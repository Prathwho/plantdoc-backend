from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io
import os
import base64
from google import genai

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

session = ort.InferenceSession("plant_disease_model.onnx")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CLIENT (New SDK)
# ─────────────────────────────────────────────────────────────────────────────

gemini_client = None
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("[Gemini] ✅ Client initialized with google-genai SDK")
    else:
        print("[Gemini] ⚠️ No GEMINI_API_KEY found")
except Exception as e:
    print(f"[Gemini] ❌ Failed to init: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def detect_skin_pixels(image_bytes: bytes) -> float:
    """Detects human skin-tone pixels using multiple color-space rules."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((100, 100))
        pixels = np.array(img).astype(np.float32)
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

        # Rule 1: Kovac skin detection (light skin)
        kovac = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 15) &
            (r - b > 15)
        )

        # Rule 2: Warm brown skin tones
        warm = (
            (r > 80) & (g > 30) & (b > 15) &
            (r > g) & (g > b) &
            (r - b > 20) & (r < 240)
        )

        # Rule 3: YCbCr-like skin range
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
        cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
        ycbcr = (
            (y > 80) &
            (cb > 85) & (cb < 135) &
            (cr > 135) & (cr < 180)
        )

        skin_mask = kovac | warm | ycbcr
        return float(np.sum(skin_mask) / 10000.0)
    except:
        return 0.0


def compute_green_ratio(image_bytes: bytes) -> dict:
    """Comprehensive green/organic color analysis."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((100, 100))
        arr = np.array(img).astype(np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Natural green (chlorophyll-like)
        natural_green = np.sum(
            (g > r) & (g > b) & (g > 50) &
            (np.abs(g - r) > 10) & (np.abs(g - b) > 10)
        ) / 10000.0

        # Brownish organic tones (dried leaves, bark, soil)
        brownish = np.sum(
            (r > b) & (r > g * 0.85) & (r > 60) & (r < 200) &
            (g > 30) & (b < r * 0.8)
        ) / 10000.0

        # Texture roughness (gradient magnitude)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        dx = np.diff(gray, axis=1)[:-1, :]
        dy = np.diff(gray, axis=0)[:, :-1]
        gradient = np.sqrt(dx**2 + dy**2)
        roughness = float(np.mean(gradient) / 255.0)

        # Edge density (how many strong edges exist)
        edge_threshold = 15.0
        edge_density = float(np.sum(gradient > edge_threshold) / gradient.size)

        # Color variance (low = uniform/artificial, high = natural)
        color_std = float(np.mean([np.std(r), np.std(g), np.std(b)]))

        return {
            "natural_green": natural_green,
            "brownish": brownish,
            "roughness": roughness,
            "edge_density": edge_density,
            "color_std": color_std,
        }
    except:
        return {"natural_green": 0, "brownish": 0, "roughness": 0, "edge_density": 0, "color_std": 0}


def quick_is_plant_heuristic(image_bytes: bytes) -> dict:
    """
    IRONCLAD GATEKEEPER HEURISTIC (v3)
    Returns dict with 'is_plant' bool and 'reason' string.
    This is a PRE-FILTER only — Gemini is the final authority.
    """
    skin_score = detect_skin_pixels(image_bytes)
    colors = compute_green_ratio(image_bytes)

    reasons = []

    # ── HARD REJECT: High skin presence ──────────────────────────────
    if skin_score > 0.12:
        return {"is_plant": False, "reason": f"High skin pixel ratio ({skin_score:.2f})", "skin": skin_score, **colors}

    # ── HARD REJECT: Very smooth + uniform (artificial object/clothing) ──
    if colors["roughness"] < 0.015 and colors["color_std"] < 25:
        return {"is_plant": False, "reason": "Extremely smooth and uniform — likely artificial object", "skin": skin_score, **colors}

    # ── HARD REJECT: Green but too flat (green shirt, green wall) ────
    if colors["natural_green"] > 0.3 and colors["roughness"] < 0.02 and colors["edge_density"] < 0.08:
        return {"is_plant": False, "reason": "Green but lacks leaf texture — likely clothing/flat surface", "skin": skin_score, **colors}

    # ── SOFT CHECK: Low organic colors + low texture ─────────────────
    organic_score = colors["natural_green"] + colors["brownish"] * 0.7
    if organic_score < 0.05 and colors["roughness"] < 0.03:
        return {"is_plant": False, "reason": f"No organic colors or texture detected (organic={organic_score:.3f})", "skin": skin_score, **colors}

    # ── MARGINAL: Some green but combined with skin ──────────────────
    if skin_score > 0.06 and colors["natural_green"] < 0.15:
        return {"is_plant": False, "reason": f"Skin detected ({skin_score:.2f}) with low plant signal", "skin": skin_score, **colors}

    return {"is_plant": True, "reason": "Passed heuristic checks", "skin": skin_score, **colors}


def gemini_validate_plant(image_bytes: bytes, ml_prediction: str = "") -> dict:
    """
    THE PRIMARY GATEKEEPER — uses Gemini Vision to validate if an image
    actually contains a plant. Returns a dict with validation result.
    """
    if not gemini_client:
        return {"validated": False, "is_plant": None, "error": "Gemini client not available"}

    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize for faster processing (Gemini doesn't need full res for validation)
        max_dim = 512
        if max(img_pil.size) > max_dim:
            ratio = max_dim / max(img_pil.size)
            new_size = (int(img_pil.size[0] * ratio), int(img_pil.size[1] * ratio))
            img_pil = img_pil.resize(new_size, Image.LANCZOS)

        validation_prompt = """STRICT BOTANICAL VALIDATOR — Answer with EXACTLY one of these two formats:

FORMAT A (if the image shows a real plant, leaf, flower, tree, fruit, or any botanical subject):
PLANT: YES
SPECIES: [common name of the plant]

FORMAT B (if the image shows ANYTHING else — a human, person, animal, object, clothing, car, food, building, artwork, drawing, screenshot, text, or any non-plant subject):
PLANT: NO
REASON: [what the image actually shows]

CRITICAL RULES:
- A person wearing green is NOT a plant
- A green object is NOT a plant
- An animal near plants is NOT a plant — the MAIN SUBJECT must be botanical
- A drawing or painting of a plant does NOT count
- Be EXTREMELY strict. When in doubt, say NO."""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[validation_prompt, img_pil]
        )

        result_text = response.text.strip().upper()
        print(f"[Gemini Validator] Raw response: {response.text.strip()}")

        if "PLANT: YES" in result_text or "PLANT:YES" in result_text:
            # Extract species if provided
            species = "Unknown"
            for line in response.text.strip().split("\n"):
                if line.strip().upper().startswith("SPECIES:"):
                    species = line.split(":", 1)[1].strip()
                    break
            return {"validated": True, "is_plant": True, "species": species}

        elif "PLANT: NO" in result_text or "PLANT:NO" in result_text:
            reason = "Non-plant subject detected"
            for line in response.text.strip().split("\n"):
                if line.strip().upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                    break
            return {"validated": True, "is_plant": False, "reason": reason}

        else:
            # Ambiguous response — treat as rejection for safety
            print(f"[Gemini Validator] Ambiguous response, rejecting for safety")
            return {"validated": True, "is_plant": False, "reason": "Could not confirm this is a plant"}

    except Exception as e:
        print(f"[Gemini Validator] Error: {e}")
        return {"validated": False, "is_plant": None, "error": str(e)}


def gemini_generate_report(image_bytes: bytes, ml_prediction: str, species_hint: str = "") -> str:
    """Uses Gemini Vision to generate the full plant care report."""
    if not gemini_client:
        return ""

    try:
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        REPORT_FORMAT = """
### FINAL_DIAGNOSIS ### 
[Specific Disease name, or "Healthy"]

### PLANT_SPECIES ### 
[Common name of the plant]

### PLANT_IDENTIFIED ###
[Detailed plant name and variety]

### DISEASE_ANALYSIS ###
[Symptoms seen and severity]

### TREATMENT_NOW ###
- [Step 1]
- [Step 2]
- [Step 3]

### PREVENTION ###
[Prevention advice]

### WATERING_CARE ###
[Watering instructions]

### SUNLIGHT_NEEDS ###
[Sunlight instructions]

### MINERALS_FERTILIZERS ###
[Mineral/Fertilizer advice]

### EXPERT_TIPS ###
[Expert tip and Fun fact]
"""

        report_prompt = f"""You are PlantDoc, a world-class botanist and plant pathologist.
Analyze this plant image and provide a comprehensive care report.

The ML scanner suggested: "{ml_prediction}"
{f'Species hint: {species_hint}' if species_hint else ''}

Override the ML prediction if it's clearly wrong. Be accurate and professional.
NEVER OMIT A SECTION. Follow this EXACT format:
{REPORT_FORMAT}"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[report_prompt, img_pil]
        )

        return response.text.strip()

    except Exception as e:
        print(f"[Gemini Report] Error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "PlantDoc Smart Gardening Advisor API is running! 🌿"}


@app.post("/chat")
def chat(chat_message: ChatMessage):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are PlantDoc, an expert AI plant doctor and smart gardening advisor.
                    When diagnosing a problem always follow this structure:
                    1. 🌿 What I think this is
                    2. 🔍 Why this is happening
                    3. 💊 What to do now
                    4. 🛡️ How to prevent it next time
                    5. 💬 One thing to watch
                    Be warm, encouraging and use simple language."""
                },
                {"role": "user", "content": chat_message.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # ── STEP 1: Quick heuristic pre-filter ──────────────────────────
        heuristic = quick_is_plant_heuristic(image_bytes)
        if not heuristic["is_plant"]:
            return {
                "disease": "Not recognized",
                "confidence": 0,
                "advice": f"⚠️ This doesn't look like a plant image. {heuristic['reason']}",
                "is_plant": False
            }

        # ── STEP 2: Gemini validation (THE authority) ───────────────────
        gemini_check = gemini_validate_plant(image_bytes)
        if gemini_check.get("validated") and gemini_check.get("is_plant") is False:
            return {
                "disease": "Not recognized",
                "confidence": 0,
                "advice": f"⚠️ {gemini_check.get('reason', 'Not a plant image.')} Please upload a clear photo of a plant leaf.",
                "is_plant": False
            }

        # If Gemini couldn't validate (API error), only allow if heuristic was strongly positive
        if not gemini_check.get("validated"):
            colors = compute_green_ratio(image_bytes)
            if colors["natural_green"] < 0.20:
                return {
                    "disease": "Not recognized",
                    "confidence": 0,
                    "advice": "⚠️ Could not verify this is a plant. Please try again with a clearer plant photo.",
                    "is_plant": False
                }

        img_array = preprocess_image(image_bytes)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0][0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        disease = class_names[str(predicted_class)]

        advice_prompt = f"Provide a detailed treatment report for {disease}."
        advice_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are PlantDoc, an expert plant doctor."},
                {"role": "user", "content": advice_prompt}
            ]
        )
        return {
            "disease": disease,
            "confidence": round(confidence, 2),
            "advice": advice_response.choices[0].message.content,
            "is_plant": True
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/identify-image")
async def identify_image(file: UploadFile = File(...)):
    """
    MASTER IDENTIFY ENDPOINT (v3 — IRONCLAD)
    ──────────────────────────────────────────
    Step 1: Quick local heuristic (catches obvious non-plants instantly)
    Step 2: Gemini Vision VALIDATES it's a real plant (PRIMARY AUTHORITY)
    Step 3: Only if validated → run ONNX model + generate care report
    Step 4: If Gemini unavailable → REJECT unless heuristic is very confident
    """
    try:
        image_bytes = await file.read()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Quick local heuristic pre-filter
        # ═══════════════════════════════════════════════════════════════════
        heuristic = quick_is_plant_heuristic(image_bytes)
        print(f"[Heuristic] is_plant={heuristic['is_plant']}, reason={heuristic['reason']}")

        if not heuristic["is_plant"]:
            return {
                "response": f"This doesn't appear to be a plant. {heuristic['reason']}. Please upload a clear photo of a plant leaf.",
                "final_name": "Not a plant",
                "is_plant": False,
                "ml_result": {"confidence": 0, "all_predictions": []},
                "audit_method": "Heuristic",
                "debug_heuristic": heuristic
            }

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: GEMINI VISION — THE PRIMARY GATEKEEPER
        # ═══════════════════════════════════════════════════════════════════
        gemini_validation = gemini_validate_plant(image_bytes)
        print(f"[Gemini Validation] {gemini_validation}")

        if gemini_validation.get("validated"):
            if gemini_validation.get("is_plant") is False:
                # GEMINI SAYS NOT A PLANT → HARD REJECT
                reason = gemini_validation.get("reason", "This is not a plant image.")
                return {
                    "response": f"{reason}. Please upload a clear photo of a plant leaf for identification.",
                    "final_name": "Not a plant",
                    "is_plant": False,
                    "ml_result": {"confidence": 0, "all_predictions": []},
                    "audit_method": "Gemini Validator",
                    "debug_heuristic": heuristic
                }
            # Gemini confirmed it's a plant — continue
            gemini_species = gemini_validation.get("species", "Unknown")
        else:
            # Gemini API failed — ONLY allow if heuristic strongly suggests plant
            print(f"[Gemini] API unavailable, falling back to strict heuristic")
            if heuristic.get("natural_green", 0) < 0.20 or heuristic.get("skin", 0) > 0.05:
                return {
                    "response": "Could not verify this image. Please try again with a clear plant photo.",
                    "final_name": "Not a plant",
                    "is_plant": False,
                    "ml_result": {"confidence": 0, "all_predictions": []},
                    "audit_method": "Heuristic (Gemini unavailable)",
                    "debug_heuristic": heuristic
                }
            gemini_species = "Unknown"

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: VALIDATED AS PLANT — Run ONNX model
        # ═══════════════════════════════════════════════════════════════════
        img_array = preprocess_image(image_bytes)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0][0]
        top5_idx = np.argsort(predictions)[::-1][:5]

        all_predictions = []
        for idx in top5_idx:
            label = class_names.get(str(idx), f"Class {idx}")
            all_predictions.append({
                "label": label.replace("_", " ").title(),
                "confidence": round(float(predictions[idx]) * 100, 1)
            })

        ml_top = all_predictions[0]["label"]
        ml_confidence = all_predictions[0]["confidence"]

        ml_result = {
            "top_prediction": ml_top,
            "confidence": ml_confidence,
            "all_predictions": all_predictions,
            "is_plant_color_heuristic": True,
            "species_name": gemini_species if gemini_species != "Unknown" else ml_top.split("___")[0].replace("_", " ").title()
        }

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Generate full care report
        # ═══════════════════════════════════════════════════════════════════
        report_text = ""
        audit_method = "Local ML"
        species_name = gemini_species if gemini_species != "Unknown" else ml_top.split("___")[0].replace("_", " ").title()
        final_name = ml_top

        # Try Gemini report first
        report_text = gemini_generate_report(image_bytes, ml_top, species_hint=gemini_species)
        if report_text and "### FINAL_DIAGNOSIS ###" in report_text:
            audit_method = "Gemini"
        else:
            # Groq fallback for report generation
            print("[Fallback] Generating report with Groq")
            try:
                REPORT_FORMAT = """
### FINAL_DIAGNOSIS ### 
[Specific Disease name, or "Healthy"]

### PLANT_SPECIES ### 
[Common name of the plant]

### PLANT_IDENTIFIED ###
[Detailed plant name and variety]

### DISEASE_ANALYSIS ###
[Symptoms seen and severity]

### TREATMENT_NOW ###
- [Step 1]
- [Step 2]
- [Step 3]

### PREVENTION ###
[Prevention advice]

### WATERING_CARE ###
[Watering instructions]

### SUNLIGHT_NEEDS ###
[Sunlight instructions]

### MINERALS_FERTILIZERS ###
[Mineral/Fertilizer advice]

### EXPERT_TIPS ###
[Expert tip and Fun fact]
"""
                fallback_prompt = f"""You are PlantDoc. Construct a full plant care report for:
                SPECIES: {species_name}
                DIAGNOSIS: {final_name}
                
                Follow this EXACT format:
                {REPORT_FORMAT}
                
                Be expert and professional."""

                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": fallback_prompt}],
                    max_tokens=1200
                )
                report_text = res.choices[0].message.content
                audit_method = "Groq"
            except Exception as groq_err:
                print(f"[Groq Error] {groq_err}")

        # ── Extract Final Metadata ──────────────────────────────────────
        is_model_reliable = True
        if report_text:
            try:
                if "FINAL_DIAGNOSIS:" in report_text:
                    extracted = report_text.split("FINAL_DIAGNOSIS:")[1].split("\n")[0].strip().replace("*", "")
                    if len(extracted) > 2:
                        final_name = extracted

                if "PLANT_SPECIES:" in report_text:
                    extracted = report_text.split("PLANT_SPECIES:")[1].split("\n")[0].strip().replace("*", "")
                    if len(extracted) > 2:
                        species_name = extracted

                ml_species_base = ml_top.split("___")[0].split(" ")[0].lower()
                if ml_species_base in species_name.lower() or species_name.lower() in ml_species_base:
                    is_model_reliable = True
                else:
                    is_model_reliable = False
            except Exception as extract_err:
                print(f"[Extraction Error] {extract_err}")

        # Final cleanup if EVERYTHING failed
        if not report_text:
            report_text = f"Identified as: {final_name}. Please check internet connection for full care guide."

        ml_result["species_name"] = species_name

        return {
            "response": report_text,
            "final_name": final_name,
            "identified_species": species_name,
            "is_model_reliable": is_model_reliable,
            "ml_result": ml_result,
            "is_plant": True,
            "audit_method": audit_method
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/describe-plant")
async def describe_plant(data: dict):
    try:
        description = data.get("description", "")
        prompt = f"Describe plant: {description}"
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}