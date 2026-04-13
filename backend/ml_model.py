import numpy as np
from PIL import Image
import io
import onnxruntime as ort
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv
 
load_dotenv()
 
# Load your existing model and class names
print("[INFO] Loading plant ML model...")
plant_session = ort.InferenceSession("plant_disease_model.onnx")
 
with open("class_names.json", "r") as f:
    class_names = json.load(f)
 
print("[INFO] Model loaded!")
print(f"[INFO] Classes available: {len(class_names)}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
 
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: GEMINI VISION GATEKEEPER (Primary — Most Reliable)
# This is the most important check. Gemini can actually SEE the image and
# determine if it's a plant leaf vs human/animal/object/clothing/background.
# ─────────────────────────────────────────────────────────────────────────────
 
def check_plant_via_gemini(image_bytes: bytes) -> tuple[bool, str, str]:
    """
    Uses Gemini Vision to determine if the image is a genuine plant/leaf photo.
    Returns: (is_plant: bool, reason: str, species_guess: str)
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None, "Gemini key missing", "Unknown"
 
        genai.configure(api_key=api_key)
 
        gatekeeper_prompt = """You are PlantDoc, the world's best AI plant doctor. Analyze this image carefully.

STEP 1 — VALIDATION:
Check if the image contains any botanical subject (leaf, plant, stem, flower, fruit).
- If it is clearly NOT botanical (e.g., just a person, a car, a room with no plants, etc.) → reply with:
  NOT_A_PLANT: [brief reason]
- If there is a plant/leaf present, even if someone is holding it or it's on a table, PROCEED TO STEP 2.

STEP 2 — INDEPENDENT ANALYSIS & REPORT:
Identify the plant species and its health status visually.
If the suggestion provided seems wrong, TRUST YOUR VISUAL ANALYSIS and override it.

Write a complete, helpful report in this EXACT format:

VERDICT: PLANT
FINAL_DIAGNOSIS: [Disease name or "Healthy"]
SPECIES: [Specific plant species name, e.g. "Banana", "Tomato", "Rose"]
REASON: [Short visual reasoning, e.g. "Green leaf with typical yellow spots"]

🌿 **PLANT IDENTIFIED:** [Common plant name and brief description]
🦠 **DISEASE ANALYSIS:** [What is wrong, how it looks, and why it's happening]
🛡️ **TREATMENT NOW:** [3 specific actionable steps to treat it immediately]
🔒 **PREVENTION:** [How to prevent it from coming back]
💧 **WATERING CARE:** [Specific watering tips for this plant]
☀️ **SUNLIGHT NEEDS:** [Exact sunlight requirements]
🧪 **MINERALS & FERTILIZERS:** [Best nutrients for this species]
💡 **EXPERT TIPS & FUN FACTS:** [Pro tips about this plant]

Be specific, warm, and professional."""
 
        img = Image.open(io.BytesIO(image_bytes))
 
        # Try Gemini 2.0 Flash first, fallback to 1.5 Flash
        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([gatekeeper_prompt, img])
                text = response.text.strip().upper()
 
                # Parse verdict
                is_plant = "VERDICT: PLANT" in text
                is_rejected = "VERDICT: NOT_PLANT" in text
 
                if not is_plant and not is_rejected:
                    # Fallback keyword parse
                    is_plant = "PLANT" in text and "NOT_PLANT" not in text
 
                # Extract species
                species = "Unknown"
                for line in response.text.split("\n"):
                    if "SPECIES:" in line.upper():
                        species = line.split(":", 1)[-1].strip()
                        if not species or species.lower() == "unknown":
                            species = "Unknown"
                        break
 
                # Extract reason
                reason = "Gemini vision check"
                for line in response.text.split("\n"):
                    if "REASON:" in line.upper():
                        reason = line.split(":", 1)[-1].strip()
                        break
 
                return is_plant, reason, species
 
            except Exception:
                continue  # Try next model
 
        return None, "Gemini unavailable", "Unknown"
 
    except Exception as e:
        return None, f"Gemini error: {str(e)}", "Unknown"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: PLANTNET API CHECK (Secondary — Botanical Database)
# If Gemini is unavailable, try PlantNet. It only matches real plants.
# ─────────────────────────────────────────────────────────────────────────────
 
def check_plant_via_plantnet(image_bytes: bytes) -> tuple[bool, str, str]:
    """
    Checks PlantNet botanical database. Only real plants will match.
    Returns: (is_plant: bool, reason: str, species_name: str)
    """
    api_key = os.getenv("PLANTNET_API_KEY")
    if not api_key:
        return None, "PlantNet key missing", "Unknown"
 
    try:
        url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}&organs=leaf"
        files = [('images', ('image.jpg', image_bytes, 'image/jpeg'))]
        response = requests.post(url, files=files, timeout=8)
 
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                best = results[0]
                score = best.get("score", 0)
                latin = best.get("species", {}).get("scientificNameWithoutAuthor", "Unknown")
                common = best.get("species", {}).get("commonNames", [])
                cname = common[0].title() if common else latin
                species_name = f"{cname} ({latin})"
 
                # Score > 0.05 means a confident botanical match
                if score > 0.05:
                    return True, f"PlantNet match: {score:.2f}", species_name
                elif score > 0.01:
                    # Weak match — still a plant but low confidence
                    return True, f"PlantNet weak match: {score:.2f}", species_name
                else:
                    return False, "PlantNet: no credible match", "Unknown"
 
        elif response.status_code == 404:
            # PlantNet returns 404 when it finds NO plant in the image
            return False, "PlantNet: image not recognized as a plant", "Unknown"
 
        return None, f"PlantNet HTTP {response.status_code}", "Unknown"
 
    except Exception as e:
        return None, f"PlantNet error: {str(e)}", "Unknown"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: STRUCTURAL HEURISTICS (Last Resort — Conservative)
# Only used when both API layers fail. Much stricter than before.
# This layer REJECTS unless it finds strong evidence of organic leaf structure.
# ─────────────────────────────────────────────────────────────────────────────
 
def detect_surface_irregularities(image_bytes: bytes) -> float:
    """Measures texture roughness. Real leaves have natural texture variance."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((100, 100))
        arr = np.array(img).astype(np.float32)
        dx = np.diff(arr, axis=1)[:-1, :]
        dy = np.diff(arr, axis=0)[:, :-1]
        gradient = np.sqrt(dx**2 + dy**2)
        return float(np.mean(gradient) / 255.0)
    except:
        return 0.0
 
 
def detect_skin_pixels(image_bytes: bytes) -> float:
    """Detects human skin-tone pixels. High score = likely a person."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((100, 100))
        pixels = np.array(img).astype(np.int32)
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        # Classic Kovac skin detection formula
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r.astype(int) - g.astype(int)) > 15) &
            (r > 220) | (  # bright skin
                (r > 95) & (g > 40) & (b > 20) &
                (r > g) & (r > b) &
                (np.abs(r.astype(int) - g.astype(int)) > 15)
            )
        )
        return float(np.sum(skin_mask) / 10000.0)
    except:
        return 0.0
 
 
def detect_fabric_texture(image_bytes: bytes) -> float:
    """
    Detects uniform woven/fabric patterns (clothing).
    Fabric has repetitive fine texture unlike organic leaf veins.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((64, 64))
        arr = np.array(img).astype(np.float32)
 
        # Check for periodic patterns using row/column variance
        row_vars = np.var(arr, axis=1)
        col_vars = np.var(arr, axis=0)
 
        # Fabric has very consistent variance across rows/columns
        row_consistency = 1.0 - (np.std(row_vars) / (np.mean(row_vars) + 1e-6))
        col_consistency = 1.0 - (np.std(col_vars) / (np.mean(col_vars) + 1e-6))
 
        return float((row_consistency + col_consistency) / 2.0)
    except:
        return 0.0
 
 
def detect_leaf_vein_structure(image_bytes: bytes) -> float:
    """
    Detects leaf vein-like branching structures using edge analysis.
    Real leaves have organic branching edge patterns.
    Returns a score 0-1 where higher = more leaf-like structure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((100, 100))
        arr = np.array(img).astype(np.float32)
 
        # Sobel-like edge detection
        dx = np.diff(arr, axis=1)[:-1, :]
        dy = np.diff(arr, axis=0)[:, :-1]
        edges = np.sqrt(dx**2 + dy**2)
 
        # Organic structures have moderate, distributed edges
        # (not zero like plain backgrounds, not maxed like sharp geometry)
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
 
        # Leaf-like: moderate edges (15-60 range) with good spread
        if 10 < edge_mean < 80 and edge_std > 8:
            return min(1.0, (edge_mean / 60.0) * (edge_std / 20.0))
        return 0.0
    except:
        return 0.0
 
 
def detect_uniform_solid_color(image_bytes: bytes) -> bool:
    """
    Detects if the image is a solid/near-solid color (plain background, wallpaper).
    Returns True if it looks like a plain background.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((50, 50))
        pixels = np.array(img).astype(np.float32)
        overall_std = np.std(pixels)
        return overall_std < 18.0  # Very low variance = solid color
    except:
        return False
 
 
def heuristic_plant_check(image_bytes: bytes) -> tuple[bool, str, str]:
    """
    Conservative structural heuristic check.
    Only passes images with genuine botanical structure evidence.
    Rejects by default unless strong plant signals are found.
    """
    # Reject solid color backgrounds/wallpapers immediately
    if detect_uniform_solid_color(image_bytes):
        return False, "Plain/uniform background detected", "Unknown"
 
    skin_score = detect_skin_pixels(image_bytes)
    roughness = detect_surface_irregularities(image_bytes)
    leaf_structure = detect_leaf_vein_structure(image_bytes)
    fabric_score = detect_fabric_texture(image_bytes)
 
    # Strong skin presence = human photo
    if skin_score > 0.12:
        return False, f"Human/skin detected (score: {skin_score:.2f})", "Unknown"
 
    # High fabric consistency + no strong leaf structure = clothing
    if fabric_score > 0.85 and leaf_structure < 0.2:
        return False, f"Fabric/clothing texture detected", "Unknown"
 
    # Positive signal: genuine leaf-like vein structure
    if leaf_structure > 0.35:
        return True, f"Leaf structure detected (score: {leaf_structure:.2f})", "Unknown"
 
    # Moderate evidence: some roughness but no skin/fabric — cautiously accept
    if roughness > 0.08 and skin_score < 0.05 and fabric_score < 0.7:
        return True, f"Organic texture detected (roughness: {roughness:.2f})", "Unknown"
 
    # Default: reject — we cannot confirm it's a plant
    return False, "No botanical structure detected", "Unknown"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MASTER GATEKEEPER — Combines all three layers
# ─────────────────────────────────────────────────────────────────────────────
 
def check_is_valid_plant_image(image_bytes: bytes) -> tuple[bool, str, str]:
    """
    Multi-layer plant validation:
    Layer 1: Gemini Vision (best — actually sees the image)
    Layer 2: PlantNet API (botanical database lookup)
    Layer 3: Structural heuristics (conservative fallback)
 
    Returns: (is_plant: bool, reason: str, species_name: str)
    """
 
    # ── LAYER 1: Gemini Vision ──────────────────────────────────────────────
    gemini_result, gemini_reason, gemini_species = check_plant_via_gemini(image_bytes)
 
    if gemini_result is True:
        print(f"[GATEKEEPER] Gemini ACCEPTED: {gemini_reason}")
        return True, f"Vision AI: {gemini_reason}", gemini_species
 
    if gemini_result is False:
        print(f"[GATEKEEPER] Gemini REJECTED: {gemini_reason}")
        return False, f"Vision AI rejected: {gemini_reason}", "Unknown"
 
    # Gemini unavailable — try next layer
    print(f"[GATEKEEPER] Gemini unavailable ({gemini_reason}), trying PlantNet...")
 
    # ── LAYER 2: PlantNet API ───────────────────────────────────────────────
    plantnet_result, plantnet_reason, plantnet_species = check_plant_via_plantnet(image_bytes)
 
    if plantnet_result is True:
        print(f"[GATEKEEPER] PlantNet ACCEPTED: {plantnet_reason}")
        return True, plantnet_reason, plantnet_species
 
    if plantnet_result is False:
        print(f"[GATEKEEPER] PlantNet REJECTED: {plantnet_reason}")
        return False, plantnet_reason, "Unknown"
 
    # PlantNet also unavailable — use heuristics
    print(f"[GATEKEEPER] PlantNet unavailable ({plantnet_reason}), using heuristics...")
 
    # ── LAYER 3: Structural Heuristics ──────────────────────────────────────
    heuristic_result, heuristic_reason, _ = heuristic_plant_check(image_bytes)
    print(f"[GATEKEEPER] Heuristic result: {heuristic_result} — {heuristic_reason}")
    return heuristic_result, heuristic_reason, "Unknown"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN IDENTIFICATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
 
def identify_plant_from_image(image_bytes: bytes) -> dict:
    try:
        img_array = preprocess_image(image_bytes)
        input_name = plant_session.get_inputs()[0].name
        outputs = plant_session.run(None, {input_name: img_array})
        predictions = outputs[0][0]
        top5_idx = np.argsort(predictions)[::-1][:5]
 
        results = []
        for idx in top5_idx:
            label = class_names.get(str(idx), class_names.get(idx, f"Class {idx}"))
            results.append({
                "label": label.replace("_", " ").title(),
                "confidence": round(float(predictions[idx]) * 100, 1)
            })
 
        # Run the multi-layer gatekeeper
        is_plant, reason, species_name = check_is_valid_plant_image(image_bytes)
        roughness = detect_surface_irregularities(image_bytes)
        visual_type = "Large/Tropical" if roughness < 0.04 else "Small/Textured"
 
        return {
            "success": True,
            "top_prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "all_predictions": results,
            "is_plant_color_heuristic": is_plant,
            "gatekeeper_reason": reason,
            "species_name": species_name,
            "leaf_type": visual_type,
            "roughness": roughness
        }
    except Exception as e:
        return {"success": False, "error": str(e)}