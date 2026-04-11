import numpy as np
from PIL import Image
import io
import onnxruntime as ort
import json

# Load your existing model and class names
print("[INFO] Loading plant ML model...")
plant_session = ort.InferenceSession("plant_disease_model.onnx")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("[INFO] Model loaded!")
print(f"[INFO] Classes available: {len(class_names)}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini Vision mapping
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
vision_model = genai.GenerativeModel('gemini-1.5-flash')

def check_is_valid_plant_image(image_bytes: bytes) -> bool:
    """
    Acts as a strict gatekeeper to ensure only plants/leaves are processed.
    Rejects humans, animals, objects, and generic backgrounds.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = """
        ACT AS A PLANT ADVISOR. 
        Is this a picture of a PLANT, a PLANT LEAF, or a PLANT DISEASE?
        
        ✅ YES: If it is any kind of plant leaf, whole plant, or flower. 
        Accept leaves even if they have HOLES, SPOTS, DISCOLORATION, or are DAMAGED. These are the most important images for us!
        
        ❌ NO: If the main focus is a HUMAN face/body, an ANIMAL, an OBJECT (like a car, toy, or furniture), or a generic building/room with no plant.
        
        Only answer NO if it is clearly NOT a plant. If it looks like a leaf, even a sick one, answer YES.
        Answer with exactly one word: YES or NO.
        """
        response = vision_model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=5
            )
        )
        answer = response.text.strip().upper()
        # Log for debugging if needed (check your Render logs)
        print(f"[GATEKEEPER] Vision analysis result: {answer}")
        return "YES" in answer
    except Exception as e:
        print(f"[GATEKEEPER] Gemini Vision Error: {e}")
        return False # Fail-Safe: Reject if we can't verify (prevents leaks)

def identify_plant_from_image(image_bytes: bytes) -> dict:
    try:
        img_array = preprocess_image(image_bytes)

        input_name = plant_session.get_inputs()[0].name
        outputs = plant_session.run(None, {input_name: img_array})
        predictions = outputs[0][0]

        # Get top 5 predictions
        top5_idx = np.argsort(predictions)[::-1][:5]
        results = []
        for idx in top5_idx:
            label = class_names.get(str(idx), class_names.get(idx, f"Class {idx}"))
            results.append({
                "label": label.replace("_", " ").title(),
                "confidence": round(float(predictions[idx]) * 100, 1)
            })

        is_plant = check_is_valid_plant_image(image_bytes)

        return {
            "success": True,
            "top_prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "all_predictions": results,
            "is_plant_color_heuristic": is_plant
        }
    except Exception as e:
        return {"success": False, "error": str(e)}