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
    Very simple gatekeeper to avoid 'Celebrity' leaks while allowing all leaves.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[GATEKEEPER] CRITICAL: GEMINI_API_KEY is missing!")
        return True # Temporarily pass if key is missing so we can at least use the ML model
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ultra simple prompt
        prompt = "Is this a picture of a plant or a plant leaf? Answer strictly YES or NO."
        
        response = vision_model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=5
            )
        )
        answer = response.text.upper().strip()
        print(f"[GATEKEEPER] AI Response: '{answer}'")
        
        if "YES" in answer:
            return True
        return False
    except Exception as e:
        print(f"[GATEKEEPER] API Error: {e}")
        return True # Default to PASS on technical error so the app doesn't break

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