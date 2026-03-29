import numpy as np
from PIL import Image
import io
import onnxruntime as ort
import json

# Load your existing model and class names
print("🌿 Loading plant ML model...")
plant_session = ort.InferenceSession("plant_disease_model.onnx")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("✅ Model loaded!")
print(f"📋 Classes available: {len(class_names)}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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

        return {
            "success": True,
            "top_prediction": results[0]["label"],
            "confidence": results[0]["confidence"],
            "all_predictions": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}