from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import onnxruntime as ort
import numpy as np
from PIL import Image
from ml_model import identify_plant_from_image
import json
import io
import os

load_dotenv()

# Load Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load disease ML model
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

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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
        img_array = preprocess_image(image_bytes)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0][0]

        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        disease = class_names[str(predicted_class)]

        from ml_model import check_is_valid_plant_image
        if not check_is_valid_plant_image(image_bytes):
            return {
                "disease": "Not recognized",
                "confidence": round(confidence, 2),
                "advice": "⚠️ PlantDoc could not identify this as a plant leaf. Please upload a clear close-up photo of a plant leaf only. Screenshots, objects, animals, and humans are not supported.",
                "is_plant": False
            }

        advice_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are PlantDoc, an expert plant doctor. Give brief treatment advice."
                },
                {
                    "role": "user",
                    "content": f"My plant has been diagnosed with {disease}. Give me a brief treatment plan in 3 steps."
                }
            ]
        )
        advice = advice_response.choices[0].message.content

        return {
            "disease": disease,
            "confidence": round(confidence, 2),
            "advice": advice,
            "is_plant": True
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/identify-image")
async def identify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        ml_result = identify_plant_from_image(image_bytes)

        if not ml_result["success"]:
            return {
                "response": "⚠️ PlantDoc could not process this image. Please upload a clear close-up photo of a plant leaf.",
                "ml_result": ml_result,
                "is_plant": False
            }

        plant_name = ml_result["top_prediction"]
        confidence = ml_result["confidence"]
        others = ", ".join([p["label"] for p in ml_result["all_predictions"][1:3]])

        # Use LLM to intelligently handle and respond — it will naturally say
        # if the image doesn't look like a plant based on the prediction labels
        prompt = f"""A user uploaded an image to PlantDoc, a plant disease detection app.
The ML model's top prediction: {plant_name} (confidence: {confidence}%)
Other possibilities: {others}

If the prediction looks like a valid plant disease/condition name (e.g. Healthy, Powdery, Rust), provide:
1. 🌿 Plant Identity — what plant or condition this likely is in simple language
2. 🎯 Confidence note — is {confidence}% high or low?
3. 💧 Watering guide
4. ☀️ Sunlight needs
5. 🌡️ Ideal temperature
6. 🌱 Best soil type
7. ⚠️ Disease & pest warnings
8. 💡 Fun facts

If the confidence is very low (under 40%) or the prediction doesn't make sense for a plant, gently let the user know and suggest uploading a clearer plant leaf photo.

Use simple everyday language, not scientific terms."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        # Determine is_plant based on the color heuristic run in ml_model
        is_plant = ml_result.get("is_plant_color_heuristic", True)

        return {
            "response": response.choices[0].message.content,
            "ml_result": ml_result,
            "is_plant": is_plant
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/describe-plant")
async def describe_plant(data: dict):
    try:
        description = data.get("description", "")

        prompt = f"""A user described their plant as: "{description}"

Based on this description please provide:
1. 🌿 Most likely plant — top 2-3 matches with common names
2. 🔍 Why it matches based on their description
3. 💧 Watering guide
4. ☀️ Sunlight needs
5. 🌡️ Ideal temperature
6. 🌱 Best soil type
7. ⚠️ Disease & pest warnings
8. 💡 Fun facts

Use simple everyday language."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}