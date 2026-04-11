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

        advice_prompt = f"""
        A plant has been diagnosed with: {disease} (Confidence: {confidence}%).
        
        Provide a detailed treatment and care report:
        1. 🔬 MINERAL ANALYSIS: What minerals might the plant be lacking?
        2. 🛠️ 3-STEP RECOVERY: Immediate actions to take.
        3. 🛡️ PREVENTION: How to stop this from returning.
        4. 💧 WATERING & ☀️ SUNLIGHT: Specific needs during recovery.
        
        Keep it warm, simple, and expert."""

        advice_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are PlantDoc, an expert plant doctor."},
                {"role": "user", "content": advice_prompt}
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

        # Use LLM to intelligently handle and respond
        prompt = f"""
        A user uploaded a plant photo for expert analysis.
        
        ML PREDICTION: {plant_name}
        CONFIDENCE: {confidence}%
        
        Please provide a comprehensive Plant Care & Health Guide including:
        
        1. 🌿 IDENTITY: Confirm the plant type and what condition {plant_name} indicates.
        2. 🎯 ACCURACY: Is {confidence}% a reliable score for this diagnosis?
        3. 🔬 MINERAL ANALYSIS: Based on the symptoms described/shown, which minerals are likely missing (e.g., Nitrogen, Phosphorus, Potassium, Magnesium)?
        4. 💧 WATERING: Specific frequency and method for this plant.
        5. ☀️ SUNLIGHT: Ideal light conditions (Direct/Indirect/Shade).
        6. 🌱 SOIL & RE-POTTING: Best soil mixture and when to re-pot.
        7. 🛠️ ACTION PLAN: 3 immediate steps for the user to help the plant recover or thrive.
        8. 💡 FUN FACT: One interesting fact about this plant.
        
        Use warm, encouraging, and simple language suitable for a home gardener. Avoid overly technical jargon.
        """

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