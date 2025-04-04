import firebase_admin 
from firebase_admin import credentials, firestore
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
import os
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

firebase_key_path = os.getenv("FIREBASE_KEY_PATH", "multi-disease-predictor-db-firebase-adminsdk-fbsvc-88e7976178.json")
# Initialize Firebase
cred = credentials.Certificate(firebase_key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the disease prediction model
with open("disease_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load datasets
disease_symptoms_df = pd.read_csv("DiseaseAndSymptoms.csv")
precaution_df = pd.read_csv("Disease precaution.csv")

# Process disease data
disease_info = {}
for _, row in disease_symptoms_df.iterrows():
    disease = row["Disease"]
    symptoms = [symptom for symptom in row[1:].dropna().tolist()]

    precautions = precaution_df[precaution_df["Disease"] == disease]
    precaution_list = precautions.iloc[:, 1:].dropna(axis=1).values.flatten().tolist() if not precautions.empty else []

    disease_info[disease] = {
        "symptoms": symptoms,
        "precautions": precaution_list,
        "recommendations": [],
        "medications": []
    }

# FastAPI instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sahooananya.github.io"],  # Or use specific domain like ["https://yourusername.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_disease(
    age: int = Form(...),
    gender: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    hemoglobin: float = Form(...),
    rbc_count: float = Form(...),
    wbc_count: float = Form(...),
    platelets: float = Form(...),
    iron_level: float = Form(...),
    vitamin_b12: float = Form(...),
    folate: float = Form(...),
    images: Optional[List[UploadFile]] = None
):
    try:
        # Log received input data
        logging.info(f"Received Data - Age: {age}, Gender: {gender}, Weight: {weight}, Height: {height}, Hemoglobin: {hemoglobin}, RBC: {rbc_count}, WBC: {wbc_count}, Platelets: {platelets}, Iron: {iron_level}, B12: {vitamin_b12}, Folate: {folate}")
        
        input_data = np.array([
            age, weight, height, hemoglobin, rbc_count, wbc_count, platelets, iron_level, vitamin_b12, folate
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        confidence = float(np.max(model.predict_proba(input_data)) * 100)

        disease_data = disease_info.get(prediction, {
            "symptoms": [],
            "precautions": [],
            "recommendations": ["Consult a doctor", "Follow a healthy diet"],
            "medications": [{"name": "Paracetamol", "dosage": "500mg", "frequency": "Twice a day"}]
        })

        prediction_result = {
            "disease": prediction,
            "confidence": confidence,
            "recommendations": disease_data["recommendations"],
            "medications": disease_data["medications"]
        }

        # Save to Firestore
        db.collection("predictions").add({
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "hemoglobin": hemoglobin,
            "rbc_count": rbc_count,
            "wbc_count": wbc_count,
            "platelets": platelets,
            "iron_level": iron_level,
            "vitamin_b12": vitamin_b12,
            "folate": folate,
            "disease": prediction,
            "confidence": confidence,
        })

        # Log prediction result
        logging.info(f"Prediction: {prediction}, Confidence: {confidence}%")

        if images:
            for image in images:
                logging.info(f"Received image: {image.filename}")

        return {"status": "success", "result": prediction_result}

    except Exception as e:
        logging.error("ERROR: " + str(e))
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
