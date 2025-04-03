import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import pandas as pd
import os
from fastapi import HTTPException

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

# Input schema for prediction request
class PredictionRequest(BaseModel):
    symptoms: list  # Example: [1, 0, 1, 0, 1] (Binary input for symptoms)

# Predict disease and store user data in Firebase
from fastapi import HTTPException

@app.post("/predict")
def predict_disease(data: PredictionRequest):
    try:
        symptoms_array = np.array(data.symptoms).reshape(1, -1)
        prediction = model.predict(symptoms_array)[0]
        confidence = float(np.max(model.predict_proba(symptoms_array)) * 100)

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

        # Store result in Firebase
        db.collection("predictions").add({
            "symptoms": data.symptoms,
            "disease": prediction,
            "confidence": confidence,
        })

        return prediction_result

    except Exception as e:
        print("ERROR:", str(e))  # Logs error
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



