import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
import pandas as pd

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\KIIT\PycharmProjects\Multi_disease_predictor\multi-disease-predictor-db-firebase-adminsdk-fbsvc-88e7976178.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the disease prediction model
with open("disease_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load datasets
disease_symptoms_df = pd.read_csv(r"C:\Users\KIIT\Downloads\archive (2)\DiseaseAndSymptoms.csv")
precaution_df = pd.read_csv(r"C:\Users\KIIT\Downloads\archive (2)\Disease precaution.csv")

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
@app.post("/predict")
def predict_disease(data: PredictionRequest):
    symptoms_array = np.array(data.symptoms).reshape(1, -1)
    prediction = model.predict(symptoms_array)[0]
    confidence = float(np.max(model.predict_proba(symptoms_array)) * 100)

    disease_data = disease_info.get(prediction, {"symptoms": [], "precautions": [], "recommendations": [], "medications": []})

    prediction_result = {
        "disease": prediction,
        "confidence": confidence,
        "symptoms": disease_data["symptoms"],
        "precautions": disease_data["precautions"],
        "recommendations": disease_data["recommendations"],
        "medications": disease_data["medications"]
    }

    # Store result in Firebase
    doc_ref = db.collection("predictions").add({
        "symptoms": data.symptoms,
        "disease": prediction,
        "confidence": confidence,
    })

    return {"prediction": prediction_result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



