from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging
import time

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# -------------------------------
# Input schema
# -------------------------------
class InputData(BaseModel):
    gender: str
    age: int
    salary: float
    tenure: int


# -------------------------------
# Load artifacts
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))


# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.get("/")
def home():
    return {"status": "healthy"}

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(input_data: InputData):
    try:
        user_data = input_data.dict()
        start_time = time.time()

        # Convert to DataFrame
        df = pd.DataFrame([user_data])

        # 🔹 Ensure all expected columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = 0

        # 🔹 Reorder columns exactly as training
        df = df[columns]

        # 🔹 Predict probability (pipeline handles scaling)
        churn_proba = model.predict_proba(df)[0][1]
        end_time = time.time()
        latency = end_time - start_time
        
        # 🔹 Threshold tuning
        threshold = 0.3
        prediction = 1 if churn_proba > threshold else 0
        
        logging.info(f"Input: {input_data}")
        logging.info(f"Prediction: {prediction}")
        logging.info(f"Latency: {latency}")
        
        return {
            "prediction": prediction,
            "churn_probability": float(churn_proba)
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}