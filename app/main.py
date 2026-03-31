from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

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

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))


# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(input_data: InputData):
    try:
        user_data = input_data.dict()

        # Convert to DataFrame
        df = pd.DataFrame([user_data])

        # 🔹 Ensure all expected columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = 0

        # 🔹 Reorder columns exactly as training
        df = df[columns]

        # 🔹 Apply scaling
        df_scaled = scaler.transform(df)

        # 🔹 Predict probability
        churn_proba = model.predict_proba(df_scaled)[0][1]

        # 🔹 Threshold tuning
        threshold = 0.3
        prediction = 1 if churn_proba > threshold else 0

        return {
            "prediction": int(prediction),
            "churn_probability": float(churn_proba)
        }

    except Exception as e:
        return {"error": str(e)}