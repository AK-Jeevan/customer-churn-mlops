from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

class InputData(BaseModel):
    gender: str
    age: int
    salary: float
    tenure: int


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(input_data: InputData):
    try:
        user_data = input_data.dict()

        # Fill missing columns with None
        full_data = {col: user_data.get(col, None) for col in columns}

        df = pd.DataFrame([full_data])

        # Get churn probability
        churn_proba = model.predict_proba(df)[0][1]

        # 🔥 Threshold tuning (important)
        threshold = 0.3
        prediction = 1 if churn_proba > threshold else 0

        return {
            "prediction": prediction,
            "churn_probability": float(churn_proba)
        }

    except Exception as e:
        return {"error": str(e)}