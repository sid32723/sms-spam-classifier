import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# Load trained model
MODEL_PATH = PROJECT_ROOT / "models" / "final_spam_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)


# Create FastAPI application
app = FastAPI()


# Define expected request format
class SMSRequest(BaseModel):
    message: str = Field(min_length=1)

class PredictionResponse(BaseModel):
    prediction: str
    spam_probability: float


@app.get("/")
def home():
    return {
        "name": "SMS Spam Classifier API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SMSRequest):

    message = pd.Series([request.message])

    prediction = pipeline.predict(message)[0]

    probabilities = pipeline.predict_proba(message)[0]

    spam_index = list(pipeline.classes_).index("spam")
    spam_probability = probabilities[spam_index]

    return {
        "prediction": prediction,
        "spam_probability": round(float(spam_probability),4)
    }