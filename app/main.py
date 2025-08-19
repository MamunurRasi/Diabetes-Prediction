import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.schemas import InputFeatures, OutputPrediction
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os

app = FastAPI(
    title="Diabetes Detection API",
    description="A simple API that predicts diabetes using a trained Random Forest model.",
    version="1.0"
)

# Define the expected feature names in the correct order
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from disk
model_path = os.path.join("app", "model", "diabetes_model.joblib")
try:
    rf_model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Serve HTML frontend
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return os.path.join(static_path, "index.html")

# Info route
@app.get("/info")
def get_info():
    return {
        "message": "Diabetes Detection API",
        "version": "1.0",
        "author": "Your Name or Team",
        "features": feature_names
    }

# Prediction route
@app.post("/predict", response_model=OutputPrediction)
def diabetes_prediction(input_features: InputFeatures):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features.dict()], columns=feature_names)
        
        # Make prediction
        prediction = int(rf_model.predict(input_df)[0])
        probability = float(rf_model.predict_proba(input_df)[0][1])
        
        return OutputPrediction(prediction=prediction, probability=probability)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
