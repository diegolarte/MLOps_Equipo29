
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Define input data model
class PredictionInput(BaseModel):
    features: dict

# Load models
models = {}
model_path = "models"
try:
    for model_file in os.listdir(model_path):
        if model_file.endswith(".pkl"):
            model_name = model_file.split("_")[0]  # Use file prefix as model name
            with open(os.path.join(model_path, model_file), "rb") as f:
                models[model_name] = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

# Endpoint for health check
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Endpoint for predictions
@app.post("/predict/{model_name}")
def predict(model_name: str, input_data: PredictionInput):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Extract features from input
    try:
        features_df = pd.DataFrame([input_data.features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    # Make prediction
    try:
        model = models[model_name]
        prediction = model.predict(features_df)
        return {"model": model_name, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
