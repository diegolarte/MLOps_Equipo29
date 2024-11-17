from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys

# Add the current directory to the Python path to locate Insurance_refactor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Insurance_refactor import DataExplorer, XGBClassifier, mlflow

# Initialize FastAPI app
app = FastAPI()

# Load or initialize the model
model_path = "models/insurance_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "API is up and running"}

# Endpoint for data exploration
@app.post("/explore")
async def explore_data(file: UploadFile = File(...)):
    try:
        # Load CSV file
        data = pd.read_csv(file.file)
        explorer = DataExplorer(data)
        summary = {
            "head": data.head().to_dict(),
            "description": data.describe().to_dict(),
            "info": str(data.info())
        }
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for training the model
@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = Form("target")  # Default target column
):
    try:
        # Load CSV file
        data = pd.read_csv(file.file)
        if target not in data.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")

        # Split data
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        global model
        model = XGBClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Save the model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)

        # Log metrics with MLflow
        with mlflow.start_run():
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            mlflow.sklearn.log_model(model, "model")

        return {"message": "Model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=400, detail="Model not found. Please train a model first.")
    try:
        # Load CSV file
        data = pd.read_csv(file.file)
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

