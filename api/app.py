# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
import numpy as np
import mlflow.pyfunc

app = FastAPI()

# 🔥 Load from MLflow Registry (Staging)
MODEL_NAME = "fraud-model"
MODEL_STAGE = "Staging"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# -----------------------------
# ✅ Schemas with strict validation
# -----------------------------

# Exactly 30 features
FeatureVector = conlist(float, min_length=30, max_length=30)

class InputData(BaseModel):
    data: FeatureVector = Field(..., description="30 numerical features")

class BatchInput(BaseModel):
    data: list[FeatureVector] = Field(..., description="List of 30-length feature vectors")

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API (MLflow Registry)"}

@app.post("/predict")
def predict(input: InputData):
    try:
        x = np.array(input.data).reshape(1, -1)
        pred = model.predict(x)
        return {
            "fraud": int(pred[0]),
            "model_source": f"{MODEL_NAME} ({MODEL_STAGE})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(input: BatchInput):
    try:
        X = np.array(input.data)  # shape: (n, 30)
        preds = model.predict(X)
        return {
            "fraud_predictions": [int(p) for p in preds],
            "count": len(preds),
            "model_source": f"{MODEL_NAME} ({MODEL_STAGE})"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))