from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("models/model.pkl", "rb"))

# ✅ Define request schema
class InputData(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "Fraud Detection API"}

@app.post("/predict")
def predict(input: InputData):
    data = np.array(input.data).reshape(1, -1)
    pred = model.predict(data)
    return {"fraud": int(pred[0])}