from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

class PredictRequest(BaseModel):
    smiles: str

@app.get("/")
def root():
    return {"status": "ok", "mode": "mock"}

@app.post("/predict")
def predict(req: PredictRequest):
    return {
        "smiles": req.smiles,
        "vdss_kg": round(random.uniform(0.8, 1.2), 3),
        "note": "model disabled"
    }
