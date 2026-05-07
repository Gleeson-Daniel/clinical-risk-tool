from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from schemas import PatientInput, PredictionResult

app = FastAPI(title="Clinical Risk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = Path("/ml/model.joblib")
artifacts = joblib.load(str(MODEL_PATH))
model = artifacts['model']
feature_names = artifacts['feature_names']
model_auc = artifacts['auc']

explainer = shap.TreeExplainer(model)

def risk_label(score: float) -> str:
    if score < 0.35: return "Low"
    if score < 0.65: return "Moderate"
    return "High"

@app.get("/")
def root():
    return {"status": "Clinical Risk API running", "model_auc": model_auc}

@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientInput):
    values = [[getattr(patient, f) for f in feature_names]]
    df_input = pd.DataFrame(values, columns=feature_names)
    risk_score = float(model.predict_proba(df_input)[0, 1])
    shap_vals = explainer.shap_values(df_input)[0]
    shap_dict = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_vals)
    }
    return PredictionResult(
        risk_score=round(risk_score, 4),
        risk_label=risk_label(risk_score),
        shap_values=shap_dict,
        model_auc=model_auc
    )