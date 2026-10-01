import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import shap
import joblib
from schemas import PatientInput, PredictionResult

# Defaults work both in Docker (/app/main.py -> /ml) and from a local checkout.
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    Path(__file__).resolve().parent.parent / "ml" / "model.joblib"
))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:5174"
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="Clinical Risk API")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

artifacts = joblib.load(str(MODEL_PATH))
model = artifacts['model']
feature_names = artifacts['feature_names']
metrics = artifacts['metrics']
# Platt scaling: calibrated log-odds = cal_a * raw log-odds + cal_b
cal_a = artifacts['calibration']['a']
cal_b = artifacts['calibration']['b']

explainer = shap.TreeExplainer(model)

def risk_label(score: float) -> str:
    if score < 0.35: return "Low"
    if score < 0.65: return "Moderate"
    return "High"

@app.get("/")
def root():
    return {"status": "Clinical Risk API running", "metrics": metrics}

@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientInput):
    values = [[getattr(patient, f) for f in feature_names]]
    df_input = pd.DataFrame(values, columns=feature_names, dtype=float)
    margin = float(model.predict(df_input, output_margin=True)[0])
    risk_score = float(1 / (1 + np.exp(-(cal_a * margin + cal_b))))
    # SHAP values are in raw log-odds; scaling by cal_a keeps them
    # consistent with the calibrated score.
    shap_vals = explainer.shap_values(df_input)[0] * cal_a
    shap_dict = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_vals)
    }
    return PredictionResult(
        risk_score=round(risk_score, 4),
        risk_label=risk_label(risk_score),
        shap_values=shap_dict,
        model_auc=metrics['cv_auc_mean'],
        model_auc_std=metrics['cv_auc_std']
    )
