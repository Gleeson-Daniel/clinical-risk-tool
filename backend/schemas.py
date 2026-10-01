from typing import Literal

from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    age: float = Field(..., ge=18, le=100, examples=[52], description="Age in years")
    sex: int = Field(..., ge=0, le=1, examples=[1], description="1=male, 0=female")
    cp: int = Field(..., ge=1, le=4, examples=[2], description="Chest pain: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic")
    trestbps: float = Field(..., ge=80, le=220, examples=[130], description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=100, le=600, examples=[245], description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, examples=[0], description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., ge=0, le=2, examples=[0], description="Resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy")
    thalach: float = Field(..., ge=60, le=220, examples=[160], description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, examples=[0], description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, le=7, examples=[1.4], description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, examples=[2], description="Slope of peak exercise ST segment (1/2/3)")
    ca: int = Field(..., ge=0, le=3, examples=[0], description="Number of major vessels colored (0-3)")
    thal: Literal[3, 6, 7] = Field(..., examples=[3], description="Thal: 3=normal, 6=fixed defect, 7=reversable defect")

class PredictionResult(BaseModel):
    risk_score: float               # 0.0–1.0, calibrated probability
    risk_label: str                 # "Low", "Moderate", "High"
    shap_values: dict[str, float]   # {feature: contribution to log-odds}
    model_auc: float                # 5-fold cross-validated AUC (mean)
    model_auc_std: float            # spread of AUC across the 5 folds
