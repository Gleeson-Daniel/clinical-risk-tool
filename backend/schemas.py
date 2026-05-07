from pydantic import BaseModel, Field

class PatientInput(BaseModel):
    age: float = Field(..., example=52, description="Age in years")
    sex: float = Field(..., example=1, description="1=male, 0=female")
    cp: float = Field(..., example=2, description="Chest pain: 1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic")
    trestbps: float = Field(..., example=130, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., example=245, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., example=0, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: float = Field(..., example=0, description="Resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy")
    thalach: float = Field(..., example=160, description="Maximum heart rate achieved")
    exang: float = Field(..., example=0, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., example=1.4, description="ST depression induced by exercise")
    slope: float = Field(..., example=2, description="Slope of peak exercise ST segment (1/2/3)")
    ca: float = Field(..., example=0, description="Number of major vessels colored (0-3)")
    thal: float = Field(..., example=3, description="Thal: 3=normal, 6=fixed defect, 7=reversable defect")

class PredictionResult(BaseModel):
    risk_score: float        # 0.0–1.0
    risk_label: str          # "Low", "Moderate", "High"
    shap_values: dict        # {feature: shap_value}
    model_auc: float
