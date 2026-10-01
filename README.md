# clinical-risk-tool

A demo web app that estimates a patient's risk of heart disease from 13 clinical
measurements and shows which factors drove the estimate.

> **Not for clinical use.** The model is trained on 297 patients from the 1988
> Cleveland heart disease dataset and has not been clinically validated. Do not
> use it to make decisions about real patients.

## How it fits together

| Part | Folder | What it does |
| --- | --- | --- |
| Frontend | [frontend/](frontend/) | React + Vite form and results page |
| Backend | [backend/](backend/) | FastAPI service that validates input, scores the patient and returns SHAP values |
| Training | [ml/](ml/) | Script that trains the XGBoost model and writes `model.joblib` |

## Run it

You need Docker Desktop running. From the project root:

```
docker compose up --build -d
```

- App: http://localhost:5174
- API: http://localhost:8001 (interactive docs at http://localhost:8001/docs)

The trained model is committed, so there is nothing to train first.

## Run without Docker

Use Python 3.11, which is what the Docker images use. The pinned versions in
the requirements files do not install on Python 3.13.

```
# Terminal 1: backend
cd backend
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# Terminal 2: frontend
cd frontend
npm install
npm run dev
```

## Configuration

| Variable | Used by | Default | Purpose |
| --- | --- | --- | --- |
| `MODEL_PATH` | backend | `ml/model.joblib` in this repo | Where to load the model from |
| `CORS_ORIGINS` | backend | `http://localhost:5173,http://localhost:5174` | Comma-separated origins allowed to call the API |
| `VITE_API_URL` | frontend | `http://localhost:8001` | API base URL, fixed at build time |

For Docker, set these in [docker-compose.yml](docker-compose.yml). For local
frontend work, copy [frontend/.env.example](frontend/.env.example) to
`frontend/.env.local`.

## Tests

```
docker compose run --rm --no-deps backend sh -c "pip install -q -r requirements-dev.txt && python -m pytest"
```

The tests in [backend/tests/](backend/tests/) cover valid predictions,
rejection of out-of-range and malformed input, and the risk label thresholds.

## Retrain the model

```
docker compose run --rm train
docker compose restart backend
```

This rewrites `ml/model.joblib` and `ml/metrics.json`. Train in Docker rather
than a local virtualenv so the model is saved with the same library versions
the backend loads it with.

## The model

XGBoost classifier on the 13 features of the
[UCI Cleveland heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(297 patients after dropping rows with missing values, 46% with disease).

Its raw output is calibrated with Platt scaling so the score can be read as a
probability. The risk label is Low below 35%, Moderate from 35% to 65%, and
High above that.

Results from 5-fold cross-validation, with calibration fitted inside each fold:

| Model | AUC (mean ± std) | Brier score (lower is better) |
| --- | --- | --- |
| XGBoost, uncalibrated | 0.879 ± 0.046 | 0.146 |
| XGBoost, calibrated (deployed) | 0.879 ± 0.046 | 0.138 |
| Logistic regression baseline | 0.898 ± 0.042 | 0.126 |

The logistic regression baseline does slightly better than XGBoost on both
measures. The difference is within the fold-to-fold spread, but it means
XGBoost is not buying any accuracy on a dataset this small.

The risk-factor chart shows SHAP values in log-odds: a positive bar pushes the
risk up, a negative bar pushes it down. They are not percentage points.

## API

`POST /predict` takes a JSON body with these fields. Anything outside the
allowed values is rejected with HTTP 422.

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `age` | Age in years | 18 to 100 |
| `sex` | Sex | 0 = female, 1 = male |
| `cp` | Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal, 4 = asymptomatic |
| `trestbps` | Resting blood pressure (mmHg) | 80 to 220 |
| `chol` | Serum cholesterol (mg/dl) | 100 to 600 |
| `fbs` | Fasting blood sugar above 120 mg/dl | 0 = no, 1 = yes |
| `restecg` | Resting ECG | 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| `thalach` | Maximum heart rate (bpm) | 60 to 220 |
| `exang` | Exercise-induced angina | 0 = no, 1 = yes |
| `oldpeak` | ST depression induced by exercise | 0 to 7 |
| `slope` | Slope of peak exercise ST segment | 1 = upsloping, 2 = flat, 3 = downsloping |
| `ca` | Major vessels colored by fluoroscopy | 0 to 3 |
| `thal` | Thallium stress test | 3 = normal, 6 = fixed defect, 7 = reversible defect |

Response:

```json
{
  "risk_score": 0.1044,
  "risk_label": "Low",
  "shap_values": { "age": -0.56, "sex": 0.2316, "...": 0 },
  "model_auc": 0.8794,
  "model_auc_std": 0.0457
}
```

`GET /` returns a status message and the model's cross-validation metrics.
