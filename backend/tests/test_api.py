import math

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

TYPICAL = {
    "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 245,
    "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
    "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 3,
}
LOW_RISK = {
    "age": 35, "sex": 0, "cp": 3, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
    "oldpeak": 0, "slope": 1, "ca": 0, "thal": 3,
}
HIGH_RISK = {
    "age": 65, "sex": 1, "cp": 4, "trestbps": 160, "chol": 300,
    "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1,
    "oldpeak": 3, "slope": 2, "ca": 3, "thal": 7,
}


def test_root_reports_status_and_metrics():
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "Clinical Risk API running"
    assert 0.5 < body["metrics"]["cv_auc_mean"] <= 1


def test_predict_returns_score_label_and_shap_values():
    res = client.post("/predict", json=TYPICAL)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_label"] == main.risk_label(body["risk_score"])
    assert set(body["shap_values"]) == set(TYPICAL)
    assert body["model_auc"] == main.metrics["cv_auc_mean"]
    assert body["model_auc_std"] == main.metrics["cv_auc_std"]


def test_known_low_and_high_risk_patients():
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()
    assert low["risk_label"] == "Low"
    assert high["risk_label"] == "High"
    assert low["risk_score"] < high["risk_score"]


def test_shap_values_add_up_to_the_calibrated_score():
    body = client.post("/predict", json=TYPICAL).json()
    base = float(main.explainer.expected_value)
    log_odds = main.cal_a * base + main.cal_b + sum(body["shap_values"].values())
    score = 1 / (1 + math.exp(-log_odds))
    assert score == pytest.approx(body["risk_score"], abs=0.01)


@pytest.mark.parametrize("field,value", [
    ("age", -5),          # below range
    ("age", 150),         # above range
    ("chol", 0),          # below range
    ("oldpeak", 20),      # above range
    ("cp", 2.7),          # categorical must be a whole number
    ("cp", 5),            # not a chest pain type
    ("sex", 2),           # not 0/1
    ("thal", 4),          # only 3, 6, 7 are valid
    ("age", "old"),       # not a number
])
def test_predict_rejects_invalid_values(field, value):
    res = client.post("/predict", json={**TYPICAL, field: value})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", field]


def test_predict_rejects_missing_field():
    patient = {k: v for k, v in TYPICAL.items() if k != "thal"}
    res = client.post("/predict", json=patient)
    assert res.status_code == 422


@pytest.mark.parametrize("score,label", [
    (0.0, "Low"), (0.349, "Low"),
    (0.35, "Moderate"), (0.649, "Moderate"),
    (0.65, "High"), (1.0, "High"),
])
def test_risk_label_thresholds(score, label):
    assert main.risk_label(score) == label
