import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "heart.csv"
MODEL_PATH = HERE / "model.joblib"
METRICS_PATH = HERE / "metrics.json"

# ── 1. Load UCI Heart Disease dataset (download if missing) ───────────
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/heart-disease/processed.cleveland.data")

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
CATEGORICAL = ['cp', 'restecg', 'slope', 'thal']

if not DATA_PATH.exists():
    print("Downloading dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)
df = pd.read_csv(DATA_PATH, names=COLUMNS, na_values='?')

# ── 2. Clean ──────────────────────────────────────────────────────────
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['target'] = (df['target'] > 0).astype(int)  # Binary: 0=no disease, 1=disease

print(f"Dataset: {len(df)} patients, {df['target'].mean():.1%} disease prevalence")

X = df.drop('target', axis=1)
y = df['target']


# ── 3. Models ─────────────────────────────────────────────────────────
def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=1  # ~300 rows: extra threads only add contention
    )


def make_baseline():
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    return make_pipeline(
        ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
            ('num', StandardScaler(), numeric),
        ]),
        LogisticRegression(max_iter=1000)
    )


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def oof_margins(X, y, seed):
    """Out-of-fold XGBoost log-odds, so the calibrator never sees a
    prediction for a patient the model was trained on."""
    margins = np.zeros(len(X))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(X, y):
        m = make_xgb().fit(X.iloc[train_idx], y.iloc[train_idx])
        margins[test_idx] = m.predict(X.iloc[test_idx], output_margin=True)
    return margins


def fit_platt(margins, y):
    """Platt scaling: calibrated log-odds = a * margin + b."""
    lr = LogisticRegression(C=1e6).fit(margins.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ── 4. Evaluate with 5-fold CV (calibration nested inside each fold) ──
raw_prob = np.zeros(len(X))
cal_prob = np.zeros(len(X))
base_prob = np.zeros(len(X))
xgb_aucs, base_aucs = [], []

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in outer.split(X, y):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    a, b = fit_platt(oof_margins(X_train, y_train, seed=0), y_train)
    model = make_xgb().fit(X_train, y_train)
    margin = model.predict(X_test, output_margin=True)
    raw_prob[test_idx] = sigmoid(margin)
    cal_prob[test_idx] = sigmoid(a * margin + b)
    xgb_aucs.append(roc_auc_score(y_test, cal_prob[test_idx]))

    baseline = make_baseline().fit(X_train, y_train)
    base_prob[test_idx] = baseline.predict_proba(X_test)[:, 1]
    base_aucs.append(roc_auc_score(y_test, base_prob[test_idx]))

metrics = {
    'n_patients': int(len(df)),
    'prevalence': round(float(y.mean()), 4),
    'cv_auc_mean': round(float(np.mean(xgb_aucs)), 4),
    'cv_auc_std': round(float(np.std(xgb_aucs)), 4),
    'brier_uncalibrated': round(float(brier_score_loss(y, raw_prob)), 4),
    'brier_calibrated': round(float(brier_score_loss(y, cal_prob)), 4),
    'baseline_cv_auc_mean': round(float(np.mean(base_aucs)), 4),
    'baseline_cv_auc_std': round(float(np.std(base_aucs)), 4),
    'baseline_brier': round(float(brier_score_loss(y, base_prob)), 4),
}

print("\n=== Model Performance (5-fold CV) ===")
print(f"XGBoost AUC:             {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
print(f"Logistic regression AUC: {metrics['baseline_cv_auc_mean']:.4f} ± {metrics['baseline_cv_auc_std']:.4f}")
print("\nBrier score (lower is better)")
print(f"XGBoost uncalibrated:    {metrics['brier_uncalibrated']:.4f}")
print(f"XGBoost calibrated:      {metrics['brier_calibrated']:.4f}")
print(f"Logistic regression:     {metrics['baseline_brier']:.4f}")

# ── 5. Fit final model + calibrator on all data ───────────────────────
cal_a, cal_b = fit_platt(oof_margins(X, y, seed=42), y)
model = make_xgb().fit(X, y)

# ── 6. Save model + calibration + metrics ─────────────────────────────
joblib.dump({
    'model': model,
    'feature_names': list(X.columns),
    'calibration': {'a': cal_a, 'b': cal_b},
    'metrics': metrics,
}, MODEL_PATH)
METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

print(f"\nModel saved to {MODEL_PATH.name}, metrics to {METRICS_PATH.name}")
