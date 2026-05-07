import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import joblib
import urllib.request

# ── 1. Download UCI Heart Disease dataset ──────────────────────────────
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/heart-disease/processed.cleveland.data")

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

print("Downloading dataset...")
urllib.request.urlretrieve(URL, "heart.csv")
df = pd.read_csv("heart.csv", names=COLUMNS, na_values='?')

# ── 2. Clean ──────────────────────────────────────────────────────────
df.dropna(inplace=True)
df['target'] = (df['target'] > 0).astype(int)  # Binary: 0=no disease, 1=disease

print(f"Dataset: {len(df)} patients, {df['target'].mean():.1%} disease prevalence")

# ── 3. Split ──────────────────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train XGBoost ──────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# ── 5. Evaluate ──────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

print(f"\n=== Model Performance ===")
print(f"Test AUC:       {auc:.4f}")
print(f"CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ── 6. Save model + feature names ────────────────────────────────────
joblib.dump({
    'model': model,
    'feature_names': list(X.columns),
    'auc': round(auc, 4)
}, 'model.joblib')

print("\nModel saved to model.joblib")