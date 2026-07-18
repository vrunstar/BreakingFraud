import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

MODEL_DIR = Path("backend/models")

model = joblib.load(MODEL_DIR / "xgb_class.pkl")
X_test_scaled = joblib.load(MODEL_DIR / "X_test_scaled.pkl")
y_test = joblib.load(MODEL_DIR / "y_test.pkl")

fraud_rows = X_test_scaled[y_test == 1]
legit_rows = X_test_scaled[y_test == 0]
legit_sample = legit_rows.sample(min(900, len(legit_rows)), random_state=42)
X_sample = pd.concat([fraud_rows, legit_sample])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap_fraud = shap_values[:, :, 1] if np.ndim(shap_values) == 3 else shap_values

mean_abs = np.abs(shap_fraud).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:15]

for i in order:
    print(f"{X_sample.columns[i]:35s} {mean_abs[i]:.4f}")