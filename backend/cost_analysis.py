import joblib
from pathlib import Path
from src.cost_matrix import find_optimal_threshold

MODEL_PATH = Path("models")

X_test_scaled = joblib.load(MODEL_PATH / "X_test_scaled.pkl")
y_test = joblib.load(MODEL_PATH / "y_test.pkl")

for name in ["xgb_class", "xgb_smote", "rf_class", "rf_smote", "linreg"]:
    model = joblib.load(MODEL_PATH / f"{name}.pkl")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    best_threshold, best_cost, thresholds, costs = find_optimal_threshold(
        y_test, y_proba, fn_cost=10000, fp_cost=500
    )

    print(f"--- {name} ---")
    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"  Cost at optimal:   ₹{best_cost:,.0f}")
    print()