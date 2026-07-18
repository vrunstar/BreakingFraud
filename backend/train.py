from src.data_loader import load_raw_data, basic_schema_check
from src.preprocessing import split_data, scale_features, apply_smote
from src.models import train_baseline, train_gradient, train_random_forest, evaluate_model
import joblib
from pathlib import Path

df = load_raw_data()
basic_schema_check(df)

X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled, preprocessor = scale_features(X_train, X_test)
X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

models = {
    "rf_smote": train_random_forest(X_train_smote, y_train_smote),
    "rf_class": train_random_forest(X_train_scaled, y_train, class_weight="balanced"),
    "xgb_smote": train_gradient(X_train_smote, y_train_smote),
    "xgb_class": train_gradient(X_train_scaled, y_train, scale_pos_weight=scale_pos_weight),
    "linreg": train_baseline(X_train_scaled, y_train, class_weight="balanced"),
}

MODEL_PATH = Path("backend/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    joblib.dump(model, MODEL_PATH / f"{name}.pkl")
    print(f"Saved {name}.pkl")

joblib.dump(preprocessor, MODEL_PATH / "preprocessor.pkl")
joblib.dump(X_test_scaled, MODEL_PATH / "X_test_scaled.pkl")
joblib.dump(y_test, MODEL_PATH / "y_test.pkl")

for name, model in models.items():
    print(f"----- {name} -----")
    results = evaluate_model(model, X_test_scaled, y_test)
    for k, v in results.items():
        if k != "Confusion Matrix":
            print(f" {k} : {v:.4f}")
        else:
            print(f" {k} : \n{v}")
    print()