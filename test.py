from src.data_loader import load_raw_data
from src.preprocessing import split_data, scale_features
from src.models import train_random_forest
from src.explainability import get_shap_values, plot_global_importance, explain_single_prediction
import numpy as np
import pandas as pd

df = load_raw_data()
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

model4 = train_random_forest(X_train_scaled, y_train, class_weight='balanced')

fraud_rows = X_test_scaled[y_test == 1]
legit_sample = X_test_scaled[y_test == 0].sample(900, random_state=42)

X_sample = pd.concat([fraud_rows, legit_sample])
y_sample = y_test.loc[X_sample.index]

explainer, shap_values = get_shap_values(model4, X_sample)

# Find an actual fraud case in the sample
fraud_indices = np.where(y_sample.values == 1)[0]
print(f"Found {len(fraud_indices)} fraud cases in this sample")

if len(fraud_indices) > 0:
    explain_single_prediction(explainer, shap_values, X_sample, index=fraud_indices[0])