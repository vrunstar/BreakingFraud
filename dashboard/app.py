import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_raw_data
from src.preprocessing import split_data, scale_features, apply_smote
from src.models import train_random_forest
from src.cost_matrix import find_optimal_threshold
from src.stat_tests import run_all_feature_tests
from src.viz import (
    plot_confusion_matrix,
    plot_cost_curve,
    plot_shap_importance,
    plot_single_explanation,
    plot_feature_test_table,
)

st.set_page_config(page_title="Breaking Fraud", layout="wide")


@st.cache_resource
def load_and_train():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    rf_smote = train_random_forest(X_train_smote, y_train_smote)
    rf_class_weight = train_random_forest(X_train_scaled, y_train, class_weight='balanced')

    return {
        "df": df,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "rf_smote": rf_smote,
        "rf_class_weight": rf_class_weight,
    }


@st.cache_data
def get_feature_test_results(df):
    return run_all_feature_tests(df)


@st.cache_resource
def compute_shap(_model, model_name, X_sample):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


data = load_and_train()

st.title("Breaking Fraud")
st.caption("Credit card fraud detection — model comparison, cost-based thresholding, and explainability")

# --- Sidebar controls ---
model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest (SMOTE)", "Random Forest (class weight)"]
)
threshold = st.sidebar.slider(
    "Threshold", min_value=0.01, max_value=0.99, value=0.5, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.caption("Cost assumptions")
fn_cost = st.sidebar.number_input("False negative cost (₹)", value=5000, step=100)
fp_cost = st.sidebar.number_input("False positive cost (₹)", value=50, step=10)

model = data["rf_smote"] if model_choice == "Random Forest (SMOTE)" else data["rf_class_weight"]

# --- Compute predictions at chosen threshold ---
y_proba = model.predict_proba(data["X_test"])[:, 1]
y_pred = (y_proba >= threshold).astype(int)

precision = precision_score(data["y_test"], y_pred)
recall = recall_score(data["y_test"], y_pred)
f1 = f1_score(data["y_test"], y_pred)
cm = confusion_matrix(data["y_test"], y_pred)
tn, fp, fn, tp = cm.ravel()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Cost analysis", "Feature insights", "Explain transaction", "Live feed"]
)

# ============= TAB 1: OVERVIEW =============
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Precision", f"{precision:.3f}")
    with m2:
        st.metric("Recall", f"{recall:.3f}")
    with m3:
        st.metric("F1 Score", f"{f1:.3f}")
    with m4:
        st.metric("Fraud caught", f"{tp} / {tp + fn}")

    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

# ============= TAB 2: COST ANALYSIS =============
with tab2:
    best_threshold, best_cost, thresholds, costs = find_optimal_threshold(
        data["y_test"], y_proba, fn_cost=fn_cost, fp_cost=fp_cost
    )
    current_cost = (fn * fn_cost) + (fp * fp_cost)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Optimal threshold", f"{best_threshold:.2f}")
    with c2:
        st.metric("Cost at optimal", f"₹{best_cost:,.0f}")
    with c3:
        st.metric("Cost at current threshold", f"₹{current_cost:,.0f}")

    st.plotly_chart(
        plot_cost_curve(thresholds, costs, best_threshold),
        use_container_width=True
    )
    st.caption(
        f"Assumes each missed fraud costs ₹{fn_cost:,.0f} and each false alarm costs ₹{fp_cost:,.0f}. "
        "Adjust these in the sidebar to see the optimal threshold shift."
    )

# ============= TAB 3: FEATURE INSIGHTS =============
with tab3:
    left, right = st.columns(2)

    X_test = data["X_test"]
    y_test = data["y_test"]

    fraud_rows = X_test[y_test == 1]
    legit_sample = X_test[y_test == 0].sample(min(900, len(X_test)), random_state=42)
    X_sample = pd.concat([fraud_rows, legit_sample])

    explainer, shap_values = compute_shap(model, model_choice, X_sample)
    shap_fraud = shap_values[:, :, 1] if np.ndim(shap_values) == 3 else shap_values

    with left:
        st.plotly_chart(
            plot_shap_importance(shap_fraud, X_sample.columns.tolist()),
            use_container_width=True
        )

    with right:
        results_df = get_feature_test_results(data["df"])
        st.markdown("**Top features by Mann-Whitney p-value**")
        st.plotly_chart(plot_feature_test_table(results_df), use_container_width=True)

    st.caption(
        "Comparing statistical significance (right) against SHAP importance (left) — "
        "features appearing at the top of both are the strongest, most trustworthy fraud signals."
    )

# ============= TAB 4: EXPLAIN TRANSACTION =============
with tab4:
    y_sample = y_test.loc[X_sample.index]
    fraud_positions = np.where(y_sample.values == 1)[0]

    if len(fraud_positions) == 0:
        st.info("No fraud cases available in the current sample.")
    else:
        selected = st.selectbox(
            "Pick a fraud transaction to explain",
            options=list(range(len(fraud_positions))),
            format_func=lambda i: f"Transaction #{fraud_positions[i]}"
        )
        idx = fraud_positions[selected]

        values = shap_fraud[idx]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        prediction = base_value + values.sum()

        st.plotly_chart(
            plot_single_explanation(X_sample.columns.tolist(), values, base_value, prediction),
            use_container_width=True
        )
        st.caption(
            "Red bars push the prediction toward fraud, blue bars push away from it. "
            "Bars are sorted by absolute contribution size."
        )

# ============= TAB 5: LIVE FEED =============
with tab5:
    st.caption("Simulates a stream of incoming transactions being scored in real time.")

    if st.button("Generate next batch"):
        batch = data["X_test"].sample(10)
        batch_proba = model.predict_proba(batch)[:, 1]
        batch_flag = (batch_proba >= threshold).astype(int)

        display_df = pd.DataFrame({
            "Amount": batch["Amount"].values,
            "Fraud probability": batch_proba.round(3),
            "Flagged": ["Fraud" if f == 1 else "Legit" for f in batch_flag],
        })
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Click the button to simulate an incoming batch of transactions.")

st.markdown("---")
st.caption("Data: Kaggle Credit Card Fraud Detection dataset · Model: Random Forest")