import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

from backend.src.viz import (
    plot_confusion_matrix,
    plot_cost_curve,
    plot_shap_importance,
    plot_single_explanation,
)
from backend.src.cost_matrix import find_optimal_threshold
from backend.src.styles import CUSTOM_CSS

st.set_page_config(page_title="Breaking Fraud", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


MODEL_DIR = Path("backend/models")

MODEL_META = {
    "Random Forest (SMOTE)":        {"key": "rf_smote",            "file": "rf_smote.pkl",   "badge": "badge-rf"},
    "Random Forest (class weight)": {"key": "rf_class_weight",     "file": "rf_class.pkl",   "badge": "badge-rf"},
    "XGBoost (SMOTE)":              {"key": "xgb_smote",           "file": "xgb_smote.pkl",  "badge": "badge-xgb"},
    "XGBoost (class weight)":       {"key": "xgb_class_weight",    "file": "xgb_class.pkl",  "badge": "badge-xgb"},
    "Logistic Regression":          {"key": "logreg_class_weight", "file": "linreg.pkl",     "badge": "badge-lr"},
}


@st.cache_resource
def load_and_train():
    """Loads pre-trained models + preprocessor + test set from disk.
    Training now happens once in backend/train.py, not on every app run."""
    models = {
        meta["key"]: joblib.load(MODEL_DIR / meta["file"])
        for meta in MODEL_META.values()
    }
    X_test_scaled = joblib.load(MODEL_DIR / "X_test_scaled.pkl")
    y_test = joblib.load(MODEL_DIR / "y_test.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

    return {
        "X_test": X_test_scaled,
        "y_test": y_test,
        "models": models,
        "preprocessor": preprocessor,
    }


@st.cache_resource
def compute_shap(_model, model_name, X_sample):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


@st.cache_data
def get_optimal_threshold(model_name, y_proba_hash, fn_cost, fp_cost, _y_test, _y_proba):
    return find_optimal_threshold(_y_test, _y_proba, fn_cost=fn_cost, fp_cost=fp_cost)


@st.cache_data
def get_all_model_summaries(_models, _X_test, _y_test, threshold):
    rows = []
    for label, meta in MODEL_META.items():
        m = _models[meta["key"]]
        proba = m.predict_proba(_X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        rows.append({
            "Model": label,
            "Precision": precision_score(_y_test, pred, zero_division=0),
            "Recall": recall_score(_y_test, pred, zero_division=0),
            "F1": f1_score(_y_test, pred, zero_division=0),
            "PR-AUC": average_precision_score(_y_test, proba),
        })
    return pd.DataFrame(rows)


def inverse_transform_amount(preprocessor, batch):
    """Undo scaling on TransactionAmt so the Live feed tab can show real ₹ values
    instead of scaled ones like -0.34."""
    num_pipeline = preprocessor.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scale"]
    amt_col_idx = list(preprocessor.transformers_[0][2]).index("TransactionAmt")

    scaled_amt = batch[["num__TransactionAmt"]].values
    dummy = np.zeros((len(batch), scaler.mean_.shape[0]))
    dummy[:, amt_col_idx] = scaled_amt[:, 0]
    real_values = scaler.inverse_transform(dummy)[:, amt_col_idx]
    return real_values


data = load_and_train()

st.markdown("""
<div class="hero">
    <h1>BREAKING FRAUD</h1>
    <p>Credit card fraud detection — 5-model comparison, cost-based thresholding, and explainability</p>
</div>
""", unsafe_allow_html=True)

# --- Controls (moved from sidebar to center) ---
st.markdown("#### Controls")

# Row 1: Model (pills) + Threshold (slider)
r1c1, r1c2 = st.columns(2)
with r1c1:
    model_choice = st.radio(
        "Model",
        options=list(MODEL_META.keys()),
        horizontal=True,
    )

with r1c2:
    threshold = st.slider("Threshold", min_value=0.01, max_value=0.99, value=0.5, step=0.01)

# Row 2: False negative cost + False positive cost (both int inputs)
r2c1, r2c2 = st.columns(2)
with r2c1:
    fn_cost = st.number_input("False negative cost (₹)", value=5000, step=100)
with r2c2:
    fp_cost = st.number_input("False positive cost (₹)", value=50, step=10)

st.markdown("---")
#


model = data["models"][MODEL_META[model_choice]["key"]]

# --- Compute predictions at chosen threshold ---
y_proba = model.predict_proba(data["X_test"])[:, 1]
y_pred = (y_proba >= threshold).astype(int)

precision = precision_score(data["y_test"], y_pred, zero_division=0)
recall = recall_score(data["y_test"], y_pred, zero_division=0)
f1 = f1_score(data["y_test"], y_pred, zero_division=0)
prauc = average_precision_score(data["y_test"], y_proba)
cm = confusion_matrix(data["y_test"], y_pred)
tn, fp, fn, tp = cm.ravel()

# --- Shared sample + SHAP, built once, used by both Feature Insights and Explain Transaction ---
X_test = data["X_test"]
y_test = data["y_test"]

fraud_rows = X_test[y_test == 1]
legit_rows = X_test[y_test == 0]
legit_sample = legit_rows.sample(min(900, len(legit_rows)), random_state=42)
X_sample = pd.concat([fraud_rows, legit_sample])
y_sample = y_test.loc[X_sample.index]

explainer, shap_values = compute_shap(model, model_choice, X_sample)
shap_fraud = shap_values[:, :, 1] if np.ndim(shap_values) == 3 else shap_values

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Model comparison", "Overview", "Cost analysis", "Feature insights", "Explain transaction", "Live feed"]
)

# ============= TAB 0: MODEL COMPARISON =============
with tab0:
    st.markdown("**All 5 models, evaluated at the current threshold**")
    summary_df = get_all_model_summaries(data["models"], X_test, y_test, threshold)
    summary_df_display = summary_df.copy()
    for col in ["Precision", "Recall", "F1", "PR-AUC"]:
        summary_df_display[col] = summary_df_display[col].round(3)

    best_prauc_model = summary_df.loc[summary_df["PR-AUC"].idxmax(), "Model"]

    st.dataframe(
        summary_df_display.style.highlight_max(
            subset=["Precision", "Recall", "F1", "PR-AUC"], color="#2B1810"
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Best PR-AUC at this threshold: **{best_prauc_model}**. "
               "PR-AUC is the most reliable metric here since fraud is rare — "
               "it isn't sensitive to threshold choice the way precision/recall/F1 are.")

# ============= TAB 1: OVERVIEW =============
with tab1:
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Precision", f"{precision:.3f}")
    with m2:
        st.metric("Recall", f"{recall:.3f}")
    with m3:
        st.metric("F1 Score", f"{f1:.3f}")
    with m4:
        st.metric("PR-AUC", f"{prauc:.3f}")
    with m5:
        st.metric("Fraud caught", f"{tp} / {tp + fn}")

    st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

# ============= TAB 2: COST ANALYSIS =============
with tab2:
    y_proba_hash = hash(y_proba.tobytes())
    best_threshold, best_cost, thresholds, costs = get_optimal_threshold(
        model_choice, y_proba_hash, fn_cost, fp_cost, data["y_test"], y_proba
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
    st.plotly_chart(
        plot_shap_importance(shap_fraud, X_sample.columns.tolist()),
        use_container_width=True
    )
    st.info(
        "Statistical significance testing (Mann-Whitney / chi-square per feature) "
        "is being updated for the new column set — coming back shortly."
    )

# ============= TAB 4: EXPLAIN TRANSACTION =============
with tab4:
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

        prediction_label = "Flagged" if prediction >= 0.5 else "Cleared"
        stamp_class = "stamp-flagged" if prediction >= 0.5 else "stamp-cleared"
        st.markdown(f'<span class="stamp {stamp_class}">{prediction_label}</span>', unsafe_allow_html=True)

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
        real_amounts = inverse_transform_amount(data["preprocessor"], batch)

        for i in range(len(batch)):
            stamp_class = "stamp-flagged" if batch_flag[i] == 1 else "stamp-cleared"
            stamp_text = "Flagged" if batch_flag[i] == 1 else "Cleared"
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"**₹{real_amounts[i]:,.2f}**")
            with c2:
                st.markdown(f"Fraud probability: `{batch_proba[i]:.3f}`")
            with c3:
                st.markdown(f'<span class="stamp {stamp_class}">{stamp_text}</span>', unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("Click the button to simulate an incoming batch of transactions.")

st.markdown("---")
st.caption("Data: IEEE-CIS Fraud Detection dataset · Models: Random Forest · XGBoost · Logistic Regression")