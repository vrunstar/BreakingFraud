# Breaking Fraud — Credit Card Fraud Detection

An end-to-end fraud detection pipeline covering SQL, statistical hypothesis testing, imbalanced classification, cost-based decision thresholds, and model explainability — packaged into an interactive Streamlit dashboard.

**[Live Dashboard →](#)** *(add your deployed Render/Streamlit Cloud link here)*

---

## Overview

Credit card fraud is a classic — and genuinely hard — imbalanced classification problem: only 0.17% of transactions in this dataset are fraudulent. This project doesn't just train a model on it; it walks the full analytical process a fraud/risk analyst would actually go through:

1. Prove features are statistically meaningful before modeling
2. Compare imbalance-handling strategies (SMOTE vs class weighting) across two model types
3. Move past a default 0.5 threshold to one chosen by real business cost tradeoffs
4. Explain individual model decisions with SHAP, not just report a black-box score

Full write-up of results: [`reports/findings.md`](reports/findings.md)

---

## Key Results

| Metric | Value |
|---|---|
| Best model | Random Forest + class weighting |
| Precision / Recall (default threshold) | 0.336 / 0.878 |
| PR-AUC | 0.654 |
| Cost reduction from threshold tuning | Up to ~28% (cost-ratio dependent) |
| Top predictive features | V14, V4, V12, V11, V10 — confirmed independently by both Mann-Whitney testing and SHAP |

---

## Tech Stack

- **Data**: pandas, SQL (SQLAlchemy)
- **Stats**: SciPy (Mann-Whitney U test)
- **ML**: scikit-learn (Logistic Regression, Random Forest), imbalanced-learn (SMOTE)
- **Explainability**: SHAP (TreeExplainer)
- **Visualization / Dashboard**: Plotly, Streamlit
- **Deployment**: Render

---

## Project Structure

```
fraud-detection-analysis/
├── data/raw/              # creditcard.csv (not committed — see Setup)
├── notebooks/              # EDA, stats, segmentation exploration
├── src/
│   ├── data_loader.py       # load + schema validation
│   ├── preprocessing.py     # split, scale, SMOTE
│   ├── stat_tests.py         # Mann-Whitney feature testing
│   ├── models.py             # Logistic Regression / Random Forest training + evaluation
│   ├── cost_matrix.py        # cost-based threshold optimization
│   ├── explainability.py     # SHAP value computation
│   └── viz.py                 # Plotly chart functions
├── dashboard/
│   └── app.py                # Streamlit app — 5 tabs (Overview, Cost, Features, Explain, Live feed)
├── reports/
│   └── findings.md            # consolidated results and conclusions
└── requirements.txt
```

---

## Dashboard

Five tabs, all driven by live sidebar controls (model choice, threshold, cost assumptions):

- **Overview** — precision/recall/F1, confusion matrix, live at your chosen threshold
- **Cost analysis** — editable false-negative/false-positive costs, optimal threshold calculation, cost-vs-threshold curve
- **Feature insights** — SHAP global importance next to the Mann-Whitney significance table
- **Explain transaction** — pick any caught fraud case, see exactly which features drove that specific prediction
- **Live feed** — simulated real-time transaction scoring

*(Add 1–2 screenshots here once you're happy with the visual polish)*

---

## Setup

```bash
git clone <your-repo-url>
cd fraud-detection-analysis

python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows

pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

Run the dashboard:
```bash
streamlit run dashboard/app.py
```

---

## Methodology Notes

- **Data leakage avoided**: scaling and SMOTE are both fit only on training data; the test set is never resampled and stays at its true 0.17% fraud rate for honest evaluation.
- **PR-AUC over accuracy/ROC-AUC**: with this level of imbalance, accuracy is meaningless (99.83% by predicting "not fraud" always) and ROC-AUC can look misleadingly good. PR-AUC is the primary comparison metric here.
- **Cost-based thresholding**: the "right" decision threshold isn't 0.5 by default — it depends on the real cost of a missed fraud vs. a false alarm. See `reports/findings.md` for the sensitivity analysis across cost ratios.

---

## Data Source

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (anonymized European cardholder transactions, PCA-transformed features).