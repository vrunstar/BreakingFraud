# Breaking Fraud

A cost-aware, explainable credit card fraud detection system — built as a case study on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection), with a companion Streamlit dashboard and a business-facing PDF report.

## What this is

Most fraud-detection demos stop at "here's a model with 95% accuracy." This project goes further: it deliberately restricts modeling to features that are **deployable** (computable on a live incoming transaction) and **interpretable** (explainable to a non-technical stakeholder or auditor) — excluding the dataset's large block of anonymized, undocumented engineered features that can't be reproduced outside Kaggle's competition environment.

It then compares 5 models, tunes the decision threshold against real business cost tradeoffs (missed fraud vs. false alarms), and explains every prediction with SHAP.

## Key findings

- **XGBoost (class-weighted)** was the lowest-cost model across three different cost scenarios tested — not just the highest-accuracy model on paper.
- The optimal decision threshold is **never** the default 0.5, and shifts predictably as the relative cost of false alarms changes.
- One counterintuitive result: `TransactionAmt` shows no statistically significant relationship with fraud on its own, yet is the model's single most important SHAP feature — likely an interaction effect, flagged in the report as the top thing to validate on real data.

Full analysis in [`reports/report.pdf`](reports/report.pdf) and [`reports/findings.md`](reports/findings.md).

## Project structure

```
backend/
├── data/               # raw IEEE-CIS CSVs (not tracked in git)
├── src/                # data loading, preprocessing, models, cost analysis, SHAP, stats
├── dashboard/           # Streamlit app
├── models/              # trained model artifacts (not tracked in git — run train.py)
├── train.py             # trains and saves all 5 models
└── cost_analysis.py     # cost-threshold sweep across saved models
reports/
├── generate_report.py   # builds the PDF report
├── report.pdf
└── findings.md
```

## Models compared

| Model | Imbalance handling |
|---|---|
| Random Forest | SMOTE / class-weighting |
| XGBoost | SMOTE / class-weighting |
| Logistic Regression | class-weighting |

## Running it

```bash
pip install -r requirements.txt

# 1. Download train_transaction.csv + train_identity.csv from Kaggle
#    into backend/data/raw/

# 2. Train all 5 models (saves to backend/models/)
python backend/train.py

# 3. Launch the dashboard
streamlit run backend/dashboard/app.py

# 4. Generate the PDF report
python reports/generate_report.py
```

## Dashboard

Interactive comparison across all 5 models: live threshold tuning, cost-scenario analysis, SHAP feature importance, per-transaction explanations, and a simulated live transaction feed.

## Tech stack

Python · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · SHAP · Streamlit · ReportLab

## Limitations

This uses a public dataset, not real transaction data — the findings demonstrate methodology and are not a validated production model. See `reports/report.pdf` for the full discussion of what a real deployment would require.