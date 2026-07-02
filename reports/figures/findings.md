# Findings — Credit Card Fraud Detection

## Problem

Detect fraudulent credit card transactions in a highly imbalanced dataset (284,807 transactions, 492 fraud — 0.17%). The core challenge throughout this project was that naive approaches (accuracy-optimized models, default thresholds) fail badly on this kind of imbalance, so every stage required imbalance-aware techniques.

---

## 1. Statistical Validation (Mann-Whitney U Test)

Before modeling, each feature was tested for whether its distribution genuinely differs between fraud and legitimate transactions, using the Mann-Whitney U test (chosen over a t-test since transaction features are non-normally distributed).

**Top discriminating features (smallest p-values):**

| Rank | Feature | p-value |
|---|---|---|
| 1 | V14 | 1.47e-260 |
| 2 | V4 | 3.63e-248 |
| 3 | V12 | 8.42e-247 |
| 4 | V11 | 4.91e-226 |
| 5 | V10 | 9.61e-222 |

All top features returned p-values effectively at zero, confirming strong, statistically significant separation between classes — evidence that a predictive model has real signal to learn from, not noise.

---

## 2. Model Comparison — SMOTE vs Class Weighting

Two imbalance-handling strategies were tested across two model types (Logistic Regression, Random Forest), evaluated on a held-out, untouched-by-resampling test set.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression + SMOTE | 0.058 | 0.918 | 0.109 | **0.725** |
| Logistic Regression + class weight | 0.061 | 0.918 | 0.114 | 0.716 |
| Random Forest + SMOTE | 0.214 | 0.888 | 0.345 | 0.708 |
| **Random Forest + class weight** | **0.336** | 0.878 | **0.486** | 0.654 |

**Key takeaways:**
- Logistic Regression achieves the highest recall and PR-AUC, but at unusable precision (~6%) — over 1,300 false alarms per ~90 fraud cases caught. Not production-viable as-is.
- Random Forest + class weighting gives the best precision/recall balance (F1 = 0.486) and the fewest false positives (170), making it the strongest default candidate.
- **Class weighting outperformed SMOTE on precision for both model types at their default thresholds** — a real, dataset-specific finding, not a universal rule.
- PR-AUC and F1 don't always agree on model quality: PR-AUC reflects performance *across all thresholds*, while F1 is a snapshot at one threshold. Logistic Regression's high PR-AUC suggests untapped potential that a better threshold could unlock.

---

## 3. Cost-Based Threshold Optimization

Rather than accepting the default 0.5 decision threshold, thresholds were swept and evaluated against assumed real-world costs: a missed fraud (false negative) vs. a false alarm (false positive).

| FN:FP cost ratio | Optimal threshold | Cost at 0.5 | Cost at optimal | Savings |
|---|---|---|---|---|
| 100:1 (₹5,000 / ₹50) | 0.47 | ₹68,500 | ₹65,650 | ~4% |
| 400:1 (₹10,000 / ₹25) | 0.30 | ₹124,250 | ₹99,875 | ~20% |
| ~1000:1+ | 0.22 | ₹304,250 | ₹217,775 | ~28% |

**Key takeaway:** the "correct" decision threshold is not a fixed model property — it depends entirely on the business's cost assumptions. As the relative cost of missing fraud increases, the optimal threshold moves lower (the model should flag more aggressively), and the savings from tuning the threshold instead of using a default grow substantially. This is implemented as a live, adjustable control in the dashboard.

---

## 4. Explainability (SHAP)

SHAP (TreeExplainer) was used to explain both global model behavior and individual predictions from the Random Forest model.

**Global importance** closely mirrored the statistical test results: `V14`, `V4`, `V10`, `V12`, `V11` ranked as the top contributors in both methods — independent confirmation from a classical statistical test and a modern explainability technique.

**Individual case example:** for one caught fraud transaction, the model moved from a 50.2% baseline to a 99.8% fraud prediction, driven primarily by `V14` (+0.13), `V10` (+0.09), and `V17`/`V4`/`V12` (+0.06 each) — a fully traceable, auditable explanation for the decision.

---

## 5. Overall Conclusion

- **Recommended model:** Random Forest + class weighting, threshold tuned to the business's actual cost assumptions rather than left at 0.5.
- **Recommended process:** don't optimize for accuracy or even F1 alone on imbalanced problems — use PR-AUC to compare models, and a cost matrix to pick the final operating threshold.
- **Explainability isn't optional** for a fraud system deployed in practice — SHAP provides the "why" behind each flag, which both statistical testing and global importance independently corroborate.
- All of the above is explorable interactively in the accompanying Streamlit dashboard (model switch, threshold slider, live cost recalculation, SHAP breakdowns).