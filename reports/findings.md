# Findings — Credit Card Fraud Detection (IEEE-CIS)

## Problem

Detect fraudulent transactions in the IEEE-CIS Fraud Detection dataset (590,540 transactions, 20,663 fraud — 3.5%), merged from `train_transaction.csv` and `train_identity.csv` on `TransactionID`. Unlike a typical anonymized fraud benchmark, this dataset was deliberately restricted to a subset of features that are both **deployable** (computable on a live, incoming transaction) and **interpretable** (explainable to a business stakeholder) — excluding Vesta's proprietary, undocumented engineered features (`C1`–`C14`, `V1`–`V339`, most `id_*` columns), which cannot be recomputed outside Vesta's internal pipeline regardless of how predictive they are.

**Features used for modeling (14):** `TransactionAmt`, `card1`–`card6`, `addr1`, `addr2`, `dist1`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `ProductCD`, and a derived `hour_of_day` feature.

---

## 1. Statistical Validation (Mann-Whitney U + Chi-square)

Numeric features (`TransactionAmt`, `hour_of_day`) were tested with the Mann-Whitney U test; categorical features (`ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`) were tested with a chi-square test of independence, since Mann-Whitney doesn't apply to categorical data.

| Feature | Test | p-value |
|---|---|---|
| ProductCD | Chi-square | ~0 |
| R_emaildomain | Chi-square | ~0 |
| P_emaildomain | Chi-square | ~0 |
| card6 | Chi-square | ~0 |
| DeviceType | Chi-square | ~0 |
| card4 | Chi-square | 1.45e-78 |
| hour_of_day | Mann-Whitney U | 1.31e-07 |
| TransactionAmt | Mann-Whitney U | 0.226 (not significant) |

**Key takeaway:** every categorical feature shows an extremely strong statistical association with fraud. `hour_of_day` is significant but weaker. **`TransactionAmt` alone shows no significant relationship with fraud** — a genuinely counterintuitive result, since transaction amount is commonly assumed to be a strong fraud signal. This became an important cross-check against the SHAP results (see Section 4).

---

## 2. Model Comparison — 5 Models, 2 Imbalance Strategies

Random Forest, XGBoost, and Logistic Regression were trained, with Random Forest and XGBoost each tested under both SMOTE oversampling and class-weighting.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **XGBoost + class weight** | 0.121 | **0.693** | 0.206 | **0.246** |
| XGBoost + SMOTE | **0.181** | 0.487 | **0.264** | 0.214 |
| Random Forest + class weight | 0.090 | 0.582 | 0.155 | 0.160 |
| Random Forest + SMOTE | 0.092 | 0.603 | 0.160 | 0.158 |
| Logistic Regression + class weight | 0.090 | 0.632 | 0.157 | 0.145 |

**Key takeaways:**
- **XGBoost beats Random Forest across every metric** — consistent with gradient boosting's typical edge on structured, imbalanced tabular data.
- XGBoost + class weight has the best PR-AUC; XGBoost + SMOTE has the best F1 — a genuine tradeoff between catching more fraud (higher recall) vs. being more confident when it does flag something (higher precision).
- These PR-AUC scores (~0.15–0.25) are meaningfully lower than typical IEEE-CIS leaderboard results (~0.7+). This is expected and intentional: it's the measurable cost of restricting the model to deployable, explainable features instead of Vesta's full opaque feature set (see Problem section).

---

## 3. Cost-Based Threshold Optimization

Thresholds were swept and evaluated against three different assumed cost scenarios (false negative cost = missed fraud, false positive cost = false alarm), to test whether the model recommendation is robust or scenario-dependent.

| Scenario | FN:FP cost | XGBoost (class wt) optimal threshold | Cost at optimal |
|---|---|---|---|
| 1 | ₹5,000 : ₹50 | 0.26 | ₹4,701,700 |
| 2 | ₹25,000 : ₹125 | 0.20 | ₹13,308,500 |
| 3 | ₹10,000 : ₹500 | 0.54 | ₹22,896,000 |

**XGBoost + class weight was the lowest-cost model in all three scenarios** — beating the next-best model by ₹524K–₹1.98M depending on scenario. In Scenario 1, it saved ~14% (₹785,350) over the worst-performing model (Logistic Regression).

**Key takeaway:** the optimal threshold shifts meaningfully across scenarios (0.20–0.54) — always well below the naive default of 0.5, and rising as the relative cost of false alarms increases (Scenario 3). Both effects are economically consistent, which supports confidence in the cost model itself, not just the resulting model recommendation. This is implemented as a live, adjustable control in the dashboard.

---

## 4. Explainability (SHAP)

SHAP (TreeExplainer) was used on the recommended model (XGBoost + class weight) to explain both global feature importance and individual predictions.

**Top features by mean absolute SHAP value:**

| Rank | Feature | SHAP importance |
|---|---|---|
| 1 | TransactionAmt | 0.326 |
| 2 | card6 (credit) | 0.291 |
| 3 | ProductCD (C) | 0.257 |
| 4 | R_emaildomain (Unknown) | 0.215 |
| 5 | card3 | 0.199 |
| 6 | R_emaildomain (gmail.com) | 0.194 |

**Strong agreement with statistical testing:** `card6`, `ProductCD`, `R_emaildomain`, and `DeviceType` rank highly by both SHAP importance and statistical significance — independent confirmation across two different methods that these features carry genuine fraud signal.

**A discrepancy worth flagging:** `TransactionAmt` is the single most important SHAP feature, despite showing *no* statistically significant relationship with fraud on its own (Section 1, p = 0.226). This most likely reflects an interaction effect — amount matters in combination with other features (e.g. unusual for a given product category) rather than on its own, which SHAP can capture but a simple two-group statistical test cannot. It's also a plausible sign of overfitting, and is flagged as the top validation priority if this model is ever retrained on real transaction data.

Several anonymized identifier columns (`card1`, `card2`, `card3`) also rank highly despite carrying no human-readable meaning — a direct illustration of the interpretability-vs-accuracy tradeoff: the model benefits from these deployable-but-opaque features even though individual predictions involving them can't be fully explained in plain language.

---

## 5. Overall Conclusion

- **Recommended model:** XGBoost with class weighting — the only model that was cost-optimal across all three tested business scenarios, not just the highest-PR-AUC model in isolation.
- **Recommended threshold:** business-dependent — use the dashboard's live threshold slider with real cost assumptions rather than a fixed default.
- **Recommended process:** evaluate with PR-AUC (not accuracy) given the class imbalance, test the model recommendation against multiple cost scenarios rather than a single assumption, and validate SHAP-reported importance against independent statistical tests before trusting a feature's apparent importance.
- **Explainability caveat:** the model's top feature (`TransactionAmt`) lacks independent statistical support — a genuine limitation to disclose, not a result to present without qualification. This is the single most important thing to re-validate if this pipeline is ever retrained on real transaction data.
- All of the above is explorable interactively in the accompanying Streamlit dashboard (model switch, threshold slider, live cost recalculation, SHAP breakdowns).