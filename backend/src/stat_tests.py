from scipy.stats import mannwhitneyu, chi2_contingency
import pandas as pd

NUMERIC_COLS = ["TransactionAmt", "hour_of_day"]
CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]


def compare_numeric_distributions(df, column, target_col="isFraud"):
    group_fraud = df[df[target_col] == 1][column].dropna()
    group_legit = df[df[target_col] == 0][column].dropna()

    result = mannwhitneyu(group_legit, group_fraud)
    return result.statistic, result.pvalue


def compare_categorical_association(df, column, target_col="isFraud"):
    contingency = pd.crosstab(df[column].fillna("Unknown"), df[target_col])
    chi2, pvalue, dof, expected = chi2_contingency(contingency)
    return chi2, pvalue


def run_all_feature_tests(df, target_col="isFraud"):
    results = []

    for col in NUMERIC_COLS:
        stat, pval = compare_numeric_distributions(df, col, target_col)
        results.append({"feature": col, "test": "Mann-Whitney U", "statistic": stat, "p_value": pval})

    for col in CATEGORICAL_COLS:
        stat, pval = compare_categorical_association(df, col, target_col)
        results.append({"feature": col, "test": "Chi-square", "statistic": stat, "p_value": pval})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="p_value", ascending=True)
    return results_df

