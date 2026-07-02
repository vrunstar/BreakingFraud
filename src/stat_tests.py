from scipy.stats import mannwhitneyu
import pandas as pd

def compare_distributions(df, column, target_col="Class"):
    group_fraud = df[df[target_col] == 1][column] 
    group_legit = df[df[target_col] == 0][column]

    result = mannwhitneyu(group_legit, group_fraud)

    return result.statistic, result.pvalue

def run_all_feature_tests(df, target_col="Class"):
    columns_to_test = [f"V{i}" for i in range (1, 29)] + ["Amount"]
    results = []

    for col in columns_to_test:
        stat, pval = compare_distributions(df, col, target_col)
        results.append({"feature": col, "statistic": stat, "p_value": pval})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="p_value", ascending=True)
    return results_df