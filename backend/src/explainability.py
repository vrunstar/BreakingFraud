import shap

def get_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def plot_global_importance(shap_values, X_sample):
    shap.summary_plot(shap_values[:, :, 1], X_sample)


def explain_single_prediction(explainer, shap_values, X_sample, index=0):
    single_values = shap_values[index, :, 1]
    single_data = X_sample.iloc[index]
    base_value = explainer.expected_value[1]

    shap.plots.waterfall(shap.Explanation(
        values=single_values,
        base_values=base_value,
        data=single_data,
        feature_names=X_sample.columns.tolist()
    ))