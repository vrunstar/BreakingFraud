import plotly.graph_objects as go
import numpy as np


def plot_confusion_matrix(cm):
    labels = ["Not Fraud", "Fraud"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale="Blues",
        showscale=False,
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        height=350,
    )

    return fig


def plot_cost_curve(thresholds, costs, best_threshold):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=costs,
        mode="lines",
        name="Total cost",
        line=dict(color="#D85A30", width=2),
    ))

    fig.add_vline(
        x=best_threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Optimal: {best_threshold:.2f}",
    )

    fig.update_layout(
        title="Expected Cost vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Total Expected Cost (₹)",
        height=350,
    )

    return fig


def plot_shap_importance(shap_values, feature_names, top_n=15):
    # shap_values here should already be the fraud-class slice, shape (n_samples, n_features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [feature_names[i] for i in order][::-1]
    top_values = [mean_abs[i] for i in order][::-1]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation="h",
        marker_color="#378ADD",
    ))

    fig.update_layout(
        title="Mean |SHAP value| — global feature importance",
        xaxis_title="Mean |SHAP value|",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_single_explanation(feature_names, values, base_value, prediction, top_n=10):
    order = np.argsort(np.abs(values))[::-1][:top_n]
    top_features = [feature_names[i] for i in order][::-1]
    top_values = [values[i] for i in order][::-1]
    colors = ["#D85A30" if v > 0 else "#378ADD" for v in top_values]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation="h",
        marker_color=colors,
    ))

    fig.update_layout(
        title=f"Base rate {base_value:.3f} → Prediction {prediction:.3f}",
        xaxis_title="SHAP value (push toward fraud →, away ←)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_feature_test_table(results_df, top_n=10):
    top = results_df.head(top_n)

    fig = go.Figure(data=[go.Table(
        header=dict(values=["Feature", "Statistic", "p-value"], align="left",
                    fill_color="#F1EFE8", font=dict(size=13)),
        cells=dict(values=[top["feature"], top["statistic"].round(1),
                            top["p_value"].apply(lambda x: f"{x:.2e}")],
                   align="left", height=28),
    )])

    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    return fig