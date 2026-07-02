import plotly.graph_objects as go
import numpy as np

# --- Shared palette, keeps every chart visually consistent ---
ACCENT_RED = "#D85A30"
ACCENT_BLUE = "#4A9EDE"
NEUTRAL = "#8A8A8A"
TABLE_HEADER = "#2B2B2B"

BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=13),
    margin=dict(l=10, r=10, t=45, b=10),
)


def plot_confusion_matrix(cm):
    labels = ["Not Fraud", "Fraud"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[[0, "#1a2f4a"], [1, ACCENT_BLUE]],
        showscale=False,
    ))

    fig.update_layout(
        **BASE_LAYOUT,
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
        line=dict(color=ACCENT_RED, width=2.5),
    ))

    fig.add_vline(
        x=best_threshold,
        line_dash="dash",
        line_color=NEUTRAL,
        annotation_text=f"Optimal: {best_threshold:.2f}",
        annotation_font_color=NEUTRAL,
    )

    fig.update_layout(
        **BASE_LAYOUT,
        title="Expected Cost vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Total Expected Cost (₹)",
        height=350,
    )

    return fig


def plot_shap_importance(shap_values, feature_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [feature_names[i] for i in order][::-1]
    top_values = [mean_abs[i] for i in order][::-1]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation="h",
        marker_color=ACCENT_BLUE,
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title="Mean |SHAP value| — global feature importance",
        xaxis_title="Mean |SHAP value|",
        height=450,
    )

    return fig


def plot_single_explanation(feature_names, values, base_value, prediction, top_n=10):
    order = np.argsort(np.abs(values))[::-1][:top_n]
    top_features = [feature_names[i] for i in order][::-1]
    top_values = [values[i] for i in order][::-1]
    colors = [ACCENT_RED if v > 0 else ACCENT_BLUE for v in top_values]

    fig = go.Figure(go.Bar(
        x=top_values,
        y=top_features,
        orientation="h",
        marker_color=colors,
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        title=f"Base rate {base_value:.3f} → Prediction {prediction:.3f}",
        xaxis_title="SHAP value (push toward fraud →, away ←)",
        height=400,
    )

    return fig


def plot_feature_test_table(results_df, top_n=10):
    top = results_df.head(top_n)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Feature", "Statistic", "p-value"],
            align="left",
            fill_color=TABLE_HEADER,
            font=dict(size=13, color="white"),
        ),
        cells=dict(
            values=[top["feature"], top["statistic"].round(1),
                    top["p_value"].apply(lambda x: f"{x:.2e}")],
            align="left",
            height=28,
            fill_color="rgba(0,0,0,0)",
            font=dict(color="white"),
        ),
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig