"""
Generates the Breaking Fraud case study report as a PDF.
Run from repo root: python backend/generate_report.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================
# PALETTE
# ============================================================
STAMP_RED = colors.HexColor("#C23B22")
DIM_GREY = colors.HexColor("#555555")
INK = colors.HexColor("#1A1A1A")

# ============================================================
# 1. GENERATE CHARTS
# ============================================================
def generate_charts():
    plt.rcParams.update({
        "figure.facecolor": "#0B1220", "axes.facecolor": "#131F33",
        "text.color": "#EDE6D6", "axes.labelcolor": "#EDE6D6",
        "xtick.color": "#8B93A6", "ytick.color": "#8B93A6",
        "axes.edgecolor": "#22314A", "grid.color": "#22314A",
        "font.size": 11,
    })

    models = ["RF\n(SMOTE)", "RF\n(class wt)", "XGBoost\n(SMOTE)", "XGBoost\n(class wt)", "LogReg"]
    prauc = [0.1577, 0.1596, 0.2139, 0.2455, 0.1446]
    bar_colors = ["#4A9C8C", "#4A9C8C", "#C23B22", "#C23B22", "#C9A227"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, prauc, color=bar_colors, width=0.6)
    ax.set_ylabel("PR-AUC Score")
    ax.set_title("Model Comparison — PR-AUC (higher is better)", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, prauc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                 ha="center", fontsize=10, fontweight="bold", color="#EDE6D6")
    plt.tight_layout()
    plt.savefig("reports/assets/chart_model_comparison.png", dpi=180, facecolor="#0B1220")
    plt.close()

    scenarios = ["Scenario 1\n(FN=Rs5k, FP=Rs50)", "Scenario 2\n(FN=Rs25k, FP=Rs125)", "Scenario 3\n(FN=Rs10k, FP=Rs500)"]
    xgb_class = [4701700, 13308500, 22896000]
    xgb_smote = [5225850, 14049250, 24879500]
    rf_class = [5418900, 14196750, 28190000]
    rf_smote = [5433450, 14245375, 28166000]
    linreg = [5487050, 14022125, 27639000]

    x = np.arange(len(scenarios))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 2*width, np.array(xgb_class)/1e6, width, label="XGBoost (class wt)", color="#C23B22")
    ax.bar(x - width, np.array(xgb_smote)/1e6, width, label="XGBoost (SMOTE)", color="#E8785A")
    ax.bar(x, np.array(rf_class)/1e6, width, label="RF (class wt)", color="#4A9C8C")
    ax.bar(x + width, np.array(rf_smote)/1e6, width, label="RF (SMOTE)", color="#7BC2B4")
    ax.bar(x + 2*width, np.array(linreg)/1e6, width, label="Logistic Regression", color="#C9A227")

    ax.set_ylabel("Cost at Optimal Threshold (Rs, Millions)")
    ax.set_title("Cost-Optimal Performance Across 3 Business Scenarios", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.legend(facecolor="#131F33", edgecolor="#22314A", labelcolor="#EDE6D6", fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("reports/assets/chart_cost_scenarios.png", dpi=180, facecolor="#0B1220")
    plt.close()


# ============================================================
# 2. STYLES
# ============================================================
def make_callout(text, styles):
    p = Paragraph(text, styles["CalloutText"])
    t = Table([[p]], colWidths=[6.4*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F3EC")),
        ("LINEBEFORE", (0, 0), (0, -1), 4, STAMP_RED),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    return t

def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=28, leading=32, fontName="Helvetica-Bold", textColor=INK, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CaseNumber", fontSize=10, fontName="Courier", textColor=STAMP_RED, spaceAfter=20))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=16, fontName="Helvetica-Bold", textColor=INK, spaceBefore=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportBody", fontSize=10.3, leading=15, fontName="Helvetica", textColor=INK, spaceAfter=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Disclaimer", fontSize=9, leading=13, fontName="Helvetica-Oblique", textColor=DIM_GREY, spaceAfter=8, borderPadding=8))
    styles.add(ParagraphStyle(name="Caption", fontSize=8.5, fontName="Helvetica-Oblique", textColor=DIM_GREY, spaceAfter=14, alignment=TA_CENTER))
    styles.add(ParagraphStyle(
        name="CalloutText", fontSize=9, leading=17, fontName="Helvetica-Oblique",
        textColor=INK,
    ))
    return styles


def table_style(header_bg=INK, header_color=colors.white):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_color),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F2ED")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#C23B22"))
    canvas.rect(0, letter[1] - 4, letter[0], 4, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(0.9*inch, 0.6*inch, "Breaking Fraud")
    canvas.drawRightString(letter[0] - 0.9*inch, 0.6*inch, f"Page {doc.page}")
    canvas.restoreState()


# ============================================================
# 3. BUILD THE REPORT
# ============================================================
def build_report():
    generate_charts()
    styles = build_styles()
    story = []

    # ---------------- COVER ----------------
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("BREAKING FRAUD", styles["ReportTitle"]))
    story.append(Paragraph(
        "A Cost-Aware, Explainable Fraud Detection System — Case Study on the IEEE-CIS Transaction Dataset",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=STAMP_RED))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<b>Prepared for:</b> Sattva Pay (fictional company)<br/>"
        "<b>Prepared by:</b> Varun Shakya<br/>"
        "<b>Date:</b> 15 July 2026",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "<b>Important note on business context:</b> \"Sattva Pay\" is a fictional company created "
        "for illustrative purposes only, so this report can demonstrate business-oriented reasoning "
        "(cost tradeoffs, deployment constraints, stakeholder recommendations) grounded in real dollar "
        "figures rather than abstract model metrics alone. All modeling, statistical testing, and cost "
        "analysis in this report were performed on real, public transaction data (the IEEE-CIS Fraud "
        "Detection dataset). No real company data was used or is represented here.",
        styles["Disclaimer"]
    ))
    story.append(PageBreak())

    # ---------------- 1. EXECUTIVE SUMMARY ----------------
    story.append(Paragraph("1. Executive Summary", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "Sattva Pay processes a high volume of digital transactions and, like any payments business, "
        "faces a constant tradeoff: flag too little and fraud losses mount; flag too much and legitimate "
        "customers are wrongly blocked, damaging trust. This report evaluates five machine learning "
        "approaches to fraud detection, selects the most cost-effective model and operating threshold, "
        "and explains the reasoning behind every flagged transaction.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(make_callout(
        "XGBoost with class-weighting was the lowest-cost model across every cost scenario "
        "tested — not just the most accurate on paper, but the most cost-effective in three different "
        "economic assumptions ranging from Rs5,000 to Rs25,000 per missed fraud case. At its optimal "
        "threshold (varying from 0.20 to 0.54 depending on cost assumptions), it consistently "
        "outperformed Random Forest and Logistic Regression baselines.",
        styles
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This model was deliberately restricted to features that are (a) computable on a live, "
        "incoming transaction and (b) explainable to a human reviewer or auditor — a design constraint "
        "that trades some raw accuracy for real-world deployability and regulatory defensibility.",
        styles["ReportBody"]
    ))

    # ---------------- 2. BUSINESS CONTEXT ----------------
    story.append(Paragraph("2. Business Context", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "Sattva Pay currently reviews flagged transactions using a static rule-based threshold, which "
        "does not adapt to the actual cost of errors on either side: missing genuine fraud (chargebacks, "
        "refunds, regulatory exposure) versus wrongly blocking legitimate customers (support costs, "
        "customer churn, reputational damage). This project evaluates whether a machine-learned, "
        "cost-calibrated approach can reduce total fraud-related losses relative to this baseline.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())

    # ---------------- 3. DATA & METHODOLOGY ----------------
    story.append(Paragraph("3. Data & Methodology", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "<b>Dataset:</b> IEEE-CIS Fraud Detection (public, Kaggle). 590,540 transactions after merging "
        "transaction and identity tables on TransactionID, with a fraud rate of 3.5% (20,663 fraud cases).",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Feature selection:</b> Of the dataset's 434 raw columns, the majority (the C1-C14, V1-V339, "
        "and most id_* columns) are Vesta's proprietary, undocumented engineered features. While often "
        "predictive, these cannot be recomputed for a new, live transaction without Vesta's internal "
        "pipeline — making any model trained on them undeployable in practice, regardless of its "
        "reported accuracy. This report restricts modeling to 14 features that are directly observable "
        "at transaction time or trivially derivable: transaction amount, product category, card network "
        "and type, billing address codes, a distance metric, purchaser/recipient email domain, device "
        "type, and hour of day (derived from the transaction timestamp).",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Class imbalance:</b> Two strategies were tested per model family — SMOTE oversampling and "
        "class-weighting — since fraud represents a small minority of transactions.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Missing data:</b> Several features (notably distance and device type) are missing for a "
        "large share of transactions. Numeric gaps were median-imputed; categorical gaps were encoded "
        "as an explicit \"Unknown\" category rather than dropped, since absence of identity data is "
        "itself a potentially meaningful signal.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())

    # ---------------- 4. DATA TRENDS ----------------
    story.append(Paragraph("4. Data Trends", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "Two statistical tests were applied to check whether each feature genuinely differs between "
        "fraudulent and legitimate transactions: the Mann-Whitney U test for the two numeric features "
        "(TransactionAmt, hour_of_day), and a chi-square test of independence for the six categorical "
        "features.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "All six categorical features showed an extremely strong statistical association with fraud "
        "(p &lt; 0.001 in every case, with five of the six essentially at p ~ 0). ProductCD, "
        "R_emaildomain, P_emaildomain, card6, and DeviceType all posted the largest chi-square "
        "statistics, indicating fraud rates vary sharply across product categories, email domains, "
        "card type, and device type — exactly the kind of signal a fraud analyst could act on "
        "operationally.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "hour_of_day was also statistically significant (p = 1.31 x 10^-7), supporting the common "
        "assumption that fraud clusters at certain times of day, though the effect is weaker than any "
        "of the categorical features.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>The most notable finding: TransactionAmt showed no statistically significant relationship "
        "with fraud (p = 0.226).</b> This runs counter to the common assumption that fraudulent "
        "transactions tend to be unusually large or small — in this dataset, transaction amount alone "
        "is not a reliable fraud indicator. This is a useful sanity check for interpreting the SHAP "
        "results in Section 7: if the model places heavy weight on TransactionAmt despite this null "
        "statistical result, that is worth flagging as a possible sign of overfitting rather than "
        "genuine signal.",
        styles["ReportBody"]
    ))

    story.append(PageBreak())

    # ---------------- 5. MODEL COMPARISON ----------------
    story.append(Paragraph("5. Model Comparison", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "Five models were trained and evaluated on a held-out 20% test set. PR-AUC (precision-recall "
        "area under curve) is the primary metric, since it is far more informative than accuracy or a "
        "single precision/recall pair when the positive class (fraud) is rare.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))

    model_table_data = [
        ["Model", "Precision", "Recall", "F1", "PR-AUC"],
        ["XGBoost (class weight)", "0.121", "0.693", "0.206", "0.246"],
        ["XGBoost (SMOTE)", "0.181", "0.487", "0.264", "0.214"],
        ["RF (class weight)", "0.090", "0.582", "0.155", "0.160"],
        ["RF (SMOTE)", "0.092", "0.603", "0.160", "0.158"],
        ["Logistic Regression", "0.090", "0.632", "0.157", "0.145"],
    ]
    t = Table(model_table_data, colWidths=[1.8*inch, 1*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    t.setStyle(table_style())
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Image("reports/assets/chart_model_comparison.png", width=6.2*inch, height=3.5*inch))
    story.append(Paragraph("Figure 1: PR-AUC comparison across all 5 trained models.", styles["Caption"]))
    story.append(Paragraph(
        "XGBoost (class weight) has the highest PR-AUC (0.246), meaningfully ahead of both Random "
        "Forest variants and Logistic Regression. XGBoost (SMOTE) has the highest F1 score, reflecting "
        "a more conservative precision/recall balance. Both XGBoost variants outperform Random Forest "
        "across every metric, consistent with gradient boosting's typical advantage on structured, "
        "imbalanced tabular data.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())

    # ---------------- 6. COST-THRESHOLD ANALYSIS ----------------
    story.append(Paragraph("6. Cost-Threshold Analysis", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "A model's raw metrics do not translate directly into a deployment decision — the decision "
        "threshold, and the relative cost of a missed fraud versus a false alarm, determine what a "
        "model actually costs a business. Three cost scenarios were tested to check whether the "
        "model recommendation holds up under different assumptions.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))

    cost_table_data = [
        ["Model", "Scenario 1\n(FN=Rs5k/FP=Rs50)", "Scenario 2\n(FN=Rs25k/FP=Rs125)", "Scenario 3\n(FN=Rs10k/FP=Rs500)"],
        ["XGBoost (class weight)", "Rs 4,701,700", "Rs 13,308,500", "Rs 22,896,000"],
        ["XGBoost (SMOTE)", "Rs 5,225,850", "Rs 14,049,250", "Rs 24,879,500"],
        ["RF (class weight)", "Rs 5,418,900", "Rs 14,196,750", "Rs 28,190,000"],
        ["RF (SMOTE)", "Rs 5,433,450", "Rs 14,245,375", "Rs 28,166,000"],
        ["Logistic Regression", "Rs 5,487,050", "Rs 14,022,125", "Rs 27,639,000"],
    ]
    t2 = Table(cost_table_data, colWidths=[1.7*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t2.setStyle(table_style())
    story.append(t2)
    story.append(Spacer(1, 15))
    story.append(Image("reports/assets/chart_cost_scenarios.png", width=6*inch, height=3.3*inch))
    story.append(Paragraph("Figure 2: Total cost at each model's optimal threshold, across three cost scenarios.", styles["Caption"]))
    story.append(Paragraph(
        "XGBoost (class weight) is the lowest-cost model in all three scenarios, saving approximately "
        "Rs785,000 (14%) over the worst-performing model in Scenario 1 alone. The optimal threshold for "
        "every model falls well below the default 0.5 cutoff in all scenarios, and rises as the relative "
        "cost of false alarms increases (Scenario 3) — both effects are economically consistent, "
        "supporting confidence in the cost model itself, not just the resulting recommendation.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Recommendation:</b> Deploy XGBoost (class weight). The exact operating threshold should be "
        "set using Sattva Pay's real cost figures once available — the dashboard accompanying this "
        "report allows that threshold to be tuned interactively.",
        styles["ReportBody"]
    ))

    story.append(PageBreak())

    # ---------------- 7. EXPLAINABILITY ----------------
    story.append(Paragraph("7. Explainability", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "Every prediction from the recommended model can be broken down using SHAP (SHapley Additive "
        "exPlanations), which attributes the model's output to individual feature contributions for a "
        "specific transaction. This matters for two practical reasons: it lets a fraud analyst justify "
        "why a transaction was flagged rather than trusting a black-box score, and it satisfies the "
        "kind of explainability that compliance and audit functions typically require before a model "
        "can be used in a regulated financial context.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Strong agreement with the statistical findings:</b> card6 (card type), ProductCD (product "
        "category), R_emaildomain, and DeviceType all rank highly by both SHAP importance and "
        "statistical significance (Section 4). This convergence across two independent methods — one "
        "measuring the model's actual reliance on a feature, the other measuring the feature's raw "
        "statistical relationship with fraud — is a strong signal these features carry genuine, "
        "trustworthy fraud signal rather than noise the model happened to latch onto.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>A discrepancy worth flagging directly:</b> TransactionAmt is the single most important "
        "feature by SHAP value (0.326, well above every other feature), despite showing no "
        "statistically significant relationship with fraud on its own in Section 4 (p = 0.226). This "
        "is not necessarily a flaw — it likely means transaction amount matters in combination with "
        "other features rather than on its own. For example, an unusual amount for a specific product "
        "category or card type may be predictive even though amount alone, averaged across all "
        "transaction types, is not. SHAP can capture this kind of interaction effect; a simple "
        "two-group statistical test cannot. That said, this pattern is also consistent with a model "
        "that has learned a spurious or overly specific relationship in the training data, so this is "
        "the single most important thing to validate if this model is ever retrained on Sattva Pay's "
        "real transaction data.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "card2, card3, and card1 also rank among the top features despite being anonymized identifiers "
        "with no human-readable meaning — a reminder that the model benefits from these "
        "deployable-but-opaque signals even though they cannot be explained to a business stakeholder "
        "individually (Section 3). This is the practical cost of the interpretability-vs-accuracy "
        "tradeoff made explicit: some of the model's real predictive power comes from features that "
        "improve deployability and accuracy but not explainability.",
        styles["ReportBody"]
    ))
    story.append(PageBreak())

    # ---------------- 8. LIMITATIONS ----------------
    story.append(Paragraph("8. Limitations & Next Steps", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=STAMP_RED, spaceAfter=10))
    story.append(Paragraph(
        "<b>Public dataset, not Sattva Pay's data.</b> These results demonstrate the approach and "
        "methodology; actual performance on Sattva Pay's real transaction data would need to be "
        "validated before deployment.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Deployability tradeoff.</b> Restricting to explainable, computable features (Section 3) "
        "likely costs meaningful accuracy compared to using the dataset's full 434-column feature set, "
        "including Vesta's opaque engineered features. This is a deliberate tradeoff favoring "
        "real-world usability over maximum benchmark accuracy.",
        styles["ReportBody"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Recommended next step:</b> A limited pilot retraining this same pipeline on a sample of "
        "Sattva Pay's actual historical transactions, to validate whether the cost-threshold "
        "recommendation and feature importance findings hold on real operational data — with "
        "particular attention to whether TransactionAmt remains the top predictive feature, as "
        "flagged in Section 7.",
        styles["ReportBody"]
    ))

    doc = SimpleDocTemplate(
        "reports/fraud_report.pdf", pagesize=letter,
        topMargin=0.9*inch, bottomMargin=0.9*inch,
        leftMargin=0.9*inch, rightMargin=0.9*inch,
    )
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print("Report generated: reports/fraud_report.pdf")


if __name__ == "__main__":
    build_report()