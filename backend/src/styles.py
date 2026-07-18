CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root {
    --ink-navy:      #0B1220;
    --panel-navy:    #131F33;
    --panel-border:  #22314A;
    --stamp-red:     #C23B22;
    --verdigris:     #4A9C8C;
    --brass:         #C9A227;
    --paper:         #EDE6D6;
    --paper-dim:     #8B93A6;
}

/* ---------- Base app shell ---------- */
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(194,59,34,0.06), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(74,156,140,0.06), transparent 40%),
        var(--ink-navy) !important;
    color: var(--paper) !important;
    font-family: 'Inter', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: #0E1728 !important;
    border-right: 1px solid var(--panel-border) !important;
}

/* ---------- Headings ---------- */
h1, h2, h3 {
    font-family: 'Oswald', sans-serif !important;
    letter-spacing: 0.5px !important;
    color: var(--paper) !important;
}

/* ---------- Tabs: styled like case-file dividers ---------- */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    color: var(--paper-dim) !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--paper) !important;
    border-bottom: 2px solid var(--stamp-red) !important;
}
div[data-baseweb="tab-highlight"] { background-color: transparent !important; }
div[data-baseweb="tab-border"] { background-color: var(--panel-border) !important; }

/* ---------- Metric cards: ledger-entry look ---------- */
div[data-testid="stMetric"] {
    background: var(--panel-navy) !important;
    border: 1px solid var(--panel-border) !important;
    border-left: 3px solid var(--brass) !important;
    border-radius: 4px !important;
    padding: 14px 16px !important;
    transition: transform 0.15s ease, border-left-color 0.15s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    border-left-color: var(--stamp-red) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: var(--paper) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: var(--paper-dim) !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--panel-navy) !important;
    border: 1px solid var(--stamp-red) !important;
    color: var(--paper) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    border-radius: 3px !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--stamp-red) !important;
    border-color: var(--stamp-red) !important;
    color: var(--paper) !important;
}

/* ---------- Sliders / number inputs / labels ---------- */
div[data-testid="stSlider"] span,
label {
    font-family: 'Inter', sans-serif !important;
    color: var(--paper-dim) !important;
}

/* ---------- Dataframes: ledger table ---------- */
div[data-testid="stDataFrame"] {
    font-family: 'IBM Plex Mono', monospace !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 4px !important;
}

/* ---------- Signature element: ink-stamp verdict badge ----------
   Element-type selectors (span.stamp, not just .stamp) + !important,
   needed to beat Streamlit's own stMarkdownContainer-scoped defaults. */
span.stamp {
    display: inline-block !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    padding: 4px 14px !important;
    border-radius: 2px !important;
    transform: rotate(-3deg) !important;
    text-transform: uppercase !important;
}
span.stamp-flagged {
    color: var(--stamp-red) !important;
    border: 2.5px solid var(--stamp-red) !important;
    background: rgba(194,59,34,0.08) !important;
}
span.stamp-cleared {
    color: var(--verdigris) !important;
    border: 2.5px solid var(--verdigris) !important;
    background: rgba(74,156,140,0.08) !important;
}

/* ---------- Model badges (used in sidebar) ---------- */
span.model-badge {
    display: inline-block !important;
    padding: 3px 12px !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    margin-right: 6px !important;
}
span.badge-rf  { background: rgba(74,156,140,0.12) !important; color: var(--verdigris) !important; border: 1px solid var(--verdigris) !important; }
span.badge-xgb { background: rgba(194,59,34,0.12) !important;  color: var(--stamp-red) !important; border: 1px solid var(--stamp-red) !important; }
span.badge-lr  { background: rgba(201,162,39,0.12) !important; color: var(--brass) !important;     border: 1px solid var(--brass) !important; }

/* ---------- Hero header ---------- */
div.hero {
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
    padding: 32px 32px !important;
}
div.hero h1 {
    font-size: 36px !important;
    color: var(--paper) !important;
}
div.case-number {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: var(--brass) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}
div.hero p {
    color: var(--paper-dim) !important;
    margin: 0 !important;
}
</style>
"""