CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Anton&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Title uses Anton for a bold display look */
h1 {
    font-family: 'Anton', sans-serif;
    font-weight: 400;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #2A2F3A;
    border-radius: 10px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] {
    color: #9CA3AF;
    font-size: 13px;
}
div[data-testid="stMetricValue"] {
    color: #F3F4F6;
    font-family: 'Anton', sans-serif;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #D85A30 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #D85A30 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid #2A2F3A;
}

/* Buttons */
button[kind="secondary"], button[kind="primary"] {
    border-radius: 8px;
}

/* Number inputs / sliders spacing tightened */
div[data-testid="stSidebar"] .stSlider, 
div[data-testid="stSidebar"] .stNumberInput {
    margin-bottom: 4px;
}
</style>
"""