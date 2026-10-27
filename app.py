st.markdown("""
<style>
/* ================================
   GPR CRYPTO - HIGH CONTRAST UI
   ================================ */

/* Metric cards */
[data-testid="stMetric"] {
    background: #162235 !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    min-height: 120px !important;
}

/* Metric label */
[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #22C55E !important;
}

/* General text */
.stMarkdown,
.stText,
p,
label {
    color: #E2E8F0;
}

/* Headers */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #FF4B4B !important;
}

/* Dataframe / tables */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

/* Slider text */
[data-testid="stSlider"] label {
    color: #CBD5E1 !important;
}

/* Info / success / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
}

/* Main scan button */
.stButton > button[kind="primary"] {
    color: #FFFFFF !important;
}

/* Make small secondary text readable */
.small-text,
.caption,
[data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)
