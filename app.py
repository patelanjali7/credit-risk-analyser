"""
CreditRisk Analyser — Streamlit Application
Professional banking-grade AI-powered credit risk assessment.
"""
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prediction_helper import (
    GRADE_RATE_MAP,
    THRESHOLD,
    calculate_emi,
    derive_loan_grade,
    generate_decision_report,
    predict_batch_fast,
    predict_risk,
)

# Plotly chart config — applied to every chart to remove logo and theme the modebar
_CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "filename": "creditrisk_chart", "scale": 2},
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditRisk Analyser",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar — settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Display Settings")
    st.markdown("**Theme**")
    display_mode = st.radio(
        "Theme",
        options=["☀️ Light", "🌙 Dark", "⚫ Contrast"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    mode_key = "light" if "Light" in display_mode else ("dark" if "Dark" in display_mode else "contrast")
    st.markdown("---")
    st.markdown("**Approval Tiers**")
    st.markdown(
        "🟢 **Approved** — PD < 40%  \n"
        "🟡 **Review** — PD 40–55%  \n"
        "🟠 **Conditional** — PD 55–70%  \n"
        "🔴 **Rejected** — PD ≥ 70%"
    )
    st.markdown("---")

# ── CSS ───────────────────────────────────────────────────────────────────────
LIGHT_CSS = """
<style>
  .stApp { background: #F0F4F8; }
  .block-container { padding-top: 0 !important; max-width: 1280px; }
  #MainMenu, footer, header { visibility: hidden; }

  .topbar {
    background: linear-gradient(90deg, #0F2748 0%, #1A3F7A 100%);
    color: white; padding: 14px 32px; margin: -60px -50px 0 -50px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .topbar-title  { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.5px; }
  .topbar-meta   { font-size: 0.72rem; opacity: 0.65; text-align: right; line-height: 1.6; }

  .stTabs [data-baseweb="tab-list"] {
    background: white; border-bottom: 2px solid #D1D9E6;
    padding: 0 4px; gap: 2px; margin-top: 12px;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 22px; font-size: 0.85rem; font-weight: 600;
    color: #5A6A85; border-radius: 4px 4px 0 0;
  }
  .stTabs [aria-selected="true"] { color: #0F2748; border-bottom: 3px solid #0F2748 !important; }

  .sec-label {
    background: #0F2748; color: white; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.8px; padding: 6px 14px; border-radius: 4px 4px 0 0;
    margin-bottom: 0; display: inline-block; width: 100%;
  }

  .banner { padding: 18px 24px; border-radius: 8px; margin: 8px 0 16px; }
  .banner-approved   { background:#F0FDF4; border-left: 6px solid #16A34A; }
  .banner-review     { background:#FFFBEB; border-left: 6px solid #D97706; }
  .banner-conditional{ background:#FFF7ED; border-left: 6px solid #EA580C; }
  .banner-rejected   { background:#FEF2F2; border-left: 6px solid #DC2626; }

  div[data-testid="metric-container"] {
    background: white; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }

  label { font-size: 0.8rem !important; font-weight: 600 !important;
          color: #374151 !important; text-transform: uppercase;
          letter-spacing: 0.3px !important; }

  /* All buttons — base sizing */
  .stButton > button,
  [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-secondary"] {
    border-radius: 6px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; min-height: 44px !important;
    transition: all 0.2s ease !important; border: none !important;
  }
  /* Primary buttons (Submit Application) — navy gradient */
  button[kind="primary"],
  [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0F2748 0%, #1A3F7A 100%) !important;
    color: white !important;
  }
  button[kind="primary"]:hover,
  [data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #1A3F7A 0%, #2A5FAA 100%) !important;
    box-shadow: 0 4px 14px rgba(15,39,72,0.40) !important;
  }
  /* Secondary buttons (New Application) — amber gradient */
  button[kind="secondary"],
  [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #B45309 0%, #D97706 100%) !important;
    color: white !important;
  }
  button[kind="secondary"]:hover,
  [data-testid="stBaseButton-secondary"]:hover {
    background: linear-gradient(135deg, #92400E 0%, #B45309 100%) !important;
    box-shadow: 0 4px 14px rgba(180,83,9,0.40) !important;
  }
  /* Download buttons override — green gradient */
  [data-testid="stDownloadButton"] button,
  [data-testid="stDownloadButton"] [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #15803D 0%, #16A34A 100%) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 0.9rem !important; min-height: 44px !important;
  }
  [data-testid="stDownloadButton"] button:hover,
  [data-testid="stDownloadButton"] [data-testid="stBaseButton-secondary"]:hover {
    background: linear-gradient(135deg, #166534 0%, #15803D 100%) !important;
    box-shadow: 0 4px 14px rgba(21,128,61,0.40) !important;
  }

  /* Dropdown popup (light) */
  [data-baseweb="popover"] ul, [data-baseweb="menu"] { background: #FFFFFF !important; }
  [data-baseweb="popover"] li { background: #FFFFFF !important; color: #1E293B !important; }
  [data-baseweb="popover"] li:hover { background: #F0F4F8 !important; }

  /* Plotly chart modebar — light theme */
  .modebar { background: rgba(255,255,255,0.9) !important; border-radius: 4px; }
  .modebar-btn path { fill: #5A6A85 !important; }
  .modebar-btn:hover { background: #E8EEF6 !important; }
  .modebar-btn.active path { fill: #0F2748 !important; }
</style>
"""

DARK_CSS = """
<style>
  .stApp, [data-testid="stAppViewContainer"] { background: #0F172A !important; }
  .block-container { padding-top: 0 !important; max-width: 1280px; background: #0F172A !important; }
  #MainMenu, footer, header { visibility: hidden; }

  section[data-testid="stSidebar"] { background: #1E293B !important; }
  section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

  .topbar {
    background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
    color: white; padding: 14px 32px; margin: -60px -50px 0 -50px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .topbar-title  { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.5px; }
  .topbar-meta   { font-size: 0.72rem; opacity: 0.65; text-align: right; line-height: 1.6; }

  .stTabs [data-baseweb="tab-list"] {
    background: #1E293B !important; border-bottom: 2px solid #334155 !important;
    padding: 0 4px; gap: 2px; margin-top: 12px;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 22px; font-size: 0.85rem; font-weight: 600;
    color: #94A3B8 !important; border-radius: 4px 4px 0 0;
  }
  .stTabs [aria-selected="true"] { color: #60A5FA !important; border-bottom: 3px solid #60A5FA !important; }

  .sec-label {
    background: #1E3A5F !important; color: white; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.8px; padding: 6px 14px; border-radius: 4px 4px 0 0;
    margin-bottom: 0; display: inline-block; width: 100%;
  }

  .banner { padding: 18px 24px; border-radius: 8px; margin: 8px 0 16px; }
  .banner-approved   { background:#14532D !important; border-left: 6px solid #16A34A; }
  .banner-review     { background:#78350F !important; border-left: 6px solid #D97706; }
  .banner-conditional{ background:#7C2D12 !important; border-left: 6px solid #EA580C; }
  .banner-rejected   { background:#7F1D1D !important; border-left: 6px solid #DC2626; }

  div[data-testid="metric-container"] {
    background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.3);
  }
  div[data-testid="metric-container"] * { color: #E2E8F0 !important; }

  label { font-size: 0.8rem !important; font-weight: 600 !important;
          color: #CBD5E1 !important; text-transform: uppercase; letter-spacing: 0.3px !important; }

  p, div, span, h1, h2, h3, h4 { color: #E2E8F0; }
  .stMarkdown p { color: #E2E8F0 !important; }
  .stCaption p  { color: #94A3B8 !important; }

  /* Containers with border */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #1E293B !important; border-color: #334155 !important; border-radius: 8px !important;
  }

  /* Number inputs */
  [data-testid="stNumberInput"] input {
    background: #334155 !important; color: #E2E8F0 !important;
    border-color: #475569 !important;
  }
  [data-testid="stNumberInput"] { background: transparent !important; }

  /* Select boxes */
  [data-testid="stSelectbox"] [data-baseweb="select"] { background: #334155 !important; }
  [data-testid="stSelectbox"] [data-baseweb="select"] div { background: #334155 !important; color: #E2E8F0 !important; }
  [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: #E2E8F0 !important; }

  /* Radio buttons */
  [data-testid="stRadio"] div { color: #E2E8F0 !important; }
  [data-testid="stRadio"] label { color: #CBD5E1 !important; }

  /* Expanders */
  [data-testid="stExpander"] { border-color: #334155 !important; background: #1E293B !important; }
  [data-testid="stExpander"] summary { color: #E2E8F0 !important; background: #1E293B !important; }

  /* Info / success / warning / error boxes */
  [data-testid="stAlertContainer"] { background: #1E3A5F !important; color: #E2E8F0 !important; }
  [data-testid="stAlertContainer"] p { color: #E2E8F0 !important; }

  /* DataFrames */
  [data-testid="stDataFrame"] { background: #1E293B !important; }
  [data-testid="stDataFrame"] * { color: #E2E8F0 !important; }

  /* Dividers */
  [data-testid="stDivider"] { border-color: #334155 !important; }

  /* General baseweb inputs */
  [data-baseweb="input"] input, [data-baseweb="select"] div,
  [data-baseweb="textarea"] textarea { background: #334155 !important; color: #E2E8F0 !important; }

  /* Dropdown menu list */
  [data-baseweb="popover"] li { background: #334155 !important; color: #E2E8F0 !important; }
  [data-baseweb="popover"] li:hover { background: #475569 !important; }
  [data-baseweb="menu"] { background: #334155 !important; }

  /* Buttons — dark theme keeps the same color logic so they stay visible */
  .stButton > button { border-radius: 6px; font-weight: 700; font-size: 0.9rem; min-height: 44px; }
  /* Buttons — dark theme */
  .stButton > button,
  [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-secondary"] {
    border-radius: 6px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; min-height: 44px !important; border: none !important;
  }
  button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #1E3A5F 0%, #2A5FAA 100%) !important;
    color: white !important;
  }
  button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #B45309 0%, #D97706 100%) !important;
    color: white !important;
  }
  [data-testid="stDownloadButton"] button,
  [data-testid="stDownloadButton"] [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #15803D 0%, #16A34A 100%) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 0.9rem !important; min-height: 44px !important;
  }

  /* Plotly chart modebar — dark theme */
  .modebar { background: #1E293B !important; border-radius: 4px; border: 1px solid #334155; }
  .modebar-btn path { fill: #94A3B8 !important; }
  .modebar-btn:hover { background: #334155 !important; }
  .modebar-btn.active path { fill: #60A5FA !important; }
</style>
"""

CONTRAST_CSS = """
<style>
  .stApp, [data-testid="stAppViewContainer"] { background: #000000 !important; }
  .block-container { padding-top: 0 !important; max-width: 1280px; background: #000000 !important; }
  #MainMenu, footer, header { visibility: hidden; }

  section[data-testid="stSidebar"] { background: #1A1A1A !important; }
  section[data-testid="stSidebar"] * { color: #FFFF00 !important; }

  .topbar {
    background: linear-gradient(90deg, #000000 0%, #1A1A1A 100%);
    color: #FFFF00; padding: 14px 32px; margin: -60px -50px 0 -50px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 3px solid #FFFF00;
  }
  .topbar-title  { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.5px; color: #FFFF00; }
  .topbar-meta   { font-size: 0.72rem; opacity: 0.9; text-align: right; line-height: 1.6; color: #FFFF00; }

  .stTabs [data-baseweb="tab-list"] {
    background: #1A1A1A !important; border-bottom: 3px solid #FFFF00 !important;
    padding: 0 4px; gap: 2px; margin-top: 12px;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 22px; font-size: 0.85rem; font-weight: 600;
    color: #FFFFFF !important; border-radius: 4px 4px 0 0;
  }
  .stTabs [aria-selected="true"] { color: #FFFF00 !important; border-bottom: 3px solid #FFFF00 !important; }

  .sec-label {
    background: #1A1A1A !important; color: #FFFF00; font-size: 0.73rem; font-weight: 700;
    letter-spacing: 0.8px; padding: 6px 14px; border-radius: 4px 4px 0 0;
    margin-bottom: 0; display: inline-block; width: 100%; border: 2px solid #FFFF00;
  }

  .banner { padding: 18px 24px; border-radius: 8px; margin: 8px 0 16px; }
  .banner-approved   { background:#000000 !important; border: 3px solid #00FF00; }
  .banner-review     { background:#000000 !important; border: 3px solid #FFFF00; }
  .banner-conditional{ background:#000000 !important; border: 3px solid #FF9900; }
  .banner-rejected   { background:#000000 !important; border: 3px solid #FF0000; }

  div[data-testid="metric-container"] {
    background: #1A1A1A !important; border: 2px solid #FFFF00 !important; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 0 10px rgba(255,255,0,.3);
  }

  div[data-testid="metric-container"] * { color: #FFFF00 !important; }

  label { font-size: 0.8rem !important; font-weight: 600 !important;
          color: #FFFF00 !important; text-transform: uppercase; letter-spacing: 0.3px !important; }

  p, div, span, h1, h2, h3, h4 { color: #FFFFFF; }
  .stMarkdown p { color: #FFFFFF !important; }
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #FFFF00 !important; }

  /* Containers with border */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #1A1A1A !important; border-color: #FFFF00 !important; border-radius: 8px !important;
  }

  /* Number inputs */
  [data-testid="stNumberInput"] input {
    background: #1A1A1A !important; color: #FFFF00 !important; border: 2px solid #FFFF00 !important;
  }

  /* Select boxes */
  [data-testid="stSelectbox"] [data-baseweb="select"] { background: #1A1A1A !important; }
  [data-testid="stSelectbox"] [data-baseweb="select"] div { background: #1A1A1A !important; color: #FFFF00 !important; }
  [data-testid="stSelectbox"] [data-baseweb="select"] svg { fill: #FFFF00 !important; }

  /* Radio buttons */
  [data-testid="stRadio"] div { color: #FFFF00 !important; }
  [data-testid="stRadio"] label { color: #FFFF00 !important; }

  /* Expanders */
  [data-testid="stExpander"] { border-color: #FFFF00 !important; background: #1A1A1A !important; }
  [data-testid="stExpander"] summary { color: #FFFF00 !important; background: #1A1A1A !important; }

  /* Alert boxes */
  [data-testid="stAlertContainer"] { background: #1A1A1A !important; color: #FFFF00 !important; border-color: #FFFF00 !important; }
  [data-testid="stAlertContainer"] p { color: #FFFF00 !important; }

  /* DataFrames */
  [data-testid="stDataFrame"] { background: #1A1A1A !important; }
  [data-testid="stDataFrame"] * { color: #FFFF00 !important; }

  /* Dividers */
  [data-testid="stDivider"] { border-color: #FFFF00 !important; }

  /* Dropdown menu list */
  [data-baseweb="popover"] li { background: #1A1A1A !important; color: #FFFF00 !important; }
  [data-baseweb="popover"] li:hover { background: #333300 !important; }
  [data-baseweb="menu"] { background: #1A1A1A !important; }

  [data-baseweb="input"] input, [data-baseweb="select"] div,
  [data-baseweb="textarea"] textarea {
    background: #1A1A1A !important; color: #FFFF00 !important; border: 2px solid #FFFF00 !important;
  }

  /* Buttons — contrast theme */
  .stButton > button,
  [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-secondary"] {
    border-radius: 6px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; min-height: 44px !important;
  }
  button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: #FFFF00 !important; color: #000000 !important; border: 2px solid #FFFF00 !important;
  }
  button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
    background: #FF9900 !important; color: #000000 !important; border: 2px solid #FF9900 !important;
  }
  [data-testid="stDownloadButton"] button,
  [data-testid="stDownloadButton"] [data-testid="stBaseButton-secondary"] {
    background: #00FF00 !important; color: #000000 !important; border: 2px solid #00FF00 !important;
    font-weight: 700 !important; font-size: 0.9rem !important; min-height: 44px !important;
  }

  /* Plotly chart modebar — contrast theme */
  .modebar { background: #1A1A1A !important; border-radius: 4px; border: 1px solid #FFFF00; }
  .modebar-btn path { fill: #FFFF00 !important; }
  .modebar-btn:hover { background: #333300 !important; }
  .modebar-btn.active path { fill: #FFFFFF !important; }
</style>
"""

# ── Apply selected theme CSS ─────────────────────────────────────────────────────────────
if mode_key == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
elif mode_key == "contrast":
    st.markdown(CONTRAST_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ── Top navigation bar ────────────────────────────────────────────────────────
st.markdown(
    '<div class="topbar">'
    '  <div><div class="topbar-title">🏛 CREDITRISK ANALYSER</div>'
    '  <div style="font-size:1.05rem;opacity:.9;margin-top:3px;letter-spacing:.3px">'
    '    Basel III Compliant Credit Risk Assessment Platform</div></div>'
    '  <div class="topbar-meta">AUC: 0.88 &nbsp;|&nbsp; Recall: 83% &nbsp;|&nbsp; Threshold: 40%<br>'
    '    Training set: 32,581 records</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("show_result", False), ("result", None),
             ("form_data", None), ("app_id", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


def reset():
    st.session_state.show_result = False
    st.session_state.result = None


def make_app_id():
    sfx = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
    return f"CR-{datetime.now().strftime('%Y%m%d')}-{sfx}"


# ── Lookup maps ───────────────────────────────────────────────────────────────
HOME_MAP = {
    "Renting":                "RENT",
    "Own (No Mortgage)":      "OWN",
    "Own (With Mortgage)":    "MORTGAGE",
    "Other":                  "OTHER",
}
INTENT_MAP = {
    "Personal Expenses":         "PERSONAL",
    "Education / Student Loan":  "EDUCATION",
    "Medical / Healthcare":      "MEDICAL",
    "Business Venture":          "VENTURE",
    "Home Improvement":          "HOMEIMPROVEMENT",
    "Debt Consolidation":        "DEBTCONSOLIDATION",
}
DEFAULT_MAP = {"No prior defaults": "N", "Has prior default on file": "Y"}

# ── Currency conversion rates (to USD) ───────────────────────────────────────
CURRENCY_RATES = {
    "USD": 1.0,
    "EUR": 1.08,  # Approximate rates as of 2024
    "GBP": 1.27,
    "CAD": 0.73,
    "AUD": 0.66,
    "JPY": 0.0067,
    "INR": 0.012,
}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_app, tab_sens, tab_bulk = st.tabs([
    "  📋  New Application  ",
    "  📊  Sensitivity Analysis  ",
    "  📁  Bulk Scoring  ",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — APPLICATION + RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_app:

    # ── APPLICATION FORM ───────────────────────────────────────────────────
    if not st.session_state.show_result:

        new_id = make_app_id()
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"**Application Reference:** `{new_id}`")
        h2.markdown(f"<div style='text-align:right;font-size:.8rem;color:#6B7280'>"
                    f"{datetime.now().strftime('%d %b %Y  %H:%M')}</div>",
                    unsafe_allow_html=True)
        st.divider()

        # Section A — Personal Information
        st.markdown('<div class="sec-label">A — PERSONAL INFORMATION</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                person_age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=100,
                    value=18,
                    step=1,
                    help="Applicant's age. Must be 18 or older to apply for a loan."
                )
                person_emp_length = st.number_input(
                    "Employment Length (years)",
                    min_value=0,
                    max_value=None,
                    value=0,
                    step=1,
                    help="Total years in current or most recent job. Enter 0 if less than 1 year employed."
                )
            with c2:
                currency = st.selectbox(
                    "Currency",
                    options=list(CURRENCY_RATES.keys()),
                    index=0,  # USD default
                    help="Select the currency for income and loan amount. Values will be converted to USD."
                )
                person_income = st.number_input(
                    f"Annual Gross Income ({currency})",
                    min_value=0,
                    max_value=None,
                    value=0,
                    step=1_000,
                    help="Total annual income before tax. No upper limit — enter actual figure."
                )
                ownership_label = st.selectbox(
                    "Residential Status",
                    options=list(HOME_MAP.keys()),
                    index=None,
                    placeholder="Select residential status…"
                )
            with c3:
                cred_hist = st.number_input(
                    "Years of Credit History",
                    min_value=0,
                    max_value=None,
                    value=0,
                    step=1,
                    help=(
                        "How many years ago you first opened ANY credit account "
                        "(credit card, loan, mortgage, etc.). "
                        "This is the age of your oldest credit account — "
                        "the longer the track record, the lower the perceived risk. "
                        "Example: if your first credit card was opened 8 years ago, enter 8."
                    )
                )
                default_label = st.selectbox(
                    "Prior Default on Record",
                    options=list(DEFAULT_MAP.keys()),
                    index=None,
                    placeholder="Select…"
                )

        # Section B — Loan Request
        st.markdown('<div class="sec-label" style="margin-top:12px">B — LOAN REQUEST</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            b1, b2 = st.columns(2)
            with b1:
                loan_amnt = st.number_input(
                    f"Loan Amount Requested ({currency})",
                    min_value=0,
                    max_value=None,
                    value=0,
                    step=500,
                    help="Total loan amount being requested. No upper limit enforced — the model evaluates affordability."
                )
            with b2:
                intent_label = st.selectbox(
                    "Purpose of Loan",
                    options=list(INTENT_MAP.keys()),
                    index=None,
                    placeholder="Select loan purpose…"
                )

        # Safe fallbacks for pre-assessment while form is being filled
        person_income_usd = person_income * CURRENCY_RATES[currency]
        loan_amnt_usd = loan_amnt * CURRENCY_RATES[currency]
        _inc_safe  = max(person_income_usd, 1)
        _amt_safe  = max(loan_amnt_usd, 1)
        _def_safe  = DEFAULT_MAP.get(default_label, "N") if default_label else "N"
        _hist_safe = max(cred_hist, 1)

        grade       = derive_loan_grade(_inc_safe, _amt_safe, _def_safe, _hist_safe, person_emp_length)
        int_rate    = GRADE_RATE_MAP[grade]
        dti         = loan_amnt / _inc_safe
        emi_preview = calculate_emi(_amt_safe, int_rate, 36)
        int_rate_x  = int_rate * (_amt_safe / _inc_safe)

        # Section C — Pre-Assessment Snapshot
        st.markdown('<div class="sec-label" style="margin-top:12px">C — PRE-ASSESSMENT SNAPSHOT (AUTO-CALCULATED)</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            if person_income == 0 or loan_amnt == 0:
                st.info("Enter income and loan amount above to see your real-time pre-assessment snapshot.")
            else:
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Debt-to-Income",       f"{dti:.1%}",
                          delta="Elevated" if dti > 0.35 else "Normal", delta_color="inverse")
                p2.metric("Preliminary Grade",    grade,
                          help="Internal risk tier A–G auto-derived from your profile. A = lowest risk, G = highest.")
                p3.metric("Indicative Rate",      f"{int_rate:.1f}% p.a.",
                          help="Interest rate band mapped from the preliminary loan grade.")
                p4.metric("Est. Monthly EMI (36m)", f"${emi_preview:,.0f}",
                          help="Estimated equated monthly instalment over 36 months at the indicative rate.")
                p5.metric("Rate × Income-Stress", f"{int_rate_x:.2f}",
                          help="Key model signal: interest rate × debt-to-income. Higher = more stressed.")

        st.divider()

        # Submit + validation
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("Submit Application for Risk Assessment",
                         use_container_width=True, type="primary"):
                errors = []
                if person_income <= 0:
                    errors.append("Annual income must be greater than $0.")
                if loan_amnt <= 0:
                    errors.append("Loan amount must be greater than $0.")
                if ownership_label is None:
                    errors.append("Please select a residential status.")
                if intent_label is None:
                    errors.append("Please select a loan purpose.")
                if default_label is None:
                    errors.append("Please indicate prior default status.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    person_income_usd = person_income * CURRENCY_RATES[currency]
                    loan_amnt_usd = loan_amnt * CURRENCY_RATES[currency]
                    h_own    = HOME_MAP[ownership_label]
                    l_intent = INTENT_MAP[intent_label]
                    p_def    = DEFAULT_MAP[default_label]
                    g        = derive_loan_grade(person_income_usd, loan_amnt_usd, p_def, cred_hist, person_emp_length)
                    rate     = GRADE_RATE_MAP[g]

                    inp = {
                        "person_age":                 person_age,
                        "person_income":              person_income_usd,
                        "person_emp_length":          person_emp_length,
                        "loan_amnt":                  loan_amnt_usd,
                        "loan_int_rate":              rate,
                        "cb_person_cred_hist_length": cred_hist,
                        "person_home_ownership":      h_own,
                        "loan_intent":                l_intent,
                        "loan_grade":                 g,
                        "cb_person_default_on_file":  p_def,
                        "emp_length_missing":         0,
                        "int_rate_missing":           0,
                    }
                    st.session_state.result = predict_risk(inp)
                    st.session_state.form_data = {
                        **inp,
                        "person_income_original": person_income,
                        "loan_amnt_original": loan_amnt,
                        "currency": currency,
                        "ownership_display": ownership_label,
                        "intent_display":    intent_label,
                        "default_display":   default_label,
                    }
                    st.session_state.app_id      = new_id
                    st.session_state.show_result = True
                    st.rerun()

    # ── RESULTS ───────────────────────────────────────────────────────────
    else:
        result = st.session_state.result
        fd     = st.session_state.form_data
        app_id = st.session_state.app_id
        terms  = result["loan_terms"]
        prob   = result["default_probability"]
        dti    = fd["loan_amnt"] / max(fd["person_income"], 1)

        # Action bar
        a1, a2, a3 = st.columns([3, 1, 1])
        a1.markdown(f"**Application:** `{app_id}` &nbsp;·&nbsp; "
                    f"**Assessed:** `{datetime.now().strftime('%d %b %Y %H:%M')}`")
        with a2:
            report_txt = generate_decision_report(result, fd, app_id)
            st.download_button("⬇ Decision Report", data=report_txt,
                               file_name=f"{app_id}.txt", mime="text/plain",
                               use_container_width=True)
        a3.button("🔄 New Application", on_click=reset, use_container_width=True, type="secondary")

        st.divider()

        # Decision banner
        approval = result["approval"]
        css_cls  = {"APPROVED": "banner-approved",
                    "APPROVED WITH REVIEW": "banner-review",
                    "CONDITIONAL APPROVAL": "banner-conditional",
                    "REJECTED": "banner-rejected"}.get(approval, "banner-review")
        icon     = {"APPROVED": "✅", "APPROVED WITH REVIEW": "⚠",
                    "CONDITIONAL APPROVAL": "🔶", "REJECTED": "❌"}.get(approval, "⚠")

        st.markdown(
            f'<div class="banner {css_cls}">'
            f'  <div style="display:flex;justify-content:space-between;align-items:center">'
            f'    <div><div style="font-size:1.35rem;font-weight:800">{icon} {approval}</div>'
            f'         <div style="margin-top:4px;font-size:.9rem">{result["approval_reason"]}</div></div>'
            f'    <div style="text-align:right">'
            f'      <div style="font-size:2rem;font-weight:800">{result["credit_score"]}</div>'
            f'      <div style="font-size:.8rem">{result["score_band"]} Credit Score</div>'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Key metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Default Probability",  f"{prob:.1%}", delta="Threshold 40%", delta_color="off")
        m2.metric("Credit Score",         result["credit_score"], delta=result["score_band"], delta_color="off")
        m3.metric("Risk Classification",  result["risk_band"])
        m4.metric("Debt-to-Income",       f"{dti:.1%}",
                  delta="Elevated" if dti > 0.35 else "Normal", delta_color="inverse")
        m5.metric("Internal Grade",       fd["loan_grade"],
                  delta=f"Rate: {fd['loan_int_rate']:.1f}%", delta_color="off")

        st.divider()

        # Gauges
        g1, g2 = st.columns(2)

        def _gauge(value, ref, title, suffix="%", rng=(0, 100)):
            bar_c = "#DC2626" if value >= ref else "#16A34A"
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                number={"suffix": suffix, "valueformat": ".1f", "font": {"size": 34}},
                delta={"reference": ref, "valueformat": ".1f", "suffix": suffix},
                title={"text": title, "font": {"size": 13}},
                gauge={
                    "axis": {"range": list(rng), "ticksuffix": suffix},
                    "bar": {"color": bar_c, "thickness": 0.22},
                    "bgcolor": "white", "borderwidth": 1, "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [rng[0], ref], "color": "#DCFCE7"},
                        {"range": [ref, rng[0] + (rng[1] - rng[0]) * 0.7], "color": "#FEF9C3"},
                        {"range": [rng[0] + (rng[1] - rng[0]) * 0.7, rng[1]], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "#1E293B", "width": 3},
                                  "thickness": 0.78, "value": ref},
                },
            ))
            fig.update_layout(height=270, margin=dict(t=50, b=10, l=10, r=10),
                              paper_bgcolor="white", plot_bgcolor="white")
            return fig

        with g1:
            st.subheader("Probability of Default")
            st.plotly_chart(_gauge(prob * 100, 40, "PD (%)"), use_container_width=True)
        with g2:
            st.subheader("Credit Score  (300 – 850)")
            fig_sc = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["credit_score"],
                number={"font": {"size": 34}},
                title={"text": result["score_band"], "font": {"size": 13}},
                gauge={
                    "axis": {"range": [300, 850]},
                    "bar": {"color": result["band_colour"], "thickness": 0.22},
                    "bgcolor": "white", "borderwidth": 1, "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [300, 580], "color": "#FEE2E2"},
                        {"range": [580, 670], "color": "#FEF9C3"},
                        {"range": [670, 740], "color": "#F0FDF4"},
                        {"range": [740, 800], "color": "#DCFCE7"},
                        {"range": [800, 850], "color": "#BBF7D0"},
                    ],
                },
            ))
            fig_sc.update_layout(height=270, margin=dict(t=50, b=10, l=10, r=10),
                                  paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig_sc, use_container_width=True, config=_CHART_CONFIG)

        st.divider()

        # Feature contributions + EMI
        fc_col, emi_col = st.columns([3, 2])

        with fc_col:
            st.subheader("Risk Factor Decomposition")
            contribs = result["feature_contributions"]
            feats    = list(contribs.keys())
            vals     = list(contribs.values())
            colors   = ["#EF4444" if v > 0 else "#22C55E" for v in vals]

            fig_fc = go.Figure(go.Bar(
                x=vals, y=feats, orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in vals],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
            ))
            fig_fc.update_layout(
                xaxis_title="Log-Odds Contribution  (red ↑ risk · green ↓ risk)",
                height=400, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#94A3B8", zeroline=True),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=70, t=10, b=40),
                font=dict(size=12),
            )
            fig_fc.add_vline(x=0, line_color="#94A3B8", line_width=1)
            st.plotly_chart(fig_fc, use_container_width=True, config=_CHART_CONFIG)
            st.caption(
                "Contribution = coefficient × standardised feature value. "
                "Mathematically equivalent to SHAP for linear models (exact, not approximated). "
                "Red bars increase default risk; green bars reduce it."
            )

        with emi_col:
            st.subheader("Loan Terms & Affordability")
            st.info(f"Suggested rate band: **{terms['rate_range'][0]}% – {terms['rate_range'][1]}%** "
                    f"(mid: {terms['suggested_rate']}%) based on credit score {result['credit_score']}")

            rows = []
            for t in [12, 24, 36, 48, 60]:
                emi_v  = calculate_emi(fd["loan_amnt"], terms["suggested_rate"], t)
                ratio  = emi_v / (fd["person_income"] / 12)
                status = "🔴 Stressed" if ratio > 0.40 else ("🟡 Moderate" if ratio > 0.25 else "🟢 Healthy")
                rows.append({
                    "Tenure": f"{t} months",
                    "Monthly EMI": f"${emi_v:,.0f}",
                    "EMI / Income": f"{ratio:.1%}",
                    "Affordability": status,
                })
            st.table(pd.DataFrame(rows).set_index("Tenure"))

            if approval != "REJECTED":
                st.metric("Maximum Approved Amount", f"${terms['max_approved_amount']:,.0f}")

            emi36 = calculate_emi(fd["loan_amnt"], terms["suggested_rate"], 36)
            tot   = emi36 * 36
            st.markdown(
                f"**36-month summary** — EMI ${emi36:,.0f} &nbsp;·&nbsp; "
                f"Total interest ${tot - fd['loan_amnt']:,.0f} &nbsp;·&nbsp; "
                f"Total repayment ${tot:,.0f}"
            )

        st.divider()

        # Application summary tables
        st.subheader("Application Summary")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Personal Information**")
            personal_df = pd.DataFrame({
                "Field": ["Age", "Annual Income", "Employment Length",
                          "Home Ownership", "Years of Credit History", "Prior Default"],
                "Value": [f"{fd['person_age']} yrs", f"${fd['person_income']:,.0f}",
                          f"{fd['person_emp_length']} yrs", fd["ownership_display"],
                          f"{fd['cb_person_cred_hist_length']} yrs", fd["default_display"]],
            }).set_index("Field")
            st.table(personal_df)
            st.download_button(
                "⬇ Download CSV",
                data=personal_df.to_csv(),
                file_name=f"{app_id}_personal_info.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with s2:
            st.markdown("**Loan Details & Derived Risk Signals**")
            int_x = fd["loan_int_rate"] * (fd["loan_amnt"] / fd["person_income"])
            r_idx = fd["loan_int_rate"] * (fd["loan_amnt"] / fd["person_income"]) / (fd["person_income"] / 1e4)
            loan_df = pd.DataFrame({
                "Field": ["Loan Amount", "Purpose", "Internal Grade", "Indicative Rate",
                          "Loan-to-Income", "Rate×Income-Stress", "Composite Risk Index"],
                "Value": [f"${fd['loan_amnt']:,.0f}", fd["intent_display"],
                          fd["loan_grade"], f"{fd['loan_int_rate']:.1f}%",
                          f"{dti:.1%}", f"{int_x:.3f}", f"{r_idx:.3f}"],
            }).set_index("Field")
            st.table(loan_df)
            st.download_button(
                "⬇ Download CSV",
                data=loan_df.to_csv(),
                file_name=f"{app_id}_loan_details.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_sens:
    st.subheader("What-If Sensitivity Analysis")

    with st.expander("ℹ How Sensitivity Analysis Works — click to expand"):
        st.markdown("""
**What it shows:** How your Probability of Default (PD) responds to changes in a single input variable,
while all other factors remain constant (ceteris paribus). This is the standard "what-if" technique
used by bank risk teams to understand model sensitivity.

**Methodology:**
1. Each chart scans **50 evenly-spaced values** across a realistic range for that variable.
2. For every value, the loan grade and interest rate are **automatically re-derived** via the same
   heuristic underwriting rules — so results reflect realistic rate changes, not just input changes.
3. All 50 predictions are computed in a **single vectorised batch call** (no Python loop overhead).
4. The <span style='color:#DC2626'>**red dashed line**</span> marks the **40% PD approval threshold** —
   points above this line would be declined or flagged for review.
5. The <span style='color:#F59E0B'>**amber dotted line**</span> marks the **current applicant's position**
   (shown if an application has been submitted in Tab 1).

**How to use it:**
- Submit an application in Tab 1 first — the charts will anchor to that applicant's profile.
- Look for where each curve crosses the red threshold to understand the margin of safety.
- A steep curve means that variable is highly sensitive; a flat curve means it has little impact.
        """, unsafe_allow_html=True)

    st.caption("Grade and interest rate update automatically as income, loan amount, "
               "employment, and credit history change.")

    if st.session_state.form_data:
        fd0 = st.session_state.form_data
    else:
        fd0 = {
            "person_age": 32, "person_income": 60_000, "person_emp_length": 5,
            "loan_amnt": 12_000, "loan_int_rate": 13.0, "cb_person_cred_hist_length": 6,
            "person_home_ownership": "RENT", "loan_intent": "PERSONAL", "loan_grade": "C",
            "cb_person_default_on_file": "N", "emp_length_missing": 0, "int_rate_missing": 0,
        }
        st.info("No application submitted yet — charts use a sample applicant profile "
                "(Income $60k, Loan $12k, Grade C). Submit an application in Tab 1 to anchor to your profile.")

    def _sens_batch(base: dict, key: str, values) -> list:
        inps = []
        for v in values:
            d = {**base, key: v}
            if key in ("person_income", "loan_amnt", "person_emp_length",
                       "cb_person_cred_hist_length", "cb_person_default_on_file"):
                g = derive_loan_grade(
                    d["person_income"], d["loan_amnt"],
                    d["cb_person_default_on_file"],
                    d["cb_person_cred_hist_length"],
                    d["person_emp_length"],
                )
                d["loan_grade"]    = g
                d["loan_int_rate"] = GRADE_RATE_MAP[g]
            inps.append(d)
        return predict_batch_fast(inps)

    THRESH_LINE = dict(y=THRESHOLD * 100, line_dash="dash",
                       line_color="#DC2626", annotation_text="Approval Threshold (40%)",
                       annotation_position="top right")
    CURR_LINE   = dict(line_dash="dot", line_color="#F59E0B",
                       annotation_text="Current", annotation_position="top left")

    layout_base = dict(
        yaxis=dict(title="Probability of Default (%)", range=[0, 100]),
        height=310, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9"), margin=dict(t=30, b=40, l=10, r=10),
    )

    row1c1, row1c2 = st.columns(2)

    with row1c1:
        incomes = np.linspace(10_000, 300_000, 50)
        pds     = [p * 100 for p in _sens_batch(fd0, "person_income", incomes)]
        fig = go.Figure(go.Scatter(x=incomes / 1_000, y=pds, mode="lines",
                                   line=dict(color="#1E3A5F", width=2.5)))
        fig.add_hline(**THRESH_LINE)
        fig.add_vline(x=fd0["person_income"] / 1_000, **CURR_LINE)
        fig.update_layout(title="Annual Income ($K) vs PD", xaxis_title="Income ($K)", **layout_base)
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    with row1c2:
        amounts = np.linspace(500, 100_000, 50)
        pds     = [p * 100 for p in _sens_batch(fd0, "loan_amnt", amounts)]
        fig = go.Figure(go.Scatter(x=amounts / 1_000, y=pds, mode="lines",
                                   line=dict(color="#7C3AED", width=2.5)))
        fig.add_hline(**THRESH_LINE)
        fig.add_vline(x=fd0["loan_amnt"] / 1_000, **CURR_LINE)
        fig.update_layout(title="Loan Amount ($K) vs PD", xaxis_title="Loan Amount ($K)", **layout_base)
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    row2c1, row2c2 = st.columns(2)

    with row2c1:
        hists = np.arange(1, 31)
        pds   = [p * 100 for p in _sens_batch(fd0, "cb_person_cred_hist_length", hists.astype(int))]
        fig   = go.Figure(go.Scatter(x=list(hists), y=pds, mode="lines+markers",
                                     line=dict(color="#059669", width=2.5),
                                     marker=dict(size=4)))
        fig.add_hline(**THRESH_LINE)
        fig.add_vline(x=fd0["cb_person_cred_hist_length"], **CURR_LINE)
        fig.update_layout(title="Years of Credit History vs PD",
                          xaxis_title="Credit History (years)", **layout_base)
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)

    with row2c2:
        emps = np.arange(0, 41)
        pds  = [p * 100 for p in _sens_batch(fd0, "person_emp_length", emps.astype(float))]
        fig  = go.Figure(go.Scatter(x=list(emps), y=pds, mode="lines+markers",
                                    line=dict(color="#D97706", width=2.5),
                                    marker=dict(size=4)))
        fig.add_hline(**THRESH_LINE)
        fig.add_vline(x=fd0["person_emp_length"], **CURR_LINE)
        fig.update_layout(title="Employment Length (yrs) vs PD",
                          xaxis_title="Employment Length (years)", **layout_base)
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BULK SCORING
# ═════════════════════════════════════════════════════════════════════════════
with tab_bulk:
    st.subheader("Bulk Application Scoring")
    st.caption(
        "Score entire loan portfolios in a single batch — "
        "grade and interest rate are **auto-derived** by the risk engine."
    )

    with st.expander("📌 What is Bulk Scoring and why does it matter? — click to expand"):
        st.markdown("""
**Bulk Scoring** (also called *portfolio scoring* or *batch underwriting*) is the process of running
credit risk assessments on hundreds or thousands of loan applications simultaneously.

**Why banks use it:**
- **Pre-screening pipelines** — score all incoming applications overnight before any human review
- **Portfolio stress-testing** — re-score existing loans under new economic scenarios (rate rises, recession)
- **Credit policy calibration** — test the impact of changing the PD threshold from 40% to 35% across the whole book
- **Regulatory reporting** — Basel III requires institutions to quantify expected credit loss (ECL) across the portfolio
- **Risk concentration analysis** — identify if too many high-risk loans are concentrated in one income band or loan purpose

**How it works here:**
1. Download the CSV template → fill in applicant data → upload
2. The engine scores all rows in a **single vectorised call** (fast, no looping)
3. Each applicant is auto-assigned a loan grade and interest rate using the same rules as the single-application form
4. Results include PD, credit score, score band, and approval decision — plus portfolio-level analytics charts below
        """)


    # Valid values reference
    with st.expander("📖 Column Reference & Valid Values — click to expand"):
        st.markdown("""
| Column | Type | Valid Values | Notes |
|--------|------|-------------|-------|
| `person_age` | integer | 18 – 100 | Applicant's age |
| `person_income` | integer | Any positive value | Annual gross income in $ |
| `person_emp_length` | float | 0 – any | Years employed; 0 = less than 1 year |
| `person_home_ownership` | string | `RENT` `OWN` `MORTGAGE` `OTHER` | Exact values, case-insensitive |
| `loan_amnt` | integer | Any positive value | Loan amount in $ |
| `loan_intent` | string | `PERSONAL` `EDUCATION` `MEDICAL` `VENTURE` `HOMEIMPROVEMENT` `DEBTCONSOLIDATION` | Exact values, case-insensitive |
| `cb_person_cred_hist_length` | integer | 0 – any | Years since first credit account was opened |
| `cb_person_default_on_file` | string | `Y` or `N` | Y = has prior default; N = no prior default |

> **Loan grade and interest rate are auto-assigned** by the model — do not add these columns.
> The engine applies the same underwriting heuristics as the single-application form.
        """)

    REQUIRED = ["person_age", "person_income", "person_emp_length",
                "person_home_ownership", "loan_amnt", "loan_intent",
                "cb_person_cred_hist_length", "cb_person_default_on_file"]

    VALID_OWNERSHIP = {"RENT", "OWN", "MORTGAGE", "OTHER"}
    VALID_INTENT    = {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"}
    VALID_DEFAULT   = {"Y", "N"}

    # Template
    template = pd.DataFrame([
        {"person_age": 32, "person_income": 60_000, "person_emp_length": 5,
         "person_home_ownership": "RENT", "loan_amnt": 12_000,
         "loan_intent": "PERSONAL", "cb_person_cred_hist_length": 6,
         "cb_person_default_on_file": "N"},
        {"person_age": 45, "person_income": 95_000, "person_emp_length": 12,
         "person_home_ownership": "MORTGAGE", "loan_amnt": 8_000,
         "loan_intent": "HOMEIMPROVEMENT", "cb_person_cred_hist_length": 15,
         "cb_person_default_on_file": "N"},
        {"person_age": 28, "person_income": 28_000, "person_emp_length": 1,
         "person_home_ownership": "RENT", "loan_amnt": 15_000,
         "loan_intent": "PERSONAL", "cb_person_cred_hist_length": 2,
         "cb_person_default_on_file": "Y"},
    ])

    dl_col, up_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "⬇ Download CSV Template",
            data=template.to_csv(index=False),
            file_name="credit_risk_template.csv",
            mime="text/csv",
            help="Download a pre-filled example with the correct column names and value formats."
        )

    uploaded = up_col.file_uploader(
        "Upload Applications CSV", type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)

            # Column check
            missing_cols = [c for c in REQUIRED if c not in batch.columns]
            if missing_cols:
                st.error(f"Missing required columns: `{'`, `'.join(missing_cols)}`  \n"
                         f"Download the template above to see the expected format.")
                st.stop()

            # Coerce numeric
            batch["person_emp_length"] = pd.to_numeric(
                batch["person_emp_length"], errors="coerce").fillna(0)

            # Validate categorical values
            val_warnings = []
            for i, row in batch.iterrows():
                own   = str(row["person_home_ownership"]).strip().upper()
                intent = str(row["loan_intent"]).strip().upper()
                pdef  = str(row["cb_person_default_on_file"]).strip().upper()
                if own not in VALID_OWNERSHIP:
                    val_warnings.append(f"Row {i + 1}: `person_home_ownership` = '{row['person_home_ownership']}' "
                                        f"— expected one of: RENT, OWN, MORTGAGE, OTHER")
                if intent not in VALID_INTENT:
                    val_warnings.append(f"Row {i + 1}: `loan_intent` = '{row['loan_intent']}' "
                                        f"— expected one of: PERSONAL, EDUCATION, MEDICAL, VENTURE, "
                                        f"HOMEIMPROVEMENT, DEBTCONSOLIDATION")
                if pdef not in VALID_DEFAULT:
                    val_warnings.append(f"Row {i + 1}: `cb_person_default_on_file` = '{row['cb_person_default_on_file']}' "
                                        f"— expected Y or N")

            if val_warnings:
                st.warning(
                    f"⚠ **{len(val_warnings)} validation warning(s)** — these rows will use fallback values:\n\n"
                    + "\n".join(val_warnings[:15])
                    + ("\n\n_… and more_" if len(val_warnings) > 15 else "")
                )

            st.success(f"Loaded **{len(batch)} applications** — running risk assessment…")
            prog = st.progress(0)

            inps = []
            for _, row in batch.iterrows():
                inc   = float(row["person_income"])
                amt   = float(row["loan_amnt"])
                hist  = int(row["cb_person_cred_hist_length"])
                emp   = float(row["person_emp_length"])
                pdef  = str(row["cb_person_default_on_file"]).strip().upper()
                own   = str(row["person_home_ownership"]).strip().upper()
                intnt = str(row["loan_intent"]).strip().upper()

                # Fallback to safe defaults for invalid categoricals
                if own   not in VALID_OWNERSHIP: own   = "RENT"
                if intnt not in VALID_INTENT:    intnt = "PERSONAL"
                if pdef  not in VALID_DEFAULT:   pdef  = "N"

                g = derive_loan_grade(inc, amt, pdef, hist, emp)
                inps.append({
                    "person_age":                 int(row["person_age"]),
                    "person_income":              inc,
                    "person_emp_length":          emp,
                    "loan_amnt":                  amt,
                    "loan_int_rate":              GRADE_RATE_MAP[g],
                    "cb_person_cred_hist_length": hist,
                    "person_home_ownership":      own,
                    "loan_intent":                intnt,
                    "loan_grade":                 g,
                    "cb_person_default_on_file":  pdef,
                    "emp_length_missing":         0,
                    "int_rate_missing":           0,
                })

            probs = predict_batch_fast(inps)
            prog.progress(1.0)

            records = []
            for i, (inp, pd_val) in enumerate(zip(inps, probs)):
                from prediction_helper import calculate_credit_score, get_score_band
                score   = calculate_credit_score(pd_val)
                band, _ = get_score_band(score)
                if pd_val >= 0.70:        verdict = "REJECTED"
                elif pd_val >= 0.55:      verdict = "CONDITIONAL"
                elif pd_val >= THRESHOLD: verdict = "REVIEW"
                else:                     verdict = "APPROVED"
                dti_val = inp["loan_amnt"] / max(inp["person_income"], 1)
                records.append({
                    "#":              i + 1,
                    "Income":         f"${inp['person_income']:,.0f}",
                    "Loan Amt":       f"${inp['loan_amnt']:,.0f}",
                    "DTI":            f"{dti_val:.1%}",
                    "Auto Grade":     inp["loan_grade"],
                    "Rate":           f"{inp['loan_int_rate']:.1f}%",
                    "PD (%)":         f"{pd_val * 100:.1f}%",
                    "Credit Score":   score,
                    "Score Band":     band,
                    "Decision":       verdict,
                })

            results_df = pd.DataFrame(records)

            total    = len(results_df)
            approved = sum(1 for r in records if r["Decision"] == "APPROVED")
            rejected = sum(1 for r in records if r["Decision"] == "REJECTED")
            review   = total - approved - rejected

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total Scored", total)
            sm2.metric("Approved",  approved, delta=f"{approved/total:.0%}", delta_color="normal")
            sm3.metric("Rejected",  rejected, delta=f"{rejected/total:.0%}", delta_color="inverse")
            sm4.metric("Review / Conditional", review)

            st.dataframe(results_df, hide_index=True, use_container_width=True)

            st.caption(
                "**Auto Grade** = loan grade derived by the risk engine from each applicant's profile. "
                "**Rate** = indicative interest rate mapped from that grade. "
                "PD is driven by all 28 model features — the grade/rate interaction is the strongest signal."
            )

            # ── Portfolio Analytics ───────────────────────────────────────────
            st.divider()
            st.subheader("Portfolio Analytics")

            pd_numeric = [float(r["PD (%)"].replace("%", "")) for r in records]
            grade_list = [inp["loan_grade"] for inp in inps]
            decision_list = [r["Decision"] for r in records]

            pa1, pa2 = st.columns(2)

            with pa1:
                # PD distribution histogram
                fig_hist = go.Figure(go.Histogram(
                    x=pd_numeric, nbinsx=20,
                    marker_color="#1E3A5F", marker_line_color="#E2E8F0", marker_line_width=0.5,
                    hovertemplate="PD range: %{x:.1f}%<br>Applications: %{y}<extra></extra>",
                ))
                fig_hist.add_vline(x=THRESHOLD * 100, line_dash="dash", line_color="#DC2626",
                                   annotation_text="Approval threshold", annotation_position="top right")
                fig_hist.update_layout(
                    title="Portfolio PD Distribution",
                    xaxis_title="Probability of Default (%)", yaxis_title="Number of Applications",
                    height=300, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#F1F5F9"), margin=dict(t=40, b=40, l=10, r=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True, config=_CHART_CONFIG)

            with pa2:
                # Decision breakdown donut
                dec_counts  = {d: decision_list.count(d) for d in ["APPROVED", "REVIEW", "CONDITIONAL", "REJECTED"]}
                dec_labels  = [k for k, v in dec_counts.items() if v > 0]
                dec_values  = [dec_counts[k] for k in dec_labels]
                dec_colours = {"APPROVED": "#16A34A", "REVIEW": "#D97706",
                               "CONDITIONAL": "#EA580C", "REJECTED": "#DC2626"}
                fig_pie = go.Figure(go.Pie(
                    labels=dec_labels, values=dec_values,
                    marker_colors=[dec_colours[d] for d in dec_labels],
                    hole=0.55,
                    hovertemplate="%{label}: %{value} applications (%{percent})<extra></extra>",
                ))
                fig_pie.update_layout(
                    title="Decision Breakdown",
                    height=300, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=40, b=10, l=10, r=10), showlegend=True,
                    legend=dict(orientation="v", x=1.0, y=0.5),
                )
                st.plotly_chart(fig_pie, use_container_width=True, config=_CHART_CONFIG)

            pa3, pa4 = st.columns(2)

            with pa3:
                # Average PD by loan grade
                grade_order = ["A", "B", "C", "D", "E", "F", "G"]
                grade_pd = {}
                for g, p in zip(grade_list, pd_numeric):
                    grade_pd.setdefault(g, []).append(p)
                avg_pd_by_grade = {g: round(sum(v) / len(v), 1) for g, v in grade_pd.items()}
                g_sorted = [g for g in grade_order if g in avg_pd_by_grade]
                g_vals   = [avg_pd_by_grade[g] for g in g_sorted]
                g_cols   = ["#16A34A" if v < 40 else ("#D97706" if v < 55 else "#DC2626") for v in g_vals]
                fig_bar = go.Figure(go.Bar(
                    x=g_sorted, y=g_vals, marker_color=g_cols,
                    text=[f"{v:.1f}%" for v in g_vals], textposition="outside",
                    hovertemplate="Grade %{x}: avg PD %{y:.1f}%<extra></extra>",
                ))
                fig_bar.add_hline(y=THRESHOLD * 100, line_dash="dash", line_color="#DC2626",
                                  annotation_text="40% threshold")
                fig_bar.update_layout(
                    title="Average PD by Loan Grade",
                    xaxis_title="Internal Loan Grade", yaxis_title="Average PD (%)",
                    height=300, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#F1F5F9"), margin=dict(t=40, b=40, l=10, r=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True, config=_CHART_CONFIG)

            with pa4:
                # Risk concentration: % of portfolio in each PD band
                bands = {
                    "Low (< 20%)": sum(1 for p in pd_numeric if p < 20),
                    "Moderate (20–40%)": sum(1 for p in pd_numeric if 20 <= p < 40),
                    "Elevated (40–55%)": sum(1 for p in pd_numeric if 40 <= p < 55),
                    "High (55–70%)": sum(1 for p in pd_numeric if 55 <= p < 70),
                    "Very High (≥70%)": sum(1 for p in pd_numeric if p >= 70),
                }
                band_cols = ["#16A34A", "#65A30D", "#D97706", "#EA580C", "#DC2626"]
                fig_conc = go.Figure(go.Bar(
                    x=list(bands.keys()), y=list(bands.values()),
                    marker_color=band_cols,
                    text=list(bands.values()), textposition="outside",
                    hovertemplate="%{x}: %{y} applications<extra></extra>",
                ))
                fig_conc.update_layout(
                    title="Risk Concentration by PD Band",
                    xaxis_title="PD Band", yaxis_title="Number of Applications",
                    height=300, plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#F1F5F9", tickangle=-15),
                    margin=dict(t=40, b=60, l=10, r=10),
                )
                st.plotly_chart(fig_conc, use_container_width=True, config=_CHART_CONFIG)

            # ── Downloads ─────────────────────────────────────────────────────
            st.divider()
            ts_str = datetime.now().strftime("%Y%m%d_%H%M")

            # Build HTML report (open in browser → Ctrl+P → Save as PDF)
            html_rows = "".join(
                f"<tr><td>{r['#']}</td><td>{r['Income']}</td><td>{r['Loan Amt']}</td>"
                f"<td>{r['DTI']}</td><td>{r['Auto Grade']}</td><td>{r['Rate']}</td>"
                f"<td>{r['PD (%)']}</td><td>{r['Credit Score']}</td><td>{r['Score Band']}</td>"
                f"<td style='color:{'#16A34A' if r['Decision']=='APPROVED' else '#DC2626' if r['Decision']=='REJECTED' else '#D97706'};font-weight:600'>{r['Decision']}</td></tr>"
                for r in records
            )
            html_report = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>CreditRisk Analyser — Portfolio Report</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 12px; margin: 20px; color: #1E293B; }}
  h1 {{ font-size: 18px; color: #0F2748; border-bottom: 2px solid #0F2748; padding-bottom: 6px; }}
  h2 {{ font-size: 14px; color: #0F2748; margin-top: 20px; }}
  .meta {{ color: #6B7280; font-size: 11px; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 24px; margin: 16px 0; }}
  .stat {{ background: #F0F4F8; border-radius: 6px; padding: 10px 16px; text-align: center; }}
  .stat .val {{ font-size: 20px; font-weight: 700; color: #0F2748; }}
  .stat .lbl {{ font-size: 10px; color: #6B7280; text-transform: uppercase; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th {{ background: #0F2748; color: white; padding: 6px 8px; text-align: left; font-size: 11px; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #E2E8F0; font-size: 11px; }}
  tr:nth-child(even) {{ background: #F8FAFC; }}
  .footer {{ margin-top: 24px; font-size: 10px; color: #94A3B8; text-align: center; }}
  @media print {{ body {{ margin: 0; }} }}
</style></head><body>
<h1>🏛 CreditRisk Analyser — Portfolio Scoring Report</h1>
<div class='meta'>Generated: {datetime.now().strftime('%d %b %Y %H:%M')} &nbsp;|&nbsp;
AUC: 0.88 &nbsp;|&nbsp; Threshold: PD ≥ 40% &nbsp;|&nbsp; Basel III compliant</div>
<h2>Portfolio Summary</h2>
<div class='summary'>
  <div class='stat'><div class='val'>{total}</div><div class='lbl'>Total Scored</div></div>
  <div class='stat'><div class='val' style='color:#16A34A'>{approved}</div><div class='lbl'>Approved</div></div>
  <div class='stat'><div class='val' style='color:#D97706'>{review}</div><div class='lbl'>Review / Conditional</div></div>
  <div class='stat'><div class='val' style='color:#DC2626'>{rejected}</div><div class='lbl'>Rejected</div></div>
  <div class='stat'><div class='val'>{sum(pd_numeric)/len(pd_numeric):.1f}%</div><div class='lbl'>Avg Portfolio PD</div></div>
</div>
<h2>Application Results</h2>
<table><thead><tr>
  <th>#</th><th>Income</th><th>Loan Amt</th><th>DTI</th><th>Grade</th>
  <th>Rate</th><th>PD (%)</th><th>Credit Score</th><th>Score Band</th><th>Decision</th>
</tr></thead><tbody>{html_rows}</tbody></table>
<div class='footer'>CreditRisk Analyser v3.0 &nbsp;·&nbsp; Logistic Regression + Feature Engineering &nbsp;·&nbsp;
Trained on 32,581 records &nbsp;·&nbsp; To save as PDF: File → Print → Save as PDF</div>
</body></html>"""

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download CSV",
                    data=results_df.to_csv(index=False),
                    file_name=f"scored_{ts_str}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "⬇ Download Report (HTML → print as PDF)",
                    data=html_report,
                    file_name=f"portfolio_report_{ts_str}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Open the downloaded file in a browser, then File → Print → Save as PDF",
                )

        except Exception as e:
            st.error(f"Processing error: {e}")
            raise

