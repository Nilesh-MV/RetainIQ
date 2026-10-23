# ============================================
# RETAINIQ - CUSTOMER CHURN INTELLIGENCE
# Redesigned Streamlit dashboard using the existing trained pipeline.
# ============================================

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RetainIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DESIGN TOKENS ----------------
INK, MUTED, LINE = "#12212B", "#5B6B76", "#DCE2E6"
TEAL = "#0F766E"
LOW, MID, HIGH = "#2F855A", "#B7791F", "#C53030"

# ---------------- LOAD EXISTING MODEL ----------------
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "logistic_regression.pkl")
)


@st.cache_resource
def load_model(model_path: str, modified_time: float):
    """Load the fitted sklearn pipeline once per app process."""
    del modified_time  # Used in the cache key to refresh after the file changes.
    return joblib.load(model_path)


if not os.path.isfile(MODEL_PATH):
    st.error(
        "The trained model was not found at `models/logistic_regression.pkl`. "
        "Keep this app at `app/app.py` and verify that the model file exists."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))
except Exception as exc:
    st.error(
        "RetainIQ could not load the saved model. Check that the model was created "
        "with a compatible scikit-learn version and that the required packages are installed."
    )
    st.exception(exc)
    st.stop()

MODEL_NAME = "Logistic Regression"

# Actual evaluation values from the project's notebook evaluation.
ROC_AUC = 0.8362
TEST_ACCURACY = 0.8031
CV_ROC_AUC = 0.8459

# Illustrative assumptions only: the dataset does not contain retention-offer costs,
# customer lifetime value, or offer success rates.
DEFAULT_BUSINESS = {
    "retained_value": 600,
    "offer_cost": 50,
    "offer_success": 0.30,
}

# ---------------- STYLING ----------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Manrope:wght@400;500;600;700&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button,
.stApp [data-testid="stMetricValue"] {{
    font-family: 'Manrope', system-ui, sans-serif;
}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .riq-display {{
    font-family: 'Bricolage Grotesque', 'Manrope', sans-serif;
    color: {INK};
    letter-spacing: -0.01em;
}}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: 1px solid {LINE}; border-radius: 8px; background: #FFF;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{ font-weight: 600; }}

.riq-top {{
    display:flex; align-items:baseline; justify-content:space-between; gap:16px;
    flex-wrap:wrap; margin-bottom:22px;
}}
.riq-brand {{ font-size:28px; font-weight:700; }}
.riq-sub {{ color:{MUTED}; font-size:14px; margin-top:2px; }}
.riq-chip {{
    font-size:13px; color:{MUTED}; border:1px solid {LINE}; background:#FFF;
    padding:6px 12px; border-radius:999px;
}}
.riq-chip b {{ color:{INK}; }}

.riq-card {{ background:#FFF; border:1px solid {LINE}; border-radius:14px; padding:26px 28px 30px; }}
.riq-row {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px; flex-wrap:wrap; }}
.riq-prob {{
    font-family:'Bricolage Grotesque',sans-serif; font-size:68px; font-weight:700;
    line-height:1; letter-spacing:-0.03em; color:{INK};
}}
.riq-prob-label {{ color:{MUTED}; font-size:14px; margin-top:6px; }}
.riq-pill {{ font-size:14px; font-weight:700; padding:7px 14px; border-radius:999px; color:#FFF; }}
.riq-msg {{ color:{INK}; font-size:15px; line-height:1.55; margin:18px 0 30px; max-width:640px; }}
.ruler {{ position:relative; height:16px; border-radius:8px; display:flex; }}
.ruler .z {{ height:100%; }}
.ruler .z.first {{ border-radius:8px 0 0 8px; }}
.ruler .z.last {{ border-radius:0 8px 8px 0; }}
.ruler .tick {{
    position:absolute; top:-4px; width:2px; height:24px; background:{INK};
    opacity:.45; transform:translateX(-50%);
}}
.ruler .pin {{
    position:absolute; top:-8px; width:6px; height:32px; background:{INK};
    border-radius:3px; transform:translateX(-50%); box-shadow:0 0 0 3px #FFF;
}}
.ruler-zones {{ position:relative; height:20px; margin-top:14px; font-size:12px; font-weight:600; color:{MUTED}; }}
.ruler-zones span {{ position:absolute; transform:translateX(-50%); white-space:nowrap; }}
.ruler-note {{ margin-top:8px; font-size:13px; color:{MUTED}; line-height:1.5; }}

.riq-tile {{ background:#FFF; border:1px solid {LINE}; border-radius:10px; padding:16px 18px; }}
.riq-tile .l {{ color:{MUTED}; font-size:13px; }}
.riq-tile .v {{ font-family:'Bricolage Grotesque',sans-serif; font-size:26px; font-weight:700; margin-top:4px; }}
.riq-tile .n {{ color:{MUTED}; font-size:12px; margin-top:2px; }}
.riq-action {{
    display:flex; gap:14px; align-items:flex-start; background:#FFF;
    border:1px solid {LINE}; border-left:4px solid {TEAL}; border-radius:8px;
    padding:14px 18px; margin-bottom:10px; font-size:15px; line-height:1.5;
}}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:10px 14px; }}
[data-testid="stMetric"] {{ background:#FFF; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:8px; }}
.riq-foot {{ color:{MUTED}; font-size:12px; margin-top:40px; padding-top:16px; border-top:1px solid {LINE}; }}
@media (max-width:640px) {{ .riq-prob {{ font-size:52px; }} .riq-card {{ padding:20px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}

.riq-tiles {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin-top:14px; }}
.riq-snap {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px; }}
.riq-snap span {{
    font-size:13px; color:{INK}; background:#FFF; border:1px solid {LINE};
    border-radius:999px; padding:5px 12px;
}}
.riq-disclaimer {{ color:{MUTED}; font-size:12px; margin-top:6px; line-height:1.5; }}
[data-testid="stSidebarUserContent"] {{ padding-top:1rem; }}
footer {{ visibility:hidden; }}
.stTabs {{ margin-top:10px; }}
@media (max-width:760px) {{ .riq-tiles {{ grid-template-columns:1fr; }} }}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig, height=320, **layout):
    """Apply consistent RetainIQ styling to Plotly charts."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=24, t=14, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK, size=13),
        **layout,
    )
    fig.update_xaxes(gridcolor="#E7ECEF", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#E7ECEF", zeroline=False, linecolor=LINE)
    return fig


def show(fig):
    """Display a Plotly chart without the toolbar."""
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_html(markup: str):
    """Render trusted, app-generated HTML with st.markdown (works on older Streamlit versions).

    Indentation and blank lines are stripped first: Markdown treats indented lines as code
    blocks and ends an HTML block at a blank line, which would break the cards.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ---------------- CUSTOMER PRESETS ----------------
PRESETS = {
    "Blank customer": {},
    "New fibre customer, month-to-month": {
        "tenure": 3,
        "internet": "Fiber optic",
        "contract": "Month-to-month",
        "payment": "Electronic check",
        "monthly": 89.0,
        "total": 267.0,
        "security": "No",
        "backup": "No",
        "device": "No",
        "tech": "No",
        "tv": "No",
        "movies": "No",
        "partner": "No",
        "dependents": "No",
        "paperless": "Yes",
    },
    "Long-term customer on a two-year plan": {
        "tenure": 60,
        "internet": "DSL",
        "contract": "Two year",
        "payment": "Bank transfer (automatic)",
        "monthly": 58.0,
        "total": 3480.0,
        "security": "Yes",
        "backup": "Yes",
        "device": "Yes",
        "tech": "Yes",
        "tv": "No",
        "movies": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "paperless": "No",
    },
}

# Keys and defaults for the sidebar widgets. Using session state lets the
# preset selector update the actual widget values rather than just its labels.
WIDGET_DEFAULTS = {
    "tenure": 12,
    "contract": "Month-to-month",
    "monthly": 70.0,
    "total": 840.0,
    "payment": "Electronic check",
    "paperless": "Yes",
    "internet": "DSL",
    "security": "No",
    "backup": "No",
    "device": "No",
    "tech": "No",
    "tv": "No",
    "movies": "No",
    "phone": "Yes",
    "lines": "No",
    "gender": "Male",
    "senior": 0,
    "partner": "No",
    "dependents": "No",
    # Business assumptions (same keys as DEFAULT_BUSINESS). Presets must not reset these.
    **DEFAULT_BUSINESS,
}


def apply_preset():
    """Copy the selected example profile into the sidebar widget state."""
    selected = st.session_state.get("customer_preset", "Blank customer")
    values = {**WIDGET_DEFAULTS, **PRESETS.get(selected, {})}

    # Service add-ons must be valid for the selected internet service.
    if values["internet"] == "No":
        for key in ("security", "backup", "device", "tech", "tv", "movies"):
            values[key] = "No internet service"
    else:
        for key in ("security", "backup", "device", "tech", "tv", "movies"):
            if values[key] == "No internet service":
                values[key] = "No"

    if values["phone"] == "No":
        values["lines"] = "No phone service"
    elif values["lines"] == "No phone service":
        values["lines"] = "No"

    for key, value in values.items():
        if key not in DEFAULT_BUSINESS:  # keep the user's business assumptions
            st.session_state[key] = value


# Initialize state once, before creating the widgets.
for key, value in WIDGET_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
if "customer_preset" not in st.session_state:
    st.session_state["customer_preset"] = "Blank customer"

sb = st.sidebar
sb.markdown("### Customer profile")
sb.selectbox(
    "Start from an example",
    list(PRESETS.keys()),
    key="customer_preset",
    on_change=apply_preset,
)

# NOTE: inside a `with sb.expander(...)` block, widgets must use `st.` (not `sb.`),
# otherwise they are drawn in the sidebar root and the expander stays empty.
with sb.expander("Account and billing", expanded=True):
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, key="tenure")
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"], key="contract"
    )
    monthly_charges = st.number_input(
        "Monthly charges ($)", min_value=0.0, max_value=200.0, step=0.5, key="monthly"
    )
    total_charges = st.number_input(
        "Total charges to date ($)",
        min_value=0.0,
        max_value=10000.0,
        step=10.0,
        key="total",
        help="Use the customer's recorded total charges when known. For the example profiles, this is prefilled.",
    )
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        key="payment",
    )
    paperless_billing = st.selectbox(
        "Paperless billing", ["Yes", "No"], key="paperless"
    )

with sb.expander("Services", expanded=True):
    internet_service = st.selectbox(
        "Internet service", ["DSL", "Fiber optic", "No"], key="internet"
    )

    def service_addon(label, key):
        if internet_service == "No":
            # Force the exact category used in the original training data.
            st.session_state[key] = "No internet service"
            return st.selectbox(label, ["No internet service"], key=key, disabled=True)
        options = ["Yes", "No"]
        if st.session_state.get(key) not in options:
            st.session_state[key] = "No"
        return st.selectbox(label, options, key=key)

    online_security = service_addon("Online security", "security")
    online_backup = service_addon("Online backup", "backup")
    device_protection = service_addon("Device protection", "device")
    tech_support = service_addon("Tech support", "tech")
    streaming_tv = service_addon("Streaming TV", "tv")
    streaming_movies = service_addon("Streaming movies", "movies")

    phone_service = st.selectbox("Phone service", ["Yes", "No"], key="phone")
    if phone_service == "No":
        st.session_state["lines"] = "No phone service"
        multiple_lines = st.selectbox(
            "Multiple lines", ["No phone service"], key="lines", disabled=True
        )
    else:
        if st.session_state.get("lines") == "No phone service":
            st.session_state["lines"] = "No"
        multiple_lines = st.selectbox("Multiple lines", ["Yes", "No"], key="lines")

with sb.expander("Household", expanded=False):
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior_citizen = st.selectbox("Senior citizen", [0, 1], key="senior")
    partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
    dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")

with sb.expander("Business assumptions", expanded=False):
    st.caption(
        "Illustrative assumptions only. The Telco dataset does not contain retention-offer costs, "
        "customer value, or offer success data."
    )
    retained_value = st.number_input(
        "Value of a retained customer ($)",
        min_value=50,
        max_value=5000,
        step=50,
        key="retained_value",
    )
    offer_cost = st.number_input(
        "Cost of one retention offer ($)",
        min_value=1,
        max_value=1000,
        step=5,
        key="offer_cost",
    )
    offer_success = st.slider(
        "Offer success rate",
        min_value=0.05,
        max_value=1.0,
        step=0.05,
        key="offer_success",
    )

# ---------------- BUILD MODEL INPUT ----------------
# Keep these names and categories aligned with the original training dataset.
customer_data = pd.DataFrame(
    {
        "gender": [gender],
        "SeniorCitizen": [int(senior_citizen)],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [int(tenure)],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [float(monthly_charges)],
        "TotalCharges": [float(total_charges)],
    }
)

try:
    churn_probability = float(model.predict_proba(customer_data)[0][1])
except Exception as exc:
    st.error(
        "The model could not score the customer profile. This usually means the saved "
        "pipeline expects different feature names or categories. Check that the model was "
        "trained on the same 19 input columns used by this app."
    )
    st.exception(exc)
    st.stop()

# ---------------- BUSINESS THRESHOLD ----------------
threshold = offer_cost / (offer_success * retained_value)
high_cut = min(threshold, 1.0)
low_cut = high_cut / 2.0
expected_gain = churn_probability * offer_success * retained_value - offer_cost
break_even_unreachable = threshold >= 1.0

if break_even_unreachable:
    risk_label, risk_color = "No offer priority", MUTED
    risk_message = (
        "The break-even threshold is at or above 100%, so no possible churn probability "
        "can produce positive expected value under these assumptions."
    )
elif churn_probability < low_cut:
    risk_label, risk_color = "Low offer priority", LOW
    risk_message = "The predicted churn probability is well below the illustrative break-even point. Continue regular engagement."
elif churn_probability < high_cut:
    risk_label, risk_color = "Medium offer priority", MID
    risk_message = "The predicted churn probability is below the illustrative break-even point. Consider low-cost actions and continue monitoring."
else:
    risk_label, risk_color = "High offer priority", HIGH
    risk_message = (
        "The predicted churn probability meets or exceeds the illustrative break-even point. "
        "A retention offer may be worth considering under these assumptions."
    )

# ---------------- HEADER ----------------
render_html(
    f"""
<div class="riq-top">
  <div>
    <div class="riq-brand riq-display">RetainIQ</div>
    <div class="riq-sub">Score a telecom customer, understand the model's signal, and explore retention decisions.</div>
  </div>
  <div class="riq-chip"><b>{MODEL_NAME}</b> &nbsp;|&nbsp; ROC-AUC {ROC_AUC:.3f}</div>
</div>
"""
)

# Quick summary of the profile being scored, so the page makes sense without the sidebar.
snapshot = [
    f"{int(tenure)} months tenure",
    contract,
    "No internet service"
    if internet_service == "No"
    else f"{internet_service} internet",
    f"&#36;{monthly_charges:,.2f} per month",
    payment_method,
]
render_html(
    '<div class="riq-snap">'
    + "".join(f"<span>{item}</span>" for item in snapshot)
    + "</div>"
)

# ---------------- RESULT CARD ----------------
p_pct, low_pct, high_pct = churn_probability * 100, low_cut * 100, high_cut * 100
ruler_low = max(0.0, min(low_pct, 100.0))
ruler_high = max(ruler_low, min(high_pct, 100.0))

zone_labels = "".join(
    f'<span style="left:{(start + end) / 2}%">{name}</span>'
    for name, start, end in (
        ("Low", 0, ruler_low),
        ("Medium", ruler_low, ruler_high),
        ("High", ruler_high, 100),
    )
    if end - start >= 8
)
tick = (
    ""
    if break_even_unreachable
    else f'<div class="tick" style="left:{ruler_high}%"></div>'
)
ruler_note = (
    "No churn probability below 100% reaches break-even under these assumptions."
    if break_even_unreachable
    else f"The dark tick marks break-even at {threshold:.1%}. Above it, an offer is expected to pay for itself."
)

render_html(
    f"""
<div class="riq-card">
  <div class="riq-row">
    <div>
      <div class="riq-prob">{p_pct:.1f}%</div>
      <div class="riq-prob-label">Estimated chance this customer churns</div>
    </div>
    <span class="riq-pill" style="background:{risk_color}">{risk_label}</span>
  </div>
  <div class="riq-msg">{risk_message}</div>
  <div class="ruler">
    <div class="z first" style="width:{ruler_low}%; background:{LOW}55"></div>
    <div class="z" style="width:{ruler_high - ruler_low}%; background:{MID}66"></div>
    <div class="z last" style="width:{100 - ruler_high}%; background:{HIGH}66"></div>
    {tick}
    <div class="pin" style="left:{p_pct}%"></div>
  </div>
  <div class="ruler-zones">{zone_labels}</div>
  <div class="ruler-note">{ruler_note}</div>
</div>
"""
)

decision = (
    "No profitable offer"
    if break_even_unreachable
    else ("Consider an offer" if churn_probability >= threshold else "Hold off")
)
gain_color = TEAL if expected_gain >= 0 else HIGH
render_html(
    f"""
<div class="riq-tiles">
  <div class="riq-tile"><div class="l">Illustrative decision</div>
    <div class="v" style="color:{risk_color}">{decision}</div>
    <div class="n">Based on the assumptions you selected</div></div>
  <div class="riq-tile"><div class="l">Expected net value per targeted customer</div>
    <div class="v" style="color:{gain_color}">{"+" if expected_gain >= 0 else "-"}&#36;{abs(expected_gain):,.2f}</div>
    <div class="n">churn probability × offer success × retained value − offer cost</div></div>
  <div class="riq-tile"><div class="l">Break-even churn probability</div>
    <div class="v">{"Above 100%" if break_even_unreachable else f"{threshold:.1%}"}</div>
    <div class="n">offer cost ÷ (success rate × retained value)</div></div>
</div>
"""
)

st.caption(
    "Business-value figures are scenario estimates, not measured RetainIQ outcomes. "
    "They assume a successful offer prevents the otherwise expected loss of the stated retained value."
)

# ---------------- TABS ----------------
tab_why, tab_act, tab_whatif, tab_model = st.tabs(
    ["Why this score", "Recommended actions", "What if tenure changes", "Model health"]
)

# ---------------- WHY: GLOBAL SHAP IMPORTANCE ----------------
with tab_why:
    st.markdown("#### Features with the largest average influence")
    st.caption(
        "Global mean absolute SHAP values calculated during model development. "
        "These explain average model-output influence across the evaluation data, not this individual prediction."
    )
    importance = pd.DataFrame(
        {
            "Feature": [
                "Tenure",
                "Total charges",
                "Two-year contract",
                "Fiber optic internet",
                "One-year contract",
                "Electronic check payment",
                "Online security",
            ],
            "Mean |SHAP|": [
                1.2049,
                0.5110,
                0.4342,
                0.4311,
                0.2876,
                0.1643,
                0.1619,
            ],
        }
    ).sort_values("Mean |SHAP|", ascending=True)

    importance_fig = go.Figure(
        go.Bar(
            x=importance["Mean |SHAP|"],
            y=importance["Feature"],
            orientation="h",
            marker=dict(color=TEAL),
            text=importance["Mean |SHAP|"].round(3),
            textposition="outside",
            cliponaxis=False,
        )
    )
    style_fig(
        importance_fig,
        360,
        xaxis_title="Mean |SHAP| (model output space)",
        showlegend=False,
    )
    show(importance_fig)
    st.caption(
        "SHAP importance is associational: it describes how the model uses features, "
        "not whether a feature causes churn. The values are fixed results from your notebook."
    )

# ---------------- RECOMMENDED ACTIONS ----------------
with tab_act:
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append(
            "Consider an incentive for moving to a longer-term contract."
        )
    if payment_method == "Electronic check":
        recommendations.append(
            "Promote automatic payment options if they suit the customer's preferences."
        )
    if internet_service == "Fiber optic":
        recommendations.append(
            "Review fiber pricing and service satisfaction with the customer."
        )
    if internet_service != "No" and tech_support == "No":
        recommendations.append(
            "Consider a support-focused plan or make technical support options clearer."
        )
    if internet_service != "No" and online_security == "No":
        recommendations.append(
            "Consider whether an online-security add-on or bundle would meet the customer's needs."
        )
    if tenure < 12:
        recommendations.append(
            "The customer is relatively new; consider onboarding support and early engagement."
        )
    if monthly_charges > 80:
        recommendations.append(
            "Review the bill and plan fit to identify avoidable cost or service dissatisfaction."
        )
    if not recommendations:
        recommendations.append(
            "Continue regular engagement and monitor the customer's predicted churn risk."
        )

    st.caption(
        "These are rule-based suggestions derived from profile fields, not recommendations generated by the ML model."
    )
    for recommendation in recommendations:
        st.markdown(
            f'<div class="riq-action"><span>→</span><span>{recommendation}</span></div>',
            unsafe_allow_html=True,
        )

# ---------------- WHAT-IF TENURE ANALYSIS ----------------
with tab_whatif:
    st.markdown("#### Estimated sensitivity to tenure")
    st.caption(
        "Illustrative scenario: tenure changes and TotalCharges is estimated as tenure × current monthly charges. "
        "Real billing histories may differ, so treat this as a sensitivity analysis—not a factual forecast."
    )

    rows = []
    for hypothetical_tenure in range(0, 73, 3):
        hypothetical_row = customer_data.copy()
        hypothetical_row["tenure"] = hypothetical_tenure
        hypothetical_row["TotalCharges"] = hypothetical_tenure * monthly_charges
        probability = float(model.predict_proba(hypothetical_row)[0][1]) * 100
        rows.append(
            {
                "Tenure (months)": hypothetical_tenure,
                "Churn probability (%)": probability,
            }
        )

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_fig = go.Figure()
    sensitivity_fig.add_trace(
        go.Scatter(
            x=sensitivity_df["Tenure (months)"],
            y=sensitivity_df["Churn probability (%)"],
            mode="lines",
            line=dict(color=TEAL, width=3),
            fill="tozeroy",
            fillcolor="rgba(15,118,110,0.10)",
            hovertemplate="%{x} months: %{y:.1f}%<extra></extra>",
            name="Scenario",
        )
    )
    if not break_even_unreachable:
        sensitivity_fig.add_hline(
            y=high_pct,
            line_dash="dash",
            line_color=HIGH,
            annotation_text="Illustrative break-even",
            annotation_position="top right",
        )
    sensitivity_fig.add_trace(
        go.Scatter(
            x=[tenure],
            y=[churn_probability * 100],
            mode="markers",
            marker=dict(size=13, color=risk_color, line=dict(width=3, color="#FFFFFF")),
            hovertemplate="Current profile: %{y:.1f}%<extra></extra>",
            name="Current profile",
        )
    )
    style_fig(
        sensitivity_fig,
        360,
        xaxis_title="Tenure (months)",
        yaxis_title="Churn probability (%)",
        showlegend=False,
    )
    sensitivity_fig.update_yaxes(rangemode="tozero")
    show(sensitivity_fig)

# ---------------- MODEL HEALTH ----------------
with tab_model:
    st.markdown("#### Logistic Regression performance")
    st.caption("Evaluation results from the project's existing test set.")

    # Verified test-set metrics from the notebook
    ROC_AUC = 0.8362
    TEST_ACCURACY = 0.8031
    PRECISION = 0.6474
    RECALL = 0.5695
    F1_SCORE = 0.6060
    CV_ROC_AUC = 0.8459

    # First row: main performance metrics
    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("ROC-AUC", f"{ROC_AUC:.2%}")
    metric2.metric("Test accuracy", f"{TEST_ACCURACY:.2%}")
    metric3.metric("Cross-validation ROC-AUC", f"{CV_ROC_AUC:.2%}")

    # Second row: classification metrics
    metric4, metric5, metric6 = st.columns(3)

    metric4.metric("Precision", f"{PRECISION:.2%}")
    metric5.metric("Recall", f"{RECALL:.2%}")
    metric6.metric("F1-score", f"{F1_SCORE:.2%}")

    st.markdown("#### Confusion matrix")

    # Rows = actual class; columns = predicted class
    confusion_df = pd.DataFrame(
        [[917, 116], [161, 213]],
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"],
    )

    st.dataframe(confusion_df, use_container_width=True)

    st.markdown("#### Understanding the results")

    st.write(
        "- **Accuracy (80.31%):** The percentage of all test customers "
        "classified correctly."
    )

    st.write(
        "- **Precision (64.74%):** Of the customers predicted to churn, "
        "64.74% actually churned."
    )

    st.write(
        "- **Recall (56.95%):** The model identified 56.95% of the "
        "customers who actually churned."
    )

    st.write("- **F1-score (60.60%):** A combined measure of precision and recall.")

    st.write(
        "- **ROC-AUC (83.62%):** Measures how well the model distinguishes "
        "between churners and non-churners across classification thresholds."
    )

    st.warning(
        "The model misses 161 actual churners in this test set. "
        "Improving recall could help RetainIQ identify more at-risk customers, "
        "although it may also increase false-positive predictions."
    )

    st.caption(
        "These metrics are reported from the notebook evaluation. "
        "They are displayed as fixed results and are not recalculated "
        "when a customer profile changes."
    )

# ---------------- MODEL INPUT PREVIEW ----------------
with st.expander("Inputs sent to the model", expanded=False):
    st.dataframe(customer_data, use_container_width=True, hide_index=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="riq-foot">RetainIQ · Telecom customer churn prediction and retention analytics · '
    "Built with scikit-learn, SHAP and Streamlit"
    '<div class="riq-disclaimer">Predictions are estimates from a model trained on the historical Telco '
    "Customer Churn dataset. They are not guarantees that a customer will leave.</div></div>",
    unsafe_allow_html=True,
)
