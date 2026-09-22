# ============================================
# RETAINIQ - STREAMLIT APP
# ============================================

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="RetainIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "logistic_regression.pkl",
)

model = joblib.load(MODEL_PATH)


# ============================================
# GLOBAL DASHBOARD STYLING
# ============================================

st.markdown(
    """
    <style>

    /* ========================================
       GLOBAL
    ======================================== */

    .main {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================
       HERO SECTION
    ======================================== */

    .retainiq-hero {
        padding: 36px 40px;
        margin: 5px 0 35px 0;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.20), rgba(124, 58, 237, 0.17));
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.18);
    }

    .retainiq-brand {
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 14px;
    }

    .retainiq-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.12;
        margin-bottom: 14px;
        letter-spacing: -1.2px;
    }

    .retainiq-description {
        font-size: 16px;
        line-height: 1.65;
        opacity: 0.76;
        max-width: 760px;
        margin-bottom: 22px;
    }

    .retainiq-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.13);
        font-size: 14px;
        font-weight: 650;
    }


    /* ========================================
       SECTION LABELS
    ======================================== */

    .section-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 14px;
        opacity: 0.60;
        margin-bottom: 18px;
    }


    /* ========================================
       INPUT SECTION
    ======================================== */

    [data-testid="stSelectbox"] > div,
    [data-testid="stNumberInput"] > div {
        border-radius: 10px;
    }


    /* ========================================
       PREDICT BUTTON
    ======================================== */

    div.stButton > button {
        min-height: 3.3rem;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 700;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.20);
    }


    /* ========================================
       RISK ANALYSIS CARD (color set dynamically via inline style)
    ======================================== */

    .risk-card {
        padding: 32px;
        margin: 10px 0 24px 0;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.88), rgba(15, 23, 42, 0.96));
        border-left: 5px solid var(--risk-color, #eab308);
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
        height: 100%;
    }

    .risk-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }

    .risk-title {
        font-size: 19px;
        font-weight: 800;
        color: var(--risk-color, #eab308);
        letter-spacing: 0.3px;
    }

    .risk-model {
        font-size: 12px;
        opacity: 0.65;
        padding: 6px 11px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.20);
    }

    .risk-probability {
        font-size: 56px;
        font-weight: 800;
        letter-spacing: -1.8px;
        line-height: 1;
    }

    .risk-probability-label {
        font-size: 13px;
        opacity: 0.58;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .risk-message {
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.75;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        padding-top: 16px;
        margin-top: 4px;
    }


    /* ========================================
       SIDE METRIC STACK
    ======================================== */

    .side-metric {
        padding: 16px 18px;
        margin-bottom: 14px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(148, 163, 184, 0.035);
    }

    .side-metric-label {
        font-size: 12px;
        opacity: 0.55;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .side-metric-value {
        font-size: 24px;
        font-weight: 750;
    }


    /* ========================================
       KPI CARDS (native st.metric, kept for model perf section)
    ======================================== */

    [data-testid="stMetric"] {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(148, 163, 184, 0.035);
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px;
    }


    /* ========================================
       RECOMMENDATION CARDS
    ======================================== */

    .recommendation-card {
        padding: 17px 20px;
        margin: 9px 0;
        border-radius: 13px;
        background: rgba(37, 99, 235, 0.07);
        border: 1px solid rgba(96, 165, 250, 0.14);
        font-size: 14px;
        line-height: 1.55;
    }

    .recommendation-icon {
        margin-right: 8px;
    }


    /* ========================================
       MODEL INSIGHTS CARD
    ======================================== */

    .insight-intro {
        padding: 17px 20px;
        margin-bottom: 18px;
        border-radius: 13px;
        background: rgba(124, 58, 237, 0.07);
        border: 1px solid rgba(167, 139, 250, 0.14);
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.85;
    }


    /* ========================================
       DATAFRAME
    ======================================== */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* ========================================
       DIVIDERS
    ======================================== */

    hr {
        margin-top: 28px;
        margin-bottom: 28px;
    }


    /* ========================================
       FOOTER
    ======================================== */

    .retainiq-footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        font-size: 12px;
        opacity: 0.45;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================
# DASHBOARD HEADER
# ============================================

st.html(
    """
    <div class="retainiq-hero">
        <div class="retainiq-brand">📊 RetainIQ</div>
        <div class="retainiq-title">Customer Churn Intelligence</div>
        <div class="retainiq-description">
            Predict customer churn risk, understand the factors
            influencing predictions, and identify retention actions.
        </div>
        <div class="retainiq-badge">
            🧠 Logistic Regression &nbsp; • &nbsp; ROC-AUC 83.62%
        </div>
    </div>
    """
)


# ============================================
# CUSTOMER INFORMATION (collapsible once a prediction exists)
# ============================================

if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False

input_expanded = not st.session_state.has_predicted

with st.expander("Customer Profile & Inputs", expanded=input_expanded):
    st.markdown(
        '<div class="section-label">CUSTOMER PROFILE</div>', unsafe_allow_html=True
    )
    st.subheader("Customer Information")
    st.caption(
        "Enter the customer's service and demographic information to estimate churn risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    st.divider()

    st.markdown(
        '<div class="section-label">BILLING & CONTRACT</div>', unsafe_allow_html=True
    )
    st.subheader("Billing & Contract Information")
    st.caption("Provide the customer's contract and billing details.")

    col3, col4 = st.columns(2)

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col4:
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.1
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0,
            step=10.0,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">INPUT PREVIEW</div>', unsafe_allow_html=True
    )
    st.subheader("Customer Data")

    customer_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
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
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        }
    )

    st.dataframe(customer_data, use_container_width=True, hide_index=True)

    st.divider()

    predict_clicked = st.button(
        "🔍 Predict Churn Risk", type="primary", use_container_width=True
    )

    if predict_clicked:
        st.session_state.has_predicted = True


# ============================================
# CHURN PREDICTION + RESULTS
# ============================================

if st.session_state.has_predicted:
    churn_probability = model.predict_proba(customer_data)[0][1]

    if churn_probability < 0.30:
        risk_level = "Low Risk"
        risk_label = "LOW RISK"
        risk_emoji = "🟢"
        risk_color = "#22c55e"
        risk_message = "Customer shows a relatively low probability of churn. Continue regular engagement."
    elif churn_probability < 0.60:
        risk_level = "Medium Risk"
        risk_label = "MEDIUM RISK"
        risk_emoji = "🟡"
        risk_color = "#eab308"
        risk_message = "Customer shows a moderate probability of churn. Targeted retention actions may be useful."
    else:
        risk_level = "High Risk"
        risk_label = "HIGH RISK"
        risk_emoji = "🔴"
        risk_color = "#ef4444"
        risk_message = "Customer shows a high probability of churn. Prioritized retention attention is recommended."

    st.divider()
    st.markdown(
        '<div class="section-label">PREDICTION RESULT</div>', unsafe_allow_html=True
    )
    st.subheader("Churn Risk Analysis")

    # ----------------------------------------
    # RESULT LAYOUT: risk card (left) + gauge (right)
    # ----------------------------------------

    result_col1, result_col2 = st.columns([1.1, 1])

    with result_col1:
        st.html(
            f"""
            <div class="risk-card" style="--risk-color: {risk_color};">
                <div class="risk-card-header">
                    <span class="risk-title">{risk_emoji} {risk_label}</span>
                    <span class="risk-model">Logistic Regression</span>
                </div>
                <div class="risk-probability">{churn_probability:.1%}</div>
                <div class="risk-probability-label">Estimated Churn Probability</div>
                <div class="risk-message">{risk_message}</div>
            </div>
            """
        )

    with result_col2:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=churn_probability * 100,
                number={"suffix": "%", "font": {"size": 42}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": risk_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(34, 197, 94, 0.18)"},
                        {"range": [30, 60], "color": "rgba(234, 179, 8, 0.18)"},
                        {"range": [60, 100], "color": "rgba(239, 68, 68, 0.18)"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Risk Level", risk_level)
        with m2:
            st.metric("Model", "LogReg")

    # ----------------------------------------
    # RISK INTERPRETATION
    # ----------------------------------------

    st.subheader("Risk Interpretation")

    if risk_level == "Low Risk":
        st.success(
            "🟢 Low risk: Continue regular customer engagement and monitor the customer's churn risk."
        )
    elif risk_level == "Medium Risk":
        st.warning(
            "🟡 Medium risk: Consider targeted retention actions to reduce the possibility of customer churn."
        )
    else:
        st.error("🔴 High risk: Customer may require prioritized retention attention.")

    # ----------------------------------------
    # RETENTION RECOMMENDATIONS
    # ----------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-label">BUSINESS ACTIONS</div>', unsafe_allow_html=True
    )
    st.subheader("Retention Recommendations")
    st.caption(
        "Potential actions are generated from the customer's current service and billing characteristics."
    )

    recommendations = []

    if contract == "Month-to-month":
        recommendations.append(
            "Consider offering a discount or incentive for a longer-term contract."
        )
    if payment_method == "Electronic check":
        recommendations.append(
            "Consider promoting automatic payment methods for a smoother billing experience."
        )
    if internet_service == "Fiber optic":
        recommendations.append(
            "Review fiber-optic pricing and service satisfaction for this customer."
        )
    if tech_support == "No":
        recommendations.append(
            "Consider offering technical support or a support-focused plan."
        )
    if online_security == "No":
        recommendations.append(
            "Consider offering an online security add-on or bundled plan."
        )
    if tenure < 12:
        recommendations.append(
            "Customer has relatively low tenure; consider an early-stage retention offer."
        )
    if monthly_charges > 80:
        recommendations.append(
            "Review whether the customer's current monthly charges can be optimized."
        )
    if not recommendations:
        recommendations.append(
            "Continue regular customer engagement and monitor churn risk."
        )

    for recommendation in recommendations:
        st.html(
            f"""
            <div class="recommendation-card">
                <span class="recommendation-icon">💡</span>
                {recommendation}
            </div>
            """
        )

    # ----------------------------------------
    # TENURE SENSITIVITY CHART
    # ----------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-label">WHAT-IF ANALYSIS</div>', unsafe_allow_html=True
    )
    st.subheader("Churn Probability vs. Tenure")
    st.caption(
        "Shows how this customer's predicted churn probability would change "
        "at different tenure lengths, holding all other attributes fixed."
    )

    tenure_range = list(range(0, 73, 4))
    sensitivity_rows = []
    for t in tenure_range:
        row = customer_data.copy()
        row["tenure"] = t
        prob = model.predict_proba(row)[0][1]
        sensitivity_rows.append({"tenure": t, "probability": prob})

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    sens_fig = go.Figure()
    sens_fig.add_trace(
        go.Scatter(
            x=sensitivity_df["tenure"],
            y=sensitivity_df["probability"] * 100,
            mode="lines+markers",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.12)",
        )
    )
    sens_fig.add_trace(
        go.Scatter(
            x=[tenure],
            y=[churn_probability * 100],
            mode="markers",
            marker=dict(size=14, color=risk_color, line=dict(width=2, color="white")),
            name="This customer",
        )
    )
    sens_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis_title="Tenure (months)",
        yaxis_title="Churn Probability (%)",
        showlegend=False,
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
    )
    st.plotly_chart(sens_fig, use_container_width=True)

    # ----------------------------------------
    # MODEL INSIGHTS - SHAP BAR CHART
    # ----------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-label">MODEL EXPLAINABILITY</div>', unsafe_allow_html=True
    )
    st.subheader("Model Insights")

    st.html(
        """
        <div class="insight-intro">
            <strong>How the model makes predictions</strong><br>
            SHAP analysis was used during model development to
            understand which features contributed most strongly
            to the Logistic Regression model's predictions.
            Higher mean absolute SHAP values indicate greater
            average influence on the model output.
        </div>
        """
    )

    model_insights = pd.DataFrame(
        {
            "Feature": [
                "Tenure",
                "Total Charges",
                "Two-Year Contract",
                "Fiber Optic Internet",
                "One-Year Contract",
                "Electronic Check Payment",
                "Online Security",
            ],
            "Mean Absolute SHAP": [
                1.2049,
                0.5110,
                0.4342,
                0.4311,
                0.2876,
                0.1643,
                0.1619,
            ],
        }
    ).sort_values("Mean Absolute SHAP", ascending=True)

    shap_fig = go.Figure(
        go.Bar(
            x=model_insights["Mean Absolute SHAP"],
            y=model_insights["Feature"],
            orientation="h",
            marker=dict(
                color=model_insights["Mean Absolute SHAP"],
                colorscale=[[0, "#2563eb"], [1, "#a78bfa"]],
            ),
            text=model_insights["Mean Absolute SHAP"].round(3),
            textposition="outside",
        )
    )
    shap_fig.update_layout(
        height=340,
        margin=dict(l=10, r=40, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="",
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    st.caption(
        "Note: SHAP importance values represent average feature "
        "contribution magnitude in the model's output space; "
        "they are not churn probabilities or percentages."
    )

    # ----------------------------------------
    # MODEL PERFORMANCE
    # ----------------------------------------

    st.divider()
    st.markdown(
        '<div class="section-label">MODEL INFORMATION</div>', unsafe_allow_html=True
    )
    st.subheader("Model Performance")

    performance_col1, performance_col2, performance_col3 = st.columns(3)

    with performance_col1:
        st.metric("ROC-AUC", "83.62%")
    with performance_col2:
        st.metric("Test Accuracy", "80.31%")
    with performance_col3:
        st.metric("Cross-Validation ROC-AUC", "84.59%")

    st.caption(
        "Performance metrics were calculated during model evaluation on the Telco Customer Churn dataset."
    )


# ============================================
# FOOTER
# ============================================

st.html(
    """
    <div class="retainiq-footer">
        RetainIQ • Telecom Customer Churn Prediction & Retention Analytics
        <br>
        Machine Learning • Explainable AI • Streamlit
    </div>
    """
)
