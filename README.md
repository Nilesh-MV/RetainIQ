# RetainIQ — Telecom Customer Churn Prediction & Retention Analytics

RetainIQ is a machine learning project that predicts the probability of customer churn for a telecom company and provides data-driven retention recommendations.

The project covers data cleaning, exploratory data analysis, feature preprocessing, machine learning model comparison, cross-validation, SHAP-based explainability, and customer-level churn risk analysis through an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Customer churn is a major challenge for telecom companies because losing existing customers can negatively impact revenue and customer lifetime value.

RetainIQ aims to identify customers who may be at risk of churning and provide actionable insights that can support customer retention strategies.

The system aims to:

- Predict the probability of customer churn.
- Classify customers into different risk levels.
- Identify important factors influencing churn predictions.
- Compare multiple machine learning classification models.
- Explain model predictions using SHAP.
- Generate customer-specific retention recommendations.
- Provide an interactive dashboard for churn risk analysis.

---

## 🎯 Objectives

1. Clean and preprocess telecom customer data.
2. Perform exploratory data analysis (EDA).
3. Handle numerical and categorical features using appropriate preprocessing techniques.
4. Train and compare multiple classification models.
5. Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Use stratified cross-validation to evaluate model stability.
7. Apply SHAP for model explainability.
8. Build an interactive Streamlit dashboard.
9. Generate rule-based retention recommendations based on customer characteristics.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer information related to:

- Customer demographics
- Tenure
- Phone services
- Internet services
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Billing information
- Payment method
- Monthly charges
- Total charges
- Churn status

### Dataset After Cleaning

After data cleaning, the dataset contains:

- **7,032 customers**
- **19 input features**
- **1 target variable**

### Target Variable

```text
Churn
