# RetainIQ — Telecom Customer Churn Prediction & Retention Analytics

RetainIQ is a machine learning project that predicts the probability of customer churn for a telecom company and provides data-driven retention recommendations.

The project performs data cleaning, exploratory data analysis, feature engineering, machine learning model comparison, cross-validation, SHAP-based explainability, and interactive customer-level churn risk analysis through a Streamlit dashboard.

---

## 📌 Problem Statement

Customer churn is a major challenge for telecom companies because losing existing customers can negatively impact revenue and customer lifetime value.

RetainIQ aims to identify customers who may be at higher risk of churn and provide insights that can help businesses take proactive retention actions.

The system:

- Predicts whether a customer is likely to churn.
- Estimates the customer's churn probability.
- Identifies important factors influencing predictions.
- Compares multiple machine learning models.
- Provides customer-specific retention recommendations.
- Presents the results through an interactive dashboard.

---

## 🎯 Objectives

1. Clean and preprocess telecom customer data.
2. Perform exploratory data analysis (EDA).
3. Prepare numerical and categorical features for machine learning.
4. Train multiple classification models.
5. Compare model performance using appropriate evaluation metrics.
6. Use cross-validation to assess model stability.
7. Apply SHAP for model explainability.
8. Build an interactive Streamlit dashboard.
9. Generate rule-based retention recommendations based on customer characteristics.

---

## 📊 Dataset

RetainIQ uses the **Telco Customer Churn** dataset from Kaggle.

The dataset contains customer information related to:

- Demographics
- Customer tenure
- Phone services
- Multiple lines
- Internet services
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn status

### Dataset after cleaning

- **7,032 customers**
- **19 input features**
- **1 target variable**

### Target Variable

The original `Churn` column contains:

```text
Yes
No
