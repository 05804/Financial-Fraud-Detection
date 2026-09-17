import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ========================================================
# PAGE CONFIGURATION
# ========================================================

st.set_page_config(
    page_title="Financial Fraud Detector",
    page_icon="🔐",
    layout="wide"
)


# ========================================================
# FILE PATHS
# ========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


# ========================================================
# LOAD MODEL
# ========================================================

@st.cache_resource
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["threshold"]


model, threshold = load_model()


# ========================================================
# LOAD DATA
# ========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ========================================================
# TITLE
# ========================================================

st.title("🔐 Financial Fraud Detection System")

st.markdown(
    """
    **Machine Learning-powered transaction risk analysis**

    This application uses a trained **XGBoost model** to estimate
    fraud probability and assign a risk level to a transaction.
    """
)


# ========================================================
# SIDEBAR
# ========================================================

st.sidebar.header("Transaction Selection")

transaction_index = st.sidebar.number_input(
    "Transaction Index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)


# ========================================================
# SELECT TRANSACTION
# ========================================================

transaction = df.iloc[[transaction_index]].copy()


# ========================================================
# FEATURE ENGINEERING
# ========================================================

transaction["Amount_Log"] = np.log1p(
    transaction["Amount"]
)

transaction["Hour"] = (
    transaction["Time"] // 3600
) % 24

transaction["Hour_Sin"] = np.sin(
    2 * np.pi * transaction["Hour"] / 24
)

transaction["Hour_Cos"] = np.cos(
    2 * np.pi * transaction["Hour"] / 24
)


# ========================================================
# PREPARE MODEL FEATURES
# ========================================================

X_transaction = transaction.drop(
    columns=["Class"]
)


# ========================================================
# PREDICTION
# ========================================================

fraud_probability = model.predict_proba(
    X_transaction
)[:, 1][0]

prediction = int(
    fraud_probability >= threshold
)

risk_score = fraud_probability * 100


# ========================================================
# RISK LEVEL
# ========================================================

if risk_score < 30:
    risk_level = "Low Risk"
elif risk_score < 70:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"


# ========================================================
# TRANSACTION INFORMATION
# ========================================================

st.subheader("Transaction Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Transaction Index",
        transaction_index
    )

with col2:
    st.metric(
        "Transaction Amount",
        f"{transaction['Amount'].iloc[0]:,.2f}"
    )

with col3:
    st.metric(
        "Approx. Hour",

        f"{transaction['Hour'].iloc[0]:.0f}:00"
    )


# ========================================================
# PREDICTION RESULTS
# ========================================================

st.subheader("Fraud Assessment")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

with col2:
    st.metric(
        "Risk Score",
        f"{risk_score:.2f}/100"
    )

with col3:
    st.metric(
        "Risk Level",
        risk_level
    )

with col4:
    prediction_text = (
        "🚨 FRAUD"
        if prediction == 1
        else "✅ NORMAL"
    )

    st.metric(
        "Prediction",
        prediction_text
    )


# ========================================================
# VISUAL RISK BAR
# ========================================================

st.subheader("Risk Score")

st.progress(
    min(int(risk_score), 100)
)


# ========================================================
# INTERPRETATION
# ========================================================

if prediction == 1:

    st.error(
        "🚨 This transaction has been classified as potentially fraudulent."
    )

elif risk_score >= 70:

    st.warning(
        "⚠️ This transaction has a high predicted risk."
    )

elif risk_score >= 30:

    st.warning(
        "⚠️ This transaction has a medium predicted risk."
    )

else:

    st.success(
        "✅ This transaction has been classified as normal."
    )

# ========================================================
# RISK LEVEL GUIDE
# ========================================================

st.divider()

st.subheader("📌 Risk Level Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        "**LOW RISK**\n\n"
        "Risk Score: 0–29.99"
    )

with col2:
    st.warning(
        "**MEDIUM RISK**\n\n"
        "Risk Score: 30–69.99"
    )

with col3:
    st.error(
        "**HIGH RISK**\n\n"
        "Risk Score: 70–100"
    )


st.caption(
    "Risk scores are derived from the model's estimated fraud probability. "
    "The decision threshold for fraud classification is 0.2."
)
# ========================================================
# MODEL INFORMATION
# ========================================================

st.divider()

# ========================================================
# FRAUD ANALYTICS
# ========================================================

st.divider()

st.subheader("📊 Fraud Analytics")

# Transaction counts
total_transactions = len(df)
fraud_transactions = int((df["Class"] == 1).sum())
normal_transactions = int((df["Class"] == 0).sum())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:
    st.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )

with col3:
    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )


# Fraud distribution
st.markdown("### Transaction Class Distribution")

class_distribution = pd.DataFrame(
    {
        "Transaction Type": [
            "Normal",
            "Fraud"
        ],
        "Count": [
            normal_transactions,
            fraud_transactions
        ]
    }
)

st.bar_chart(
    class_distribution.set_index("Transaction Type")
)


# Model performance
st.markdown("### Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "PR-AUC",
        "0.8771"
    )

with col2:
    st.metric(
        "Precision",
        "88%"
    )

with col3:
    st.metric(
        "Recall",
        "85%"
    )

with col4:
    st.metric(
        "F1 Score",
        "0.86"
    )

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "Decision Threshold",
        f"{threshold:.1f}"
    )

with col3:
    st.metric(
        "Test PR-AUC",
        "0.8771"
    )


# ========================================================
# ORIGINAL TRANSACTION DETAILS
# ========================================================

with st.expander("View Transaction Features"):

    display_transaction = transaction.drop(
        columns=["Class"],
        errors="ignore"
    )

    st.dataframe(
        display_transaction.T,
        use_container_width=True
    )


# ========================================================
# ACTUAL DATASET LABEL
# ========================================================

with st.expander("Dataset Reference"):

    actual_class = transaction["Class"].iloc[0]

    if actual_class == 1:
        st.write("Actual dataset label: **Fraud**")
    else:
        st.write("Actual dataset label: **Normal**")