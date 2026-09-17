import pandas as pd
import numpy as np
import joblib


# ========================================================
# LOAD SAVED MODEL
# ========================================================

model_data = joblib.load("models/fraud_model.pkl")

model = model_data["model"]
threshold = model_data["threshold"]

print("Model loaded successfully!")
print("Fraud detection threshold:", threshold)


# ========================================================
# LOAD ONE TRANSACTION
# ========================================================

df = pd.read_csv("data/raw/creditcard.csv")

transaction = df.iloc[[0]].copy()


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
# PREPARE FEATURES
# ========================================================

X_transaction = transaction.drop(
    columns=["Class"]
)


# ========================================================
# MAKE PREDICTION
# ========================================================

fraud_probability = model.predict_proba(
    X_transaction
)[:, 1][0]

prediction = int(
    fraud_probability >= threshold
)


# ========================================================
# CALCULATE RISK
# ========================================================

risk_score = fraud_probability * 100

if risk_score < 30:
    risk_level = "Low Risk"
elif risk_score < 70:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"


# ========================================================
# DISPLAY RESULT
# ========================================================

print("\n============================================")
print("FINANCIAL FRAUD PREDICTION")
print("============================================")

print(
    f"Transaction Amount: {transaction['Amount'].iloc[0]:.2f}"
)

print(
    f"Fraud Probability: {fraud_probability * 100:.2f}%"
)

print(
    f"Risk Score: {risk_score:.2f}"
)

print(
    f"Risk Level: {risk_level}"
)

print(
    f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}"
)