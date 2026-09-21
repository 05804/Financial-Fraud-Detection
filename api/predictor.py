import joblib
import numpy as np
import pandas as pd
import boto3
from pathlib import Path


S3_BUCKET = "financial-fraud-detection-169523632531"
S3_KEY = "models/fraud_model.pkl"

LOCAL_MODEL_PATH = Path("/tmp/fraud_model.pkl")


# Download the model from private S3 when the API starts
s3 = boto3.client("s3")
s3.download_file(
    S3_BUCKET,
    S3_KEY,
    str(LOCAL_MODEL_PATH)
)


# Load the model once when the API starts
model_data = joblib.load(LOCAL_MODEL_PATH)

model = model_data["model"]
threshold = model_data["threshold"]


def predict_transaction(transaction: dict):
    transaction_df = pd.DataFrame([transaction])

    transaction_df["Amount_Log"] = np.log1p(
        transaction_df["Amount"]
    )

    transaction_df["Hour"] = (
        transaction_df["Time"] // 3600
    ) % 24

    transaction_df["Hour_Sin"] = np.sin(
        2 * np.pi * transaction_df["Hour"] / 24
    )

    transaction_df["Hour_Cos"] = np.cos(
        2 * np.pi * transaction_df["Hour"] / 24
    )

    transaction_df = transaction_df[model.feature_names_in_]

    fraud_probability = model.predict_proba(
        transaction_df
    )[:, 1][0]

    prediction = int(
        fraud_probability >= threshold
    )

    risk_score = fraud_probability * 100

    if risk_score < 30:
        risk_level = "Low Risk"
    elif risk_score < 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_score": round(float(risk_score), 2),
        "risk_level": risk_level,
        "prediction": "FRAUD" if prediction == 1 else "NORMAL"
    }