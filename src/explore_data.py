import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv("data/raw/creditcard.csv")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Convert transaction amount into a log-scaled feature
# This reduces the effect of very large transaction amounts.
df["Amount_Log"] = np.log1p(df["Amount"])


# Convert Time (seconds) into approximate hour of the day
df["Hour"] = (df["Time"] // 3600) % 24

df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)


# ============================================================
# 2A. ANALYZE FRAUD TRANSACTIONS BY HOUR
# ============================================================

# Count fraud transactions for each hour
fraud_by_hour = (
    df[df["Class"] == 1]
    .groupby("Hour")
    .size()
)

print("\nFraud transactions by hour:")
print(fraud_by_hour)


# ============================================================
# 2B. CALCULATE FRAUD RATE BY HOUR
# ============================================================

# Calculate total transactions and fraud transactions
# for every hour.
hourly_stats = (
    df.groupby("Hour")["Class"]
    .agg(
        total_transactions="count",
        fraud_transactions="sum"
    )
)

# Calculate percentage of transactions that are fraud
# for each hour.
hourly_stats["fraud_rate"] = (
    hourly_stats["fraud_transactions"]
    / hourly_stats["total_transactions"]
    * 100
)

print("\nFraud rate by hour:")
print(
    hourly_stats
    .sort_values("fraud_rate", ascending=False)
)


# ============================================================
# 2C. VISUALIZE FRAUD RATE BY HOUR
# ============================================================

plt.figure(figsize=(10, 5))

hourly_stats["fraud_rate"].sort_index().plot(
    kind="bar"
)

plt.title("Fraud Rate by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 2D. VERIFY NEW FEATURES
# ============================================================

print("\nNew features created:")

print(df[[
    "Amount",
    "Amount_Log",
    "Time",
    "Hour",
    "Hour_Sin",
    "Hour_Cos"
]].head())



print("\nFirst 5 transactions:")
print(df.head())


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nTransaction class distribution:")
print(df["Class"].value_counts())

print("\nMissing values in each column:")
print(df.isnull().sum())


# ============================================================
# 3. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\nTransaction amount statistics:")

print("\nNormal transactions:")
print(df[df["Class"] == 0]["Amount"].describe())

print("\nFraudulent transactions:")
print(df[df["Class"] == 1]["Amount"].describe())


# ============================================================
# 4. VISUALIZE NORMAL VS FRAUD
# ============================================================

class_counts = df["Class"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal", "Fraud"],
    class_counts.values
)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# ============================================================
# 5. TRANSACTION PERCENTAGES
# ============================================================

class_percentage = df["Class"].value_counts(normalize=True) * 100

print("\nTransaction percentages:")
print(class_percentage)


# ============================================================
# 6. COMPARE FEATURES
# ============================================================

normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

features = [f"V{i}" for i in range(1, 29)]

comparison = pd.DataFrame({
    "Normal Mean": normal[features].mean(),
    "Fraud Mean": fraud[features].mean()
})

comparison["Difference"] = abs(
    comparison["Fraud Mean"] -
    comparison["Normal Mean"]
)

comparison = comparison.sort_values(
    "Difference",
    ascending=False
)

print("\nFeatures with the largest difference between normal and fraud:")
print(comparison.head(10))


# ============================================================
# 7. VISUALIZE TOP FEATURES
# ============================================================

top_features = comparison.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features.index,
    top_features["Difference"]
)

plt.title("Top 10 Features with Largest Difference")
plt.xlabel("Absolute Difference in Mean")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeature data shape (X):")
print(X.shape)

print("\nTarget data shape (y):")
print(y.shape)

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)


# ============================================================
# 10. TRAINING / VALIDATION SPLIT
# ============================================================

X_train_model, X_val, y_train_model, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nModel training data shape:")
print(X_train_model.shape)

print("\nValidation data shape:")
print(X_val.shape)

print("\nModel training target shape:")
print(y_train_model.shape)

print("\nValidation target shape:")
print(y_val.shape)


# ============================================================
# 11. SMOTE
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_model,
    y_train_model
)

print("\nClass distribution before SMOTE:")
print(y_train_model.value_counts())

print("\nClass distribution after SMOTE:")
print(y_train_smote.value_counts())

# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

lr_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")


# ============================================================
# 12. LOGISTIC REGRESSION TEST PREDICTIONS
# ============================================================

lr_test_pred = lr_model.predict(X_test)

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_test_pred))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_test_pred))


# ============================================================
# 13. LOGISTIC REGRESSION PR-AUC
# ============================================================

lr_test_probabilities = lr_model.predict_proba(X_test)[:, 1]

lr_pr_auc = average_precision_score(
    y_test,
    lr_test_probabilities
)

print("\nLogistic Regression PR-AUC:")
print(f"{lr_pr_auc:.4f}")


# ============================================================
# 14. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train ONLY on model-training data
rf_model.fit(
    X_train_model,
    y_train_model
)

print("\nRandom Forest trained using model training data!")

# ========================================================
# SMOTE RANDOM FOREST
# ========================================================

smote_rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train using the SMOTE-balanced training data
smote_rf_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nSMOTE Random Forest trained successfully!")


# ============================================================
# 15. RANDOM FOREST VALIDATION PREDICTIONS
# ============================================================

rf_val_probabilities = rf_model.predict_proba(X_val)[:, 1]

rf_val_pred = (rf_val_probabilities >= 0.5).astype(int)

print("\nRandom Forest Validation Classification Report:")
print(
    classification_report(
        y_val,
        rf_val_pred
    )
)

print("\nRandom Forest Validation Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        rf_val_pred
    )
)
print(
    "\nRandom Forest Validation PR-AUC:",
    average_precision_score(
        y_val,
        rf_val_probabilities
    )
)
# ========================================================
# SMOTE RANDOM FOREST VALIDATION
# ========================================================

smote_rf_val_probabilities = smote_rf_model.predict_proba(X_val)[:, 1]

smote_rf_val_pred = (
    smote_rf_val_probabilities >= 0.5
).astype(int)

print("\nSMOTE Random Forest Validation Classification Report:")

print(
    classification_report(
        y_val,
        smote_rf_val_pred
    )
)

print("\nSMOTE Random Forest Validation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        smote_rf_val_pred
    )
)

print(
    "\nSMOTE Random Forest Validation PR-AUC:",
    average_precision_score(
        y_val,
        smote_rf_val_probabilities
    )
)
# ========================================================
# XGBOOST MODEL
# ========================================================

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

# Train XGBoost on the original model-training data
xgb_model.fit(
    X_train_model,
    y_train_model
)

print("\nXGBoost trained successfully!")

# ========================================================
# SAVE FINAL MODEL
# ========================================================

# ========================================================
# XGBOOST VALIDATION
# ========================================================

xgb_val_probabilities = xgb_model.predict_proba(X_val)[:, 1]

xgb_val_pred = (
    xgb_val_probabilities >= 0.5
).astype(int)

print("\nXGBoost Validation Classification Report:")

print(
    classification_report(
        y_val,
        xgb_val_pred
    )
)

print("\nXGBoost Validation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        xgb_val_pred
    )
)

print(
    "\nXGBoost Validation PR-AUC:",
    average_precision_score(
        y_val,
        xgb_val_probabilities
    )
)# ========================================================
# XGBOOST VALIDATION THRESHOLD ANALYSIS
# ========================================================

print("\n============================================")
print("XGBOOST VALIDATION THRESHOLD ANALYSIS")
print("============================================")

xgb_best_threshold = 0.5
xgb_best_f1 = 0

for threshold in np.arange(0.1, 1.0, 0.1):

    xgb_threshold_pred = (
        xgb_val_probabilities >= threshold
    ).astype(int)

    xgb_precision = precision_score(
        y_val,
        xgb_threshold_pred,
        zero_division=0
    )

    xgb_recall = recall_score(
        y_val,
        xgb_threshold_pred,
        zero_division=0
    )

    xgb_f1 = f1_score(
        y_val,
        xgb_threshold_pred,
        zero_division=0
    )

    print(f"\nThreshold: {threshold:.1f}")
    print(f"Precision: {xgb_precision:.4f}")
    print(f"Recall:    {xgb_recall:.4f}")
    print(f"F1-score:  {xgb_f1:.4f}")

    if xgb_f1 > xgb_best_f1:
        xgb_best_f1 = xgb_f1
        xgb_best_threshold = threshold

print("\n============================================")
print("XGBOOST SELECTED THRESHOLD")
print("============================================")

print(
    f"Selected threshold: {xgb_best_threshold:.1f}"
)

print(
    f"Validation F1-score: {xgb_best_f1:.4f}"
)
# ========================================================
# TRAINING FINAL XGBOOST
# ========================================================

print("\n============================================")
print("TRAINING FINAL XGBOOST")
print("============================================")

xgb_final_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

# Train on the complete training dataset
xgb_final_model.fit(
    X_train,
    y_train
)

print("\nFinal XGBoost trained successfully!")
# ========================================================
# SAVE FINAL MODEL
# ========================================================

final_model_data = {
    "model": xgb_final_model,
    "threshold": xgb_best_threshold
}

joblib.dump(
    final_model_data,
    "models/fraud_model.pkl"
)

print("\nFinal XGBoost model saved successfully!")
print("Saved to: models/fraud_model.pkl")


# ========================================================
# FINAL XGBOOST TEST RESULTS
# ========================================================

xgb_test_probabilities = (
    xgb_final_model.predict_proba(X_test)[:, 1]
)

xgb_test_pred = (
    xgb_test_probabilities >= xgb_best_threshold
).astype(int)

print("\n============================================")
print("FINAL XGBOOST TEST RESULTS")
print("============================================")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        xgb_test_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_test_pred
    )
)

print(
    "\nXGBoost PR-AUC:",
    average_precision_score(
        y_test,
        xgb_test_probabilities
    )
)
# ============================================================
# 16. THRESHOLD ANALYSIS USING VALIDATION DATA
# ============================================================

print("\n============================================")
print("RANDOM FOREST VALIDATION THRESHOLD ANALYSIS")
print("============================================")

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    threshold_pred = (
        rf_val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Automatically remember the threshold
    # with the highest F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\n============================================")
print("SELECTED THRESHOLD")
print("============================================")

print(f"Selected threshold: {best_threshold}")
print(f"Validation F1-score: {best_f1:.4f}")


# ============================================================
# 17. RETRAIN FINAL RANDOM FOREST
# ============================================================

print("\n============================================")
print("TRAINING FINAL RANDOM FOREST")
print("============================================")

final_rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Now we can use ALL original training data.
# The validation set has already helped us choose
# the threshold.
final_rf_model.fit(
    X_train,
    y_train
)

print("\nFinal Random Forest trained successfully!")


# ============================================================
# 18. FINAL TEST PREDICTIONS
# ============================================================

final_rf_probabilities = (
    final_rf_model.predict_proba(X_test)[:, 1]
)

# Apply the threshold selected using validation data
final_rf_pred = (
    final_rf_probabilities >= best_threshold
).astype(int)


# ============================================================
# 19. FINAL RANDOM FOREST EVALUATION
# ============================================================

print("\n============================================")
print("FINAL RANDOM FOREST TEST RESULTS")
print("============================================")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        final_rf_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_rf_pred
    )
)


# ============================================================
# 20. FINAL RANDOM FOREST PR-AUC
# ============================================================

rf_pr_auc = average_precision_score(
    y_test,
    final_rf_probabilities
)

print("\nRandom Forest PR-AUC:")
print(f"{rf_pr_auc:.4f}")


# ============================================================
# 21. MODEL COMPARISON
# ============================================================

print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

print(
    f"Logistic Regression PR-AUC: "
    f"{lr_pr_auc:.4f}"
)

print(
    f"Random Forest PR-AUC:       "
    f"{rf_pr_auc:.4f}"
)


# ============================================================
# 22. FINAL RISK SCORE
# ============================================================

risk_scores = final_rf_probabilities * 100


# ============================================================
# 23. RISK LEVEL FUNCTION
# ============================================================

def get_risk_level(score):

    if score < 30:
        return "Low Risk"

    elif score < 70:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# 24. CREATE FINAL RESULTS TABLE
# ============================================================

results = pd.DataFrame({
    "Actual Class": y_test.values,
    "Predicted Class": final_rf_pred,
    "Fraud Probability": final_rf_probabilities,
    "Risk Score": risk_scores
})

results["Risk Level"] = (
    results["Risk Score"]
    .apply(get_risk_level)
)


# ============================================================
# 25. SHOW FIRST 20 RESULTS
# ============================================================

print("\n============================================")
print("FINAL PREDICTION RESULTS")
print("============================================")

print(
    results.head(20)
)


# ============================================================
# 26. RISK LEVEL DISTRIBUTION
# ============================================================

risk_distribution = (
    results["Risk Level"]
    .value_counts()
)

print("\nRisk Level Distribution:")
print(risk_distribution)


# ============================================================
# 27. FRAUD DETECTION RATE
# ============================================================

fraud_actual = (
    y_test == 1
).sum()

fraud_detected = (
    (y_test == 1) &
    (final_rf_pred == 1)
).sum()

fraud_detection_rate = (
    fraud_detected /
    fraud_actual
) * 100

print("\nFinal Fraud Detection Rate:")
print(
    f"{fraud_detection_rate:.2f}%"
)


# ============================================================
# 28. FINAL PROJECT SUMMARY
# ============================================================

print("\n============================================")
print("PROJECT SUMMARY")
print("============================================")

print(
    f"Dataset size: {df.shape[0]:,} transactions"
)

print(
    f"Fraud transactions: "
    f"{(y == 1).sum():,}"
)

print(
    f"Normal transactions: "
    f"{(y == 0).sum():,}"
)

print(
    f"Selected threshold: "
    f"{best_threshold}"
)

print(
    f"Random Forest PR-AUC: "
    f"{rf_pr_auc:.4f}"
)

print(
    f"Fraud detection rate: "
    f"{fraud_detection_rate:.2f}%"
)