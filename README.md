# Financial Fraud Detection & Risk Prediction

An end-to-end machine learning system for detecting potentially fraudulent financial transactions and assigning a risk score.

This project covers exploratory data analysis, feature engineering, class-imbalance handling, model comparison, threshold optimization, final model evaluation, and interactive prediction through a Streamlit dashboard.

---

## Project Overview

Financial fraud detection is a highly imbalanced binary classification problem because fraudulent transactions are much rarer than normal transactions.

This project:

- Performs exploratory data analysis
- Creates transaction-based features
- Handles severe class imbalance
- Trains Logistic Regression, Random Forest, and XGBoost models
- Compares models using fraud-focused evaluation metrics
- Optimizes the fraud classification threshold
- Evaluates the final model on an untouched test set
- Generates fraud probability and risk scores
- Provides an interactive Streamlit dashboard

---

## Dataset

The project uses the Credit Card Fraud Detection dataset.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total transactions | 284,807 |
| Normal transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud percentage | 0.1727% |

The dataset contains anonymized transaction features `V1` through `V28`, together with `Time`, `Amount`, and `Class`.

- `Class = 0` → Normal transaction
- `Class = 1` → Fraudulent transaction

The raw dataset is not included in the GitHub repository.

Place the dataset at:

`data/raw/creditcard.csv`

---

## Feature Engineering

### Amount_Log

A logarithmic transformation of the transaction amount using `log1p`.

### Hour

The `Time` feature is converted into an approximate transaction hour.

### Hour_Sin and Hour_Cos

Cyclical transformations are used to represent the 24-hour nature of transaction time.

---

## Machine Learning Models

### Logistic Regression

Used as the baseline classification model.

**PR-AUC: 0.7190**

### Random Forest

A tree-based ensemble model using class weighting to address the severe class imbalance.

**Final Test PR-AUC: 0.8629**

### XGBoost

A gradient boosting model used as the final prediction model.

**Final Test PR-AUC: 0.8771**

---

## Class Imbalance

Fraudulent transactions represent only about **0.17%** of all transactions.

Because of this severe imbalance, accuracy alone is not sufficient for evaluating fraud detection performance.

The project therefore focuses on:

- Precision
- Recall
- F1-score
- PR-AUC

SMOTE was also evaluated as an imbalance-handling experiment.

---

## Threshold Optimization

Instead of relying only on the default probability threshold of `0.5`, multiple thresholds were evaluated using validation data.

For the final XGBoost model:

- Selected threshold: **0.2**
- Validation F1-score: **0.8456**

The selected threshold was then used for the final test-set evaluation.

---

## Final XGBoost Results

The final XGBoost model was evaluated on the untouched test set.

| Metric | Result |
|---|---:|
| Precision | 88% |
| Recall | 85% |
| F1-score | 0.86 |
| PR-AUC | 0.8771 |

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 56,853 | 11 |
| Actual Fraud | 15 | 83 |

The model correctly detected **83 fraudulent transactions** and missed **15 fraudulent transactions** in the final test set.

---

## Risk Scoring

The application converts the model's fraud probability into a risk score from **0 to 100**.

| Risk Score | Risk Level |
|---:|---|
| 0–29.99 | Low Risk |
| 30–69.99 | Medium Risk |
| 70–100 | High Risk |

The fraud classification threshold used by the final model is **0.2**.

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for transaction-level fraud analysis.

The dashboard provides:

- Transaction selection
- Transaction amount
- Approximate transaction hour
- Fraud probability
- Risk score
- Risk level
- Fraud/normal prediction
- Transaction feature inspection
- Dataset statistics
- Fraud analytics
- Model performance
- Risk level guide

To run the dashboard:

`streamlit run app/app.py`

---

## Dashboard Screenshots

### Main Dashboard

![Financial Fraud Detection Dashboard](images/dashboard-main.png)

### Fraud Analytics

![Fraud Analytics Dashboard](images/dashboard-analytics.png)

### Fraud Detection Example

![Fraud Detection Example](images/fraud-example.png)

### Additional Dashboard View

![Additional Dashboard View](images/fraud-example%20extn.png)

---

## Project Structure

Financial-Fraud-Detection/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── images/
│   ├── dashboard-main.png
│   ├── dashboard-analytics.png
│   ├── fraud-example.png
│   └── fraud-example extn.png
├── models/
│   └── fraud_model.pkl
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── explore_data.py
│   └── predict.py
├── .gitignore
├── README.md
├── requirements.txt
└── test.py

---

## How to Run

### 1. Clone the Repository

`git clone https://github.com/05804/Financial-Fraud-Detection.git`

### 2. Open the Project

`cd Financial-Fraud-Detection`

### 3. Create a Virtual Environment

`python -m venv venv`

### 4. Activate the Virtual Environment

Windows PowerShell:

`venv\Scripts\activate`

### 5. Install Dependencies

`pip install -r requirements.txt`

### 6. Add the Dataset

Place the dataset at:

`data/raw/creditcard.csv`

### 7. Run the Prediction Pipeline

`python src/predict.py`

### 8. Run the Streamlit Dashboard

`streamlit run app/app.py`

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Key Concepts Demonstrated

- Binary classification
- Exploratory Data Analysis
- Feature engineering
- Highly imbalanced datasets
- SMOTE
- Class weighting
- Logistic Regression
- Random Forest
- XGBoost
- Precision and Recall
- F1-score
- PR-AUC
- Decision threshold optimization
- Model persistence
- Prediction pipelines
- Streamlit application development

---

## Author
**Shaik Junaid**

B.Tech - Computer Science & Engineering (Data Science)



