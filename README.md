## Overview

This project detects potentially fraudulent credit-card transactions
and converts model probability into an interpretable 0–100 risk score.

The system evolved from an ML experimentation project into a
production-style application with:

- XGBoost fraud detection
- Streamlit dashboard
- FastAPI inference API
- Docker containers
- Amazon S3 model storage
- Amazon ECR
- EC2 deployment
- GitHub Actions CI/CD
- AWS OIDC authentication
- CloudWatch logging
- Nginx + HTTPS
- Automated rollback
- Docker image cleanup


Then a compact Results section rather than spreading metrics throughout the README:

Model	PR-AUC
Logistic Regression	0.7190
Random Forest	0.8629
XGBoost	0.8771

And:

Final XGBoost	Result
Precision	88%
Recall	85%
F1	0.86
PR-AUC	0.8771

That would look much cleaner and more professional.

Also, I would keep the AWS/DevOps material, because that's actually one of the strongest parts of your project—but present it as an architecture + deployment section, rather than dozens of separate headings.

If you want, I'll
rewrite the entire existing README into this polished GitHub-style format, while keeping all your real project details and metrics, and then we'll replace the current README and commit the improved version.

Rewrite the entire existing README into this polished GitHub-style format

Absolutely. Let's make it look like a serious GitHub portfolio project rather than a report.

Below is the complete replacement for your existing README.md. It keeps your actual ML results and the AWS/DevOps work, but uses a much cleaner hierarchy, fewer headings, compact tables, badges, and better spacing.

Open the existing README:

notepad README.md

Delete everything and paste this entire version:

# Financial Fraud Detection & Risk Prediction

End-to-end fraud detection system combining machine learning, REST APIs, containerization, AWS infrastructure, monitoring, and automated CI/CD.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-189FDD)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)

**Live API:** https://skfrauddetection.duckdns.org
**Repository:** https://github.com/05804/Financial-Fraud-Detection

---

## Overview

Financial fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small fraction of total transactions.

This project detects potentially fraudulent transactions and converts the model's fraud probability into an interpretable **0–100 risk score**.

The project evolved from an exploratory machine learning pipeline into a production-style application with:

- Machine learning model development and evaluation
- Feature engineering and threshold optimization
- Interactive Streamlit dashboard
- FastAPI inference service
- Docker containerization
- Private model storage in Amazon S3
- Amazon ECR container registry
- AWS EC2 deployment
- AWS Systems Manager deployment
- CloudWatch logging
- Nginx reverse proxy and HTTPS
- GitHub Actions CI/CD
- AWS OIDC authentication
- Trivy container security scanning
- Automated health checks
- Deployment rollback
- Automatic Docker image cleanup

---

## System Architecture

```text
                         ┌──────────────────────┐
                         │      GitHub Repo     │
                         │ Financial Fraud App  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   GitHub Actions     │
                         │      CI / CD         │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
           Pytest                 Trivy             Docker Build
                                                        │
                                                        ▼
                                                 ┌──────────────┐
                                                 │ Amazon ECR   │
                                                 └──────┬───────┘
                                                        │
                                                        ▼
                                                 ┌──────────────┐
                                                 │ AWS Systems  │
                                                 │   Manager    │
                                                 └──────┬───────┘
                                                        │
                                                        ▼
                  ┌──────────────────────────────────────────────────┐
                  │                     AWS EC2                      │
                  │                                                  │
                  │                  Nginx + HTTPS                   │
                  │                        │                         │
                  │                        ▼                         │
                  │                  FastAPI Docker                  │
                  │                        │                         │
                  └────────────────────────┼─────────────────────────┘
                                           │
                           ┌───────────────┼───────────────┐
                           │               │               │
                           ▼               ▼               ▼
                       Amazon S3      CloudWatch        Public API
                    Model Storage       Logs             HTTPS
Machine Learning
Dataset

The project uses the Credit Card Fraud Detection dataset.

Metric	Value
Total transactions	284,807
Normal transactions	284,315
Fraudulent transactions	492
Fraud percentage	0.1727%

The dataset contains anonymized features V1 through V28, together with Time, Amount, and Class.

Class = 0 → Normal transaction
Class = 1 → Fraudulent transaction

The raw dataset is not intended to be committed to the public repository.

Expected location:

data/raw/creditcard.csv
Feature Engineering

The prediction pipeline generates the following engineered features:

Feature	Description
Amount_Log	Log-transformed transaction amount
Hour	Approximate transaction hour derived from Time
Hour_Sin	Cyclical representation of hour
Hour_Cos	Cyclical representation of hour

The same feature engineering logic is applied during API inference to ensure consistency between model training and production predictions.

Models

Three classification approaches were evaluated:

Model	PR-AUC
Logistic Regression	0.7190
Random Forest	0.8629
XGBoost	0.8771

XGBoost was selected as the final prediction model.

Imbalanced Classification

Fraudulent transactions represent only approximately 0.17% of the dataset.

The project therefore focuses on fraud-sensitive metrics rather than accuracy alone:

Precision
Recall
F1-score
PR-AUC

Class weighting and SMOTE were also evaluated as approaches for handling class imbalance.

Threshold Optimization

The default probability threshold of 0.5 was not assumed to be optimal.

Multiple thresholds were evaluated using validation data.

The selected threshold for the final XGBoost model was:

0.2

Validation F1-score:

0.8456

The optimized threshold is stored with the trained model and used during API inference.

Model Performance

The final XGBoost model was evaluated on an untouched test set.

Metric	Result
Precision	88%
Recall	85%
F1-score	0.86
PR-AUC	0.8771
Confusion Matrix
	Predicted Normal	Predicted Fraud
Actual Normal	56,853	11
Actual Fraud	15	83

The model correctly detected 83 fraudulent transactions and missed 15 fraudulent transactions in the final test set.

Risk Scoring

The application converts fraud probability into a score between 0 and 100.

Risk Score = Fraud Probability × 100
Score	Risk Level
0–29.99	Low Risk
30–69.99	Medium Risk
70–100	High Risk

The risk level is derived from the probability score, while the fraud/normal classification uses the optimized threshold of 0.2.

Application
Streamlit Dashboard

The project includes an interactive dashboard for transaction-level fraud analysis.

Features include:

Transaction selection
Transaction amount
Approximate transaction hour
Fraud probability
Risk score
Risk level
Fraud/normal prediction
Transaction feature inspection
Dataset statistics
Fraud analytics
Model performance
Risk-level guide

Run locally:

streamlit run app/app.py
Dashboard Screenshots
Main Dashboard

Fraud Analytics

Fraud Detection Example

Additional Dashboard View

FastAPI

The machine learning model is exposed through a REST API built with FastAPI.

Endpoints
Method	Endpoint	Purpose
GET	/	API status
GET	/health	Health check
POST	/predict	Fraud prediction
Health Check
GET /health

Response:

{
  "status": "healthy"
}
Prediction Request
POST /predict

The request accepts:

Time
Amount
V1 through V28

The API performs the same feature engineering used during model development and returns:

{
  "fraud_probability": 0.1234,
  "risk_score": 12.34,
  "risk_level": "Low Risk",
  "prediction": "NORMAL"
}
Production Model Loading

The API retrieves the trained model from a private Amazon S3 location:

models/fraud_model.pkl

The model is downloaded only when prediction is first requested and is then cached locally at:

/tmp/fraud_model.pkl

The stored model contains both the trained model and the optimized classification threshold.

Cloud Deployment

The production API is deployed on AWS using a containerized architecture.

Service	Purpose
Amazon EC2	Hosts the production API
Amazon ECR	Stores versioned Docker images
Amazon S3	Private ML model storage
AWS Systems Manager	Remote deployment execution
Amazon CloudWatch	Application and Nginx logging
IAM	Access control
Nginx	Reverse proxy
HTTPS	Secure public API access

The production endpoint is:

https://skfrauddetection.duckdns.org

Health check:

https://skfrauddetection.duckdns.org/health
CI/CD

GitHub Actions automates testing, security scanning, image publishing, deployment, verification, and cleanup.

Git Push
   │
   ▼
Checkout
   │
   ▼
Run Tests
   │
   ▼
Build Docker Images
   │
   ▼
Trivy Security Scan
   │
   ▼
AWS OIDC Authentication
   │
   ▼
Amazon ECR Login
   │
   ▼
Push Versioned Image
   │
   ▼
AWS Systems Manager Deployment
   │
   ▼
Local Health Check
   │
   ├── Healthy ──► Cleanup Previous Images
   │
   └── Failed ───► Automatic Rollback
   │
   ▼
External HTTPS Health Check
AWS OIDC

GitHub Actions authenticates with AWS using OpenID Connect rather than storing long-lived AWS access keys in GitHub.

Security Scanning

Docker images are scanned using Trivy before deployment.

Deployment Safety

The deployment process records the currently running image before replacing it.

If the new container fails its health check:

The failed container is removed.
The previous image is restored.
The previous container is started.
A rollback health check is performed.

This provides automatic rollback protection during deployments.

Infrastructure Monitoring

Application and reverse-proxy logs are collected in Amazon CloudWatch.

Log Groups
/financial-fraud/api
/financial-fraud/nginx/access
/financial-fraud/nginx/error

The FastAPI container sends application logs directly to CloudWatch using the AWS Logs Docker driver.

Nginx access and error logs are collected through the CloudWatch Agent.

Docker Resource Management

The EC2 deployment automatically removes the previous Docker image after a successful deployment.

Unused Docker images and build artifacts are also cleaned up.

This prevents repeated deployments from continuously consuming EC2 storage.

During deployment verification, disk usage was reduced from approximately 98% to 41% after removing obsolete Docker images and unused artifacts.

Final Docker resource verification showed:

Images:       1
Containers:   1
Volumes:      0
Build Cache:  0
Testing

API tests are located in:

tests/test_api.py

Run the test suite:

pytest
Project Structure
Financial-Fraud-Detection/
│
├── api/
│   ├── Dockerfile
│   ├── main.py
│   ├── predictor.py
│   ├── requirements.txt
│   └── __init__.py
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── images/
│   ├── dashboard-main.png
│   ├── dashboard-analytics.png
│   ├── fraud-example.png
│   └── fraud-example extn.png
│
├── models/
│   └── fraud_model.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── explore_data.py
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── README.md
└── requirements.txt
Local Development
Clone
git clone https://github.com/05804/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
Create Environment
python -m venv .venv
Activate on Windows
.venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Add Dataset

Place the dataset at:

data/raw/creditcard.csv
Run Prediction Pipeline
python src/predict.py
Run Dashboard
streamlit run app/app.py
Run Tests
pytest
Technology Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
imbalanced-learn
Joblib
Application
Streamlit
FastAPI
Pydantic
Uvicorn
DevOps
Docker
GitHub Actions
Trivy
Nginx
AWS
Amazon EC2
Amazon ECR
Amazon S3
AWS Systems Manager
Amazon CloudWatch
IAM
AWS OIDC
Engineering Concepts

This project demonstrates practical experience with:

Imbalanced binary classification
Feature engineering
Model evaluation
Threshold optimization
Model persistence
REST API development
Docker containerization
Cloud deployment
CI/CD automation
AWS IAM and OIDC
Container security scanning
Application monitoring
Health checks
Deployment rollback
Infrastructure maintenance
Automated resource cleanup
Limitations
The dataset contains anonymized features, limiting real-world interpretability.
Model performance depends on the characteristics of the training dataset.
The fraud threshold represents a trade-off between precision and recall.
This project is a demonstration system and is not intended to replace a production banking fraud engine without additional validation, monitoring, security controls, and domain-specific requirements.
Project Evolution
ML Pipeline
    ↓
Feature Engineering
    ↓
Model Comparison
    ↓
Threshold Optimization
    ↓
Streamlit Dashboard
    ↓
FastAPI API
    ↓
Docker
    ↓
AWS Deployment
    ↓
CI/CD
    ↓
Cloud Monitoring
    ↓
Rollback Protection
    ↓
Automated Resource Cleanup

The final system demonstrates the progression from a machine learning experiment to a deployable, monitored, and automated application.

Author

Shaik Junaid

B.Tech — Computer Science & Engineering (Data Science)

⭐ Project

Financial Fraud Detection & Risk Prediction

Built as an end-to-end Data Science + Machine Learning + Cloud/DevOps project.
