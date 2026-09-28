# Financial Fraud Detection & Risk Prediction

> End-to-end fraud detection system combining machine learning, REST APIs, Docker, AWS infrastructure, monitoring, and automated CI/CD.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-189FDD)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)

**Live API:** https://skfrauddetection.duckdns.org  
**Swagger API Docs:** https://skfrauddetection.duckdns.org/docs  
**Repository:** https://github.com/05804/Financial-Fraud-Detection

---

## Overview

Financial fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a very small fraction of the total dataset.

This project detects potentially fraudulent credit-card transactions and converts the model's fraud probability into an interpretable **0-100 risk score**.

The project evolved from an exploratory machine learning pipeline into a production-style application with:

- Machine learning model development and evaluation
- Feature engineering and threshold optimization
- Interactive Streamlit dashboard
- FastAPI inference API
- Docker containerization
- Private model storage in Amazon S3
- Amazon ECR container registry
- AWS EC2 deployment
- AWS Systems Manager deployment
- Amazon CloudWatch logging
- Nginx reverse proxy and HTTPS
- GitHub Actions CI/CD
- AWS OIDC authentication
- Trivy container security scanning
- Automated health checks
- Deployment rollback
- Automatic Docker image cleanup

---

## Key Results

### Model Comparison

| Model | PR-AUC |
|---|---:|
| Logistic Regression | 0.7190 |
| Random Forest | 0.8629 |
| **XGBoost** | **0.8771** |

### Final XGBoost Performance

| Metric | Result |
|---|---:|
| Precision | 88% |
| Recall | 85% |
| F1-score | 0.86 |
| PR-AUC | 0.8771 |

The final XGBoost model was evaluated on an untouched test set using a classification threshold of **0.2**.

**Validation F1-score:** 0.8456

### Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| **Actual Normal** | 56,853 | 11 |
| **Actual Fraud** | 15 | 83 |

---

## System Architecture

```text
                         GitHub Repository
                                |
                                v
                         GitHub Actions
                           CI / CD Pipeline
                                |
              +-----------------+-----------------+
              |                 |                 |
              v                 v                 v
           Pytest             Trivy         Docker Build
                                                |
                                                v
                                           Amazon ECR
                                                |
                                                v
                                      AWS Systems Manager
                                                |
                                                v
                                           Amazon EC2
                                                |
                                  +-------------+-------------+
                                  |                           |
                                  v                           v
                               Nginx                     FastAPI
                                  |                           |
                                HTTPS                         |
                                                              v
                                                         Amazon S3
                                                       Model Storage
                                                              |
                                                              v
                                                        CloudWatch
                                                           Logs
Production Flow
Code is pushed to GitHub.
GitHub Actions runs tests and security scans.
Docker images are built and pushed to Amazon ECR.
AWS Systems Manager deploys the new image to EC2.
The new container is health-checked.
If healthy, the deployment completes and old Docker images are cleaned up.
If unhealthy, the previous image is automatically restored.
Nginx provides HTTPS access to the FastAPI service.
FastAPI loads the trained model from private Amazon S3 storage.
Application and Nginx logs are collected in CloudWatch.
Dataset

The project uses the Credit Card Fraud Detection dataset.

Metric	Value
Total transactions	284,807
Normal transactions	284,315
Fraudulent transactions	492
Fraud percentage	0.1727%

The dataset contains anonymized features V1-V28, together with Time, Amount, and Class.

Class = 0 -> Normal transaction
Class = 1 -> Fraudulent transaction

The dataset is used locally for model development and is not required by the production API.

Expected local location:

data/raw/creditcard.csv

Feature Engineering

The prediction pipeline creates additional features from the original transaction data:

Feature	Description
Amount_Log	Log-transformed transaction amount
Hour	Approximate transaction hour derived from Time
Hour_Sin	Cyclical representation of transaction hour
Hour_Cos	Cyclical representation of transaction hour

The same feature engineering logic is applied during API inference to maintain consistency between model training and production predictions.

Model Development

Three classification models were evaluated:

Model	PR-AUC
Logistic Regression	0.7190
Random Forest	0.8629
XGBoost	0.8771

XGBoost achieved the highest PR-AUC among the evaluated models and was selected as the final model.

Class Imbalance

Fraudulent transactions represent only 0.1727% of the dataset.

Because of this severe class imbalance, model evaluation focuses on:

Precision
Recall
F1-score
PR-AUC

Class weighting and SMOTE were also evaluated as approaches for handling class imbalance.

Threshold Optimization

The default classification threshold of 0.5 was not assumed to be optimal.

Multiple probability thresholds were evaluated using validation data.

The selected threshold for the final XGBoost model is:

0.2

The selected threshold achieved a validation F1-score of:

0.8456

The optimized threshold is stored with the trained model and used during API inference.

Final Model Performance

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

The application converts the model's fraud probability into an interpretable score between 0 and 100.

Risk Score = Fraud Probability x 100

Risk Score	Risk Level
0-29.99	Low Risk
30-69.99	Medium Risk
70-100	High Risk

The risk level is derived from the probability score, while the fraud/normal classification uses the optimized probability threshold of 0.2.

Streamlit Dashboard

The project includes an interactive Streamlit dashboard for transaction-level fraud analysis.

Dashboard Features
Transaction selection
Transaction amount and transaction time
Fraud probability
Risk score
Risk level
Fraud/normal prediction
Transaction feature inspection
Dataset statistics
Fraud analytics
Model performance
Risk-level guide
Main Dashboard

Fraud Analytics

Fraud Detection Example

Additional Dashboard View

Run Locally
streamlit run app/app.py
FastAPI REST API

The project exposes the trained fraud detection model through a production-ready FastAPI service.

API Endpoints
Method	Endpoint	Description
GET	/	API status
GET	/health	Health check
POST	/predict	Predict fraud risk for a transaction
Prediction Response

The /predict endpoint returns:

{
  "fraud_probability": 0.9904,
  "risk_score": 99.04,
  "risk_level": "High Risk",
  "prediction": "FRAUD"
}
Interactive API Documentation

FastAPI automatically provides Swagger documentation at:

https://skfrauddetection.duckdns.org/docs

Example Request
{
  "Time": 406,
  "Amount": 0.0,
  "V1": -2.31,
  "V2": 1.95,
  "V3": -1.61,
  "V4": 3.99,
  "V5": -0.52,
  "V6": -1.43,
  "V7": -2.54,
  "V8": 0.39,
  "V9": -1.35,
  "V10": -2.77,
  "V11": 3.20,
  "V12": -2.90,
  "V13": -0.60,
  "V14": -4.29,
  "V15": 0.39,
  "V16": -1.14,
  "V17": -2.83,
  "V18": -0.02,
  "V19": 0.42,
  "V20": 0.13,
  "V21": 0.52,
  "V22": -0.04,
  "V23": -0.46,
  "V24": 0.32,
  "V25": 0.04,
  "V26": 0.18,
  "V27": 0.21,
  "V28": -0.07
}

The API validates incoming requests using Pydantic and returns HTTP 422 responses when required fields are missing or contain invalid data.

Docker & Containerization

The application is containerized using Docker to provide a consistent runtime environment across development and production.

Containerized Services
Service	Purpose
Streamlit	Interactive fraud detection dashboard
FastAPI	Production inference API
FastAPI Container

The FastAPI service is packaged into a Docker image and deployed to Amazon ECR.

The production container:

Runs the FastAPI application on port 8000
Uses a production runtime environment
Loads the trained model from private Amazon S3 storage
Sends application logs to Amazon CloudWatch
Is deployed and managed through AWS Systems Manager
Build and Run Locally

Build the FastAPI image:

docker build -t financial-fraud-api ./api

Run the container:

docker run -p 8000:8000 financial-fraud-api

The API can then be accessed locally at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs

AWS Infrastructure & Deployment

The production application is deployed on AWS using a combination of managed services and infrastructure components.

AWS Services Used
AWS Service	Purpose
Amazon EC2	Hosts the production containers
Amazon ECR	Stores Docker images
Amazon S3	Stores the private trained model
AWS Systems Manager	Performs remote deployments
Amazon CloudWatch	Collects application and Nginx logs
AWS IAM	Controls access and GitHub OIDC permissions
Production Deployment Flow
GitHub
   |
   v
GitHub Actions
   |
   +-- Run Tests
   +-- Trivy Security Scan
   +-- Build Docker Image
   +-- Push Image
          |
          v
      Amazon ECR
          |
          v
AWS Systems Manager
          |
          v
      Amazon EC2
          |
          v
       Nginx
          |
          v
       FastAPI
          |
          v
     Private S3 Model
Private Model Storage

The trained model is stored privately in Amazon S3.

models/fraud_model.pkl

The production API downloads the model from S3 when a prediction is requested and caches it locally for subsequent predictions.

HTTPS

Nginx acts as a reverse proxy in front of FastAPI and provides the public HTTPS endpoint:

https://skfrauddetection.duckdns.org

CI/CD Pipeline

The project uses GitHub Actions to automate testing, security scanning, Docker image publishing, deployment, and post-deployment verification.

Pipeline Stages
Code Push
    |
    v
GitHub Actions
    |
    +-- Pytest
    |
    +-- Trivy Security Scan
    |
    +-- Docker Build
    |
    +-- AWS OIDC Authentication
    |
    +-- Push Image -> Amazon ECR
    |
    +-- Deploy -> AWS Systems Manager
    |
    +-- Health Check
    |
    +-- Rollback if Deployment Fails
Automated Deployment

Every push to the main branch triggers the CI/CD workflow.

The pipeline:

Installs the Python dependencies.
Runs the automated API test suite.
Builds the Docker image.
Scans the container image using Trivy.
Authenticates with AWS using GitHub OIDC.
Pushes the Docker image to Amazon ECR.
Deploys the image to EC2 using AWS Systems Manager.
Performs an application health check.
Automatically rolls back to the previous image if the deployment fails.
Removes unused Docker images after a successful deployment.
Deployment Safety

The deployment process includes:

Automated health checks
Previous-image tracking
Automatic rollback
CloudWatch logging
Docker image cleanup
External HTTPS verification

This provides a repeatable deployment process without requiring manual SSH access to the production server.

Monitoring, Logging & Reliability

The production deployment includes centralized logging and automated health monitoring.

CloudWatch Logging

Application and infrastructure logs are collected using Amazon CloudWatch.

Log Source	CloudWatch Log Group
FastAPI	/financial-fraud/api
Nginx Access	/financial-fraud/nginx/access
Nginx Error	/financial-fraud/nginx/error

These logs help monitor API requests, application behavior, HTTP errors, and reverse-proxy activity.

Health Checks

The API exposes a dedicated health endpoint:

GET /health

A successful response indicates that the FastAPI service is running correctly.

The CI/CD pipeline also performs an external HTTPS health check after deployment.

Automatic Rollback

If the newly deployed container fails its health check:

The failed container is stopped and removed.
The previous Docker image is restored.
The previous container is started again.
The rollback deployment is health-checked.
The deployment fails safely if the rollback is also unhealthy.
Docker Image Cleanup

After a successful deployment, unused Docker images are automatically removed from the EC2 instance.

This prevents unnecessary Docker images from consuming the server's limited disk space over time.

Security

Security was considered across the application, container, AWS infrastructure, and CI/CD pipeline.

Security Measures
AWS IAM permissions are used to control access to AWS resources.
GitHub Actions authenticates with AWS using OIDC instead of storing long-lived AWS access keys.
The trained model is stored in a private Amazon S3 bucket.
Docker images are scanned using Trivy during CI/CD.
The production API runs inside a Docker container.
Nginx provides HTTPS access to the public API.
Sensitive files such as credentials and private keys are excluded from Git tracking.
API request validation is handled using Pydantic.
Deployment health checks help prevent unhealthy containers from remaining in production.
GitHub OIDC

GitHub Actions uses an AWS IAM role through OpenID Connect (OIDC):

GitHub Actions
      |
      v
GitHub OIDC
      |
      v
AWS IAM Role
      |
      v
AWS Resources

This avoids storing permanent AWS access keys as GitHub repository secrets.

Testing

The project includes automated tests for the FastAPI service using Pytest and FastAPI's TestClient.

Test Coverage

The API test suite covers:

Root endpoint availability
Health endpoint availability
Valid transaction prediction requests
Missing required fields
Invalid transaction amount

Run the tests locally with:

pytest

The current test suite contains 5 automated tests.

Example result:

5 passed

The same test suite is executed automatically as part of the GitHub Actions CI/CD pipeline before deployment.

Project Structure
Financial-Fraud-Detection/
|
+-- api/
|   +-- Dockerfile
|   +-- main.py
|   +-- predictor.py
|   +-- requirements.txt
|   +-- __init__.py
|
+-- app/
|   +-- app.py
|
+-- data/
|   +-- raw/
|       +-- creditcard.csv
|
+-- images/
|   +-- dashboard-analytics.png
|   +-- dashboard-main.png
|   +-- fraud-example.png
|   +-- fraud-example extn.png
|
+-- models/
|   +-- fraud_model.pkl
|
+-- notebooks/
|   +-- fraud_detection.ipynb
|
+-- src/
|   +-- explore_data.py
|   +-- predict.py
|
+-- tests/
|   +-- test_api.py
|
+-- .gitignore
+-- README.md
+-- requirements.txt
Directory Overview
Directory	Purpose
api/	FastAPI application and Docker configuration
app/	Streamlit dashboard
data/	Local dataset used for model development
images/	README and dashboard screenshots
models/	Trained machine learning model
notebooks/	Exploratory model development
src/	Data exploration and prediction scripts
tests/	Automated API tests
Local Setup & Usage
1. Clone the Repository
git clone https://github.com/05804/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
2. Create a Virtual Environment

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux / macOS:

python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Streamlit Dashboard
streamlit run app/app.py
5. Run the FastAPI Service

Install the API dependencies:

pip install -r api/requirements.txt

Start the API:

uvicorn api.main:app --reload

The API will be available at:

http://localhost:8000

Swagger documentation:

http://localhost:8000/docs

6. Run Tests
pytest
Production API

The deployed production API is available at:

https://skfrauddetection.duckdns.org

Interactive Swagger documentation:

https://skfrauddetection.duckdns.org/docs

Live Deployment Verification

The production API was tested after deployment to verify both normal and fraudulent transaction scenarios.

Health Check

GET /health

Response:

{
  "status": "healthy"
}
Normal Transaction

A normal transaction returned:

Fraud Probability: 0.0
Risk Score: 0.0
Risk Level: Low Risk
Prediction: NORMAL
Fraudulent Transaction

A fraud-like transaction returned:

Fraud Probability: 0.9904
Risk Score: 99.04
Risk Level: High Risk
Prediction: FRAUD
Validation Testing

The production API also correctly rejected invalid requests:

Test	HTTP Status
Missing required fields	422
Invalid Amount value	422
Production Endpoint

API: https://skfrauddetection.duckdns.org

Swagger: https://skfrauddetection.duckdns.org/docs

Technologies Used
Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Application Development
Streamlit
FastAPI
Pydantic
Uvicorn
DevOps & Containerization
Docker
Git
GitHub Actions
Trivy
AWS
Amazon EC2
Amazon ECR
Amazon S3
AWS Systems Manager
AWS IAM
Amazon CloudWatch
Infrastructure & Networking
Nginx
HTTPS
DuckDNS
Future Improvements

Potential improvements for the project include:

Real-time transaction streaming
Model retraining automation
Automated model monitoring and drift detection
Feature store integration
Batch prediction pipelines
Expanded fraud investigation workflows
Authentication and authorization for the API
Rate limiting and API request monitoring
Infrastructure as Code using Terraform
Kubernetes-based deployment
Automated model performance monitoring
Additional machine learning models and ensemble approaches
Author

Shaik Junaid

B.Tech - Computer Science & Engineering (Data Science)

Interested in:

Machine Learning
Data Science
MLOps
AWS Cloud
DevOps
Generative AI
Project

Financial Fraud Detection & Risk Prediction

An end-to-end machine learning and cloud deployment project demonstrating the complete workflow from model development to production deployment.

License

This project is intended for educational, portfolio, and demonstration purposes.

The dataset used in this project is subject to its original dataset license and terms of use.

Links
Live API: https://skfrauddetection.duckdns.org
Swagger API Documentation: https://skfrauddetection.duckdns.org/docs
GitHub Repository: https://github.com/05804/Financial-Fraud-Detection

