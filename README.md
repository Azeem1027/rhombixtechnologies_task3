# 💳 Credit Card Approval Prediction System

An end-to-end Machine Learning web application that predicts credit card approvals based on applicant demographic and financial data. Built using scikit-learn's `Logistic Regression` with automated hyperparameter tuning via `GridSearchCV`, a Python `Flask` API backend, and an interactive, modern frontend dashboard.

---

## 📌 Project Overview

This application automates the credit approval evaluation process by leveraging historical data from the UCI Credit Screening dataset (`crx.data`). The system preprocesses applicant feature data, handles missing values, normalizes numerical attributes, encodes categorical data, and predicts the approval outcome alongside a calibrated confidence probability score.

### Key Features
* **Machine Learning Pipeline:** Data imputation, scaling (`MinMaxScaler`), categorical encoding (`OneHotEncoder`), and hyperparameter search (`GridSearchCV`).
* **Backend API:** Lightweight Flask REST API serving endpoints for predictions (`/api/predict`) and performance evaluation metrics (`/api/metrics`).
* **Interactive Frontend:** Responsive, glassmorphism-themed web dashboard with real-time prediction output, dynamic probability gauge, and live model evaluation analytics.
* **Automated Testing:** Dedicated test suite (`test_pipeline.py`) ensuring zero data leakage and reliable pipeline execution.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    A[Raw Data: crx.data] --> B[train_model.py]
    B --> C[Data Preprocessing Pipeline: Imputation, Encoding, Scaling]
    C --> D[Model Training: Logistic Regression + GridSearchCV]
    D --> E[Saved Artifacts: model.joblib, pipeline.joblib, metrics.json]
    E --> F[Flask Backend: app.py]
    F --> G[Web API: /predict & /metrics]
    G --> H[Frontend Dashboard: index.html, style.css, app.js]
    H --> I[User Input Form]
    H --> J[Prediction Result Card & Radial Gauge]
    H --> K[Model Evaluation Dashboard]
