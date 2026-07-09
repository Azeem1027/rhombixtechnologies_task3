import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve
)

# Constants
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
DATA_FILE = "crx.data"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")

def build_and_train():
    # Load dataset
    # The dataset has no header, and missing values are marked as '?'
    df = pd.read_csv(DATA_FILE, header=None, na_values="?")
    
    # Feature Mapping: map indices to readable feature names
    feature_names = [
        "Gender", "Age", "Debt", "Married", "BankCustomer", "EducationLevel", 
        "Ethnicity", "YearsEmployed", "PriorDefault", "Employed", "CreditScore", 
        "DriversLicense", "Citizen", "ZipCode", "Income"
    ]
    df.columns = feature_names + ["Approved"]
    
    # Preprocessing column conversions
    # Force Age and ZipCode to be numeric (just in case pandas read them as objects due to '?')
    df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
    df["ZipCode"] = pd.to_numeric(df["ZipCode"], errors='coerce')
    
    # Convert Approved label to binary (1 for Approved '+', 0 for Denied '-')
    # Map '+' to 1 and '-' to 0
    df["Approved"] = df["Approved"].map({"+": 1, "-": 0})
    
    # Drop rows where target variable is missing (if any)
    df = df.dropna(subset=["Approved"])
    
    # Split into features X and target y
    X = df[feature_names]
    y = df["Approved"]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()} (1 = Approved, 0 = Denied)")
    
    # Define numeric and categorical columns
    numeric_cols = ["Age", "Debt", "YearsEmployed", "CreditScore", "ZipCode", "Income"]
    categorical_cols = ["Gender", "Married", "BankCustomer", "EducationLevel", "Ethnicity", "PriorDefault", "Employed", "DriversLicense", "Citizen"]
    
    # Split into train and test sets (stratified to maintain label ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # robust to outliers
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Define final model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    # Hyperparameter search grid
    param_grid = [
        {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['saga', 'lbfgs']
        }
    ]
    
    # Set up GridSearchCV
    print("Starting Grid Search CV...")
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Extract best estimator
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Grid search complete. Best parameters: {best_params}")
    
    # Make predictions on test set
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert numpy array to list for JSON serialization
    
    # Compute ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # Downsample ROC curve slightly to prevent huge JSON file (e.g. max 50 points)
    downsample_factor = max(1, len(fpr) // 50)
    roc_data = {
        "fpr": fpr[::downsample_factor].tolist() + [fpr[-1]],
        "tpr": tpr[::downsample_factor].tolist() + [tpr[-1]]
    }
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    
    # Save the complete pipeline (preprocessor + best classifier)
    print(f"Saving model pipeline to {MODEL_FILE}...")
    joblib.dump(best_pipeline, MODEL_FILE)
    
    # Prepare metrics object
    # Clean up param keys for display (remove classifier__ prefix)
    clean_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    
    metrics = {
        "best_params": clean_params,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "class_distribution": {
            "approved": int((y == 1).sum()),
            "denied": int((y == 0).sum())
        },
        "dataset_size": int(df.shape[0]),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0])
    }
    
    # Save metrics to JSON file
    print(f"Saving metrics to {METRICS_FILE}...")
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Training process finished successfully.")

if __name__ == "__main__":
    download_data()
    build_and_train()
