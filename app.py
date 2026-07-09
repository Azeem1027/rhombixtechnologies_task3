import os
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Initialize Flask app
# Serving static files from the 'static' folder directly at the root URL
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"

# Load the trained machine learning pipeline and metrics
model_pipeline = None
metrics = None

if os.path.exists(MODEL_FILE):
    try:
        model_pipeline = joblib.load(MODEL_FILE)
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
else:
    print(f"Warning: {MODEL_FILE} not found. Please run train_model.py first.")

if os.path.exists(METRICS_FILE):
    try:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
        print("Metrics loaded successfully.")
    except Exception as e:
        print(f"Error loading metrics: {e}")
else:
    print(f"Warning: {METRICS_FILE} not found. Please run train_model.py first.")

# Feature names as expected by the trained pipeline
FEATURE_NAMES = [
    "Gender", "Age", "Debt", "Married", "BankCustomer", "EducationLevel", 
    "Ethnicity", "YearsEmployed", "PriorDefault", "Employed", "CreditScore", 
    "DriversLicense", "Citizen", "ZipCode", "Income"
]

@app.route("/")
def index():
    """Serve the index.html from static folder."""
    return app.send_static_file("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Endpoint to fetch ML model evaluation metrics and training parameters."""
    if metrics is None:
        return jsonify({"error": "Metrics not loaded. Make sure the model is trained."}), 500
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict credit card approval based on input features.
    Expects a JSON body with keys matching the FEATURE_NAMES.
    """
    if model_pipeline is None:
        return jsonify({"error": "Model not loaded. Make sure the model is trained."}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract features and format them correctly
        input_data = {}
        for col in FEATURE_NAMES:
            if col not in data:
                # If a feature is missing, we pass None so SimpleImputer handles it
                input_data[col] = None
            else:
                val = data[col]
                # Convert numeric fields to float, keeping None if empty string
                if col in ["Age", "Debt", "YearsEmployed", "CreditScore", "ZipCode", "Income"]:
                    if val == "" or val is None:
                        input_data[col] = None
                    else:
                        try:
                            input_data[col] = float(val)
                        except ValueError:
                            input_data[col] = None
                else:
                    # Categorical feature
                    input_data[col] = str(val) if val is not None else None

        # Convert dictionary to DataFrame (as expected by ColumnTransformer)
        df_input = pd.DataFrame([input_data])
        
        # Ensure correct column order
        df_input = df_input[FEATURE_NAMES]
        
        # Get class prediction (0 or 1) and probability distribution
        prediction = int(model_pipeline.predict(df_input)[0])
        probabilities = model_pipeline.predict_proba(df_input)[0]
        approval_probability = float(probabilities[1]) # Probability of being approved (class 1)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": approval_probability
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
