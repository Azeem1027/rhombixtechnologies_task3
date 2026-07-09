import os
import json
import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class TestCreditCardPipeline(unittest.TestCase):
    MODEL_FILE = "model.joblib"
    METRICS_FILE = "metrics.json"
    
    @classmethod
    def setUpClass(cls):
        # Check if training artifacts exist before running tests
        cls.model_exists = os.path.exists(cls.MODEL_FILE)
        cls.metrics_exists = os.path.exists(cls.METRICS_FILE)
        
        if cls.model_exists:
            cls.pipeline = joblib.load(cls.MODEL_FILE)
            
        if cls.metrics_exists:
            with open(cls.METRICS_FILE, "r") as f:
                cls.metrics = json.load(f)

    def test_artifacts_exist(self):
        """Verify that model training output files exist."""
        self.assertTrue(self.model_exists, f"{self.MODEL_FILE} is missing. Please run train_model.py first.")
        self.assertTrue(self.metrics_exists, f"{self.METRICS_FILE} is missing. Please run train_model.py first.")

    def test_loaded_pipeline_type(self):
        """Verify that the loaded model is a scikit-learn Pipeline."""
        self.assertTrue(self.model_exists)
        self.assertIsInstance(self.pipeline, Pipeline)
        
        # Verify the pipeline has the expected steps
        steps = [step[0] for step in self.pipeline.steps]
        self.assertIn("preprocessor", steps)
        self.assertIn("classifier", steps)

    def test_prediction_output(self):
        """Test making a prediction with valid complete mock inputs."""
        self.assertTrue(self.model_exists)
        
        # Formulate a valid mock application (all keys match features)
        mock_input = pd.DataFrame([{
            "Gender": "a",
            "Age": 30.0,
            "Debt": 5.0,
            "Married": "u",
            "BankCustomer": "g",
            "EducationLevel": "c",
            "Ethnicity": "v",
            "YearsEmployed": 2.5,
            "PriorDefault": "f",
            "Employed": "t",
            "CreditScore": 4,
            "DriversLicense": "t",
            "Citizen": "g",
            "ZipCode": 280,
            "Income": 1200
        }])
        
        # Predict classes
        pred = self.pipeline.predict(mock_input)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1], "Prediction class must be 0 or 1.")
        
        # Predict probabilities
        proba = self.pipeline.predict_proba(mock_input)
        self.assertEqual(proba.shape, (1, 2))
        self.assertTrue(0.0 <= proba[0][0] <= 1.0)
        self.assertTrue(0.0 <= proba[0][1] <= 1.0)
        self.assertAlmostEqual(proba[0][0] + proba[0][1], 1.0, places=5)

    def test_missing_values_imputation(self):
        """Test that the pipeline successfully imputes and processes missing values (None/NaN)."""
        self.assertTrue(self.model_exists)
        
        # Mock inputs containing missing numerical (Age, ZipCode) and categorical (Gender, Married) fields
        mock_input_with_missing = pd.DataFrame([{
            "Gender": None,               # Missing categorical
            "Age": np.nan,                # Missing numerical
            "Debt": 0.5,
            "Married": None,              # Missing categorical
            "BankCustomer": "g",
            "EducationLevel": "w",
            "Ethnicity": "h",
            "YearsEmployed": 0.0,
            "PriorDefault": "t",
            "Employed": "f",
            "CreditScore": 0,
            "DriversLicense": "f",
            "Citizen": "g",
            "ZipCode": np.nan,            # Missing numerical
            "Income": 0
        }])
        
        # This will raise an error if imputers do not handle None/NaN values properly
        try:
            pred = self.pipeline.predict(mock_input_with_missing)
            proba = self.pipeline.predict_proba(mock_input_with_missing)
            success = True
        except Exception as e:
            success = False
            print(f"Error during missing values test: {e}")
            
        self.assertTrue(success, "Pipeline failed to process data containing missing values.")
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1])

    def test_metrics_integrity(self):
        """Verify that training metrics are structured correctly."""
        self.assertTrue(self.metrics_exists)
        
        expected_keys = [
            "best_params", "accuracy", "precision", "recall", 
            "f1_score", "roc_auc", "confusion_matrix", "roc_curve",
            "dataset_size", "train_size", "test_size"
        ]
        for key in expected_keys:
            self.assertIn(key, self.metrics)
            
        # Metric bounds checking
        cls_accuracy = self.metrics["accuracy"]
        self.assertTrue(0.0 <= cls_accuracy <= 1.0)
        self.assertTrue(0.0 <= self.metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= self.metrics["recall"] <= 1.0)
        self.assertTrue(0.0 <= self.metrics["f1_score"] <= 1.0)
        self.assertTrue(0.0 <= self.metrics["roc_auc"] <= 1.0)
        
        # Confusion matrix checks
        cm = self.metrics["confusion_matrix"]
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)
        self.assertEqual(len(cm[1]), 2)
        
        # Param check
        self.assertIn("C", self.metrics["best_params"])

if __name__ == "__main__":
    unittest.main()
