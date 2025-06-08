import os
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from typing import List, Any

app = Flask(__name__, template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models/")

RF_MODEL_FILE = os.path.join(MODEL_PATH, "breast_cancer_model.joblib")
RF_SCALER_FILE = os.path.join(MODEL_PATH, "scaler.joblib")

rf_model: Any = None
rf_scaler: Any = None
rf_feature_names: List[str] = [
    'worst perimeter', 'worst radius', 'worst concave points', 'worst area',
    'mean concave points', 'mean concavity', 'mean area', 'area error',
    'mean perimeter', 'mean radius', 'worst concavity', 'worst texture',
    'mean texture'
]

def load_rf_resources():
    """Load the Random Forest model and scaler at application startup."""
    global rf_model, rf_scaler
    print("Loading resources for the Random Forest model...")

    if os.path.exists(RF_SCALER_FILE):
        rf_scaler = joblib.load(RF_SCALER_FILE)
        print("Random Forest scaler loaded successfully.")
    else:
        print(f"FATAL: Random Forest scaler not found at {RF_SCALER_FILE}")
        rf_scaler = None

    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        print("Random Forest model loaded successfully.")
    else:
        print(f"FATAL: Random Forest model not found at {RF_MODEL_FILE}")
        rf_model = None

load_rf_resources()

@app.route("/", methods=["GET"])
def index():
    """Serve the main HTML page."""
    return render_template("index.html", features=rf_feature_names, prediction_result=None, form_data=None)

@app.route("/predict_rf", methods=["POST"])
def predict_random_forest():
    """
    Predict breast cancer diagnosis using the Random Forest model from form data.
    """
    global rf_model, rf_scaler, rf_feature_names

    if not rf_model or not rf_scaler:
        print("Error: Random Forest model or scaler not loaded.")
        return render_template("index.html", features=rf_feature_names, error="Model is not available. Please check server logs.")

    try:
        input_dict = {feature: float(request.form[feature.replace(" ", "_")]) for feature in rf_feature_names}
    except (KeyError, ValueError) as e:
        print(f"Error processing form data: {e}")
        return render_template("index.html", features=rf_feature_names, error="Invalid or missing form data. Please fill all fields with numeric values.")

    input_data = [input_dict[feature] for feature in rf_feature_names]
    df_input = pd.DataFrame([input_data], columns=rf_feature_names)

    try:
        scaled_input_np = rf_scaler.transform(df_input)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return render_template("index.html", features=rf_feature_names, error=f"Error during data scaling: {e}")

    try:
        probabilities = rf_model.predict_proba(scaled_input_np)[0]
        prediction_val = rf_model.predict(scaled_input_np)[0]

        prob_benign = float(probabilities[1])
        prob_malignant = float(probabilities[0])
        
        result_text = "Benign" if prediction_val == 1 else "Malignant"
        
        prediction_result = {
            "prediction_text": result_text,
            "probability_benign": f"{prob_benign:.2%}",
            "probability_malignant": f"{prob_malignant:.2%}"
        }

    except Exception as e:
        print(f"Error during model prediction: {str(e)}")
        return render_template("index.html", features=rf_feature_names, error=f"Error during model prediction: {e}")
    
    return render_template("index.html", features=rf_feature_names, prediction_result=prediction_result, form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
