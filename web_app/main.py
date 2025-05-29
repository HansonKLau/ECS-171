import os
from flask import Flask, request, jsonify, send_from_directory, abort # Added Flask imports
from pydantic import BaseModel, Field, ValidationError # Added ValidationError
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- Application Setup ---
app = Flask(__name__) # Flask app initialization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models/") # if app is in web_app folder
# MODEL_PATH = "saved_models/" # if app is at project root

STATIC_DIR = os.path.join(BASE_DIR, "static") # Define static directory path

# For clarity, define specific filenames
FEATURE_NAMES_FILE = os.path.join(MODEL_PATH, "best_logreg_feature_names.joblib")
SCALER_FILE = os.path.join(MODEL_PATH, "best_logreg_scaler.joblib")
MODEL_FILE = os.path.join(MODEL_PATH, "best_logreg_model.joblib")

# Global variables to hold loaded resources
model: Any = None
scaler: Any = None
feature_names: List[str] = []

def load_resources():
    """Load the model, scaler, and feature names at application startup."""
    global model, scaler, feature_names
    print("Loading resources for the best logistic regression model...")

    if os.path.exists(FEATURE_NAMES_FILE):
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        print(f"Loaded feature names: {feature_names}")
    else:
        print(f"Warning: Feature names file not found at {FEATURE_NAMES_FILE}")

    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print("Scaler loaded successfully.")
    else:
        print(f"Warning: Scaler file not found at {SCALER_FILE}")

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("Logistic regression model loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_FILE}")

load_resources() # Call directly for Flask
# --- Pydantic Input Model ---
class BestLogRegFeaturesInput(BaseModel):
    radius_error: float = Field(..., example=0.5)
    worst_texture: float = Field(..., example=25.0)
    worst_area: float = Field(..., example=900.0)
    worst_smoothness: float = Field(..., example=0.15)
    worst_concave_points: float = Field(..., example=0.2)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "radius_error": 0.2699, # from chris's set4model example
                    "worst_texture": 19.26,
                    "worst_area": 711.2,
                    "worst_smoothness": 0.144,
                    "worst_concave_points": 0.1288
                }
            ]
        }
# --- Prediction Response Model ---
class PredictionResponse(BaseModel):
    prediction: int = Field(..., example=0, description="0 for malignant, 1 for benign")
    probability_benign: float = Field(..., example=0.95, description="Predicted probability of being benign (class 1)")
    probability_malignant: float = Field(..., example=0.05, description="Predicted probability of being malignant (class 0)")
    model_used: str = Field(..., example="Best Logistic Regression")
    features_received: Dict[str, float]
    features_expected_by_model: List[str]


# --- API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict_best_logistic_regression():
    """
    Predict breast cancer diagnosis using the best logistic regression model
    trained on ['radius error', 'worst texture', 'worst area', 'worst smoothness', 'worst concave points'].
    """
    global model, scaler, feature_names # access the loaded resources

    if not model or not scaler or not feature_names or not feature_names: # check feature_names too
        print("Error: Model, scaler, or feature names not loaded.") # Log for server visibility
        abort(503, description="Model, scaler, or feature names not loaded. API is not operational.")

    raw_input_data = request.get_json()
    if not raw_input_data:
        abort(400, description="Invalid JSON input or empty request body.")

    try:
        input_model = BestLogRegFeaturesInput.model_validate(raw_input_data)
        input_dict = input_model.model_dump()
    except ValidationError as e:
        abort(400, description=f"Input validation error: {e.errors()}")
        
    df_temp = pd.DataFrame([input_dict])

    rename_map = {fn.replace(" ", "_"): fn for fn in feature_names}

    try:
        df_renamed = df_temp.rename(columns=rename_map)
        # make sure all expected columns are present and in the correct order
        df_input_ordered = df_renamed[feature_names]
    except KeyError as e:
        abort(400, description=f"Feature mismatch. Expected features like: {feature_names}. Error: {e}. Provided features (after internal mapping): {list(df_renamed.columns)}")
    except Exception as e: # Catch other potential errors during DataFrame manipulation
        abort(500, description=f"Error preparing DataFrame: {e}")

    try:
        scaled_input_np = scaler.transform(df_input_ordered)
    except ValueError as e:
        abort(500, description=f"Error during scaling. Scaler expected different features. Features given to scaler: {list(df_input_ordered.columns)}. Error: {e}")
    except Exception as e:
        abort(500, description=f"Unexpected error during scaling: {e}")

    # Make prediction
    try:
        probabilities = model.predict_proba(scaled_input_np)[0] # Prob for [class_0_malignant, class_1_benign]
        prob_malignant = float(probabilities[0])
        prob_benign = float(probabilities[1])
        prediction = 1 if prob_benign > prob_malignant else 0 # Benign if its probability is higher
    except Exception as e:
        abort(500, description=f"Error during model prediction: {str(e)}")

    response_data = PredictionResponse(
        prediction=prediction,
        probability_benign=prob_benign,
        probability_malignant=prob_malignant,
        model_used="Best Logistic Regression",
        features_received=input_dict, # Send back the validated and structured input
        features_expected_by_model=feature_names
    )
    return jsonify(response_data.model_dump())

# --- Root Endpoint to serve index.html ---
@app.route("/", methods=["GET"])
def read_index():
    index_html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_html_path):
        print(f"Warning: index.html not found at {index_html_path}. Serving error message.")
        return jsonify({
            "message": "Welcome to the Simple Breast Cancer Prediction API! Frontend (index.html) not found in static folder.",
            "status": "error"
        }), 404
    return send_from_directory(STATIC_DIR, "index.html")

# flask handles static files by default if the static folder is in the same directory as the app
if not (os.path.exists(STATIC_DIR) and os.path.isdir(STATIC_DIR)):
    print(f"Warning: Static directory '{STATIC_DIR}' not found or is not a directory. Static files (like CSS, JS) might not be served correctly from the /static URL path.")
else:
    print(f"Static files expected to be served from '{STATIC_DIR}' via Flask's default /static route.")

if __name__ == "__main__":
    # make sure this port isn't in use or change it if you can't run the flask app
    app.run(debug=True, host="0.0.0.0", port=8000)
