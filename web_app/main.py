import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- Application Setup ---
app = FastAPI(
    title="Simple Breast Cancer Prediction API (Logistic Regression)",
    description="API for predicting breast cancer diagnosis using the best logistic regression model.",
)

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

@app.on_event("startup")
def load_resources():
    """Load the model, scaler, and feature names at application startup."""
    global model, scaler, feature_names
    print("Loading resources for the best logistic regression model...")

    if os.path.exists(FEATURE_NAMES_FILE):
        feature_names = joblib.load(FEATURE_NAMES_FILE)
        print(f"Loaded feature names: {feature_names}")

    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
        print("Scaler loaded successfully.")


    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("Logistic regression model loaded successfully.")



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
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_best_logistic_regression(input_data: BestLogRegFeaturesInput):
    """
    Predict breast cancer diagnosis using the best logistic regression model
    trained on ['radius error', 'worst texture', 'worst area', 'worst smoothness', 'worst concave points'].
    """
    global model, scaler, feature_names # access the loaded resources

    if not model or not scaler or not feature_names:
        raise HTTPException(
            status_code=503, # 503 is service unavailable
            detail="Model, scaler, or feature names not loaded. API is not operational."
        )

    # convert pydantic model to a dictionary
    input_dict = input_data.model_dump()
    df_temp = pd.DataFrame([input_dict])

    # not sure if it's specific to fastapi but it acts out when the feature names are not written with underscores 
    # so we just replace the spaces with underscores here
    rename_map = {fn.replace(" ", "_"): fn for fn in feature_names}

    try:
        df_renamed = df_temp.rename(columns=rename_map)
        # make sure all expected columns are present and in the correct order
        df_input_ordered = df_renamed[feature_names]
    except KeyError as e:
        # This error means a column expected by feature_names was not found after renaming
        # or an input feature wasn't provided that was expected.
        raise HTTPException(
            status_code=400,
            detail=f"Feature mismatch after renaming. Expected features: {feature_names}. Error: {e}. "
                   f"Renamed df_temp columns were: {list(df_renamed.columns)}"
        )
    except Exception as e: # Catch other potential errors during DataFrame manipulation
        raise HTTPException(status_code=500, detail=f"Error preparing DataFrame: {e}")


    # Scale the features
    try:
        scaled_input_np = scaler.transform(df_input_ordered)
    except ValueError as e:
        # This can happen if feature names still don't match what scaler expects
        raise HTTPException(
            status_code=500,
            detail=f"Error during scaling. Scaler expected different features than provided. "
                   f"Features passed to scaler: {list(df_input_ordered.columns)}. Error: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during scaling: {e}")


    # Make prediction
    try:
        probabilities = model.predict_proba(scaled_input_np)[0] # Prob for [class_0_malignant, class_1_benign]
        prob_malignant = float(probabilities[0])
        prob_benign = float(probabilities[1])
        prediction = 1 if prob_benign > prob_malignant else 0 # Benign if its probability is higher
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    return PredictionResponse(
        prediction=prediction,
        probability_benign=prob_benign,
        probability_malignant=prob_malignant,
        model_used="Best Logistic Regression",
        features_received=input_dict,
        features_expected_by_model=feature_names
    )

# --- Root Endpoint to serve index.html ---
@app.get("/", response_class=FileResponse, tags=["General"])
async def read_index():
    index_html_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_html_path):
        print(f"Warning: index.html not found at {index_html_path}. Serving /docs link.")
        return {
            "message": "Welcome to the Simple Breast Cancer Prediction API! Frontend (index.html) not found.",
            "try_documentation_at": "/docs"
        }
    return FileResponse(index_html_path)

# --- Mount static files directory (CSS, JS, images etc.) ---
if os.path.exists(STATIC_DIR) and os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static_assets") # Changed name to avoid conflict if "static" is used elsewhere
    print(f"Static files mounted from '{STATIC_DIR}' at '/static'")
else:
    print(f"Warning: Static directory '{STATIC_DIR}' not found or is not a directory. Static files will not be served from /static path.")
