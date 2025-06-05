# Breast Cancer Prediction Web Application

This project is a Flask-based web application that serves a machine learning model to predict whether a breast tissue sample is malignant or benign based on 13 key features. The model is a Random Forest Classifier trained on the Wisconsin Breast Cancer dataset.

## Project Structure

```
.
├── appModel.ipynb
├── saved_models
│   ├── breast_cancer_model.joblib
│   └── scaler.joblib
├── web_app
│   ├── main.py
│   ├── README.md
│   ├── requirements.txt
│   └── static
│       └── index.html
```

- `appModel.ipynb`: A Jupyter Notebook used to train the Random Forest model and the associated scaler. Running this notebook generates the `.joblib` files.
- `saved_models/`: Directory containing the serialized model and scaler.
- `web_app/main.py`: The Flask application script. It loads the model, defines API endpoints, and serves the frontend.
- `web_app/static/index.html`: The user-facing HTML page with a form to input features and see the prediction.
- `web_app/requirements.txt`: A list of Python dependencies required for the application.
- `web_app/README.md`: This file.

## Setup and Running the Application

### 1. Generate Model Files

Before running the web application, you must first generate the model and scaler files.

1.  Ensure you have the necessary dependencies installed (`jupyter`, `scikit-learn`, `numpy`, `joblib`). You can often install these via `pip`.
2.  Open and run the `appModel.ipynb` notebook from the project's root directory (`ECS-171/`).
3.  This will create two files: `breast_cancer_model.joblib` and `scaler.joblib` and place them in the `saved_models/` directory.

### 2. Run the Web Application

1.  **Navigate to the `web_app` Directory:**
    Open your terminal and change to the `ECS-171/web_app` directory.

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask Application:**
    From within the `ECS-171/web_app/` directory:
    ```bash
    flask run
    ```
    Or, for development with auto-reloading:
    ```bash
    flask --app main --debug run
    ```

5.  **Access the Application:**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

## API Endpoint

The application exposes a single API endpoint for predictions.

-   **`POST /predict_rf`**:
    -   **Description:** Accepts 13 feature values and returns a breast cancer diagnosis prediction (Malignant/Benign) using the trained Random Forest model.
    -   **Request Body (Form Data):** The request is sent as `application/x-www-form-urlencoded` from the HTML form. The keys are the feature names: `worst_perimeter`, `worst_radius`, `worst_concave_points`, `worst_area`, `mean_concave_points`, `mean_concavity`, `mean_area`, `area_error`, `mean_perimeter`, `mean_radius`, `worst_concavity`, `worst_texture`, `mean_texture`.
    -   **Response:** The endpoint renders the `index.html` template again, this time with the prediction result displayed.
