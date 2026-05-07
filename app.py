# ============================================================
# VinoVeritas 🍷
# Part 2: Flask REST API for Wine Quality Prediction
# ============================================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ------------------------------------------------------------
# 1. CREATE FLASK APPLICATION
# ------------------------------------------------------------

app = Flask(__name__)

# ------------------------------------------------------------
# 2. LOAD THE SAVED MODEL AND SCALER
# ------------------------------------------------------------

model = joblib.load("vino_veritas_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------------------------------
# 3. DEFINE FEATURE ORDER
# ------------------------------------------------------------

feature_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# ------------------------------------------------------------
# 4. HOME ROUTE
# ------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the VinoVeritas Wine Quality Prediction API",
        "status": "API is running",
        "prediction_endpoint": "/predict"
    })

# ------------------------------------------------------------
# 5. PREDICTION ROUTE
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from request
        data = request.get_json()

        # Check if request body is empty
        if data is None:
            return jsonify({
                "error": "No input data provided. Please send wine features in JSON format."
            }), 400

        # Check for missing features
        missing_features = [feature for feature in feature_columns if feature not in data]

        if missing_features:
            return jsonify({
                "error": "Missing required wine features",
                "missing_features": missing_features
            }), 400

        # Arrange input data in the same order used during training
        input_data = pd.DataFrame(
            [[data[feature] for feature in feature_columns]],
            columns=feature_columns
        )

        # Scale input data using saved scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Convert numeric prediction into readable label
        if prediction == 1:
            result = "Good Quality"
        else:
            result = "Bad Quality"

        # Return prediction result
        return jsonify({
            "quality_prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ------------------------------------------------------------
# 6. RUN THE APP
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)