from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model
try:
    model = joblib.load("phishing_detector.pkl")
    print("\n✅ Model loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Model file not found! Train the model first.")
    exit()

# Load test dataset to get feature names
try:
    test_data = joblib.load("test_data.pkl")
    X_test = test_data["X_test"]
    feature_names = list(X_test.columns)  # Get feature names from dataset
    print("\n✅ Feature names loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Test dataset not found! Run 'phishing_detection.py' first.")
    exit()

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Phishing Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input into a Pandas DataFrame with correct feature names
        new_website_df = pd.DataFrame([data["features"]], columns=feature_names)

        # Make prediction
        prediction = model.predict(new_website_df)

        # Return result
        result = {"prediction": int(prediction[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

import os

if __name__ == '_main_':
    port = int(os.environ.get('PORT', 5000))  # Render uses a dynamic port
    app.run(host='0.0.0.0', port=port)