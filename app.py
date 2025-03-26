from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Define paths
output_dir = r"C:\Users\prash\OneDrive\Documents\Project\Backend\outputs"
model_path = os.path.join(output_dir, "model.pkl")
preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")

# Load model and preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Load column names from training data (replace with actual column names)
expected_columns = [
    "category", "amt", "city", "job", "merchant", "hour", "day", "month", "day_of_week", 
    "lat", "long", "distance_from_home", "merch_long", "dob", "city_pop", "cc_num", 
    "unix_time", "state", "gender", "Unnamed: 0", "merch_lat"
]

# Default values for missing columns
default_values = {
    "category": "unknown",
    "amt": 0.0,
    "city": "unknown",
    "job": "unknown",
    "merchant": "unknown",
    "hour": 0,
    "day": 1,
    "month": 1,
    "day_of_week": 0,
    "lat": 0.0,
    "long": 0.0,
    "distance_from_home": 0.0,
    "merch_long": 0.0,
    "dob": "2000-01-01",  # Example default for date fields
    "city_pop": 0,
    "cc_num": 0,
    "unix_time": 0,
    "state": "unknown",
    "gender": "unknown",
    "Unnamed: 0": 0,
    "merch_lat": 0.0
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Add missing columns with default values
        for col in expected_columns:
            if col not in input_df:
                input_df[col] = default_values[col]

        # Ensure correct column order
        input_df = input_df[expected_columns]

        # Apply preprocessing
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]  # Probability of fraud

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

if __name__ == '__main__':
    app.run(debug=True)


