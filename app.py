# -*- coding: utf-8 -*-
"""
==========================================================================
Flask Backend for Uplift Prediction
==========================================================================

Purpose: This Flask application serves the uplift prediction model.
         - It loads the pre-trained models and feature columns on startup.
         - It provides a '/' route to serve the frontend HTML.
         - It provides a '/predict' API endpoint to handle prediction requests.

"""
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model Assets on Startup ---
try:
    print("Loading model assets...")
    control_model = joblib.load('model_assets/control_model.joblib')
    treatment_model = joblib.load('model_assets/treatment_model.joblib')
    feature_columns = joblib.load('model_assets/feature_columns.joblib')
    print("Model assets loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Model assets not found. Please run 'training_script.py' first.")
    # In a real app, you might want more robust error handling or a startup check
    control_model = None
    treatment_model = None
    feature_columns = None

# --- Hardcoded Thresholds (Derived from offline analysis) ---
# These would be the output of the training script's analysis phase.
model_thresholds = {
    'highUplift': 0.0118,
    'lowUplift': -0.0035,
    'highControlProb': 0.021,
    'negativeUplift': -0.001
}

def classify_user(uplift_score, p_control):
    """Classifies a user into a segment based on their scores."""
    if uplift_score >= model_thresholds['highUplift']:
        return 'Persuadables'
    if uplift_score < model_thresholds['lowUplift']:
        if p_control >= model_thresholds['highControlProb']:
            return 'Sure Things'
        if uplift_score < model_thresholds['negativeUplift']:
            return 'Sleeping Dogs'
        return 'Lost Causes'
    return 'Neutral / Low Impact'

@app.route('/')
def home():
    """Serves the frontend HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests."""
    if not all([control_model, treatment_model, feature_columns is not None]):
        return jsonify({'error': 'Models not loaded. Server is not ready.'}), 500

    # Get data from the POST request
    data = request.get_json(force=True)
    
    # Create a DataFrame from the input
    input_df = pd.DataFrame([data])

    # --- Preprocess the input data ---
    # This must match the training script's preprocessing exactly
    input_df['most ads day'] = pd.Categorical(input_df['most ads day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    input_df['most ads hour'] = pd.Categorical(input_df['most ads hour'], categories=list(range(24)))
    
    # One-hot encode
    input_processed = pd.get_dummies(input_df, columns=['most ads day', 'most ads hour'])
    
    # Align columns with the training data
    # This adds missing columns (with value 0) and ensures order is the same
    input_aligned = input_processed.reindex(columns=feature_columns, fill_value=0)

    # --- Make Predictions ---
    p_control = control_model.predict_proba(input_aligned)[:, 1][0]
    p_treatment = treatment_model.predict_proba(input_aligned)[:, 1][0]
    uplift_score = p_treatment - p_control

    # --- Classify and Respond ---
    segment = classify_user(uplift_score, p_control)
    
    response = {
        'segment': segment,
        'uplift_score': round(uplift_score, 5),
        'p_control': round(p_control, 5),
        'p_treatment': round(p_treatment, 5)
    }
    
    return jsonify(response)


if __name__ == '__main__':
    # To run this app: `flask run` in your terminal
    # Or `python app.py` if you have debug=True
    app.run(debug=True)
