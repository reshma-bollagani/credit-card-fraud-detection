# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# --- 1. Create a Flask App ---
app = Flask(__name__)

# --- 2. Load The Trained Model and Scaler ---
print("[INFO] Loading model and scaler...")
try:
    model = joblib.load('models/fraud_detection_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    print("[SUCCESS] Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"[ERROR] Could not load model or scaler: {e}")
    print("[ERROR] Please make sure 'fraud_detection_model.joblib' and 'scaler.joblib' are in the 'models' directory.")
    # Exit if models are not found, as the app is useless without them.
    exit()


# --- 3. Define a Root Route ---
# This route will serve the main HTML page for our web app.
@app.route('/')
def home():
    # We will create this 'index.html' file in the next phase.
    # The `render_template` function looks for HTML files in a 'templates' folder.
    return render_template('index.html')


# --- 4. Define The Prediction Route ---
# This route will handle the prediction requests from the frontend.
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives transaction data from a POST request, uses the loaded model
    to make a prediction, and returns the prediction as JSON.
    """
    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()
        
        # The frontend will send the data as a dictionary.
        # We need to extract the values in the correct order.
        # The order must match the order of columns used during model training.
        
        # The original training columns (excluding original Time/Amount and Class)
        # It's crucial this order is maintained.
        feature_order = [
            'scaled_Time', 'scaled_Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
        ]

        # The data from the form will not have scaled values.
        # We must scale 'Time' and 'Amount' using our saved scaler.
        time_val = float(data['time'])
        amount_val = float(data['amount'])

        # The scaler expects a 2D array, so we reshape.
        scaled_time = scaler.transform(np.array([[time_val]]))[0][0]
        scaled_amount = scaler.transform(np.array([[amount_val]]))[0][0]

        # Create a dictionary for the input features
        input_features = {
            'scaled_Time': scaled_time,
            'scaled_Amount': scaled_amount
        }
        # Add the 'V' features from the input data
        for i in range(1, 29):
            v_feature = f'V{i}'
            input_features[v_feature] = float(data.get(v_feature, 0.0)) # Use .get for safety

        # Create a pandas DataFrame from the dictionary, ensuring column order
        final_features_df = pd.DataFrame([input_features])
        final_features_df = final_features_df[feature_order] # Enforce the correct column order

        # Make a prediction
        prediction = model.predict(final_features_df)
        prediction_proba = model.predict_proba(final_features_df)

        # Interpret the prediction
        if prediction[0] == 1:
            result = 'Fraudulent Transaction'
        else:
            result = 'Normal Transaction'
            
        # Get the confidence score
        confidence = prediction_proba[0][prediction[0]] * 100

        # Return the result as a JSON object
        return jsonify({
            'prediction': result,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        # If anything goes wrong, return an error message.
        print(f"[ERROR] An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Check the server logs.'}), 500


# --- 5. Run the App ---
if __name__ == '__main__':
    # The host='0.0.0.0' makes the app accessible from your local network.
    # The port can be any number, 5000 is a common choice for Flask.
    app.run(host='0.0.0.0', port=5000, debug=True)

