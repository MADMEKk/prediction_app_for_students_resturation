from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins, methods, and headers globally

# Load the model and encoders
model = joblib.load('meal_prediction_model.joblib')
meal_type_encoder = joblib.load('meal_type_encoder.joblib')
meal_encoder = joblib.load('meal_encoder.joblib')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'Flask server is running!'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract encoded data directly from the request
        meal_type_encoded = data['meal_type']  # Already encoded as an integer
        meal_encoded = data['meal']           # Already encoded as an integer
        day_of_week = data['Day_of_Week']     # Directly use the integer value
        
        # Prepare features for prediction
        features = np.array([[day_of_week, meal_type_encoded, meal_encoded]])
        prediction = model.predict(features)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_swipes': int(prediction),
            'meal_type_encoded': meal_type_encoded,
            'meal_encoded': meal_encoded,
            'Day_of_Week': day_of_week
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400





if __name__ == '__main__':
    print("Server starting at http://localhost:8080")
    app.run(host='0.0.0.0', port=5000, debug=True)
