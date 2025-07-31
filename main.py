"""
Taxi Fare Prediction API using Flask
Serves a PyTorch model trained on NYC taxi data for fare predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import traceback
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None
distance_matrix = None
feature_order = [
    'passenger_count', 'trip_distance',
    'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'payment_type', 'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee',
    'trip_duration_minutes', 'pickup_hour', 'pickup_day', 'pickup_month'
]

# Model Architecture (same as training)
class TaxiFareModel(nn.Module):
    """
    Deep Neural Network for taxi fare prediction
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super(TaxiFareModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()

def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler, distance_matrix
    
    try:
        # Load model
        input_size = len(feature_order)  # Updated to 14 features
        model = TaxiFareModel(input_size=input_size)
        
        # Load trained weights
        model_path = 'best_taxi_fare_model.pth'
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading model weights: {e}. Using untrained model.")
        else:
            logger.warning(f"Model file {model_path} not found. Using untrained model.")
        
        # Load scaler
        scaler_path = '../best_models/scaler.pkl'
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                logger.warning(f"Scaler file corrupted or invalid: {e}. Creating new scaler.")
                # Create a new scaler if the file is corrupted
                scaler = StandardScaler()
                # Fit with dummy data that matches expected ranges (14 features)
                dummy_data = np.array([[
                    1, 5, 0.5, 0.5, 2, 0, 1, 2.5, 0, 0.75, 20, 12, 3, 1
                ]])
                scaler.fit(dummy_data)
        else:
            # Create a dummy scaler if not found
            scaler = StandardScaler()
            # Fit with dummy data that matches expected ranges (14 features)
            dummy_data = np.array([[
                1, 5, 0.5, 0.5, 2, 0, 1, 2.5, 0, 0.75, 20, 12, 3, 1
            ]])
            scaler.fit(dummy_data)
            logger.warning("Scaler not found. Created dummy scaler.")
        
        # Load distance matrix
        distance_matrix_path = '../distances/full_taxi_zone_distance_matrix.csv'
        if os.path.exists(distance_matrix_path):
            try:
                distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
                logger.info("Distance matrix loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading distance matrix: {e}. Location-based predictions will use default distance.")
                distance_matrix = None
        else:
            logger.warning("Distance matrix not found. Location-based predictions will use default distance.")
            distance_matrix = None
            
    except Exception as e:
        logger.error(f"Error loading model/scaler: {str(e)}")
        raise

def preprocess_input(data):
    """
    Preprocess input data for prediction
    """
    try:
        # Handle day name to number conversion
        if 'pickup_day' in data and isinstance(data['pickup_day'], str):
            day_mapping = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6,
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            data['pickup_day'] = day_mapping.get(data['pickup_day'], 0)
        
        # Create feature array in correct order
        features = []
        for feature in feature_order:
            if feature in data:
                features.append(float(data[feature]))
            else:
                # Default values for missing features (updated for 14 features)
                defaults = {
                    'passenger_count': 1, 'trip_distance': 5.0, 'extra': 0.5, 'mta_tax': 0.5,
                    'tip_amount': 2.0, 'tolls_amount': 0.0, 'payment_type': 1, 
                    'congestion_surcharge': 2.5, 'Airport_fee': 0.0, 'cbd_congestion_fee': 0.75, 
                    'trip_duration_minutes': 20, 'pickup_hour': 12, 'pickup_day': 3, 'pickup_month': 1
                }
                features.append(defaults.get(feature, 0.0))
        
        # Convert to numpy array
        features_array = np.array([features], dtype=np.float32)
        
        # Log the feature array before scaling
        logger.info(f"Features before scaling: {features}")
        logger.info(f"Passenger count in features[0]: {features[0]}")
        
        # Apply scaling
        if scaler:
            features_array = scaler.transform(features_array)
            logger.info(f"Features after scaling: {features_array[0]}")
        
        return features_array
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

def make_prediction(features_array):
    """
    Make prediction using the loaded model
    """
    try:
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.tensor(features_array, dtype=torch.float32)
            
            # Make prediction
            prediction = model(input_tensor)
            
            # Return as float
            return float(prediction.item())
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise

# API Routes

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Taxi Fare Prediction API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'distance_matrix_loaded': distance_matrix is not None,
        'total_features': len(feature_order),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_fare():
    """
    Main prediction endpoint
    Expected JSON format:
    {
        "PULocationID": 161,
        "DOLocationID": 230,
        "passenger_count": 2,
        "trip_distance": 5.5,
        "extra": 0.5,
        "mta_tax": 0.5,
        "tip_amount": 3.0,
        "tolls_amount": 0.0,
        "total_amount": 25.0,
        "payment_type": 1,
        "trip_type": 1,
        "congestion_surcharge": 2.5,
        "cbd_congestion_fee": 0.75,
        "trip_duration_minutes": 22,
        "pickup_hour": 14,
        "pickup_day": "Friday",
        "pickup_month": 1
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Log the request
        logger.info(f"Prediction request: {data}")
        
        # Preprocess input
        features_array = preprocess_input(data)
        
        # Make prediction
        predicted_fare = make_prediction(features_array)
        
        # Ensure prediction is reasonable (basic validation)
        if predicted_fare < 0:
            predicted_fare = abs(predicted_fare)
        if predicted_fare > 1000:  # Cap at $1000
            predicted_fare = 1000
        
        # Prepare response
        response = {
            'status': 'success',
            'predicted_fare': round(predicted_fare, 2),
            'currency': 'USD',
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: ${predicted_fare:.2f}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Prediction error: {error_msg}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {error_msg}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_from_locations', methods=['POST'])
def predict_fare_from_locations():
    """
    Location-based prediction endpoint for simplified UI
    Expected JSON format:
    {
        "pickup_location_id": 161,
        "dropoff_location_id": 230,
        "passenger_count": 1,
        "pickup_hour": 14,
        "pickup_day": "Friday",
        "pickup_month": 1
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Required fields
        required_fields = ['pickup_location_id', 'dropoff_location_id']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Extract location IDs
        pickup_id = int(data['pickup_location_id'])
        dropoff_id = int(data['dropoff_location_id'])
        
        # Get distance from matrix
        trip_distance = 5.0  # Default distance
        if distance_matrix is not None:
            try:
                trip_distance = distance_matrix.loc[pickup_id, str(dropoff_id)]
                if pd.isna(trip_distance):
                    trip_distance = 5.0  # Fallback for missing distances
            except (KeyError, IndexError):
                logger.warning(f"Distance not found for {pickup_id} -> {dropoff_id}, using default")
                trip_distance = 5.0
        
        # Estimate trip duration (rough estimate: 2.5 minutes per mile + base time)
        trip_duration_minutes = max(5, int(trip_distance * 2.5))
        
        # Get other parameters with defaults
        passenger_count = data.get('passenger_count', 1)
        pickup_hour = data.get('pickup_hour', 14)
        pickup_day = data.get('pickup_day', 'Friday')
        pickup_month = data.get('pickup_month', 1)
        
        # Log received passenger count
        logger.info(f"Received passenger_count: {passenger_count}")
        
        # Create trip features with intelligent estimates
        trip_features = {
            'passenger_count': passenger_count,
            'trip_distance': trip_distance,
            'extra': 0.5,  # Standard extra charge
            'mta_tax': 0.5,  # Standard MTA tax
            'tip_amount': max(2.0, trip_distance * 0.3),  # Estimated tip (30% of distance)
            'tolls_amount': 0.0,  # Default no tolls
            'payment_type': 1,  # Default credit card
            'congestion_surcharge': 2.5 if 6 <= pickup_hour <= 20 else 0.0,  # Peak hours
            'Airport_fee': 5.0 if pickup_id in [1, 132, 138] or dropoff_id in [1, 132, 138] else 0.0,
            'cbd_congestion_fee': 0.75 if pickup_id <= 100 or dropoff_id <= 100 else 0.0,  # Manhattan zones
            'trip_duration_minutes': trip_duration_minutes,
            'pickup_hour': pickup_hour,
            'pickup_day': pickup_day,
            'pickup_month': pickup_month
        }
        
        # Preprocess and predict
        features_array = preprocess_input(trip_features)
        predicted_fare = make_prediction(features_array)
        
        # Ensure prediction is reasonable
        if predicted_fare < 0:
            predicted_fare = abs(predicted_fare)
        if predicted_fare > 1000:
            predicted_fare = 1000
        
        # Prepare response
        response = {
            'success': True,
            'status': 'success',
            'predicted_fare': round(predicted_fare, 2),
            'currency': 'USD',
            'trip_details': {
                'pickup_location_id': pickup_id,
                'dropoff_location_id': dropoff_id,
                'trip_distance': round(trip_distance, 2),
                'trip_duration_minutes': trip_duration_minutes,
                'passenger_count': passenger_count,
                'pickup_hour': pickup_hour,
                'pickup_day': pickup_day,
                'pickup_month': pickup_month
            },
            'estimated_features': {
                'congestion_surcharge': trip_features['congestion_surcharge'],
                'airport_fee': trip_features['Airport_fee'],
                'cbd_congestion_fee': trip_features['cbd_congestion_fee'],
                'estimated_tip': trip_features['tip_amount']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Location-based prediction: {pickup_id}->{dropoff_id} = ${predicted_fare:.2f}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Location-based prediction error: {error_msg}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'status': 'error',
            'message': f'Location-based prediction failed: {error_msg}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Expected JSON format:
    {
        "trips": [
            {trip_data_1},
            {trip_data_2},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'trips' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No trips data provided. Expected format: {"trips": [...]}'
            }), 400
        
        trips = data['trips']
        predictions = []
        
        for i, trip in enumerate(trips):
            try:
                features_array = preprocess_input(trip)
                predicted_fare = make_prediction(features_array)
                
                # Basic validation
                if predicted_fare < 0:
                    predicted_fare = abs(predicted_fare)
                if predicted_fare > 1000:
                    predicted_fare = 1000
                
                predictions.append({
                    'trip_index': i,
                    'predicted_fare': round(predicted_fare, 2),
                    'input_data': trip
                })
                
            except Exception as e:
                predictions.append({
                    'trip_index': i,
                    'error': str(e),
                    'input_data': trip
                })
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'total_trips': len(trips),
            'successful_predictions': len([p for p in predictions if 'predicted_fare' in p]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get information about required features"""
    feature_info = {
        'required_features': feature_order,
        'feature_descriptions': {
            'passenger_count': 'Number of passengers (1-6)',
            'trip_distance': 'Trip distance in miles',
            'extra': 'Extra charges',
            'mta_tax': 'MTA tax (usually 0.5)',
            'tip_amount': 'Tip amount',
            'tolls_amount': 'Toll charges',
            'payment_type': 'Payment type (1=Credit, 2=Cash)',
            'congestion_surcharge': 'Congestion surcharge',
            'Airport_fee': 'Airport fee',
            'cbd_congestion_fee': 'CBD congestion fee',
            'trip_duration_minutes': 'Trip duration in minutes',
            'pickup_hour': 'Pickup hour (0-23)',
            'pickup_day': 'Day of week (Monday-Sunday or 0-6)',
            'pickup_month': 'Month (1-12)'
        },
        'example_request': {
            'passenger_count': 2,
            'trip_distance': 5.5,
            'extra': 0.5,
            'mta_tax': 0.5,
            'tip_amount': 3.0,
            'tolls_amount': 0.0,
            'payment_type': 1,
            'congestion_surcharge': 2.5,
            'Airport_fee': 0.0,
            'cbd_congestion_fee': 0.75,
            'trip_duration_minutes': 22,
            'pickup_hour': 14,
            'pickup_day': 'Friday',
            'pickup_month': 1
        },
        'location_based_example': {
            'pickup_location_id': 161,
            'dropoff_location_id': 230,
            'passenger_count': 1,
            'pickup_hour': 14,
            'pickup_day': 'Friday',
            'pickup_month': 1
        }
    }
    
    return jsonify(feature_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': [
            'GET /',
            'POST /predict',
            'POST /predict_from_locations',
            'POST /predict/batch',
            'GET /features'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# Initialize the app
if __name__ == '__main__':
    try:
        logger.info("Starting Taxi Fare Prediction API...")
        
        # Load model and scaler
        load_model_and_scaler()
        
        # Start the Flask app
        port = int(os.environ.get('PORT', 5000))

        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise

