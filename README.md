---
title: NYC Taxi Fare Prediction API
emoji: �
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# NYC Taxi Fare Prediction API

This Space provides an API for predicting NYC taxi fares using a deep neural network trained on NYC taxi trip data. The model takes various trip features and estimates the fare amount in USD.

## Features

- **Deep Neural Network**: PyTorch-based model with 14 input features
- **Model Compression**: Uses quantized model for faster inference and smaller memory footprint
- **Location-based Predictions**: Simplified endpoint for location ID-based predictions
- **Batch Processing**: Support for multiple predictions in a single request
- **CORS Enabled**: Ready for frontend integration

## Usage

### Web Interface
Visit the Space URL to access the health check endpoint and API documentation.

### API Endpoints

#### GET /
Health check endpoint that returns API status and model information.

**Response:**
```json
{
  "status": "success",
  "message": "Taxi Fare Prediction API is running",
  "model_loaded": true,
  "scaler_loaded": true,
  "distance_matrix_loaded": false,
  "total_features": 14,
  "timestamp": "2025-08-01T12:00:00.000000"
}
```

#### POST /predict
Main prediction endpoint with full feature control.

**Request:**
```json
{
  "passenger_count": 2,
  "trip_distance": 5.5,
  "extra": 0.5,
  "mta_tax": 0.5,
  "tip_amount": 3.0,
  "tolls_amount": 0.0,
  "payment_type": 1,
  "congestion_surcharge": 2.5,
  "Airport_fee": 0.0,
  "cbd_congestion_fee": 0.75,
  "trip_duration_minutes": 22,
  "pickup_hour": 14,
  "pickup_day": "Friday",
  "pickup_month": 1
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_fare": 18.45,
  "currency": "USD",
  "input_data": {...},
  "timestamp": "2025-08-01T12:00:00.000000"
}
```

#### POST /predict_from_locations
Simplified prediction endpoint using pickup and dropoff location IDs.

**Request:**
```json
{
  "pickup_location_id": 161,
  "dropoff_location_id": 230,
  "passenger_count": 1,
  "pickup_hour": 14,
  "pickup_day": "Friday",
  "pickup_month": 1
}
```

**Response:**
```json
{
  "success": true,
  "status": "success",
  "predicted_fare": 15.20,
  "currency": "USD",
  "trip_details": {
    "pickup_location_id": 161,
    "dropoff_location_id": 230,
    "trip_distance": 4.2,
    "trip_duration_minutes": 18,
    "passenger_count": 1,
    "pickup_hour": 14,
    "pickup_day": "Friday",
    "pickup_month": 1
  },
  "estimated_features": {
    "congestion_surcharge": 2.5,
    "airport_fee": 0.0,
    "cbd_congestion_fee": 0.75,
    "estimated_tip": 1.26
  },
  "timestamp": "2025-08-01T12:00:00.000000"
}
```

#### POST /predict/batch
Batch prediction endpoint for multiple trips.

**Request:**
```json
{
  "trips": [
    {
      "passenger_count": 1,
      "trip_distance": 3.2,
      "pickup_hour": 9,
      "pickup_day": "Monday"
    },
    {
      "passenger_count": 2,
      "trip_distance": 7.8,
      "pickup_hour": 18,
      "pickup_day": "Friday"
    }
  ]
}
```

#### GET /features
Returns information about required features and example requests.

## Model Features

The model uses 14 input features:

- **passenger_count**: Number of passengers (1-6)
- **trip_distance**: Trip distance in miles
- **extra**: Extra charges
- **mta_tax**: MTA tax (usually 0.5)
- **tip_amount**: Tip amount
- **tolls_amount**: Toll charges
- **payment_type**: Payment type (1=Credit, 2=Cash)
- **congestion_surcharge**: Congestion surcharge
- **Airport_fee**: Airport fee
- **cbd_congestion_fee**: CBD congestion fee
- **trip_duration_minutes**: Trip duration in minutes
- **pickup_hour**: Pickup hour (0-23)
- **pickup_day**: Day of week (Monday-Sunday or 0-6)
- **pickup_month**: Month (1-12)

## Technical Details

- **Framework**: Flask with PyTorch
- **Model**: Deep Neural Network (128→64→32→1 neurons)
- **Compression**: INT8 quantization for 71% size reduction
- **CORS**: Enabled for cross-origin requests
- **Logging**: Comprehensive request/error logging

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

The API will be available at `http://localhost:5000`

## Docker Deployment

```bash
# Build the image
docker build -t taxi-fare-api .

# Run the container
docker run -p 7860:7860 taxi-fare-api
```