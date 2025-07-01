
import pandas as pd
import numpy as np
from prophet import Prophet
import argparse
import os
import json
import pickle

def model_fn(model_dir):
    """Load the Prophet model from the model_dir."""
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as fin:
        model = pickle.load(fin)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data payload."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction using the Prophet model."""
    forecast = model.predict(input_data)
    return forecast

def output_fn(prediction, accept):
    """Format prediction output."""
    if accept == 'application/json':
        # Convert DataFrame to dict and handle timestamp serialization
        prediction_dict = prediction.to_dict('records')
        
        # Convert any timestamp objects to ISO format strings
        for record in prediction_dict:
            for key, value in record.items():
                if hasattr(value, 'isoformat'):  # Check if it's a timestamp-like object
                    record[key] = value.isoformat()
        
        return json.dumps(prediction_dict)
    raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()

    # Load training data
    training_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    
    # Initialize and train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.fit(training_data)
    
    # Save model using pickle instead of json
    with open(os.path.join(args.model_dir, 'model.pkl'), 'wb') as fout:
        pickle.dump(model, fout)
