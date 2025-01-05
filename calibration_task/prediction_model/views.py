from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import joblib
import numpy as np
import pandas as pd
import json
import os
import sys
from .utils import TemperatureScaling

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'prediction_model', 'models', 'final_model_with_temp_scaling.pkl')

# Features to drop
features_to_drop = [
    'Stddev No. Of Symbols per Categorical Features', 
    'Stddev No. Of Significant Lags in Target', 
    'Stddev No. Of Insignificant Lags in Target', 
    'Stddev No. Of Seasonality Components in Target'
]

def load_model():
    """Load the model only when needed"""
    # Add the TemperatureScaling class to both the global namespace and sys.modules
    sys.modules['__main__'].TemperatureScaling = TemperatureScaling
    globals()['TemperatureScaling'] = TemperatureScaling
    
    try:
        model_data = joblib.load(MODEL_PATH)
        return model_data['model'], model_data['temperature_scaler']
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    try:
        # Load model only when the endpoint is called
        loaded_final_model, loaded_temperature_scaler = load_model()
        
        # Parse the input features from JSON
        input_data = json.loads(request.body)
        
        # Prepare the predictions dictionary
        predictions = {}
        
        # List of classifiers
        class_names = ['ELASTICNETCV', 'HUBERREGRESSOR', 'LASSO', 'LinearSVR', 'QUANTILEREGRESSOR', 'XGBRegressor']
        
        for key, sample in input_data.items():
            # Convert the sample to a DataFrame for processing
            sample_df = pd.DataFrame([sample])
            
            # Drop unwanted features
            sample_df = sample_df.drop(columns=features_to_drop, errors='ignore')
            
            # Get the raw probabilities (logits)
            logits = np.log(loaded_final_model.predict_proba(sample_df) + 1e-8)
            
            # Apply temperature scaling to adjust probabilities
            y_pred_proba_scaled = loaded_temperature_scaler.transform(logits)
            
            # Store the probabilities for each classifier
            predictions[key] = {class_name: float(prob) for class_name, prob in zip(class_names, y_pred_proba_scaled[0])}
        
        return JsonResponse({'status': 'success', 'predictions': predictions})
    
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")  # Add logging
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)