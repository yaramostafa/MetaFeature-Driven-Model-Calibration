from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

app = Flask(__name__)

# Temperature Scaling Class
class TemperatureScaling:
    def __init__(self):
        self.temperature = None

    def fit(self, logits, y_true):
        def loss_fn(T):
            scaled_logits = logits / T
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
            log_likelihood = -np.log(probs[np.arange(len(y_true)), y_true])
            return np.mean(log_likelihood)

        result = minimize(loss_fn, x0=np.ones(1), bounds=[(0.1, 10)])
        self.temperature = result.x[0]

    def transform(self, logits):
        scaled_logits = logits / self.temperature
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        return probs

# Load the saved model and temperature scaler
loaded_model_and_scaler = joblib.load('final_model_with_temp_scaling.pkl')
loaded_final_model = loaded_model_and_scaler['model']
loaded_temperature_scaler = loaded_model_and_scaler['temperature_scaler']

# List of features to drop
features_to_drop = [
    'Stddev No. Of Symbols per Categorical Features', 
    'Stddev No. Of Significant Lags in Target', 
    'Stddev No. Of Insignificant Lags in Target', 
    'Stddev No. Of Seasonality Components in Target'
]

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input features from JSON
    input_data = request.get_json()

    # Prepare the predictions dictionary
    predictions = {}

    # List of classifiers (you may want to adjust based on your actual class names)
    class_names = ['ELASTICNETCV', 'HUBERREGRESSOR', 'LASSO', 'LinearSVR', 'QUANTILEREGRESSOR', 'XGBRegressor']

    for key, sample in input_data.items():
        # Convert the sample to a DataFrame for processing
        sample_df = pd.DataFrame([sample])

        # Drop unwanted features
        sample_df = sample_df.drop(columns=features_to_drop, errors='ignore')

        # Get the raw probabilities (logits)
        logits = np.log(loaded_final_model.predict_proba(sample_df) + 1e-8)  # Add epsilon for numerical stability

        # Apply temperature scaling to adjust probabilities
        y_pred_proba_scaled = loaded_temperature_scaler.transform(logits)

        # Store the probabilities for each classifier
        predictions[key] = {class_name: prob for class_name, prob in zip(class_names, y_pred_proba_scaled[0])}

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
