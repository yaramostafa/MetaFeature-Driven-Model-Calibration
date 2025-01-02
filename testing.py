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

# Load models
loaded_final_model = joblib.load('final_model.pkl')
loaded_temperature_scaler = joblib.load('temperature_scaler.pkl')

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

    # List of classifiers
    class_names = ['ELASTICNETCV', 'HUBERREGRESSOR', 'LASSO', 'LinearSVR', 'QUANTILEREGRESSOR', 'XGBRegressor']

    for key, sample in input_data.items():
        # Convert the sample to a DataFrame for processing
        sample_df = pd.DataFrame([sample])

        # Drop unwanted features
        sample_df = sample_df.drop(columns=features_to_drop, errors='ignore')

        # Apply the model
        logits = np.log(loaded_final_model.predict_proba(sample_df) + 1e-8)

        # Apply temperature scaling
        y_pred_proba_scaled = loaded_temperature_scaler.transform(logits)

        # Store the probabilities for each classifier
        predictions[key] = {class_name: prob for class_name, prob in zip(class_names, y_pred_proba_scaled[0])}

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
