import numpy as np
from scipy.optimize import minimize

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