from django.apps import AppConfig


class PredictionModelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction_model'
    def ready(self):
        # Import the TemperatureScaling class when the app is ready
        from .utils import TemperatureScaling
