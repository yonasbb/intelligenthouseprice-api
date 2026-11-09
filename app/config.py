"""Deployment configuration comes from environment variables."""
import os
from pathlib import Path

MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'model_artifacts/houseprice_model.joblib'))
API_KEY = os.environ.get('API_KEY', '')
PORT = int(os.environ.get('PORT', '8000'))
if not 1 <= PORT <= 65535:
    raise ValueError('PORT must be between 1 and 65535.')
