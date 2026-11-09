"""FastAPI backend for the beginner house-price prototype."""
from contextlib import asynccontextmanager
import json
import joblib
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from houseprice_score import FEATURES, predict_house, predict_csv
from app.config import MODEL_PATH, API_KEY
import secrets


@asynccontextmanager
async def lifespan(app):
    app.state.saved = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title='INTELLIGENT HOUSE PRICE PREDICTION', lifespan=lifespan)


@app.exception_handler(Exception)
async def internal_error(request, error):
    logging.getLogger(__name__).error('Prediction service failure', exc_info=error)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error. Please try again later.'})


@app.get('/')
def home():
    return {'message': 'House price API is running'}


@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.get('/features')
def features():
    saved = app.state.saved
    return {'required_features': FEATURES, 'target': 'SalePrice', 'currency': 'USD',
            'defaults': saved['defaults'], 'categories': saved['categories']}


@app.post('/predict')
def predict(house: dict):
    try:
        price, warnings = predict_house(house, app.state.saved)
        return {'estimated_price_usd': price, 'warnings': warnings}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post('/predict/batch')
def batch(file: UploadFile = File(...)):
    try:
        result = predict_csv(file.file.read(10 * 1024 * 1024 + 1), app.state.saved)
        # Pandas converts missing prices to JSON null, avoiding invalid NaN JSON.
        return {'total': len(result), 'results': json.loads(result.to_json(orient='records'))}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        file.file.close()


@app.middleware('http')
async def check_api_key(request, call_next):
    if request.url.path.startswith('/predict') and API_KEY:
        if not secrets.compare_digest(request.headers.get('X-API-Key', ''), API_KEY):
            return JSONResponse(status_code=401, content={'detail': 'Invalid or missing API key.'})
    return await call_next(request)
