"""Small helper shared by the Streamlit form and the tests."""
from pathlib import Path
import math
from io import StringIO
import csv
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NUMERIC = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
           'YearBuilt', 'FullBath', 'LotArea', 'OverallCond']
CATEGORIES = ['Neighborhood', 'KitchenQual']
FEATURES = NUMERIC + CATEGORIES
LABELS = ['Overall quality (1-10)', 'Living area (sq ft)', 'Basement area (sq ft)',
          'Garage car capacity', 'Year built', 'Full bathrooms above ground',
          'Lot area (sq ft)', 'Overall condition (1-10)']
LIMITS = [(1, 10), (100, 20000), (0, 20000), (0, 10),
          (1800, 2010), (0, 10), (100, 1000000), (1, 10)]

def validate_house(values, saved):
    """Check the inputs, predict log(price), then convert back to dollars."""
    clean = {}
    warnings = []
    for name, limits in zip(NUMERIC, LIMITS):
        try:
            raw = values.get(name)
            if raw is None or isinstance(raw, bool):
                raise ValueError()
            number = float(raw)
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError()
            if not limits[0] <= number <= limits[1]:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f'{name}: enter a whole number from {limits[0]} to {limits[1]}.') from None
        clean[name] = int(number)
        low, high = saved['ranges'][name]
        if number < low or number > high:
            warnings.append(f'{name} is outside the training range ({low:g}-{high:g}).')
    for name in CATEGORIES:
        category = values.get(name)
        if category not in saved['categories'][name]:
            raise ValueError(f'{name}: choose one of the listed categories.')
        clean[name] = category
    return clean, warnings

def predict_house(values, saved):
    clean, warnings = validate_house(values, saved)
    row = pd.DataFrame([clean], columns=FEATURES)
    log_price = saved['model'].predict(row)[0]
    price = float(np.exp(log_price))
    if not math.isfinite(price) or price <= 0:
        raise ValueError('A valid price could not be calculated.')
    return round(price, 2), warnings

def predict_csv(file_bytes, saved):
    """Validate rows individually and predict valid houses in one model call."""
    if len(file_bytes) > 10*1024*1024:
        raise ValueError('Upload a CSV no larger than 10 MB.')
    try:
        text = file_bytes.decode('utf-8-sig')
        reader = csv.reader(StringIO(text), strict=True)
        header = [name.strip() for name in next(reader)]
        if len(header) != len(set(header)):
            raise ValueError('Column names must be unique.')
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError('Every CSV row must match the header field count.')
        data = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeDecodeError, StopIteration, csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError):
        raise ValueError('Upload a readable UTF-8 CSV with a header and at least one house.') from None
    data.columns = data.columns.str.strip()
    missing = [name for name in FEATURES if name not in data.columns]
    if missing:
        raise ValueError('Missing required columns: '+', '.join(missing))
    if not 1 <= len(data) <= 10000:
        raise ValueError('The file must contain 1 to 10,000 houses.')
    outputs = ['EstimatedPriceUSD', 'PredictionStatus', 'PredictionMessage']
    if any(name in data.columns for name in outputs):
        raise ValueError('Remove old prediction columns before uploading again.')
    result = data.copy()
    result['EstimatedPriceUSD'] = np.nan
    result['PredictionStatus'] = 'error'
    result['PredictionMessage'] = ''
    valid, positions = [], []
    for position, row in data.iterrows():
        try:
            clean, warnings = validate_house({name: row[name].strip() for name in FEATURES}, saved)
            valid.append(clean)
            positions.append(position)
            result.loc[position, 'PredictionStatus'] = 'warning' if warnings else 'ok'
            result.loc[position, 'PredictionMessage'] = '; '.join(warnings)
        except ValueError as error:
            result.loc[position, 'PredictionMessage'] = str(error)
    if valid:
        prices = np.exp(saved['model'].predict(pd.DataFrame(valid, columns=FEATURES)))
        for position, price in zip(positions, prices):
            if math.isfinite(price) and price > 0:
                result.loc[position, 'EstimatedPriceUSD'] = round(float(price), 2)
            else:
                result.loc[position, 'PredictionStatus'] = 'error'
                result.loc[position, 'PredictionMessage'] = 'Unable to calculate a valid price.'
    return result
