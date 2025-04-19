from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import os
import time
from datetime import datetime, timedelta

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI(title="Stock Return Predictor API")

# Input model
class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int

# Cache directory
CACHE_DIR = "stock_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(ticker: str, period: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.csv")

def is_cache_valid(path: str, max_age_hours: int = 24) -> bool:
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=max_age_hours)

def fetch_stock_data(ticker: str, period="5y") -> pd.DataFrame:
    cache_path = get_cache_path(ticker, period)

    if is_cache_valid(cache_path):
        try:
            logger.info(f"Using cached data for {ticker}")
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")

    try:
        logger.info(f"Fetching data using yf.download for {ticker}")
        df = yf.download(ticker, period=period, progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df.to_csv(cache_path, index=False)
            return df
    except Exception as e:
        logger.warning(f"yf.download failed: {e}")

    logger.warning("Using synthetic fallback data.")
    dates = pd.date_range(end=datetime.today(), periods=252)
    prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1e6, 1e7, len(dates)),
        'synthetic': True
    })
    df.to_csv(os.path.join(CACHE_DIR, "synthetic_data.csv"), index=False)
    return df

def engineer_features(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Volatility'] = df['Return'].rolling(10).std()
    df[f'Target_{days}d'] = df['Close'].shift(-days) / df['Close'] - 1
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Not enough data after processing. Try shorter investment days.")

    return df

def train_model(df: pd.DataFrame, days: int):
    features = ['Return', 'MA_5', 'MA_10', 'Volatility']
    target = f'Target_{days}d'

    X, y = df[features], df[target]
    if X.empty or y.empty:
        raise ValueError("Insufficient data for training.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"Model trained with RMSE: {rmse:.4f}")
    return model, df

def assess_risk(pred_return: float) -> str:
    if pred_return > 0.2:
        return "High Gain (High Risk)"
    elif pred_return > 0.05:
        return "Moderate Gain (Medium Risk)"
    elif pred_return > -0.05:
        return "Low Gain or Neutral (Low Risk)"
    else:
        return "Potential Loss (High Risk)"

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Stock Return Predictor API 🚀",
        "usage": "POST /predict with ticker, investment_amount, and investment_days."
    }

@app.post("/predict")
def predict_return(request: PredictionRequest):
    try:
        logger.info(f"Received request: {request}")
        if request.investment_amount <= 0 or request.investment_days <= 0:
            raise ValueError("Investment amount and days must be greater than zero.")

        ticker = request.ticker.strip().upper()
        df = fetch_stock_data(ticker)
        using_real_data = 'synthetic' not in df.columns

        df = engineer_features(df, request.investment_days)
        model, df = train_model(df, request.investment_days)

        latest = df.iloc[-1][['Return', 'MA_5', 'MA_10', 'Volatility']].values.reshape(1, -1)
        predicted_return = model.predict(latest)[0]
        predicted_value = request.investment_amount * (1 + predicted_return)
        risk = assess_risk(predicted_return)

        response = {
            "ticker": ticker,
            "investment_amount": request.investment_amount,
            "investment_days": request.investment_days,
            "predicted_return_percent": round(predicted_return * 100, 2),
            "predicted_value": round(predicted_value, 2),
            "risk_level": risk,
            "data_points_used": len(df),
            "data_source": "real market data" if using_real_data else "synthetic fallback data"
        }

        logger.info(f"Response: {response}")
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unhandled exception during prediction")
        raise HTTPException(status_code=500, detail="Internal Server Error")
