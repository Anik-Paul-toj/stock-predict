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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Return Predictor API")

class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int

CACHE_DIR = "stock_data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_path(ticker: str, period: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.csv")

def is_cache_valid(cache_path: str, max_age_hours: int = 24) -> bool:
    if not os.path.exists(cache_path):
        return False
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    return age < timedelta(hours=max_age_hours)

def fetch_stock_data(ticker: str, period="5y", max_retries=3) -> pd.DataFrame:
    cache_path = get_cache_path(ticker, period)
    if is_cache_valid(cache_path):
        try:
            logger.info(f"Using cached data for {ticker}")
            df = pd.read_csv(cache_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if not df.empty:
                df.reset_index(inplace=True)
                df.to_csv(cache_path, index=False)
                return df
        except Exception as e:
            logger.warning(f"Ticker method failed: {e}")

        try:
            df = yf.download(ticker, period=period)
            if not df.empty:
                df.reset_index(inplace=True)
                df.to_csv(cache_path, index=False)
                return df
        except Exception as e:
            logger.warning(f"Download method failed: {e}")

        retry_count += 1
        time.sleep(2 ** retry_count)

    logger.warning("Falling back to synthetic data")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252)
    close_prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))

    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'synthetic': True
    })

    df.to_csv(os.path.join(CACHE_DIR, "sample_data.csv"), index=False)
    return df

def engineer_features(df: pd.DataFrame, investment_days: int) -> pd.DataFrame:
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df[f'Target_{investment_days}d'] = df['Close'].shift(-investment_days) / df['Close'] - 1
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Insufficient data after feature engineering. Try a shorter investment period or different stock.")

    return df

def train_model(df: pd.DataFrame, investment_days: int):
    features = ['Return', 'MA_5', 'MA_10', 'Volatility']
    target = f'Target_{investment_days}d'

    X = df[features]
    y = df[target]

    if X.empty or y.empty:
        raise ValueError("Training data is empty after processing.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"Trained model RMSE: {rmse:.4f}")

    return model, df

def assess_risk(predicted_return: float) -> str:
    if predicted_return > 0.2:
        return "High Gain (High Risk)"
    elif predicted_return > 0.05:
        return "Moderate Gain (Medium Risk)"
    elif predicted_return > -0.05:
        return "Low Gain or Neutral (Low Risk)"
    else:
        return "Potential Loss (High Risk)"

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Stock Return Predictor API 🚀",
        "endpoints": {
            "POST /predict": "Send a ticker, investment amount, and investment days to get a return prediction."
        }
    }

@app.post("/predict")
def predict_return(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request: {request}")
        if request.investment_amount <= 0:
            raise ValueError("Investment amount must be positive.")
        if request.investment_days <= 0:
            raise ValueError("Investment days must be positive.")

        ticker = request.ticker.strip().upper()
        df = fetch_stock_data(ticker, period="5y")

        using_real_data = 'synthetic' not in df.columns

        df = engineer_features(df, request.investment_days)
        model, df = train_model(df, request.investment_days)

        latest_data = df.iloc[-1][['Return', 'MA_5', 'MA_10', 'Volatility']].values.reshape(1, -1)
        predicted_return = model.predict(latest_data)[0]
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
            "data_source": "real market data" if using_real_data else "synthetic demonstration data"
        }

        logger.info(f"Prediction completed: {response}")
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unhandled error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
