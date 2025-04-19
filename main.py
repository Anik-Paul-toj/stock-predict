from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Return Predictor API")

class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int

# Your original ML functions
def fetch_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

def engineer_features(df, investment_days):
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df[f'Target_{investment_days}d'] = df['Close'].shift(-investment_days) / df['Close'] - 1
    df.dropna(inplace=True)
    return df

def train_model(df, investment_days):
    features = ['Return', 'MA_5', 'MA_10', 'Volatility']
    target = f'Target_{investment_days}d'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Model RMSE: {rmse:.4f}")
    return model, df

def assess_risk(predicted_return):
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
            "data_source": "real market data"
        }

        logger.info(f"Response: {response}")
        return response

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unhandled exception during prediction")
        raise HTTPException(status_code=500, detail="Internal Server Error")
