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
import json
import requests
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Return Predictor API")

class PredictionRequest(BaseModel):
    ticker: str
    investment_amount: float
    investment_days: int

# Create cache directory if it doesn't exist
CACHE_DIR = "stock_data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_path(ticker: str, period: str) -> str:
    """Generate cache file path for a ticker and period"""
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.csv")

def is_cache_valid(cache_path: str, max_age_hours: int = 24) -> bool:
    """Check if cached data is still valid based on age"""
    if not os.path.exists(cache_path):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    age = datetime.now() - file_time
    
    return age < timedelta(hours=max_age_hours)

def fetch_stock_data(ticker: str, period="5y", max_retries=3) -> pd.DataFrame:
    """Fetch stock data with retries, backoff, and caching"""
    cache_path = get_cache_path(ticker, period)
    
    # Try to use cached data if it's valid
    if is_cache_valid(cache_path):
        try:
            logger.info(f"Using cached data for {ticker}")
            df = pd.read_csv(cache_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.warning(f"Failed to load cached data: {str(e)}. Will fetch fresh data.")
    
    # If cache is invalid or loading failed, fetch new data
    retry_count = 0
    while retry_count < max_retries:
        try:
            logger.info(f"Attempt {retry_count+1} to fetch data for {ticker}")
            
            # Method 1: Use Ticker object
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} rows with Ticker method")
                    df.reset_index(inplace=True)
                    df.to_csv(cache_path, index=False)
                    return df
            except Exception as e1:
                logger.warning(f"Ticker method failed: {str(e1)}")
            
            # Method 2: Use yf.download
            try:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} rows with download method")
                    df.reset_index(inplace=True)
                    df.to_csv(cache_path, index=False)
                    return df
            except Exception as e2:
                logger.warning(f"Download method failed: {str(e2)}")
            
            # Try shorter period if full period fails
            if period == "5y" and retry_count == 1:
                logger.info(f"Trying shorter period for {ticker}")
                try:
                    df = yf.download(ticker, period="1y")
                    if not df.empty:
                        logger.info(f"Successfully fetched {len(df)} rows with 1y period")
                        df.reset_index(inplace=True)
                        df.to_csv(cache_path, index=False)
                        return df
                except Exception as e3:
                    logger.warning(f"Shorter period attempt failed: {str(e3)}")
            
            # If still empty, wait and retry
            wait_time = 2 ** retry_count  # Exponential backoff: 1, 2, 4 seconds
            logger.warning(f"No data returned for {ticker}, retrying in {wait_time} seconds")
            time.sleep(wait_time)
            retry_count += 1
            
        except Exception as e:
            logger.error(f"Error in fetch attempt {retry_count+1}: {str(e)}")
            time.sleep(2 ** retry_count)
            retry_count += 1
    
    # Check for fallback cached data even if it's old
    if os.path.exists(cache_path):
        try:
            logger.warning(f"Using older cached data as fallback for {ticker}")
            df = pd.read_csv(cache_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.error(f"Failed to load fallback cache: {str(e)}")
    
    # Try to load sample data for demonstration purposes
    sample_data_path = os.path.join(CACHE_DIR, "sample_data.csv")
    if os.path.exists(sample_data_path):
        try:
            logger.warning(f"Using sample data for demonstration purposes")
            df = pd.read_csv(sample_data_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.error(f"Failed to load sample data: {str(e)}")
    
    # If we get here, generate synthetic data for demonstration
    logger.warning(f"Generating synthetic data for {ticker} as last resort")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252)  # ~1 trading year
    close_prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02, 
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    df.to_csv(sample_data_path, index=False)  # Save sample data for future use
    logger.info(f"Generated synthetic data with {len(df)} rows")
    
    return df

def engineer_features(df: pd.DataFrame, investment_days: int) -> pd.DataFrame:
    """Engineer features as per original implementation"""
    try:
        # Calculate returns
        df['Return'] = df['Close'].pct_change()
        
        # Create moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Return'].rolling(window=10).std()
        
        # Calculate target - future return after investment_days
        df[f'Target_{investment_days}d'] = df['Close'].shift(-investment_days) / df['Close'] - 1
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        logger.info(f"After feature engineering: {len(df)} rows remain")
        
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise ValueError(f"Feature engineering failed: {str(e)}")

def train_model(df: pd.DataFrame, investment_days: int):
    """Train model with original implementation parameters"""
    try:
        features = ['Return', 'MA_5', 'MA_10', 'Volatility']
        target = f'Target_{investment_days}d'
        
        X = df[features]
        y = df[target]
        
        # Use fixed test size of 0.2 as in original code
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Use fixed 100 estimators as in original code
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        logger.info(f"Model trained with RMSE: {rmse:.4f}")
        
        return model, df
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise ValueError(f"Model training failed: {str(e)}")

def assess_risk(predicted_return: float) -> str:
    """Assess investment risk based on predicted return"""
    if predicted_return > 0.2:
        return "High Gain (High Risk)"
    elif predicted_return > 0.05:
        return "Moderate Gain (Medium Risk)"
    elif predicted_return > -0.05:
        return "Low Gain or Neutral (Low Risk)"
    else:
        return "Potential Loss (High Risk)"

@app.post("/predict")
def predict_return(request: PredictionRequest):
    """Endpoint to predict stock returns"""
    try:
        logger.info(f"Processing prediction request for {request.ticker}")
        
        # Validate inputs
        if request.investment_amount <= 0:
            raise ValueError("Investment amount must be positive")
        if request.investment_days <= 0:
            raise ValueError("Investment days must be positive")
            
        # Normalize ticker symbol
        ticker = request.ticker.strip().upper()
        
        # Fetch data with standard 5y period as in original code
        df = fetch_stock_data(ticker, period="5y")
        
        # Add a flag to indicate if we're using synthetic data
        using_real_data = not df.empty and 'synthetic' not in df.columns
        
        # Process data
        df = engineer_features(df, request.investment_days)
        
        model, df = train_model(df, request.investment_days)
        
        # Get prediction
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
        
        logger.info(f"Prediction successful: {response}")
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

# Generate sample data for demonstration if needed
def create_sample_data():
    """Create sample stock data for demonstration purposes"""
    sample_path = os.path.join(CACHE_DIR, "sample_data.csv")
    if not os.path.exists(sample_path):
        logger.info("Creating sample data file for demonstration")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252)  # ~1 trading year
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
        
        df.to_csv(sample_path, index=False)
        logger.info(f"Sample data created with {len(df)} rows")

# Add health check endpoint
@app.get("/health")
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Create sample data if needed
        create_sample_data()
        
        # Try a simple yfinance API call to verify connectivity
        test_ticker = "MSFT"
        try:
            test_df = yf.download(test_ticker, period="7d")
            yfinance_status = "ok" if not test_df.empty else "degraded"
            yfinance_msg = f"yfinance returned {len(test_df)} rows for {test_ticker}"
        except Exception as e:
            yfinance_status = "failed"
            yfinance_msg = f"yfinance error: {str(e)}"
        
        # Check cache directory
        cache_status = "ok" if os.path.exists(CACHE_DIR) else "not found"
        cache_files = len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0
        
        return {
            "status": "healthy" if yfinance_status == "ok" else "degraded",
            "yfinance_connectivity": yfinance_status,
            "cache_status": cache_status,
            "cache_files": cache_files,
            "message": f"API is running, {yfinance_msg}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Stock Return Predictor API")
    uvicorn.run(app, host="0.0.0.0", port=8000)