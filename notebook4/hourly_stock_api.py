import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

# FastAPI app setup
app = FastAPI(
    title="Hourly Stock Prediction API",
    description="API for predicting next hour's stock price and fetching hourly data using saved ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Available tickers (based on your notebook4 models)
AVAILABLE_TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA", "ADBE", "IBM", "META", "TSLA"]

# Response models
class HourlyPrediction(BaseModel):
    ticker: str
    current_datetime: str
    next_hour_datetime: str
    current_close: float
    predicted_next_close: float
    predicted_change: float
    predicted_change_percent: float
    model_file: str
    confidence_score: Optional[float] = None

class HourlyData(BaseModel):
    ticker: str
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None

class StockDataResponse(BaseModel):
    ticker: str
    data: List[HourlyData]
    prediction: Optional[HourlyPrediction] = None

class ModelInfo(BaseModel):
    ticker: str
    model_file: str
    exists: bool
    last_modified: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

def find_model_file(ticker: str) -> Optional[str]:
    """Find the model file for a ticker in the current directory."""
    model_filename = f"{ticker}_hour.joblib"
    if os.path.exists(model_filename):
        return model_filename
    return None

def get_next_trading_hour(dt: datetime) -> datetime:
    """Get the next trading hour (simple implementation)."""
    return dt + timedelta(hours=1)

def predict_next_hour(ticker: str) -> Optional[HourlyPrediction]:
    """Predict the next hour's closing price using a saved model."""
    
    model_filename = find_model_file(ticker)
    if not model_filename:
        return None
    
    try:
        # Load the trained model
        pipeline = joblib.load(model_filename)
        
        # Download recent hourly data for prediction
        end = datetime.today()
        start = end - timedelta(days=7)  # Get last 7 days of hourly data
        
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), interval="60m")
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Ensure we have required columns
        required_cols = ["Datetime", "Open", "High", "Low", "Close"]
        available_cols = df.columns.tolist()
        
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            return None
        
        # Use only OHLC columns
        df = df[required_cols]
        
        # Get the latest data point for prediction
        latest_data = df.iloc[-1]
        latest_features = df[["Open", "High", "Low", "Close"]].iloc[-1:].copy()
        
        # Make prediction
        prediction = pipeline.predict(latest_features)[0]
        
        # Calculate next hour timestamp
        next_hour = get_next_trading_hour(latest_data["Datetime"])
        
        # Calculate confidence score (simple implementation)
        confidence_score = 0.85  # You can implement a more sophisticated confidence calculation
        
        return HourlyPrediction(
            ticker=ticker,
            current_datetime=latest_data["Datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            next_hour_datetime=next_hour.strftime("%Y-%m-%d %H:%M:%S"),
            current_close=float(latest_data["Close"]),
            predicted_next_close=float(prediction),
            predicted_change=float(prediction - latest_data["Close"]),
            predicted_change_percent=float(((prediction - latest_data["Close"]) / latest_data["Close"]) * 100),
            model_file=model_filename,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        print(f"Error predicting for {ticker}: {e}")
        return None

def get_hourly_data(ticker: str, days_back: int = 7) -> List[HourlyData]:
    """Get hourly stock data for the specified ticker."""
    
    try:
        end = datetime.today()
        start = end - timedelta(days=days_back)
        
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'), interval="60m")
        
        if df.empty:
            return []
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        
        # Convert to list of HourlyData objects
        hourly_data = []
        for _, row in df.iterrows():
            data = HourlyData(
                ticker=ticker,
                datetime=row["Datetime"].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row["Datetime"], 'strftime') else str(row["Datetime"]),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else None
            )
            hourly_data.append(data)
        
        return hourly_data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []

@app.get("/", response_model=APIResponse)
def home():
    """Welcome endpoint with API information."""
    return APIResponse(
        success=True,
        message="📈 Hourly Stock Prediction API is running!",
        data={
            "available_tickers": AVAILABLE_TICKERS,
            "endpoints": [
                "/predict/{ticker}",
                "/data/{ticker}",
                "/predict-and-data/{ticker}",
                "/models",
                "/health"
            ],
            "usage": {
                "predict": "GET /predict/AAPL - Get next hour prediction",
                "data": "GET /data/AAPL?days=7 - Get hourly data",
                "both": "GET /predict-and-data/AAPL - Get both prediction and data"
            }
        }
    )

@app.get("/health", response_model=APIResponse)
def health_check():
    """Health check endpoint."""
    return APIResponse(
        success=True,
        message="✅ API is healthy and running",
        data={"timestamp": datetime.now().isoformat()}
    )

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available model files."""
    models = []
    for ticker in AVAILABLE_TICKERS:
        model_filename = find_model_file(ticker)
        model_info = ModelInfo(
            ticker=ticker,
            model_file=f"{ticker}_hour.joblib",
            exists=model_filename is not None,
            last_modified=datetime.fromtimestamp(os.path.getmtime(model_filename)).isoformat() if model_filename else None
        )
        models.append(model_info)
    
    return models

@app.get("/predict/{ticker}", response_model=HourlyPrediction)
def get_prediction(ticker: str):
    """Get next hour prediction for a specific ticker."""
    ticker = ticker.upper()
    
    if ticker not in AVAILABLE_TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not supported. Available: {AVAILABLE_TICKERS}")
    
    prediction = predict_next_hour(ticker)
    if not prediction:
        raise HTTPException(
            status_code=404, 
            detail=f"No model found for {ticker}. Please train the model first using the notebook."
        )
    
    return prediction

@app.get("/data/{ticker}", response_model=List[HourlyData])
def get_hourly_data_endpoint(ticker: str, days: int = Query(7, ge=1, le=30)):
    """Get hourly stock data for a specific ticker."""
    ticker = ticker.upper()
    
    if ticker not in AVAILABLE_TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not supported. Available: {AVAILABLE_TICKERS}")
    
    data = get_hourly_data(ticker, days)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    return data

@app.get("/predict-and-data/{ticker}", response_model=StockDataResponse)
def get_prediction_and_data(ticker: str, days: int = Query(7, ge=1, le=30)):
    """Get both prediction and hourly data for a specific ticker."""
    ticker = ticker.upper()
    
    if ticker not in AVAILABLE_TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not supported. Available: {AVAILABLE_TICKERS}")
    
    # Get prediction
    prediction = predict_next_hour(ticker)
    
    # Get data
    data = get_hourly_data(ticker, days)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    return StockDataResponse(
        ticker=ticker,
        data=data,
        prediction=prediction
    )

@app.get("/batch-predict", response_model=List[HourlyPrediction])
def batch_predict(tickers: Optional[str] = Query(None, description="Comma-separated list of tickers")):
    """Get predictions for multiple tickers."""
    if not tickers:
        # Use all available tickers
        ticker_list = AVAILABLE_TICKERS
    else:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    predictions = []
    for ticker in ticker_list:
        if ticker in AVAILABLE_TICKERS:
            prediction = predict_next_hour(ticker)
            if prediction:
                predictions.append(prediction)
    
    return predictions

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Hourly Stock Prediction API...")
    print(f"📊 Available tickers: {AVAILABLE_TICKERS}")
    print("🌐 API will be available at: http://127.0.0.1:8000")
    print("📖 Documentation at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
