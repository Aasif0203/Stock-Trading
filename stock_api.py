import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

# FastAPI app
app = FastAPI(title="Stock Prediction API", description="API for predicting stock prices")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Settings
TICKERS = ["AAPL", "GOOG", "MSFT", "NVDA", "AMZN", "ADBE", "IBM", "META", "TSLA"]
FEATURES = ["Open", "High", "Low", "Close"]
MODELS_DIR = Path(".")

# Request models
class StockRequest(BaseModel):
    ticker: str
    start_date: str = "2020-01-01"
    end_date: str = None

class PredictionResponse(BaseModel):
    ticker: str
    last_close: float
    last_date: str
    predicted_next_close: float
    change: float
    change_percent: float
    model_used: str

def find_model_file(ticker: str, models_dir: Path) -> Path:
    """Find the model file for a ticker, handling different naming patterns."""
    # Try exact match first
    exact_path = models_dir / f"{ticker}.joblib"
    if exact_path.exists():
        return exact_path
    
    # Try with date suffix (most recent)
    pattern = f"{ticker}_*.joblib"
    matching_files = list(models_dir.glob(pattern))
    if matching_files:
        # Return the most recent file (by modification time)
        return max(matching_files, key=lambda x: x.stat().st_mtime)
    
    raise FileNotFoundError(f"No model file found for ticker {ticker}")

def load_model(path: Path):
    """Load a trained model from file."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

def fetch_latest_ohlc(ticker: str, start_date: str = "2020-01-01", end_date: str = None) -> pd.DataFrame:
    """Download recent daily data and return the latest row with OHLC."""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise ValueError("No price data returned from yfinance.")

    # Ensure standard columns and drop rows with missing OHLC
    cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = cols
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        raise ValueError("Price data missing required OHLC columns.")

    return df

@app.get("/")
def home():
    """Welcome endpoint."""
    return {
        "message": "📈 Stock Prediction API is running!",
        "available_tickers": TICKERS,
        "endpoints": [
            "/predict/{ticker}",
            "/predict",
            "/models"
        ]
    }

@app.get("/models")
def list_models():
    """List all available model files."""
    models = {}
    for ticker in TICKERS:
        available_models = []
        exact_path = MODELS_DIR / f"{ticker}.joblib"
        if exact_path.exists():
            available_models.append(exact_path.name)
        
        pattern = f"{ticker}_*.joblib"
        matching_files = list(MODELS_DIR.glob(pattern))
        available_models.extend([f.name for f in matching_files])
        
        if available_models:
            models[ticker] = available_models
    
    return {"available_models": models}

@app.get("/predict/{ticker}")
def predict_stock(ticker: str, start_date: str = "2020-01-01", end_date: str = None):
    """Predict next day's stock price for a given ticker."""
    try:
        ticker = ticker.upper()
        
        # Find and load model
        model_path = find_model_file(ticker, MODELS_DIR)
        pipeline = load_model(model_path)
        
        # Fetch latest data
        df = fetch_latest_ohlc(ticker, start_date, end_date)
        
        # Get latest features for prediction
        latest_features = df[FEATURES].iloc[-1:].copy()
        
        # Make prediction
        prediction = float(pipeline.predict(latest_features)[0])
        
        # Get current data
        last_close = float(df["Close"].iloc[-1])
        last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
        
        # Calculate changes
        change = prediction - last_close
        change_percent = (change / last_close) * 100 if last_close else 0.0
        
        return PredictionResponse(
            ticker=ticker,
            last_close=round(last_close, 2),
            last_date=last_date,
            predicted_next_close=round(prediction, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            model_used=model_path.name
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict")
def predict_stock_post(request: StockRequest):
    """Predict next day's stock price using POST request."""
    return predict_stock(request.ticker, request.start_date, request.end_date)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
