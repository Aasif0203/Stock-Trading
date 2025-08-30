import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta

def predict_next_hour(ticker, model_filename=None):
    """
    Predict the next hour's closing price using a saved model.
    
    Args:
        ticker (str): Stock ticker symbol
        model_filename (str): Path to the saved model file. If None, uses default naming.
    
    Returns:
        dict: Prediction results
    """
    
    # Use default model filename if not provided
    if model_filename is None:
        model_filename = f"{ticker}_hour.joblib"
    
    try:
        # Load the trained model
        pipeline = joblib.load(model_filename)
        print(f"✅ Loaded model: {model_filename}")
    except FileNotFoundError:
        print(f"❌ Model file {model_filename} not found. Please train the model first.")
        return None
    
    # Download recent hourly data for prediction
    end = datetime.today()
    start = end - timedelta(days=7)  # Get last 7 days of hourly data
    
    print(f"Downloading recent {ticker} hourly data...")
    
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
        print(f"❌ Missing columns: {missing_cols}")
        return None
    
    # Use only OHLC columns
    df = df[required_cols]
    
    # Get the latest data point for prediction
    latest_data = df.iloc[-1]
    latest_features = df[["Open", "High", "Low", "Close"]].iloc[-1:].copy()
    
    # Make prediction
    prediction = pipeline.predict(latest_features)[0]
    
    # Calculate next hour timestamp
    next_hour = latest_data["Datetime"] + timedelta(hours=1)
    
    # Prepare results
    results = {
        "ticker": ticker,
        "current_datetime": latest_data["Datetime"],
        "next_hour_datetime": next_hour,
        "current_close": latest_data["Close"],
        "predicted_next_close": prediction,
        "predicted_change": prediction - latest_data["Close"],
        "predicted_change_percent": ((prediction - latest_data["Close"]) / latest_data["Close"]) * 100,
        "model_file": model_filename
    }
    
    return results

def print_prediction_results(results):
    """Pretty print the prediction results."""
    if results is None:
        return
    
    print("\n" + "="*50)
    print(f"📈 HOURLY PREDICTION FOR {results['ticker']}")
    print("="*50)
    print(f"Current Time: {results['current_datetime']}")
    print(f"Next Hour:    {results['next_hour_datetime']}")
    print(f"Current Close: ${results['current_close']:.2f}")
    print(f"Predicted Close: ${results['predicted_next_close']:.2f}")
    
    change = results['predicted_change']
    change_pct = results['predicted_change_percent']
    
    if change > 0:
        print(f"Predicted Change: +${change:.2f} (+{change_pct:.2f}%) 📈")
    else:
        print(f"Predicted Change: ${change:.2f} ({change_pct:.2f}%) 📉")
    
    print(f"Model: {results['model_file']}")
    print("="*50)

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    
    # Try to predict next hour
    results = predict_next_hour(ticker)
    
    if results:
        print_prediction_results(results)
    else:
        print(f"\nTo train a model for {ticker}, run:")
        print(f"python fixed_hourly_prediction.py")
