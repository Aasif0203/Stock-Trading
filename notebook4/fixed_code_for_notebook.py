# ============================================================================
# FIXED CODE FOR NOTEBOOK - Copy this into your notebook cells
# ============================================================================

# CELL 1: Training Function (Copy this into your first cell)
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta

def train_hourly_model(ticker="AAPL", days_back=90):
    """
    Train an hourly prediction model for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days to look back for training data
    
    Returns:
        tuple: (model_filename, best_model_name, prediction)
    """
    
    # 1. Download hourly stock data
    end = datetime.today()
    start = end - timedelta(days=days_back)
    
    print(f"Downloading {ticker} hourly data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'), interval="60m")

    # Properly handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex, get the first level (actual column names)
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to get a datetime column
    df = df.reset_index()

    # Ensure we have the required columns and they are unique
    required_cols = ["Datetime", "Open", "High", "Low", "Close"]
    available_cols = df.columns.tolist()

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in available_cols]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {available_cols}")
        raise ValueError(f"Required columns {missing_cols} not found in data")

    # Use only OHLC columns
    df = df[required_cols]

    # 2. Create target = next hour's Close
    df["Next_Close"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    print(f"Training data shape: {df.shape}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

    # 3. Prepare features and target
    features = ["Open", "High", "Low", "Close"]
    X = df[features]
    y = df["Next_Close"]

    # Standardize the numeric features
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)],
                                     remainder="drop")

    # 4. Define candidate models
    lr    = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=30000)
    rf    = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    gb    = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    voting_reg = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)])
    weighted_voting_reg = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
                                         weights=[1,2,2])
    stacking_reg = StackingRegressor(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        final_estimator=Ridge(alpha=1.0)
    )
    models = {
        "Linear Regression": lr,
        "Ridge Regression": ridge,
        "Lasso Regression": lasso,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting": voting_reg,
        "Weighted Voting": weighted_voting_reg,
        "Stacking": stacking_reg
    }

    # 5. Evaluate each model with time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    for name, model in models.items():
        mae_scores, r2_scores = [], []
        pipeline = Pipeline(steps=[("scale", preprocessor), ("model", model)])
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        results[name] = {"MAE": np.mean(mae_scores), "R²": np.mean(r2_scores)}

    results_df = pd.DataFrame(results).T.sort_values(by="MAE")
    print("\nModel CV performance (lower MAE is better):")
    print(results_df)

    # 6. Select best model and retrain on all data
    best_model_name = results_df.index[0]
    print(f"\n✅ Best model: {best_model_name}")
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[("scale", preprocessor), ("model", best_model)])
    final_pipeline.fit(X, y)

    # 7. Save the trained model pipeline
    model_filename = f"{ticker}_hour.joblib"
    joblib.dump(final_pipeline, model_filename)
    print(f"💾 Saved model as {model_filename}")

    # 8. Predict next hour's closing price
    latest_features = df[features].iloc[-1:].copy()
    pred_next_hour = final_pipeline.predict(latest_features)[0]
    print(f"📈 Predicted next close (next hour) for {ticker}: {pred_next_hour:.2f}")
    
    return model_filename, best_model_name, pred_next_hour

# Train model for AAPL
model_file, best_model, prediction = train_hourly_model("AAPL", days_back=90)
print(f"\nSummary:")
print(f"Model file: {model_file}")
print(f"Best model: {best_model}")
print(f"Next hour prediction: ${prediction:.2f}")

# ============================================================================
# CELL 2: Prediction Function (Copy this into your second cell)
# ============================================================================

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

# Example usage - predict next hour for AAPL
results = predict_next_hour("AAPL")

if results:
    print_prediction_results(results)
else:
    print(f"\nTo train a model for AAPL, run the training cell above first.")

# ============================================================================
# CELL 3: Visualization (Copy this into your third cell)
# ============================================================================

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_hourly_predictions(ticker="AAPL", days_back=7):
    """
    Plot recent hourly data and predictions.
    """
    
    # Download recent data
    end = datetime.today()
    start = end - timedelta(days=days_back)
    
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'), interval="60m")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Create plot
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=(f'{ticker} Hourly OHLC', 'Volume'),
                        vertical_spacing=0.1)
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df['Datetime'],
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'],
                                  name='OHLC'),
                   row=1, col=1)
    
    # Volume chart
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df['Datetime'], y=df['Volume'], name='Volume'),
                       row=2, col=1)
    
    fig.update_layout(
        title=f'{ticker} Hourly Data (Last {days_back} Days)',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )
    
    fig.show()

# Plot recent hourly data
plot_hourly_predictions("AAPL", days_back=7)
