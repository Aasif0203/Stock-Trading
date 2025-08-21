# graph_predict_api.py

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# CORS for frontend (e.g., React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def model_predict(df):
    features = ["Open", "High", "Low", "Close"]
    X = df[features]
    y = df["Next_Close"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), features)],
        remainder="drop"
    )
    # Candidate models
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=30000)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    voting_reg = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)])
    weighted_voting_reg = VotingRegressor(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        weights=[1, 2, 2]
    )
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
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    for name, model in models.items():
        mae_scores = []
        for train_idx, test_idx in tscv.split(X):
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae_scores.append(np.mean(np.abs(y_test - y_pred)))
        results[name] = np.mean(mae_scores)
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    final_pipeline.fit(X, y)
    latest_features = df[features].iloc[-1:]
    next_day_prediction = float(final_pipeline.predict(latest_features)[0])
    return next_day_prediction, best_model_name, final_pipeline

def plot_stock(df, next_day_prediction):
    # Plot the Close price and predicted next-day close
    plt.figure(figsize=(9,4))
    plt.plot(df["Date"], df["Close"], label="Close Price")
    plt.scatter(df["Date"].iloc[-1] + pd.Timedelta(days=1), next_day_prediction, color="red", label="Predicted Next Close")
    plt.legend()
    plt.title("Close Price and Predicted Next Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_base64

@app.get("/")
def home():
    return {"message": "📈 Stock Price Prediction API is running!", "endpoints": ["/predict/{ticker}"]}

@app.get("/predict/{ticker}")
def get_stock_prediction(ticker: str, start: str = "2020-01-01", end: str = None):
    try:
        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')
        
        # Download stock data
        df = yf.download(ticker, start=start, end=end)
        if df.empty or len(df) < 10:
            return JSONResponse(
                content={"error": f"No data found or insufficient data for ticker {ticker}."}, 
                status_code=400
            )
        
        # Process data
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ["Open", "High", "Low", "Close"]].reset_index()
        df["Next_Close"] = df["Close"].shift(-1)
        df = df.dropna().reset_index(drop=True)
        
        # Make prediction
        next_day_prediction, best_model_name, final_pipeline = model_predict(df)
        
        # Get last close and date
        last_close = float(df["Close"].iloc[-1])
        last_date = df["Date"].iloc[-1].strftime("%Y-%m-%d") if isinstance(df["Date"].iloc[-1], pd.Timestamp) else str(df["Date"].iloc[-1])
        
        # Generate plot
        try:
            img_base64 = plot_stock(df, next_day_prediction)
        except Exception as plot_error:
            print(f"Plot error: {plot_error}")
            img_base64 = None
        
        # Return response
        return {
            "ticker": ticker.upper(),
            "last_close": round(last_close, 2),
            "last_date": last_date,
            "predicted_next_close": round(next_day_prediction, 2),
            "best_model": best_model_name,
            "plot_base64": img_base64
        }
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error processing request: {str(e)}"}, 
            status_code=500
        )

