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

if __name__ == "__main__":
    # Train model for AAPL
    model_file, best_model, prediction = train_hourly_model("AAPL", days_back=90)
    print(f"\nSummary:")
    print(f"Model file: {model_file}")
    print(f"Best model: {best_model}")
    print(f"Next hour prediction: ${prediction:.2f}")
