import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- Utils ----------
def next_business_day(date_obj: datetime) -> str:
    d = date_obj + timedelta(days=1)
    # skip weekends
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")

def fetch_ohlc(ticker: str, start="2015-01-01") -> pd.DataFrame:
    end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}.")
    # flatten potential multi-index
    df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ["Open", "High", "Low", "Close"]].reset_index()
    return df

def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # target = next day's Close; features = today's OHLC
    df["Next_Close"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    features = ["Open", "High", "Low", "Close"]
    X = df[features]
    y = df["Next_Close"]
    return X, y, features

def get_models():
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=10000)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    voting = VotingRegressor(estimators=[("lr", lr), ("rf", rf), ("gb", gb)])
    stacking = StackingRegressor(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        final_estimator=Ridge(alpha=1.0)
    )
    return {
        "Linear Regression": lr,
        "Ridge Regression": ridge,
        "Lasso Regression": lasso,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting": voting,
        "Stacking": stacking
    }

def evaluate_and_select(X: pd.DataFrame, y: pd.Series, features: list[str]):
    preprocessor = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")
    models = get_models()
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    pipelines = {}

    for name, model in models.items():
        mae_scores, r2_scores = [], []
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        results[name] = {"MAE": float(np.mean(mae_scores)), "R2": float(np.mean(r2_scores))}
        pipelines[name] = pipe

    # select best by MAE
    best_name = min(results.keys(), key=lambda k: results[k]["MAE"])
    best_pipe = Pipeline([("prep", preprocessor), ("model", get_models()[best_name])])
    best_pipe.fit(X, y)  # retrain on ALL data

    return best_name, best_pipe, results

def model_paths(ticker: str):
    base = os.path.join(MODELS_DIR, f"{ticker.upper()}")
    return base + ".joblib", base + ".meta.json"

def save_model(ticker: str, pipeline: Pipeline, metadata: dict):
    model_path, meta_path = model_paths(ticker)
    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

def load_model(ticker: str):
    model_path, meta_path = model_paths(ticker)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    pipe = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return pipe, meta

def train_ml(ticker: str, start="2015-01-01"):
    df = fetch_ohlc(ticker, start)
    X, y, features = build_dataset(df)
    best_name, best_pipe, results = evaluate_and_select(X, y, features)

    last_date = pd.to_datetime(df["Date"].iloc[-1])
    prediction_date = next_business_day(last_date.to_pydatetime())

    meta = {
        "ticker": ticker.upper(),
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_samples": int(len(X)),
        "features": features,
        "best_model": best_name,
        "cv_results": results,
        "last_data_date": last_date.strftime("%Y-%m-%d"),
        "prediction_date": prediction_date
    }
    save_model(ticker, best_pipe, meta)
    return best_pipe, meta

def predict_next_close(ticker: str):
    pipe, meta = load_model(ticker)
    # If no model yet, train it
    if pipe is None:
        pipe, meta = train_ml(ticker)

    # Always fetch fresh data up to today for the latest features
    df = fetch_ohlc(ticker)
    X, _, features = build_dataset(df)

    latest_features = X.iloc[[-1]]  # last row as frame
    pred = float(pipe.predict(latest_features)[0])

    # ensure prediction date is day after df last date (business-day aware)
    last_date = pd.to_datetime(df["Date"].iloc[-1]).to_pydatetime()
    prediction_date = next_business_day(last_date)

    response = {
        "ticker": ticker.upper(),
        "prediction_date": prediction_date,
        "predicted_close": round(pred, 2),
        "best_model": meta["best_model"],
        "trained_at": meta["trained_at"],
        "last_data_date": pd.to_datetime(df["Date"].iloc[-1]).strftime("%Y-%m-%d")
    }
    return response
