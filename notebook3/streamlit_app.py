
import os
import warnings
from datetime import datetime, date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Next-Day Stock Predictor", layout="wide")
st.title("📈 Next-Day Stock Predictor (AutoML)")
st.caption("Enter a stock ticker, load historical data, AutoML-train (if needed), and predict tomorrow's close.")

# For cleaner logs in the UI
warnings.filterwarnings("ignore")

# -------------------------------
# Trading Day Utilities
# -------------------------------
def get_next_trading_day(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the next trading day after dt (skips weekends)."""
    next_day = dt + pd.Timedelta(days=1)
    # Skip Saturday (5) and Sunday (6)
    while next_day.weekday() >= 5:
        next_day += pd.Timedelta(days=1)
    return next_day

# -------------------------------
# Sidebar Inputs
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_start = date(2020, 1, 1)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=date.today())
    ticker = st.text_input("Ticker (e.g., ADBE, AAPL, GOOG)", value="ADBE").upper().strip()
    train_button = st.button("🚀 Load & Train (if needed)")

# Note: Models are saved/loaded as {ticker}.joblib

# -------------------------------
# Utility: Data Download (cached)
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLC data and build Next_Close target."""
    if not ticker:
        return pd.DataFrame()
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    # Flatten possible MultiIndex columns
    try:
        df.columns = df.columns.get_level_values(0)
    except Exception:
        pass
    # Keep OHLC & Date
    df = df.loc[:, ["Open", "High", "Low", "Close"]].reset_index()
    # Target: next day's close
    df["Next_Close"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------------
# Model Zoo (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_models():
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=30000)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
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

    return {
        "Linear Regression": lr,
        "Ridge Regression": ridge,
        "Lasso Regression": lasso,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting": voting_reg,
        "Weighted Voting": weighted_voting_reg,
        "Stacking": stacking_reg,
    }

# -------------------------------
# Training / Evaluation
# -------------------------------
def evaluate_and_select_model(X: pd.DataFrame, y: pd.Series, features: list) -> tuple[pd.DataFrame, str, Pipeline]:
    models = get_models()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), features)],
        remainder="drop",
    )

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, model in models.items():
        mae_scores, r2_scores = [], []
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        results[name] = {"MAE": np.mean(mae_scores), "R²": np.mean(r2_scores)}

    results_df = pd.DataFrame(results).T.sort_values(by="MAE", na_position="last")
    best_model_name = results_df.dropna().index[0]

    # Fit best pipeline on full data
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    final_pipeline.fit(X, y)

    return results_df, best_model_name, final_pipeline

# -------------------------------
# Main App Logic
# -------------------------------
def main():
    colA, colB = st.columns([3, 2])

    if not ticker:
        colA.warning("Please enter a ticker symbol.")
        return

    # Always fetch data through today's date for training/prediction
    effective_end = date.today()

    # Load data
    with st.spinner("Downloading historical data…"):
        df = load_data(ticker, str(start_date), str(effective_end))
        # Also fetch raw OHLC including the latest available day for correct last close display
        raw_end_plus_one = pd.to_datetime(effective_end) + pd.Timedelta(days=1)
        raw = yf.download(
            ticker,
            start=str(start_date),
            end=str(raw_end_plus_one.date()),
            progress=False,
            auto_adjust=False,
        )
        try:
            raw.columns = raw.columns.get_level_values(0)
        except Exception:
            pass
        raw = raw.loc[:, ["Open", "High", "Low", "Close"]].reset_index()

    if df.empty or len(df) < 60 or raw.empty:
        colA.error("No data (or too little data). Try a different ticker or wider date range.")
        return

    features = ["Open", "High", "Low", "Close"]
    X = df[features]
    y = df["Next_Close"]

    model_path = f"{ticker}.joblib"

    # Load or Train (auto-train if missing; button forces retrain)
    results_df = None
    best_model_name = None
    final_pipeline = None

    if os.path.exists(model_path) and not train_button:
        try:
            final_pipeline = joblib.load(model_path)
            st.success(f"Loaded saved model: {os.path.basename(model_path)}")
        except Exception:
            final_pipeline = None

    if final_pipeline is None:
        with st.spinner("Training models (AutoML)…"):
            results_df, best_model_name, final_pipeline = evaluate_and_select_model(X, y, features)
            joblib.dump(final_pipeline, model_path)
            if train_button:
                st.success(f"Model retrained and saved as {os.path.basename(model_path)}")
            else:
                st.success(f"Model trained and saved as {os.path.basename(model_path)}")

    # Current (last close) and Prediction
    # Use the most recent raw OHLC row to display last close and to predict the next trading day's close
    latest_features = raw[features].iloc[-1:]
    next_day_prediction = float(final_pipeline.predict(latest_features)[0])
    last_close = float(raw["Close"].iloc[-1])
    last_date = pd.to_datetime(raw["Date"].iloc[-1])
    next_date = get_next_trading_day(last_date)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close", f"{last_close:,.2f}")
    col2.metric("Predicted Next Close", f"{next_day_prediction:,.2f}", f"{next_day_prediction - last_close:,.2f}")
    if best_model_name:
        col3.metric("Best Model (today)", best_model_name)
    else:
        col3.metric("Best Model (today)", "Loaded model")

    # Optional: show evaluation table if we just trained
    if results_df is not None:
        st.subheader("Model Performance (Cross-Validation)")
        st.dataframe(results_df.style.format({"MAE": "{:.4f}", "R²": "{:.4f}"}), use_container_width=True)

    # Chart: historical close + predicted point
    st.subheader(f"{ticker} — Last Close and Predicted Next Close")
    plot_df = raw[["Date", "Close"]].copy()
    pred_row = pd.DataFrame({"Date": [next_date], "Close": [next_day_prediction]})
    plot_df = pd.concat([plot_df, pred_row], ignore_index=True)

    # Use Plotly for smooth UI
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], mode="lines", name="Close"))
    fig.add_trace(
        go.Scatter(x=[next_date], y=[next_day_prediction], mode="markers+text", name="Predicted",
                   text=["Predicted"], textposition="top center")
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Raw data expander
    with st.expander("Show raw data"):
        st.dataframe(raw.tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
