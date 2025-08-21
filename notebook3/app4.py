import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime
import joblib
import os
import glob

# --- App Configuration ---
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# --- App Title and Description (Always display this) ---
st.title("📈 AutoML Stock Price Predictor")
st.markdown("""
This application uses an automated machine learning (AutoML) approach to predict the next day's stock closing price.
Enter a stock ticker in the sidebar and click 'Predict' to get started.
""")

# --- Helper Functions ---

def download_data(ticker, start_date, end_date):
    """Downloads and preprocesses stock data with clear feedback."""
    st.write(f"Attempting to download data for **{ticker}** from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        st.error(f"No data found for ticker '{ticker}'. It might be an invalid symbol or delisted.")
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].copy()
    df["Next_Close"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    st.success(f"Successfully downloaded and processed {len(df)} days of data for {ticker}.")
    return df

def train_and_evaluate(X, y):
    """Trains, evaluates, and returns the best model."""
    features = ["Open", "High", "Low", "Close"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), features)], remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    n_splits = 5
    if len(X) < n_splits * 2:
        st.warning(f"Not enough data for {n_splits}-fold cross-validation. Found {len(X)} data points.")
        return None, None, None
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}
    
    progress_bar = st.progress(0, text="Starting model training...")
    
    for i, (name, model) in enumerate(models.items()):
        progress_bar.progress((i) / len(models), text=f"Training and evaluating: {name}")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        mae_scores, r2_scores = [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        
        results[name] = {"MAE": np.mean(mae_scores), "R²": np.mean(r2_scores)}

    progress_bar.progress(1.0, text="Evaluation complete!")
    
    results_df = pd.DataFrame(results).T.sort_values(by="MAE")
    best_model_name = results_df.index[0]
    best_model = models[best_model_name]
    
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])
    final_pipeline.fit(X, y)
    
    return final_pipeline, best_model_name, results_df

def get_model_filename(ticker):
    """Generates a consistent model filename."""
    today_str = datetime.today().strftime('%Y%m%d')
    return f"{ticker}_{today_str}.pkl"

def save_model(model, filename):
    """Saves the model to a file."""
    joblib.dump(model, filename)
    st.success(f"Model saved as `{filename}`")

def load_model(ticker):
    """Loads a model for the given ticker if it exists for today."""
    filename = get_model_filename(ticker)
    if os.path.exists(filename):
        st.info(f"Loading pre-trained model from today: `{filename}`")
        # Clean up old models for the same ticker
        for old_file in glob.glob(f"{ticker}_*.pkl"):
            if old_file != filename:
                try:
                    os.remove(old_file)
                except OSError as e:
                    st.warning(f"Could not remove old model file {old_file}: {e}")
        return joblib.load(filename)
    return None

def main_prediction_flow(ticker):
    """Main function to orchestrate the prediction process."""
    st.header(f"Prediction for {ticker}")

    model_pipeline = load_model(ticker)
    model_name = "Loaded Model"
    results_df = None

    if model_pipeline is None:
        st.info(f"No model found for {ticker} for today. Starting new training process...")
        start_date = "2020-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = download_data(ticker, start_date, end_date)

        if data is not None:
            features = ["Open", "High", "Low", "Close"]
            X = data[features]
            y = data["Next_Close"]
            model_pipeline, model_name, results_df = train_and_evaluate(X, y)

            if model_pipeline:
                save_model(model_pipeline, get_model_filename(ticker))
    
    if model_pipeline is not None:
        if results_df is not None:
            st.subheader("Model Performance (Cross-Validation)")
            st.dataframe(results_df.style.highlight_min(subset=['MAE'], color='lightgreen').highlight_max(subset=['R²'], color='lightgreen'))
            st.success(f"🏆 Best Model Selected: **{model_name}** (based on lowest MAE)")
        
        st.subheader("Prediction")
        with st.spinner("Fetching latest market data for prediction..."):
            latest_data_df = yf.download(ticker, period="5d", progress=False)
            if latest_data_df.empty:
                st.error("Could not fetch latest data to make a prediction.")
                return

            if isinstance(latest_data_df.columns, pd.MultiIndex):
                latest_data_df.columns = latest_data_df.columns.get_level_values(0)

            latest_features = latest_data_df[["Open", "High", "Low", "Close"]].iloc[-1:]
            prediction = model_pipeline.predict(latest_features)[0]
            last_close = latest_features['Close'].iloc[0]
            change = prediction - last_close
            percent_change = (change / last_close) * 100
            
            st.metric(
                label=f"Predicted Close Price for Tomorrow",
                value=f"${prediction:,.2f}",
                delta=f"${change:,.2f} ({percent_change:.2f}%)"
            )
            st.write("Prediction based on this data:")
            st.dataframe(latest_features)
    else:
        st.error("Could not proceed with prediction. Please check the ticker and try again.")

# --- Streamlit UI ---
st.sidebar.header("User Input")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL")
predict_button = st.sidebar.button("Predict Tomorrow's Price", type="primary")

if predict_button:
    ticker = ticker_input.upper().strip()
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        # MASTER ERROR HANDLING BLOCK
        try:
            main_prediction_flow(ticker)
        except Exception as e:
            st.error("An unexpected error occurred during the process.")
            st.exception(e)
