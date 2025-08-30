# -----------------------------------------------------------------------------
# Streamlit Prediction App
#
# Version: Final (Prediction-Only) with Ticker Dropdown
# -----------------------------------------------------------------------------

import streamlit as st
import os
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# --- App Configuration ---
st.set_page_config(page_title="Stock Hourly Predictor", layout="centered")

# --- Constants ---
TIME_STEP = 60  # Must match the model's training
MODEL_FILES = {
    "AAPL": "AAPL_hourly_lstm_model_tuned.keras",
    "TSLA": "TSLA_hourly_lstm_model_tuned.keras",
    "MSFT": "MSFT_hourly_lstm_model_tuned.keras",
    "GOOG" : "GOOG_hourly_lstm_model_tuned.keras",
    "AMZN" : "AMZN_hourly_lstm_model_tuned.keras",
    


}

# --- Caching Functions ---
@st.cache_resource
def load_keras_model(path):
    """Loads the Keras model from disk and caches it in memory."""
    try:
        model = load_model(path)
        return model
    except (IOError, ImportError):
        return None

@st.cache_data
def fetch_data(ticker):
    """
    Fetches historical hourly data to fit the scaler and get the latest sequence.
    Caches the returned dataframe.
    """
    try:
        data = yf.download(ticker, period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        return data if not data.empty else None
    except Exception:
        return None

# --- UI ---
st.title("📈 Stock Next Hour Price Predictor")
st.write("This app uses a pre-trained LSTM model to predict the stock price for the next trading hour.")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")

# Dropdown for ticker selection
TICKER = st.selectbox("Select Stock Symbol", list(MODEL_FILES.keys()))
MODEL_PATH = MODEL_FILES[TICKER]

# --- Prediction Button ---
if st.button("▶️ Predict Next Hour"):
    with st.spinner(f"Running prediction for {TICKER}..."):

        # 1. Load the pre-trained model
        model = load_keras_model(MODEL_PATH)
        if model is None:
            st.error(f"Fatal Error: Model file not found at '{MODEL_PATH}'.")
            st.stop()

        # 2. Fetch the latest data
        data = fetch_data(TICKER)
        if data is None:
            st.error(f"Fatal Error: Could not fetch data for {TICKER}.")
            st.stop()

        # 3. Prepare the Scaler
        close_data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close_data)

        # 4. Prepare last sequence
        last_sequence_df = close_data.tail(TIME_STEP)
        last_sequence = last_sequence_df.values
        last_sequence_scaled = scaler.transform(last_sequence)
        X_pred = np.array([last_sequence_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

        # 5. Predict
        pred_price_scaled = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]

        # 6. Calculate change & sentiment
        last_actual_price = close_data['Close'].iloc[-1]
        last_actual_timestamp = close_data.index[-1]
        change = predicted_price - last_actual_price
        percent_change = (change / last_actual_price) * 100
        sentiment = "Bullish 🐂" if change > 0 else "Bearish 🐻"

    # --- Display ---
    st.success("Prediction Complete!")
    st.subheader(f"{TICKER} Prediction Details")
    col1, col2 = st.columns(2)
    col1.metric(
        "Last Known Price",
        f"${last_actual_price:.2f}",
        help=f"The last price recorded at {last_actual_timestamp.strftime('%Y-%m-%d %H:%M %Z')}"
    )
    col2.metric(
        "Predicted Price (Next Hour)",
        f"${predicted_price:.2f}",
        f"{change:+.2f} USD ({percent_change:+.2f}%)"
    )

    if sentiment.startswith("Bullish"):
        st.info(f"**Sentiment for Next Hour: {sentiment}**", icon="📈")
    else:
        st.warning(f"**Sentiment for Next Hour: {sentiment}**", icon="📉")

    st.write("---")
    st.write(f"#### Last {TIME_STEP} Hours of Data Used for Prediction:")
    st.dataframe(last_sequence_df)
