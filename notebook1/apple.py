# -----------------------------------------------------------------------------
# Stock Prophet: Final Streamlit Application
#
# Version: 4.0 (Robust)
# This script incorporates all fixes, including the critical one for handling
# multi-level column indices returned by yfinance. It uses a simplified
# load-or-train model strategy without Keras Tuner for faster execution.
# -----------------------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import os

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Scikit-learn import
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- App Configuration and Title ---
st.set_page_config(page_title="Stock Prophet", layout="wide")
st.title("🔮 Stock Prophet: Price Prediction & Analysis")
st.write("""
Welcome to Stock Prophet. This application loads a pre-trained model or trains a standard one if not found.
- **Daily Model**: Predicts the closing price for the next trading day.
- **Hourly Model**: Predicts the opening price for the next trading day.
""")
st.info("**Pro Tip:** If the app ever behaves unexpectedly after a code change, try clearing the cache via the (☰) menu in the top-right corner.")

# --- Sidebar for User Input ---
st.sidebar.header("User Controls")
TICKER = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
PREDICTION_MODE = st.sidebar.selectbox(
    "Select Prediction Mode",
    ["Predict Next Day's CLOSE (Daily Model)", "Predict Next Day's OPEN (Hourly Model)"]
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(ticker, period, interval):
    """
    Fetches stock data from Yahoo Finance, flattens multi-level columns if
    they exist, and caches the result. This is the robust, corrected version.
    """
    st.write(f"Fetching {interval} data for {ticker}...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    # --- CRITICAL FIX FOR MULTI-INDEX COLUMNS ---
    if isinstance(data.columns, pd.MultiIndex):
        st.write("Multi-level columns detected. Flattening to single-level.")
        data.columns = data.columns.get_level_values(0)
    # --------------------------------------------

    if data.empty:
        return None
        
    data.dropna(inplace=True)
    return data

@st.cache_resource
def get_or_train_model(ticker, mode, x_train, y_train):
    """
    Loads a pre-trained model from disk or trains a new standard model.
    The trained model object is cached in memory for the session.
    """
    is_hourly = (mode == "Hourly")
    model_name_suffix = "hourly" if is_hourly else "daily"
    MODEL_PATH = f"{ticker}_{model_name_suffix}.keras"

    if os.path.exists(MODEL_PATH):
        st.write(f"Loading pre-trained model `{MODEL_PATH}` from disk...")
        model = load_model(MODEL_PATH)
    else:
        st.write(f"No pre-trained model found at `{MODEL_PATH}`. Training a new standard model...")
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stop], verbose=0)
        
        model.save(MODEL_PATH)
        st.success(f"New model trained and saved as `{MODEL_PATH}`")
        
    return model

# --- Main App Logic ---

is_hourly_mode = "Hourly" in PREDICTION_MODE
period, interval, time_step = ("730d", "1h", 60) if is_hourly_mode else ("10y", "1d", 60)

data = load_data(TICKER, period, interval)

if data is None:
    st.error(f"Could not load data for {TICKER}. Please check the ticker symbol.")
else:
    st.subheader(f"Raw {interval} Data for {TICKER}")
    st.write(data.tail())

    if st.button("🚀 Start Analysis and Prediction"):
        with st.spinner("Processing data and preparing model..."):
            
            close_data = data.filter(['Close'])

            if close_data.empty:
                st.error(f"Fatal Error: Could not find the 'Close' column in the downloaded data for {TICKER}. "
                         f"Please check the data source. Available columns might be: {data.columns.tolist()}")
                st.stop()

            dataset = close_data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_dataset = scaler.fit_transform(dataset)

            training_data_len = int(np.ceil(len(scaled_dataset) * 0.8))
            train_data = scaled_dataset[0:training_data_len, :]
            x_train, y_train = [], []
            for i in range(time_step, len(train_data)):
                x_train.append(train_data[i-time_step:i, 0])
                y_train.append(train_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model_mode = "Hourly" if is_hourly_mode else "Daily"
            model = get_or_train_model(TICKER, model_mode, x_train, y_train)

        st.subheader("Model Evaluation")
        test_data = scaled_dataset[training_data_len - time_step:, :]
        x_test, y_test = [], dataset[training_data_len:, :]
        for i in range(time_step, len(test_data)):
            x_test.append(test_data[i-time_step:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        st.metric("Test Set Root Mean Squared Error (RMSE)", f"${rmse:.2f}")

        st.subheader("Prediction vs. Actual Price (Test Set)")
        train = close_data[:training_data_len]
        valid = close_data[training_data_len:].copy()
        valid['Predictions'] = predictions
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title(f'Model Prediction vs Actual ({model_mode} Data)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price USD ($)')
        ax.plot(train['Close'], label='Train')
        ax.plot(valid[['Close', 'Predictions']])
        ax.legend(['Train', 'Actual', 'Predicted'], loc='lower right')
        st.pyplot(fig)

        st.subheader("Final Prediction")
        last_sequence = close_data[-time_step:].values
        last_sequence_scaled = scaler.transform(last_sequence)
        X_pred = np.array([last_sequence_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

        pred_price = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price)[0][0]
        
        last_actual_price = data['Close'].iloc[-1]
        last_actual_timestamp = data.index[-1]
        change = predicted_price - last_actual_price
        percent_change = (change / last_actual_price) * 100
        sentiment = "Bullish 🐂" if change > 0 else "Bearish 🐻"
        
        col1, col2 = st.columns(2)
        col1.metric("Last Known Price", f"${last_actual_price:.2f}", f"at {last_actual_timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        if is_hourly_mode:
            next_day_str = "Next Day's OPEN"
            col2.metric(f"Predicted Price ({next_day_str})", f"${predicted_price:.2f}", f"{change:.2f} USD ({percent_change:.2f}%)")
            st.info(f"Sentiment for Market Open: **{sentiment}**", icon="📈")
            st.caption("Note: This is a prediction for the start of the next trading day, NOT the final closing price.")
        else:
            next_day_str = "Next Day's CLOSE"
            col2.metric(f"Predicted Price ({next_day_str})", f"${predicted_price:.2f}", f"{change:.2f} USD ({percent_change:.2f}%)")
            st.info(f"Sentiment for Next Trading Day: **{sentiment}**", icon="📅")