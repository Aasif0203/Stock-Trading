# -----------------------------------------------------------------------------
# Streamlit Multi-Stock Prediction App
#
# Version: 7.0 (Multi-Stock)
# This app can train, save, and load separate models for different stocks.
# It uses a dropdown to select the ticker and manages models like
# 'AAPL_hourly.model', 'MSFT_hourly.model', etc.
# -----------------------------------------------------------------------------

import streamlit as st
import os
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# --- App Configuration ---
st.set_page_config(page_title="Multi-Stock Predictor", layout="centered")

# --- Model and Data Constants ---
TIME_STEP = 60  # The sequence length used for model training

# --- Caching Functions for Performance ---

@st.cache_resource
def get_or_train_model(ticker, x_train, y_train):
    """
    Loads a ticker-specific model if it exists. If not, it trains, saves,
    and returns a new standard LSTM model.
    """
    MODEL_PATH = f"{ticker}_hourly_lstm_model_tuned.keras"

    if os.path.exists(MODEL_PATH):
        st.write(f"Loading pre-trained model for `{ticker}`...")
        model = load_model(MODEL_PATH)
    else:
        st.write(f"No pre-trained model found for `{ticker}`. Training a new model...")
        
        # Define a standard, robust LSTM architecture
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        # verbose=0 in fit hides the epoch-by-epoch training log in the UI
        model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stop], verbose=0)
        
        model.save(MODEL_PATH)
        st.success(f"New model for `{ticker}` trained and saved as `{MODEL_PATH}`")
        
    return model

@st.cache_data
def fetch_and_process_data(ticker):
    """
    Fetches, processes, and prepares data for a given ticker.
    Returns all necessary components: scaler, datasets, etc.
    """
    try:
        data = yf.download(ticker, period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if data.empty: return None

        close_data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit scaler on the entire dataset to capture its full range
        scaler.fit(close_data)
        # Transform the dataset for training
        scaled_dataset = scaler.transform(close_data)

        training_data_len = int(np.ceil(len(scaled_dataset) * 0.8))
        train_data = scaled_dataset[0:training_data_len, :]
        x_train, y_train = [], []
        for i in range(TIME_STEP, len(train_data)):
            x_train.append(train_data[i-TIME_STEP:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return {
            "scaler": scaler,
            "data": data,
            "x_train": x_train,
            "y_train": y_train
        }

    except Exception:
        return None

# --- Main App Interface ---
st.title("📈 Multi-Stock Next Hour Predictor")

# --- User Input ---
selected_ticker = st.selectbox(
    "Select Stock Ticker",
    ["AAPL", "MSFT","GOOG","TSLA","AMZN"],
    help="The app will train and save a separate model for each ticker."
)

if st.button(f"Predict Next Hour for {selected_ticker}"):
    with st.spinner(f"Fetching data and preparing model for {selected_ticker}..."):
        
        # 1. Fetch and process data
        processed_data = fetch_and_process_data(selected_ticker)
        if processed_data is None:
            st.error(f"Could not fetch or process data for {selected_ticker}.")
            st.stop()
        
        # Unpack processed data
        scaler = processed_data["scaler"]
        data = processed_data["data"]
        x_train = processed_data["x_train"]
        y_train = processed_data["y_train"]

        # 2. Load or train the specific model for the chosen ticker
        model = get_or_train_model(selected_ticker, x_train, y_train)

    # 3. Prepare latest sequence for prediction
    close_data = data[['Close']]
    last_sequence = close_data.tail(TIME_STEP).values
    last_sequence_scaled = scaler.transform(last_sequence)
    X_pred = np.array([last_sequence_scaled])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    # 4. Make the prediction
    with st.spinner("Making final prediction..."):
        pred_price_scaled = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]

    # 5. Display the results
    st.success("Prediction Complete!")
    last_actual_price = close_data['Close'].iloc[-1]
    last_actual_timestamp = close_data.index[-1]
    change = predicted_price - last_actual_price
    percent_change = (change / last_actual_price) * 100
    sentiment = "Bullish 🐂" if change > 0 else "Bearish 🐻"
    
    st.subheader(f"Prediction for {selected_ticker}")
    col1, col2 = st.columns(2)
    col1.metric(
        "Last Known Price",
        f"${last_actual_price:.2f}",
        help=f"Last price at {last_actual_timestamp.strftime('%Y-%m-%d %H:%M %Z')}"
    )
    col2.metric(
        "Predicted Price (Next Hour)",
        f"${predicted_price:.2f}",
        f"{change:+.2f} USD ({percent_change:+.2f}%)"
    )
    st.info(f"**Sentiment for Next Hour: {sentiment}**", icon="🧭")