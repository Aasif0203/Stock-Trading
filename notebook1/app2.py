# -----------------------------------------------------------------------------
# Streamlit App: Multi-Stock Predictor for Tomorrow's Opening Price
#
# Version: Final (Multi-Stock, Tomorrow's Open)
# This app includes a dropdown to select a stock (AAPL or MSFT) and manages
# separate pre-trained HOURLY models to forecast the OPENING price of the
# next trading day for the selected stock.
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
from datetime import date, timedelta

# --- App Configuration ---
st.set_page_config(page_title="Tomorrow's Open Predictor", layout="centered")

# --- Model and Data Constants ---
TIME_STEP = 60  # The sequence length used for model training

# --- Helper Function ---
def get_next_trading_day(today):
    """Calculates the next trading day, skipping weekends."""
    if today.weekday() < 4: return today + timedelta(days=1)
    else: return today + timedelta(days=(7 - today.weekday()))

# --- Caching Functions for Performance ---
@st.cache_resource
def get_or_train_model(ticker, x_train, y_train):
    """
    Loads a ticker-specific model if it exists. If not, it trains, saves,
    and returns a new standard LSTM model.
    """
    MODEL_PATH = f"{ticker}_hourly_lstm_model_tuned.keras" # e.g., AAPL_hourly.model

    if os.path.exists(MODEL_PATH):
        st.write(f"Loading pre-trained model for `{ticker}`...")
        model = load_model(MODEL_PATH)
    else:
        st.write(f"No pre-trained model found for `{ticker}`. Training a new model...")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)), Dropout(0.2),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(25, activation='relu'), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stop], verbose=0)
        model.save(MODEL_PATH)
        st.success(f"New model for `{ticker}` trained and saved as `{MODEL_PATH}`")
        
    return model

@st.cache_data
def fetch_and_process_data(ticker):
    """Fetches and prepares all necessary data for a given ticker."""
    try:
        data = yf.download(ticker, period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        if data.empty: return None

        close_data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(close_data)
        
        # Prepare training data needed for the get_or_train_model function
        scaled_dataset = scaler.transform(close_data)
        x_train, y_train = [], []
        for i in range(TIME_STEP, len(scaled_dataset)):
            x_train.append(scaled_dataset[i-TIME_STEP:i, 0])
            y_train.append(scaled_dataset[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return {"scaler": scaler, "data": data, "x_train": x_train, "y_train": y_train}
    except Exception:
        return None

# --- Main App Interface ---
st.title("📈 Tomorrow's Open Price Predictor")

# --- User Input Dropdown ---
selected_ticker = st.selectbox(
    "Select Stock Ticker",
    ["AAPL", "MSFT", "GOOG","AMZN", "NVDA"],
    help="The app trains and saves a separate model for each stock."
)

st.write(f"This app will use a pre-trained hourly LSTM model to forecast the opening price for **{selected_ticker}**.")

if st.button(f"Predict Tomorrow's Opening Price for {selected_ticker}"):
    with st.spinner(f"Fetching data and preparing model for {selected_ticker}..."):
        
        # 1. Fetch and process all data
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

    # 3. Prepare the most recent sequence for prediction
    with st.spinner("Making final prediction..."):
        close_data = data[['Close']]
        last_sequence = close_data.tail(TIME_STEP).values
        last_sequence_scaled = scaler.transform(last_sequence)
        X_pred = np.array([last_sequence_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

        # 4. Make the prediction
        pred_price_scaled = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(pred_price_scaled)[0][0]

    # 5. Display the results
    st.success("Prediction Complete!")
    last_actual_price = close_data['Close'].iloc[-1]
    last_actual_timestamp = close_data.index[-1]
    change = predicted_price - last_actual_price
    percent_change = (change / last_actual_price) * 100
    sentiment = "Bullish 🐂" if change > 0 else "Bearish 🐻"
    
    next_day = get_next_trading_day(last_actual_timestamp.date())
    
    st.subheader(f"Forecast for {selected_ticker} on {next_day.strftime('%A, %B %d, %Y')}")
    col1, col2 = st.columns(2)
    col1.metric(
        "Last Known Closing Price",
        f"${last_actual_price:.2f}",
        help=f"Closing price on {last_actual_timestamp.strftime('%Y-%m-%d')}"
    )
    col2.metric(
        "Predicted Opening Price",
        f"${predicted_price:.2f}",
        f"{change:+.2f} USD ({percent_change:+.2f}%)"
    )

    if sentiment == "Bullish 🐂":
        st.info(f"**Sentiment for Market Open: {sentiment}**", icon="📈")
    else:
        st.warning(f"**Sentiment for Market Open: {sentiment}**", icon="📉")