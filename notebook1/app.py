import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# --- Constants ---
TIME_STEP = 60 # Must match the training time_step
MODEL_FILENAME = 'lgbm_model.joblib'

# --- Model and Scaler Loading ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the generic saved model and scaler."""
    if os.path.exists(MODEL_FILENAME):
        data = joblib.load(MODEL_FILENAME)
        return data['model'], data['scaler']
    else:
        return None, None

# --- Data Fetching ---
@st.cache_data(ttl=900) # Cache data for 15 minutes
def fetch_data(ticker):
    """Fetches historical daily and recent hourly data for a ticker."""
    try:
        end_date = datetime.now()
        # Fetch enough daily data to have a buffer
        daily_start_date = end_date - timedelta(days=150)
        # Fetch recent hourly data
        intraday_start_date = end_date - timedelta(days=59)

        daily_data = yf.download(ticker, start=daily_start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)
        intraday_data = yf.download(ticker, start=intraday_start_date, end=end_date, interval='1h', auto_adjust=True, progress=False)

        if daily_data.empty and intraday_data.empty:
            return None

        # Combine, making sure to keep the more granular intraday data
        combined_data = pd.concat([daily_data, intraday_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data.sort_index(inplace=True)
        return combined_data[['Close']]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Main App ---
st.title("📈 Generalized Stock Price Predictor")
st.markdown("Predicts the next price point for selected tech stocks using a single LightGBM model trained on combined daily and hourly data from all stocks.")
st.markdown("---")

# --- Load Model ---
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error(f"**Model not found!** Please ensure the file `{MODEL_FILENAME}` is in the same directory as this app. You must run the Jupyter Notebook first to train and save the model.")
else:
    # --- Sidebar for User Input ---
    st.sidebar.header("Select Stock")
    ticker = st.sidebar.selectbox(
        "Choose a stock ticker:",
        ('MSFT', 'GOOGL', 'CSCO') # Dropdown list
    )
    predict_button = st.sidebar.button("Predict Next Price", type="primary")

    st.header(f"Prediction for {ticker}")

    if predict_button:
        with st.spinner(f"Fetching latest data for {ticker}..."):
            close_prices = fetch_data(ticker)

        if close_prices is None or len(close_prices) < TIME_STEP:
            st.error(f"Could not fetch sufficient data for {ticker}. Need at least {TIME_STEP} data points.")
        else:
            st.success(f"Latest data for {ticker} fetched successfully!")

            # --- Prediction Logic ---
            # 1. Scale the data using the loaded scaler
            scaled_data = scaler.transform(close_prices)

            # 2. Prepare the last sequence for prediction
            last_sequence = scaled_data[-TIME_STEP:]
            input_data = last_sequence.reshape(1, -1)

            # 3. Predict and inverse transform
            predicted_price_scaled = model.predict(input_data)
            prediction = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

            # 4. Determine prediction date/time
            last_date = close_prices.index[-1]
            # Since we have hourly data, the next prediction is for the next interval
            prediction_date = last_date + timedelta(hours=1) 
            
            # --- Display Results ---
            col1, col2 = st.columns(2)
            
            last_close_price = close_prices['Close'].iloc[-1]
            price_change = prediction - last_close_price
            percent_change = (price_change / last_close_price) * 100
            
            with col1:
                st.metric(
                    label=f"Predicted Price for next interval ({prediction_date.strftime('%Y-%m-%d %H:%M')})",
                    value=f"${prediction:.2f}",
                    delta=f"{price_change:.2f} ({percent_change:.2f}%)"
                )
            with col2:
                 st.metric(
                    label=f"Last Known Price ({last_date.strftime('%Y-%m-%d %H:%M')})",
                    value=f"${last_close_price:.2f}"
                )

            # --- Visualization ---
            st.subheader("Price History & Prediction")
            fig, ax = plt.subplots(figsize=(14, 7))
            # Plot last 300 data points for better context
            ax.plot(close_prices.index[-300:], close_prices['Close'][-300:], label='Historical Close Price')
            ax.plot(prediction_date, prediction, 'ro', markersize=8, label=f'Predicted Price: ${prediction:.2f}')
            
            ax.set_title(f'{ticker} - Price Prediction', fontsize=16)
            ax.set_xlabel('Date and Time')
            ax.set_ylabel('Price (USD)')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            
            st.pyplot(fig)

    else:
        st.info("Select a stock from the sidebar and click 'Predict' to start.")