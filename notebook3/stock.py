import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path

st.set_page_config(page_title="Next-Day Stock Close • Pretrained Models", page_icon="📈", layout="centered")

st.title("📈 Next-Day Stock Prediction (Pretrained)")
st.caption("Select a ticker. We'll load the saved model (`<TICKER>.joblib`), fetch the latest OHLC, and predict tomorrow's close.")

# -------------------------------
# Settings
# -------------------------------
TICKERS = ["AAPL", "GOOG", "MSFT", "NVDA", "AMZN", "ADBE", "IBM", "META", "TSLA"]
FEATURES = ["Open", "High", "Low", "Close"]
MODELS_DIR = Path(".")  # change if your models are elsewhere

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
    ticker = st.selectbox("Choose a ticker", TICKERS, index=0)
    
    # Show available models for the selected ticker
    available_models = []
    exact_path = MODELS_DIR / f"{ticker}.joblib"
    if exact_path.exists():
        available_models.append(exact_path.name)
    
    pattern = f"{ticker}_*.joblib"
    matching_files = list(MODELS_DIR.glob(pattern))
    available_models.extend([f.name for f in matching_files])
    
    if available_models:
        st.write("Available models:")
        for model in available_models:
            st.write(f"• {model}")
    else:
        st.warning(f"No models found for {ticker}")
    
    run_btn = st.button("🔮 Predict Tomorrow's Close")

# -------------------------------
# Helper functions
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_latest_ohlc(ticker: str) -> pd.DataFrame:
    """Download recent daily data and return the latest row with OHLC in a single-row DataFrame."""
    # Pull a small recent window to be safe across weekends/holidays
    df = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No price data returned from yfinance.")

    # Ensure standard columns and drop rows with missing OHLC
    cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = cols
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        raise ValueError("Price data missing required OHLC columns.")

    latest_row = df.iloc[[-1]][FEATURES]
    latest_row.index = [df.index[-1]]  # Preserve the date
    return latest_row


def find_model_file(ticker: str, models_dir: Path) -> Path:
    """Find the model file for a ticker, handling different naming patterns."""
    # Try exact match first
    exact_path = models_dir / f"{ticker}.joblib"
    if exact_path.exists():
        return exact_path
    
    # Try with date suffix (most recent)
    pattern = f"{ticker}_*.joblib"
    matching_files = list(models_dir.glob(pattern))
    if matching_files:
        # Return the most recent file (by modification time)
        return max(matching_files, key=lambda x: x.stat().st_mtime)
    
    raise FileNotFoundError(f"No model file found for ticker {ticker}")

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")

# -------------------------------
# Main Action
# -------------------------------
if run_btn:
    with st.spinner("Loading model and fetching latest data…"):
        try:
            model_path = find_model_file(ticker, MODELS_DIR)
            st.info(f"Using model: {model_path.name}")
            pipeline = load_model(model_path)
            latest_features = fetch_latest_ohlc(ticker)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Predict next close
    try:
        pred = float(pipeline.predict(latest_features[FEATURES])[0])
    except Exception as e:
        st.error(f"Prediction failed. Ensure the saved model expects features {FEATURES}.\nDetails: {e}")
        st.stop()

    last_trade_date = latest_features.index[-1]
    last_close = float(latest_features["Close"].iloc[-1])
    change = pred - last_close
    pct = (change / last_close) * 100 if last_close else 0.0

    st.subheader(f"{ticker} • Prediction")
    st.write(f"Last market day: **{last_trade_date.date()}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close", f"{last_close:,.2f}")
    c2.metric("Predicted Next Close", f"{pred:,.2f}", delta=f"{change:,.2f}")
    c3.metric("Change %", f"{pct:,.2f}%")

    with st.expander("Show latest input features"):
        st.dataframe(latest_features.reset_index().rename(columns={"index": "Date"}))

    # Optional: small recent chart for context
    try:
        recent = yf.download(ticker, period="3mo", interval="1d", auto_adjust=False, progress=False)
        if recent is not None and not recent.empty:
            st.write("Recent Close Prices (3M)")
            st.line_chart(recent["Close"].dropna())
    except Exception:
        pass

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
**Notes**
- Place your pretrained models in the app directory, named exactly as `<TICKER>.joblib` (e.g., `AAPL.joblib`).
- Models should be full `sklearn` pipelines that accept columns `['Open','High','Low','Close']` as input.
- We fetch the latest available daily OHLC via `yfinance` and predict the *next* close.
""")
