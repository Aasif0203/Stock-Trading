import streamlit as st

st.set_page_config(layout="wide")

st.title("Minimal Test Successful!")

st.write("If you can see this text, it means you can successfully run a basic `app.py` file.")

st.balloons()

st.info("This is great progress! It strongly suggests the issue is with the `yfinance` or `pandas` library import. We can fix that next.")

