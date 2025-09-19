import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import rankdata

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="NSE Stock Screener")

st.title("NSE Stock Screener for Short-Term Momentum Trading 📈")
st.write("Upload your data file to analyze stocks based on a comprehensive set of technical indicators and custom ratios.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload 'stock_data.xlsx'", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load the three sheets from the uploaded Excel file
        bhav_data = pd.read_excel(uploaded_file, sheet_name="BhavData")
        index_data = pd.read_excel(uploaded_file, sheet_name="IndexData")
        high_data = pd.read_excel(uploaded_file, sheet_name="52w High")

        st.success("File uploaded and sheets loaded successfully!")

    except Exception as e:
        st.error(f"Error loading the Excel file. Please ensure it has the correct sheet names (BhavData, IndexData, 52w High). Error: {e}")
        st.stop()

    # --- Data Cleaning and Preparation ---
    try:
        # Clean and standardize column names
        bhav_data.rename(columns={'DATE1': 'Date', 'CLOSE_PRICE': 'Close', 'TTL_TRD_QNTY': 'Volume', 'DELIV_QTY': 'Deliverable Volume', 'SYMBOL': 'Symbol', 'HIGH_PRICE': 'High', 'LOW_PRICE': 'Low'}, inplace=True)
        index_data.rename(columns={'Index Date': 'Date', 'Closing Index Value': 'Close'}, inplace=True)
        high_data.rename(columns={'SYMBOL': 'Symbol', 'Adjusted_52_Week_High': '52wH', 'Adjusted_52_Week_Low': '52wL'}, inplace=True)

        # Convert date columns to datetime objects
        bhav_data['Date'] = pd.to_datetime(bhav_data['Date'])
        index_data['Date'] = pd.to_datetime(index_data['Date'])
        high_data['Date'] = pd.to_datetime(high_data['52_Week_High_Date'])

        # Set Date as index for easy access
        bhav_data.set_index('Date', inplace=True)
        index_data.set_index('Date', inplace=True)

        # Ensure data is sorted by date
        bhav_data.sort_index(inplace=True)
        index_data.sort_index(inplace=True)

    except KeyError as e:
        st.error(f"Column not found. Please check your Excel sheet headers. Missing column: {e}")
        st.stop()

    # --- Calculation Functions ---
    def calculate_rsi(prices, period=1
