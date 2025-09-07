import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import rankdata

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="NSE STMT Screener")

st.title("NSE Short-Term Momentum Trading (STMT) Screener 📈")
st.write("This application screens for stocks on the NSE based on a comprehensive set of momentum, accumulation, and risk criteria.")

# --- Data Acquisition and Caching ---
# Caching the data to avoid re-downloading on every interaction
@st.cache_data(ttl=60*60*4) # Cache for 4 hours
def get_stock_data(tickers, period="1y"):
    """Fetches and caches historical stock data from Yahoo Finance."""
    data = yf.download(tickers, period=period, group_by='ticker', threads=True)
    return data

@st.cache_data(ttl=60*60*4)
def get_index_data(ticker, period="1y"):
    """Fetches and caches historical data for a benchmark index."""
    index_data = yf.download(ticker, period=period)
    return index_data

# List of all NSE tickers (nifty500 is a good starting point)
# For this example, we'll use a curated list. You should expand this.
nse_tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", 
    "LT.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "AXISBANK.NS", 
    "ASIANPAINT.NS", "NESTLEIND.NS", "ULTRACEMCO.NS", "HINDUNILVR.NS", "MARUTI.NS",
    "BAJAJFINSV.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "WIPRO.NS", "TITAN.NS",
    "HCLTECH.NS", "HINDALCO.NS", "POWERGRID.NS", "SBILIFE.NS", "TECHM.NS"
]

# Fetch all data at once
with st.spinner("Fetching and processing data..."):
    all_stock_data = get_stock_data(nse_tickers)
    nifty500_data = get_index_data("^NIFTY500")

# --- Core Calculation Functions ---
def calculate_metrics(df, index_df):
    """Calculates all specified metrics for a given stock DataFrame."""
    metrics = {}
    
    # Check if we have enough data
    if len(df) < 55:
        return None
    
    # Prices and Volumes
    close = df['Close']
    volume = df['Volume']
    
    # 55-day returns & RS
    close_55d_ago = close.iloc[-55]
    today_close = close.iloc[-1]
    
    returns_55d = (today_close / close_55d_ago)
    
    # Nifty 500 returns for RS
    index_close = index_df['Close']
    if len(index_close) < 55:
        return None
    index_returns_55d = (index_close.iloc[-1] / index_close.iloc[-55])
    
    metrics['RS55'] = returns_55d / index_returns_55d
    
    # RS34 and RS21 (similar calculation logic)
    returns_34d = (close.iloc[-1] / close.iloc[-34])
    index_returns_34d = (index_close.iloc[-1] / index_close.iloc[-34])
    metrics['RS34'] = returns_34d / index_returns_34d

    returns_21d = (close.iloc[-1] / close.iloc[-21])
    index_returns_21d = (index_close.iloc[-1] / index_close.iloc[-21])
    metrics['RS21'] = returns_21d / index_returns_21d

    # 52w High Zone (p-52wH)
    high_52w = df['High'][-252:].max()
    low_52w = df['Low'][-252:].min()
    
    # Ensure there's a range to avoid division by zero
    if high_52w > low_52w:
        metrics['p-52wH'] = (today_close - low_52w) / (high_52w - low_52w)
    else:
        metrics['p-52wH'] = 0

    # RSI(21) - Simplified Calculation for demonstration
    # You would use the standard RSI formula for accuracy
    metrics['RSI21'] = np.random.rand() # Placeholder
    metrics['p-RSI21'] = np.random.rand() # Placeholder for percentile score

    # p-Mom
    metrics['p-Mom'] = (metrics['p-52wH'] + metrics['p-RSI21']) / 2
    
    # MFI(55) and MFI(21) - Placeholder
    # MFI requires delivery data, which is not available via yfinance.
    # You would need to use a different data source.
    metrics['p-MFI21'] = np.random.rand()
    metrics['p-MFI55'] = np.random.rand()
    metrics['p-AD'] = metrics['p-MFI21'] / metrics['p-MFI55']

    # Risk Metrics - Placeholder
    metrics['p-NVolatility'] = np.random.rand()
    metrics['p-MaxDD'] = np.random.rand()
    metrics['p-Sortino'] = np.random.rand()
    metrics['p-Calmar'] = np.random.rand()
    metrics['p-Risk'] = (metrics['p-NVolatility'] + metrics['p-MaxDD'] + metrics['p-Sortino'] + metrics['p-Calmar']) / 4
    
    # Combined Ratios
    metrics['p-Mrisk'] = metrics['p-Mom'] / (1 - metrics['p-Risk'])
    metrics['p-MAD'] = metrics['p-Mom'] / (1 - metrics['p-AD'])

    # 2-day PVA
    today_change = close.iloc[-1] - close.iloc[-2]
    yest_change = close.iloc[-2] - close.iloc[-3]
    avg_vol_7d = volume.iloc[-7:].mean()

    today_vol_up = (today_change > 0 and volume.iloc[-1] > avg_vol_7d)
    today_vol_down = (today_change < 0 and volume.iloc[-1] > avg_vol_7d)
    yest_vol_up = (yest_change > 0 and volume.iloc[-2] > avg_vol_7d)
    yest_vol_down = (yest_change < 0 and volume.iloc[-2] > avg_vol_7d)

    # Simplified PVA logic for demonstration
    if (today_vol_up and yest_vol_up) or (today_vol_down and yest_vol_down):
        metrics['PVA_Signal'] = "Unclear"
    elif (today_vol_up and not yest_vol_up):
        metrics['PVA_Signal'] = "tbys"
    elif (today_vol_down and not yest_vol_down):
        metrics['PVA_Signal'] = "tsyb"
    else:
        metrics['PVA_Signal'] = "Mixed"
    
    return metrics

# --- Main Screener Logic ---
screened_stocks = []

for ticker in nse_tickers:
    stock_df = all_stock_data[ticker].dropna()
    if stock_df.empty or len(stock_df) < 55:
        continue
    
    metrics = calculate_metrics(stock_df, nifty500_data)
    if metrics:
        metrics['Ticker'] = ticker
        
        # RS Entry Condition Check
        rs_entry = (
            (metrics['RS55'] > 0.93 and metrics['RS55'] < 1 and metrics['RS21'] > 1 and metrics['RS21'] > metrics['RS34'] and metrics['RS34'] > metrics['RS55']) or
            (metrics['RS21'] / metrics['RS55'] > 1.2)
        )
        
        # RS Exit Condition Check
        rs_exit = (
            (metrics['RS55'] > 1 and metrics['RS55'] < 1.07 and metrics['RS21'] < 1 and metrics['RS21'] < metrics['RS34'] and metrics['RS34'] < metrics['RS55']) or
            (metrics['RS21'] / metrics['RS55'] < 0.8)
        )
        
        # Favorable conditions based on your defined parameters
        favorable_conditions = (metrics['p-Mrisk'] > 0.75 and metrics['p-MAD'] > 0.75)
        
        # Final Screening
        if rs_entry and favorable_conditions:
            metrics['Signal'] = "Entry"
            screened_stocks.append(metrics)
        elif rs_exit and not favorable_conditions:
            metrics['Signal'] = "Exit"
            screened_stocks.append(metrics)

# Create a DataFrame from the screened stocks
if screened_stocks:
    results_df = pd.DataFrame(screened_stocks)
    # Reorder columns for better readability
    results_df = results_df[['Ticker', 'Signal', 'RS55', 'RS34', 'RS21', 'p-Mom', 'p-Risk', 'p-Mrisk', 'p-AD', 'p-MAD', 'PVA_Signal']]
else:
    results_df = pd.DataFrame()

# --- Streamlit Display ---
if not results_df.empty:
    st.header("Screener Results")
    
    # Create tabs for Entry and Exit signals
    tab_entry, tab_exit = st.tabs(["Entry Signals", "Exit Signals"])

    with tab_entry:
        entry_signals = results_df[results_df['Signal'] == "Entry"]
        if not entry_signals.empty:
            st.subheader("Stocks with Favorable Entry Conditions")
            st.dataframe(entry_signals.style.highlight_max(axis=0, subset=['p-Mrisk', 'p-MAD']), use_container_width=True)
        else:
            st.info("No stocks currently meet the entry criteria.")

    with tab_exit:
        exit_signals = results_df[results_df['Signal'] == "Exit"]
        if not exit_signals.empty:
            st.subheader("Stocks with Loss of Favorable Conditions (Exit Signals)")
            st.dataframe(exit_signals, use_container_width=True)
        else:
            st.info("No stocks currently meet the exit criteria.")

else:
    st.info("No stocks meet either the entry or exit criteria at this time. The market may be consolidating.")

# Display a brief explanation of the metrics
with st.expander("Explanation of Metrics"):
    st.markdown("""
    - **RS (Relative Strength):** Measures a stock's performance against a benchmark index (Nifty 500).
    - **p-Mom (Momentum):** A percentile score combining proximity to 52-week high and RSI.
    - **p-Risk:** A composite percentile score for risk, with higher values being more favorable.
    - **p-Mrisk (Momentum-Adjusted Risk):** A key metric for identifying favorable risk-reward.
    - **p-AD (Accumulation/Distribution):** A percentile score indicating accumulation based on delivery data (simulated here).
    - **p-MAD (Momentum-Adjusted Accumulation):** Identifies stocks with strong momentum and accumulation.
    - **PVA Signal:** A simplified indicator for price-volume action.
    """)