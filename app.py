import streamlit as st
import pandas as pd
import numpy as np

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
    def calculate_rsi(prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(high, low, close, volume, period=14):
        """Calculates the Money Flow Index (MFI)."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = np.where(typical_price.diff() > 0, money_flow, 0)
        negative_flow = np.where(typical_price.diff() < 0, money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    # --- Main Analysis Loop ---
    all_stocks = bhav_data['Symbol'].unique()
    results_list = []
    nifty500_data = index_data[index_data.index_name.str.contains('nifty 500', case=False)]

    if nifty500_data.empty or len(nifty500_data) < 55:
        st.warning("Nifty 500 index data not found or insufficient. Skipping RS calculations.")
        nifty500_close_55d_ago = None
        nifty500_close_21d_ago = None
    else:
        nifty500_close_55d_ago = nifty500_data['Close'].iloc[-55]
        nifty500_close_21d_ago = nifty500_data['Close'].iloc[-21]
        nifty500_close_today = nifty500_data['Close'].iloc[-1]
        I55d_Returns = nifty500_close_today / nifty500_close_55d_ago
        I21d_Returns = nifty500_close_today / nifty500_close_21d_ago

    for symbol in all_stocks:
        stock_df = bhav_data[bhav_data['Symbol'] == symbol].copy()
        high_52w = high_data[high_data['Symbol'] == symbol].iloc[0]

        if len(stock_df) < 55:
            continue
        
        today_close = stock_df['Close'].iloc[-1]
        close_55d_ago = stock_df['Close'].iloc[-55]
        close_21d_ago = stock_df['Close'].iloc[-21]
        
        # Returns
        S55d_Returns = today_close / close_55d_ago
        S21d_Returns = today_close / close_21d_ago
        
        # Relative Strength
        RS55 = S55d_Returns / I55d_Returns if I55d_Returns else np.nan
        RS21 = S21d_Returns / I21d_Returns if I21d_Returns else np.nan
        
        # Momentum & Accumulation
        _52wH = high_52w['52wH']
        LTP = today_close
        _52wHZ = _52wH / LTP if LTP > 0 else np.nan
        
        RSI21 = calculate_rsi(stock_df['Close'], period=21).iloc[-1]
        MFI55_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=55).iloc[-1]
        MFI21_V = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], period=21).iloc[-1]
        MFI21_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=21).iloc[-1]

        AD = MFI21_D / MFI55_D if MFI55_D else np.nan
        Strength_AD = MFI21_D / MFI21_V if MFI21_V else np.nan
        Mom_Conf = MFI21_D / RSI21 if RSI21 else np.nan
        Mom_Osc = RS21 / RS55 if RS55 else np.nan

        # PVA Signals
        avg_delivery_7d = stock_df['Deliverable Volume'].iloc[-7:].mean()
        today_close_prev_close = today_close / stock_df['Close'].iloc[-2] - 1
        yest_close_day_before = stock_df['Close'].iloc[-2] / stock_df['Close'].iloc[-3] - 1
        
        pva = "None"
        if today_close_prev_close > 0 and stock_df['Deliverable Volume'].iloc[-1] > avg_delivery_7d and yest_close_day_before > 0 and stock_df['Deliverable Volume'].iloc[-2] > avg_delivery_7d:
            pva = "tbyb (Buy)"
        elif today_close_prev_close < 0 and stock_df['Deliverable Volume'].iloc[-1] > avg_delivery_7d and yest_close_day_before < 0 and stock_df['Deliverable Volume'].iloc[-2] > avg_delivery_7d:
            pva = "tsys (Sell)"
        
        # Screening Conditions
        signal = "None"
        # RS Entry Condition
        rs_entry = (RS55 > 0.93 and RS55 < 1 and Mom_Osc > 1 and Mom_Conf > 1 and Strength_AD > 1 and AD > 1 and _52wHZ < 1.25)
        # RS Exit Condition
        rs_exit = (
            (RS55 > 1 and RS55 < 1.07 and Mom_Osc < 1 and Mom_Conf < 1 and Strength_AD < 1 and AD < 1 and _52wHZ > 1.1) or
            (RS55 > 1 and RS55 < 1.07 and Mom_Osc < 1) or
            (Mom_Osc > 1.2)
        )

        if rs_entry:
            signal = "Entry"
        elif rs_exit:
            signal = "Exit"
            
        results_list.append({
            'Symbol': symbol,
            'Signal': signal,
            'PVA': pva,
            'RS21': RS21,
            'RS55': RS55,
            'MFI21_D': MFI21_D,
            'MFI21_V': MFI21_V,
            'MFI55_D': MFI55_D,
            'RSI21': RSI21,
            '52wHZ': _52wHZ,
            'AD': AD,
            'Strength_AD': Strength_AD,
            'Mom_Conf': Mom_Conf,
            'Mom_Osc': Mom_Osc
        })

    # Create a DataFrame and display results
    results_df = pd.DataFrame(results_list)

    if not results_df.empty:
        # Sort symbols as requested
        entry_signals = results_df[results_df['Signal'] == 'Entry'].sort_values('Symbol')
        exit_signals = results_df[results_df['Signal'] == 'Exit'].sort_values('Symbol')

        # Display tables with tabs
        st.header("Screener Results")
        tab_entry, tab_exit, tab_all = st.tabs(["Entry Signals", "Exit Signals", "All Stocks"])

        with tab_entry:
            if not entry_signals.empty:
                st.subheader("Stocks with Favorable Entry Conditions")
                st.dataframe(entry_signals, use_container_width=True)
            else:
                st.info("No stocks currently meet the entry criteria.")

        with tab_exit:
            if not exit_signals.empty:
                st.subheader("Stocks with Loss of Favorable Conditions (Exit Signals)")
                st.dataframe(exit_signals, use_container_width=True)
            else:
                st.info("No stocks currently meet the exit criteria.")
        
        with tab_all:
            st.subheader("All Stocks and Their Metrics")
            st.dataframe(results_df, use_container_width=True)

    else:
        st.info("No stocks meet the screening criteria at this time. The market may be consolidating.")

else:
    # Instructions for first-time use
    st.info("Please upload your 'stock_data.xlsx' file to begin the analysis. The file should contain 'BhavData', 'IndexData', and '52w High' sheets with the specified columns.")
    