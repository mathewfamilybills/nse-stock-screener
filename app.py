import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="NSE Stock Screener")

st.title("NSE Stock Screener for Short-Term Momentum Trading 📈")
st.write("Upload your data file to analyze stocks based on a comprehensive set of technical indicators and custom ratios.")

# --- Calculation Functions (Corrected) ---

def calculate_rsi(prices, period=21):
    """Calculates a robust Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_mfi(high, low, close, volume, period=55):
    """Calculates a robust Money Flow Index (MFI) preserving the date index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    price_diff = typical_price.diff()
    
    positive_flow = raw_money_flow.where(price_diff > 0, 0)
    negative_flow = raw_money_flow.where(price_diff < 0, 0)
    
    positive_mf_sum = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_mf_sum = negative_flow.rolling(window=period, min_periods=1).sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        money_ratio = positive_mf_sum / (negative_mf_sum + 1e-9)
        
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload 'stock_data.xlsx'", type=["xlsx"])

if uploaded_file is not None:
    try:
        bhav_data = pd.read_excel(uploaded_file, sheet_name="BhavData")
        index_data = pd.read_excel(uploaded_file, sheet_name="IndexData")
        high_data = pd.read_excel(uploaded_file, sheet_name="52w High")
        st.success("File uploaded and sheets loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the Excel file. Please ensure it has the correct sheet names (BhavData, IndexData, 52w High). Error: {e}")
        st.stop()

    # --- Data Cleaning and Preparation ---
    try:
        bhav_data.rename(columns={'DATE1': 'Date', 'CLOSE_PRICE': 'Close', 'TTL_TRD_QNTY': 'Volume', 'DELIV_QTY': 'Deliverable Volume', 'SYMBOL': 'Symbol', 'HIGH_PRICE': 'High', 'LOW_PRICE': 'Low'}, inplace=True)
        index_data.rename(columns={'Index Date': 'Date', 'Closing Index Value': 'Close'}, inplace=True)
        high_data.rename(columns={'SYMBOL': 'Symbol', 'Adjusted_52_Week_High': '52wH'}, inplace=True)
        bhav_data['Date'] = pd.to_datetime(bhav_data['Date'])
        index_data['Date'] = pd.to_datetime(index_data['Date'])
    except KeyError as e:
        st.error(f"Column not found. Please check your Excel sheet headers. Missing column: {e}")
        st.stop()
        
    # --- Main Analysis Loop ---
    all_stocks = bhav_data['Symbol'].unique()
    results_list = []
    
    # Pre-calculate index returns
    nifty500_data = index_data[index_data['Index Name'].str.contains('nifty 500', case=False)].set_index('Date').sort_index()
    if nifty500_data.empty or len(nifty500_data) < 55:
        st.warning("Nifty 500 index data not found or insufficient. Skipping RS calculations.")
        nifty500_data['I55d_Returns'] = np.nan
        nifty500_data['I34d_Returns'] = np.nan
        nifty500_data['I21d_Returns'] = np.nan
    else:
        nifty500_data['I55d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(54)
        nifty500_data['I34d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(33)
        nifty500_data['I21d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(20)

    for symbol in all_stocks:
        stock_df = bhav_data[bhav_data['Symbol'] == symbol].set_index('Date').sort_index()
        high_52w_filtered = high_data[high_data['Symbol'] == symbol]
        if high_52w_filtered.empty or len(stock_df) < 55:
            continue
        
        # Get latest values for the stock
        today_close = stock_df['Close'].iloc[-1]
        close_55d_ago = stock_df['Close'].iloc[-55]
        close_34d_ago = stock_df['Close'].iloc[-34]
        close_21d_ago = stock_df['Close'].iloc[-21]
        
        # Get corresponding index returns for today
        latest_date = stock_df.index[-1]
        if latest_date in nifty500_data.index:
            I55d_Returns = nifty500_data.loc[latest_date, 'I55d_Returns']
            I34d_Returns = nifty500_data.loc[latest_date, 'I34d_Returns']
            I21d_Returns = nifty500_data.loc[latest_date, 'I21d_Returns']
        else:
            I55d_Returns, I34d_Returns, I21d_Returns = np.nan, np.nan, np.nan
            
        # Calculate RS and other metrics
        RS55 = (today_close / close_55d_ago) / I55d_Returns if I55d_Returns else np.nan
        RS34 = (today_close / close_34d_ago) / I34d_Returns if I34d_Returns else np.nan
        RS21 = (today_close / close_21d_ago) / I21d_Returns if I21d_Returns else np.nan
        
        # Other calculations
        _52wH = high_52w_filtered.iloc[0]['52wH']
        _52wHZ = _52wH / today_close if today_close > 0 else np.nan
        RSI21 = calculate_rsi(stock_df['Close'], period=21).iloc[-1]
        MFI55_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=55).iloc[-1]
        MFI21_V = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], period=21).iloc[-1]
        MFI21_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=21).iloc[-1]
        
        AD = MFI21_D / MFI55_D if MFI55_D else np.nan
        Strength_AD = MFI21_D / MFI21_V if MFI21_V else np.nan
        Mom_Conf = MFI21_D / RSI21 if RSI21 else np.nan
        Mom_Osc = RS21 / RS55 if RS55 else np.nan

        # Signal Logic
        rs_entry = (RS55 > 0.93 and RS55 < 1 and RS21 > RS34 and RS34 > RS55) or (Mom_Osc > 1.2)
        rs_exit = (RS55 > 1 and RS55 < 1.07 and RS21 < RS34 and RS34 < RS55) or (Mom_Osc < 0.8)
        signal = "Entry" if rs_entry else "Exit" if rs_exit else "None"
            
        results_list.append({
            'Symbol': symbol, 'Signal': signal, 'RS21': RS21, 'RS55': RS55, 'MFI21_D': MFI21_D,
            'MFI21_V': MFI21_V, 'MFI55_D': MFI55_D, 'RSI21': RSI21, '52wHZ': _52wHZ,
            'AD': AD, 'Strength_AD': Strength_AD, 'Mom_Conf': Mom_Conf, 'Mom_Osc': Mom_Osc
        })

    results_df = pd.DataFrame(results_list).dropna().round(4)

    if not results_df.empty:
        st.header("Screener Results")
        tab_entry, tab_exit, tab_all = st.tabs(["Entry Signals", "Exit Signals", "All Stocks"])

        with tab_entry:
            st.dataframe(results_df[results_df['Signal'] == 'Entry'], use_container_width=True)
        with tab_exit:
            st.dataframe(results_df[results_df['Signal'] == 'Exit'], use_container_width=True)
        with tab_all:
            st.dataframe(results_df, use_container_width=True)

    # --- Graphing Section ---
    st.sidebar.header("Generate Ratio Graphs")
    all_symbols_in_results = results_df['Symbol'].unique().tolist()
    selected_symbol = st.sidebar.selectbox("Select a stock symbol to view its graph:", options=all_symbols_in_results)
    
    if st.sidebar.button("Generate Graph"):
        st.subheader(f"Dual Momentum Ratios for {selected_symbol}")
        
        stock_df_graph = bhav_data[bhav_data['Symbol'] == selected_symbol].set_index('Date').sort_index()
        
        if len(stock_df_graph) < 55:
            st.warning("Insufficient data for the selected symbol to generate graphs.")
        else:
            # Join with index data to align dates for RS calculation
            graph_data = stock_df_graph.join(nifty500_data[['I21d_Returns', 'I55d_Returns']], how='left').ffill()

            # Calculate historical stock returns
            graph_data['S21d_Returns'] = graph_data['Close'] / graph_data['Close'].shift(20)
            graph_data['S55d_Returns'] = graph_data['Close'] / graph_data['Close'].shift(54)
            
            # Calculate historical RS
            graph_data['RS21'] = graph_data['S21d_Returns'] / graph_data['I21d_Returns']
            graph_data['RS55'] = graph_data['S55d_Returns'] / graph_data['I55d_Returns']
            
            # Calculate historical indicators
            graph_data['RSI21'] = calculate_rsi(graph_data['Close'], period=21)
            graph_data['MFI21_D'] = calculate_mfi(graph_data['High'], graph_data['Low'], graph_data['Close'], graph_data['Deliverable Volume'], period=21)
            graph_data['MFI21_V'] = calculate_mfi(graph_data['High'], graph_data['Low'], graph_data['Close'], graph_data['Volume'], period=21)
            graph_data['MFI55_D'] = calculate_mfi(graph_data['High'], graph_data['Low'], graph_data['Close'], graph_data['Deliverable Volume'], period=55)

            # Calculate final ratios for plotting
            plot_df = pd.DataFrame(index=graph_data.index)
            plot_df['AD (MFI21D/MFI55D)'] = graph_data['MFI21_D'] / graph_data['MFI55_D']
            plot_df['Strength_AD (MFI21D/MFI21V)'] = graph_data['MFI21_D'] / graph_data['MFI21_V']
            plot_df['Mom_Conf (MFI21D/RSI21)'] = graph_data['MFI21_D'] / graph_data['RSI21']
            plot_df['Mom_Osc (RS21/RS55)'] = graph_data['RS21'] / graph_data['RS55']

            plot_df = plot_df.tail(60) # Plot last 60 days

            fig = go.Figure()
            fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Crossover at 1")

            for col in plot_df.columns:
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode='lines', name=col))

            fig.update_layout(title=f'Dual Momentum Ratios for {selected_symbol}',
                              xaxis_title="Date", yaxis_title="Ratio Value", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
else:
    st.info("Please upload your 'stock_data.xlsx' file to begin the analysis.")
