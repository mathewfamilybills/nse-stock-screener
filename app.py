import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(page_title="Stock Market Technical Analysis", layout="wide")

# Title
st.title("📈 Stock Market Technical Analysis Dashboard")

# Sidebar for file upload and configuration
st.sidebar.header("Configuration")
st.sidebar.subheader("Upload stock_data.xlsx")

uploaded_file = st.sidebar.file_uploader(
    "Drag and drop file here",
    type=['xlsx'],
    help="Limit 200MB per file • XLSX"
)

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.bhav_data = None
    st.session_state.index_data = None
    st.session_state.high_52w_data = None
    st.session_state.results_df = None

# Load data when file is uploaded
if uploaded_file is not None and not st.session_state.data_loaded:
    try:
        # Read Excel file with all sheets
        xls = pd.ExcelFile(uploaded_file)
        
        # Load BhavData sheet
        bhav_data = pd.read_excel(xls, sheet_name='BhavData')
        bhav_data['DATE1'] = pd.to_datetime(bhav_data['DATE1'])
        bhav_data = bhav_data.sort_values(['SYMBOL', 'DATE1'])
        
        # Load IndexData sheet
        index_data = pd.read_excel(xls, sheet_name='IndexData')
        index_data['Index Date'] = pd.to_datetime(index_data['Index Date'])
        index_data = index_data.sort_values('Index Date')
        
        # Load 52w High sheet
        high_52w_data = pd.read_excel(xls, sheet_name='52w High')
        
        # Store in session state
        st.session_state.bhav_data = bhav_data
        st.session_state.index_data = index_data
        st.session_state.high_52w_data = high_52w_data
        st.session_state.data_loaded = True
        
        st.sidebar.success(f"✅ File uploaded successfully!")
        st.sidebar.info(f"Loaded {len(bhav_data)} records from BhavData")
        
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# India VIX input
st.sidebar.subheader("Enter India VIX Value:")
india_vix = st.sidebar.number_input(
    "India VIX",
    min_value=0.0,
    max_value=100.0,
    value=15.5,
    step=0.01,
    help="Enter the latest India VIX value"
)

# Process data if loaded
if st.session_state.data_loaded and st.session_state.results_df is None:
    with st.spinner("Calculating technical indicators..."):
        bhav_data = st.session_state.bhav_data.copy()
        index_data = st.session_state.index_data.copy()
        high_52w_data = st.session_state.high_52w_data.copy()
        
        # Get Nifty 500 data
        nifty_500 = index_data[index_data['Index Name'] == 'Nifty 500'].copy()
        nifty_500 = nifty_500.sort_values('Index Date').reset_index(drop=True)
        
        # Calculate Nifty 500 returns
        if len(nifty_500) >= 55:
            latest_nifty = nifty_500.iloc[-1]['Closing Index Value']
            nifty_55d_ago = nifty_500.iloc[-56]['Closing Index Value']
            nifty_21d_ago = nifty_500.iloc[-22]['Closing Index Value']
            
            I55d_Returns = latest_nifty / nifty_55d_ago
            I21d_Returns = latest_nifty / nifty_21d_ago
        else:
            I55d_Returns = 1.0
            I21d_Returns = 1.0
        
        # Function to calculate RSI
        def calculate_rsi(prices, period=21):
            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            if down == 0:
                return 100
            rs = up / down
            rsi = 100 - (100 / (1 + rs))
            
            for delta in deltas[period:]:
                if delta > 0:
                    upval = delta
                    downval = 0
                else:
                    upval = 0
                    downval = -delta
                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                if down == 0:
                    rs = 100
                else:
                    rs = up / down
                rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # Function to calculate MFI using delivery data
        def calculate_mfi_delivery(df, period=21):
            if len(df) < period + 1:
                return np.nan
            
            # Calculate typical price
            df = df.copy()
            df['TP'] = (df['HIGH_PRICE'] + df['LOW_PRICE'] + df['CLOSE_PRICE']) / 3
            
            # Calculate raw money flow using DELIVERY DATA
            df['RMF'] = df['TP'] * df['DELIV_QTY']
            
            # Identify positive and negative money flow
            df['TP_Change'] = df['TP'].diff()
            
            # Calculate positive and negative money flow
            positive_mf = 0
            negative_mf = 0
            
            for i in range(len(df) - period, len(df)):
                if i > 0 and df.iloc[i]['TP_Change'] > 0:
                    positive_mf += df.iloc[i]['RMF']
                elif i > 0 and df.iloc[i]['TP_Change'] < 0:
                    negative_mf += df.iloc[i]['RMF']
            
            if negative_mf == 0:
                return 100
            
            mf_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + mf_ratio))
            
            return mfi
        
        # Function to calculate CMF using delivery data
        def calculate_cmf_delivery(df, period=21):
            if len(df) < period:
                return np.nan
            
            df = df.copy()
            recent_data = df.tail(period)
            
            # Calculate Money Flow Multiplier
            mf_multiplier = ((recent_data['CLOSE_PRICE'] - recent_data['LOW_PRICE']) - 
                           (recent_data['HIGH_PRICE'] - recent_data['CLOSE_PRICE'])) / \
                          (recent_data['HIGH_PRICE'] - recent_data['LOW_PRICE'])
            
            # Replace any NaN or inf values
            mf_multiplier = mf_multiplier.fillna(0)
            mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0)
            
            # Calculate Money Flow Volume using DELIVERY DATA
            mf_volume = mf_multiplier * recent_data['DELIV_QTY']
            
            # Calculate CMF
            cmf = mf_volume.sum() / recent_data['DELIV_QTY'].sum()
            
            return cmf
        
        # Function to calculate Max Drawdown
        def calculate_max_drawdown(prices):
            if len(prices) < 2:
                return 0
            cummax = np.maximum.accumulate(prices)
            drawdown = (prices - cummax) / cummax
            return drawdown.min()
        
        # Function to calculate Sortino Ratio
        def calculate_sortino(returns, target_return=0):
            if len(returns) < 2:
                return 0
            excess_returns = returns - target_return
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0
            return excess_returns.mean() / downside_returns.std()
        
        # Function to calculate Beta
        def calculate_beta(stock_returns, market_returns):
            if len(stock_returns) < 2 or len(market_returns) < 2:
                return 1.0
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            if market_variance == 0:
                return 1.0
            return covariance / market_variance
        
        # Function to calculate Alpha (annualized)
        def calculate_alpha(stock_returns, market_returns, beta, rf_rate=0.05):
            if len(stock_returns) < 2:
                return 0
            stock_annual_return = (1 + stock_returns.mean()) ** 252 - 1
            market_annual_return = (1 + market_returns.mean()) ** 252 - 1
            alpha = stock_annual_return - (rf_rate + beta * (market_annual_return - rf_rate))
            return alpha
        
        # Process each stock
        results = []
        symbols = bhav_data['SYMBOL'].unique()
        
        # Get Nifty 500 returns for beta calculation
        nifty_500_sorted = nifty_500.sort_values('Index Date').tail(56)
        nifty_returns = nifty_500_sorted['Closing Index Value'].pct_change().dropna().values
        
        for symbol in symbols:
            stock_data = bhav_data[bhav_data['SYMBOL'] == symbol].copy()
            stock_data = stock_data.sort_values('DATE1').reset_index(drop=True)
            
            # Need at least 56 days for calculations
            if len(stock_data) < 56:
                continue
            
            # Get latest data
            latest_close = stock_data.iloc[-1]['CLOSE_PRICE']
            
            # Calculate returns
            if len(stock_data) >= 56:
                close_55d_ago = stock_data.iloc[-56]['CLOSE_PRICE']
                S55d_Returns = latest_close / close_55d_ago
            else:
                S55d_Returns = 1.0
            
            if len(stock_data) >= 22:
                close_21d_ago = stock_data.iloc[-22]['CLOSE_PRICE']
                S21d_Returns = latest_close / close_21d_ago
            else:
                S21d_Returns = 1.0
            
            # Calculate RS
            RS55 = S55d_Returns / I55d_Returns if I55d_Returns != 0 else 1.0
            RS21 = S21d_Returns / I21d_Returns if I21d_Returns != 0 else 1.0
            
            # Calculate 52-week High Zone
            high_52w_row = high_52w_data[high_52w_data['SYMBOL'] == symbol]
            if len(high_52w_row) > 0:
                high_52w = high_52w_row.iloc[0]['Adjusted_52_Week_High']
                wHZ_52 = high_52w / latest_close if latest_close != 0 else 1.0
            else:
                wHZ_52 = 1.0
            
            # Calculate RSI21
            if len(stock_data) >= 22:
                prices_for_rsi = stock_data['CLOSE_PRICE'].values[-22:]
                RSI21 = calculate_rsi(prices_for_rsi, period=21)
            else:
                RSI21 = 50
            
            # Calculate MFI using delivery data
            MFI21_D = calculate_mfi_delivery(stock_data, period=21)
            MFI55_D = calculate_mfi_delivery(stock_data, period=55)
            
            # Calculate CMF using delivery data
            CMF21_D = calculate_cmf_delivery(stock_data, period=21)
            CMF55_D = calculate_cmf_delivery(stock_data, period=55)
            
            # Calculate derived ratios
            AD = MFI21_D / MFI55_D if MFI55_D != 0 and not np.isnan(MFI55_D) else 1.0
            Strength_Mom = MFI21_D / RSI21 if RSI21 != 0 and not np.isnan(MFI21_D) else 1.0
            Mom_Osc = RS21 / RS55 if RS55 != 0 else 1.0
            PSR = CMF21_D / CMF55_D if CMF55_D != 0 and not np.isnan(CMF55_D) else 1.0
            
            # Calculate PMA
            if not np.isnan(CMF21_D) and not np.isnan(MFI21_D):
                PMA = ((CMF21_D + (MFI21_D / 50)) * ((70 - RSI21) / 40))
            else:
                PMA = 0
            
            # Risk Parameters
            recent_55_prices = stock_data['CLOSE_PRICE'].values[-55:]
            max_dd = calculate_max_drawdown(recent_55_prices)
            
            recent_55_returns = stock_data['CLOSE_PRICE'].pct_change().dropna().values[-55:]
            sortino = calculate_sortino(recent_55_returns)
            
            beta = calculate_beta(recent_55_returns, nifty_returns[-55:])
            alpha = calculate_alpha(recent_55_returns, nifty_returns[-55:], beta)
            
            # Calculate 7-day average delivery for PVA signals
            avg_7d_deliv = stock_data['DELIV_QTY'].tail(7).mean()
            
            # PVA Signals
            tbyb = False
            tsys = False
            
            if len(stock_data) >= 3:
                # Today's data (index -1)
                today_close = stock_data.iloc[-1]['CLOSE_PRICE']
                today_deliv = stock_data.iloc[-1]['DELIV_QTY']
                
                # Yesterday's data (index -2)
                yesterday_close = stock_data.iloc[-2]['CLOSE_PRICE']
                yesterday_deliv = stock_data.iloc[-2]['DELIV_QTY']
                
                # Day before yesterday (index -3)
                day_before_close = stock_data.iloc[-3]['CLOSE_PRICE']
                
                # tbyb: Buy signal
                if (today_close > yesterday_close and today_deliv > avg_7d_deliv and
                    yesterday_close > day_before_close and yesterday_deliv > avg_7d_deliv):
                    tbyb = True
                
                # tsys: Sell signal
                if (today_close < yesterday_close and today_deliv > avg_7d_deliv and
                    yesterday_close < day_before_close and yesterday_deliv > avg_7d_deliv):
                    tsys = True
            
            # Determine PVA signal
            if tbyb:
                pva_signal = "Buy"
            elif tsys:
                pva_signal = "Sell"
            else:
                pva_signal = "Neutral"
            
            # Store results with all calculated indicators
            results.append({
                'SYMBOL': symbol,
                'LTP': latest_close,
                'RS55': RS55,
                'RS21': RS21,
                'RS21_prev': stock_data.iloc[-2]['CLOSE_PRICE'] / stock_data.iloc[-23]['CLOSE_PRICE'] / I21d_Returns if len(stock_data) >= 23 else RS21,
                'RS55_prev': stock_data.iloc[-2]['CLOSE_PRICE'] / stock_data.iloc[-57]['CLOSE_PRICE'] / I55d_Returns if len(stock_data) >= 57 else RS55,
                '52wHZ': wHZ_52,
                'RSI21': RSI21,
                'MFI21_D': MFI21_D,
                'MFI55_D': MFI55_D,
                'CMF21_D': CMF21_D,
                'CMF55_D': CMF55_D,
                'AD': AD,
                'Strength_Mom': Strength_Mom,
                'Mom_Osc': Mom_Osc,
                'PSR': PSR,
                'PMA': PMA,
                'MaxDD': max_dd,
                'Sortino': sortino,
                'Beta': beta,
                'Alpha': alpha,
                'PVA': pva_signal,
                'tbyb': tbyb,
                'tsys': tsys
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate percentile scores for risk parameters
        results_df['p_MaxDD'] = results_df['MaxDD'].rank(pct=True, ascending=False)  # Less drawdown = higher score
        results_df['p_Sortino'] = results_df['Sortino'].rank(pct=True, ascending=True)  # Higher Sortino = higher score
        results_df['p_Beta'] = results_df['Beta'].rank(pct=True, ascending=False)  # Lower Beta = higher score
        results_df['p_Alpha'] = results_df['Alpha'].rank(pct=True, ascending=True)  # Higher Alpha = higher score
        
        # Calculate RATS
        results_df['RATS_raw'] = (results_df['p_MaxDD'] + results_df['p_Sortino'] + 
                                   results_df['p_Beta'] + results_df['p_Alpha'])
        results_df['RATS'] = results_df['RATS_raw'].rank(pct=True, ascending=True)
        
        # Calculate vRATS
        results_df['vRATS'] = results_df['RATS'] * (india_vix / 15.5)
        
        # Detect crossovers
        results_df['RS21_crosses_above_1'] = (results_df['RS21'] > 1) & (results_df['RS21_prev'] <= 1)
        results_df['RS21_crosses_below_1'] = (results_df['RS21'] < 1) & (results_df['RS21_prev'] >= 1)
        results_df['RS55_crosses_above_1'] = (results_df['RS55'] > 1) & (results_df['RS55_prev'] <= 1)
        results_df['RS55_crosses_below_1'] = (results_df['RS55'] < 1) & (results_df['RS55_prev'] >= 1)
        
        st.session_state.results_df = results_df

# Display results if data is processed
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    
    # Apply screening conditions
    # Entry Condition
    entry_condition = (
        (results_df['PMA'] > 0.5) &
        (results_df['PSR'] > 0.2) &
        (results_df['AD'] > 1) &
        (results_df['Strength_Mom'] > 1.05) &
        (
            ((results_df['RS55'] < 1) & results_df['RS21_crosses_above_1']) |
            ((results_df['Mom_Osc'] > 1.2) & results_df['RS21_crosses_above_1'])
        ) &
        (results_df['Mom_Osc'] > 1) &
        (results_df['52wHZ'] < 1.3)
    )
    
    # Exit Condition
    exit_condition = (
        (results_df['PMA'] < 0.5) &
        ((results_df['Mom_Osc'] < 1) | (results_df['Mom_Osc'] < 0.85)) &
        results_df['RS21_crosses_below_1'] &
        (results_df['Strength_Mom'] < 1.05)
    )
    
    # Add Stocks Condition
    add_condition = (
        (results_df['PMA'] > 0.5) &
        (results_df['PSR'] > 0.2) &
        (results_df['RS21'] > 1) &
        (results_df['Mom_Osc'] > 1) &
        (results_df['Strength_Mom'] > 1.07) &
        (results_df['52wHZ'] < 0.75) &
        (results_df['PVA'] == 'Buy')
    )
    
    # Book Profits Condition
    book_condition = (
        ((results_df['RS55'] > 1.07) & 
         (results_df['Mom_Osc'] < 1.07) & 
         results_df['RS21_crosses_below_1']) |
        ((results_df['PSR'] < 0.1) & (results_df['PVA'] == 'Sell'))
    )
    
    # Create display columns
    display_cols = ['SYMBOL', 'PVA', 'RS21', 'RS55', 'MFI21_D', 'MFI55_D', 
                    'RSI21', 'CMF21_D', 'CMF55_D', '52wHZ', 'AD', 
                    'Strength_Mom', 'Mom_Osc', 'PSR', 'PMA', 'vRATS']
    
    # Display Trading Signals
    st.header("📊 Trading Signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Entry Signals")
        entry_stocks = results_df[entry_condition][display_cols].sort_values('SYMBOL')
        if len(entry_stocks) > 0:
            entry_stocks_display = entry_stocks.copy()
            entry_stocks_display['Signal'] = 'Entry'
            st.dataframe(entry_stocks_display, use_container_width=True, hide_index=True)
        else:
            st.info("No entry signals found.")
    
    with col2:
        st.subheader("🔴 Exit Signals")
        exit_stocks = results_df[exit_condition][display_cols].sort_values('SYMBOL')
        if len(exit_stocks) > 0:
            exit_stocks_display = exit_stocks.copy()
            exit_stocks_display['Signal'] = 'Exit'
            st.dataframe(exit_stocks_display, use_container_width=True, hide_index=True)
        else:
            st.info("No exit signals found.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("➕ Add Stocks")
        add_stocks = results_df[add_condition][display_cols].sort_values('SYMBOL')
        if len(add_stocks) > 0:
            add_stocks_display = add_stocks.copy()
            add_stocks_display['Signal'] = 'Add'
            st.dataframe(add_stocks_display, use_container_width=True, hide_index=True)
        else:
            st.info("No add signals found.")
    
    with col4:
        st.subheader("💰 Book Profits")
        book_stocks = results_df[book_condition][display_cols].sort_values('SYMBOL')
        if len(book_stocks) > 0:
            book_stocks_display = book_stocks.copy()
            book_stocks_display['Signal'] = 'Book'
            st.dataframe(book_stocks_display, use_container_width=True, hide_index=True)
        else:
            st.info("No book profit signals found.")
    
    # Graph Selection Section
    st.header("📈 Generate Ratio Graphs")
    
    # Show all stocks data
    st.subheader("All Stocks Data")
    all_stocks_display = results_df[display_cols].sort_values('SYMBOL')
    st.dataframe(all_stocks_display, use_container_width=True, hide_index=True)
    
    # Sidebar stock selection
    st.sidebar.header("Graph Selection")
    available_symbols = sorted(results_df['SYMBOL'].unique())
    
    if len(available_symbols) > 0:
        selected_symbol = st.sidebar.selectbox(
            "Select a stock symbol to view its graph:",
            options=available_symbols
        )
        
        if st.sidebar.button("Generate Graph"):
            # Get stock data
            stock_data = st.session_state.bhav_data[
                st.session_state.bhav_data['SYMBOL'] == selected_symbol
            ].copy()
            stock_data = stock_data.sort_values('DATE1').reset_index(drop=True)
            
            if len(stock_data) >= 55:
                # Calculate historical ratios for last 60 days
                hist_data = []
                for i in range(max(0, len(stock_data) - 60), len(stock_data)):
                    window_data = stock_data.iloc[:i+1]
                    
                    if len(window_data) >= 55:
                        # Calculate ratios
                        MFI21 = calculate_mfi_delivery(window_data, 21)
                        MFI55 = calculate_mfi_delivery(window_data, 55)
                        CMF21 = calculate_cmf_delivery(window_data, 21)
                        CMF55 = calculate_cmf_delivery(window_data, 55)
                        
                        prices_rsi = window_data['CLOSE_PRICE'].values[-22:]
                        RSI21_val = calculate_rsi(prices_rsi, 21)
                        
                        # Calculate returns
                        latest = window_data.iloc[-1]['CLOSE_PRICE']
                        close_55 = window_data.iloc[-56]['CLOSE_PRICE']
                        close_21 = window_data.iloc[-22]['CLOSE_PRICE']
                        
                        S55_ret = latest / close_55
                        S21_ret = latest / close_21
                        
                        RS55_val = S55_ret / I55d_Returns
                        RS21_val = S21_ret / I21d_Returns
                        
                        AD_val = MFI21 / MFI55 if MFI55 != 0 else 1
                        Strength_val = MFI21 / RSI21_val if RSI21_val != 0 else 1
                        Mom_val = RS21_val / RS55_val if RS55_val != 0 else 1
                        PSR_val = CMF21 / CMF55 if CMF55 != 0 else 1
                        PMA_val = ((CMF21 + (MFI21 / 50)) * ((70 - RSI21_val) / 40))
                        
                        # Get vRATS for this date
                        vRATS_val = results_df[results_df['SYMBOL'] == selected_symbol]['vRATS'].values[0]
                        
                        hist_data.append({
                            'Date': window_data.iloc[-1]['DATE1'],
                            'AD': AD_val,
                            'Strength_Mom': Strength_val,
                            'Mom_Osc': Mom_val,
                            'PSR': PSR_val,
                            'PMA': PMA_val,
                            'vRATS': vRATS_val
                        })
                
                hist_df = pd.DataFrame(hist_data)
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['AD'],
                    mode='lines', name='AD Ratio',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['Strength_Mom'],
                    mode='lines', name='Strength Mom',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['Mom_Osc'],
                    mode='lines', name='Mom Oscillator',
                    line=dict(color='orange', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['PSR'],
                    mode='lines', name='PSR',
                    line=dict(color='purple', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['PMA'],
                    mode='lines', name='PMA',
                    line=dict(color='cyan', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist_df['Date'], y=hist_df['vRATS'],
                    mode='lines', name='vRATS',
                    line=dict(color='black', width=3)
                ))
                
                # Add threshold lines
                fig.add_hline(y=1, line_dash="dash", line_color="red",
                            annotation_text="Threshold at 1")
                fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                            annotation_text="Threshold at 0.5")
                
                fig.update_layout(
                    title=f"Technical Ratios for {selected_symbol}",
                    xaxis_title="Date",
                    yaxis_title="Ratio Value",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display 21-day table
                st.subheader(f"21-Day Data for {selected_symbol}")
