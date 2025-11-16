import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(page_title="Stock Market Technical Analysis", layout="wide")

# Title
st.title("📈 Stock Market Technical Analysis Dashboard")

# Sidebar for file upload and India VIX input
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload stock_data.xlsx", type=['xlsx'])

# India VIX input
india_vix = st.sidebar.number_input("Enter India VIX Value:", min_value=0.0, value=15.5, step=0.1)

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully!")
    
    try:
        # Read Excel sheets
        bhav_data = pd.read_excel(uploaded_file, sheet_name='BhavData')
        index_data = pd.read_excel(uploaded_file, sheet_name='IndexData')
        high_52w_data = pd.read_excel(uploaded_file, sheet_name='52w High')
        
        st.sidebar.info(f"Loaded {len(bhav_data)} records from BhavData")
        
        # Data preprocessing
        bhav_data['DATE1'] = pd.to_datetime(bhav_data['DATE1'])
        index_data['Index Date'] = pd.to_datetime(index_data['Index Date'])
        
        # Filter for Nifty 500 index
        nifty_500 = index_data[index_data['Index Name'] == 'Nifty 500'].copy()
        nifty_500 = nifty_500.sort_values('Index Date').reset_index(drop=True)
        
        # Get unique symbols
        symbols = bhav_data['SYMBOL'].unique()
        
        # Initialize results dataframe
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Ensure we have enough Nifty 500 data
        if len(nifty_500) < 56:
            st.error("Insufficient Nifty 500 data. Need at least 56 days of index data.")
            st.stop()
        
        # Calculate technical indicators for each symbol
        for idx, symbol in enumerate(symbols):
            status_text.text(f"Processing {symbol}... ({idx+1}/{len(symbols)})")
            progress_bar.progress((idx + 1) / len(symbols))
            
            # Get stock data
            stock_data = bhav_data[bhav_data['SYMBOL'] == symbol].copy()
            stock_data = stock_data.sort_values('DATE1').reset_index(drop=True)
            
            # Skip if insufficient data
            if len(stock_data) < 56:
                continue
            
            # Get 52-week high data
            high_52w = high_52w_data[high_52w_data['SYMBOL'] == symbol]
            if len(high_52w) > 0:
                week_52_high = high_52w.iloc[0]['Adjusted_52_Week_High']
            else:
                week_52_high = stock_data['HIGH_PRICE'].max()
            
            # Calculate Returns
            current_close = stock_data.iloc[-1]['CLOSE_PRICE']
            close_55d_ago = stock_data.iloc[-56]['CLOSE_PRICE']
            close_21d_ago = stock_data.iloc[-22]['CLOSE_PRICE']
            
            S55d_Returns = current_close / close_55d_ago
            S21d_Returns = current_close / close_21d_ago
            
            # Nifty 500 Returns
            nifty_current = nifty_500.iloc[-1]['Closing Index Value']
            nifty_55d_ago = nifty_500.iloc[-56]['Closing Index Value']
            nifty_21d_ago = nifty_500.iloc[-22]['Closing Index Value']
            
            I55d_Returns = nifty_current / nifty_55d_ago
            I21d_Returns = nifty_current / nifty_21d_ago
            
            # Relative Strength
            RS55 = S55d_Returns / I55d_Returns
            RS21 = S21d_Returns / I21d_Returns
            
            # 52-week High Zone
            ltp = current_close
            wHZ_52 = week_52_high / ltp
            
            # RSI Calculation (21-day)
            def calculate_rsi(data, period=21):
                delta = data['CLOSE_PRICE'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            stock_data['RSI21'] = calculate_rsi(stock_data, 21)
            RSI21 = stock_data.iloc[-1]['RSI21']
            
            # Money Flow Index (MFI) Calculation using Delivery Data
            def calculate_mfi(data, period):
                typical_price = (data['HIGH_PRICE'] + data['LOW_PRICE'] + data['CLOSE_PRICE']) / 3
                raw_money_flow = typical_price * data['DELIV_QTY']
                
                positive_flow = []
                negative_flow = []
                
                for i in range(1, len(data)):
                    if typical_price.iloc[i] > typical_price.iloc[i-1]:
                        positive_flow.append(raw_money_flow.iloc[i])
                        negative_flow.append(0)
                    elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                        positive_flow.append(0)
                        negative_flow.append(raw_money_flow.iloc[i])
                    else:
                        positive_flow.append(0)
                        negative_flow.append(0)
                
                positive_flow = [0] + positive_flow
                negative_flow = [0] + negative_flow
                
                positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
                negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
                
                mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
                return mfi
            
            stock_data['MFI55_D'] = calculate_mfi(stock_data, 55)
            stock_data['MFI21_D'] = calculate_mfi(stock_data, 21)
            
            MFI55_D = stock_data.iloc[-1]['MFI55_D']
            MFI21_D = stock_data.iloc[-1]['MFI21_D']
            
            # Chaikin Money Flow (CMF) Calculation using Delivery Data
            def calculate_cmf(data, period):
                mf_multiplier = ((data['CLOSE_PRICE'] - data['LOW_PRICE']) - (data['HIGH_PRICE'] - data['CLOSE_PRICE'])) / (data['HIGH_PRICE'] - data['LOW_PRICE'])
                mf_multiplier = mf_multiplier.fillna(0)
                mf_volume = mf_multiplier * data['DELIV_QTY']
                cmf = mf_volume.rolling(window=period).sum() / data['DELIV_QTY'].rolling(window=period).sum()
                return cmf
            
            stock_data['CMF21_D'] = calculate_cmf(stock_data, 21)
            stock_data['CMF55_D'] = calculate_cmf(stock_data, 55)
            
            CMF21_D = stock_data.iloc[-1]['CMF21_D']
            CMF55_D = stock_data.iloc[-1]['CMF55_D']
            
            # Derived Ratios
            AD = MFI21_D / MFI55_D if MFI55_D != 0 else 0
            Strength_Mom = MFI21_D / RSI21 if RSI21 != 0 else 0
            Mom_Osc = RS21 / RS55 if RS55 != 0 else 0
            PSR = CMF21_D / CMF55_D if CMF55_D != 0 else 0
            PMA = ((CMF21_D + (MFI21_D / 50)) * ((70 - RSI21) / 40))
            
            # Calculate all ratios for historical data
            stock_data['AD'] = stock_data['MFI21_D'] / stock_data['MFI55_D']
            stock_data['Strength_Mom'] = stock_data['MFI21_D'] / stock_data['RSI21']
            stock_data['Mom_Osc'] = np.nan
            stock_data['PSR'] = stock_data['CMF21_D'] / stock_data['CMF55_D']
            stock_data['PMA'] = ((stock_data['CMF21_D'] + (stock_data['MFI21_D'] / 50)) * ((70 - stock_data['RSI21']) / 40))
            
            # Calculate RS21 and RS55 for each day
            # Make sure we align stock_data with nifty_500 by date
            stock_data_len = len(stock_data)
            nifty_len = len(nifty_500)
            
            # We'll calculate from the latest data backwards
            for i in range(stock_data_len):
                # Calculate corresponding nifty index (assuming dates are aligned)
                nifty_idx = nifty_len - stock_data_len + i
                
                if nifty_idx >= 55 and i >= 55:
                    s55_ret = stock_data.iloc[i]['CLOSE_PRICE'] / stock_data.iloc[i-55]['CLOSE_PRICE']
                    i55_ret = nifty_500.iloc[nifty_idx]['Closing Index Value'] / nifty_500.iloc[nifty_idx-55]['Closing Index Value']
                    rs55_val = s55_ret / i55_ret
                else:
                    rs55_val = np.nan
                
                if nifty_idx >= 21 and i >= 21:
                    s21_ret = stock_data.iloc[i]['CLOSE_PRICE'] / stock_data.iloc[i-21]['CLOSE_PRICE']
                    i21_ret = nifty_500.iloc[nifty_idx]['Closing Index Value'] / nifty_500.iloc[nifty_idx-21]['Closing Index Value']
                    rs21_val = s21_ret / i21_ret
                else:
                    rs21_val = np.nan
                
                if not np.isnan(rs21_val) and not np.isnan(rs55_val) and rs55_val != 0:
                    stock_data.loc[i, 'Mom_Osc'] = rs21_val / rs55_val
            
            # Risk Parameters Calculation (55 days)
            last_55_days = stock_data.tail(55).copy()
            
            # Maximum Drawdown
            cumulative_returns = (1 + last_55_days['CLOSE_PRICE'].pct_change()).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sortino Ratio
            returns = last_55_days['CLOSE_PRICE'].pct_change().dropna()
            downside_returns = returns[returns < 0]
            expected_return = returns.mean()
            downside_std = downside_returns.std()
            sortino_ratio = expected_return / downside_std if downside_std != 0 else 0
            
            # Beta calculation
            stock_returns = last_55_days['CLOSE_PRICE'].pct_change().dropna()
            nifty_returns = nifty_500.tail(55)['Closing Index Value'].pct_change().dropna()
            
            if len(stock_returns) == len(nifty_returns):
                covariance = np.cov(stock_returns, nifty_returns)[0, 1]
                market_variance = np.var(nifty_returns)
                beta = covariance / market_variance if market_variance != 0 else 1
            else:
                beta = 1
            
            # Alpha calculation (annualized)
            risk_free_rate = 0.07 / 252  # Assuming 7% annual risk-free rate
            stock_return = (last_55_days.iloc[-1]['CLOSE_PRICE'] / last_55_days.iloc[0]['CLOSE_PRICE']) - 1
            market_return = (nifty_500.tail(55).iloc[-1]['Closing Index Value'] / nifty_500.tail(55).iloc[0]['Closing Index Value']) - 1
            alpha = stock_return - (risk_free_rate + beta * (market_return - risk_free_rate))
            alpha_annualized = alpha * (252 / 55)
            
            # PVA Signals
            avg_delivery_7d = stock_data['DELIV_QTY'].tail(7).mean()
            
            today_close = stock_data.iloc[-1]['CLOSE_PRICE']
            yesterday_close = stock_data.iloc[-2]['CLOSE_PRICE']
            day_before_yesterday_close = stock_data.iloc[-3]['CLOSE_PRICE']
            
            today_delivery = stock_data.iloc[-1]['DELIV_QTY']
            yesterday_delivery = stock_data.iloc[-2]['DELIV_QTY']
            
            # tbyb (Buy PVA Signal)
            tbyb = (today_close > yesterday_close and today_delivery > avg_delivery_7d and 
                    yesterday_close > day_before_yesterday_close and yesterday_delivery > avg_delivery_7d)
            
            # tsys (Sell PVA Signal)
            tsys = (today_close < yesterday_close and today_delivery > avg_delivery_7d and 
                    yesterday_close < day_before_yesterday_close and yesterday_delivery > avg_delivery_7d)
            
            pva_signal = 'Buy' if tbyb else ('Sell' if tsys else 'None')
            
            # Check for RS21 crosses
            rs21_yesterday = np.nan
            if len(stock_data) >= 2:
                i = len(stock_data) - 2
                if i >= 21:
                    s21_ret_y = stock_data.iloc[i]['CLOSE_PRICE'] / stock_data.iloc[i-21]['CLOSE_PRICE']
                    i21_ret_y = nifty_500.iloc[i]['Closing Index Value'] / nifty_500.iloc[i-21]['Closing Index Value']
                    rs21_yesterday = s21_ret_y / i21_ret_y
            
            rs21_crosses_above_1 = (RS21 > 1 and rs21_yesterday <= 1) if not np.isnan(rs21_yesterday) else False
            rs21_crosses_below_1 = (RS21 < 1 and rs21_yesterday >= 1) if not np.isnan(rs21_yesterday) else False
            
            # Store results with risk parameters
            results.append({
                'SYMBOL': symbol,
                'RS21': RS21,
                'RS55': RS55,
                'MFI21_D': MFI21_D,
                'MFI55_D': MFI55_D,
                'RSI21': RSI21,
                'CMF21_D': CMF21_D,
                'CMF55_D': CMF55_D,
                '52wHZ': wHZ_52,
                'AD': AD,
                'Strength_Mom': Strength_Mom,
                'Mom_Osc': Mom_Osc,
                'PSR': PSR,
                'PMA': PMA,
                'PVA': pva_signal,
                'RS21_crosses_above_1': rs21_crosses_above_1,
                'RS21_crosses_below_1': rs21_crosses_below_1,
                'Max_Drawdown': max_drawdown,
                'Sortino_Ratio': sortino_ratio,
                'Beta': beta,
                'Alpha_Annualized': alpha_annualized,
                'stock_data': stock_data
            })
        
        progress_bar.empty()
        status_text.empty()
        
        # Create results dataframe
        df_results = pd.DataFrame(results)
        
        # Calculate percentiles for risk parameters
        df_results['p_MaxDD'] = df_results['Max_Drawdown'].rank(pct=True)  # Higher percentile = less drawdown (better)
        df_results['p_Sortino'] = df_results['Sortino_Ratio'].rank(pct=True)  # Higher percentile = higher Sortino (better)
        df_results['p_Beta'] = 1 - df_results['Beta'].rank(pct=True)  # Higher percentile = lower Beta (better)
        df_results['p_Alpha'] = df_results['Alpha_Annualized'].rank(pct=True)  # Higher percentile = higher Alpha (better)
        
        # Calculate RATS and vRATS
        df_results['RATS'] = (df_results['p_MaxDD'] + df_results['p_Sortino'] + 
                              df_results['p_Beta'] + df_results['p_Alpha'])
        df_results['RATS_percentile'] = df_results['RATS'].rank(pct=True)
        df_results['vRATS'] = df_results['RATS_percentile'] * (india_vix / 15.5)
        
        # Entry Condition
        entry_condition = (
            (df_results['PMA'] > 0.5) &
            (df_results['PSR'] > 0.2) &
            (df_results['AD'] > 1) &
            (df_results['Strength_Mom'] > 1.05) &
            (
                ((df_results['RS55'] < 1) & df_results['RS21_crosses_above_1']) |
                ((df_results['Mom_Osc'] > 1.2) & df_results['RS21_crosses_above_1'])
            ) &
            (df_results['Mom_Osc'] > 1) &
            (df_results['52wHZ'] < 1.3)
        )
        
        # Exit Condition
        exit_condition = (
            (df_results['PMA'] < 0.5) &
            ((df_results['Mom_Osc'] < 1) | (df_results['Mom_Osc'] < 0.85)) &
            df_results['RS21_crosses_below_1'] &
            (df_results['Strength_Mom'] < 1.05)
        )
        
        # Add Stocks Condition
        add_condition = (
            (df_results['PMA'] > 0.5) &
            (df_results['PSR'] > 0.2) &
            (df_results['RS21'] > 1) &
            (df_results['Mom_Osc'] > 1) &
            (df_results['Strength_Mom'] > 1.07) &
            (df_results['52wHZ'] < 0.75) &
            (df_results['PVA'] == 'Buy')
        )
        
        # Book Profits Condition
        book_profits_condition = (
            ((df_results['RS55'] > 1.07) & (df_results['Mom_Osc'] < 1.07) & df_results['RS21_crosses_below_1']) |
            ((df_results['PSR'] < 0.1) & (df_results['PVA'] == 'Sell'))
        )
        
        # Create filtered dataframes
        entry_df = df_results[entry_condition].copy()
        entry_df['Signal'] = 'Entry'
        
        exit_df = df_results[exit_condition].copy()
        exit_df['Signal'] = 'Exit'
        
        add_df = df_results[add_condition].copy()
        add_df['Signal'] = 'Add'
        
        book_df = df_results[book_profits_condition].copy()
        book_df['Signal'] = 'Book'
        
        # Display columns
        display_cols = ['SYMBOL', 'Signal', 'PVA', 'RS21', 'RS55', 'MFI21_D', 'MFI55_D', 
                       'RSI21', 'CMF21_D', 'CMF55_D', '52wHZ', 'AD', 'Strength_Mom', 
                       'Mom_Osc', 'PSR', 'PMA', 'vRATS']
        
        # Display tables
        st.header("📊 Trading Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Entry Signals")
            if len(entry_df) > 0:
                entry_display = entry_df[display_cols].sort_values('SYMBOL')
                st.dataframe(entry_display.style.format({
                    'RS21': '{:.4f}', 'RS55': '{:.4f}', 'MFI21_D': '{:.2f}', 'MFI55_D': '{:.2f}',
                    'RSI21': '{:.2f}', 'CMF21_D': '{:.4f}', 'CMF55_D': '{:.4f}', '52wHZ': '{:.4f}',
                    'AD': '{:.4f}', 'Strength_Mom': '{:.4f}', 'Mom_Osc': '{:.4f}', 'PSR': '{:.4f}',
                    'PMA': '{:.4f}', 'vRATS': '{:.4f}'
                }), use_container_width=True)
            else:
                st.info("No entry signals found.")
        
        with col2:
            st.subheader("🔴 Exit Signals")
            if len(exit_df) > 0:
                exit_display = exit_df[display_cols].sort_values('SYMBOL')
                st.dataframe(exit_display.style.format({
                    'RS21': '{:.4f}', 'RS55': '{:.4f}', 'MFI21_D': '{:.2f}', 'MFI55_D': '{:.2f}',
                    'RSI21': '{:.2f}', 'CMF21_D': '{:.4f}', 'CMF55_D': '{:.4f}', '52wHZ': '{:.4f}',
                    'AD': '{:.4f}', 'Strength_Mom': '{:.4f}', 'Mom_Osc': '{:.4f}', 'PSR': '{:.4f}',
                    'PMA': '{:.4f}', 'vRATS': '{:.4f}'
                }), use_container_width=True)
            else:
                st.info("No exit signals found.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("➕ Add Stocks")
            if len(add_df) > 0:
                add_display = add_df[display_cols].sort_values('SYMBOL')
                st.dataframe(add_display.style.format({
                    'RS21': '{:.4f}', 'RS55': '{:.4f}', 'MFI21_D': '{:.2f}', 'MFI55_D': '{:.2f}',
                    'RSI21': '{:.2f}', 'CMF21_D': '{:.4f}', 'CMF55_D': '{:.4f}', '52wHZ': '{:.4f}',
                    'AD': '{:.4f}', 'Strength_Mom': '{:.4f}', 'Mom_Osc': '{:.4f}', 'PSR': '{:.4f}',
                    'PMA': '{:.4f}', 'vRATS': '{:.4f}'
                }), use_container_width=True)
            else:
                st.info("No add signals found.")
        
        with col4:
            st.subheader("💰 Book Profits")
            if len(book_df) > 0:
                book_display = book_df[display_cols].sort_values('SYMBOL')
                st.dataframe(book_display.style.format({
                    'RS21': '{:.4f}', 'RS55': '{:.4f}', 'MFI21_D': '{:.2f}', 'MFI55_D': '{:.2f}',
                    'RSI21': '{:.2f}', 'CMF21_D': '{:.4f}', 'CMF55_D': '{:.4f}', '52wHZ': '{:.4f}',
                    'AD': '{:.4f}', 'Strength_Mom': '{:.4f}', 'Mom_Osc': '{:.4f}', 'PSR': '{:.4f}',
                    'PMA': '{:.4f}', 'vRATS': '{:.4f}'
                }), use_container_width=True)
            else:
                st.info("No book profit signals found.")
        
        # Graph Generation Section
        st.header("📈 Generate Ratio Graphs")
        
        # Sidebar: Symbol selection dropdown
        st.sidebar.header("Graph Selection")
        
        # Get all unique symbols from screener results
        all_symbols = pd.concat([entry_df, exit_df, add_df, book_df])['SYMBOL'].unique() if len(pd.concat([entry_df, exit_df, add_df, book_df])) > 0 else []
        
        if len(all_symbols) > 0:
            selected_symbol = st.sidebar.selectbox("Select a stock symbol to view its graph:", sorted(all_symbols))
            generate_button = st.sidebar.button("Generate Graph", type="primary")
            
            if generate_button:
                # Get stock data for selected symbol
                symbol_data = df_results[df_results['SYMBOL'] == selected_symbol].iloc[0]['stock_data']
                
                if len(symbol_data) >= 55:
                    # Get last 60 days of data
                    plot_data = symbol_data.tail(60).copy()
                    
                    # Calculate vRATS for each day
                    vrats_value = df_results[df_results['SYMBOL'] == selected_symbol].iloc[0]['vRATS']
                    plot_data['vRATS'] = vrats_value
                    
                    # Create plotly figure
                    fig = go.Figure()
                    
                    # Add traces for each metric
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['AD'], 
                                            mode='lines', name='AD Ratio', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['Strength_Mom'], 
                                            mode='lines', name='Strength Mom Ratio', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['Mom_Osc'], 
                                            mode='lines', name='Momentum Oscillator', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['PSR'], 
                                            mode='lines', name='PSR', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['PMA'], 
                                            mode='lines', name='PMA', line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=plot_data['DATE1'], y=plot_data['vRATS'], 
                                            mode='lines', name='vRATS Score', line=dict(color='black', width=3)))
                    
                    # Add threshold lines
                    fig.add_hline(y=1, line_dash="dash", line_color="red", 
                                 annotation_text="Threshold at 1", annotation_position="right")
                    fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                                 annotation_text="Threshold at 0.5", annotation_position="right")
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Technical Indicators for {selected_symbol}",
                        xaxis_title="Date",
                        yaxis_title="Ratio Value",
                        hovermode='x unified',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display 21-day table
                    st.subheader(f"Rolling 21-Day Data for {selected_symbol}")
                    table_data = plot_data.tail(21)[['DATE1', 'AD', 'Strength_Mom', 'Mom_Osc', 'PSR', 'PMA', 'vRATS']].copy()
                    table_data['DATE1'] = table_data['DATE1'].dt.strftime('%Y-%m-%d')
                    st.dataframe(table_data.style.format({
                        'AD': '{:.4f}', 'Strength_Mom': '{:.4f}', 'Mom_Osc': '{:.4f}',
                        'PSR': '{:.4f}', 'PMA': '{:.4f}', 'vRATS': '{:.4f}'
                    }), use_container_width=True)
                else:
                    st.warning(f"Insufficient data for {selected_symbol}. Need at least 55 days of historical data.")
        else:
            st.sidebar.info("No symbols available. Complete screening first.")
            st.info("No symbols available for graphing. Please check the screening results above.")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload the stock_data.xlsx file to begin analysis.")
    st.markdown("""
    ### Instructions:
    1. Upload your **stock_data.xlsx** file using the sidebar
    2. Enter the current **India VIX** value
    3. The application will automatically calculate all technical indicators
    4. View trading signals in four categories: Entry, Exit, Add Stocks, and Book Profits
    5. Select a symbol to view detailed ratio graphs
    
    ### Required Excel Sheets:
    - **BhavData**: Historical stock price data (120 days)
    - **IndexData**: Index data including Nifty 500
    - **52w High**: 52-week high/low data for stocks
    """)
