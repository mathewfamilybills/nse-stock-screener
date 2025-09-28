import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm  # Required for Alpha/Beta OLS regression
from scipy.stats import linregress

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="NSE Stock Screener")

st.title("NSE Stock Screener for Short-Term Momentum Trading 📈")
st.write("Upload your data file to analyze stocks based on a comprehensive set of technical indicators and custom ratios.")

# --- Constants for Calculations (RISK-FREE RATE) ---
ANNUAL_TRADING_DAYS = 252
RISK_FREE_RATE_ANNUAL = 0.03  # Using 3% as a proxy for the annual risk-free rate
RISK_FREE_RATE_DAILY = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / ANNUAL_TRADING_DAYS) - 1

# --- Calculation Functions ---

def calculate_rsi(prices, period=21):
    """Calculates a robust Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use rolling mean for typical smoothing
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        # Add a small constant to avoid division by zero
        rs = avg_gain / (avg_loss + 1e-9) 
    
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

def calculate_alpha_beta(stock_returns, benchmark_returns, risk_free_rate_daily, annual_trading_days=252):
    """
    Calculates Alpha and Beta using linear regression (CAPM).
    Returns (Alpha_Ann, Beta). Uses statsmodels for robustness.
    """
    # Ensure returns are aligned
    common_index = stock_returns.index.intersection(benchmark_returns.index)
    stock_returns = stock_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    if len(common_index) < 2:
        return np.nan, np.nan

    # Excess Returns
    stock_excess_returns = stock_returns - risk_free_rate_daily
    market_excess_returns = benchmark_returns - risk_free_rate_daily

    # Perform OLS Regression
    X = sm.add_constant(market_excess_returns)
    Y = stock_excess_returns
    
    try:
        # Filter out NaN/inf values which OLS can't handle
        valid_indices = X.isin([np.nan, np.inf, -np.inf]).any(axis=1) | Y.isin([np.nan, np.inf, -np.inf])
        X_clean = X[~valid_indices]
        Y_clean = Y[~valid_indices]
        
        if len(Y_clean) < 2: # Need at least 2 data points for regression
            return np.nan, np.nan
            
        model = sm.OLS(Y_clean, X_clean).fit()
        
        alpha_daily = model.params.get('const', np.nan)
        beta = model.params.iloc[1] if len(model.params) > 1 else np.nan

        # Annualize Alpha
        alpha_annual = (1 + alpha_daily)**annual_trading_days - 1
        return alpha_annual, beta
    except Exception:
        return np.nan, np.nan

def calculate_sortino(returns, period=55, mar_daily=RISK_FREE_RATE_DAILY):
    """
    Calculates the Sortino Ratio for a given period.
    Uses the daily risk-free rate as the MAR (Minimum Acceptable Return).
    """
    
    if len(returns) == 0:
        return np.nan
        
    # Filter returns that are below the MAR (downside returns)
    downside_returns = returns.where(returns < mar_daily, 0)
    
    # Calculate downside deviation (annualized)
    downside_std = downside_returns.std() * np.sqrt(ANNUAL_TRADING_DAYS)
    
    # Calculate the average excess return (annualized)
    annualized_return = (1 + returns.mean())**ANNUAL_TRADING_DAYS - 1
    
    # Sortino = (Annualized Return - Annualized MAR) / Downside Deviation
    
    if downside_std == 0:
        # If no downside deviation, return a high value if return > MAR, else NaN
        return np.inf if annualized_return > RISK_FREE_RATE_ANNUAL else np.nan
    
    return (annualized_return - RISK_FREE_RATE_ANNUAL) / downside_std


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
    
    # Pre-calculate Nifty 500 returns (for RS, Alpha, and Beta)
    nifty500_data = index_data[index_data['Index Name'].str.contains('nifty 500', case=False)].set_index('Date').sort_index()
    if nifty500_data.empty or len(nifty500_data) < 55:
        st.warning("Nifty 500 index data not found or insufficient. Skipping RS, Alpha, and Beta calculations.")
        nifty500_data['I55d_Returns'] = np.nan
        nifty500_data['I34d_Returns'] = np.nan
        nifty500_data['I21d_Returns'] = np.nan
        nifty500_daily_returns = pd.Series(dtype=float)
    else:
        # Calculate Index Returns for RS
        nifty500_data['I55d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(54)
        nifty500_data['I34d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(33)
        nifty500_data['I21d_Returns'] = nifty500_data['Close'] / nifty500_data['Close'].shift(20)
        # Calculate Daily Index Returns for Alpha/Beta/Sortino
        nifty500_daily_returns = nifty500_data['Close'].pct_change().dropna()


    for symbol in all_stocks:
        stock_df = bhav_data[bhav_data['Symbol'] == symbol].set_index('Date').sort_index()
        high_52w_filtered = high_data[high_data['Symbol'] == symbol]
        
        # Check for minimum data requirement for 55-day calculations
        if high_52w_filtered.empty or len(stock_df) < 55:
            continue
        
        # --- 1. Momentum and Ratio Calculations ---
        
        # Get latest values for the stock
        today_close = stock_df['Close'].iloc[-1]
        close_55d_ago = stock_df['Close'].iloc[-55]
        close_34d_ago = stock_df['Close'].iloc[-34]
        close_21d_ago = stock_df['Close'].iloc[-21]
        
        # Get corresponding index returns for today
        latest_date = stock_df.index[-1]
        
        I55d_Returns, I34d_Returns, I21d_Returns = np.nan, np.nan, np.nan
        if latest_date in nifty500_data.index:
            I55d_Returns = nifty500_data.loc[latest_date, 'I55d_Returns']
            I34d_Returns = nifty500_data.loc[latest_date, 'I34d_Returns']
            I21d_Returns = nifty500_data.loc[latest_date, 'I21d_Returns']
            
        # Calculate RS
        RS55 = (today_close / close_55d_ago) / I55d_Returns if I55d_Returns and close_55d_ago else np.nan
        RS34 = (today_close / close_34d_ago) / I34d_Returns if I34d_Returns and close_34d_ago else np.nan
        RS21 = (today_close / close_21d_ago) / I21d_Returns if I21d_Returns and close_21d_ago else np.nan
        
        # Other calculations
        _52wH = high_52w_filtered.iloc[0]['52wH']
        _52wHZ = _52wH / today_close if today_close > 0 else np.nan
        RSI21 = calculate_rsi(stock_df['Close'], period=21).iloc[-1]
        MFI55_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=55).iloc[-1]
        MFI21_V = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], period=21).iloc[-1]
        MFI21_D = calculate_mfi(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Deliverable Volume'], period=21).iloc[-1]
        
        # Derived Ratios
        AD = MFI21_D / MFI55_D if MFI55_D else np.nan
        Strength_AD = MFI21_D / MFI21_V if MFI21_V else np.nan
        Mom_Conf = MFI21_D / RSI21 if RSI21 else np.nan
        Mom_Osc = RS21 / RS55 if RS55 else np.nan

        # --- 2. Risk Metric Calculations (55-day window) ---
        
        stock_55d_df = stock_df.tail(55)
        stock_daily_returns = stock_55d_df['Close'].pct_change().dropna()
        
        # Align market data to the 55-day stock data dates
        market_55d_returns = nifty500_daily_returns.reindex(stock_daily_returns.index)
        
        # Calculate Alpha and Beta
        Alpha_Ann, Beta_55d = calculate_alpha_beta(
            stock_daily_returns, market_55d_returns, RISK_FREE_RATE_DAILY, ANNUAL_TRADING_DAYS
        )
        
        # Calculate Sortino Ratio
        Sortino_55d = calculate_sortino(stock_daily_returns, period=55, mar_daily=RISK_FREE_RATE_DAILY)

        # --- 3. A-RATS Calculation ---
        
        A_RATS = np.nan
        
        # Check if all required components are available and valid
        if Mom_Osc is not np.nan and Alpha_Ann is not np.nan and Beta_55d is not np.nan and Sortino_55d is not np.nan:
            
            # FIX: If Sortino is non-positive, assign A_RATS = 0.0 (low quality/uninvestable)
            if Sortino_55d <= 0:
                A_RATS = 0.0
            else:
                numerator = Mom_Osc + Alpha_Ann
                denominator = Beta_55d * (1 / Sortino_55d)
                
                # Check for near-zero denominator to prevent overflow
                if abs(denominator) < 1e-9:
                    A_RATS = 0.0 
                else:
                    A_RATS = numerator / denominator

        # --- 4. Signal Logic (Original RS conditions preserved) ---
        rs_entry = (RS55 > 0.93 and RS55 < 1 and RS21 > RS34 and RS34 > RS55) or (Mom_Osc > 1.2)
        rs_exit = (RS55 > 1 and RS55 < 1.07 and RS21 < RS34 and RS34 < RS55) or (Mom_Osc < 0.8)
        signal = "Entry" if rs_entry else "Exit" if rs_exit else "None"
            
        results_list.append({
            'Symbol': symbol, 'Signal': signal, 'RS21': RS21, 'RS55': RS55, 'MFI21_D': MFI21_D,
            'MFI21_V': MFI21_V, 'MFI55_D': MFI55_D, 'RSI21': RSI21, '52wHZ': _52wHZ,
            'AD': AD, 'Strength_AD': Strength_AD, 'Mom_Conf': Mom_Conf, 'Mom_Osc': Mom_Osc,
            'Beta_55d': Beta_55d, 'Alpha_Ann': Alpha_Ann, 'Sortino_55d': Sortino_55d, 'A_RATS': A_RATS
        })

    # Create DataFrame and prepare for display
    results_df = pd.DataFrame(results_list).dropna(subset=['RS21', 'RS55']).round(4)
    
    # Replace the numerical 0.0 with '0.0000' string for display consistency where needed
    results_df['A_RATS'] = results_df['A_RATS'].apply(lambda x: '0.0000' if x == 0.0 else x)


    if not results_df.empty:
        st.header("Screener Results")
        tab_entry, tab_exit, tab_all = st.tabs(["Entry Signals", "Exit Signals", "All Stocks"])

        with tab_entry:
            st.subheader("Stocks with Favorable Entry Conditions")
            st.dataframe(results_df[results_df['Signal'] == 'Entry'], use_container_width=True)
        with tab_exit:
            st.subheader("Stocks with Loss of Favorable Conditions (Exit Signals)")
            st.dataframe(results_df[results_df['Signal'] == 'Exit'], use_container_width=True)
        with tab_all:
            st.subheader("All Stocks and Their Metrics")
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

            
            plot_df = pd.DataFrame(index=graph_data.index)
            plot_df['RS21'] = graph_data['RS21']
            plot_df['RS55'] = graph_data['RS55']
            plot_df['AD'] = graph_data['MFI21_D'] / graph_data['MFI55_D']
            plot_df['Strength_AD'] = graph_data['MFI21_D'] / graph_data['MFI21_V']
            plot_df['Mom_Conf'] = graph_data['MFI21_D'] / graph_data['RSI21']
            plot_df['Mom_Osc'] = graph_data['RS21'] / graph_data['RS55']

            # For plotting, we use the final value of A-RATS from the main table
            final_metrics = results_df[results_df['Symbol'] == selected_symbol].iloc[0]
            
            if final_metrics['A_RATS'] is not np.nan:
                # Add A-RATS to the plotting DataFrame (as a static line for now)
                # Convert the '0.0000' string back to a float for plotting
                arats_value = float(final_metrics['A_RATS']) if isinstance(final_metrics['A_RATS'], str) else final_metrics['A_RATS']
                plot_df['A_RATS'] = arats_value
                
                # Filter to the last 60 days for a focused view
                plot_df_graph = plot_df.tail(60).dropna()
                
                # Select the ratios for plotting
                plot_df_graph_selected = pd.DataFrame(index=plot_df_graph.index)
                plot_df_graph_selected['AD (MFI21D / MFI55D)'] = plot_df_graph['AD']
                plot_df_graph_selected['Strength_AD (MFI21D / MFI21V)'] = plot_df_graph['Strength_AD']
                plot_df_graph_selected['Mom_Osc (RS21 / RS55)'] = plot_df_graph['Mom_Osc']
                plot_df_graph_selected['A_RATS'] = plot_df_graph['A_RATS'] # Add A-RATS
                
                fig = go.Figure()
                fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Crossover at 1")
                fig.add_hline(y=0.85, line_dash="dash", line_color="green", annotation_text="A-RATS Entry (0.85)")
                fig.add_hline(y=0.60, line_dash="dash", line_color="orange", annotation_text="A-RATS Exit (0.60)")

                # Define a set of darker, distinct colors
                colors = ['#ff7f0e', '#1f77b4', '#d62728', '#000000'] # Orange, blue, red, black

                for i, col in enumerate(plot_df_graph_selected.columns):
                    # Make A-RATS line thicker and black
                    line_style = dict(color=colors[i % len(colors)], width=2) if col == 'A_RATS' else dict(color=colors[i % len(colors)], width=1)
                    
                    fig.add_trace(go.Scatter(x=plot_df_graph_selected.index, y=plot_df_graph_selected[col], mode='lines', name=col,
                                             line=line_style)) 

                fig.update_layout(title=f'Dual Momentum Ratios & A-RATS for {selected_symbol}',
                                  xaxis_title="Date", yaxis_title="Ratio Value", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Display rolling 21-day tabular data below the graph ---
                st.subheader(f"Rolling 21-Day Data for {selected_symbol}")
                
                # Filter the plot_df for the last 21 days
                df_display = plot_df_graph_selected.tail(21).round(4)
                st.dataframe(df_display, use_container_width=True)
                
            else:
                st.warning("A-RATS calculation was not possible due to missing Beta/Alpha/Sortino data.")
            
else:
    st.info("Please upload your 'stock_data.xlsx' file to begin the analysis.")
