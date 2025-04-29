# dashboard/main.py

import sys
import os
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Logging Configuration ------------------------- #
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'dashboard.log')
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------- Package Imports ------------------------- #
try:
    
    from backtester.backtest import Backtester, load_trade_signals, load_historical_data
    from backtester.metrics import PerformanceMetrics
    from backtester.visualization import (
        plot_trade_history,
        plot_trade_history_interactive,
        plot_portfolio_performance,
        plot_portfolio_performance_interactive,
        plot_risk_metrics,
        plot_risk_metrics_interactive,
        plot_closed_trade_cumulative_returns,
        plot_closed_trade_cumulative_returns_interactive,
        calculate_average_weighted_probability,
        plot_average_weighted_probability,
        plot_average_weighted_probability_interactive,
        plot_decomposition_metrics,
        plot_monthly_stats
    )
    logger.info("Modules imported successfully.")
    
except ImportError as e:
    logger.error(f"❌ Critical Import Error: {e}")
    st.error(f"""
    ❌ Critical Import Error: {e}
    Please ensure you've installed the package with:
    ```
    pip install -e .
    ```
    from the repository root directory.
    """)
    sys.exit(1)

# ------------------------- Helper Functions ------------------------- #

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if 'backtest_run' not in st.session_state:
        st.session_state.backtest_run = False
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'trade_signals' not in st.session_state:
        st.session_state.trade_signals = None
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = pd.DataFrame()
    if 'metrics' not in st.session_state:
        st.session_state.metrics = pd.DataFrame()
    if 'monthly_stats' not in st.session_state:
        st.session_state.monthly_stats = pd.DataFrame()
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = pd.DataFrame()

def calculate_performance_metrics(trade_history: pd.DataFrame, max_drawdown: float) -> pd.DataFrame:
    """
    Calculate basic performance metrics.
    
    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades.
        max_drawdown (float): Maximum drawdown percentage.
    
    Returns:
        pd.DataFrame: DataFrame containing basic performance metrics.
    """
    try:
        total_trades = len(trade_history)
        profitable_trades = trade_history[trade_history['profit_pct'] > 0]
        win_rate = (len(profitable_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_profit = trade_history['profit_dollar'].sum()
        metrics = {
            'Total Trades': total_trades,
            'Profitable Trades': len(profitable_trades),
            'Win Rate (%)': round(win_rate, 2),
            'Total Profit ($)': round(total_profit, 2),
            'Max Intraday Drawdown (%)': round(max_drawdown, 2)
        }
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df = metrics_df.rename_axis('Metric').reset_index()
        logger.info("Basic performance metrics calculated successfully.")
        return metrics_df
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
        return pd.DataFrame()

# ------------------------- Main Function ------------------------- #

def main():
    """
    Main function to run the Streamlit dashboard.
    """
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Error during session state initialization: {e}")
        logger.error(f"Session state initialization error: {e}", exc_info=True)
        st.stop()

    st.set_page_config(page_title="Backtesting Framework Dashboard", layout="wide")
    st.title("📈 QST Backtesting Framework")
    logger.info("Streamlit page configured.")

    # ------------------------- Sidebar Configuration ------------------------- #

    st.sidebar.header("🔧 Configuration")
    st.sidebar.subheader("📂 Upload Data Files")
    uploaded_historical_file = st.sidebar.file_uploader(
        "Upload Historical Data CSV (e.g., 2020-2022_Long_Data.csv)",
        type=["csv"],
        key="historical_data_csv"
    )
    uploaded_trade_signals_file = st.sidebar.file_uploader(
        "Upload Trade Signals CSV (trade_signals.csv)",
        type=["csv"],
        key="trade_signals_csv"
    )

    # -------------- Load historical data if uploaded -------------- #
    if uploaded_historical_file is not None:
        try:
            historical_data = load_historical_data(uploaded_historical_file)
            if historical_data.empty:
                st.sidebar.error("Historical data CSV is empty or improperly formatted.")
                logger.error("Historical data CSV is empty or improperly formatted.")
            else:
                st.session_state.historical_data = historical_data
                st.sidebar.success("✅ Historical data loaded successfully!")
                st.sidebar.subheader("🔍 Sector Filter")
                if 'sector' in historical_data.columns:
                    available_sectors = historical_data['sector'].dropna().unique().tolist()
                    available_sectors.sort()
                    sector_options = ['All'] + available_sectors
                    selected_sector = st.sidebar.selectbox(
                        "Select Sector",
                        options=sector_options,
                        index=0,
                        key="selected_sector"
                    )
                else:
                    st.sidebar.warning("The 'sector' column is missing from the uploaded Historical Data file.")
                    selected_sector = 'All'
        except Exception as e:
            st.sidebar.error(f"Error loading historical data: {e}")
            logger.error(f"Error loading historical data: {e}", exc_info=True)
    else:
        st.sidebar.selectbox(
            "Select Sector",
            options=['All'],
            index=0,
            disabled=True,
            key="selected_sector"
        )

    # -------------- Load trade signals if uploaded -------------- #
    if uploaded_trade_signals_file is not None:
        try:
            trade_signals = load_trade_signals(uploaded_trade_signals_file)
            if trade_signals.empty:
                st.sidebar.error("Trade signals CSV is empty or improperly formatted.")
                logger.error("Trade signals CSV is empty or improperly formatted.")
            else:
                st.session_state.trade_signals = trade_signals
                st.sidebar.success("✅ Trade signals loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading trade signals: {e}")
            logger.error(f"Error loading trade signals: {e}", exc_info=True)

    # ------------------------- Backtest Period ------------------------- #
    st.sidebar.subheader("📅 Backtest Period")
    default_start_date = datetime.today() - timedelta(days=365)
    default_end_date = datetime.today()
    backtest_start_date = st.sidebar.date_input("Start Date", value=default_start_date, key="start_date")
    backtest_end_date = st.sidebar.date_input("End Date", value=default_end_date, key="end_date")

    # ------------------------- Backtest Parameters ------------------------- #
    st.sidebar.subheader("📈 Backtest Parameters")
    profit_target = st.sidebar.slider("🎯 Profit Target (%)", 1, 50, 5, key="profit_target") / 100
    max_holding_days = st.sidebar.slider("📅 Max Holding Days", 1, 120, 30, key="max_holding_days")
    initial_capital = st.sidebar.number_input("💰 Initial Capital ($)", value=5000000, min_value=1000, step=1000)
    allocation_percentage = st.sidebar.number_input(
        "📊 Allocation (%)", value=1.0, min_value=0.001, max_value=100.0, step=0.01
    ) / 100

    # ------------------------- Max Share Limit Enhancements ------------------------- #
    st.sidebar.subheader("📈 Max Share Limit Configuration")
    max_share_option = st.sidebar.radio(
        "Choose Max Share Limit Type",
        options=["Fixed Share Limit", "Percentage of 30-Day Avg Volume"],
        index=0,
        key="max_share_option"
    )

    if max_share_option == "Fixed Share Limit":
        max_share_limit = st.sidebar.number_input("📈 Fixed Max Share Limit", value=1000, min_value=1, step=1)
        use_max_share_percentage = False
        max_share_percentage = 0.10  # Default value, not used
    else:
        max_share_percentage = st.sidebar.slider(
            "📊 Max Share Percentage of 30-Day Avg Volume (%)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key="max_share_percentage_slider"
        ) / 100
        use_max_share_percentage = True
        max_share_limit = 1000  # Default value, not used

    # Long Filters
    st.sidebar.subheader("📊 Long Filters")
    long_min_price = st.sidebar.number_input("Long Min Price", value=0.0, step=0.1)
    long_max_price = st.sidebar.number_input("Long Max Price", value=1000.0, step=0.1)
    long_min_avg_volume = st.sidebar.number_input("Long Min Avg Volume", value=1000.0, step=100.0)
    long_min_vol = st.sidebar.number_input("Long Min Volatility", value=0.02, step=0.01)
    long_min_vol_regime_std = st.sidebar.number_input("Long Min Vol Regime Std", value=0.0, step=0.1)
    long_max_vol_regime_std = st.sidebar.number_input("Long Max Vol Regime Std", value=5.0, step=0.1)
    long_min_beta = st.sidebar.number_input("Long Min Beta", value=-1.0, step=0.1)
    long_max_beta = st.sidebar.number_input("Long Max Beta", value=2.0, step=0.1)
    long_min_z_score = st.sidebar.number_input("Long Min Z-Score", value=-3.0, step=0.1)
    long_max_z_score = st.sidebar.number_input("Long Max Z-Score", value=3.0, step=0.1)

    # Short Filters
    st.sidebar.subheader("📊 Short Filters")
    short_min_price = st.sidebar.number_input("Short Min Price", value=0.0, step=0.1)
    short_max_price = st.sidebar.number_input("Short Max Price", value=1000.0, step=0.1)
    short_min_avg_volume = st.sidebar.number_input("Short Min Avg Volume", value=1000.0, step=100.0)
    short_min_vol = st.sidebar.number_input("Short Min Volatility", value=0.02, step=0.01)
    short_min_vol_regime_std = st.sidebar.number_input("Short Min Vol Regime Std", value=0.0, step=0.1)
    short_max_vol_regime_std = st.sidebar.number_input("Short Max Vol Regime Std", value=5.0, step=0.1)
    short_min_beta = st.sidebar.number_input("Short Min Beta", value=-1.0, step=0.1)
    short_max_beta = st.sidebar.number_input("Short Max Beta", value=2.0, step=0.1)
    short_min_z_score = st.sidebar.number_input("Short Min Z-Score", value=-3.0, step=0.1)
    short_max_z_score = st.sidebar.number_input("Short Max Z-Score", value=3.0, step=0.1)

    # ------------------------- Trailing Stop Parameters ------------------------- #
    st.sidebar.subheader("📐 Long Trailing Stop Parameters")
    trailing_trigger = st.sidebar.slider(
        "Trailing Trigger (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        key="trailing_trigger"
    ) / 100
    trailing_stop = st.sidebar.slider(
        "Trailing Stop (%)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="trailing_stop"
    ) / 100
    trailing_move = st.sidebar.slider(
        "Trailing Move (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        key="trailing_move"
    ) / 100

    # ------------------------- Stop-Loss Parameters ------------------------- #
    st.sidebar.subheader("📉 Long Stop-Loss Parameters")
    stop_loss_pct = st.sidebar.slider(
        "Stop-Loss (%)",
        min_value=0.5,
        max_value=100.0,
        value=5.0,
        step=0.5,
        key="stop_loss_pct"
    ) / 100
    
    # ------------------------- Volatility-Adjusted Stop-Loss ------------------------- #
    st.sidebar.subheader("📈 Volatility-Adjusted Stop-Loss")
    use_volatility_adjusted_stop_loss = st.sidebar.checkbox(
        "Enable Volatility-Adjusted Stop-Loss",
        value=False,
        key="use_volatility_adjusted_stop_loss"
    )
    if use_volatility_adjusted_stop_loss:
        st.sidebar.info(
            "When enabled, the stop-loss percentage will be adjusted based on the stock's realized volatility."
        )

    # ------------------------- Rolling Close Window ------------------------- #
    st.sidebar.subheader("📐 Rolling Close Window")
    rolling_close_window = st.sidebar.slider(
        "Rolling Close Window (days)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        key="rolling_close_window"
    )

    # ------------------------- Weighted Probability Filter ------------------------- #
    st.sidebar.subheader("📐 Weighted Probability Filter")
    wp_filter_range = st.sidebar.slider(
        "Weighted Probability Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        key="wp_filter_range"
    )

    # ------------------------- Interactive Plot Selection ------------------------- #
    use_interactive = st.sidebar.checkbox("Use Interactive Plotly Charts?", value=False)

    # ------------------------- Run Backtest Button ------------------------- #
    run_backtest = st.sidebar.button("🔄 Run Backtest")

    # ------------------------- Backtest Execution ------------------------- #
    if run_backtest:
        # Ensure that both historical_data and trade_signals are loaded
        if st.session_state.historical_data is None:
            st.sidebar.error("Please upload your Historical Data CSV (e.g., 2020-2022_Long_Data.csv).")
            logger.error("Historical data CSV not uploaded.")
            st.stop()
        if st.session_state.trade_signals is None:
            st.sidebar.error("Please upload your Trade Signals CSV.")
            logger.error("Trade signals CSV not uploaded.")
            st.stop()

        try:
            historical_data = st.session_state.historical_data
            trade_signals = st.session_state.trade_signals

            # Validate and filter trade signals based on backtest period
            backtest_start_timestamp = pd.to_datetime(backtest_start_date).normalize()
            backtest_end_timestamp = pd.to_datetime(backtest_end_date).normalize()

            trade_signals_filtered = trade_signals[
                (trade_signals['trade_date'] >= backtest_start_timestamp) &
                (trade_signals['trade_date'] <= backtest_end_timestamp)
            ]

            if trade_signals_filtered.empty:
                st.error("No trade signals found in the specified period.")
                logger.warning("No trade signals in specified period.")
                st.stop()

            st.success("✅ Trade signals filtered successfully!")
            st.subheader("📊 Trade Signals Preview")
            st.dataframe(trade_signals_filtered.head())
            logger.info(f"Number of Trade Signals after filtering: {len(trade_signals_filtered)}")

            # Initialize Backtester with all relevant parameters
            backtester = Backtester(
                historical_data=historical_data,
                trade_signals=trade_signals_filtered,
                profit_target=profit_target,
                max_holding_days=max_holding_days,
                initial_capital=initial_capital,
                allocation_percentage=allocation_percentage,
                max_share_limit=max_share_limit,
                use_max_share_percentage=use_max_share_percentage,
                max_share_percentage=max_share_percentage,
                backtest_start_date=backtest_start_timestamp,
                backtest_end_date=backtest_end_timestamp,

                # Long Filters
                long_min_price=long_min_price,
                long_max_price=long_max_price,
                long_min_avg_volume=long_min_avg_volume,
                long_min_vol=long_min_vol,
                long_min_vol_regime=long_min_vol_regime_std,
                long_max_vol_regime=long_max_vol_regime_std,
                long_min_beta=long_min_beta,
                long_max_beta=long_max_beta,
                long_min_z_score=long_min_z_score,
                long_max_z_score=long_max_z_score,

                # Short Filters
                short_min_price=short_min_price,
                short_max_price=short_max_price,
                short_min_avg_volume=short_min_avg_volume,
                short_min_vol=short_min_vol,
                short_min_vol_regime=short_min_vol_regime_std,
                short_max_vol_regime=short_max_vol_regime_std,
                short_min_beta=short_min_beta,
                short_max_beta=short_max_beta,
                short_min_z_score=short_min_z_score,
                short_max_z_score=short_max_z_score,

                # Trailing Stop / Stop-Loss
                trailing_trigger=trailing_trigger,
                trailing_stop=trailing_stop,
                trailing_move=trailing_move,
                stop_loss_pct=stop_loss_pct,
                use_volatility_adjusted_stop_loss=use_volatility_adjusted_stop_loss,

                # Rolling Close Window
                rolling_close_window=rolling_close_window
            )

            logger.info("Backtester initialized.")

            # Execute Backtest
            st.subheader("⏳ Executing Backtest")
            backtest_progress_bar = st.progress(0)

            def update_backtest_progress(fraction_done):
                backtest_progress_bar.progress(fraction_done)

            with st.spinner("Executing Backtest..."):
                backtester.execute_backtest(progress_callback=update_backtest_progress)

            trade_history = backtester.get_trade_history()
            if trade_history.empty:
                st.warning("No trades were executed. Possibly all filters excluded them.")
                logger.warning("No trades executed.")
                st.session_state.backtest_run = False
                st.stop()

            # Store trade_history in session state
            st.session_state.trade_history = trade_history
            st.session_state.backtest_run = True
            logger.info("Backtest completed successfully.")
            st.success("✅ Backtest executed successfully!")

        except Exception as e:
            st.error(f"Error during backtest execution: {e}")
            logger.error(f"Backtest execution error: {e}", exc_info=True)
            st.stop()
            
    # ------------------------- Display Results ------------------------- #
    if st.session_state.get('backtest_run') and not st.session_state.trade_history.empty:
        trade_history = st.session_state.trade_history

        try:
            # Apply sector filter to trade_history
            selected_sector = st.session_state.get('selected_sector', 'All')
            if selected_sector != "All":
                trade_history_filtered = trade_history[trade_history['sector'] == selected_sector]
                st.write(f"📈 **Number of Trades in Sector '{selected_sector}':** {len(trade_history_filtered)}")
                logger.info(f"Trade history filtered by sector: {selected_sector}")
            else:
                trade_history_filtered = trade_history.copy()
                st.write(f"📈 **Total Number of Trades:** {len(trade_history_filtered)}")
                logger.info("All sectors selected. No filtering applied to trade history.")

            # Check if filtered trade history is empty
            if trade_history_filtered.empty:
                st.warning(f"No trades found for the selected sector '{selected_sector}'.")
                logger.warning(f"No trades found for the selected sector '{selected_sector}'.")
            else:
                st.subheader("📊 Trade History")
                display_columns = [
                    'ticker', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
                    'number_of_shares', 'capital_allocated', 'profit_pct', 'profit_dollar',
                    'holding_time_days', 'intraday_drawdown_pct', 'max_favorable_pct',
                    'status', 'exit_reason', 'avg_30_day_volume', 'realized_vol_30',
                    'weighted_probability',
                    'log_ret_sq', 'rolling_mean_30', 'rolling_std_30', 'vol_regime_std',
                    'hurst_exponent', 'rolling_beta_30', 'rsi_30'
                ]

                critical_columns = ['prediction', 'profit_dollar']
                for col in display_columns:
                    if col not in trade_history_filtered.columns:
                        if col in critical_columns:
                            trade_history_filtered[col] = 0
                        else:
                            trade_history_filtered[col] = 'N/A'

                trade_history_filtered['profit_dollar'] = pd.to_numeric(trade_history_filtered['profit_dollar'], errors='coerce').fillna(0)
                trade_history_filtered['prediction'] = pd.to_numeric(trade_history_filtered['prediction'], errors='coerce').fillna(0)

                st.dataframe(trade_history_filtered[display_columns].sort_values(by='exit_date', ascending=False))
                logger.info("Trade history displayed.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the trade history: {e}")
            logger.error(f"Error processing trade history: {e}", exc_info=True)

                    # ------------------------- Unique Ticker Analysis ------------------------- #
        st.subheader("📈 Daily Unique Tickers Analysis")
        try:
            # Ensure required columns exist
            required_columns = ['entry_date', 'exit_date', 'ticker', 'prediction']
            for col in required_columns:
                if col not in trade_history_filtered.columns:
                    st.error(f"Missing required column: {col}")
                    raise KeyError(f"Missing column: {col}")

            # Convert dates to datetime
            trade_history_filtered['entry_date'] = pd.to_datetime(trade_history_filtered['entry_date'])
            trade_history_filtered['exit_date'] = pd.to_datetime(trade_history_filtered['exit_date'])

            # Create date range for analysis
            date_range = pd.date_range(
                start=trade_history_filtered['entry_date'].min(),
                end=trade_history_filtered['exit_date'].max(),
                freq='D'
            )

            # Initialize dictionaries to track positions
            from collections import defaultdict
            daily_tickers = {
                'total': defaultdict(set),
                'long': defaultdict(set),
                'short': defaultdict(set)
            }

            # Process each trade
            for _, trade in trade_history_filtered.iterrows():
                ticker = trade['ticker']
                trade_dates = pd.date_range(start=trade['entry_date'], end=trade['exit_date'], freq='D')
                position_type = 'long' if trade['prediction'] > 0 else 'short'

                for date in trade_dates:
                    # Track in all categories
                    daily_tickers['total'][date].add(ticker)
                    daily_tickers[position_type][date].add(ticker)

            # Create DataFrame with counts
            position_counts = pd.DataFrame(index=date_range, columns=['total', 'long', 'short'])
            
            for date in date_range:
                position_counts.loc[date, 'total'] = len(daily_tickers['total'].get(date, set()))
                position_counts.loc[date, 'long'] = len(daily_tickers['long'].get(date, set()))
                position_counts.loc[date, 'short'] = len(daily_tickers['short'].get(date, set()))

            position_counts = position_counts.fillna(0).astype(int)

            # Display data
            st.write("### Daily Position Counts")
            st.dataframe(position_counts.head())

            # Create line chart with custom colors
            st.write("#### Daily Unique Tickers by Position Type")
            st.line_chart(position_counts, 
                        color=['#1f77b4', '#2ca02c', '#d62728'])  # Blue, Green, Red

        except KeyError as ke:
            st.error(f"Key error while processing position counts: {ke}")
            logger.error(f"Key error while processing position counts: {ke}", exc_info=True)
        except Exception as e:
            st.error(f"Error processing position counts: {e}")
            logger.error(f"Error processing position counts: {e}", exc_info=True)
        
                # ------------------------- Net Exposure and Net Beta ------------------------- #
        st.subheader("📊 Net Exposure and Net Beta")

        try:
                    # Calculate daily net exposure and net beta
                    trade_history_filtered['entry_date'] = pd.to_datetime(trade_history_filtered['entry_date'])
                    trade_history_filtered['exit_date'] = pd.to_datetime(trade_history_filtered['exit_date'])

                    # Create a date range for the backtest period
                    date_range = pd.date_range(
                        start=trade_history_filtered['entry_date'].min(),
                        end=trade_history_filtered['exit_date'].max(),
                        freq='D'
                    )

                    # Initialize DataFrame to store daily metrics
                    daily_metrics = pd.DataFrame(index=date_range)
                    daily_metrics['net_exposure'] = 0.0
                    daily_metrics['net_beta'] = 0.0
                    daily_metrics['total_capital'] = 0.0  # Track total capital for weighting

                    # Calculate daily net exposure and net beta
                    for _, trade in trade_history_filtered.iterrows():
                        trade_dates = pd.date_range(start=trade['entry_date'], end=trade['exit_date'], freq='D')
                        trade_value = trade['capital_allocated'] * trade['prediction']  # Long: +, Short: -
                        trade_beta = trade['rolling_beta_30'] * trade['prediction']  # Adjust beta for short positions

                        # Update daily metrics
                        daily_metrics.loc[trade_dates, 'net_exposure'] += trade_value
                        daily_metrics.loc[trade_dates, 'total_capital'] += trade['capital_allocated']
                        daily_metrics.loc[trade_dates, 'net_beta'] += trade_beta * trade['capital_allocated']

                    # Calculate weighted average beta
                    daily_metrics['net_beta'] = daily_metrics['net_beta'] / daily_metrics['total_capital']
                    daily_metrics['net_beta'] = daily_metrics['net_beta'].fillna(0)  # Handle cases where total_capital is 0

                    # Add month column for aggregation
                    daily_metrics['month'] = daily_metrics.index.to_period('M')

                    # Display daily net exposure and net beta
                    st.write("### Daily Net Exposure and Net Beta")
                    st.dataframe(daily_metrics[['net_exposure', 'net_beta']].head())

                    # Plot daily net exposure
                    st.write("#### Daily Net Exposure Over Time")
                    st.line_chart(daily_metrics['net_exposure'])

                    # Plot daily net beta
                    st.write("#### Daily Net Beta Over Time")
                    st.line_chart(daily_metrics['net_beta'])

                    # Calculate monthly net exposure and net beta
                    monthly_metrics = daily_metrics.groupby('month').agg({
                        'net_exposure': 'mean',
                        'net_beta': 'mean'
                    }).reset_index()
                    monthly_metrics['month'] = monthly_metrics['month'].astype(str)

                    # Display monthly net exposure and net beta
                    st.write("### Monthly Net Exposure and Net Beta")
                    st.dataframe(monthly_metrics)

                    # Plot monthly net exposure
                    st.write("#### Monthly Net Exposure Over Time")
                    st.bar_chart(monthly_metrics.set_index('month')['net_exposure'])

                    # Plot monthly net beta
                    st.write("#### Monthly Net Beta Over Time")
                    st.bar_chart(monthly_metrics.set_index('month')['net_beta'])

        except KeyError as ke:
            st.error(f"Key error while calculating net exposure and net beta: {ke}")
            logger.error(f"Key error while calculating net exposure and net beta: {ke}", exc_info=True)
        except Exception as e:
            st.error(f"Error processing net exposure and net beta: {e}")
            logger.error(f"Error processing net exposure and net beta: {e}", exc_info=True)
            
            
        # ------------------------- Sector Exposure ------------------------- #
        st.subheader("📊 Sector Exposure Analysis")

        try:
            # Ensure required columns exist
            required_columns = ['entry_date', 'exit_date', 'capital_allocated', 'prediction', 'sector']
            for col in required_columns:
                if col not in trade_history_filtered.columns:
                    st.error(f"Missing required column: {col}")
                    raise KeyError(f"Missing column: {col}")

            # Convert dates to datetime
            trade_history_filtered['entry_date'] = pd.to_datetime(trade_history_filtered['entry_date'])
            trade_history_filtered['exit_date'] = pd.to_datetime(trade_history_filtered['exit_date'])

            # Create a date range for the backtest period
            date_range = pd.date_range(
                start=trade_history_filtered['entry_date'].min(),
                end=trade_history_filtered['exit_date'].max(),
                freq='D'
            )

            # Initialize DataFrame to store daily sector metrics
            daily_sector_metrics = pd.DataFrame(index=date_range)

            # Get unique sectors
            sectors = trade_history_filtered['sector'].unique()

            # Calculate daily sector exposure
            for sector in sectors:
                sector_trades = trade_history_filtered[trade_history_filtered['sector'] == sector]
                
                # Initialize sector exposure column
                daily_sector_metrics[f'{sector}_exposure'] = 0.0
                
                # Calculate sector-specific exposure for each trade
                for _, trade in sector_trades.iterrows():
                    trade_dates = pd.date_range(start=trade['entry_date'], end=trade['exit_date'], freq='D')
                    trade_value = trade['capital_allocated'] * trade['prediction']  # Long: +, Short: -
                    
                    # Update daily sector exposure
                    daily_sector_metrics.loc[trade_dates, f'{sector}_exposure'] += trade_value

            # Add month column for aggregation
            daily_sector_metrics['month'] = daily_sector_metrics.index.to_period('M')

            # Display daily sector exposure for all sectors
            st.write("### Daily Sector Exposure Across All Sectors")
            st.dataframe(daily_sector_metrics[[col for col in daily_sector_metrics.columns if '_exposure' in col]].head())

            # Plot daily sector exposure with sector names
            st.write("#### Daily Sector Exposure Over Time")
            for sector in sectors:
                st.write(f"**{sector}**")  # Display sector name as header
                st.line_chart(daily_sector_metrics[f'{sector}_exposure'])

            # Calculate monthly sector exposure
            monthly_sector_metrics = daily_sector_metrics.groupby('month').agg({
                **{f'{sector}_exposure': 'mean' for sector in sectors}
            }).reset_index()
            monthly_sector_metrics['month'] = monthly_sector_metrics['month'].astype(str)

            # Display monthly sector exposure for all sectors
            st.write("### Monthly Sector Exposure Across All Sectors")
            st.dataframe(monthly_sector_metrics)

            # Plot monthly sector exposure with sector names
            st.write("#### Monthly Sector Exposure Over Time")
            for sector in sectors:
                st.write(f"**{sector}**")  # Display sector name as header
                st.bar_chart(monthly_sector_metrics.set_index('month')[f'{sector}_exposure'])

        except KeyError as ke:
            st.error(f"Key error while calculating sector exposure: {ke}")
            logger.error(f"Key error while calculating sector exposure: {ke}", exc_info=True)
        except Exception as e:
            st.error(f"Error processing sector exposure: {e}")
            logger.error(f"Error processing sector exposure: {e}", exc_info=True)

        # -------------- Basic Performance Metrics -------------- #
        try:
            if not trade_history_filtered.empty:
                st.subheader("📊 Performance Metrics (Basic)")
                max_drawdown = trade_history_filtered['intraday_drawdown_pct'].max() if not trade_history_filtered.empty else 0.0
                basic_metrics = calculate_performance_metrics(trade_history_filtered, max_drawdown=max_drawdown)
                st.table(basic_metrics)
                st.session_state.metrics = basic_metrics
                logger.info("Basic performance metrics displayed.")
            else:
                st.write("No performance metrics to display for the selected sector.")
                logger.info("No performance metrics to display due to empty trade history.")
        except Exception as e:
            st.error(f"Error displaying basic performance metrics: {e}")
            logger.error(f"Basic metrics display error: {e}", exc_info=True)

        # -------------- Advanced Performance Metrics -------------- #
        try:
            if not trade_history_filtered.empty:
                st.subheader("📊 Advanced Performance Metrics")
                apply_adv_filters = st.checkbox(
                    "Apply Price/Volume/Vol/Weighted_Probability filters again for advanced metrics?", 
                    value=False,
                    key="apply_adv_filters"
                )
                filtered_for_adv = trade_history_filtered.copy()

                if apply_adv_filters:
                    st.info("Re-filtering trade_history for advanced metrics.")
                    # Long Filters
                    if long_min_price > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['entry_price'] >= long_min_price]
                    if long_max_price > 0 and long_max_price < 999999:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['entry_price'] <= long_max_price]
                    if long_min_avg_volume > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['avg_30_day_volume'] >= long_min_avg_volume]
                    if long_min_vol > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['realized_vol_30'] >= long_min_vol]

                    # Short Filters
                    if short_min_price > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['entry_price'] >= short_min_price]
                    if short_max_price > 0 and short_max_price < 999999:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['entry_price'] <= short_max_price]
                    if short_min_avg_volume > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['avg_30_day_volume'] >= short_min_avg_volume]
                    if short_min_vol > 0:
                        filtered_for_adv = filtered_for_adv[filtered_for_adv['realized_vol_30'] >= short_min_vol]

                    # Weighted Probability Range Filter
                    wp_lower, wp_upper = wp_filter_range
                    filtered_for_adv = filtered_for_adv[
                        (filtered_for_adv['weighted_probability'] >= wp_lower) &
                        (filtered_for_adv['weighted_probability'] <= wp_upper)
                    ]

                    st.write(f"Trades after advanced filters: {len(filtered_for_adv)}")
                    logger.info(f"Advanced filters applied. Trades count: {len(filtered_for_adv)}")

                pm_adv = PerformanceMetrics(filtered_for_adv, initial_capital=initial_capital)
                adv_metrics_dict = pm_adv.calculate_metrics()
                adv_metrics_df = pd.DataFrame(list(adv_metrics_dict.items()), columns=["Metric", "Value"])
                st.table(adv_metrics_df)
                logger.info("Advanced performance metrics displayed.")
                
        
                
                # -------------- Decomposition DataFrames -------------- #
                decomposition_data = {
                    'Volatility_Regime_Std_Decomposition': pm_adv.calculate_vol_regime_std_decomposition(vol_bins=list(range(-5, 6, 1))),
                    'Hurst_Exponent_Decomposition': pm_adv.calculate_hurst_exponent_decomposition(hurst_bins=[0.0, 0.4, 0.5, 0.6, 0.8, 1.0]),
                    'Weighted_Probability_Decomposition': pm_adv.calculate_wp_decomposition(wp_bins=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0]),
                    'Rolling_Beta_Decomposition': pm_adv.calculate_rolling_beta_decomposition(beta_bins=[-2, -1, 0, 1, 2, 3, 4, 5]),
                    'RSI_Decomposition': pm_adv.calculate_rsi_decomposition(rsi_bins=[0, 30, 50, 70, 100]),
                    'Z_Score_Decomposition': pm_adv.calculate_z_score_decomposition(z_bins=[-3, -2, -1, 0, 1, 2, 3]),
                    'Volume_Decomposition': pm_adv.calculate_volume_decomposition(volume_bins=[0, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000]),
                    'Price_Decomposition': pm_adv.calculate_price_decomposition(price_bins=[0, 1, 5, 10, 50, 100, float('inf')]),
                    'Volume_Regime_Decomposition': pm_adv.calculate_volume_regime_decomposition(volume_regime_bins=list(range(-5, 6, 1)))
                }

                # -------------- Monthly Statistics -------------- #
                daily_returns_for_adv = pm_adv.calculate_portfolio_returns()
                monthly_stats_df = pm_adv.get_monthly_statistics(daily_returns_for_adv)
                st.session_state.monthly_stats = monthly_stats_df

                # -------------- Display Decomposition Tables -------------- #
                st.subheader("📊 Decomposition Metrics Tables")
                for dec_key, dec_df in decomposition_data.items():
                    st.markdown(f"### {dec_key}")
                    if isinstance(dec_df, pd.DataFrame) and not dec_df.empty:
                        st.table(dec_df)
                    else:
                        st.write("No data available for this decomposition metric.")

                # -------------- Optionally Plot Decompositions -------------- #
                st.subheader("📊 Decomposition Plots")
                dec_figures = plot_decomposition_metrics(decomposition_data, sector=selected_sector if selected_sector != "All" else None)
                for plot_name, fig_obj in dec_figures.items():
                    st.markdown(f"#### {plot_name}")
                    if isinstance(fig_obj, plt.Figure):
                        st.pyplot(fig_obj)
                    elif hasattr(fig_obj, 'to_html'):
                        st.plotly_chart(fig_obj, use_container_width=True)
                    else:
                        st.write("Unsupported figure type.")

                # -------------- Display (or Plot) Monthly Statistics -------------- #
                if not monthly_stats_df.empty:
                    st.subheader("📅 Monthly Statistics")
                    st.dataframe(monthly_stats_df)

                    # Optional: Create a monthly stats bar chart if desired
                    monthly_fig = plot_monthly_stats(monthly_stats_df)
                    if monthly_fig:
                        st.pyplot(monthly_fig)
        except Exception as e:
            st.error(f"Error calculating advanced metrics or monthly stats: {e}")
            logger.error(f"Advanced metrics/monthly stats error: {e}", exc_info=True)

        # ------------------------- Predictive Power of Weighted Probabilities ------------------------- #
        try:
            st.subheader("📈 Predictive Power of Weighted Probabilities for SPY")

            if 'exit_date' in trade_history_filtered.columns and 'weighted_probability' in trade_history_filtered.columns:
                daily_avg_wp = trade_history_filtered.groupby('exit_date')['weighted_probability'].mean().reset_index()
                daily_avg_wp.rename(columns={'exit_date': 'date', 'weighted_probability': 'avg_weighted_probability'}, inplace=True)
                st.write("### Daily Average Weighted Probability")
                st.dataframe(daily_avg_wp.head())
                logger.info("Daily average weighted probability computed.")

                # Fetch SPY's daily returns from historical_data
                historical_data = st.session_state.historical_data
                if ('timestamp' in historical_data.columns) and ('spy_ret' in historical_data.columns):
                    spy_data = historical_data[['timestamp', 'spy_ret']].dropna()
                    spy_data.rename(columns={'timestamp': 'date', 'spy_ret': 'SPY_return'}, inplace=True)
                    spy_data['date'] = pd.to_datetime(spy_data['date'], errors='coerce')
                    spy_data.dropna(subset=['date'], inplace=True)
                    st.write("### SPY Daily Returns")
                    st.dataframe(spy_data.head())
                    logger.info("SPY daily returns fetched.")
                else:
                    st.warning("SPY data with 'timestamp' and 'spy_ret' columns not found in historical data.")
                    logger.warning("SPY data with 'timestamp' and 'spy_ret' columns not found.")
                    spy_data = pd.DataFrame()

                if not spy_data.empty:
                    merged_data = pd.merge(daily_avg_wp, spy_data, on='date', how='inner')
                    if merged_data.empty:
                        st.warning("No overlapping dates between weighted probabilities and SPY returns.")
                        logger.warning("No overlapping dates between WP and SPY returns.")
                    else:
                        correlation = merged_data['avg_weighted_probability'].corr(merged_data['SPY_return'])
                        st.write(f"**Correlation (WP vs SPY Return):** {round(correlation, 4)}")
                        logger.info(f"Correlation computed: {correlation}")

                        fig, ax = plt.subplots()
                        ax.scatter(merged_data['avg_weighted_probability'], merged_data['SPY_return'], alpha=0.6)
                        if len(merged_data) >= 2:
                            slope, intercept = np.polyfit(merged_data['avg_weighted_probability'], merged_data['SPY_return'], 1)
                            ax.plot(
                                merged_data['avg_weighted_probability'],
                                slope * merged_data['avg_weighted_probability'] + intercept,
                                color='red'
                            )
                        ax.set_xlabel('Average Weighted Probability')
                        ax.set_ylabel('SPY Daily Return')
                        ax.set_title('Average Weighted Probability vs. SPY Daily Return')
                        st.pyplot(fig)
            else:
                st.warning("Required columns 'exit_date' and/or 'weighted_probability' not found in trade history.")
                logger.warning("Required columns for WP vs SPY correlation not found.")
        except Exception as e:
            st.error(f"Error analyzing predictive power of weighted probabilities: {e}")
            logger.error(f"Error analyzing WP predictive power: {e}", exc_info=True)
            
        # ------------------------- Monte Carlo Simulation ------------------------- #
        try:
            if not trade_history_filtered.empty:
                st.subheader("📊 Monte Carlo Simulation")
                st.write("Simulate potential future returns based on the daily returns distribution of your backtested portfolio.")
                with st.expander("Monte Carlo Options", expanded=False):
                    num_sims = st.number_input("Number of simulations", value=1000, min_value=1, step=100, key="mc_num_sims")
                    sim_days = st.number_input("Simulation Days", value=252, min_value=1, step=1, key="mc_sim_days")
                    sim_method = st.selectbox("Simulation Method", ["historical", "gaussian"], index=0, key="mc_sim_method")

                run_monte_carlo = st.button("Run Monte Carlo Simulation", key="run_monte_carlo")

                if run_monte_carlo:
                    try:
                        sim_results = backtester.run_monte_carlo_simulation(
                            trade_history=trade_history_filtered,
                            num_sims=num_sims,
                            sim_days=sim_days,
                            sim_method=sim_method,
                            seed=42
                        )
                        st.session_state.sim_results = sim_results
                        if sim_results.empty:
                            st.warning("Monte Carlo simulation could not be performed due to insufficient data.")
                            logger.warning("Monte Carlo simulation returned empty.")
                        else:
                            eq_curves = (1 + sim_results).cumprod() * 100
                            st.line_chart(eq_curves)
                            final_returns = eq_curves.iloc[-1, :] - 100
                            st.write("### Distribution of Final Returns (%)")
                            st.bar_chart(final_returns.value_counts(bins=20).sort_index())
                            st.write("**Mean Final Return**:", round(final_returns.mean(), 2), "%")
                            st.write("**Median Final Return**:", round(final_returns.median(), 2), "%")
                            st.write("**5% Quantile**:", round(final_returns.quantile(0.05), 2), "%")
                            st.write("**95% Quantile**:", round(final_returns.quantile(0.95), 2), "%")
                            logger.info("Monte Carlo simulation executed successfully.")
                    except Exception as e:
                        st.error(f"Error running Monte Carlo simulation: {e}")
                        logger.error(f"Monte Carlo simulation error: {e}", exc_info=True)
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation section: {e}")
            logger.error(f"Monte Carlo section error: {e}", exc_info=True)

        # ------------------------- Visualizations ------------------------- #
        try:
            if not trade_history_filtered.empty:
                st.subheader("📊 Visualizations")
                st.write("### Trade History Distributions")
                if use_interactive:
                    fig_profit, fig_holding = plot_trade_history_interactive(trade_history_filtered)
                    if fig_profit and fig_holding:
                        st.plotly_chart(fig_profit, use_container_width=True)
                        st.plotly_chart(fig_holding, use_container_width=True)
                else:
                    fig_static = plot_trade_history(trade_history_filtered)
                    if fig_static:
                        st.pyplot(fig_static)

                st.write("### Portfolio Cumulative Returns")
                pm_for_plots = PerformanceMetrics(trade_history=trade_history_filtered, initial_capital=initial_capital)
                daily_returns = pm_for_plots.calculate_portfolio_returns()
                if daily_returns.empty:
                    st.warning("No daily returns for portfolio performance plot.")
                else:
                    if use_interactive:
                        fig_int = plot_portfolio_performance_interactive(daily_returns)
                        if fig_int:
                            st.plotly_chart(fig_int, use_container_width=True)
                    else:
                        fig_port = plot_portfolio_performance(daily_returns)
                        if fig_port:
                            st.pyplot(fig_port)
                logger.info("Visualizations generated and displayed.")
            else:
                st.write("No visualizations available for the selected sector.")
                logger.info("No visualizations available due to empty trade history.")
        except Exception as e:
            st.error(f"Error generating visualizations: {e}")
            logger.error(f"Visualization generation error: {e}", exc_info=True)

        # ------------------------- Download Options ------------------------- #
        try:
            if not trade_history_filtered.empty:
                st.subheader("📥 Download Trade History")
                csv_data = trade_history_filtered.to_csv(index=False)
                st.download_button(
                    "Download Trade History as CSV",
                    data=csv_data,
                    file_name='trade_history.csv',
                    mime='text/csv'
                )
                logger.info("Trade history download button created.")
            else:
                st.write("No trade history available to download for the selected sector.")
                logger.info("No trade history available to download.")
        except Exception as e:
            st.error(f"Error creating download button for trade history: {e}")
            logger.error(f"Trade history download button error: {e}", exc_info=True)

        try:
            if not trade_history_filtered.empty and not st.session_state.metrics.empty:
                st.subheader("📥 Download Performance Metrics")
                csv_metrics = st.session_state.metrics.to_csv(index=False)
                st.download_button(
                    "Download Metrics as CSV",
                    data=csv_metrics,
                    file_name='performance_metrics.csv',
                    mime='text/csv'
                )
                logger.info("Performance metrics download button created.")
            else:
                st.write("No performance metrics available to download for the selected sector.")
                logger.info("No performance metrics available to download.")
        except Exception as e:
            st.error(f"Error creating metrics download button: {e}")
            logger.error(f"Performance metrics download button error: {e}", exc_info=True)

        try:
            if not st.session_state.monthly_stats.empty:
                st.subheader("📥 Download Monthly Statistics")
                csv_monthly = st.session_state.monthly_stats.to_csv(index=False)
                st.download_button(
                    "Download Monthly Stats as CSV",
                    data=csv_monthly,
                    file_name='monthly_statistics.csv',
                    mime='text/csv'
                )
                logger.info("Monthly statistics download button created.")
            else:
                st.write("No monthly statistics available to download.")
                logger.info("No monthly statistics to download.")
        except Exception as e:
            st.error(f"Error creating monthly stats download button: {e}")
            logger.error(f"Monthly statistics download button error: {e}", exc_info=True)

        try:
            if not st.session_state.sim_results.empty:
                st.subheader("📥 Download Monte Carlo Simulation Results")
                sim_results = st.session_state.sim_results
                csv_sim = sim_results.to_csv(index=False)
                st.download_button(
                    "Download Simulation Results as CSV",
                    data=csv_sim,
                    file_name='monte_carlo_simulation.csv',
                    mime='text/csv'
                )
                logger.info("Monte Carlo simulation results download button created.")
            else:
                st.write("No Monte Carlo simulation results available to download.")
                logger.info("No Monte Carlo simulation results to download.")
        except Exception as e:
            st.error(f"Error creating simulation results download button: {e}")
            logger.error(f"Monte Carlo simulation results download button error: {e}", exc_info=True)

        try:
            if not trade_history_filtered.empty:
                st.subheader("⚙️ Adjusted Trades Info")
                adjusted_trades = trade_history_filtered[
                    (trade_history_filtered['exit_date'] - trade_history_filtered['entry_date']).dt.days != max_holding_days
                ]
                if not adjusted_trades.empty:
                    st.warning("⚠️ Some trades had exit dates adjusted due to missing data or holidays:")
                    st.dataframe(adjusted_trades[['ticker','entry_date','exit_date','holding_time_days','exit_reason']])
                    logger.info("Adjusted trades displayed.")
                else:
                    st.success("✅ All trades executed as per the maximum holding period.")
                    logger.info("No adjusted trades to display.")
            else:
                st.write("No adjusted trades info available for the selected sector.")
                logger.info("No adjusted trades info available.")
        except Exception as e:
            st.error(f"Error displaying adjusted trades: {e}")
            logger.error(f"Adjusted trades display error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
