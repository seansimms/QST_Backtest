# backtest.py

import pandas as pd
from datetime import timedelta
import logging
import numpy as np
from bisect import bisect_left
from numba import njit
from joblib import Parallel, delayed
from typing import Optional, List, Tuple

# ------------------------- Logging Configuration ------------------------- #

logging.basicConfig(
    filename='backtest.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Change to DEBUG for more detailed logs during development
)
logger = logging.getLogger(__name__)

# ------------------------- Slippage Constant ------------------------- #
SLIPPAGE_BPS = 0.0075  # 0.75% slippage

# ------------------------- Standalone Functions ------------------------- #

def load_trade_signals(csv_file: pd.io.common.BytesIO) -> pd.DataFrame:
    """
    Loads trade signals from a CSV file, requiring 'weighted_probability'.
    
    Parameters:
    -----------
    csv_file : pd.io.common.BytesIO
        A file-like object representing the CSV file containing trade signals.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the trade signals with necessary columns.
    """
    try:
        # Reset file pointer to the beginning
        csv_file.seek(0)
        
        # Check if the file is empty
        first_char = csv_file.read(1)
        if not first_char:
            raise pd.errors.EmptyDataError("The uploaded trade signals CSV file is empty.")
        # Reset again after reading
        csv_file.seek(0)
        
        # Read CSV with proper encoding to handle potential BOM
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Log the column names for debugging
        logger.debug(f"Trade Signals CSV Columns: {df.columns.tolist()}")
        
        required_columns = ['trade_date', 'ticker', 'prediction', 'weighted_probability']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")
        
        # Convert 'trade_date' to datetime, localize to UTC, then remove timezone
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce', utc=True)
        
        # Handle any NaT values resulting from parsing errors
        if df['trade_date'].isnull().any():
            logger.warning("Some 'trade_date' entries could not be parsed and are set as NaT.")
        
        # Normalize to remove time component and make timezone-naive
        df['trade_date'] = df['trade_date'].dt.normalize().dt.tz_localize(None)
        
        # Drop any rows where 'trade_date' is NaT after localization
        initial_length = len(df)
        df = df.dropna(subset=['trade_date'])
        final_length = len(df)
        if final_length < initial_length:
            logger.warning(f"Dropped {initial_length - final_length} rows due to NaT 'trade_date'.")
    
        # Ensure 'prediction' is either 1 or -1
        if not df['prediction'].isin([1, -1]).all():
            raise ValueError("Column 'prediction' must contain only 1 (Long) or -1 (Short).")
        
        # Weighted probability should be numeric
        df['weighted_probability'] = pd.to_numeric(df['weighted_probability'], errors='coerce')
        if df['weighted_probability'].isnull().any():
            logger.warning("Some 'weighted_probability' entries could not be parsed. Setting them to NaN.")
        
        # Drop rows with NaN in 'weighted_probability'
        initial_length = len(df)
        df = df.dropna(subset=['weighted_probability'])
        final_length = len(df)
        if final_length < initial_length:
            logger.warning(f"Dropped {initial_length - final_length} rows due to NaN 'weighted_probability'.")
        
        # Convert 'ticker' to categorical for performance
        df['ticker'] = df['ticker'].astype('category')
        
        logger.info("Trade signals (including weighted_probability) loaded successfully from CSV.")
        return df
    except pd.errors.EmptyDataError as e:
        logger.error(f"Trade signals CSV file is empty: {e}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error loading trade signals: {e}", exc_info=True)
        raise e

def load_historical_data(csv_file: pd.io.common.BytesIO) -> pd.DataFrame:
    try:
        # Reset file pointer to the beginning
        csv_file.seek(0)
        
        # Check if the file is empty
        first_char = csv_file.read(1)
        if not first_char:
            raise pd.errors.EmptyDataError("The uploaded historical data CSV file is empty.")
        # Reset again after reading
        csv_file.seek(0)
        
        # Read CSV with proper encoding to handle potential BOM
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        # Strip whitespace and normalize column names
        df.columns = df.columns.str.strip().str.lower()
        logger.debug(f"Columns in CSV file: {df.columns.tolist()}")

        # Define the expected column name for timestamp
        expected_timestamp_col = 'timestamp'
        
        # Check if 'timestamp' exists; if not, try alternative names
        if expected_timestamp_col not in df.columns:
            alternative_names = ['timestamp', 'date', 'trade_date']
            renamed = False
            for alt in alternative_names:
                if alt in df.columns:
                    df.rename(columns={alt: 'timestamp'}, inplace=True)
                    logger.warning(f"Renamed '{alt}' to 'timestamp'.")
                    renamed = True
                    break
            if not renamed:
                logger.error("CSV must contain a column for timestamps (e.g., 'timestamp', 'date', 'trade_date').")
                raise ValueError("CSV must contain a 'timestamp' column.")
        
        # Verify required columns
        required_columns = [
            'timestamp', 'volume', 'vw', 'open', 'close', 'high', 'low', 'n',
            'avg_volume_30', 'avg_volume_90','volume_regime','ret', 'log_ret', 'log_ret_sq', 'rolling_mean_30',
            'rolling_std_30', 'vol_regime_std', 'realized_vol_30',
            'hurst_exponent', 'ticker', 'otc', 'rolling_beta_30', 'rsi_30', 'sector', 'z_score_30'
        ]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"CSV must contain '{col}' column.")
                raise ValueError(f"CSV must contain '{col}' column.")
        
        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        
        # Log rows with invalid timestamps
        if df['timestamp'].isnull().any():
            logger.warning(f"Rows with invalid 'timestamp':\n{df[df['timestamp'].isnull()]}")
        
        # Drop rows with NaT in 'timestamp'
        initial_len = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        final_len = len(df)
        if initial_len != final_len:
            logger.warning(f"Dropped {initial_len - final_len} rows due to invalid 'timestamp' values.")
        
        # Normalize timestamps and convert 'ticker' to categorical
        df['timestamp'] = df['timestamp'].dt.normalize().dt.tz_localize(None)
        df['ticker'] = df['ticker'].astype('category')

        # Ensure DataFrame is not empty
        if df.empty:
            logger.error("Historical data is empty after processing.")
            raise ValueError("Historical data is empty after processing.")

        # Sort by 'ticker' and 'timestamp'
        if 'ticker' not in df.columns or 'timestamp' not in df.columns:
            logger.error("'ticker' or 'timestamp' column is missing in the processed DataFrame.")
            raise ValueError("'ticker' or 'timestamp' column is required for sorting.")
        df.sort_values(['ticker', 'timestamp'], inplace=True)
        
        logger.info("Historical data loaded and processed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error loading historical data: {e}", exc_info=True)
        raise

# ------------------------- Numba-Optimized Functions ------------------------- #

@njit
def calculate_exit_price_numba(entry_price: float,
                               low_prices: np.ndarray,
                               high_prices: np.ndarray,
                               stop_loss_pct: float,
                               trailing_trigger: float,
                               trailing_stop: float,
                               trailing_move: float,
                               prediction: int,
                               profit_target: float) -> Tuple[float, int, int]:
    """
    Numba-optimized function to calculate exit price based on stop-loss, trailing stop, and profit target logic.
    
    Parameters:
    -----------
    entry_price : float
        The price at which the position was entered.
    low_prices : np.ndarray
        Array of low prices for each day in the holding period.
    high_prices : np.ndarray
        Array of high prices for each day in the holding period.
    stop_loss_pct : float
        Percentage below (for long) or above (for short) the entry price to trigger a stop-loss.
    trailing_trigger : float
        Profit percentage at which the trailing stop becomes active.
    trailing_stop : float
        Percentage below the max profit (for long) or above the min price (for short) for the trailing stop.
    trailing_move : float
        Minimum price movement to update the trailing stop.
    prediction : int
        1 for long position, -1 for short position.
    profit_target : float
        Percentage above (for long) or below (for short) the entry price to trigger an exit.
    
    Returns:
    --------
    exit_price : float
        The price at which the position is exited.
    exit_date_index : int
        The index (day) at which the exit occurs.
    exit_reason_code : int
        0: Max Holding, 1: Stop Loss, 2: Trailing Stop, 3: Profit Target
    """
    # Calculate stop-loss and profit target prices based on position type
    if prediction == 1:
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        profit_target_price = entry_price * (1 + profit_target)
    else:
        stop_loss_price = entry_price * (1 + stop_loss_pct)
        profit_target_price = entry_price * (1 - profit_target)
    
    # Initialize variables
    trail_active = False
    current_trail_price = 0.0
    max_favorable_pct = 0.0
    exit_price = 0.0
    exit_date_index = len(low_prices) - 1
    exit_reason_code = 0  # 0: Max Holding Period
    
    # Iterate through price data
    for i in range(len(low_prices)):
        if prediction == 1:
            # Long position
            # Check profit target
            if high_prices[i] >= profit_target_price:
                exit_price = profit_target_price
                exit_date_index = i
                exit_reason_code = 3  # Profit Target Hit
                break
            # Stop-loss
            if low_prices[i] <= stop_loss_price:
                exit_price = stop_loss_price
                exit_date_index = i
                exit_reason_code = 1  # Stop Loss Hit
                break
            # Trailing stop
            current_profit_pct = (high_prices[i] - entry_price) / entry_price
            if (not trail_active) and (current_profit_pct >= trailing_trigger):
                trail_active = True
                current_trail_price = entry_price * (1 + trailing_trigger - trailing_stop)
            if trail_active:
                new_trail_price = entry_price * (1 + current_profit_pct - trailing_stop)
                if new_trail_price > current_trail_price + trailing_move:
                    current_trail_price += trailing_move
                if low_prices[i] <= current_trail_price:
                    exit_price = current_trail_price
                    exit_date_index = i
                    exit_reason_code = 2  # Trailing Stop Hit
                    break
        else:
            # Short position
            # Check profit target
            if low_prices[i] <= profit_target_price:
                exit_price = profit_target_price
                exit_date_index = i
                exit_reason_code = 3  # Profit Target Hit
                break
            # Stop-loss
            if high_prices[i] >= stop_loss_price:
                exit_price = stop_loss_price
                exit_date_index = i
                exit_reason_code = 1  # Stop Loss Hit
                break
            # Trailing stop
            current_profit_pct = (entry_price - low_prices[i]) / entry_price
            if (not trail_active) and (current_profit_pct >= trailing_trigger):
                trail_active = True
                current_trail_price = entry_price * (1 - trailing_trigger + trailing_stop)
            if trail_active:
                new_trail_price = entry_price * (1 - current_profit_pct + trailing_stop)
                if new_trail_price < current_trail_price - trailing_move:
                    current_trail_price -= trailing_move
                if high_prices[i] >= current_trail_price:
                    exit_price = current_trail_price
                    exit_date_index = i
                    exit_reason_code = 2  # Trailing Stop Hit
                    break
        # Update max_favorable_pct (optional, can be returned if needed)
        if prediction == 1:
            favorable_pct = (high_prices[i] - entry_price) / entry_price * 100
        else:
            favorable_pct = (entry_price - low_prices[i]) / entry_price * 100
        if favorable_pct > max_favorable_pct:
            max_favorable_pct = favorable_pct
    
    # If no exit condition is met, exit at the last day's price
    if exit_price == 0.0:
        exit_price = high_prices[-1] if prediction == 1 else low_prices[-1]
    
    return exit_price, exit_date_index, exit_reason_code

# ------------------------- Backtester Class ------------------------- #

class Backtester:
    """
    A class to perform backtesting with realistic capital management, dynamic position sizing,
    and comprehensive risk controls.
    """

    EXIT_REASON_MAP = {
        0: 'Max Holding Period',
        1: 'Stop Loss Hit',
        2: 'Trailing Stop Hit',
        3: 'Profit Target Hit'
    }

    def __init__(
        self,
        historical_data: pd.DataFrame,
        trade_signals: pd.DataFrame,
        profit_target: float = 0.05,
        max_holding_days: int = 30,
        initial_capital: float = 100000.0,
        allocation_percentage: float = 0.01,
        max_share_limit: int = 1000,
        use_max_share_percentage: bool = False,
        max_share_percentage: float = 0.10,
        backtest_start_date: Optional[str] = None,
        backtest_end_date: Optional[str] = None,
        long_min_price: Optional[float] = None,
        long_max_price: Optional[float] = None,
        long_min_avg_volume: float = 1000.0,
        long_min_vol: float = 0.02,
        long_min_vol_regime: Optional[float] = None,
        long_max_vol_regime: Optional[float] = None,
        long_min_beta: Optional[float] = None,
        long_max_beta: Optional[float] = None,
        long_min_z_score: Optional[float] = None,
        long_max_z_score: Optional[float] = None,
        short_min_price: Optional[float] = None,
        short_max_price: Optional[float] = None,
        short_min_avg_volume: float = 1000.0,
        short_min_vol: float = 0.02,
        short_min_vol_regime: Optional[float] = None,
        short_max_vol_regime: Optional[float] = None,
        short_min_beta: Optional[float] = None,
        short_max_beta: Optional[float] = None,
        short_min_z_score: Optional[float] = None,
        short_max_z_score: Optional[float] = None,
        trailing_trigger: float = 0.05,
        trailing_stop: float = 0.02,
        trailing_move: float = 0.01,
        stop_loss_pct: float = 0.05,
        use_volatility_adjusted_stop_loss: bool = False,
        rolling_close_window: int = 5,
        commission_per_trade: float = 1.0,
        min_commission_percent: float = 0.0001
    ):
        # Capital management initialization
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.open_positions = []
        self.commission_per_trade = commission_per_trade
        self.min_commission_percent = min_commission_percent

        # Data validation
        self._validate_dates(historical_data, backtest_start_date, backtest_end_date)
        self.historical_data = self._preprocess_historical_data(historical_data)
        self.trade_signals = self._preprocess_trade_signals(trade_signals)

        # Parameter initialization
        self.profit_target = profit_target
        self.max_holding_days = max_holding_days
        self.allocation_percentage = allocation_percentage
        self.max_share_limit = max_share_limit
        self.use_max_share_percentage = use_max_share_percentage
        self.max_share_percentage = max_share_percentage
        self.backtest_start_date = pd.to_datetime(backtest_start_date) if backtest_start_date else None
        self.backtest_end_date = pd.to_datetime(backtest_end_date) if backtest_end_date else None

        # Risk parameters
        self.long_min_price = long_min_price
        self.long_max_price = long_max_price
        self.long_min_avg_volume = long_min_avg_volume
        self.long_min_vol = long_min_vol
        self.long_min_vol_regime = long_min_vol_regime
        self.long_max_vol_regime = long_max_vol_regime
        self.long_min_beta = long_min_beta
        self.long_max_beta = long_max_beta
        self.long_min_z_score = long_min_z_score
        self.long_max_z_score = long_max_z_score

        self.short_min_price = short_min_price
        self.short_max_price = short_max_price
        self.short_min_avg_volume = short_min_avg_volume
        self.short_min_vol = short_min_vol
        self.short_min_vol_regime = short_min_vol_regime
        self.short_max_vol_regime = short_max_vol_regime
        self.short_min_beta = short_min_beta
        self.short_max_beta = short_max_beta
        self.short_min_z_score = short_min_z_score
        self.short_max_z_score = short_max_z_score

        self.trailing_trigger = trailing_trigger
        self.trailing_stop = trailing_stop
        self.trailing_move = trailing_move
        self.stop_loss_pct = stop_loss_pct
        self.use_volatility_adjusted_stop_loss = use_volatility_adjusted_stop_loss
        self.rolling_close_window = rolling_close_window

        # Initialize trackers
        self.trades = []
        self.missing_trade_dates = []
        self.daily_values = []
        self.daily_dates = []
        self.max_intraday_drawdown = 0.0

        # Configure data
        self.historical_data['ticker'] = self.historical_data['ticker'].astype('category')
        self.trade_signals.loc[:, 'ticker'] = self.trade_signals['ticker'].astype('category')
        
        if 'timestamp' not in self.historical_data.columns:
            logger.error("Missing 'timestamp' in historical data")
            raise ValueError("Historical data requires 'timestamp' column")
            
        self.historical_data.set_index('timestamp', inplace=True)
        self._log_initial_parameters()

    def _validate_dates(self, hist_data: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]):
        """Validate backtest dates against historical data range"""
        hist_start = hist_data['timestamp'].min()
        hist_end = hist_data['timestamp'].max()
        
        if start_date and pd.to_datetime(start_date) < hist_start:
            raise ValueError(f"Start date {start_date} precedes historical data start {hist_start}")
            
        if end_date and pd.to_datetime(end_date) > hist_end:
            raise ValueError(f"End date {end_date} exceeds historical data end {hist_end}")

    def _preprocess_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare historical data with volatility scaling"""
        data['volatility_scaling'] = 0.2 / data['realized_vol_30']
        return data

    def _preprocess_trade_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate trade signals"""
        if 'weighted_probability' in signals.columns:
            if (signals['weighted_probability'].min() < 0) or (signals['weighted_probability'].max() > 1):
                raise ValueError("Weighted probabilities must be between 0-1")
                
        duplicates = signals.duplicated(subset=['trade_date', 'ticker'], keep=False)
        if duplicates.any():
            logger.warning(f"Removing {duplicates.sum()} duplicate signals")
            return signals.drop_duplicates(subset=['trade_date', 'ticker'], keep='last')
        return signals

    def _log_initial_parameters(self):
        """Log all initialization parameters"""
        params = [
            f"Profit Target: {self.profit_target*100}%",
            f"Max Holding Days: {self.max_holding_days}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Allocation Percentage: {self.allocation_percentage*100}%",
            f"Max Share Limit: {self.max_share_limit}",
            f"Use Max Share %: {self.use_max_share_percentage}",
            f"Max Share %: {self.max_share_percentage*100}%" if self.use_max_share_percentage else "",
            f"Backtest Range: {self.backtest_start_date} to {self.backtest_end_date}",
            "Long Filters:",
            f"  Price: {self.long_min_price} - {self.long_max_price}",
            f"  Volume: {self.long_min_avg_volume}",
            f"  Vol: {self.long_min_vol}",
            f"  Vol Regime: {self.long_min_vol_regime} - {self.long_max_vol_regime}",
            f"  Beta: {self.long_min_beta} - {self.long_max_beta}",
            f"  Z-Score: {self.long_min_z_score} - {self.long_max_z_score}",
            "Short Filters:",
            f"  Price: {self.short_min_price} - {self.short_max_price}",
            f"  Volume: {self.short_min_avg_volume}",
            f"  Vol: {self.short_min_vol}",
            f"  Vol Regime: {self.short_min_vol_regime} - {self.short_max_vol_regime}",
            f"  Beta: {self.short_min_beta} - {self.short_max_beta}",
            f"  Z-Score: {self.short_min_z_score} - {self.short_max_z_score}"
        ]
        logger.info("\n".join([p for p in params if p]))

    def compute_rolling_metrics(self):
        """Calculate rolling high/low closes"""
        self.historical_data['rolling_max_close'] = self.historical_data.groupby('ticker', observed=True)['close']\
            .transform(lambda x: x.shift(1).rolling(self.rolling_close_window, min_periods=1).max())
        self.historical_data['rolling_min_close'] = self.historical_data.groupby('ticker', observed=True)['close']\
            .transform(lambda x: x.shift(1).rolling(self.rolling_close_window, min_periods=1).min())
        logger.info(f"Computed {self.rolling_close_window}-day rolling metrics")

    def calculate_slippage(self, avg_volume: float, trade_size: int) -> float:
        """Dynamic slippage based on volume impact"""
        volume_ratio = trade_size / avg_volume
        if volume_ratio < 0.01: return 0.0005
        elif volume_ratio < 0.05: return 0.0015
        else: return 0.0030

    def calculate_position_size(self, entry_price: float, volatility: float, avg_volume: float) -> int:
        """Volatility-adjusted position sizing"""
        risk_capital = min(
            self.available_capital * self.allocation_percentage,
            self.available_capital
        )
        if risk_capital < entry_price:
            return 0
            
        vol_scaled = risk_capital * (0.2 / volatility)
        max_shares = min(
            int(vol_scaled / entry_price),
            int(avg_volume * self.max_share_percentage),
            self.max_share_limit,
            int(self.available_capital / entry_price)
        )
        return max_shares

    def execute_backtest(self, progress_callback=None):
        """Sequential backtest execution with capital tracking"""
        self.compute_rolling_metrics()
        signals = self.trade_signals.sort_values(['trade_date', 'weighted_probability'], 
                                               ascending=[True, False])
        
        start_date = self.backtest_start_date or signals['trade_date'].min()
        end_date = self.backtest_end_date or signals['trade_date'].max()
        
        current_date = start_date
        while current_date <= end_date:
            self._process_daily_signals(signals, current_date)
            self._update_open_positions(current_date)
            self._update_progress(progress_callback, current_date, start_date, end_date)
            current_date += pd.DateOffset(days=1)
            
            if self.available_capital < (self.initial_capital * 0.01):
                logger.warning("Capital exhausted - stopping backtest")
                break
                
        self._finalize_backtest()

    def _process_daily_signals(self, signals: pd.DataFrame, date: pd.Timestamp):
        """Process all signals for a given date"""
        daily_signals = signals[signals['trade_date'] == date]
        for _, signal in daily_signals.iterrows():
            ticker = signal['ticker']
            ticker_data = self.historical_data[self.historical_data['ticker'] == ticker]
            
            if ticker_data.empty:
                self.missing_trade_dates.append((ticker, date))
                continue
                
            trade = self._process_signal(signal, ticker_data)
            if trade:
                self.trades.append(trade)
                self.available_capital -= trade['capital_allocated'] + trade['commission']
                self.open_positions.append({
                    'ticker': ticker,
                    'exit_date': pd.to_datetime(trade['exit_date']),
                    'capital': trade['capital_allocated'],
                    'shares': trade['number_of_shares'],
                    'entry_price': trade['entry_price'],
                    'prediction': trade['prediction']
                })

    def _process_signal(self, signal: pd.Series, ticker_data: pd.DataFrame) -> Optional[dict]:
        """Process individual trade signal with full capital management and risk controls"""
        signal_date = signal['trade_date']
        prediction = signal['prediction']
        weighted_prob = signal['weighted_probability']
        ticker = signal['ticker']
        is_long = prediction == 1

        # Date validation
        if signal_date not in ticker_data.index:
            nearest_date = self.find_nearest_date(signal_date, ticker_data.index, 1)
            if not nearest_date:
                self.missing_trade_dates.append((ticker, signal_date))
                logger.warning(f"Missing data for {ticker} on {signal_date}")
                return None
            signal_date = nearest_date

        # Get market data
        try:
            market_data = ticker_data.loc[signal_date]
            avg_volume = market_data['avg_volume_30']
            realized_vol = market_data['realized_vol_30']
            price_data = {
                'open': market_data['open'],
                'high': market_data['high'],
                'low': market_data['low'],
                'close': market_data['close']
            }
        except KeyError as e:
            logger.error(f"Missing market data for {ticker} on {signal_date}: {e}")
            return None

        # Apply filters
        filters = self._get_filters(is_long)
        if not self._passes_filters(market_data, filters):
            return None

        # Determine entry price
        entry_price = self._calculate_entry_price(is_long, market_data, price_data)
        if entry_price is None:
            return None

        # Calculate position size
        shares = self.calculate_position_size(
            entry_price=entry_price,
            volatility=realized_vol,
            avg_volume=avg_volume
        )
        if shares < 1:
            logger.debug(f"Insufficient position size for {ticker} ({shares} shares)")
            return None

        # Calculate allocation and commission
        allocation = shares * entry_price
        commission = max(
            self.commission_per_trade,
            allocation * self.min_commission_percent
        )
        total_cost = allocation + commission
        
        # Check capital availability
        if total_cost > self.available_capital:
            logger.debug(f"Insufficient capital for {ticker}: Need {total_cost:.2f}, Have {self.available_capital:.2f}")
            return None

        # Apply dynamic slippage
        slippage = self.calculate_slippage(avg_volume, shares)
        if is_long:
            entry_price *= (1 + slippage)
            entry_price = min(entry_price, market_data['high'])  # Clamp to valid price range
        else:
            entry_price *= (1 - slippage)
            entry_price = max(entry_price, market_data['low'])  # Clamp to valid price range

        # Calculate exit parameters
        exit_date = min(
            signal_date + pd.DateOffset(days=self.max_holding_days),
            self.backtest_end_date or ticker_data.index[-1]
        )
        holding_data = ticker_data.loc[signal_date:exit_date]
        
        if holding_data.empty:
            logger.debug(f"No holding data for {ticker} from {signal_date} to {exit_date}")
            return None

        # Calculate exit price with numba
        exit_price, exit_idx, exit_code = calculate_exit_price_numba(
            entry_price=entry_price,
            low_prices=holding_data['low'].values,
            high_prices=holding_data['high'].values,
            stop_loss_pct=self.stop_loss_pct,
            trailing_trigger=self.trailing_trigger,
            trailing_stop=self.trailing_stop,
            trailing_move=self.trailing_move,
            prediction=prediction,
            profit_target=self.profit_target
        )

        # Apply exit slippage and price clamping
        exit_slippage = self.calculate_slippage(avg_volume, shares)
        if is_long:
            exit_price *= (1 - exit_slippage)
            exit_price = max(exit_price, holding_data['low'].min())  # Ensure realistic exit price
        else:
            exit_price *= (1 + exit_slippage)
            exit_price = min(exit_price, holding_data['high'].max())  # Ensure realistic exit price

        # Calculate P&L with slippage validation
        if is_long:
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price
            
        gross_profit = allocation * profit_pct
        net_profit = gross_profit - commission
        net_profit_pct = net_profit / allocation if allocation != 0 else 0

        # Calculate intraday drawdown
        drawdown = self._calculate_intraday_drawdown(
            entry_price=entry_price,
            is_long=is_long,
            lows=holding_data['low'].values,
            highs=holding_data['high'].values
        )

        # Build comprehensive trade record
        return {
            'ticker': ticker,
            'sector': market_data.get('sector', 'Unknown'),
            'signal_date': signal_date.strftime('%Y-%m-%d'),
            'entry_date': signal_date.strftime('%Y-%m-%d'),
            'exit_date': holding_data.index[exit_idx].strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'number_of_shares': shares,
            'capital_allocated': round(allocation, 2),
            'commission': round(commission, 2),
            'gross_profit': round(gross_profit, 2),
            'profit_dollar': round(net_profit, 2),  # Renamed for metrics compatibility
            'profit_pct': round(net_profit_pct * 100, 2),
            'holding_time_days': (pd.to_datetime(holding_data.index[exit_idx]) - signal_date).days,
            'max_drawdown_pct': round(drawdown, 2),
            'exit_reason': self.EXIT_REASON_MAP.get(exit_code, 'Unknown'),
            'prediction': prediction,
            'weighted_probability': weighted_prob,
            'vol_regime_std': market_data.get('vol_regime_std', np.nan),
            'hurst_exponent': market_data.get('hurst_exponent', np.nan),
            'volume_regime': market_data.get('volume_regime', np.nan),
            'realized_vol_30': round(realized_vol, 4),
            'avg_30_day_volume': round(avg_volume, 2),
            'volume_utilization': round(shares / avg_volume, 4),
            'beta': round(market_data.get('rolling_beta_30', 0), 4),
            'rsi_30': round(market_data.get('rsi_30', np.nan), 2),
            'z_score_30': round(market_data.get('z_score_30', 0), 2)
        }
        
    def _get_filters(self, is_long: bool) -> dict:
        """Get appropriate filter set for long/short trades"""
        return {
            'min_price': self.long_min_price if is_long else self.short_min_price,
            'max_price': self.long_max_price if is_long else self.short_max_price,
            'min_avg_volume': self.long_min_avg_volume if is_long else self.short_min_avg_volume,
            'min_vol': self.long_min_vol if is_long else self.short_min_vol,
            'min_vol_regime': self.long_min_vol_regime if is_long else self.short_min_vol_regime,
            'max_vol_regime': self.long_max_vol_regime if is_long else self.short_max_vol_regime,
            'min_beta': self.long_min_beta if is_long else self.short_min_beta,
            'max_beta': self.long_max_beta if is_long else self.short_max_beta,
            'min_z_score': self.long_min_z_score if is_long else self.short_min_z_score,
            'max_z_score': self.long_max_z_score if is_long else self.short_max_z_score,
        }

    def _passes_filters(self, market_data: pd.Series, filters: dict) -> bool:
        """Validate trade against all filters"""
        checks = [
            (market_data['avg_volume_30'] >= filters['min_avg_volume']),
            (market_data['realized_vol_30'] >= filters['min_vol']),
            (filters['min_price'] is None or market_data['close'] >= filters['min_price']),
            (filters['max_price'] is None or market_data['close'] <= filters['max_price']),
            (filters['min_vol_regime'] is None or market_data['vol_regime_std'] >= filters['min_vol_regime']),
            (filters['max_vol_regime'] is None or market_data['vol_regime_std'] <= filters['max_vol_regime']),
            (filters['min_beta'] is None or market_data['rolling_beta_30'] >= filters['min_beta']),
            (filters['max_beta'] is None or market_data['rolling_beta_30'] <= filters['max_beta']),
            (filters['min_z_score'] is None or market_data['z_score_30'] >= filters['min_z_score']),
            (filters['max_z_score'] is None or market_data['z_score_30'] <= filters['max_z_score'])
        ]
        return all(checks)

    def _calculate_entry_price(self, is_long: bool, market_data: pd.Series, price_data: dict) -> Optional[float]:
        """Determine valid entry price based on breakout rules"""
        if is_long:
            threshold = market_data['rolling_max_close']
            if price_data['high'] > threshold:
                return threshold
            if price_data['open'] > threshold:
                return price_data['open']
        else:
            threshold = market_data['rolling_min_close']
            if price_data['low'] < threshold:
                return threshold
            if price_data['open'] < threshold:
                return price_data['open']
        return None

    def _calculate_intraday_drawdown(self, entry_price: float, is_long: bool, lows: np.ndarray, highs: np.ndarray) -> float:
        """Calculate maximum intraday drawdown during holding period"""
        cumulative_max = entry_price
        max_drawdown = 0.0
        
        if is_long:
            for high, low in zip(highs, lows):
                cumulative_max = max(cumulative_max, high)
                drawdown = (cumulative_max - low) / cumulative_max
                max_drawdown = max(max_drawdown, drawdown)
        else:
            cumulative_min = entry_price
            for high, low in zip(highs, lows):
                cumulative_min = min(cumulative_min, low)
                drawdown = (high - cumulative_min) / cumulative_min
                max_drawdown = max(max_drawdown, drawdown)
                
        return max_drawdown * 100

    def _update_open_positions(self, current_date: pd.Timestamp):
        
        # Close positions that have reached their exit date
        for pos in list(self.open_positions):
            if current_date >= pos['exit_date']:
                exit_price = self._get_exit_price(pos)
                self._close_position(pos, exit_price, current_date)

        # Calculate current portfolio value (cash + open positions)
        current_market_value = self.available_capital
        for pos in self.open_positions:
            try:
                # Get ticker data with error handling
                ticker_data = self.historical_data[
                    (self.historical_data['ticker'] == pos['ticker']) & 
                    (self.historical_data.index == current_date)
                ]
                if not ticker_data.empty:
                    price = ticker_data.loc[current_date, 'close']
                    current_market_value += pos['shares'] * price
            except KeyError as e:
                logger.warning(f"Missing price data for {pos['ticker']} on {current_date}: {e}")
                continue
                
        # Record daily portfolio value with NaN handling
        if not np.isnan(current_market_value):
            self.daily_values.append(current_market_value)
            self.daily_dates.append(current_date)
        else:
            logger.error(f"Invalid portfolio value NaN detected on {current_date}")
            # Carry forward last valid value if available
            if self.daily_values:
                self.daily_values.append(self.daily_values[-1])
                self.daily_dates.append(current_date)

    def _get_exit_price(self, position: dict) -> float:
        """Get actual exit price from historical data"""
        try:
            data = self.historical_data[
                (self.historical_data['ticker'] == position['ticker']) &
                (self.historical_data.index == position['exit_date'])
            ].iloc[0]
            return data['close']
        except IndexError:
            logger.warning(f"Missing exit price for {position['ticker']}, using last known")
            return self.historical_data[
                self.historical_data['ticker'] == position['ticker']
            ]['close'].iloc[-1]

    def _close_position(self, position: dict, exit_price: float, exit_date: pd.Timestamp):
        """Calculate final P&L for closed position"""
        entry_val = position['shares'] * position['entry_price']
        exit_val = position['shares'] * exit_price
        profit = (exit_val - entry_val) * position['prediction']
        
        # Update capital
        self.available_capital += exit_val + profit
        
        # Update trade record
        for trade in self.trades:
            if (trade['ticker'] == position['ticker'] and 
                pd.to_datetime(trade['exit_date']) == position['exit_date']):
                trade['actual_exit_price'] = exit_price
                trade['actual_profit'] = profit
                trade['settlement_date'] = exit_date.strftime('%Y-%m-%d')
                break
                
        self.open_positions.remove(position)

    def _finalize_backtest(self):
        """Final cleanup and reporting"""
        self.calculate_max_intraday_drawdown()
        self._calculate_hedge_trades()
        logger.info(f"Backtest completed. Final capital: ${self.available_capital:,.2f}")
        logger.info(f"Total trades executed: {len(self.trades)}")
        logger.info(f"Max intraday drawdown: {self.max_intraday_drawdown:.2f}%")

    def calculate_max_intraday_drawdown(self):
        """Calculate maximum portfolio drawdown"""
        equity_curve = pd.Series({
            pd.to_datetime(t['entry_date']): t['capital_allocated'] 
            for t in self.trades
        }).sort_index().cumsum()
        
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        self.max_intraday_drawdown = drawdown.max() * 100

    def _calculate_hedge_trades(self):
        """Generate beta-hedging trades"""
        # Implementation depends on specific hedging strategy

    def get_trade_history(self) -> pd.DataFrame:
        """Return formatted trade history"""
        df = pd.DataFrame(self.trades)
        if not df.empty:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            df['holding_period'] = (df['exit_date'] - df['entry_date']).dt.days
        return df

    def find_nearest_date(self, target_date: pd.Timestamp, 
                        available_dates: pd.DatetimeIndex, 
                        tolerance: int=1) -> Optional[pd.Timestamp]:
        """Find nearest valid date within tolerance window"""
        dates = available_dates.sort_values()
        idx = bisect_left(dates, target_date)
        
        candidates = []
        if idx > 0: candidates.append(dates[idx-1])
        if idx < len(dates): candidates.append(dates[idx])
        
        valid = [d for d in candidates if abs((d - target_date).days) <= tolerance]
        return min(valid, key=lambda d: abs(d - target_date)) if valid else None

    def _update_progress(self, progress_callback, current_date, start_date, end_date):
        """Update progress callback"""
        if progress_callback:
            days_total = (end_date - start_date).days
            days_done = (current_date - start_date).days
            fraction = min(days_done / days_total, 1.0) if days_total > 0 else 1.0
            progress_callback(fraction)
