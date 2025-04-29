# metrics.py

import numpy as np
import pandas as pd
import logging
from bisect import bisect_left

logging.basicConfig(
    filename='metrics.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Enhanced performance metrics calculator with fixes for:
    - Sharpe/Sortino ratio calculations
    - Proper win rate handling for long/short positions
    - Monte Carlo simulation improvements
    - Edge case handling for financial ratios
    """

    def __init__(self, trade_history: pd.DataFrame, initial_capital: float = 100000.0, 
                 benchmark_returns: pd.Series = None):
        self.trade_history = trade_history.copy()
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns
        self._preprocess_trades()
        logger.info("PerformanceMetrics initialized.")

    def _preprocess_trades(self):
        """Ensure proper data types and calculate win/loss flags"""
        # Convert dates
        date_cols = ['entry_date', 'exit_date']
        for col in date_cols:
            self.trade_history[col] = pd.to_datetime(self.trade_history[col], errors='coerce')
        
        # Calculate actual win/loss based on direction
        self.trade_history['is_win'] = np.where(
            (self.trade_history['prediction'] == 1) & 
            (self.trade_history['exit_price'] > self.trade_history['entry_price']) |
            (self.trade_history['prediction'] == -1) & 
            (self.trade_history['exit_price'] < self.trade_history['entry_price']),
            1, 0
        )

    # ------------------------- Core Metric Calculations ------------------------- #

    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate daily returns from trade PnLs"""
        try:
            if self.trade_history.empty:
                return pd.Series(dtype=float)

            # Create date index
            dates = pd.date_range(
                self.trade_history['entry_date'].min(),
                self.trade_history['exit_date'].max(),
                freq='D'
            )
            
            # Accumulate PnLs by exit date
            pnl_series = pd.Series(0.0, index=dates)
            for _, trade in self.trade_history.iterrows():
                exit_date = trade['exit_date']
                if pd.notnull(exit_date) and exit_date in pnl_series.index:
                    pnl_series[exit_date] += trade['profit_dollar']

            # Calculate equity and returns
            equity = self.initial_capital + pnl_series.cumsum()
            returns = equity.pct_change().fillna(0)
            return returns

        except Exception as e:
            logger.error(f"Return calculation failed: {e}")
            return pd.Series(dtype=float)

    def calculate_sharpe_ratio(self, returns: pd.Series, rf: float = 0) -> float:
        """Annualized Sharpe Ratio with proper risk-free rate handling"""
        if returns.empty or returns.std() == 0:
            return np.nan
            
        daily_rf = (1 + rf/100) ** (1/252) - 1  # Convert annual % to daily decimal
        excess_returns = returns - daily_rf
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns: pd.Series, rf: float = 0) -> float:
        """Enhanced Sortino Ratio with validation"""
        if returns.empty or (returns == 0).all():
            return np.nan
            
        daily_rf = (1 + rf/100) ** (1/252) - 1
        excess_returns = returns - daily_rf
        downside = excess_returns[excess_returns < 0]
        
        if len(downside) < 3 or downside.std() == 0:
            return np.nan
            
        return (excess_returns.mean() / downside.std()) * np.sqrt(252)

    def calculate_win_rate(self) -> dict:
        """Accurate win rate calculation for long/short positions"""
        try:
            if self.trade_history.empty:
                return {'Overall': np.nan, 'Long': np.nan, 'Short': np.nan}

            longs = self.trade_history[self.trade_history['prediction'] == 1]
            shorts = self.trade_history[self.trade_history['prediction'] == -1]

            return {
                'Overall': self.trade_history['is_win'].mean() * 100,
                'Long': longs['is_win'].mean() * 100 if not longs.empty else np.nan,
                'Short': shorts['is_win'].mean() * 100 if not shorts.empty else np.nan
            }
        except Exception as e:
            logger.error(f"Win rate error: {e}")
            return {'Overall': np.nan, 'Long': np.nan, 'Short': np.nan}

    def calculate_beta(self, returns: pd.Series) -> float:
        """Robust Beta calculation with validation"""
        if self.benchmark_returns is None or returns.empty:
            return np.nan
            
        aligned_bench = self.benchmark_returns.reindex(returns.index).dropna()
        aligned_returns = returns.reindex(aligned_bench.index)
        
        if len(aligned_returns) < 2 or aligned_bench.var() < 1e-9:
            return np.nan
            
        cov = np.cov(aligned_returns, aligned_bench)[0, 1]
        return cov / aligned_bench.var()

    # ------------------------- Risk Metrics ------------------------- #

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Value at Risk calculation"""
        if returns.empty:
            return np.nan
        return np.percentile(returns, 100*(1-confidence))

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Maximum drawdown calculation"""
        if returns.empty:
            return np.nan
            
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        return drawdown.min() * 100  # Return as percentage

    # ------------------------- Performance Metrics Aggregation ------------------------- #

    def calculate_metrics(self) -> dict:
        """Aggregate all performance metrics"""
        metrics = {}
        returns = self.calculate_portfolio_returns()
        win_rates = self.calculate_win_rate()
        months_in_backtest = (self.trade_history['exit_date'].max() - self.trade_history['entry_date'].min()).days / 30.44  # average month length
        yearly_return = (self.trade_history['profit_dollar'].sum() / self.initial_capital * 100) * (12 / months_in_backtest)
    

        try:
            metrics.update({
                'Total Trades': len(self.trade_history),
                'Profitable Trades': self.trade_history['is_win'].sum(),
                'Win Rate (%)': win_rates['Overall'],
                'Long Win Rate (%)': win_rates['Long'],
                'Short Win Rate (%)': win_rates['Short'],
                'Total Return (%)': self.trade_history['profit_dollar'].sum() / self.initial_capital * 100,
                'Yearly Return (%)': yearly_return,
                'Annualized Return (%)': self._annualized_return(returns),
                'Sharpe Ratio': self.calculate_sharpe_ratio(returns),
                'Sortino Ratio': self.calculate_sortino_ratio(returns),
                'Max Drawdown (%)': self.calculate_max_drawdown(returns),
                'Value at Risk (95%)': self.calculate_var(returns),
                'Beta': self.calculate_beta(returns),
                'Tracking Error': self._tracking_error(returns),
            })
        except Exception as e:
            logger.error(f"Metrics aggregation error: {e}")

        return metrics

    # ------------------------- Support Calculations ------------------------- #

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return from daily returns"""
        if returns.empty:
            return np.nan
        cum_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        return ((1 + cum_return) ** (1/years) - 1) * 100 if years > 0 else 0

    def _tracking_error(self, returns: pd.Series) -> float:
        """Tracking error relative to benchmark"""
        if self.benchmark_returns is None:
            return np.nan
            
        aligned_bench = self.benchmark_returns.reindex(returns.index).fillna(0)
        return (returns - aligned_bench).std() * np.sqrt(252)

      # ------------------------- Decomposition Methods ------------------------- #

    def _decompose(self, column: str, bins: list, labels: list, sector: str = None) -> pd.DataFrame:
        """
        Generic decomposition method to avoid repetition.
        Includes both average and total profit calculations.
        """
        try:
            if self.trade_history.empty:
                logging.info(f"No trades in trade_history, returning empty {column} Decomposition.")
                return pd.DataFrame()

            if sector:
                filtered_trades = self.trade_history[self.trade_history['sector'] == sector]
                logger.info(f"Filtering {column} decomposition for sector: {sector}")
            else:
                filtered_trades = self.trade_history.copy()

            if filtered_trades.empty:
                logging.warning(f"No trades found for sector '{sector}'. Returning empty DataFrame.")
                return pd.DataFrame()

            if column not in filtered_trades.columns:
                logging.warning(f"No '{column}' column found. Cannot compute {column} decomposition.")
                return pd.DataFrame()

            bucket_column = f"{column}_bucket"
            filtered_trades[bucket_column] = pd.cut(
                filtered_trades[column],
                bins=bins,
                right=False,
                labels=labels
            )

            def _win_rate(x):
                return (x > 0).mean() * 100 if len(x) > 0 else np.nan

            def _avg_holding_days(x):
                return x.mean() if len(x) > 0 else np.nan

            def _avg_profit_pct(x):
                return x.mean() if len(x) > 0 else np.nan

            def _avg_profit_dollar(x):
                return x.mean() if len(x) > 0 else np.nan

            def _total_profit_dollar(x):
                return x.sum() if len(x) > 0 else np.nan

            decomposition = (
                filtered_trades
                .groupby(bucket_column)
                .agg(
                    Number_of_Trades=('ticker', 'count'),
                    Win_Rate_Percent=('profit_dollar', _win_rate),
                    Avg_Holding_Days=('holding_time_days', _avg_holding_days),
                    Avg_Profit_Pct=('profit_pct', _avg_profit_pct),
                    Avg_Profit_Dollar=('profit_dollar', _avg_profit_dollar),
                    Total_Profit_Dollar=('profit_dollar', _total_profit_dollar)
                )
                .reset_index()
            )
            logging.info(f"{column} Decomposition calculated successfully.")
            return decomposition

        except Exception as e:
            logging.error(f"Error calculating {column} Decomposition: {e}", exc_info=True)
            return pd.DataFrame()

    def calculate_vol_regime_std_decomposition(self, vol_bins: list = None, sector: str = None) -> pd.DataFrame:
        if vol_bins is None:
            vol_bins = list(range(-5, 6, 1))  # Example bins from -5 to 5
        labels = [f"[{vol_bins[i]}, {vol_bins[i+1]})" for i in range(len(vol_bins) - 1)]
        return self._decompose('vol_regime_std', vol_bins, labels, sector)

    def calculate_hurst_exponent_decomposition(self, hurst_bins: list = None, sector: str = None) -> pd.DataFrame:
        if hurst_bins is None:
            hurst_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = [f"[{hurst_bins[i]:.2f}, {hurst_bins[i+1]:.2f})" for i in range(len(hurst_bins) - 1)]
        return self._decompose('hurst_exponent', hurst_bins, labels, sector)

    def calculate_wp_decomposition(self, wp_bins: list = None, sector: str = None) -> pd.DataFrame:
        if wp_bins is None:
            wp_bins = [.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        labels = [f"[{wp_bins[i]:.2f}, {wp_bins[i+1]:.2f})" for i in range(len(wp_bins) - 1)]
        return self._decompose('weighted_probability', wp_bins, labels, sector)

    def calculate_rolling_beta_decomposition(self, beta_bins: list = None, sector: str = None) -> pd.DataFrame:
        if beta_bins is None:
            beta_bins = [-2, -1, 0, 1, 2, 3, 4, 5]
        labels = [f"[{beta_bins[i]}, {beta_bins[i+1]})" for i in range(len(beta_bins) - 1)]
        return self._decompose('rolling_beta_30', beta_bins, labels, sector)

    def calculate_rsi_decomposition(self, rsi_bins: list = None, sector: str = None) -> pd.DataFrame:
        if rsi_bins is None:
            rsi_bins = [0, 30, 50, 70, 100]
        labels = [f"[{rsi_bins[i]}, {rsi_bins[i+1]})" for i in range(len(rsi_bins) - 1)]
        return self._decompose('rsi_30', rsi_bins, labels, sector)

    def calculate_z_score_decomposition(self, z_bins: list = None, sector: str = None) -> pd.DataFrame:
        """
        Decompose performance by Z-Score 30-day (z_score_30) buckets.
        Bins can range as per user preference (default is [-3, -2, -1, 0, 1, 2, 3]).
        """
        if z_bins is None:
            z_bins = [-3, -2, -1, 0, 1, 2, 3]  # Example bins for Z-Score
        labels = [f"[{z_bins[i]}, {z_bins[i+1]})" for i in range(len(z_bins) - 1)]
        return self._decompose('z_score_30', z_bins, labels, sector)

    def calculate_volume_decomposition(self, volume_bins: list = None, sector: str = None) -> pd.DataFrame:
        """
        Decompose performance by Volume buckets.
        """
        if volume_bins is None:
            volume_bins = [0, 250000, 500000, 750000, 1000000, 1250000, 1500000, 1750000, 2000000]
        labels = [f"[{volume_bins[i]}, {volume_bins[i+1]})" for i in range(len(volume_bins) - 1)]
        return self._decompose('avg_30_day_volume', volume_bins, labels, sector)

    def calculate_price_decomposition(self, price_bins: list = None, sector: str = None) -> pd.DataFrame:
        """
        Decompose performance by Price buckets.
        """
        if price_bins is None:
            price_bins = [0, 1, 5, 10, 50, 100, float('inf')]
        labels = [
            "[0, 1)", "[1, 5)", "[5, 10)", "[10, 50)", "[50, 100)", "100+"
        ]
        return self._decompose('entry_price', price_bins, labels, sector)
    
    def calculate_volume_regime_decomposition(self, volume_regime_bins: list = None, sector: str = None) -> pd.DataFrame:
        """
        Decompose performance by Volume Regime buckets.
        """
        if volume_regime_bins is None:
            volume_regime_bins = [-3, -2, -1, 0, 1, 2, 3]
        labels = [f"[{volume_regime_bins[i]}, {volume_regime_bins[i+1]})" for i in range(len(volume_regime_bins) - 1)]
        return self._decompose('volume_regime', volume_regime_bins, labels, sector)


    # ------------------------- Monthly Statistics Calculation ------------------------- #

    def calculate_monthly_stats(self, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Calculate monthly statistics based on trade history and portfolio returns,
        including Percentage Long and Percentage Short.

        Parameters:
            portfolio_returns (pd.Series): Series of portfolio daily returns.

        Returns:
            pd.DataFrame: DataFrame containing monthly statistics.
        """
        try:
            if self.trade_history.empty:
                logger.info("No trades in trade_history, returning empty monthly stats.")
                return pd.DataFrame()

            # Ensure 'exit_date' is datetime
            self.trade_history['exit_date'] = pd.to_datetime(self.trade_history['exit_date'], errors='coerce')
            self.trade_history.dropna(subset=['exit_date'], inplace=True)
            self.trade_history['exit_month'] = self.trade_history['exit_date'].dt.to_period('M').astype(str)

            # Calculate Capital Long and Short
            self.trade_history['Capital_Long'] = np.where(
                self.trade_history['prediction'] == 1,
                self.trade_history['capital_allocated'], 0
            )
            self.trade_history['Capital_Short'] = np.where(
                self.trade_history['prediction'] == -1,
                self.trade_history['capital_allocated'], 0
            )

            # Group by 'exit_month' and calculate statistics
            monthly_stats = self.trade_history.groupby('exit_month').agg(
                Total_PnL_Dollar=('profit_dollar', 'sum'),
                Number_of_Trades=('profit_dollar', 'count'),
                Win_Rate_Percent=('profit_dollar', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else np.nan),
                Avg_30_Day_Volume=('avg_30_day_volume', 'mean'),
                Min_30_Day_Realized_Vol=('realized_vol_30', 'min'),
                Capital_Long=('Capital_Long', 'sum'),
                Capital_Short=('Capital_Short', 'sum')
            ).reset_index()

            # Calculate total and percentages for capital allocation
            monthly_stats['Total_Capital_Allocated'] = (
                monthly_stats['Capital_Long'] + monthly_stats['Capital_Short']
            )
            monthly_stats['Percentage_Long'] = np.where(
                monthly_stats['Total_Capital_Allocated'] > 0,
                (monthly_stats['Capital_Long'] / monthly_stats['Total_Capital_Allocated']) * 100,
                0
            )
            monthly_stats['Percentage_Short'] = np.where(
                monthly_stats['Total_Capital_Allocated'] > 0,
                (monthly_stats['Capital_Short'] / monthly_stats['Total_Capital_Allocated']) * 100,
                0
            )
            monthly_stats['Percentage_Long'] = monthly_stats['Percentage_Long'].round(2)
            monthly_stats['Percentage_Short'] = monthly_stats['Percentage_Short'].round(2)

            # Calculate monthly portfolio returns
            portfolio_returns_monthly = portfolio_returns.copy()
            portfolio_returns_monthly.index = portfolio_returns_monthly.index.to_period('M').astype(str)
            monthly_returns = portfolio_returns_monthly.groupby(portfolio_returns_monthly.index).apply(lambda x: (1 + x).prod() - 1).reset_index()
            monthly_returns.rename(columns={0: 'Monthly_Return', 'index': 'exit_month'}, inplace=True)
            monthly_returns.columns = ['exit_month', 'Monthly_Return']

            # Merge portfolio returns with trade statistics
            monthly_stats = pd.merge(monthly_stats, monthly_returns, on='exit_month', how='left')
            monthly_stats['Monthly_Return'] = monthly_stats['Monthly_Return'].fillna(0).round(2)

            logger.info("Monthly statistics calculated successfully.")
            return monthly_stats

        except Exception as e:
            logger.error(f"Error calculating monthly stats: {e}", exc_info=True)
            return pd.DataFrame()


    def get_monthly_statistics(self, portfolio_returns: pd.Series) -> pd.DataFrame:
        """
        Retrieve monthly statistics based on trade history and portfolio returns.

        Parameters:
            portfolio_returns (pd.Series): Series of portfolio daily returns.

        Returns:
            pd.DataFrame: DataFrame containing monthly statistics.
        """
        try:
            monthly_stats = self.calculate_monthly_stats(portfolio_returns)
            return monthly_stats
        except Exception as e:
            logging.error(f"Error retrieving monthly statistics: {e}", exc_info=True)
            return pd.DataFrame()

    # ------------------------- Monte Carlo Simulation ------------------------- #

    def run_monte_carlo_simulation(
        self,
        trade_history: pd.DataFrame,
        num_sims: int,
        sim_days: int,
        sim_method: str = "historical",
        seed: int = None
    ) -> pd.DataFrame:
        """
        Enhanced Monte Carlo simulation for backtested trade data using log-returns.
        """
        try:
            if seed is not None:
                np.random.seed(seed)
                logger.info(f"Setting random seed to {seed} for Monte Carlo simulation.")

            if 'profit_pct' not in trade_history.columns:
                logger.warning("No 'profit_pct' column found in trade_history. Cannot run Monte Carlo.")
                return pd.DataFrame()

            daily_returns_decimal = (trade_history['profit_pct'] / 100.0).dropna()
            if daily_returns_decimal.empty:
                logger.warning("No valid daily returns found for Monte Carlo.")
                return pd.DataFrame()

            log_returns = np.log1p(daily_returns_decimal)
            if log_returns.empty:
                logger.warning("Log returns are empty after conversion.")
                return pd.DataFrame()

            sim_returns = None
            if sim_method == "historical":
                unique_log_returns = log_returns.unique()
                if len(unique_log_returns) == 0:
                    logger.warning("No unique log-returns for historical simulation.")
                    return pd.DataFrame()

                logger.info(f"Running 'historical' Monte Carlo with {num_sims} sims, {sim_days} days, "
                            f"draws from {len(unique_log_returns)} unique log-returns.")

                draws = np.random.choice(unique_log_returns, size=(sim_days, num_sims), replace=True)
                sim_returns = pd.DataFrame(
                    data=draws,
                    columns=[f"Sim_{i+1}" for i in range(num_sims)],
                    index=pd.date_range(
                        start=pd.to_datetime(trade_history['entry_date']).min(),
                        periods=sim_days,
                        freq='B'
                    )
                )

            elif sim_method == "gaussian":
                mu = log_returns.mean()
                sigma = log_returns.std()

                if pd.isna(mu) or pd.isna(sigma):
                    logger.warning("Mean or std of log-returns is NaN for Gaussian simulation.")
                    return pd.DataFrame()

                logger.info(f"Running 'gaussian' Monte Carlo with mean={mu:.6f}, std={sigma:.6f}, "
                            f"{num_sims} sims, {sim_days} days.")

                draws = np.random.normal(loc=mu, scale=sigma, size=(sim_days, num_sims))
                sim_returns = pd.DataFrame(
                    data=draws,
                    columns=[f"Sim_{i+1}" for i in range(num_sims)],
                    index=pd.date_range(
                        start=pd.to_datetime(trade_history['entry_date']).min(),
                        periods=sim_days,
                        freq='B'
                    )
                )
            else:
                logger.error(f"Invalid simulation method '{sim_method}'. Choose 'historical' or 'gaussian'.")
                return pd.DataFrame()

            # Convert log returns back to decimal returns
            daily_decimal_returns = np.expm1(sim_returns.values)
            final_sim = pd.DataFrame(
                data=daily_decimal_returns,
                columns=sim_returns.columns,
                index=sim_returns.index
            )

            logger.info(f"Monte Carlo simulation completed with method='{sim_method}', "
                        f"days={sim_days}, sims={num_sims}, seed={'Set' if seed else 'Not Set'}.")
            return final_sim

        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}", exc_info=True)
            return pd.DataFrame()

    # ------------------------- Overall Simulation Runner ------------------------- #

    def run_portfolio_performance(self, sector: str = None) -> dict:
        """
        Run all portfolio performance metrics and decompositions.
        """
        try:
            portfolio_returns = self.calculate_portfolio_returns()
            metrics = self.calculate_metrics()
            decomposition = self.get_decomposition_metrics(sector=sector)
            monthly_stats = self.get_monthly_statistics(portfolio_returns)

            performance_report = {
                'Aggregated_Metrics': metrics,
                'Decomposition_Metrics': decomposition,
                'Monthly_Statistics': monthly_stats
            }
            logging.info("Portfolio performance and decompositions calculated successfully.")
            return performance_report

        except Exception as e:
            logging.error(f"Error running portfolio performance: {e}", exc_info=True)
            return {}

    # ------------------------- Decomposition Metrics Retrieval ------------------------- #

def get_decomposition_metrics(self, sector: str = None) -> dict:
    """
    Retrieve all decomposition metrics including existing and new metrics.
    Allows filtering by sector.
    """
    try:
        vol_decomp = self.calculate_vol_regime_std_decomposition(sector=sector)
        hurst_decomp = self.calculate_hurst_exponent_decomposition(sector=sector)
        wp_decomp = self.calculate_wp_decomposition(sector=sector)
        beta_decomp = self.calculate_rolling_beta_decomposition(sector=sector)
        rsi_decomp = self.calculate_rsi_decomposition(sector=sector)
        z_score_decomp = self.calculate_z_score_decomposition(sector=sector)
        
        # Adding new decompositions for Volume and Price
        volume_decomp = self.calculate_volume_decomposition(sector=sector)
        price_decomp = self.calculate_price_decomposition(sector=sector)
        volume_regime_decomp = self.calculate_volume_regime_decomposition(sector=sector)

        decomposition = {
            'Volatility_Regime_Std_Decomposition': vol_decomp,
            'Hurst_Exponent_Decomposition': hurst_decomp,
            'Weighted_Probability_Decomposition': wp_decomp,
            'Rolling_Beta_Decomposition': beta_decomp,
            'RSI_Decomposition': rsi_decomp,
            'Z_Score_Decomposition': z_score_decomp,
            'Volume_Decomposition': volume_decomp,
            'Price_Decomposition': price_decomp,
            'Volume_Regime_Decomposition': volume_regime_decomp
        }
        return decomposition

    except Exception as e:
        logging.error(f"Error retrieving decomposition metrics: {e}", exc_info=True)
        return {}

# ------------------------- Utility Functions ------------------------- #

def compute_hurst_exponent(ts: pd.Series) -> float:
    """
    Compute the Hurst Exponent for a given time series using Rescaled Range (R/S) analysis.
    """
    try:
        N = len(ts)
        if N < 2:
            return np.nan

        mean_ts = np.mean(ts)
        deviate = ts - mean_ts
        Z = deviate.cumsum()
        R = Z.max() - Z.min()
        S = np.std(ts)
        if S == 0:
            return np.nan
        RS = R / S
        hurst = np.log(RS) / np.log(N)
        return hurst
    except Exception as e:
        logger.error(f"Error computing Hurst Exponent: {e}", exc_info=True)
        return np.nan

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given price series.
    """
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(0)

        return rsi
    except Exception as e:
        logger.error(f"Error computing RSI: {e}", exc_info=True)
        return pd.Series(dtype=float)
