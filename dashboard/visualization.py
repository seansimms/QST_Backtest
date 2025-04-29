# scripts/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
import logging
import numpy as np
from typing import Union, List, Tuple, Dict

# Configure logging
logging.basicConfig(
    filename='visualization.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure Seaborn and Matplotlib for a professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})


def format_percentage(x, pos):
    """Helper function to format y-axis labels as percentages."""
    return f"{x:.0f}%"


# ------------------------- Trade History Plots ------------------------- #

def plot_trade_history(trade_history: pd.DataFrame) -> plt.Figure:
    """
    Generate a Matplotlib figure showing the distribution of trade profits and holding times.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with necessary metrics.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # Profit Percentage Distribution
        sns.histplot(trade_history['profit_pct'], bins=30, kde=True, color='#2E86C1', ax=axes[0])
        axes[0].set_title('Profit Percentage Distribution')
        axes[0].set_xlabel('Profit (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].xaxis.set_major_formatter(FuncFormatter(format_percentage))

        # Holding Time Distribution
        sns.histplot(trade_history['holding_time_days'], bins=30, kde=True, color='#28B463', ax=axes[1])
        axes[1].set_title('Holding Time Distribution')
        axes[1].set_xlabel('Holding Time (Days)')
        axes[1].set_ylabel('Frequency')

        # Adjust layout and remove unnecessary spines for a cleaner look
        for ax in axes:
            sns.despine(ax=ax)

        plt.tight_layout()
        logger.info("Trade history plots generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating trade history plots: {e}", exc_info=True)
        return None


def plot_trade_history_interactive(trade_history: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Generate Plotly interactive histograms for profit distribution and holding times.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with necessary metrics.

    Returns:
        Tuple[go.Figure, go.Figure]: Tuple containing two Plotly figures for profit and holding time distributions.
    """
    try:
        # Profit Percentage Distribution
        fig_profit = px.histogram(
            trade_history,
            x='profit_pct',
            nbins=30,
            title="Profit % Distribution (Interactive)",
            labels={'profit_pct': 'Profit (%)'},
            opacity=0.75,
            color_discrete_sequence=['#2E86C1']
        )
        fig_profit.update_layout(
            xaxis_title="Profit (%)",
            yaxis_title="Count",
            bargap=0.2,
            hovermode="overlay",
            template="plotly_white"
        )
        fig_profit.update_traces(marker_line_width=1, marker_line_color='white')

        # Holding Time Distribution
        fig_holding = px.histogram(
            trade_history,
            x='holding_time_days',
            nbins=30,
            title="Holding Time Distribution (Interactive)",
            labels={'holding_time_days': 'Holding Time (Days)'},
            opacity=0.75,
            color_discrete_sequence=['#28B463']
        )
        fig_holding.update_layout(
            xaxis_title="Holding Time (Days)",
            yaxis_title="Count",
            bargap=0.2,
            hovermode="overlay",
            template="plotly_white"
        )
        fig_holding.update_traces(marker_line_width=1, marker_line_color='white')

        logger.info("Interactive trade history plots generated successfully.")
        return fig_profit, fig_holding
    except Exception as e:
        logger.error(f"Error generating interactive trade history plots: {e}", exc_info=True)
        return None, None


# ------------------------- Portfolio Performance Plots ------------------------- #

def plot_portfolio_performance(portfolio_returns: pd.Series) -> plt.Figure:
    """
    Generate a Matplotlib figure of portfolio cumulative returns.

    Parameters:
        portfolio_returns (pd.Series): Series of portfolio daily returns.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        if portfolio_returns.empty:
            logger.warning("Portfolio returns are empty. Portfolio performance plot not generated.")
            return None

        cumret = (1 + portfolio_returns).cumprod() - 1
        fig, ax = plt.subplots(figsize=(24, 12))

        ax.plot(cumret.index, cumret * 100, label='Cumulative Returns', color='#1F618D', linewidth=2)
        ax.set_title('Portfolio Cumulative Returns Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns (%)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Highlight key points
        max_return = cumret.max()
        max_date = cumret.idxmax()
        ax.annotate(f'Max Return: {max_return*100:.2f}%',
                    xy=(max_date, max_return*100),
                    xytext=(max_date, max_return*100 + 5),
                    arrowprops=dict(facecolor='#28B463', shrink=0.05),
                    horizontalalignment='left')

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Portfolio performance plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating portfolio performance plot: {e}", exc_info=True)
        return None


def plot_portfolio_performance_interactive(portfolio_returns: pd.Series) -> go.Figure:
    """
    Generate a Plotly interactive figure of portfolio cumulative returns.

    Parameters:
        portfolio_returns (pd.Series): Series of portfolio daily returns.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        if portfolio_returns.empty:
            logger.warning("Portfolio returns are empty. Interactive performance plot not generated.")
            return None

        cumret = (1 + portfolio_returns).cumprod() - 1
        cumret_pct = cumret * 100

        fig = px.line(
            cumret_pct,
            x=cumret_pct.index,
            y=cumret_pct.values,
            title="Portfolio Cumulative Returns (Interactive)",
            labels={"x": "Date", "y": "Cumulative Returns (%)"},
            template="plotly_white",
            markers=True
        )
        fig.update_traces(line_color='#1F618D', line_width=2)

        # Add final return annotation
        final_return = cumret_pct.iloc[-1]
        fig.add_annotation(
            x=cumret_pct.index[-1],
            y=final_return,
            text=f"Final Return: {final_return:.2f}%",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="rgba(0,0,0,0.7)",
            font=dict(color="white")
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        logger.info("Interactive portfolio performance plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating interactive portfolio performance plot: {e}", exc_info=True)
        return None


# ------------------------- Risk Metrics Plots ------------------------- #

def plot_risk_metrics(metrics: dict) -> plt.Figure:
    """
    Generate a Matplotlib bar chart for risk metrics.

    Parameters:
        metrics (dict): Dictionary containing performance and risk metrics.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        # Extract relevant risk metrics and ensure Max Drawdown is positive
        risk_metrics = {
            'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
            'Sortino Ratio': metrics.get('Sortino Ratio', 0),
            'Value at Risk (95%)': metrics.get('Value at Risk (95%)', 0),
            'Max Drawdown (%)': abs(metrics.get('Max Drawdown (Portfolio) [% - Close]', 0))
        }

        names = list(risk_metrics.keys())
        vals = list(risk_metrics.values())

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=vals, y=names, palette='viridis', ax=ax)
        ax.set_title('Risk Metrics')
        ax.set_xlabel('Value')
        ax.set_ylabel('Metric')

        # Add data labels
        for i, v in enumerate(vals):
            ax.text(
                v + (0.02 * max(vals)), 
                i, 
                f"{v:.2f}" if 'Ratio' in names[i] else f"{v:.2f}%", 
                color='black', ha='left', va='center', fontsize=12
            )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Risk metrics plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating risk metrics plot: {e}", exc_info=True)
        return None


def plot_risk_metrics_interactive(metrics: dict) -> go.Figure:
    """
    Generate a Plotly interactive bar chart for risk metrics.

    Parameters:
        metrics (dict): Dictionary containing performance and risk metrics.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        # Extract relevant risk metrics and ensure Max Drawdown is positive
        risk_data = {
            'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
            'Sortino Ratio': metrics.get('Sortino Ratio', 0),
            'Value at Risk (95%)': metrics.get('Value at Risk (95%)', 0),
            'Max Drawdown (%)': abs(metrics.get('Max Drawdown (Portfolio) [% - Close]', 0))
        }

        names = list(risk_data.keys())
        vals = list(risk_data.values())

        fig = go.Figure(data=[
            go.Bar(
                x=vals,
                y=names,
                orientation='h',
                marker=dict(color=px.colors.sequential.Viridis),
                text=[f"{v:.2f}" if 'Ratio' in name else f"{v:.2f}%" for name, v in risk_data.items()],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Risk Metrics (Interactive)",
            xaxis_title="Value",
            yaxis_title="Metric",
            yaxis=dict(autorange="reversed"),  # To display the first metric on top
            template="plotly_white",
            showlegend=False
        )
        logger.info("Interactive risk metrics plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating interactive risk metrics plot: {e}", exc_info=True)
        return None


# ------------------------- Closed-Trade Cumulative Returns Plots ------------------------- #

def plot_closed_trade_cumulative_returns(trade_history: pd.DataFrame, initial_capital: float) -> plt.Figure:
    """
    Generate a Matplotlib figure for closed-trade cumulative returns using arithmetic returns.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with necessary metrics.
        initial_capital (float): The starting capital for the portfolio.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        if trade_history.empty:
            logger.warning("Trade history is empty. Closed-trade cumulative returns plot not generated.")
            return None

        # Prepare and clean data
        closed_df = trade_history[['exit_date', 'profit_dollar']].copy()
        closed_df['exit_date'] = pd.to_datetime(closed_df['exit_date'], errors='coerce')
        closed_df.dropna(subset=['exit_date'], inplace=True)
        closed_df.sort_values('exit_date', inplace=True)

        # Calculate cumulative returns based on arithmetic growth
        closed_df['cumulative_pnl'] = closed_df['profit_dollar'].cumsum()
        closed_df['cumulative_value'] = initial_capital + closed_df['cumulative_pnl']

        # Convert to cumulative return percentages
        closed_df['cumulative_return'] = ((closed_df['cumulative_value'] - initial_capital) / initial_capital) * 100

        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(
            closed_df['exit_date'], 
            closed_df['cumulative_return'], 
            marker='o', 
            color='#3498DB', 
            linewidth=2, 
            label='Cumulative Returns (Arithmetic)'
        )
        ax.set_title("Closed-Trade Cumulative Returns (Arithmetic)", fontsize=16)
        ax.set_xlabel("Exit Date", fontsize=14)
        ax.set_ylabel("Cumulative Return (%)", fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Annotate final cumulative return
        final_return = closed_df['cumulative_return'].iloc[-1]
        ax.annotate(
            f'Final Return: {final_return:.2f}%',
            xy=(closed_df['exit_date'].iloc[-1], final_return),
            xytext=(closed_df['exit_date'].iloc[-1], final_return + 5),
            arrowprops=dict(facecolor='#28B463', shrink=0.05),
            horizontalalignment='right'
        )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Closed-trade cumulative returns plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"Error generating closed-trade cumulative returns plot: {e}", exc_info=True)
        return None


def plot_closed_trade_cumulative_returns_interactive(self, trade_history: pd.DataFrame) -> go.Figure:
        """
        Generate a Plotly interactive figure for closed-trade cumulative returns using arithmetic returns.

        Parameters:
            trade_history (pd.DataFrame): DataFrame containing executed trades with necessary metrics.

        Returns:
            plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
        """
        try:
            if trade_history.empty:
                logger.warning("Trade history is empty. Interactive closed-trade cumulative returns plot not generated.")
                return None

            closed_df = trade_history[['exit_date', 'profit_dollar']].copy()
            closed_df['exit_date'] = pd.to_datetime(closed_df['exit_date'], errors='coerce')
            closed_df.dropna(subset=['exit_date'], inplace=True)
            closed_df.sort_values('exit_date', inplace=True)

            # Calculate cumulative PnL and returns
            closed_df['cumulative_pnl'] = closed_df['profit_dollar'].cumsum()
            closed_df['cumulative_value'] = self.initial_capital + closed_df['cumulative_pnl']
            closed_df['cumulative_return'] = ((closed_df['cumulative_value'] - self.initial_capital) / self.initial_capital) * 100

            # Create interactive Plotly figure
            fig = px.line(
                closed_df,
                x='exit_date',
                y='cumulative_return',
                title="Closed-Trade Cumulative Returns (Arithmetic, Interactive)",
                labels={"exit_date": "Exit Date", "cumulative_return": "Cumulative Return (%)"},
                template="plotly_white",
                markers=True
            )
            fig.update_traces(line_color='#3498DB', line_width=2)

            # Add final return annotation
            final_return = closed_df['cumulative_return'].iloc[-1]
            fig.add_annotation(
                x=closed_df['exit_date'].iloc[-1],
                y=final_return,
                text=f"Final Return: {final_return:.2f}%",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                bgcolor="rgba(0,0,0,0.7)",
                font=dict(color="white")
            )

            fig.update_layout(
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            )
            logger.info("Interactive closed-trade cumulative returns plot generated successfully.")
            return fig
        except Exception as e:
            logger.error(f"Error generating interactive closed-trade cumulative returns plot: {e}", exc_info=True)
            return None


# ------------------------- Max Intraday Drawdown Plots ------------------------- #

def plot_portfolio_max_intraday_drawdown(max_intraday_dd: float) -> plt.Figure:
    """
    Generate a Matplotlib bar chart for the portfolio's maximum intraday drawdown.

    Parameters:
        max_intraday_dd (float): Maximum intraday drawdown percentage.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=['Max Intraday Drawdown'], y=[max_intraday_dd], palette=['#FF5733'], ax=ax)
        ax.set_title('Max Intraday Drawdown (Portfolio)')
        ax.set_ylabel('Drawdown (%)')
        ax.set_ylim(0, max(max_intraday_dd * 1.1, 10))  # Dynamic y-axis limit for better visibility

        for i, v in enumerate([max_intraday_dd]):
            ax.text(
                i, 
                v + (0.05 * max_intraday_dd), 
                f"{v:.2f}%", 
                color='black', 
                ha='center', 
                va='bottom', 
                fontsize=12
            )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Max Intraday Drawdown plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating Max Intraday Drawdown plot: {e}", exc_info=True)
        return None


def plot_portfolio_max_intraday_drawdown_interactive(max_intraday_dd: float) -> go.Figure:
    """
    Generate a Plotly interactive bar chart for the portfolio's maximum intraday drawdown.

    Parameters:
        max_intraday_dd (float): Maximum intraday drawdown percentage.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        fig = go.Figure(data=[
            go.Bar(
                x=['Max Intraday Drawdown'],
                y=[max_intraday_dd],
                marker_color='#FF5733',
                text=[f"{max_intraday_dd:.2f}%"],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Max Intraday Drawdown (Portfolio)",
            xaxis_title="Metric",
            yaxis_title="Drawdown (%)",
            yaxis=dict(range=[0, max(max_intraday_dd * 1.2, 10)]),  # Dynamic y-axis range
            template="plotly_white",
            showlegend=False
        )
        logger.info("Interactive Max Intraday Drawdown plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating interactive Max Intraday Drawdown plot: {e}", exc_info=True)
        return None


# ------------------------- Volume vs. Return Plots ------------------------- #

def plot_volume_vs_return(trade_history: pd.DataFrame) -> plt.Figure:
    """
    Generate a Matplotlib scatter plot showing the relationship between average 30-day volume and profit percentage.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with
                                      'avg_30_day_volume' and 'profit_pct'.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        if trade_history.empty:
            logger.warning("Trade history is empty. Volume vs. Return scatter plot not generated.")
            return None

        fig, ax = plt.subplots(figsize=(18, 12))

        # Scatter plot
        sns.scatterplot(
            data=trade_history,
            x='avg_30_day_volume',
            y='profit_pct',
            color='#1F77B4',
            alpha=0.7,
            ax=ax
        )
        ax.set_title('Average 30-Day Volume vs. Profit Percentage')
        ax.set_xlabel('Average 30-Day Volume')
        ax.set_ylabel('Profit (%)')

        # Add regression line
        sns.regplot(
            data=trade_history,
            x='avg_30_day_volume',
            y='profit_pct',
            scatter=False,
            ax=ax,
            color='red'
        )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Volume vs. Return scatter plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating Volume vs. Return scatter plot: {e}", exc_info=True)
        return None


def plot_volume_vs_return_interactive(trade_history: pd.DataFrame) -> go.Figure:
    """
    Generate a Plotly interactive scatter plot showing the relationship between average 30-day volume and profit percentage.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with
                                      'avg_30_day_volume' and 'profit_pct'.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        if trade_history.empty:
            logger.warning("Trade history is empty. Interactive Volume vs. Return plot not generated.")
            return None

        fig = px.scatter(
            trade_history,
            x='avg_30_day_volume',
            y='profit_pct',
            title='Average 30-Day Volume vs. Profit Percentage (Interactive)',
            labels={
                'avg_30_day_volume': 'Average 30-Day Volume',
                'profit_pct': 'Profit (%)'
            },
            hover_data=['ticker', 'entry_date', 'exit_date'],
            trendline='ols',
            template='plotly_white',
            color_discrete_sequence=['#1F77B4']
        )

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis=dict(type='log'),  # Logarithmic scale for volume if appropriate
            hovermode="closest"
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

        logger.info("Interactive Volume vs. Return scatter plot generated successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error generating interactive Volume vs. Return scatter plot: {e}", exc_info=True)
        return None


# ------------------------- Average Weighted Probability Plots ------------------------- #

def calculate_average_weighted_probability(trade_history: pd.DataFrame, sector: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Calculate the average weighted probability for each unique day.
    Optionally filter by sector.

    Parameters:
        trade_history (pd.DataFrame): DataFrame containing executed trades with 'exit_date' and 'weighted_probability'.
        sector (Union[str, List[str]], optional): Sector name(s) to filter trades. If None, includes all sectors.

    Returns:
        pd.DataFrame: DataFrame with 'exit_date' and 'avg_weighted_probability'.
    """
    try:
        if trade_history.empty:
            logger.warning("Trade history is empty. Cannot calculate average weighted probabilities.")
            return pd.DataFrame()

        # Apply sector filter if provided
        if sector:
            if isinstance(sector, list):
                filtered_trades = trade_history[trade_history['sector'].isin(sector)].copy()
                logger.info(f"Filtering average weighted probability for sectors: {', '.join(sector)}")
            else:
                filtered_trades = trade_history[trade_history['sector'] == sector].copy()
                logger.info(f"Filtering average weighted probability for sector: {sector}")
        else:
            filtered_trades = trade_history.copy()

        if filtered_trades.empty:
            logger.warning(f"No trades found for the specified sector(s). Returning empty DataFrame.")
            return pd.DataFrame()

        # Ensure 'exit_date' is in datetime format
        filtered_trades['exit_date'] = pd.to_datetime(filtered_trades['exit_date'], errors='coerce')

        # Drop rows with invalid 'exit_date'
        filtered_trades.dropna(subset=['exit_date'], inplace=True)

        # Group by 'exit_date' and calculate the mean of 'weighted_probability'
        avg_wp = filtered_trades.groupby('exit_date')['weighted_probability'].mean().reset_index()
        avg_wp.rename(columns={'weighted_probability': 'avg_weighted_probability'}, inplace=True)

        logger.info("Average weighted probabilities calculated successfully.")
        return avg_wp

    except Exception as e:
        logger.error(f"Error calculating average weighted probabilities: {e}", exc_info=True)
        return pd.DataFrame()


def plot_average_weighted_probability(avg_wp: pd.DataFrame, sector: Union[str, List[str]] = None) -> plt.Figure:
    """
    Generate a Matplotlib figure showing the average weighted probability over time.

    Parameters:
        avg_wp (pd.DataFrame): DataFrame containing 'exit_date' and 'avg_weighted_probability'.
        sector (Union[str, List[str]], optional): Sector name(s) for plot title customization.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        if avg_wp.empty:
            logger.warning("Average Weighted Probability DataFrame is empty. Plot not generated.")
            return None

        fig, ax = plt.subplots(figsize=(24, 12))

        ax.plot(
            avg_wp['exit_date'],
            avg_wp['avg_weighted_probability'],
            marker='o',
            color='#E67E22',
            linewidth=2,
            label='Avg Weighted Probability'
        )

        # Format sector information in the title
        title = 'Average Weighted Probability Over Time'
        if sector:
            if isinstance(sector, list):
                sector_str = ', '.join(sector)
            else:
                sector_str = sector
            title += f" - Sector: {sector_str}"

        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Weighted Probability')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Highlight key points
        max_wp = avg_wp['avg_weighted_probability'].max()
        min_wp = avg_wp['avg_weighted_probability'].min()
        max_date = avg_wp.loc[avg_wp['avg_weighted_probability'] == max_wp, 'exit_date'].iloc[0]
        min_date = avg_wp.loc[avg_wp['avg_weighted_probability'] == min_wp, 'exit_date'].iloc[0]

        ax.annotate(
            f'Max Avg WP: {max_wp:.2f}',
            xy=(max_date, max_wp),
            xytext=(max_date, max_wp + 0.05),
            arrowprops=dict(facecolor='#28B463', shrink=0.05),
            horizontalalignment='left'
        )

        ax.annotate(
            f'Min Avg WP: {min_wp:.2f}',
            xy=(min_date, min_wp),
            xytext=(min_date, min_wp - 0.05),
            arrowprops=dict(facecolor='#E74C3C', shrink=0.05),
            horizontalalignment='right'
        )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Average Weighted Probability plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"Error generating Average Weighted Probability plot: {e}", exc_info=True)
        return None


def plot_average_weighted_probability_interactive(avg_wp: pd.DataFrame, sector: Union[str, List[str]] = None) -> go.Figure:
    """
    Generate a Plotly interactive figure showing the average weighted probability over time.

    Parameters:
        avg_wp (pd.DataFrame): DataFrame containing 'exit_date' and 'avg_weighted_probability'.
        sector (Union[str, List[str]], optional): Sector name(s) for plot title customization.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        if avg_wp.empty:
            logger.warning("Average Weighted Probability DataFrame is empty. Interactive plot not generated.")
            return None

        # Format sector information in the title
        title = "Average Weighted Probability Over Time (Interactive)"
        if sector:
            if isinstance(sector, list):
                sector_str = ', '.join(sector)
            else:
                sector_str = sector
            title += f" - Sector: {sector_str}"

        fig = px.line(
            avg_wp,
            x='exit_date',
            y='avg_weighted_probability',
            title=title,
            labels={"exit_date": "Date", "avg_weighted_probability": "Average Weighted Probability"},
            markers=True,
            template="plotly_white"
        )
        fig.update_traces(line_color='#E67E22', line_width=2)

        # Add annotations for max and min average weighted probabilities
        max_wp = avg_wp['avg_weighted_probability'].max()
        min_wp = avg_wp['avg_weighted_probability'].min()
        max_date = avg_wp.loc[avg_wp['avg_weighted_probability'] == max_wp, 'exit_date'].iloc[0]
        min_date = avg_wp.loc[avg_wp['avg_weighted_probability'] == min_wp, 'exit_date'].iloc[0]

        fig.add_annotation(
            x=max_date,
            y=max_wp,
            text=f"Max Avg WP: {max_wp:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="rgba(40, 180, 67, 0.7)",
            font=dict(color="white")
        )

        fig.add_annotation(
            x=min_date,
            y=min_wp,
            text=f"Min Avg WP: {min_wp:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=40,
            bgcolor="rgba(231, 76, 60, 0.7)",
            font=dict(color="white")
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        logger.info("Interactive Average Weighted Probability plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"Error generating interactive Average Weighted Probability plot: {e}", exc_info=True)
        return None


# ------------------------- Decomposition Plots ------------------------- #

def plot_decomposition_metric(
    decomposition_df: pd.DataFrame,
    metric: str,
    decomposition_type: str,
    sector: Union[str, List[str]] = None
) -> Dict[str, plt.Figure]:
    """
    Generate both Matplotlib and Plotly interactive plots for a specific decomposition metric.

    Parameters:
        decomposition_df (pd.DataFrame): DataFrame containing decomposition metrics.
        metric (str): The metric to plot (e.g., 'Total_Profit_Dollar', 'Number_of_Trades').
        decomposition_type (str): Type of decomposition (e.g., 'Hurst', 'Rolling Beta', 'Z Score').
        sector (Union[str, List[str]], optional): Sector name(s) for plot title customization.

    Returns:
        Dict[str, plt.Figure]: Dictionary containing 'matplotlib' and 'plotly' figures.
    """
    try:
        if decomposition_df.empty:
            logger.warning(f"{decomposition_type} Decomposition DataFrame is empty. Plot not generated.")
            return {}

        figures = {}

        # Matplotlib Plot
        fig_matplotlib = create_matplotlib_metric_plot(decomposition_df, metric, decomposition_type, sector)
        if fig_matplotlib:
            figures[f'{decomposition_type}_{metric}_Matplotlib'] = fig_matplotlib

        # Plotly Interactive Plot
        fig_plotly = create_plotly_metric_plot(decomposition_df, metric, decomposition_type, sector)
        if fig_plotly:
            figures[f'{decomposition_type}_{metric}_Plotly'] = fig_plotly

        logger.info(f"{decomposition_type} Decomposition Metric '{metric}' plots generated successfully.")
        return figures

    except Exception as e:
        logger.error(f"Error generating {decomposition_type} Decomposition Metric '{metric}' plots: {e}", exc_info=True)
        return {}


def create_matplotlib_metric_plot(
    decomposition_df: pd.DataFrame,
    metric: str,
    decomposition_type: str,
    sector: Union[str, List[str]] = None
) -> plt.Figure:
    """
    Create a Matplotlib bar plot for a specific decomposition metric.

    Parameters:
        decomposition_df (pd.DataFrame): DataFrame containing decomposition metrics.
        metric (str): The metric to plot.
        decomposition_type (str): Type of decomposition.
        sector (Union[str, List[str]], optional): Sector name(s) for plot title customization.

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        fig, ax = plt.subplots(figsize=(24, 12))

        # Attempt to find a column that contains 'bucket' in its name (e.g. 'z_score_30_bucket')
        bucket_column = [col for col in decomposition_df.columns if 'bucket' in col.lower()]
        if bucket_column:
            x = bucket_column[0]
        else:
            x = decomposition_df.columns[0]  # Fallback to the first column

        y = metric

        # Determine color palette based on metric
        if metric == 'Total_Profit_Dollar':
            palette = 'coolwarm'
            ylabel = 'Total Profit ($)'
            title_suffix = f' - {metric.replace("_", " ")}'
        elif metric == 'Number_of_Trades':
            palette = 'Blues_d'
            ylabel = 'Number of Trades'
            title_suffix = f' - {metric.replace("_", " ")}'
        else:
            palette = 'viridis'
            ylabel = metric.replace('_', ' ')
            title_suffix = f' - {metric.replace("_", " ")}'

        # Bar plot
        sns.barplot(
            data=decomposition_df,
            x=x,
            y=y,
            palette=palette,
            ax=ax
        )

        # Set titles and labels
        title = f'{decomposition_type} Decomposition{title_suffix}'
        if sector:
            if isinstance(sector, list):
                sector_str = ', '.join(sector)
            else:
                sector_str = sector
            title += f" - Sector: {sector_str}"
        ax.set_title(title)
        ax.set_xlabel(f'{decomposition_type} Range')
        ax.set_ylabel(ylabel)

        # Add data labels on top of each bar
        for index, row in decomposition_df.iterrows():
            if 'Profit' in y:
                label = f"${row[y]:.2f}"
            elif 'Number' in y:
                label = f"{int(row[y])}"
            else:
                label = f"{row[y]:.2f}"
            ax.text(
                index, 
                row[y] + (0.05 * decomposition_df[y].max()),
                label, 
                ha='center', 
                va='bottom', 
                fontsize=12
            )

        sns.despine(ax=ax)
        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Error creating Matplotlib {decomposition_type} Decomposition Metric '{metric}' plot: {e}", exc_info=True)
        return None


def create_plotly_metric_plot(
    decomposition_df: pd.DataFrame,
    metric: str,
    decomposition_type: str,
    sector: Union[str, List[str]] = None
) -> go.Figure:
    """
    Create a Plotly interactive bar plot for a specific decomposition metric.

    Parameters:
        decomposition_df (pd.DataFrame): DataFrame containing decomposition metrics.
        metric (str): The metric to plot.
        decomposition_type (str): Type of decomposition.
        sector (Union[str, List[str]], optional): Sector name(s) for plot title customization.

    Returns:
        plotly.graph_objects.Figure: The generated interactive figure or None if an error occurs.
    """
    try:
        # Map the decomposition type to a relevant bucket column (if needed)
        # For example, 'Z Score' might map to 'z_score_30_bucket'.
        # In your PerformanceMetrics code, the actual column name is determined by the
        # `_decompose(...)` method which sets e.g. 'z_score_30_bucket'.

        if decomposition_type == 'Weighted Probability':
            x = 'wp_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'coolwarm'
            title_suffix = ' - Total Profit ($) (Interactive)'
        elif decomposition_type == 'Hurst':
            x = 'hurst_exponent_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'magma'
            title_suffix = ' - Total Profit ($) (Interactive)'
        elif decomposition_type == 'Rolling Beta':
            x = 'rolling_beta_30_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'coolwarm'
            title_suffix = ' - Total Profit ($) (Interactive)'
        elif decomposition_type == 'Volatility Regime Std':
            x = 'vol_regime_std_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'coolwarm'
            title_suffix = ' - Total Profit ($) (Interactive)'
        elif decomposition_type == 'RSI':
            x = 'rsi_30_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'viridis'
            title_suffix = ' - Total Profit ($) (Interactive)'
        elif decomposition_type == 'Z Score':
            # If the new Z-Score decomposition is named 'z_score_30_bucket':
            x = 'z_score_30_bucket'
            y = 'Total_Profit_Dollar'
            color = 'Total_Profit_Dollar'
            color_scale = 'RdBu'
            title_suffix = ' - Total Profit ($) (Interactive)'
        else:
            logger.warning(f"Unknown decomposition type: {decomposition_type}. Interactive plot not generated.")
            return None

        # If the user wants to plot 'Number_of_Trades' or some other metric:
        if metric == 'Number_of_Trades':
            y = 'Number_of_Trades'
            title_suffix = title_suffix.replace('Total Profit ($)', 'Number of Trades')

        # Overwrite defaults if the decomposition_df uses a different bucket column
        bucket_column = [col for col in decomposition_df.columns if 'bucket' in col.lower()]
        if bucket_column and len(bucket_column) == 1:
            x = bucket_column[0]

        # Format sector information in the title
        title = f"{decomposition_type} Decomposition{title_suffix}"
        if sector:
            if isinstance(sector, list):
                sector_str = ', '.join(sector)
            else:
                sector_str = sector
            title += f" - Sector: {sector_str}"

        # Build the Plotly bar
        fig = px.bar(
            decomposition_df,
            x=x,
            y=y,
            title=title,
            labels={x: f'{decomposition_type} Range', y: metric.replace('_', ' ')},
            color=color,
            color_continuous_scale=color_scale,
            text=y,
            template='plotly_white'
        )
        # Decide on text formatting:
        if 'Profit' in y:
            fig.update_traces(texttemplate='$%{text:.2f}', textposition='auto')
        elif 'Number' in y:
            fig.update_traces(texttemplate='%{text:.0f}', textposition='auto')
        else:
            fig.update_traces(texttemplate='%{text}', textposition='auto')

        fig.update_layout(
            xaxis_title=f"{decomposition_type} Range",
            yaxis_title=metric.replace('_', ' '),
            showlegend=False
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating Plotly {decomposition_type} Decomposition plot: {e}", exc_info=True)
        return None


def plot_decomposition_metrics(decomposition_metrics: dict, sector: Union[str, List[str]] = None) -> dict:
    """
    Generate all decomposition plots for each metric, both Matplotlib and Plotly interactive.
    Allows filtering by sector.

    Parameters:
        decomposition_metrics (dict): Dictionary containing all decomposition DataFrames.
        sector (Union[str, List[str]], optional): Sector name(s) to filter and customize plots.

    Returns:
        dict: Dictionary containing all generated figures.
    """
    try:
        figures = {}

        # Add an entry for Volume and Price decompositions
        decomposition_details = {
            'Volatility_Regime_Std_Decomposition': {
                'decomposition_type': 'Volatility Regime Std',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Hurst_Exponent_Decomposition': {
                'decomposition_type': 'Hurst',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Weighted_Probability_Decomposition': {
                'decomposition_type': 'Weighted Probability',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Rolling_Beta_Decomposition': {
                'decomposition_type': 'Rolling Beta',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'RSI_Decomposition': {
                'decomposition_type': 'RSI',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Z_Score_Decomposition': {
                'decomposition_type': 'Z Score',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Volume_Decomposition': {
                'decomposition_type': 'Volume',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            },
            'Price_Decomposition': {
                'decomposition_type': 'Price',
                'metrics': ['Total_Profit_Dollar', 'Number_of_Trades']
            }
        }

        for key, details in decomposition_details.items():
            decomposition_df = decomposition_metrics.get(key)
            if decomposition_df is not None:
                dec_type = details['decomposition_type']
                for metric in details['metrics']:
                    metric_figures = plot_decomposition_metric(
                        decomposition_df,
                        metric,
                        dec_type,
                        sector
                    )
                    figures.update(metric_figures)

        logger.info("All decomposition metric plots (including Volume and Price) generated successfully.")
        return figures

    except Exception as e:
        logger.error(f"Error generating decomposition metric plots: {e}", exc_info=True)
        return {}


# ------------------------- Monthly Statistics Plot ------------------------- #

def plot_monthly_stats(monthly_stats: pd.DataFrame) -> plt.Figure:
    """
    Generate a Matplotlib bar chart for monthly returns (or other monthly statistics).

    Parameters:
        monthly_stats (pd.DataFrame): DataFrame containing monthly statistics, including
                                      'exit_month' and 'Monthly_Return' columns (among others).

    Returns:
        matplotlib.figure.Figure: The generated figure or None if an error occurs.
    """
    try:
        if monthly_stats.empty:
            logger.warning("Monthly stats DataFrame is empty. Monthly stats plot not generated.")
            return None

        # We assume there's a 'Monthly_Return' column to plot. Adjust as needed for your data structure.
        if 'Monthly_Return' not in monthly_stats.columns:
            logger.warning("'Monthly_Return' column not found in monthly stats. Plot not generated.")
            return None

        fig, ax = plt.subplots(figsize=(18, 8))
        sns.barplot(
            data=monthly_stats,
            x='exit_month',
            y='Monthly_Return',
            palette='rocket',
            ax=ax
        )
        ax.set_title("Monthly Returns")
        ax.set_xlabel("Month")
        ax.set_ylabel("Return (%)")

        # Rotate month labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add data labels on top of each bar
        for index, row in monthly_stats.iterrows():
            ax.text(
                index, 
                row['Monthly_Return'] + 0.1, 
                f"{row['Monthly_Return']:.2f}%", 
                ha='center', 
                va='bottom', 
                fontsize=11
            )

        sns.despine(ax=ax)
        plt.tight_layout()
        logger.info("Monthly statistics plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"Error generating monthly stats plot: {e}", exc_info=True)
        return None


# ------------------------- All Decomposition Metrics Plots ------------------------- #

def plot_all_decompositions(decomposition_metrics: dict, sector: Union[str, List[str]] = None) -> dict:
    """
    Generate all decomposition plots, both Matplotlib and Plotly interactive.
    Allows filtering by sector.

    Parameters:
        decomposition_metrics (dict): Dictionary containing all decomposition DataFrames.
        sector (Union[str, List[str]], optional): Sector name(s) to filter and customize plots.

    Returns:
        dict: Dictionary containing all generated figures.
    """
    try:
        figures = {}

        # If you prefer to only plot specific metrics, or if your code needed separate logic,
        # you can customize here. This example uses the new "plot_decomposition_metrics"
        # function, which now includes Z-score as well.
        dec_figures = plot_decomposition_metrics(decomposition_metrics, sector=sector)
        figures.update(dec_figures)

        return figures
    except Exception as e:
        logger.error(f"Error generating all decomposition plots: {e}", exc_info=True)
        return {}
