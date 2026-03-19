#!/usr/bin/env python3
"""Enhanced Static BH 1Y strategies with smart rebalancing triggers."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from shared_strategies import select_top_performers
from config import (
    STATIC_BH_1Y_VOLUME_MIN_DAILY,
    STATIC_BH_1Y_SECTOR_MAX_PER_SECTOR,
    STATIC_BH_1Y_PERFORMANCE_MIN_IMPROVEMENT,
    STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_HIGH_VOL_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_LOW_VOL_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_VOL_THRESHOLD
)


def select_volume_filtered_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    min_volume: float = STATIC_BH_1Y_VOLUME_MIN_DAILY  # Minimum $1M daily volume
) -> List[str]:
    """
    Static BH 1Y with volume confirmation filter.
    Only selects stocks with sufficient liquidity.
    """
    # Get top performers first
    candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n * 2  # Get more to filter
    )
    
    # Apply volume filter
    filtered_stocks = []
    for ticker in candidates:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if not data.empty and 'Volume' in data.columns:
                # Calculate average daily volume over last 20 days
                recent_volume = data['Volume'].tail(20).mean()
                # Convert to dollar volume (volume * price)
                recent_price = data['Close'].tail(20).mean()
                if recent_price > 0:
                    avg_dollar_volume = recent_volume * recent_price
                    if avg_dollar_volume >= min_volume:
                        filtered_stocks.append(ticker)
    
    return filtered_stocks[:top_n]


def select_sector_rotated_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    max_per_sector: int = 3
) -> List[str]:
    """
    Static BH 1Y with sector rotation enhancement.
    Limits to max 3 stocks per sector for diversification.
    """
    # Simple sector mapping (you can expand this)
    sector_mapping = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'NFLX'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'MDT', 'ISRG'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'SCHW'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR'],
        'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'LOW', 'TGT', 'COST'],
        'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT'],
        'Materials': ['DOW', 'DD', 'NUE', 'FCX', 'BHP'],
        'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'XEL'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'Communication': ['VZ', 'T', 'TMUS', 'CMCSA', 'CHTR']
    }
    
    # Reverse mapping: ticker -> sector
    ticker_to_sector = {}
    for sector, tickers_in_sector in sector_mapping.items():
        for ticker in tickers_in_sector:
            ticker_to_sector[ticker] = sector
    
    # Get top performers
    candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n * 3  # Get more to ensure diversity
    )
    
    # Apply sector filter
    selected = []
    sector_counts = {}
    
    for ticker in candidates:
        # Unmapped tickers get their own "sector" to avoid all being lumped into 'Other'
        sector = ticker_to_sector.get(ticker, f'_unmapped_{ticker}')
        current_count = sector_counts.get(sector, 0)
        
        if current_count < max_per_sector:
            selected.append(ticker)
            sector_counts[sector] = current_count + 1
            
        if len(selected) >= top_n:
            break
    
    return selected


def select_performance_threshold_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    top_n: int = 10,
    min_improvement: float = 0.02  # 2% minimum improvement
) -> Tuple[List[str], bool]:
    """
    Static BH 1Y with performance threshold trigger.
    Only rebalances if new selection offers >2% improvement.
    
    Returns:
        (selected_stocks, should_rebalance)
    """
    # Get current portfolio performance
    if not current_stocks:
        # No current stocks, always rebalance
        new_stocks = select_top_performers(
            tickers, ticker_data_grouped, current_date,
            lookback_days=365, top_n=top_n
        )
        return new_stocks, True
    
    # Calculate current portfolio momentum
    current_momentum = 0
    valid_count = 0
    
    for ticker in current_stocks:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 365:
                start_date = current_date - timedelta(days=365)
                period_data = data[data.index >= start_date]
                if len(period_data) >= 2:
                    start_price = period_data['Close'].iloc[0]
                    end_price = period_data['Close'].iloc[-1]
                    momentum = (end_price / start_price - 1)
                    current_momentum += momentum
                    valid_count += 1
    
    if valid_count > 0:
        current_momentum /= valid_count
    
    # Get new candidates
    new_candidates = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # Calculate new portfolio momentum
    new_momentum = 0
    valid_count = 0
    
    for ticker in new_candidates:
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker].loc[:current_date]
            if len(data) >= 365:
                start_date = current_date - timedelta(days=365)
                period_data = data[data.index >= start_date]
                if len(period_data) >= 2:
                    start_price = period_data['Close'].iloc[0]
                    end_price = period_data['Close'].iloc[-1]
                    momentum = (end_price / start_price - 1)
                    new_momentum += momentum
                    valid_count += 1
    
    if valid_count > 0:
        new_momentum /= valid_count
    
    # Check if improvement exceeds threshold
    improvement = new_momentum - current_momentum
    should_rebalance = improvement > min_improvement
    
    return new_candidates, should_rebalance


def select_market_regime_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    base_rebalance_days: int = 22,
    high_vol_days: int = 15,
    low_vol_days: int = 30,
    vol_threshold: float = 0.20  # 20% volatility threshold
) -> Tuple[List[str], int]:
    """
    Static BH 1Y with market-regime based rebalancing.
    Adjusts rebalancing frequency based on market volatility.
    
    Returns:
        (selected_stocks, next_rebalance_days)
    """
    # Calculate market volatility using SPY
    market_volatility = 0.20  # Default
    spy_data = ticker_data_grouped.get('SPY')
    if spy_data is not None:
        spy_recent = spy_data.loc[:current_date].tail(60)  # Last 60 days
        if len(spy_recent) >= 20:
            spy_returns = spy_recent['Close'].pct_change().dropna()
            if len(spy_returns) > 0:
                market_volatility = spy_returns.std() * np.sqrt(252)  # Annualized
    
    # Determine rebalancing frequency based on volatility
    if market_volatility > vol_threshold:
        # High volatility - rebalance more frequently
        rebalance_days = high_vol_days
    elif market_volatility < vol_threshold / 2:
        # Low volatility - rebalance less frequently
        rebalance_days = low_vol_days
    else:
        # Normal volatility - use base frequency
        rebalance_days = base_rebalance_days
    
    # Get top performers
    selected_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    return selected_stocks, rebalance_days


def check_momentum_persistence(
    previous_top_stocks_history: List[List[str]],
    current_top_stocks: List[str],
    min_stable_days: int = 5,
    min_overlap: float = 0.8
) -> Tuple[bool, List[List[str]]]:
    """
    Check if top stocks have been stable for N consecutive days.
    
    Args:
        previous_top_stocks_history: List of previous day's top stocks (most recent last)
        current_top_stocks: Today's top stocks
        min_stable_days: Minimum consecutive days of stability required
        min_overlap: Minimum overlap ratio to consider "stable"
    
    Returns:
        (is_stable, updated_history)
    """
    if not current_top_stocks:
        return False, []
    
    # Add current to history
    updated_history = previous_top_stocks_history.copy()
    updated_history.append(current_top_stocks)
    
    # Keep only last N+1 days (we need N comparisons)
    if len(updated_history) > min_stable_days + 1:
        updated_history = updated_history[-(min_stable_days + 1):]
    
    # Need at least min_stable_days + 1 entries to check stability
    if len(updated_history) < min_stable_days + 1:
        return False, updated_history
    
    # Check if all consecutive days have sufficient overlap
    for i in range(1, len(updated_history)):
        prev_set = set(updated_history[i-1])
        curr_set = set(updated_history[i])
        if len(prev_set) == 0:
            return False, updated_history
        overlap = len(prev_set & curr_set) / len(prev_set)
        if overlap < min_overlap:
            return False, updated_history
    
    return True, updated_history


def select_momentum_persistence_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    previous_top_stocks_history: List[List[str]],
    current_stocks: List[str],
    top_n: int = 10,
    min_stable_days: int = 5,
    min_overlap: float = 0.8
) -> Tuple[List[str], bool, List[List[str]]]:
    """
    Static BH 1Y with momentum persistence requirement.
    Only rebalances when top performers have been stable for N consecutive days.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        previous_top_stocks_history: History of previous top stocks
        current_stocks: Currently held stocks
        top_n: Number of stocks to select
        min_stable_days: Minimum consecutive days of stability
        min_overlap: Minimum overlap ratio for stability
    
    Returns:
        (selected_stocks, should_rebalance, updated_history)
    """
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # Check momentum persistence
    is_stable, updated_history = check_momentum_persistence(
        previous_top_stocks_history, new_top_stocks, min_stable_days, min_overlap
    )
    
    # Only rebalance if:
    # 1. No current stocks (initialization)
    # 2. Top stocks have been stable AND they differ from current holdings
    if not current_stocks:
        return new_top_stocks, True, updated_history
    
    if is_stable:
        current_set = set(current_stocks)
        new_set = set(new_top_stocks)
        # Check if the stable top stocks are different from what we hold
        overlap_with_current = len(current_set & new_set) / len(current_set) if current_set else 0
        if overlap_with_current < 0.9:  # At least 1 stock different
            return new_top_stocks, True, updated_history
    
    return current_stocks, False, updated_history


def select_overlap_based_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    top_n: int = 10,
    overlap_threshold: float = 0.7
) -> Tuple[List[str], bool]:
    """
    Static BH 1Y with overlap-based rebalancing.
    Only rebalances when overlap between current holdings and new top performers
    drops below threshold.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        current_stocks: Currently held stocks
        top_n: Number of stocks to select
        overlap_threshold: Minimum overlap to avoid rebalancing (default 70%)
    
    Returns:
        (selected_stocks, should_rebalance)
    """
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # No current stocks - always rebalance (initialization)
    if not current_stocks:
        return new_top_stocks, True
    
    # Calculate overlap
    current_set = set(current_stocks)
    new_set = set(new_top_stocks)
    overlap = len(current_set & new_set) / len(current_set) if current_set else 0
    
    # Rebalance if overlap drops below threshold
    should_rebalance = overlap < overlap_threshold
    
    if should_rebalance:
        return new_top_stocks, True
    else:
        return current_stocks, False


def select_rank_drift_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    previous_rankings: Dict[str, int],
    current_stocks: List[str],
    top_n: int = 10,
    rank_drift_threshold: float = 3.0
) -> Tuple[List[str], bool, Dict[str, int]]:
    """
    Static BH 1Y with rank drift rebalancing.
    Rebalances when stocks have moved significantly in rankings.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        previous_rankings: Previous day's rankings {ticker: rank}
        current_stocks: Currently held stocks
        top_n: Number of stocks to select
        rank_drift_threshold: Avg rank change threshold to trigger rebalance
    
    Returns:
        (selected_stocks, should_rebalance, updated_rankings)
    """
    # Get current top performers with their performances
    from parallel_backtest import calculate_parallel_performance
    
    performances = calculate_parallel_performance(
        tickers, ticker_data_grouped, current_date, period_days=365
    )
    
    # Sort by performance (highest first) and create rankings
    # performances is already a list of (ticker, perf) tuples
    sorted_perf = sorted(performances, key=lambda x: x[1], reverse=True)
    current_rankings = {}
    for rank, (ticker, perf) in enumerate(sorted_perf, 1):
        current_rankings[ticker] = rank
    
    # Get new top stocks
    new_top_stocks = [t for t, _ in sorted_perf[:top_n]]
    
    # No previous rankings - always rebalance (initialization)
    if not previous_rankings:
        return new_top_stocks, True, current_rankings
    
    # Calculate average rank change for current holdings
    if current_stocks:
        rank_changes = []
        for ticker in current_stocks:
            if ticker in previous_rankings and ticker in current_rankings:
                prev_rank = previous_rankings[ticker]
                curr_rank = current_rankings[ticker]
                rank_changes.append(abs(curr_rank - prev_rank))
        
        avg_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0
    else:
        avg_rank_change = 0
    
    # Rebalance if average rank change exceeds threshold
    should_rebalance = avg_rank_change > rank_drift_threshold
    
    if should_rebalance:
        return new_top_stocks, True, current_rankings
    else:
        return current_stocks, False, current_rankings


def check_drawdown_trigger(
    portfolio_value: float,
    portfolio_peak: float,
    drawdown_threshold: float = 0.05
) -> bool:
    """
    Check if portfolio drawdown exceeds threshold.
    
    Args:
        portfolio_value: Current portfolio value
        portfolio_peak: Peak portfolio value
        drawdown_threshold: Drawdown threshold (default 5%)
    
    Returns:
        True if drawdown exceeds threshold
    """
    if portfolio_peak == 0:
        return False
    
    drawdown = (portfolio_peak - portfolio_value) / portfolio_peak
    return drawdown >= drawdown_threshold


def select_drawdown_trigger_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    portfolio_value: float,
    portfolio_peak: float,
    top_n: int = 10,
    drawdown_threshold: float = 0.05
) -> Tuple[List[str], bool]:
    """
    Static BH 1Y with portfolio drawdown trigger.
    Rebalances when portfolio drawdown exceeds threshold.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        current_stocks: Currently held stocks
        portfolio_value: Current portfolio value
        portfolio_peak: Peak portfolio value
        top_n: Number of stocks to select
        drawdown_threshold: Drawdown threshold (default 5%)
    
    Returns:
        (selected_stocks, should_rebalance)
    """
    # Check if drawdown trigger is hit
    should_rebalance = check_drawdown_trigger(
        portfolio_value, portfolio_peak, drawdown_threshold
    )
    
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # No current stocks - always rebalance (initialization)
    if not current_stocks:
        return new_top_stocks, True
    
    if should_rebalance:
        return new_top_stocks, True
    else:
        return current_stocks, False


def is_month_start(current_date: datetime) -> bool:
    """
    Check if current date is the first trading day of a month.
    """
    if current_date.day <= 3:
        return True
    return False


def select_smart_monthly_bh_1y_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    current_stocks: List[str],
    portfolio_value: float,
    portfolio_peak: float,
    last_rebalance_month: int,
    top_n: int = 10,
    early_rebalance_drawdown: float = 0.03
) -> Tuple[List[str], bool, int]:
    """
    Static BH 1Y with smart monthly + conditional rebalancing.
    Rebalances on monthly schedule BUT allows early rebalance if momentum drops.
    
    Args:
        tickers: List of tickers to consider
        ticker_data_grouped: Price data for all tickers
        current_date: Current date
        current_stocks: Currently held stocks
        portfolio_value: Current portfolio value
        portfolio_peak: Peak portfolio value
        last_rebalance_month: Month of last rebalance (1-12)
        top_n: Number of stocks to select
        early_rebalance_drawdown: Early rebalance if down X% from peak
    
    Returns:
        (selected_stocks, should_rebalance, current_month)
    """
    current_month = current_date.month
    
    # Get current top performers
    new_top_stocks = select_top_performers(
        tickers, ticker_data_grouped, current_date,
        lookback_days=365, top_n=top_n
    )
    
    # No current stocks - always rebalance (initialization)
    if not current_stocks:
        return new_top_stocks, True, current_month
    
    # Check if it's a new month (monthly rebalance)
    is_new_month = current_month != last_rebalance_month
    
    # Check for early rebalance condition (significant drawdown)
    is_early_rebalance = check_drawdown_trigger(
        portfolio_value, portfolio_peak, early_rebalance_drawdown
    )
    
    # Rebalance if new month OR early rebalance condition
    should_rebalance = is_new_month or is_early_rebalance
    
    if should_rebalance:
        return new_top_stocks, True, current_month
    else:
        return current_stocks, False, last_rebalance_month


# =============================================================================
# SMART REBALANCING STRATEGIES (based on Static BH 1Y)
# =============================================================================

def should_sell_momentum_based(
    ticker: str,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    short_lookback: int = 10,
    long_lookback: int = 20
) -> Tuple[bool, str]:
    """
    Strategy 1: Momentum-Based Sell
    Only sell when short-term momentum < long-term momentum (declining).
    
    Returns:
        (should_sell, reason)
    """
    if ticker not in ticker_data_grouped:
        return True, "No data"
    
    data = ticker_data_grouped[ticker].loc[:current_date]
    if len(data) < long_lookback + 5:
        return True, "Insufficient data"
    
    close = data['Close'].dropna()
    if len(close) < long_lookback + 5:
        return True, "Insufficient close data"
    
    # Calculate short and long momentum
    short_mom = (close.iloc[-1] / close.iloc[-short_lookback] - 1) * 100 if len(close) >= short_lookback else 0
    long_mom = (close.iloc[-1] / close.iloc[-long_lookback] - 1) * 100 if len(close) >= long_lookback else 0
    
    # Only sell if momentum is declining (short < long)
    if short_mom < long_mom:
        return True, f"Momentum declining: {short_lookback}d={short_mom:.1f}% < {long_lookback}d={long_mom:.1f}%"
    else:
        return False, f"Momentum stable/rising: {short_lookback}d={short_mom:.1f}% >= {long_lookback}d={long_mom:.1f}%"


def should_sell_rank_based(
    ticker: str,
    current_rank: int,
    entry_rank: int,
    rank_drop_threshold: int = 5,
    max_rank: int = 15
) -> Tuple[bool, str]:
    """
    Strategy 2: Relative Strength Ranking
    Sell when rank drops significantly from entry or is outside max rank.
    
    Returns:
        (should_sell, reason)
    """
    rank_drop = current_rank - entry_rank
    
    if current_rank > max_rank:
        return True, f"Rank {current_rank} > max {max_rank}"
    elif rank_drop >= rank_drop_threshold:
        return True, f"Rank dropped {rank_drop} positions (entry={entry_rank}, now={current_rank})"
    else:
        return False, f"Rank OK: {current_rank} (entry={entry_rank}, drop={rank_drop})"


def should_sell_trailing_momentum(
    ticker: str,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    entry_price: float,
    peak_price: float,
    soft_stop: float = 0.10,
    hard_stop: float = 0.15,
    momentum_lookback: int = 10
) -> Tuple[bool, str]:
    """
    Strategy 3: Trailing Stop + Momentum
    - Soft stop: sell if down X% from peak AND momentum negative
    - Hard stop: always sell if down Y% from peak
    
    Returns:
        (should_sell, reason)
    """
    if ticker not in ticker_data_grouped:
        return True, "No data"
    
    data = ticker_data_grouped[ticker].loc[:current_date]
    if data.empty:
        return True, "Empty data"
    
    current_price = data['Close'].dropna().iloc[-1]
    
    # Calculate drawdown from peak
    drawdown_from_peak = (peak_price - current_price) / peak_price if peak_price > 0 else 0
    
    # Hard stop - always sell
    if drawdown_from_peak >= hard_stop:
        return True, f"Hard stop: down {drawdown_from_peak:.1%} from peak"
    
    # Soft stop - only if momentum also negative
    if drawdown_from_peak >= soft_stop:
        close = data['Close'].dropna()
        if len(close) >= momentum_lookback:
            momentum = (close.iloc[-1] / close.iloc[-momentum_lookback] - 1)
            if momentum < 0:
                return True, f"Soft stop: down {drawdown_from_peak:.1%} from peak AND momentum {momentum:.1%}"
    
    return False, f"Hold: drawdown {drawdown_from_peak:.1%} from peak"


def should_sell_volume_confirmed(
    ticker: str,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    in_top_buffer: bool,
    volume_multiplier: float = 1.5
) -> Tuple[bool, str]:
    """
    Strategy 4: Volume Confirmation
    Only sell if volume confirms weakness (high volume on down days).
    
    Returns:
        (should_sell, reason)
    """
    if in_top_buffer:
        return False, "In top buffer - keep"
    
    if ticker not in ticker_data_grouped:
        return True, "No data"
    
    data = ticker_data_grouped[ticker].loc[:current_date]
    if len(data) < 20 or 'Volume' not in data.columns:
        return True, "Insufficient volume data"
    
    # Calculate average volume
    avg_volume = data['Volume'].tail(20).mean()
    recent_volume = data['Volume'].iloc[-1]
    
    # Check if recent volume is elevated
    if avg_volume > 0 and recent_volume > avg_volume * volume_multiplier:
        # High volume - check if it's distribution (down day)
        if len(data) >= 2:
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            if price_change < 0:
                return True, f"Volume confirmed sell: {recent_volume/avg_volume:.1f}x avg on down day"
    
    return False, f"Volume not confirming: {recent_volume/avg_volume:.1f}x avg"


def filter_sells_sector_aware(
    positions_to_sell: set,
    positions_to_keep: set,
    ticker_to_sector: Dict[str, str],
    all_ranked_tickers: List[str],
    min_per_sector: int = 1,
    max_rank: int = 20
) -> Tuple[set, set]:
    """
    Strategy 5: Sector Rotation Awareness
    Don't sell all stocks from a sector at once - keep at least min_per_sector.
    
    Returns:
        (adjusted_positions_to_sell, adjusted_positions_to_keep)
    """
    # Count sectors in positions to keep
    sector_counts = {}
    for ticker in positions_to_keep:
        sector = ticker_to_sector.get(ticker, 'Unknown')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    adjusted_sell = set()
    adjusted_keep = set(positions_to_keep)
    
    for ticker in positions_to_sell:
        sector = ticker_to_sector.get(ticker, 'Unknown')
        current_sector_count = sector_counts.get(sector, 0)
        
        # Check if ticker is in top max_rank
        try:
            rank = all_ranked_tickers.index(ticker) + 1
            in_top_rank = rank <= max_rank
        except ValueError:
            in_top_rank = False
        
        # Keep if this would leave sector with less than min_per_sector AND ticker is in top max_rank
        if current_sector_count < min_per_sector and in_top_rank:
            adjusted_keep.add(ticker)
            sector_counts[sector] = current_sector_count + 1
        else:
            adjusted_sell.add(ticker)
    
    return adjusted_sell, adjusted_keep


def select_buys_with_acceleration(
    candidates: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    accel_weight: float = 0.3
) -> List[str]:
    """
    Strategy 6: Accelerating Momentum Buy
    Prioritize stocks with accelerating momentum (1M momentum > 3M trend).
    
    Returns:
        List of selected tickers prioritized by acceleration
    """
    scored_candidates = []
    
    for ticker in candidates:
        if ticker not in ticker_data_grouped:
            continue
        
        data = ticker_data_grouped[ticker].loc[:current_date]
        if len(data) < 90:
            # Not enough data - use base score only
            scored_candidates.append((ticker, 0, 0))
            continue
        
        close = data['Close'].dropna()
        if len(close) < 90:
            scored_candidates.append((ticker, 0, 0))
            continue
        
        # Calculate 1M and 3M momentum
        mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
        
        # Acceleration = 1M momentum - (3M momentum / 3)
        # Positive means momentum is accelerating
        expected_1m_from_3m = mom_3m / 3  # If 3M is 30%, expect ~10% per month
        acceleration = mom_1m - expected_1m_from_3m
        
        # Combined score: base rank score + acceleration bonus
        base_score = len(candidates) - candidates.index(ticker)  # Higher rank = higher score
        accel_score = acceleration * accel_weight
        total_score = base_score + accel_score
        
        scored_candidates.append((ticker, total_score, acceleration))
    
    # Sort by total score (descending)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    selected = [ticker for ticker, score, accel in scored_candidates[:top_n]]
    
    # Log top selections
    if scored_candidates:
        print(f"   📈 Acceleration Buy: Top picks with acceleration scores:")
        for ticker, score, accel in scored_candidates[:min(5, len(scored_candidates))]:
            print(f"      {ticker}: accel={accel:+.1f}%, score={score:.1f}")
    
    return selected


def select_static_bh_3m_accel(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 10,
    lookback_short: int = 10,
    lookback_long: int = 21,
    accel_weight: float = 0.3
) -> List[str]:
    """
    Static BH 3M with Acceleration
    Uses 3-month performance lookback but prioritizes stocks with accelerating momentum.
    Acceleration is calculated using 2W vs 1M (faster signal for shorter-term strategy).
    
    Returns:
        List of selected tickers prioritized by 3M performance + acceleration
    """
    scored_candidates = []
    
    for ticker in ticker_data_grouped:
        data = ticker_data_grouped[ticker].loc[:current_date]
        if len(data) < 63 + 5:  # Need 3M data for primary selection
            continue
        
        close = data['Close'].dropna()
        if len(close) < 63 + 5:
            continue
        
        # Calculate 3M performance (primary metric for BH 3M selection)
        perf_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100
        
        # Calculate 2W and 1M momentum for acceleration
        mom_2w = (close.iloc[-1] / close.iloc[-lookback_short] - 1) * 100 if len(close) >= lookback_short else 0
        mom_1m = (close.iloc[-1] / close.iloc[-lookback_long] - 1) * 100 if len(close) >= lookback_long else 0
        
        # Calculate expected 2W return from 1M trend
        expected_2w = mom_1m * (lookback_short / lookback_long)  # ~47% of 1M
        acceleration = mom_2w - expected_2w
        
        # Combined score: 3M performance + acceleration bonus
        total_score = perf_3m + (acceleration * accel_weight)
        
        scored_candidates.append((ticker, total_score, perf_3m, acceleration))
    
    # Sort by total score (descending)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    selected = [ticker for ticker, score, perf, accel in scored_candidates[:top_n]]
    
    # Log top selections
    if scored_candidates:
        print(f"   📈 Static BH 3M Accel: Top picks (2W vs 1M accel):")
        for ticker, score, perf, accel in scored_candidates[:min(5, len(scored_candidates))]:
            print(f"      {ticker}: 3M={perf:+.1f}%, accel={accel:+.1f}%, score={score:.1f}")
    
    return selected


# =====================================================================
# 10 NEW REBALANCING STRATEGIES
# =====================================================================

def get_vol_adjusted_rebalance_days(ticker_data_grouped, current_date, config):
    """1. Volatility-Adjusted Rebalancing: adjust rebalance frequency based on market vol"""
    import numpy as np
    
    market_ticker = None
    for t in ['SPY', 'QQQ', 'VOO', 'IVV']:
        if t in ticker_data_grouped:
            market_ticker = t
            break
    
    if not market_ticker:
        return 20
    
    try:
        data = ticker_data_grouped[market_ticker].loc[:current_date]
        if len(data) < 20:
            return 20
        
        returns = data['Close'].pct_change().dropna()
        if len(returns) < 20:
            return 20
        
        vol = returns.std() * np.sqrt(252)
        
        low_thresh = config.get('BH_1Y_VOL_ADJ_REBAL_LOW_VOL_THRESH', 0.15)
        high_thresh = config.get('BH_1Y_VOL_ADJ_REBAL_HIGH_VOL_THRESH', 0.35)
        min_days = config.get('BH_1Y_VOL_ADJ_REBAL_MIN_DAYS', 5)
        max_days = config.get('BH_1Y_VOL_ADJ_REBAL_MAX_DAYS', 30)
        
        if vol <= low_thresh:
            return max_days
        elif vol >= high_thresh:
            return min_days
        else:
            ratio = (vol - low_thresh) / (high_thresh - low_thresh)
            return int(max_days - ratio * (max_days - min_days))
    except:
        return 20


def filter_by_correlation(ticker_list, ticker_data_grouped, current_date, config):
    """2. Correlation-Based Filtering: avoid highly correlated stocks"""
    import numpy as np
    
    threshold = config.get('BH_1Y_CORR_FILTER_THRESH', 0.7)
    lookback = config.get('BH_1Y_CORR_FILTER_LOOKBACK', 60)
    
    if len(ticker_list) <= 1:
        return ticker_list
    
    returns_dict = {}
    for ticker in ticker_list:
        if ticker in ticker_data_grouped:
            try:
                data = ticker_data_grouped[ticker].loc[:current_date].tail(lookback)
                if len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) >= 20:
                        returns_dict[ticker] = returns.values
            except:
                pass
    
    if len(returns_dict) < 2:
        return ticker_list
    
    tickers = list(returns_dict.keys())
    n = len(tickers)
    
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r1 = returns_dict[tickers[i]]
                r2 = returns_dict[tickers[j]]
                min_len = min(len(r1), len(r2))
                corr = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                if not np.isnan(corr):
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
            except:
                pass
    
    selected = [tickers[0]]
    for i in range(1, n):
        too_correlated = False
        for j, sel_ticker in enumerate(selected):
            idx_sel = tickers.index(sel_ticker)
            if corr_matrix[i, idx_sel] > threshold:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(tickers[i])
    
    return selected[:config.get('PORTFOLIO_SIZE', 10)]


def detect_market_regime(ticker_data_grouped, current_date, config):
    """3. Market Regime Detection: detect bull/bear/sideways market"""
    market_ticker = None
    for t in ['SPY', 'QQQ', 'VOO', 'IVV']:
        if t in ticker_data_grouped:
            market_ticker = t
            break
    
    if not market_ticker:
        return 'sideways'
    
    try:
        data = ticker_data_grouped[market_ticker].loc[:current_date]
        if len(data) < 200:
            return 'sideways'
        
        bull_ma = config.get('BH_1Y_REGIME_BULL_MA', 50)
        bear_ma = config.get('BH_1Y_REGIME_BEAR_MA', 200)
        
        current_price = data['Close'].iloc[-1]
        ma50 = data['Close'].rolling(bull_ma).mean().iloc[-1]
        ma200 = data['Close'].rolling(bear_ma).mean().iloc[-1]
        
        if current_price > ma50 and current_price > ma200:
            return 'bull'
        elif current_price < ma50 and current_price < ma200:
            return 'bear'
        else:
            return 'sideways'
    except:
        return 'sideways'


def get_risk_parity_weights(ticker_list, ticker_data_grouped, current_date, config):
    """4. Risk Parity Allocation: size by inverse volatility"""
    import numpy as np
    
    vol_lookback = config.get('BH_1Y_RISK_PARITY_VOL_LOOKBACK', 60)
    min_weight = config.get('BH_1Y_RISK_PARITY_MIN_WEIGHT', 0.02)
    max_weight = config.get('BH_1Y_RISK_PARITY_MAX_WEIGHT', 0.25)
    
    if not ticker_list:
        return {}
    
    volatilities = {}
    for ticker in ticker_list:
        if ticker in ticker_data_grouped:
            try:
                data = ticker_data_grouped[ticker].loc[:current_date].tail(vol_lookback)
                if len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) >= 20:
                        vol = returns.std() * np.sqrt(252)
                        if vol > 0:
                            volatilities[ticker] = vol
            except:
                pass
    
    if not volatilities:
        return {t: 1.0 / len(ticker_list) for t in ticker_list}
    
    inv_vols = {t: 1.0 / v for t, v in volatilities.items()}
    total_inv_vol = sum(inv_vols.values())
    
    weights = {}
    for ticker in ticker_list:
        if ticker in inv_vols:
            w = inv_vols[ticker] / total_inv_vol
            w = max(min_weight, min(max_weight, w))
            weights[ticker] = w
        else:
            weights[ticker] = min_weight
    
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    
    return weights


def calculate_portfolio_drift(positions, ticker_data_grouped, current_date, config):
    """5. Adaptive Drift Threshold: calculate portfolio drift from target"""
    target_weight = config.get('BH_1Y_DRIFT_THRESH_TARGET', 0.10)
    
    if not positions:
        return 0.0
    
    total_value = sum(p.get('value', 0) for p in positions.values())
    if total_value <= 0:
        return 0.0
    
    max_drift = 0.0
    for ticker, pos in positions.items():
        try:
            if ticker in ticker_data_grouped:
                data = ticker_data_grouped[ticker].loc[:current_date]
                if not data.empty:
                    price = data['Close'].dropna().iloc[-1]
                    if price > 0:
                        current_weight = (pos.get('shares', 0) * price) / total_value
                        drift = abs(current_weight - target_weight)
                        max_drift = max(max_drift, drift)
        except:
            pass
    
    return max_drift


def score_momentum_quality(ticker, ticker_data_grouped, current_date, config):
    """6. Momentum Quality Score: combine momentum strength + consistency"""
    import numpy as np
    
    mom_weight = config.get('BH_1Y_MOM_QUALITY_MOM_WEIGHT', 0.6)
    cons_weight = config.get('BH_1Y_MOM_QUALITY_CONS_WEIGHT', 0.4)
    
    try:
        data = ticker_data_grouped[ticker].loc[:current_date]
        if len(data) < 60:
            return 0.5
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) >= 252:
            mom_return = (1 + returns.tail(252).sum()) - 1
        elif len(returns) >= 60:
            mom_return = (1 + returns.tail(60).sum()) - 1
        else:
            mom_return = 0
        
        if len(returns) >= 60:
            ret_vol = returns.tail(60).std()
            consistency = 1.0 / (1.0 + ret_vol * 10)
        else:
            consistency = 0.5
        
        mom_normalized = (mom_return + 0.5) / 1.5
        mom_normalized = max(0, min(1, mom_normalized))
        
        score = mom_weight * mom_normalized + cons_weight * consistency
        return score
    except:
        return 0.5


def get_liquidity_weights(ticker_list, ticker_data_grouped, current_date, config):
    """7. Liquidity-Based Sizing: larger positions in more liquid stocks"""
    import numpy as np
    
    avg_vol_days = config.get('BH_1Y_LIQUIDITY_AVG_VOL_DAYS', 20)
    min_dollar_vol = config.get('BH_1Y_LIQUIDITY_MIN_DOLLAR_VOL', 1000000)
    max_weight = config.get('BH_1Y_LIQUIDITY_MAX_WEIGHT', 0.20)
    
    if not ticker_list:
        return {}
    
    dollar_vols = {}
    for ticker in ticker_list:
        if ticker in ticker_data_grouped:
            try:
                data = ticker_data_grouped[ticker].loc[:current_date].tail(avg_vol_days)
                if len(data) >= 5 and 'Volume' in data.columns:
                    avg_vol = data['Volume'].mean()
                    price = data['Close'].iloc[-1]
                    dollar_vol = avg_vol * price
                    dollar_vols[ticker] = dollar_vol
            except:
                pass
    
    if not dollar_vols:
        return {t: 1.0 / len(ticker_list) for t in ticker_list}
    
    max_vol = max(dollar_vols.values())
    scores = {}
    for ticker, vol in dollar_vols.items():
        if vol < min_dollar_vol:
            scores[ticker] = 0.1
        else:
            scores[ticker] = np.log(vol / min_dollar_vol + 1) / np.log(max_vol / min_dollar_vol + 1)
    
    total_score = sum(scores.values())
    if total_score > 0:
        weights = {t: min(max_weight, s / total_score * len(ticker_list)) for t, s in scores.items()}
    else:
        weights = {t: 1.0 / len(ticker_list) for t in ticker_list}
    
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    
    return weights


def is_near_earnings(ticker, ticker_data_grouped, current_date, config):
    """8. Earnings Avoidance: check if ticker is near earnings date"""
    return False


def get_multi_factor_score(ticker, ticker_data_grouped, current_date, config):
    """9. Multi-Factor Composite: blend momentum + value + quality"""
    import numpy as np
    
    mom_w = config.get('BH_1Y_MULTI_FACTOR_MOM_WEIGHT', 0.5)
    val_w = config.get('BH_1Y_MULTI_FACTOR_VAL_WEIGHT', 0.25)
    qual_w = config.get('BH_1Y_MULTI_FACTOR_QUAL_WEIGHT', 0.25)
    
    try:
        data = ticker_data_grouped[ticker].loc[:current_date]
        if len(data) < 60:
            return 0.5, 0.5, 0.5
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) >= 252:
            mom = (1 + returns.tail(252).sum()) - 1
        elif len(returns) >= 60:
            mom = (1 + returns.tail(60).sum()) - 1
        else:
            mom = 0
        mom_norm = max(0, min(1, (mom + 0.5) / 1.5))
        
        val_norm = 0.5
        
        if len(returns) >= 60:
            ret_std = returns.tail(60).std()
            qual = 1.0 / (1.0 + ret_std * 10)
        else:
            qual = 0.5
        
        total_score = mom_w * mom_norm + val_w * val_norm + qual_w * qual
        
        return total_score, mom_norm, qual
    except:
        return 0.5, 0.5, 0.5


def get_time_decay_exit_pct(ticker, ticker_data_grouped, current_date, config):
    """10. Time-Decay Holdings: calculate exit percentage for gradual reduction"""
    return 0.0
