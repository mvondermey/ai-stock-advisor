"""
Shared Strategy Implementations
Used by both backtesting and live trading to ensure identical logic.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Import config for strategy parameters
from config import (
    RISK_ADJ_MOM_PERFORMANCE_WINDOW, RISK_ADJ_MOM_VOLATILITY_WINDOW,
    RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_CONFIRM_SHORT,
    RISK_ADJ_MOM_CONFIRM_MEDIUM, RISK_ADJ_MOM_CONFIRM_LONG, RISK_ADJ_MOM_MIN_CONFIRMATIONS,
    RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION, RISK_ADJ_MOM_VOLUME_WINDOW, RISK_ADJ_MOM_VOLUME_MULTIPLIER,
    RISK_ADJ_MOM_MIN_SCORE, VOLATILITY_ADJ_MOM_LOOKBACK, VOLATILITY_ADJ_MOM_VOL_WINDOW,
    VOLATILITY_ADJ_MOM_MIN_SCORE, DATA_FRESHNESS_MAX_DAYS,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M, MIN_DATA_DAYS_1M, MIN_DATA_DAYS_GENERAL,
    ENABLE_INVERSE_ETF_HEDGE, INVERSE_ETF_HEDGE_THRESHOLD_LOW, INVERSE_ETF_HEDGE_BASE_ALLOCATION, INVERSE_ETF_HEDGE_PREFERENCE
)

# Import AI Elite helper functions
from ai_elite_strategy import _calculate_market_return


# Global tracking for inverse ETF hedges used
_inverse_etf_hedge_log = []  # List of (date, etf, market_decline) tuples


def get_inverse_etf_hedge_log() -> List[tuple]:
    """Get the log of inverse ETF hedges used."""
    return _inverse_etf_hedge_log.copy()


def clear_inverse_etf_hedge_log():
    """Clear the inverse ETF hedge log (call at start of backtest)."""
    global _inverse_etf_hedge_log
    _inverse_etf_hedge_log = []


def apply_inverse_etf_hedge(
    selected_stocks: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    portfolio_size: int = 10
) -> List[str]:
    """
    Add inverse ETFs to selected stocks during market downturns.

    This replaces stop losses by hedging with inverse ETFs when market crashes.

    Args:
        selected_stocks: Stocks selected by strategy
        ticker_data_grouped: All ticker data
        current_date: Current analysis date
        portfolio_size: Target portfolio size

    Returns:
        Updated stock list with inverse ETFs if market is down
    """
    global _inverse_etf_hedge_log

    if not ENABLE_INVERSE_ETF_HEDGE:
        return selected_stocks

    # Get market conditions
    market_conditions = get_market_conditions(ticker_data_grouped, current_date)

    # Check if market is down significantly
    market_decline = 0
    for metric, value in market_conditions.items():
        if '_3m' in metric and value < 0:  # Negative 3-month performance
            market_decline = max(market_decline, abs(value))

    # Add hedge if market is down more than threshold
    if market_decline <= INVERSE_ETF_HEDGE_THRESHOLD_LOW:
        return selected_stocks

    # Calculate how many positions to replace with hedge
    num_hedge_positions = max(1, int(portfolio_size * INVERSE_ETF_HEDGE_BASE_ALLOCATION))

    # Remove worst performers to make room
    updated_stocks = selected_stocks[:-num_hedge_positions] if len(selected_stocks) > portfolio_size - num_hedge_positions else selected_stocks.copy()

    # Add preferred inverse ETFs
    for etf in INVERSE_ETF_HEDGE_PREFERENCE:
        if etf not in updated_stocks and etf in ticker_data_grouped and len(updated_stocks) < portfolio_size:
            updated_stocks.append(etf)
            print(f"   🛡️ Adding hedge {etf} (market down {market_decline:.1%})")
            # Log the hedge
            _inverse_etf_hedge_log.append((current_date, etf, market_decline))
            break  # Add only one hedge ETF for simplicity

    return updated_stocks


def get_market_conditions(ticker_data_grouped: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
    """Get current market conditions using major indices."""
    conditions = {}

    # Check major indices
    indices = {
        'SPY': 'sp500',
        'QQQ': 'nasdaq',
        'IWM': 'russell2000'
    }

    # Convert current_date to pandas Timestamp
    current_ts = pd.Timestamp(current_date)

    for ticker, name in indices.items():
        if ticker in ticker_data_grouped:
            data = ticker_data_grouped[ticker]

            # Filter data up to current_date (important for backtesting)
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(data.index.tz)

            data_until_now = data[data.index <= current_ts]

            # 3-month performance (90 calendar days)
            start_3m = current_ts - timedelta(days=90)
            data_3m = data_until_now[data_until_now.index >= start_3m]
            if len(data_3m) >= 10:
                perf_3m = (data_3m['Close'].iloc[-1] / data_3m['Close'].iloc[0] - 1)
                conditions[f'{name}_3m'] = perf_3m

            # 1-month performance (30 calendar days)
            start_1m = current_ts - timedelta(days=30)
            data_1m = data_until_now[data_until_now.index >= start_1m]
            if len(data_1m) >= 5:
                perf_1m = (data_1m['Close'].iloc[-1] / data_1m['Close'].iloc[0] - 1)
                conditions[f'{name}_1m'] = perf_1m

    return conditions


def calculate_risk_adjusted_momentum_score(ticker_data: pd.DataFrame, current_date: datetime = None, skip_freshness_check: bool = False) -> tuple:
    """
    Calculate risk-adjusted momentum score for a ticker.

    Args:
        ticker_data: DataFrame with price data indexed by date
        current_date: Current date for analysis (None uses data's max date)
        skip_freshness_check: If True, skip data freshness validation (for live trading)

    Returns:
        tuple: (score, return_pct, volatility_pct) or (0, 0, 0) if insufficient data
    """
    from config import MIN_DATA_DAYS_MOMENTUM_CONFIRM
    if len(ticker_data) < MIN_DATA_DAYS_MOMENTUM_CONFIRM:
        return 0.0, 0.0, 0.0

    # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
    end_date = ticker_data.index.max()

    # If current_date is provided, try to use it (must convert to pandas Timestamp first)
    if current_date is not None:
        try:
            # Convert to pandas Timestamp (handles both datetime and Timestamp)
            current_ts = pd.Timestamp(current_date)

            # If data has timezone, ensure current_ts matches
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    # current_ts is naive, localize to data's timezone
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    # current_ts has timezone, convert to data's timezone
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)

            # Check data freshness - warn if data is older than configured limit
            if not skip_freshness_check:
                data_age_days = (current_ts - end_date).total_seconds() / 86400
                if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                    # Data is stale - raise error to be caught by caller
                    raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days). Data end: {end_date}, Current: {current_ts}")
        except ValueError:
            # Re-raise ValueError (stale data error) so caller can handle it
            raise
        except Exception as e:
            # For other errors (timezone conversion, etc), stick with data's max date
            print(f"Error calculating risk-adjusted momentum score: {e}")
            pass

    # Calculate 1-year performance (rolling window)
    start_date = end_date - timedelta(days=RISK_ADJ_MOM_PERFORMANCE_WINDOW)

    # ✅ Use boolean indexing instead of .loc[] to avoid KeyError with timezone mismatches
    perf_data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= end_date)]

    from config import MIN_DATA_DAYS_PERFORMANCE_DATA, MIN_DATA_DAYS_VALID_CLOSE
    if len(perf_data) < MIN_DATA_DAYS_PERFORMANCE_DATA:
        return 0.0, 0.0, 0.0

    valid_close = perf_data['Close'].dropna()
    if len(valid_close) < MIN_DATA_DAYS_VALID_CLOSE:
        return 0.0, 0.0, 0.0

    start_price = valid_close.iloc[0]
    end_price = valid_close.iloc[-1]

    if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
        return 0.0, 0.0, 0.0

    # FIXED: Allow negative returns with good risk-adjusted scores
    # Calculate risk-adjusted score for all returns, not just positive ones
    basic_return = ((end_price - start_price) / start_price) * 100

    # Calculate volatility
    daily_returns = valid_close.pct_change().dropna()
    from config import MIN_DATA_DAYS_DAILY_RETURNS
    if len(daily_returns) <= MIN_DATA_DAYS_DAILY_RETURNS:
        return 0.0, 0.0, 0.0

    # Calculate volatility using full performance window
    volatility = daily_returns.std() * 100

    # Risk-adjusted score - allow negative returns but penalize appropriately
    # Use absolute return in numerator but keep sign in score
    risk_adj_score = basic_return / (volatility**0.5 + 0.001)

    return risk_adj_score, basic_return, volatility


def check_momentum_confirmation(ticker_data: pd.DataFrame, current_date: datetime = None) -> int:
    """
    Check momentum confirmation across multiple timeframes.

    Returns:
        int: Number of timeframes with positive momentum
    """
    if not RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
        return 1  # No confirmation required

    momentum_confirmations = 0

    # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
    end_date = ticker_data.index.max()

    # If current_date is provided, try to use it (must convert to pandas Timestamp first)
    if current_date is not None:
        try:
            # Convert to pandas Timestamp (handles both datetime and Timestamp)
            current_ts = pd.Timestamp(current_date)

            # If data has timezone, ensure current_ts matches
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    # current_ts is naive, localize to data's timezone
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    # current_ts has timezone, convert to data's timezone
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)

            # Check data freshness - warn if data is older than configured limit
            data_age_days = (current_ts - end_date).total_seconds() / 86400
            if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                # Data is stale - raise error to be caught by caller
                raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")

            # Use the minimum of current_ts and data's max (can't use future data)
            end_date = min(current_ts, end_date)
        except ValueError:
            # Re-raise ValueError (stale data error) so caller can handle it
            raise
        except Exception as e:
            # For other errors (timezone conversion, etc), stick with data's max date
            print(f"Error checking momentum confirmation: {e}")
            pass

    # 3-month momentum check (using calendar days = 90 days)
    if RISK_ADJ_MOM_CONFIRM_SHORT:
        start_3m = end_date - timedelta(days=90)
        data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= end_date)]
        if len(data_3m) >= 30:  # Need ~30 trading days in 90 calendar days
            valid_close = data_3m['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_3m = ((end_price - start_price) / start_price) * 100
                    if return_3m > 0:
                        momentum_confirmations += 1

    # 6-month momentum check (using calendar days = 180 days)
    if RISK_ADJ_MOM_CONFIRM_MEDIUM:
        start_6m = end_date - timedelta(days=180)
        data_6m = ticker_data[(ticker_data.index >= start_6m) & (ticker_data.index <= end_date)]
        if len(data_6m) >= 60:  # Need ~60 trading days in 180 calendar days
            valid_close = data_6m['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_6m = ((end_price - start_price) / start_price) * 100
                    if return_6m > 0:
                        momentum_confirmations += 1

    # 1-year momentum check (using calendar days = 365 days)
    if RISK_ADJ_MOM_CONFIRM_LONG:
        start_1y = end_date - timedelta(days=365)
        data_1y = ticker_data[(ticker_data.index >= start_1y) & (ticker_data.index <= end_date)]
        if len(data_1y) >= 100:  # Need ~100 trading days in 365 calendar days
            valid_close = data_1y['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_1y = ((end_price - start_price) / start_price) * 100
                    if return_1y > 0:
                        momentum_confirmations += 1

    return momentum_confirmations


def check_volume_confirmation(ticker_data: pd.DataFrame) -> bool:
    """
    Check volume confirmation criteria.

    Returns:
        bool: True if volume confirmation passes, False otherwise
    """
    if not RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION:
        return True  # No volume confirmation required

    volume_data = ticker_data['Volume'].dropna()
    from config import MIN_DATA_DAYS_VOLUME_CONFIRM
    if len(volume_data) < MIN_DATA_DAYS_VOLUME_CONFIRM:
        return True  # Insufficient data, pass by default

    recent_volume = volume_data.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
    avg_volume = volume_data.head(len(volume_data) - RISK_ADJ_MOM_VOLUME_WINDOW).mean()

    if avg_volume <= 0:
        return True  # Invalid average, pass by default

    return recent_volume >= avg_volume * RISK_ADJ_MOM_VOLUME_MULTIPLIER


def calculate_volatility_adjusted_momentum(ticker_data, current_date=None, lookback_days=90, vol_window=20):
    """
    Calculate volatility-adjusted momentum score for a ticker.
    Uses calendar days (timedelta) for consistency with other calculations.

    Args:
        ticker_data: DataFrame with price data for a single ticker
        current_date: Current date for calculation (uses data max if None)
        lookback_days: Period for momentum calculation in calendar days (default 90)
        vol_window: Period for volatility calculation in calendar days (default 20)

    Returns:
        Volatility-adjusted momentum score
    """
    try:
        # Ensure index is DatetimeIndex in UTC
        if not isinstance(ticker_data.index, pd.DatetimeIndex):
            ticker_data = ticker_data.copy()
            ticker_data.index = pd.to_datetime(ticker_data.index, utc=True)
        elif ticker_data.index.tz is None:
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_localize('UTC')

        # Determine end date (ensure UTC)
        if current_date is not None:
            end_date = pd.Timestamp(current_date)
            if end_date.tz is None:
                end_date = end_date.tz_localize('UTC')
            else:
                end_date = end_date.tz_convert('UTC')
            ticker_data = ticker_data.loc[ticker_data.index <= end_date]
        else:
            end_date = ticker_data.index.max()

        if len(ticker_data) < 30:
            return 0.0

        # Calculate momentum return using calendar days
        start_date = end_date - timedelta(days=lookback_days)
        momentum_data = ticker_data[ticker_data.index >= start_date]

        if len(momentum_data) < 10:  # Need at least 10 trading days
            return 0.0

        close_prices = momentum_data['Close'].dropna()
        if len(close_prices) < 2:
            return 0.0

        momentum_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1)

        # Calculate volatility using calendar days
        vol_start_date = end_date - timedelta(days=vol_window)
        vol_data = ticker_data[ticker_data.index >= vol_start_date]
        daily_returns = vol_data['Close'].pct_change().dropna()

        if len(daily_returns) < 5:
            return 0.0

        volatility = daily_returns.std()

        # Avoid division by zero
        if volatility <= 0:
            return 0.0

        # Volatility-adjusted momentum (higher is better)
        vol_adjusted_score = momentum_return / (volatility ** 0.5)

        return vol_adjusted_score

    except Exception as e:
        print(f"Error calculating volatility-adjusted momentum: {e}")
        return 0.0


def select_volatility_adj_mom_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                                     current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Volatility-Adjusted Momentum stock selection logic.
    Uses calendar days for all calculations.
    Uses select_top_performers to avoid strict performance filtering issues.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Use select_top_performers to get candidates without strict filtering
    # This allows the strategy to work even in early backtest days
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "Vol-Adj Mom"
    )

    # No fallback - if filtering removes all tickers, return empty list

    current_top_performers = []

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Calculate volatility-adjusted momentum score using calendar days
            vol_adj_score = calculate_volatility_adjusted_momentum(
                ticker_data,
                current_date=current_date,
                lookback_days=VOLATILITY_ADJ_MOM_LOOKBACK,
                vol_window=VOLATILITY_ADJ_MOM_VOL_WINDOW
            )

            # Use same threshold as backtesting (accept any positive score)
            if vol_adj_score > 0:
                current_top_performers.append((ticker, vol_adj_score))

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Sort by volatility-adjusted score and get top N
    if current_top_performers:
        current_top_performers.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score in current_top_performers[:top_n]]

        print(f"   📊 Top {top_n} volatility-adjusted momentum: {selected_tickers}")
        for ticker, score in current_top_performers[:top_n]:
            print(f"      {ticker}: score={score:.3f}")

        return selected_tickers
    else:
        print(f"   ❌ No stocks passed volatility-adjusted momentum criteria")
        return []


def select_risk_adj_mom_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                               current_date: datetime = None, top_n: int = 20, lookback_days: int = 365,
                               strategy_name: str = "Risk-Adj Mom") -> List[str]:
    """
    Shared Risk-Adjusted Momentum stock selection logic.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, strategy_name
    )

    current_top_performers = []

    # Always use parallel processing
    from parallel_backtest import calculate_parallel_risk_adjusted_scores

    scores_data = calculate_parallel_risk_adjusted_scores(
        filtered_tickers,
        ticker_data_grouped,
        current_date,
        lookback_days=lookback_days
    )

    # Apply filters
    momentum_filtered = 0
    volume_filtered = 0
    data_issues = 0

    for ticker, score, return_pct, volatility_pct in scores_data:
        try:
            ticker_data = ticker_data_grouped[ticker]

            # Check momentum confirmation
            momentum_confirmations = check_momentum_confirmation(ticker_data, current_date)

            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                if momentum_confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    momentum_filtered += 1
                    continue

            # Check volume confirmation
            if not check_volume_confirmation(ticker_data):
                volume_filtered += 1
                continue

            if score > RISK_ADJ_MOM_MIN_SCORE:
                current_top_performers.append((ticker, score, return_pct, volatility_pct))

        except Exception as e:
            data_issues += 1
            print(f"Error processing {ticker}: {e}")
            continue

    analyzed_count = len(scores_data)

    # Sort by risk-adjusted score and get top N
    if current_top_performers:
        current_top_performers.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, ret, vol in current_top_performers[:top_n]]

        # Debug info
        confirm_parts = []
        if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
            confirm_parts.append("momentum")
        if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION:
            confirm_parts.append("volume")
        confirm_text = f" (with {' + '.join(confirm_parts)} confirmation)" if confirm_parts else ""

        print(f"   📊 Analysis: {analyzed_count} processed (PARALLEL), {momentum_filtered} momentum filtered, {volume_filtered} volume filtered, {data_issues} data issues")
        print(f"   🎯 Selected {len(selected_tickers)} stocks{confirm_text}:")
        for ticker, score, ret, vol in current_top_performers[:top_n]:
            print(f"      {ticker}: score={score:.2f}, return={ret:.1f}%, vol={vol:.1f}%")

        return selected_tickers
    else:
        print(f"   ❌ No stocks passed filtering criteria (analyzed: {analyzed_count})")
        print(f"   🔍 Debug: {momentum_filtered} momentum filtered, {volume_filtered} volume filtered, {data_issues} data issues, min_score={RISK_ADJ_MOM_MIN_SCORE}")
        return []


def select_mean_reversion_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                                current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Mean Reversion stock selection logic.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "Mean Reversion"
    )

    oversold_candidates = []

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in filtered_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            from config import MIN_DATA_DAYS_PERFORMANCE_DATA
            if len(ticker_data) < MIN_DATA_DAYS_PERFORMANCE_DATA:  # Need at least 50 days of data
                continue

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate recent performance (last 20 days) using date filtering
            recent_start = current_date_tz - timedelta(days=20)
            recent_data = ticker_data[(ticker_data.index >= recent_start) & (ticker_data.index <= current_date_tz)]
            if len(recent_data) >= 2:
                recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
            else:
                recent_return = 0.0

            # Calculate longer-term performance (last 100 days) using date filtering
            longer_start = current_date_tz - timedelta(days=100)
            longer_data = ticker_data[(ticker_data.index >= longer_start) & (ticker_data.index <= current_date_tz)]
            if len(longer_data) >= 2:
                longer_return = (longer_data['Close'].iloc[-1] / longer_data['Close'].iloc[0] - 1) * 100
            else:
                longer_return = 0.0

            # FIXED: Mean reversion signal - look for oversold conditions, not just recent decline
            # Use RSI-like logic: recent decline but not too severe, with longer-term strength
            if recent_return < -5 and recent_return > -25 and longer_return > -5:  # Moderate recent drop, not severe long-term decline
                # Better scoring: prioritize stocks with larger recent drops but solid long-term performance
                reversion_score = (-recent_return * 0.7) + (longer_return * 0.3)  # Weight recent drop more
                oversold_candidates.append((ticker, reversion_score, recent_return, longer_return))

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Sort by reversion score and get top N
    if oversold_candidates:
        oversold_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, recent_ret, longer_ret in oversold_candidates[:top_n]]

        print(f"   📊 Top {top_n} mean reversion candidates: {selected_tickers}")
        for ticker, score, recent_ret, longer_ret in oversold_candidates[:top_n]:
            print(f"      {ticker}: score={score:.2f}, recent={recent_ret:.1f}%, longer={longer_ret:.1f}%")

        return selected_tickers
    else:
        print(f"   ❌ No oversold candidates found")
        return []


def select_quality_momentum_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                                   current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Quality + Momentum stock selection logic.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "Quality+Mom"
    )

    quality_momentum_candidates = []

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in filtered_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            if len(ticker_data) < MIN_DATA_DAYS_GENERAL:  # Need enough data for quality assessment
                continue

            # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
            current_date_tz = ticker_data.index.max()

            # If current_date is provided, try to use it (must convert to pandas Timestamp first)
            if current_date is not None:
                try:
                    # Convert to pandas Timestamp (handles both datetime and Timestamp)
                    current_ts = pd.Timestamp(current_date)

                    # If data has timezone, ensure current_ts matches
                    if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                        if current_ts.tz is None:
                            # current_ts is naive, localize to data's timezone
                            current_ts = current_ts.tz_localize(ticker_data.index.tz)
                        else:
                            # current_ts has timezone, convert to data's timezone
                            current_ts = current_ts.tz_convert(ticker_data.index.tz)

                    # Check data freshness - warn if data is older than configured limit
                    data_age_days = (current_ts - current_date_tz).total_seconds() / 86400
                    if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                        # Data is stale - raise error
                        raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")

                    # Use the minimum of current_ts and data's max (can't use future data)
                    current_date_tz = min(current_ts, current_date_tz)
                except ValueError:
                    # Re-raise ValueError (stale data error) so it gets caught in outer exception handler
                    raise
                except Exception as e:
                    # For other errors (timezone conversion, etc), stick with data's max date
                    print(f"Error processing {ticker}: {e}")
                    pass

            # Momentum calculation (1-year) using date filtering
            momentum_start = current_date_tz - timedelta(days=365)  # 1 year for better performance measurement
            momentum_data = ticker_data[(ticker_data.index >= momentum_start) & (ticker_data.index <= current_date_tz)]

            if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                print(f"   🔍 DEBUG: {ticker} data range: {ticker_data.index.min()} to {ticker_data.index.max()}")
                print(f"   🔍 DEBUG: {ticker} momentum range: {momentum_start} to {current_date_tz}")
                print(f"   🔍 DEBUG: {ticker} momentum_data points: {len(momentum_data)}")
            if len(momentum_data) >= 2:
                # Drop NaN values from Close prices
                valid_prices = momentum_data['Close'].dropna()
                if len(valid_prices) < 2:
                    momentum_return = 0.0
                else:
                    start_price = valid_prices.iloc[0]
                    end_price = valid_prices.iloc[-1]
                    if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                        momentum_calc = (end_price / start_price - 1) * 100
                        print(f"   🔍 DEBUG: {ticker} start_price={start_price}, end_price={end_price}")
                        print(f"   🔍 DEBUG: {ticker} momentum={momentum_calc:.1f}%")
                    if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
                        momentum_return = 0.0
                    else:
                        momentum_return = (end_price / start_price - 1) * 100
            else:
                momentum_return = 0.0

            # Check for NaN values
            if pd.isna(momentum_return):
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum=nan% (NaN value, filtered)")
                continue

            # Only include stocks with positive momentum
            if momentum_return <= 0:
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum={momentum_return:.1f}% (<=0, filtered)")
                continue

            # Quality indicators (simplified)
            # 1. Price stability (lower volatility is better)
            daily_returns = ticker_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100

            # 2. Trend consistency (positive recent performance) using date filtering
            short_start = current_date_tz - timedelta(days=30)
            short_data = ticker_data[(ticker_data.index >= short_start) & (ticker_data.index <= current_date_tz)]
            if len(short_data) >= 2:
                short_trend = (short_data['Close'].iloc[-1] / short_data['Close'].iloc[0] - 1) * 100
            else:
                short_trend = 0.0

            # Quality score: momentum with stability bonus
            stability_bonus = max(0, 50 - volatility) / 50  # Higher bonus for lower volatility
            trend_bonus = max(0, short_trend) / 100  # Bonus for positive short trend

            quality_score = momentum_return * (1 + stability_bonus + trend_bonus)

            if momentum_return > 0:  # Only consider positive momentum (lowered from 5%)
                quality_momentum_candidates.append((ticker, quality_score, momentum_return, volatility))
            else:
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum={momentum_return:.1f}% (<=0%, filtered)")

        except Exception as e:
            if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                print(f"   🔍 DEBUG: {ticker} exception: {type(e).__name__}: {e}")
            continue

    # Sort by quality score and get top N
    if quality_momentum_candidates:
        quality_momentum_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, mom, vol in quality_momentum_candidates[:top_n]]

        print(f"   📊 Top {top_n} quality + momentum: {selected_tickers}")
        for ticker, score, mom, vol in quality_momentum_candidates[:top_n]:
            print(f"      {ticker}: score={score:.1f}, momentum={mom:.1f}%, vol={vol:.1f}%")

        return selected_tickers
    else:
        print(f"   ❌ No quality + momentum candidates found")
        return []


def select_sector_rotation_etfs(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                                current_date: datetime, top_n: int = 5) -> List[str]:
    """
    Select top performing sector ETFs based on momentum.
    PROPOSAL 2: Sector Rotation Strategy
    """
    from config import SECTOR_ROTATION_MOMENTUM_WINDOW, SECTOR_ROTATION_MIN_MOMENTUM

    print(f"   🔍 Sector Rotation Debug: Looking for ETFs in {len(all_tickers)} available tickers")
    print(f"   🔍 Momentum window: {SECTOR_ROTATION_MOMENTUM_WINDOW} days, Min threshold: {SECTOR_ROTATION_MIN_MOMENTUM}%")

    # Check if sector ETFs are in the available tickers
    sector_etfs = [
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB',
        'GDX', 'USO', 'TLT'
    ]

    available_sector_etfs = [etf for etf in sector_etfs if etf in all_tickers]
    print(f"   🔍 Available sector ETFs: {available_sector_etfs}")

    if not available_sector_etfs:
        print(f"   ❌ No sector ETFs found in ticker list! Strategy cannot execute.")
        return []

    sector_performance = []
    found_etfs = []

    for etf in available_sector_etfs:
        if etf in all_tickers and etf in ticker_data_grouped:
            found_etfs.append(etf)
            try:
                etf_data = ticker_data_grouped[etf]

                # Convert current_date to pandas Timestamp with timezone
                current_date_tz = pd.Timestamp(current_date)
                if hasattr(etf_data.index, 'tz') and etf_data.index.tz is not None:
                    if current_date_tz.tz is None:
                        current_date_tz = current_date_tz.tz_localize(etf_data.index.tz)
                    else:
                        current_date_tz = current_date_tz.tz_convert(etf_data.index.tz)

                # Filter data for momentum calculation
                start_date = current_date_tz - timedelta(days=SECTOR_ROTATION_MOMENTUM_WINDOW + 30)

                # Use index-based filtering since date is the index
                etf_filtered = etf_data[(etf_data.index >= start_date) & (etf_data.index <= current_date_tz)]

                print(f"   🔍 {etf}: {len(etf_filtered)} data points available")

                if len(etf_filtered) >= 20:  # Need sufficient data
                    start_price = etf_filtered['Close'].iloc[0]
                    end_price = etf_filtered['Close'].iloc[-1]
                    momentum_pct = ((end_price - start_price) / start_price) * 100

                    print(f"   🔍 {etf}: momentum = {momentum_pct:.1f}% (threshold: {SECTOR_ROTATION_MIN_MOMENTUM}%)")

                    if momentum_pct >= SECTOR_ROTATION_MIN_MOMENTUM:
                        sector_performance.append((etf, momentum_pct))
                else:
                    print(f"   🔍 {etf}: insufficient data (need 20, have {len(etf_filtered)})")

            except Exception as e:
                print(f"   🔍 {etf}: error calculating momentum - {e}")
                continue

    print(f"   🔍 Found {len(found_etfs)} sector ETFs: {found_etfs}")
    print(f"   🔍 {len(sector_performance)} ETFs met momentum threshold")

    # Sort by momentum and get top N
    if sector_performance:
        sector_performance.sort(key=lambda x: x[1], reverse=True)
        selected_etfs = [etf for etf, momentum in sector_performance[:top_n]]

        print(f"   🏢 Top {len(selected_etfs)} sector ETFs by {SECTOR_ROTATION_MOMENTUM_WINDOW}-day momentum:")
        for etf, momentum in sector_performance[:top_n]:
            print(f"      {etf}: {momentum:+.1f}%")

        print(f"   ✅ Sector Rotation selected {len(selected_etfs)} ETFs: {selected_etfs}")
        return selected_etfs
    else:
        print(f"   ❌ No sector ETFs met minimum momentum threshold ({SECTOR_ROTATION_MIN_MOMENTUM}%)")
        print(f"   ❌ Total ETFs analyzed: {len(sector_performance)}")
        return []


def select_3m_1y_ratio_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                              current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    3M/1Y Ratio Strategy: Select tickers with highest ratio of 3-month performance to 1-year performance.

    This strategy identifies stocks that are showing strong recent momentum (3M) relative to
    their longer-term performance (1Y), which can indicate accelerating momentum or
    reversal from longer-term trends.

    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data (with date as index)
        current_date: Current date for analysis (None for last available)
        top_n: Number of stocks to select

    Returns:
        List[str]: Selected ticker symbols
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "3M/1Y Ratio"
    )

    ratio_candidates = []

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []

    print(f"   📊 3M/1Y Ratio Strategy analyzing {len(filtered_tickers)} tickers")

    analysis_count = 0
    filtered_3m_negative = 0
    filtered_1y_negative = 0
    filtered_ratio_negative = 0
    filtered_ratio_too_high = 0
    data_insufficient = 0

    for ticker in filtered_tickers:
        try:
            analysis_count += 1

            if ticker not in ticker_data_grouped:
                data_insufficient += 1
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Need at least 1 year of data
            if len(ticker_data) < MIN_DATA_DAYS_1Y:
                data_insufficient += 1
                continue

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate 3-month performance
            three_month_start = current_date_tz - timedelta(days=90)
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) &
                                         (ticker_data.index <= current_date_tz)]

            from config import MIN_DATA_DAYS_THREE_MONTH_POINTS
            if len(three_month_data) < MIN_DATA_DAYS_THREE_MONTH_POINTS:  # Need at least 10 data points
                data_insufficient += 1
                continue

            three_month_valid = three_month_data['Close'].dropna()
            if len(three_month_valid) < 2:
                data_insufficient += 1
                continue

            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]

            if three_month_start_price <= 0 or pd.isna(three_month_start_price) or pd.isna(three_month_end_price):
                data_insufficient += 1
                continue

            three_month_performance = ((three_month_end_price - three_month_start_price) /
                                     three_month_start_price) * 100

            # Calculate 1-year performance
            one_year_start = current_date_tz - timedelta(days=365)
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) &
                                      (ticker_data.index <= current_date_tz)]

            from config import MIN_DATA_DAYS_ONE_YEAR_POINTS
            if len(one_year_data) < MIN_DATA_DAYS_ONE_YEAR_POINTS:  # Need at least 50 data points
                data_insufficient += 1
                continue

            one_year_valid = one_year_data['Close'].dropna()
            if len(one_year_valid) < 2:
                data_insufficient += 1
                continue

            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]

            if one_year_start_price <= 0 or pd.isna(one_year_start_price) or pd.isna(one_year_end_price):
                data_insufficient += 1
                continue

            one_year_performance = ((one_year_end_price - one_year_start_price) /
                                  one_year_start_price) * 100

            # Calculate ratio (handle division by zero and negative 1Y performance)
            if abs(one_year_performance) < 0.1:  # Avoid division by very small numbers
                continue

            ratio = three_month_performance / one_year_performance

            # Better approach: Annualized 3M performance vs 1Y performance
            # This compares like-for-like annualized rates
            annualized_3m = three_month_performance * (365/90)  # Annualize 3M performance
            momentum_acceleration = annualized_3m - one_year_performance

            # Debug first few stocks
            if analysis_count <= 5:
                print(f"   🔍 DEBUG {ticker}: 3M={three_month_performance:+.1f}%, annualized_3M={annualized_3m:+.1f}%, 1Y={one_year_performance:+.1f}%, acceleration={momentum_acceleration:+.1f}%")

            # Track filtering reasons
            if three_month_performance <= 0:
                filtered_3m_negative += 1
                continue
            if one_year_performance <= 0:
                filtered_1y_negative += 1
                continue

            # Better approach: Strong base + annualized acceleration
            # Require minimum 1Y performance AND positive annualized acceleration - RELAXED
            if (one_year_performance > 5 and  # Reduced: Minimum 5% 1Y performance vs 10%
                momentum_acceleration > 5):  # Reduced: At least 5% annualized acceleration vs 10%
                ratio_candidates.append((ticker, momentum_acceleration, annualized_3m, one_year_performance))
            else:
                # Track why filtered
                if momentum_acceleration <= 5 or one_year_performance <= 5:
                    filtered_ratio_negative += 1  # Reuse this counter for low acceleration/weak 1Y

        except Exception as e:
            data_insufficient += 1
            continue

    print(f"   📊 Analysis Summary:")
    print(f"      Total analyzed: {analysis_count}")
    print(f"      Data insufficient: {data_insufficient}")
    print(f"      3M negative: {filtered_3m_negative}")
    print(f"      1Y negative: {filtered_1y_negative}")
    print(f"      Weak 1Y/Low acceleration: {filtered_ratio_negative}")
    print(f"      Valid candidates: {len(ratio_candidates)}")

    # Sort by momentum acceleration (highest first) and get top N
    if ratio_candidates:
        ratio_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, acceleration, annualized_3m, y1_perf in ratio_candidates[:top_n]]

        print(f"   📊 Top {top_n} Annualized Acceleration candidates:")
        for ticker, acceleration, annualized_3m, y1_perf in ratio_candidates[:top_n]:
            print(f"      {ticker}: acceleration={acceleration:+.1f}%, annualized_3M={annualized_3m:+.1f}%, 1Y={y1_perf:+.1f}%")

        print(f"   ✅ Annualized Acceleration selected {len(selected_tickers)} tickers: {selected_tickers}")
        return selected_tickers
    else:
        print(f"   ❌ No Annualized Acceleration candidates found")
        print(f"   ❌ Analyzed {len(filtered_tickers)} tickers, found {len(ratio_candidates)} valid candidates")
        return []


def select_turnaround_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Turnaround Strategy: Select stocks with low 3Y performance but high 1Y performance.
    This identifies stocks that may be emerging from a long decline with strong recent momentum.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "Turnaround"
    )

    turnaround_candidates = []
    data_insufficient = 0
    filtered_3y_positive = 0
    filtered_1y_low = 0

    print(f"   🔍 Turnaround: Analyzing {len(filtered_tickers)} tickers")

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in filtered_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate 1-year and 3-year start dates
            one_year_start = current_date_tz - timedelta(days=365)
            three_year_start = current_date_tz - timedelta(days=1095)  # 3 years

            # Get 1-year data
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) & (ticker_data.index <= current_date_tz)]
            one_year_valid = one_year_data['Close'].dropna()

            # Get 3-year data
            three_year_data = ticker_data[(ticker_data.index >= three_year_start) & (ticker_data.index <= current_date_tz)]
            three_year_valid = three_year_data['Close'].dropna()

            # Check data sufficiency (reduced requirements for available data)
            if len(one_year_valid) < 100 or len(three_year_valid) < 100:  # Reduced requirements
                data_insufficient += 1
                continue

            # Calculate performances
            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]
            three_year_start_price = three_year_valid.iloc[0]
            three_year_end_price = three_year_valid.iloc[-1]

            if any(price <= 0 or pd.isna(price) for price in [one_year_start_price, one_year_end_price, three_year_start_price, three_year_end_price]):
                data_insufficient += 1
                continue

            one_year_performance = ((one_year_end_price - one_year_start_price) / one_year_start_price) * 100
            three_year_performance = ((three_year_end_price - three_year_start_price) / three_year_start_price) * 100

            # Debug first few stocks
            if len(turnaround_candidates) < 5:
                print(f"   🔍 DEBUG {ticker}: 3Y={three_year_performance:+.1f}%, 1Y={one_year_performance:+.1f}%")

            # Turnaround criteria (relaxed for more candidates):
            # 1. Poor 3Y performance (negative or low - below 30%)
            # 2. Strong 1Y performance (positive - minimum 10%)
            # 3. Recovery ratio: 1Y performance should be better than 3Y annualized
            if three_year_performance > 30:  # 3Y should be below 30% (relaxed from 0%)
                filtered_3y_positive += 1
                continue

            if one_year_performance < 10:  # Need positive 1Y performance (relaxed from 20%)
                filtered_1y_low += 1
                continue

            # Calculate recovery score: 1Y performance - (3Y performance / 3)
            # This compares 1Y performance to average annual performance over 3 years
            three_year_annual = three_year_performance / 3
            recovery_score = one_year_performance - three_year_annual

            # Add to candidates if recovery is positive (relaxed from 30%)
            if recovery_score > 10:  # 1Y should be at least 10% better than 3Y annual average
                turnaround_candidates.append((ticker, recovery_score, one_year_performance, three_year_performance))

        except Exception as e:
            data_insufficient += 1
            continue

    # Sort by recovery score (highest first)
    turnaround_candidates.sort(key=lambda x: x[1], reverse=True)

    if turnaround_candidates:
        print(f"   📊 Turnaround: Selected {len(turnaround_candidates)} candidates")
        print(f"   📊 Filter breakdown: {filtered_3y_positive} filtered (3Y positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        print(f"   🎯 Selected {min(len(turnaround_candidates), top_n)} turnaround stocks:")
        for ticker, score, one_year, three_year in turnaround_candidates[:top_n]:
            print(f"      {ticker}: recovery={score:.1f}%, 1Y={one_year:+.1f}%, 3Y={three_year:+.1f}%")

        return [ticker for ticker, _, _, _ in turnaround_candidates[:top_n]]
    else:
        print(f"   ❌ No turnaround candidates found")
        print(f"   📊 Filter breakdown: {filtered_3y_positive} filtered (3Y positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        return []


def select_1m_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    1M/3M Ratio Strategy: Select stocks with strong 1M performance relative to 3M performance.
    This identifies stocks with short-term momentum acceleration - recent outperformance.

    Positive 1M/3M ratio means 1M is outpacing 3M (accelerating momentum).
    Useful for catching stocks that are breaking out or gaining momentum recently.
    """
    from config import INVERSE_ETFS

    # Exclude inverse ETFs
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "1M/3M Ratio"
    )

    ratio_candidates = []
    data_insufficient = 0
    filtered_1m_negative = 0
    filtered_3m_negative = 0

    print(f"   🔍 1M/3M Ratio: Analyzing {len(filtered_tickers)} tickers")

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in filtered_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is timezone-aware
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate 1-month and 3-month start dates
            one_month_start = current_date_tz - timedelta(days=30)
            three_month_start = current_date_tz - timedelta(days=90)

            # Get 1-month data
            one_month_data = ticker_data[(ticker_data.index >= one_month_start) &
                                        (ticker_data.index <= current_date_tz)]

            # Get 3-month data
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) &
                                          (ticker_data.index <= current_date_tz)]

            # Check data sufficiency
            if len(one_month_data) < MIN_DATA_DAYS_1M or len(three_month_data) < MIN_DATA_DAYS_3M:
                data_insufficient += 1
                continue

            one_month_valid = one_month_data['Close'].dropna()
            three_month_valid = three_month_data['Close'].dropna()

            if len(one_month_valid) < 2 or len(three_month_valid) < 2:
                data_insufficient += 1
                continue

            # Calculate performances
            one_month_start_price = one_month_valid.iloc[0]
            one_month_end_price = one_month_valid.iloc[-1]
            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]

            if any(price <= 0 or pd.isna(price) for price in [one_month_start_price, one_month_end_price,
                                                               three_month_start_price, three_month_end_price]):
                data_insufficient += 1
                continue

            one_month_performance = ((one_month_end_price - one_month_start_price) /
                                    one_month_start_price) * 100
            three_month_performance = ((three_month_end_price - three_month_start_price) /
                                      three_month_start_price) * 100

            # Annualize 1M for fair comparison with 3M
            annualized_1m = one_month_performance * 3  # 1M * 3 = 3M equivalent

            # Calculate acceleration: how much 1M is outpacing 3M
            acceleration = annualized_1m - three_month_performance

            # Debug first few stocks
            if len(ratio_candidates) < 5:
                print(f"   🔍 DEBUG {ticker}: 1M={one_month_performance:+.1f}%, 3M={three_month_performance:+.1f}%, accel={acceleration:+.1f}%")

            # Selection criteria:
            # 1. Positive 1M performance (recent momentum)
            # 2. Positive acceleration (1M outpacing 3M)
            if one_month_performance <= 0:
                filtered_1m_negative += 1
                continue

            if three_month_performance < -20:  # Avoid stocks in freefall
                filtered_3m_negative += 1
                continue

            # Add to candidates if showing acceleration
            if acceleration > 0:  # 1M is outpacing 3M
                ratio_candidates.append((ticker, acceleration, one_month_performance, three_month_performance))

        except Exception as e:
            data_insufficient += 1
            continue

    # Sort by acceleration (highest first)
    ratio_candidates.sort(key=lambda x: x[1], reverse=True)

    if ratio_candidates:
        print(f"   📊 1M/3M Ratio: Selected {len(ratio_candidates)} acceleration candidates")
        print(f"   📊 Filter breakdown: {filtered_1m_negative} filtered (1M negative), {filtered_3m_negative} filtered (3M crash), {data_insufficient} insufficient data")
        print(f"   🎯 Selected {min(len(ratio_candidates), top_n)} accelerating stocks:")
        for ticker, accel, one_month, three_month in ratio_candidates[:top_n]:
            print(f"      {ticker}: accel={accel:+.1f}%, 1M={one_month:+.1f}%, 3M={three_month:+.1f}%")

        return [ticker for ticker, _, _, _ in ratio_candidates[:top_n]]
    else:
        print(f"   ❌ No acceleration candidates found")
        print(f"   📊 Filter breakdown: {filtered_1m_negative} filtered (1M negative), {filtered_3m_negative} filtered (3M crash), {data_insufficient} insufficient data")
        return []


def select_1y_performers_ranked_by_1m3m_ratio(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Strategy: Select top 1Y performers, then rank by 1M/3M ratio (momentum acceleration).

    Step 1: Get top 1Y performers (larger pool, e.g., top 50)
    Step 2: Calculate 1M/3M ratio for each
    Step 3: Rank by 1M/3M ratio (acceleration) and return top N

    This combines long-term strength (1Y performance) with short-term momentum (1M/3M acceleration).
    """
    from parallel_backtest import calculate_parallel_performance
    from config import INVERSE_ETFS

    # Exclude inverse ETFs
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Step 1: Get top 1Y performers (larger pool - 3x the final selection)
    pool_size = top_n * 3
    performances = calculate_parallel_performance(
        tickers_to_use, ticker_data_grouped, current_date, period_days=365
    )

    if not performances:
        print(f"   ❌ 1Y+1M3M Ratio: No 1Y performance data available")
        return []

    # Sort by 1Y performance and get top pool
    performances.sort(key=lambda x: x[1], reverse=True)
    top_1y_pool = [ticker for ticker, _ in performances[:pool_size]]

    print(f"   📊 1Y+1M3M Ratio: Selected top {len(top_1y_pool)} 1Y performers for ranking")

    # Step 2: Calculate 1M/3M ratio for each ticker in the pool
    ratio_candidates = []
    data_insufficient = 0

    # Ensure current_date is timezone-aware
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in top_1y_pool
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    for ticker in top_1y_pool:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate 1-month and 3-month start dates
            one_month_start = current_date_tz - timedelta(days=30)
            three_month_start = current_date_tz - timedelta(days=90)

            # Get 1-month data
            one_month_data = ticker_data[(ticker_data.index >= one_month_start) &
                                        (ticker_data.index <= current_date_tz)]

            # Get 3-month data
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) &
                                          (ticker_data.index <= current_date_tz)]

            # Check data sufficiency
            if len(one_month_data) < MIN_DATA_DAYS_1M or len(three_month_data) < MIN_DATA_DAYS_3M:
                data_insufficient += 1
                continue

            one_month_valid = one_month_data['Close'].dropna()
            three_month_valid = three_month_data['Close'].dropna()

            if len(one_month_valid) < 2 or len(three_month_valid) < 2:
                data_insufficient += 1
                continue

            # Calculate performances
            one_month_start_price = one_month_valid.iloc[0]
            one_month_end_price = one_month_valid.iloc[-1]
            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]

            if any(price <= 0 or pd.isna(price) for price in [one_month_start_price, one_month_end_price,
                                                               three_month_start_price, three_month_end_price]):
                data_insufficient += 1
                continue

            one_month_performance = ((one_month_end_price - one_month_start_price) /
                                    one_month_start_price) * 100
            three_month_performance = ((three_month_end_price - three_month_start_price) /
                                      three_month_start_price) * 100

            # Annualize 1M for fair comparison with 3M
            annualized_1m = one_month_performance * 3  # 1M * 3 = 3M equivalent

            # Calculate acceleration: how much 1M is outpacing 3M
            acceleration = annualized_1m - three_month_performance

            # Get 1Y performance for this ticker
            perf_1y = next((p for t, p in performances if t == ticker), 0)

            ratio_candidates.append((ticker, acceleration, one_month_performance, three_month_performance, perf_1y))

        except Exception as e:
            data_insufficient += 1
            continue

    # Step 3: Sort by 1M/3M acceleration (highest first)
    ratio_candidates.sort(key=lambda x: x[1], reverse=True)

    if ratio_candidates:
        print(f"   📊 1Y+1M3M Ratio: Ranked {len(ratio_candidates)} stocks by acceleration")
        print(f"   🎯 Top {min(len(ratio_candidates), top_n)} stocks (1Y performers ranked by 1M/3M):")
        for ticker, accel, one_month, three_month, perf_1y in ratio_candidates[:top_n]:
            print(f"      {ticker}: 1Y={perf_1y:+.1f}%, accel={accel:+.1f}%, 1M={one_month:+.1f}%, 3M={three_month:+.1f}%")

        return [ticker for ticker, _, _, _, _ in ratio_candidates[:top_n]]
    else:
        print(f"   ❌ 1Y+1M3M Ratio: No candidates found ({data_insufficient} insufficient data)")
        return []


def select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    1Y/3M Ratio Strategy: Select stocks with strong 1Y performance but weak 3M performance.
    This identifies stocks in long-term uptrends that recently pulled back (buy on dip).
    """
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "1Y/3M Ratio"
    )

    ratio_candidates = []
    data_insufficient = 0
    filtered_3m_positive = 0
    filtered_1y_low = 0

    print(f"   🔍 1Y/3M Ratio: Analyzing {len(filtered_tickers)} tickers")

    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in filtered_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)

    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]

            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)

            # Calculate 3-month and 1-year start dates
            three_month_start = current_date_tz - timedelta(days=90)
            one_year_start = current_date_tz - timedelta(days=365)

            # Get 3-month data
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) &
                                         (ticker_data.index <= current_date_tz)]

            # Get 1-year data
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) &
                                      (ticker_data.index <= current_date_tz)]

            # Check data sufficiency
            if len(three_month_data) < MIN_DATA_DAYS_3M or len(one_year_data) < MIN_DATA_DAYS_1Y:
                data_insufficient += 1
                continue

            three_month_valid = three_month_data['Close'].dropna()
            one_year_valid = one_year_data['Close'].dropna()

            if len(three_month_valid) < 2 or len(one_year_valid) < 2:
                data_insufficient += 1
                continue

            # Calculate performances
            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]
            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]

            if any(price <= 0 or pd.isna(price) for price in [three_month_start_price, three_month_end_price, one_year_start_price, one_year_end_price]):
                data_insufficient += 1
                continue

            three_month_performance = ((three_month_end_price - three_month_start_price) /
                                      three_month_start_price) * 100
            one_year_performance = ((one_year_end_price - one_year_start_price) /
                                  one_year_start_price) * 100

            # Debug first few stocks
            if len(ratio_candidates) < 5:
                print(f"   🔍 DEBUG {ticker}: 3M={three_month_performance:+.1f}%, 1Y={one_year_performance:+.1f}%")

            # Buy on dip criteria - RELAXED:
            # 1. Strong 1Y performance (positive and significant)
            # 2. Weak or negative 3M performance (pullback)
            # 3. Dip ratio: 1Y performance should be much better than 3M
            if three_month_performance > 30:  # Relaxed: 3M should be weak or negative (30% vs 20%)
                filtered_3m_positive += 1
                continue

            if one_year_performance < 10:  # Relaxed: Need strong 1Y performance (10% vs 15%)
                filtered_1y_low += 1
                continue

            # Calculate dip opportunity score: 1Y performance - 3M performance
            # Higher score means stronger 1Y trend with bigger recent pullback
            dip_score = one_year_performance - three_month_performance

            # Add to candidates if dip opportunity is strong - RELAXED
            if dip_score > 15:  # Relaxed: 1Y should be at least 15% better than 3M (15% vs 20%)
                ratio_candidates.append((ticker, dip_score, one_year_performance, three_month_performance))

        except Exception as e:
            data_insufficient += 1
            continue

    # Sort by dip score (highest first)
    ratio_candidates.sort(key=lambda x: x[1], reverse=True)

    if ratio_candidates:
        print(f"   📊 1Y/3M Ratio: Selected {len(ratio_candidates)} dip candidates")
        print(f"   📊 Filter breakdown: {filtered_3m_positive} filtered (3M positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        print(f"   🎯 Selected {min(len(ratio_candidates), top_n)} buy-on-dip stocks:")
        for ticker, score, one_year, three_month in ratio_candidates[:top_n]:
            print(f"      {ticker}: dip={score:.1f}%, 1Y={one_year:+.1f}%, 3M={three_month:+.1f}%")

        return [ticker for ticker, _, _, _ in ratio_candidates[:top_n]]
    else:
        print(f"   ❌ No buy-on-dip candidates found")
        print(f"   📊 Filter breakdown: {filtered_3m_positive} filtered (3M positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        return []


def select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Momentum-Volatility Strategy: Combines strong momentum with controlled volatility.
    """
    if current_date is None:
        current_date = datetime.now()

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Mom-Vol Hybrid"
    )

    candidates = []

    # Use filtered tickers for analysis
    tickers_to_scan = set(filtered_tickers)
    if isinstance(ticker_data_grouped, dict):
        tickers_to_scan.update(ticker_data_grouped.keys())

    for ticker in tickers_to_scan:
        try:
            # Get ticker data - handle both dict and GroupBy objects
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None

            if ticker_data is None or len(ticker_data) == 0:
                continue

            # ✅ FIX: Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data

            # Use dropna'd Close series for all calculations
            close_prices = ticker_data_filtered['Close'].dropna()
            n_prices = len(close_prices)

            if n_prices < 60:
                continue

            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue

            # Calculate 3M performance (approx 63 trading days)
            lookback_3m = min(63, n_prices - 1)
            if lookback_3m < 20:
                continue
            price_3m_ago = close_prices.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue
            performance_3m = (latest_price - price_3m_ago) / price_3m_ago
            annualized_3m = (1 + performance_3m) ** (252 / lookback_3m) - 1

            # Calculate 1Y performance (approx 252 trading days)
            lookback_1y = min(252, n_prices - 1)
            if lookback_1y < 60:
                continue
            price_1y_ago = close_prices.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago

            # Calculate volatility (using daily returns)
            daily_returns = close_prices.pct_change().dropna()
            if len(daily_returns) < 30:
                continue
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility

            # Calculate average volume
            avg_volume = ticker_data['Volume'].dropna().mean() if 'Volume' in ticker_data.columns else 100000

            # Apply filters - RELAXED criteria
            if (annualized_3m > 0.0 and  # Any positive 3M momentum
                performance_1y > -0.3 and  # Allow up to 30% loss in 1Y
                volatility < 3.0 and  # Volatility < 300%
                avg_volume > 10000):  # Low volume threshold

                # Calculate composite score
                momentum_score = annualized_3m * 0.6 + max(performance_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)

                candidates.append({
                    'ticker': ticker,
                    'score': composite_score,
                    'annualized_3m': annualized_3m,
                    'performance_1y': performance_1y,
                    'volatility': volatility
                })

        except Exception as e:
            continue

    # Sort by composite score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Debug output
    if candidates:
        print(f"   🎯 Momentum-Volatility Hybrid: Found {len(candidates)} candidates")
        if len(candidates) >= 3:
            print(f"   Top 3: {candidates[0]['ticker']} ({candidates[0]['score']:.3f}), "
                  f"{candidates[1]['ticker']} ({candidates[1]['score']:.3f}), "
                  f"{candidates[2]['ticker']} ({candidates[2]['score']:.3f})")
    else:
        print(f"   ⚠️ Momentum-Volatility Hybrid: No candidates found (checked {len(all_tickers)} tickers)")

    return [c['ticker'] for c in candidates[:top_n]]


def select_momentum_volatility_hybrid_6m_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Momentum-Volatility Strategy (6M variant): Combines strong 6-month momentum with controlled volatility.
    Same as 3M variant but uses 6-month lookback as primary momentum signal.
    """
    if current_date is None:
        current_date = datetime.now()

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Mom-Vol Hybrid 6M"
    )

    candidates = []

    # Use filtered tickers for analysis
    tickers_to_scan = set(filtered_tickers)
    if isinstance(ticker_data_grouped, dict):
        tickers_to_scan.update(ticker_data_grouped.keys())

    for ticker in tickers_to_scan:
        try:
            # Get ticker data - handle both dict and GroupBy objects
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None

            if ticker_data is None or len(ticker_data) == 0:
                continue

            # ✅ FIX: Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data

            # Use dropna'd Close series for all calculations
            close_prices = ticker_data_filtered['Close'].dropna()
            n_prices = len(close_prices)

            if n_prices < 60:
                continue

            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue

            # Calculate 6M performance (approx 126 trading days)
            lookback_6m = min(126, n_prices - 1)
            if lookback_6m < 40:
                continue
            price_6m_ago = close_prices.iloc[-lookback_6m]
            if price_6m_ago <= 0:
                continue
            performance_6m = (latest_price - price_6m_ago) / price_6m_ago
            annualized_6m = (1 + performance_6m) ** (252 / lookback_6m) - 1

            # Calculate 1Y performance (approx 252 trading days)
            lookback_1y = min(252, n_prices - 1)
            if lookback_1y < 60:
                continue
            price_1y_ago = close_prices.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago

            # Calculate volatility (using daily returns)
            daily_returns = close_prices.pct_change().dropna()
            if len(daily_returns) < 30:
                if ticker in ['SNDK', 'MTS.MC']:
                    print(f"   ⚠️ DEBUG {ticker}: SKIPPED - daily_returns < 30 ({len(daily_returns)})")
                continue
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility

            # Calculate average volume
            avg_volume = ticker_data['Volume'].dropna().mean() if 'Volume' in ticker_data.columns else 100000

            # Debug for specific tickers
            if ticker in ['SNDK', 'MTS.MC']:
                print(f"   🔍 DEBUG {ticker}: price={latest_price:.2f}, 6m_ann={annualized_6m:.4f}, 1y={performance_1y:.4f}, vol={volatility:.4f}, avg_vol={avg_volume:.0f}")
                print(f"   🔍 DEBUG {ticker}: data_len={len(ticker_data)}, close_len={len(ticker_data['Close'].dropna())}, cols={list(ticker_data.columns[:10])}")

            # Apply filters - RELAXED criteria
            if (annualized_6m > 0.0 and  # Any positive 6M momentum
                performance_1y > -0.3 and  # Allow up to 30% loss in 1Y
                volatility < 3.0 and  # Volatility < 300%
                avg_volume > 10000):  # Low volume threshold

                # Calculate composite score (6M weighted higher)
                momentum_score = annualized_6m * 0.6 + max(performance_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)

                if ticker in ['SNDK', 'MTS.MC']:
                    print(f"   ✅ DEBUG {ticker}: PASSED all filters, score={composite_score:.4f}")

                candidates.append({
                    'ticker': ticker,
                    'score': composite_score,
                    'annualized_6m': annualized_6m,
                    'performance_1y': performance_1y,
                    'volatility': volatility
                })
            else:
                if ticker in ['SNDK', 'MTS.MC']:
                    fails = []
                    if annualized_6m <= 0.0: fails.append(f"6m_ann={annualized_6m:.4f}<=0")
                    if performance_1y <= -0.3: fails.append(f"1y={performance_1y:.4f}<=-0.3")
                    if volatility >= 3.0: fails.append(f"vol={volatility:.4f}>=3.0")
                    if avg_volume <= 10000: fails.append(f"vol={avg_volume:.0f}<=10000")
                    print(f"   ❌ DEBUG {ticker}: FAILED filters: {', '.join(fails)}")

        except Exception as e:
            if ticker in ['SNDK', 'MTS.MC']:
                import traceback
                print(f"   ⚠️ DEBUG {ticker}: EXCEPTION in scoring: {e}")
                traceback.print_exc()
            continue

    # Sort by composite score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Debug: Check if specific tickers made it
    candidate_tickers = {c['ticker'] for c in candidates}
    for debug_ticker in ['SNDK', 'MTS.MC']:
        if debug_ticker in candidate_tickers:
            c = next(c for c in candidates if c['ticker'] == debug_ticker)
            print(f"   🔍 DEBUG {debug_ticker}: IN candidates, score={c['score']:.4f}")
        else:
            print(f"   🔍 DEBUG {debug_ticker}: NOT in candidates")

    # Debug output
    if candidates:
        print(f"   🎯 Mom-Vol Hybrid 6M: Found {len(candidates)} candidates")
        if len(candidates) >= 3:
            print(f"   Top 3: {candidates[0]['ticker']} ({candidates[0]['score']:.3f}), "
                  f"{candidates[1]['ticker']} ({candidates[1]['score']:.3f}), "
                  f"{candidates[2]['ticker']} ({candidates[2]['score']:.3f})")
    else:
        print(f"   ⚠️ Mom-Vol Hybrid 6M: No candidates found (checked {len(all_tickers)} tickers)")

    return [c['ticker'] for c in candidates[:top_n]]


def select_momentum_volatility_hybrid_1y3m_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Momentum-Volatility Strategy (1Y/3M Ratio variant): Combines strong 1-year momentum with weak 3-month
    performance (buy on dip) and controlled volatility.

    Strategy: Find stocks with good 1Y performance but recent 3M pullback, filtered by low volatility.
    """
    if current_date is None:
        current_date = datetime.now()

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Mom-Vol Hybrid 1Y/3M"
    )

    candidates = []

    # Use filtered tickers for analysis
    tickers_to_scan = set(filtered_tickers)
    if isinstance(ticker_data_grouped, dict):
        tickers_to_scan.update(ticker_data_grouped.keys())

    for ticker in tickers_to_scan:
        try:
            # Get ticker data - handle both dict and GroupBy objects
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None

            if ticker_data is None or len(ticker_data) == 0:
                continue

            # ✅ FIX: Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data

            # Use dropna'd Close series for all calculations
            close_prices = ticker_data_filtered['Close'].dropna()
            n_prices = len(close_prices)

            if n_prices < 60:
                continue

            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue

            # Calculate 1Y performance (approx 252 trading days)
            lookback_1y = min(252, n_prices - 1)
            if lookback_1y < 60:
                continue
            price_1y_ago = close_prices.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago

            # Calculate 3M performance (approx 63 trading days)
            lookback_3m = min(63, n_prices - 1)
            if lookback_3m < 20:
                continue
            price_3m_ago = close_prices.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue
            performance_3m = (latest_price - price_3m_ago) / price_3m_ago

            # Calculate volatility (using daily returns)
            daily_returns = ticker_data_filtered['Close'].pct_change().dropna()
            if len(daily_returns) < 30:
                continue
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility

            # Calculate average volume
            avg_volume = ticker_data['Volume'].mean() if 'Volume' in ticker_data.columns else 100000

            # Apply filters - looking for strong 1Y but weak 3M (buy on dip)
            if (performance_1y > 0.05 and  # At least 5% gain in 1Y
                performance_3m < 0.10 and  # Less than 10% gain in 3M (or negative = pullback)
                performance_1y > performance_3m and  # 1Y must be stronger than 3M
                volatility < 2.0 and  # Volatility < 200%
                avg_volume > 10000):  # Low volume threshold

                # Calculate composite score: reward high 1Y/3M ratio with volatility penalty
                # Higher ratio = stronger long-term trend with recent pullback
                ratio_1y_3m = (1 + performance_1y) / (1 + performance_3m) if performance_3m > -0.5 else 2.0
                volatility_penalty = min(volatility, 1.0)
                composite_score = ratio_1y_3m * (1 - volatility_penalty * 0.3)

                candidates.append({
                    'ticker': ticker,
                    'score': composite_score,
                    'performance_1y': performance_1y,
                    'performance_3m': performance_3m,
                    'ratio_1y_3m': ratio_1y_3m,
                    'volatility': volatility
                })

        except Exception as e:
            continue

    # Sort by composite score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Debug output
    if candidates:
        print(f"   🎯 Mom-Vol Hybrid 1Y/3M: Found {len(candidates)} candidates")
        if len(candidates) >= 3:
            print(f"   Top 3: {candidates[0]['ticker']} (ratio={candidates[0]['ratio_1y_3m']:.2f}, score={candidates[0]['score']:.3f}), "
                  f"{candidates[1]['ticker']} (ratio={candidates[1]['ratio_1y_3m']:.2f}, score={candidates[1]['score']:.3f}), "
                  f"{candidates[2]['ticker']} (ratio={candidates[2]['ratio_1y_3m']:.2f}, score={candidates[2]['score']:.3f})")
    else:
        print(f"   ⚠️ Mom-Vol Hybrid 1Y/3M: No candidates found (checked {len(all_tickers)} tickers)")

    return [c['ticker'] for c in candidates[:top_n]]


def select_momentum_volatility_hybrid_1y_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Momentum-Volatility Strategy (1Y variant): Combines strong 1-year momentum with controlled volatility.
    Same as 3M/6M variants but uses 1-year lookback as primary momentum signal.
    """
    if current_date is None:
        current_date = datetime.now()

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Mom-Vol Hybrid 1Y"
    )

    candidates = []

    # Use filtered tickers for analysis
    tickers_to_scan = set(filtered_tickers)
    if isinstance(ticker_data_grouped, dict):
        tickers_to_scan.update(ticker_data_grouped.keys())

    for ticker in tickers_to_scan:
        try:
            # Get ticker data - handle both dict and GroupBy objects
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None

            if ticker_data is None or len(ticker_data) == 0:
                continue

            # Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data

            # Use dropna'd Close series for all calculations
            close_prices = ticker_data_filtered['Close'].dropna()
            n_prices = len(close_prices)

            if n_prices < 100:
                continue

            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue

            # Calculate 1Y performance (approx 252 trading days)
            lookback_1y = min(252, n_prices - 1)
            if lookback_1y < 100:
                continue
            price_1y_ago = close_prices.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago

            # Calculate 3Y performance for context (approx 756 trading days)
            lookback_3y = min(756, n_prices - 1)
            if lookback_3y >= 200:
                price_3y_ago = close_prices.iloc[-lookback_3y]
                if price_3y_ago > 0:
                    performance_3y = (latest_price - price_3y_ago) / price_3y_ago
                else:
                    performance_3y = 0.0
            else:
                performance_3y = 0.0

            # Calculate volatility (using daily returns)
            daily_returns = ticker_data['Close'].pct_change().dropna()
            if len(daily_returns) < 60:
                continue
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility

            # Calculate average volume
            avg_volume = ticker_data['Volume'].mean() if 'Volume' in ticker_data.columns else 100000

            # Apply filters - RELAXED criteria for 1Y
            if (performance_1y > 0.0 and  # Any positive 1Y momentum
                performance_3y > -0.5 and  # Allow up to 50% loss in 3Y (or no 3Y data)
                volatility < 2.5 and  # Volatility < 250%
                avg_volume > 10000):  # Low volume threshold

                # Calculate composite score (1Y weighted higher, 3Y provides context)
                momentum_score = performance_1y * 0.7 + max(performance_3y, 0) * 0.3
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)

                candidates.append({
                    'ticker': ticker,
                    'score': composite_score,
                    'performance_1y': performance_1y,
                    'performance_3y': performance_3y,
                    'volatility': volatility
                })

        except Exception as e:
            continue

    # Sort by composite score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Debug output
    if candidates:
        print(f"   🎯 Mom-Vol Hybrid 1Y: Found {len(candidates)} candidates")
        if len(candidates) >= 3:
            print(f"   Top 3: {candidates[0]['ticker']} ({candidates[0]['score']:.3f}), "
                  f"{candidates[1]['ticker']} ({candidates[1]['score']:.3f}), "
                  f"{candidates[2]['ticker']} ({candidates[2]['score']:.3f})")
    else:
        print(f"   ⚠️ Mom-Vol Hybrid 1Y: No candidates found (checked {len(all_tickers)} tickers)")

    return [c['ticker'] for c in candidates[:top_n]]


def select_price_acceleration_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Price Acceleration Strategy: Uses velocity (price change) and acceleration (velocity change)
    to identify stocks with increasing momentum.

    Physics-inspired approach:
    - Velocity = price.pct_change() (daily return rate)
    - Acceleration = velocity.diff() (change in velocity)
    - Buy when acceleration is positive (momentum increasing)
    """
    from config import INVERSE_ETFS, MIN_DATA_DAYS_PERIOD_DATA

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use, ticker_data_grouped, current_date, "Price Acceleration"
    )

    candidates = []
    data_insufficient = 0
    low_velocity = 0
    negative_acceleration = 0

    print(f"   🚀 Price Acceleration: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")
    print(f"   📐 Formula: velocity = price.pct_change(), acceleration = velocity.diff()")

    for ticker in filtered_tickers:
        try:
            # Get ticker data
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None

            from config import MIN_DATA_DAYS_PERIOD_DATA
            if ticker_data is None or len(ticker_data) < MIN_DATA_DAYS_PERIOD_DATA:
                data_insufficient += 1
                continue

            # ✅ FIX: Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data

            # Calculate velocity (daily returns)
            prices = ticker_data_filtered['Close'].dropna()
            if len(prices) < 30:
                data_insufficient += 1
                continue

            velocity = prices.pct_change().dropna()
            if len(velocity) < 20:
                data_insufficient += 1
                continue

            # Calculate acceleration (change in velocity)
            acceleration = velocity.diff().dropna()
            if len(acceleration) < 10:
                data_insufficient += 1
                continue

            # Get recent metrics (last 10 days)
            recent_velocity = velocity.tail(10).mean()
            recent_acceleration = acceleration.tail(5).mean()  # Last 5 days acceleration
            latest_acceleration = acceleration.iloc[-1]  # Most recent day

            # Calculate trend consistency (how many recent days have positive acceleration)
            recent_accel_series = acceleration.tail(5)
            positive_accel_days = (recent_accel_series > 0).sum()
            consistency_score = positive_accel_days / 5  # 0.0 to 1.0

            # Debug first few stocks
            if len(candidates) < 3:
                print(f"   🔍 DEBUG {ticker}: velocity={recent_velocity:.4f}, accel={recent_acceleration:.6f}, "
                      f"latest={latest_acceleration:.6f}, consistency={consistency_score:.1%}")

            # Selection criteria:
            # 1. Positive recent velocity (price going up)
            # 2. Positive acceleration (momentum increasing)
            # 3. Recent consistency (at least 3 of 5 days with positive acceleration)
            # 4. Strong latest acceleration signal

            if recent_velocity <= 0.001:  # At least 0.1% average daily gain
                low_velocity += 1
                continue

            if recent_acceleration <= 0:  # Must have positive acceleration
                negative_acceleration += 1
                continue

            # Calculate composite acceleration score
            # Weights: 40% avg acceleration, 40% latest acceleration, 20% consistency
            accel_score = (recent_acceleration * 0.4 +
                          latest_acceleration * 0.4 +
                          consistency_score * recent_acceleration * 0.2)

            # Scale by velocity (stronger acceleration matters more with higher velocity)
            final_score = accel_score * (1 + recent_velocity * 100)

            candidates.append({
                'ticker': ticker,
                'score': final_score,
                'velocity': recent_velocity,
                'acceleration': recent_acceleration,
                'latest_accel': latest_acceleration,
                'consistency': consistency_score
            })

        except Exception as e:
            data_insufficient += 1
            continue

    # Sort by acceleration score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"   📊 Analysis Summary:")
    print(f"      Total analyzed: {len(all_tickers)}")
    print(f"      Data insufficient: {data_insufficient}")
    print(f"      Low velocity: {low_velocity}")
    print(f"      Negative acceleration: {negative_acceleration}")
    print(f"      Valid candidates: {len(candidates)}")

    if candidates:
        print(f"   📊 Top {min(len(candidates), top_n)} Price Acceleration candidates:")
        for i, c in enumerate(candidates[:top_n], 1):
            print(f"      {i}. {c['ticker']}: score={c['score']:.4f}, "
                  f"velocity={c['velocity']:.4f}, accel={c['acceleration']:.6f}, "
                  f"consistency={c['consistency']:.0%}")

        selected = [c['ticker'] for c in candidates[:top_n]]
        print(f"   ✅ Price Acceleration selected {len(selected)} tickers: {selected}")
        return selected
    else:
        print(f"   ❌ No Price Acceleration candidates found")
        return []


def select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, period='1y', current_date=None, top_n=20):
    """
    Shared Dynamic Buy & Hold stock selection logic.
    """
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, f"Dynamic BH {period}"
    )

    performances = []

    # Use current date or last available date
    if current_date is None:
        # Find the latest date across all tickers
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []

    # Determine lookback period
    if period == '1y':
        lookback_days = 365
    elif period == '6m':
        lookback_days = 180
    elif period == '3m':
        lookback_days = 90
    elif period == '1m':
        lookback_days = 30
    else:
        lookback_days = 365

    stale_data_count = 0
    analyzed_count = 0

    for ticker in filtered_tickers:  # Process filtered tickers
        try:
            if ticker not in ticker_data_grouped:
                continue

            analyzed_count += 1
            ticker_data = ticker_data_grouped[ticker]

            # Check data freshness before processing
            data_max_date = ticker_data.index.max()
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)

            data_age_days = (current_ts - data_max_date).total_seconds() / 86400
            if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                stale_data_count += 1
                continue  # Skip stale data instead of raising exception

            # Calculate start date based on lookback period
            start_date = current_date - timedelta(days=lookback_days)

            # Filter data to the exact period - be more flexible with date range
            available_start = ticker_data.index.min()
            available_end = ticker_data.index.max()

            # Use available data if it covers most of the period
            if available_start > start_date:
                start_date = available_start

            period_data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= current_ts)]

            # Reduce minimum data requirement for live trading
            min_data_points = max(5, lookback_days // 30)  # At least 5 points or 1 per month
            if len(period_data) < min_data_points:
                continue

            valid_close = period_data['Close'].dropna()
            if len(valid_close) < 2:
                continue

            start_price = valid_close.iloc[0]
            end_price = valid_close.iloc[-1]

            if start_price > 0:
                performance = ((end_price - start_price) / start_price) * 100
                # Include all stocks (not just positive performance) to ensure we have picks
                performances.append((ticker, performance))

        except Exception as e:
            continue

    # Check if all data was stale
    if stale_data_count > 0 and stale_data_count == analyzed_count:
        print(f"   ❌ No stocks passed filtering (stale data)")
        return []

    # Sort by performance and get top N
    if performances:
        performances.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, _ in performances[:top_n]]

        print(f"   📊 Top {top_n} performers ({period}): {selected_tickers}")
        for i, (ticker, perf) in enumerate(performances[:top_n], 1):
            print(f"      {i}. {ticker}: {perf:+.1f}%")

        return selected_tickers
    else:
        if stale_data_count > 0:
            print(f"   ❌ No valid performance data found for {period} (stale data: {stale_data_count})")
        else:
            print(f"   ❌ No valid performance data found for {period}")
        return []


def select_voting_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Voting Ensemble Strategy: Combines stock picks from multiple strategies
    and selects stocks with the highest consensus votes.

    Strategy List:
    - momentum_volatility_hybrid
    - price_acceleration
    - static_bh_6m
    - ratio_3m
    - vol_adj_mom
    - momentum_ai_hybrid
    - static_bh_3m
    - risk_adj_mom
    - dynamic_bh_1y_trailing_stop
    - dynamic_bh_1y_vol_filter
    - adaptive_ensemble
    - sentiment_ensemble

    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dictionary of ticker data
        current_date: Current date for analysis
        top_n: Number of stocks to return

    Returns:
        List of selected ticker symbols with highest consensus votes
    """
    print(f"   🗳️  Voting Ensemble: Analyzing {len(all_tickers)} tickers")
    print(f"   📊 Combining picks from 10 strategies...")

    # Collect stock picks from each strategy
    strategy_picks = {}
    vote_counts = {}
    total_strategies = 0

    # Strategy 1: Momentum Volatility Hybrid
    try:
        print(f"   🔄 Getting picks from momentum_volatility_hybrid...")
        stocks = select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["momentum_volatility_hybrid"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ momentum_volatility_hybrid: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ momentum_volatility_hybrid failed: {e}")

    # Strategy 2: Price Acceleration
    try:
        print(f"   🔄 Getting picks from price_acceleration...")
        stocks = select_price_acceleration_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["price_acceleration"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ price_acceleration: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ price_acceleration failed: {e}")

    # Strategy 3: 3M/1Y Ratio
    try:
        print(f"   🔄 Getting picks from ratio_3m_1y...")
        stocks = select_3m_1y_ratio_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["ratio_3m_1y"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ ratio_3m_1y: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ ratio_3m_1y failed: {e}")

    # Strategy 4: Vol-Adj Mom
    try:
        print(f"   🔄 Getting picks from vol_adj_mom...")
        stocks = select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["vol_adj_mom"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ vol_adj_mom: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ vol_adj_mom failed: {e}")

    # Strategy 5: Quality Momentum
    try:
        print(f"   🔄 Getting picks from quality_momentum...")
        stocks = select_quality_momentum_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["quality_momentum"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ quality_momentum: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ quality_momentum failed: {e}")

    # Strategy 6: Turnaround
    try:
        print(f"   🔄 Getting picks from turnaround...")
        stocks = select_turnaround_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["turnaround"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ turnaround: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ turnaround failed: {e}")

    # Strategy 7: 1Y/3M Ratio
    try:
        print(f"   🔄 Getting picks from ratio_1y_3m...")
        stocks = select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        if stocks:
            strategy_picks["ratio_1y_3m"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ ratio_1y_3m: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ ratio_1y_3m failed: {e}")

    # Strategy 8: Dynamic BH 6M
    try:
        print(f"   🔄 Getting picks from dynamic_bh_6m...")
        stocks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, "6m", current_date, top_n)
        if stocks:
            strategy_picks["dynamic_bh_6m"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ dynamic_bh_6m: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ dynamic_bh_6m failed: {e}")

    # Strategy 9: Dynamic BH 3M
    try:
        print(f"   🔄 Getting picks from dynamic_bh_3m...")
        stocks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, "3m", current_date, top_n)
        if stocks:
            strategy_picks["dynamic_bh_3m"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ dynamic_bh_3m: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ dynamic_bh_3m failed: {e}")

    # Strategy 10: Dynamic BH 1Y
    try:
        print(f"   🔄 Getting picks from dynamic_bh_1y...")
        stocks = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, "1y", current_date, top_n)
        if stocks:
            strategy_picks["dynamic_bh_1y"] = stocks
            for stock in stocks:
                vote_counts[stock] = vote_counts.get(stock, 0) + 1
            total_strategies += 1
            print(f"   ✅ dynamic_bh_1y: {len(stocks)} stocks")
    except Exception as e:
        print(f"   ⚠️ dynamic_bh_1y failed: {e}")

    print(f"   📊 {total_strategies} strategies contributed votes")

    # Sort stocks by vote count (descending), then by ticker symbol for consistency
    sorted_stocks = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))

    # Select top N stocks with highest votes
    top_voted_stocks = [stock for stock, votes in sorted_stocks[:top_n]]

    # Display voting results
    if top_voted_stocks:
        print(f"\n   📊 VOTING RESULTS (Top {len(top_voted_stocks)} stocks):")
        print(f"   {'Rank':<5} {'Ticker':<8} {'Votes':<8} {'Strategies':<50}")
        print(f"   {'-'*5} {'-'*8} {'-'*8} {'-'*50}")

        for rank, (stock, votes) in enumerate(sorted_stocks[:top_n], 1):
            # Find which strategies voted for this stock
            voting_strategies = []
            for strategy_name, picks in strategy_picks.items():
                if stock in picks:
                    voting_strategies.append(strategy_name.replace('_', ' ').title())

            strategies_str = ', '.join(voting_strategies[:3])  # Show first 3 strategies
            if len(voting_strategies) > 3:
                strategies_str += f" (+{len(voting_strategies)-3} more)"

            vote_bar = "█" * votes + "░" * (total_strategies - votes)
            print(f"   {rank:<5} {stock:<8} {votes}/{total_strategies} {vote_bar} {strategies_str}")

        print(f"\n   🗳️  Voting Ensemble selected {len(top_voted_stocks)} tickers:")
        print(f"   ✅ {top_voted_stocks}")
    else:
        print(f"   ❌ No stocks received votes from any strategy")

    return top_voted_stocks


def select_top_performers(all_tickers, ticker_data_grouped, current_date, lookback_days, top_n=10,
                          apply_performance_filter=False, filter_label="Strategy",
                          exclude_inverse_etfs=True):
    """
    Shared stock selection by historical performance: calculate_parallel_performance + sort + top N.

    This is the SINGLE source of truth for performance-based stock selection.
    Used by Static BH, Dynamic BH, Monthly variants, and their extensions.

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        lookback_days: Number of days to look back for performance (365=1Y, 180=6M, 90=3M, 30=1M)
        top_n: Number of stocks to select
        apply_performance_filter: If True, apply performance_filters before ranking
        filter_label: Label for performance filter logging
        exclude_inverse_etfs: If True, exclude inverse ETFs from selection (they decay over time)

    Returns:
        List of selected ticker symbols (top performers)
    """
    from parallel_backtest import calculate_parallel_performance
    from config import INVERSE_ETFS

    # Exclude inverse ETFs from buy-and-hold strategies (they decay over time)
    if exclude_inverse_etfs:
        tickers_to_rank = [t for t in all_tickers if t not in INVERSE_ETFS]
    else:
        tickers_to_rank = all_tickers
    if apply_performance_filter:
        from performance_filters import filter_tickers_by_performance
        tickers_to_rank = filter_tickers_by_performance(
            tickers_to_rank, ticker_data_grouped, current_date, filter_label
        )

    performances = calculate_parallel_performance(
        tickers_to_rank, ticker_data_grouped, current_date, period_days=lookback_days
    )

    if performances:
        performances.sort(key=lambda x: x[1], reverse=True)
        selected = [ticker for ticker, _ in performances[:top_n]]
        return selected

    return []


def select_top_performers_with_scores(all_tickers, ticker_data_grouped, current_date, lookback_days, top_n=10,
                                       apply_performance_filter=False, filter_label="Strategy"):
    """
    Same as select_top_performers but also returns performance scores.

    Returns:
        List of (ticker, performance_pct) tuples, sorted descending
    """
    from parallel_backtest import calculate_parallel_performance

    tickers_to_rank = all_tickers
    if apply_performance_filter:
        from performance_filters import filter_tickers_by_performance
        tickers_to_rank = filter_tickers_by_performance(
            all_tickers, ticker_data_grouped, current_date, filter_label
        )

    performances = calculate_parallel_performance(
        tickers_to_rank, ticker_data_grouped, current_date, period_days=lookback_days
    )

    if performances:
        performances.sort(key=lambda x: x[1], reverse=True)
        return performances[:top_n]

    return []


def select_top_performers_vol_filtered(all_tickers, ticker_data_grouped, current_date, lookback_days,
                                        max_volatility, top_n=10):
    """
    Performance-based stock selection with volatility filter.
    Calculates performances, filters by max annualized volatility, then returns top N.

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        lookback_days: Number of days for performance calculation
        max_volatility: Maximum annualized volatility (%) to pass filter
        top_n: Number of stocks to select

    Returns:
        (selected_tickers, stocks_passed, stocks_evaluated) tuple
    """
    import pandas as pd
    from parallel_backtest import calculate_parallel_performance
    from datetime import timedelta

    performances = calculate_parallel_performance(
        all_tickers, ticker_data_grouped, current_date, period_days=lookback_days
    )

    filtered = []
    stocks_evaluated = 0
    stocks_passed = 0

    for ticker, perf_pct in performances:
        try:
            ticker_data = ticker_data_grouped[ticker]
            perf_start_date = current_date - timedelta(days=lookback_days)
            perf_data = ticker_data.loc[perf_start_date:current_date]

            if len(perf_data) >= 50:
                valid_close = perf_data['Close'].dropna()
                if len(valid_close) >= 2:
                    stocks_evaluated += 1
                    daily_returns = valid_close.pct_change(fill_method=None).dropna()
                    if len(daily_returns) > 10:
                        annualized_volatility = daily_returns.std() * (252 ** 0.5) * 100
                        if annualized_volatility <= max_volatility:
                            filtered.append((ticker, perf_pct, annualized_volatility))
                            stocks_passed += 1
        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    if filtered:
        filtered.sort(key=lambda x: x[1], reverse=True)
        selected = [ticker for ticker, _, _ in filtered[:top_n]]
        return selected, stocks_passed, stocks_evaluated

    return [], stocks_passed, stocks_evaluated


def select_momentum_ai_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=10):
    """
    Momentum AI Hybrid Strategy: Selects top momentum stocks based on lookback period.

    Shared function used by both backtesting and live trading.

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select

    Returns:
        List of selected ticker symbols
    """
    import pandas as pd
    from config import MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK, INVERSE_ETFS

    # Exclude inverse ETFs - they should only be in the inverse_etf_hedge strategy
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]

    momentum_scores = []

    for ticker in tickers_to_use:
        try:
            if ticker not in ticker_data_grouped:
                continue
            ticker_history = ticker_data_grouped[ticker]

            # Filter to current date
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                ticker_history = ticker_history[ticker_history.index <= current_ts]

            ticker_history = ticker_history.tail(MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK + 10)

            if len(ticker_history) >= MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK:
                lookback_data = ticker_history.tail(MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK)
                start_price = lookback_data.iloc[0]['Close']
                end_price = lookback_data.iloc[-1]['Close']

                if start_price > 0:
                    momentum_return = (end_price - start_price) / start_price
                    momentum_scores.append((ticker, momentum_return))

        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    if momentum_scores:
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        top_stocks = [ticker for ticker, score in momentum_scores[:top_n]]
        print(f"   📈 Momentum AI Hybrid: Top {top_n} stocks: {[(t, f'{s*100:.1f}%') for t, s in momentum_scores[:top_n]]}")
        return top_stocks

    return []


def select_ai_elite_with_training(
    all_tickers: list,
    ticker_data_grouped: dict,
    current_date=None,
    top_n: int = 10,
    ai_elite_models: dict = None,
    force_train: bool = False,
    model_path_suffix: str = ""
) -> tuple:
    """
    AI Elite Strategy: Full pipeline — load model, train if needed, select stocks.

    This is the SINGLE source of truth for AI Elite. Both backtesting and live trading call this.

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        ai_elite_models: Dict of models (mutated in-place). Pass {} on first call.
        force_train: If True, always retrain even if model exists on disk
        model_path_suffix: Suffix for model file (e.g. "_monthly" for AI Elite Monthly)

    Returns:
        (selected_stocks, ai_elite_models) — selected tickers and updated models dict
    """
    from ai_elite_strategy_per_ticker import train_shared_base_model, collect_ticker_training_data
    from ai_elite_strategy import select_ai_elite_stocks
    from config import AI_ELITE_TRAINING_LOOKBACK, AI_ELITE_FORWARD_DAYS
    import os
    import pickle
    from datetime import timedelta, timezone as tz_utc

    if ai_elite_models is None:
        ai_elite_models = {}

    models_dir = "logs/models"
    base_model_path = os.path.join(models_dir, f"_shared_base_ai_elite{model_path_suffix}.joblib")

    # Step 1: Try loading saved model from disk if not already in memory
    if ai_elite_models.get('_shared_base') is None and not force_train:
        print(f"   🔍 AI Elite: Looking for model at {base_model_path}")
        if os.path.exists(base_model_path):
            try:
                print(f"   📁 AI Elite: Found model file, loading...")
                with open(base_model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Handle both old format (direct model) and new format (model + metadata)
                if isinstance(model_data, dict) and 'all_models' in model_data:
                    # New format - model dict with metadata at top level
                    loaded_model = model_data
                    metadata = model_data.get('metadata', {})

                    # Display training/update info
                    if 'trained' in metadata:
                        trained_date = metadata['trained'][:10]  # YYYY-MM-DD format
                        msg = f"   ✅ AI Elite: Loaded model from {base_model_path} (trained {trained_date}"
                        if 'train_start' in metadata and 'train_end' in metadata:
                            start_date = metadata['train_start'][:10]
                            end_date = metadata['train_end'][:10]
                            msg += f", data {start_date} to {end_date}"
                        msg += ")"
                        print(msg)
                    elif 'updated' in metadata:
                        updated_date = metadata['updated'][:10]
                        msg = f"   ✅ AI Elite: Loaded model from {base_model_path} (updated {updated_date}"
                        if 'train_start' in metadata and 'train_end' in metadata:
                            start_date = metadata['train_start'][:10]
                            end_date = metadata['train_end'][:10]
                            msg += f", data {start_date} to {end_date}"
                        msg += ")"
                        print(msg)
                    else:
                        print(f"   ✅ AI Elite: Loaded model from {base_model_path}")
                elif isinstance(model_data, dict) and 'model' in model_data:
                    # Wrapped format - extract the actual model
                    loaded_model = model_data['model']
                    metadata = model_data.get('metadata', {})
                    print(f"   ✅ AI Elite: Loaded model from {base_model_path} (wrapped format)")
                else:
                    # Old format - direct model
                    loaded_model = model_data
                    print(f"   ✅ AI Elite: Loaded model from {base_model_path} (legacy format)")

                ai_elite_models['_shared_base'] = loaded_model
                for ticker in all_tickers:
                    ai_elite_models[ticker] = loaded_model
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to load model: {e}")

    # Step 2: Always train (incrementally if model exists, fresh if not)
    # This ensures model adapts to recent market conditions
    print(f"   🎓 AI Elite: Training shared base model...")

    # Determine training window
    if current_date is None:
        import pandas as pd
        from datetime import datetime
        current_date = datetime.now(tz_utc.utc)

    train_end = current_date
    train_start = train_end - timedelta(days=AI_ELITE_TRAINING_LOOKBACK)

    # Pre-compute market returns (same as backtesting)
    market_returns = {}
    sample_date_iter = train_start
    while sample_date_iter <= train_end:
        mr = _calculate_market_return(ticker_data_grouped, sample_date_iter, AI_ELITE_FORWARD_DAYS)
        utc_key = sample_date_iter.replace(tzinfo=tz_utc.utc) if sample_date_iter.tzinfo is None else sample_date_iter
        market_returns[utc_key] = mr if mr is not None else 0.0
        sample_date_iter += timedelta(days=2)

    # Collect training data from all tickers (PARALLEL with multiprocessing.Pool)
    from multiprocessing import Pool, cpu_count
    import time

    n_workers = max(1, cpu_count() - 2)
    print(f"   📊 AI Elite: Collecting data from {len(all_tickers)} tickers ({n_workers} processes, {AI_ELITE_TRAINING_LOOKBACK}d lookback)...")
    start_time = time.time()

    all_training_data = []
    ticker_samples_map = {}

    # Prepare args for parallel workers
    collect_args = [
        (t, ticker_data_grouped.get(t), train_start, train_end, AI_ELITE_FORWARD_DAYS, market_returns)
        for t in all_tickers
    ]

    with Pool(processes=n_workers) as pool:
        from backtesting import _collect_data_worker
        results = pool.map(_collect_data_worker, collect_args)

    for ticker, samples in results:
        if samples:
            all_training_data.extend(samples)
            ticker_samples_map[ticker] = samples

    elapsed = time.time() - start_time
    print(f"   📊 AI Elite: Collected {len(all_training_data)} samples from {len(ticker_samples_map)} tickers ({elapsed:.1f}s)")

    # Train shared base model
    os.makedirs(models_dir, exist_ok=True)
    existing_base = ai_elite_models.get('_shared_base')
    base_model, base_r2 = train_shared_base_model(
        all_training_data, save_path=base_model_path,
        existing_model=existing_base, train_start=train_start, train_end=train_end
    )

    if base_model:
        ai_elite_models['_shared_base'] = base_model
        for ticker in all_tickers:
            ai_elite_models[ticker] = base_model
        print(f"   ✅ AI Elite: Model trained (R² {base_r2:.3f})")
    else:
        print(f"   ⚠️ AI Elite: Training failed, no model produced")

    # Step 3: Select stocks using trained model
    selected = select_ai_elite_stocks(
        all_tickers, ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        per_ticker_models=ai_elite_models
    )

    return selected, ai_elite_models


def select_bh_1y_volsweet_accel_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
) -> List[str]:
    """
    BH 1Y VolSweet Acceleration Strategy:
    Combines Static BH 1Y and 1M VolSweet tickers, ranks by acceleration score.

    Selection process:
    1. Get top performers from Static BH 1Y (1-year lookback)
    2. Get top performers from 1M VolSweet (1-month risk-adjusted momentum + vol filter)
    3. Combine unique tickers from both strategies
    4. Rank by momentum acceleration score
    5. Return top N

    Returns:
        List of selected tickers
    """
    from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks

    if current_date is None:
        current_date = datetime.now(timezone.utc)

    print(f"   📊 BH 1Y VolSweet Accel: Getting Static BH 1Y tickers...")
    bh_1y_tickers = select_top_performers(
        all_tickers, ticker_data_grouped,
        current_date=current_date,
        lookback_days=365,
        top_n=top_n * 2,  # Get more for combination
        apply_performance_filter=False
    )

    print(f"   📊 BH 1Y VolSweet Accel: Getting 1M VolSweet tickers...")
    volsweet_tickers = select_risk_adj_mom_1m_vol_sweet_stocks(
        all_tickers, ticker_data_grouped,
        current_date=current_date,
        top_n=top_n * 2
    )

    # Combine unique tickers from both strategies
    combined_tickers = list(set(bh_1y_tickers) | set(volsweet_tickers))
    print(f"   📊 BH 1Y VolSweet Accel: Combined {len(combined_tickers)} unique tickers")

    # Calculate acceleration scores for combined tickers
    scored_candidates = []

    for ticker in combined_tickers:
        if ticker not in ticker_data_grouped:
            continue

        try:
            data = ticker_data_grouped[ticker].loc[:current_date]
            close = data['Close'].dropna()

            if len(close) < 60:
                continue

            # Calculate momentum metrics
            mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0

            # Calculate acceleration: 1M momentum vs expected from 3M trend
            expected_1m_from_3m = mom_3m / 3 if mom_3m != 0 else 0
            acceleration = mom_1m - expected_1m_from_3m

            # Calculate velocity (recent momentum strength)
            recent_velocity = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0

            # Calculate consistency (how consistent is the momentum)
            returns_5d = close.pct_change(5).dropna()
            if len(returns_5d) >= 20:
                accel_series = returns_5d.rolling(5).mean().diff().iloc[-20:]
                consistency = (accel_series > 0).mean()
            else:
                consistency = 0.5

            # Composite acceleration score (same weights as shared_strategies.py)
            # 40% avg acceleration, 40% latest acceleration, 20% consistency
            accel_score = (acceleration * 0.4 +
                          acceleration * 0.4 +
                          consistency * acceleration * 0.2)

            # Scale by velocity
            final_score = accel_score * (1 + recent_velocity * 10)

            scored_candidates.append({
                'ticker': ticker,
                'score': final_score,
                'acceleration': acceleration,
                'velocity': recent_velocity,
                'consistency': consistency,
                'mom_1m': mom_1m,
                'mom_3m': mom_3m
            })

        except Exception as e:
            continue

    # Sort by acceleration score (highest first)
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Log top selections
    if scored_candidates:
        print(f"   📈 BH 1Y VolSweet Accel: Top picks with acceleration:")
        for c in scored_candidates[:min(5, len(scored_candidates))]:
            print(f"      {c['ticker']}: accel={c['acceleration']:+.1f}%, velocity={c['velocity']:+.1f}%, score={c['score']:.4f}")

    # Return top N
    selected = [c['ticker'] for c in scored_candidates[:top_n]]

    print(f"   ✅ BH 1Y VolSweet Accel: Selected {len(selected)} tickers: {selected}")
    return selected


def calculate_rebalance_signal(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    market_ticker: str = "SPY"
) -> Dict:
    """
    Calculate dynamic rebalance signal based on market conditions.

    Returns dict with:
    - should_rebalance: bool - whether to rebalance today
    - signal_strength: float - how strong the signal is (0-1)
    - reasons: list of strings - why we should/shouldn't rebalance
    """
    from config import MARKET_FILTER_TICKER

    # Use SPY as market proxy
    market_ticker = MARKET_FILTER_TICKER

    signal = {
        'should_rebalance': False,
        'signal_strength': 0.0,
        'reasons': [],
        'days_since_last': 0
    }

    if market_ticker not in ticker_data_grouped:
        # No market data - use time-based fallback
        signal['reasons'].append('No market data - using time-based')
        return signal

    try:
        data = ticker_data_grouped[market_ticker].loc[:current_date]
        close = data['Close'].dropna()

        if len(close) < 60:
            signal['reasons'].append('Insufficient data')
            return signal

        # 1. Momentum change signal
        # Compare recent momentum to longer-term momentum
        mom_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        mom_21d = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0
        mom_63d = (close.iloc[-1] / close.iloc[-64] - 1) * 100 if len(close) >= 64 else 0

        # If 5d momentum is much stronger than 63d, momentum is accelerating
        if mom_63d > 0:
            momentum_signal = min(1.0, (mom_5d - mom_63d/3) / 10)  # Normalize
        else:
            momentum_signal = 0.0

        # 2. Volatility signal
        # High volatility suggests more frequent rebalancing needed
        returns = close.pct_change().dropna()
        vol_5d = returns.iloc[-5:].std() * np.sqrt(252) * 100 if len(returns) >= 5 else 20
        vol_21d = returns.iloc[-21:].std() * np.sqrt(252) * 100 if len(returns) >= 21 else 20

        # If recent vol is much higher than normal, rebalance more often
        vol_signal = min(1.0, max(0, (vol_21d - vol_5d) / vol_21d)) if vol_21d > 0 else 0

        # 3. Trend change signal
        # If market crossed its moving average, might need rebalance
        sma_21 = close.rolling(21).mean().iloc[-1]
        sma_63 = close.rolling(63).mean().iloc[-1]

        # Current position vs 21d SMA
        price_vs_sma = (close.iloc[-1] / sma_21 - 1) * 100 if not np.isnan(sma_21) else 0

        # If price just crossed SMA, strong signal
        prev_price_vs_sma = (close.iloc[-2] / sma_21 - 1) * 100 if len(close) >= 3 and not np.isnan(sma_21) else 0

        trend_signal = 0.0
        if prev_price_vs_sma < 0 and price_vs_sma > 0:
            trend_signal = 0.8  # Bullish crossover
        elif prev_price_vs_sma > 0 and price_vs_sma < 0:
            trend_signal = 0.8  # Bearish crossover
        elif abs(price_vs_sma) > 5:
            trend_signal = 0.3  # Far from SMA

        # Combine signals
        combined_signal = (momentum_signal * 0.4 + vol_signal * 0.3 + trend_signal * 0.3)

        signal['signal_strength'] = combined_signal
        signal['should_rebalance'] = combined_signal > 0.4  # Threshold

        if momentum_signal > 0.3:
            signal['reasons'].append(f"Momentum accelerating: {momentum_signal:.2f}")
        if vol_signal > 0.3:
            signal['reasons'].append(f"Volatility increasing: {vol_signal:.2f}")
        if trend_signal > 0.3:
            signal['reasons'].append(f"Trend change: {trend_signal:.2f}")

    except Exception as e:
        signal['reasons'].append(f'Error: {str(e)[:30]}')

    return signal


def select_bh_1y_dynamic_accel_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    days_since_rebalance: int = 0,
    min_days: int = 5,
    max_days: int = 44,
) -> Tuple[List[str], bool]:
    """
    BH 1Y Dynamic Acceleration Strategy:
    Combines Static BH 1Y and 1M VolSweet tickers, ranks by acceleration.
    Dynamically decides when to rebalance based on market conditions.

    Returns:
        Tuple of (selected_tickers, should_rebalance)
    """
    from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks

    if current_date is None:
        current_date = datetime.now(timezone.utc)

    # Calculate rebalance signal
    rebalance_signal = calculate_rebalance_signal(ticker_data_grouped, current_date)

    # Force rebalance if too long since last rebalance
    force_rebalance = days_since_rebalance >= max_days

    # Don't rebalance if too soon (unless forced or day 1)
    if days_since_rebalance < min_days and days_since_rebalance > 0 and not force_rebalance and not rebalance_signal['should_rebalance']:
        print(f"   ⏳ BH 1Y Dynamic Accel: Skipping (days={days_since_rebalance}, signal={rebalance_signal['signal_strength']:.2f})")
        return [], False

    # Day 0 = initial selection, always proceed
    is_initial = days_since_rebalance == 0
    should_rebalance = is_initial or force_rebalance or rebalance_signal['should_rebalance']

    if not should_rebalance:
        print(f"   ⏳ BH 1Y Dynamic Accel: No rebalance signal (days={days_since_rebalance}, signal={rebalance_signal['signal_strength']:.2f})")
        if rebalance_signal['reasons']:
            print(f"      Reasons: {rebalance_signal['reasons']}")
        return [], False

    print(f"   🔄 BH 1Y Dynamic Accel: REBALANCING (days={days_since_rebalance}, signal={rebalance_signal['signal_strength']:.2f})")
    if rebalance_signal['reasons']:
        print(f"      Reasons: {rebalance_signal['reasons']}")

    # Get tickers from Static BH 1Y
    print(f"   📊 BH 1Y Dynamic Accel: Getting Static BH 1Y tickers...")
    bh_1y_tickers = select_top_performers(
        all_tickers, ticker_data_grouped,
        current_date=current_date,
        lookback_days=365,
        top_n=top_n * 2,
        apply_performance_filter=False
    )

    # Get tickers from 1M VolSweet
    print(f"   📊 BH 1Y Dynamic Accel: Getting 1M VolSweet tickers...")
    volsweet_tickers = select_risk_adj_mom_1m_vol_sweet_stocks(
        all_tickers, ticker_data_grouped,
        current_date=current_date,
        top_n=top_n * 2
    )

    # Combine unique tickers
    combined_tickers = list(set(bh_1y_tickers) | set(volsweet_tickers))
    print(f"   📊 BH 1Y Dynamic Accel: Combined {len(combined_tickers)} unique tickers")

    # Calculate acceleration scores
    scored_candidates = []

    for ticker in combined_tickers:
        if ticker not in ticker_data_grouped:
            continue

        try:
            data = ticker_data_grouped[ticker].loc[:current_date]
            close = data['Close'].dropna()

            if len(close) < 60:
                continue

            # Calculate momentum metrics
            mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
            mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0

            # Acceleration
            expected_1m_from_3m = mom_3m / 3 if mom_3m != 0 else 0
            acceleration = mom_1m - expected_1m_from_3m

            # Velocity
            recent_velocity = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0

            # Consistency
            returns_5d = close.pct_change(5).dropna()
            if len(returns_5d) >= 20:
                accel_series = returns_5d.rolling(5).mean().diff().iloc[-20:]
                consistency = (accel_series > 0).mean()
            else:
                consistency = 0.5

            # Composite score
            accel_score = (acceleration * 0.4 + acceleration * 0.4 + consistency * acceleration * 0.2)
            final_score = accel_score * (1 + recent_velocity * 10)

            scored_candidates.append({
                'ticker': ticker,
                'score': final_score,
                'acceleration': acceleration,
                'velocity': recent_velocity,
                'consistency': consistency
            })

        except Exception:
            continue

    # Sort by score
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Log top picks
    if scored_candidates:
        print(f"   📈 BH 1Y Dynamic Accel: Top picks with acceleration:")
        for c in scored_candidates[:min(5, len(scored_candidates))]:
            print(f"      {c['ticker']}: accel={c['acceleration']:+.1f}%, velocity={c['velocity']:+.1f}%, score={c['score']:.4f}")

    selected = [c['ticker'] for c in scored_candidates[:top_n]]

    print(f"   ✅ BH 1Y Dynamic Accel: Selected {len(selected)} tickers: {selected}")
    return selected, True


# =============================================================================
# STRATEGY REGISTRY - Maps strategy names to selection functions
# =============================================================================

def _get_strategy_registry():
    """
    Returns a dictionary mapping strategy names to their selection functions.
    This is the single source of truth for all strategy implementations.
    Both backtesting and live trading should use this registry.
    """
    return {
        # Static BH strategies (lookback-based)
        'static_bh_1y': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_6m': lambda t, d, dt, n: select_top_performers(t, d, dt, 180, n),
        'static_bh_3m': lambda t, d, dt, n: select_top_performers(t, d, dt, 90, n),
        'static_bh_1m': lambda t, d, dt, n: select_top_performers(t, d, dt, 30, n),

        # Static BH Monthly variants
        'static_bh_1y_monthly': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_6m_monthly': lambda t, d, dt, n: select_top_performers(t, d, dt, 180, n),
        'static_bh_3m_monthly': lambda t, d, dt, n: select_top_performers(t, d, dt, 90, n),
        'static_bh_1m_monthly': lambda t, d, dt, n: select_top_performers(t, d, dt, 30, n),

        # Dynamic BH strategies
        'dynamic_bh_1y': lambda t, d, dt, n: select_dynamic_bh_stocks(t, d, '1y', dt, n),
        'dynamic_bh_6m': lambda t, d, dt, n: select_dynamic_bh_stocks(t, d, '6m', dt, n),
        'dynamic_bh_3m': lambda t, d, dt, n: select_dynamic_bh_stocks(t, d, '3m', dt, n),
        'dynamic_bh_1m': lambda t, d, dt, n: select_dynamic_bh_stocks(t, d, '1m', dt, n),
        'dynamic_bh_1y_vol_filter': lambda t, d, dt, n: select_top_performers_vol_filtered(t, d, dt, 365, 0.4, n)[0],
        'dynamic_bh_1y_trailing_stop': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),

        # Risk-Adjusted Momentum strategies
        'risk_adj_mom': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 365, "Risk-Adj Mom"),
        'risk_adj_mom_6m': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 180, "Risk-Adj Mom 6M"),
        'risk_adj_mom_3m': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 90, "Risk-Adj Mom 3M"),
        'risk_adj_mom_1m': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 30, "Risk-Adj Mom 1M"),
        'risk_adj_mom_6m_monthly': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 180, "RiskAdj 6M Mth"),
        'risk_adj_mom_3m_monthly': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 90, "RiskAdj 3M Mth"),
        'risk_adj_mom_1m_monthly': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 30, "RiskAdj 1M Mth"),
        'risk_adj_mom_3m_sentiment': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 90, "RiskAdj 3M Sent"),
        'risk_adj_mom_3m_market_up': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 90, "RiskAdj 3M Up"),
        'risk_adj_mom_3m_with_stops': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 90, "RiskAdj 3M Stop"),
        'risk_adj_mom_sentiment': lambda t, d, dt, n: select_risk_adj_mom_stocks(t, d, dt, n, 365, "RiskAdj Sent"),

        # Vol Sweet strategies
        'risk_adj_mom_1m_vol_sweet': lambda t, d, dt, n: _select_risk_adj_mom_1m_vol_sweet(t, d, dt, n),
        'vol_sweet_mom': lambda t, d, dt, n: _select_risk_adj_mom_1m_vol_sweet(t, d, dt, n),
        'bh_1y_volsweet_accel': lambda t, d, dt, n: select_bh_1y_volsweet_accel_stocks(t, d, dt, n),
        'bh_1y_dynamic_accel': lambda t, d, dt, n: select_bh_1y_dynamic_accel_stocks(t, d, dt, n, 0, 0, 44)[0],
        'bh_1y_accel': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),  # Alias for bh_1y_accel_buy

        # Mean Reversion & Quality
        'mean_reversion': lambda t, d, dt, n: select_mean_reversion_stocks(t, d, dt, n),
        'quality_momentum': lambda t, d, dt, n: select_quality_momentum_stocks(t, d, dt, n),
        'volatility_adj_mom': lambda t, d, dt, n: select_volatility_adj_mom_stocks(t, d, dt, n),

        # Ratio strategies
        'ratio_3m_1y': lambda t, d, dt, n: select_3m_1y_ratio_stocks(t, d, dt, n),
        '3m_1y_ratio': lambda t, d, dt, n: select_3m_1y_ratio_stocks(t, d, dt, n),
        'ratio_1y_3m': lambda t, d, dt, n: select_1y_3m_ratio_stocks(t, d, dt, n),
        '1y_3m_ratio': lambda t, d, dt, n: select_1y_3m_ratio_stocks(t, d, dt, n),
        'ratio_1m_3m': lambda t, d, dt, n: select_1m_3m_ratio_stocks(t, d, dt, n),
        '1m_3m_ratio': lambda t, d, dt, n: select_1m_3m_ratio_stocks(t, d, dt, n),

        # Momentum-Volatility Hybrid strategies
        'momentum_volatility_hybrid': lambda t, d, dt, n: select_momentum_volatility_hybrid_stocks(t, d, dt, n),
        'momentum_volatility_hybrid_6m': lambda t, d, dt, n: select_momentum_volatility_hybrid_6m_stocks(t, d, dt, n),
        'momentum_volatility_hybrid_1y': lambda t, d, dt, n: select_momentum_volatility_hybrid_1y_stocks(t, d, dt, n),
        'momentum_volatility_hybrid_1y3m': lambda t, d, dt, n: select_momentum_volatility_hybrid_1y3m_stocks(t, d, dt, n),

        # Other strategies
        'turnaround': lambda t, d, dt, n: select_turnaround_stocks(t, d, dt, n),
        'price_acceleration': lambda t, d, dt, n: select_price_acceleration_stocks(t, d, dt, n),
        'sector_rotation': lambda t, d, dt, n: select_sector_rotation_etfs(t, d, dt, n),
        'voting_ensemble': lambda t, d, dt, n: select_voting_ensemble_stocks(t, d, dt, n),
        'momentum_ai_hybrid': lambda t, d, dt, n: select_momentum_ai_hybrid_stocks(t, d, dt, n),

        # AI Elite strategies
        'ai_elite': lambda t, d, dt, n: select_ai_elite_with_training(t, d, dt, n)[0],
        'ai_elite_monthly': lambda t, d, dt, n: select_ai_elite_with_training(t, d, dt, n)[0],
        'ai_elite_filtered': lambda t, d, dt, n: select_ai_elite_with_training(t, d, dt, n)[0],
        'ai_elite_market_up': lambda t, d, dt, n: select_ai_elite_with_training(t, d, dt, n)[0],

        # BH 1Y Adaptive Rebalancing variants (all use same base selection)
        'static_bh_1y_volatility': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_performance': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_momentum': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_atr': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_hybrid': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_volume_filter': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_sector_rotation': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_performance_threshold': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_market_regime': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_momentum_persist': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_overlap': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_rank_drift': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_drawdown': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_1y_smart_monthly': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),

        # BH 1Y Smart Rebalancing variants
        'bh_1y_mom_sell': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_rank_sell': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_trailing_mom': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_volume_confirm': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_sector_aware': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_accel_buy': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'static_bh_3m_accel': lambda t, d, dt, n: select_top_performers(t, d, dt, 90, n),

        # Rebal 1Y variants
        'bh_1y_vol_adj_rebal': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'ratio_1m3m_vol_adj_rebal': lambda t, d, dt, n: select_1y_performers_ranked_by_1m3m_ratio(t, d, dt, n),
        'bh_1y_corr_filter': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_regime_aware': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_risk_parity': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_drift_thresh': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_mom_quality': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_liquidity': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_earnings_avoid': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_multi_factor': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
        'bh_1y_time_decay': lambda t, d, dt, n: select_top_performers(t, d, dt, 365, n),
    }


def _select_risk_adj_mom_1m_vol_sweet(all_tickers, ticker_data_grouped, current_date, top_n):
    """Wrapper for risk_adj_mom_1m_vol_sweet strategy."""
    try:
        from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
        return select_risk_adj_mom_1m_vol_sweet_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
    except ImportError:
        print("   ⚠️ risk_adj_mom_1m_vol_sweet_strategy not available, using risk_adj_mom_1m")
        return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, 30, "Risk-Adj Mom 1M")


def get_strategy_tickers(strategy_name: str, all_tickers: list, ticker_data_grouped: dict,
                         current_date=None, top_n: int = 10) -> list:
    """
    Get tickers for a strategy using the registry.

    This is the main entry point for live execution (--live-run mode).
    For JSON reading (--live-trading mode), use load_strategy_selections_from_json() directly.

    Args:
        strategy_name: Name of the strategy (e.g., 'static_bh_1y', 'risk_adj_mom')
        all_tickers: List of all available tickers
        ticker_data_grouped: Dict mapping ticker -> DataFrame
        current_date: Current date for analysis (None uses latest data)
        top_n: Number of tickers to select

    Returns:
        List of selected tickers, or empty list if strategy not found
    """
    from datetime import datetime, timezone

    # Default to current time if not provided
    if current_date is None:
        current_date = datetime.now(timezone.utc)

    # Get the strategy registry
    registry = _get_strategy_registry()

    # Check if strategy exists in registry
    if strategy_name not in registry:
        print(f"   ⚠️ Strategy '{strategy_name}' not found in registry")
        print(f"   Available strategies: {sorted(registry.keys())}")
        return []

    # Execute the strategy
    try:
        result = registry[strategy_name](all_tickers, ticker_data_grouped, current_date, top_n)
        if result:
            return result
        # Debug: log when result is empty/None
        print(f"   ⚠️ Strategy '{strategy_name}' returned empty result (type: {type(result).__name__})")
        return []
    except Exception as e:
        print(f"   ⚠️ Strategy '{strategy_name}' execution failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_available_strategies() -> list:
    """Return list of all available strategy names in the registry."""
    return sorted(_get_strategy_registry().keys())
