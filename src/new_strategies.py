"""
New Advanced Trading Strategies

1. Momentum Acceleration - 3M momentum with acceleration filter
2. Concentrated 3M - Fewer positions with volatility filter
3. Dual Momentum - Absolute + relative momentum (Antonacci style)
4. Trend Following ATR - Trend following with ATR trailing stops
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_history_up_to,
    get_cached_values_between,
    get_cached_window,
    resolve_cache_current_date,
)

from config import (
    PORTFOLIO_SIZE,
    MOM_ACCEL_LOOKBACK_DAYS, MOM_ACCEL_SHORT_LOOKBACK, MOM_ACCEL_MIN_ACCELERATION,
    CONCENTRATED_3M_MAX_VOLATILITY, CONCENTRATED_3M_REBALANCE_DAYS,
    DUAL_MOM_LOOKBACK_DAYS, DUAL_MOM_ABSOLUTE_THRESHOLD, DUAL_MOM_RISK_OFF_TICKER,
    TREND_ATR_LOOKBACK_DAYS, TREND_ATR_PERIOD, TREND_ATR_TRAILING_MULT, TREND_ATR_ENTRY_BREAKOUT,
)


# ============================================
# 1. MOMENTUM ACCELERATION STRATEGY
# ============================================

def select_momentum_acceleration_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    Momentum Acceleration Strategy:
    - Require positive 3M momentum
    - Require momentum acceleration (current 1M > previous 1M)
    - Rank by acceleration-adjusted momentum score

    Returns:
        List of selected tickers
    """
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Momentum Acceleration",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []

    candidates = []

    for ticker in filtered_tickers:
        try:
            data_3m = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                MOM_ACCEL_LOOKBACK_DAYS,
                field_name="close",
                min_rows=5,
            )
            if data_3m is None or data_3m.size < 2:
                continue

            momentum_3m = (data_3m[-1] / data_3m[0] - 1) * 100

            # Skip if 3M momentum is negative
            if momentum_3m <= 0:
                continue

            # Calculate current 1M momentum (last 21 days)
            valid_1m = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                MOM_ACCEL_SHORT_LOOKBACK,
                field_name="close",
                min_rows=10,
            )
            if valid_1m is None or valid_1m.size < 2:
                continue

            momentum_1m_current = (valid_1m[-1] / valid_1m[0] - 1) * 100

            # Calculate previous 1M momentum (21-42 days ago)
            valid_1m_prev = get_cached_values_between(
                price_history_cache,
                ticker,
                current_date - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK * 2),
                current_date - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK),
                field_name="close",
                min_rows=10,
            )
            if valid_1m_prev is None or valid_1m_prev.size < 2:
                continue

            momentum_1m_prev = (valid_1m_prev[-1] / valid_1m_prev[0] - 1) * 100

            # Calculate acceleration
            acceleration = momentum_1m_current - momentum_1m_prev

            # Skip if acceleration is below threshold
            if acceleration < MOM_ACCEL_MIN_ACCELERATION:
                continue

            # Score: 3M momentum weighted by acceleration
            score = momentum_3m * (1 + acceleration / 100)

            candidates.append((ticker, score, momentum_3m, acceleration))

        except Exception:
            continue

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _, _, _ in candidates[:top_n]]

        print(f"   📈 Momentum Acceleration: Selected {len(selected)} stocks")
        for t, score, mom3m, accel in candidates[:min(5, len(candidates))]:
            print(f"      {t}: score={score:.1f}, 3M={mom3m:+.1f}%, accel={accel:+.1f}%")

        return selected

    print(f"   ❌ Momentum Acceleration: No candidates found")
    return []


# ============================================
# 2. CONCENTRATED 3M + VOL FILTER STRATEGY
# ============================================

def select_concentrated_3m_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = None,
    price_history_cache=None,
) -> List[str]:
    """
    Concentrated 3M Strategy:
    - Top performers by 3M momentum
    - Volatility filter (max 40% annualized)
    - Fewer positions (5 instead of 10)

    Returns:
        List of selected tickers
    """
    if top_n is None:
        top_n = PORTFOLIO_SIZE

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Concentrated 3M",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []

    candidates = []

    for ticker in filtered_tickers:
        try:
            valid_close = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                90,
                field_name="close",
                min_rows=5,
            )
            if valid_close is None or valid_close.size < 2:
                continue

            momentum_3m = (valid_close[-1] / valid_close[0] - 1) * 100

            # Skip if 3M momentum is negative
            if momentum_3m <= 0:
                continue

            # Calculate volatility (30-day)
            returns = np.diff(valid_close) / valid_close[:-1]
            if returns.size < 20:
                continue

            volatility = float(np.std(returns, ddof=1) * np.sqrt(252))  # Annualized

            # Skip if volatility exceeds threshold
            if volatility > CONCENTRATED_3M_MAX_VOLATILITY:
                continue

            candidates.append((ticker, momentum_3m, volatility))

        except Exception:
            continue

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _, _ in candidates[:top_n]]

        print(f"   🎯 Concentrated 3M: Selected {len(selected)} stocks (max {top_n})")
        for t, mom3m, vol in candidates[:min(5, len(candidates))]:
            print(f"      {t}: 3M={mom3m:+.1f}%, vol={vol*100:.1f}%")

        return selected

    print(f"   ❌ Concentrated 3M: No candidates found")
    return []


# ============================================
# 3. DUAL MOMENTUM STRATEGY
# ============================================

def select_dual_momentum_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = None,
    price_history_cache=None,
) -> Tuple[List[str], bool]:
    """
    Dual Momentum Strategy (Antonacci style):
    - Absolute momentum: Only buy if 3M return > threshold
    - Relative momentum: Pick top N by 3M return
    - Risk-off: Return empty list if market momentum negative (caller handles cash)

    Returns:
        Tuple of (selected tickers, is_risk_on)
    """
    if top_n is None:
        top_n = PORTFOLIO_SIZE

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Dual Momentum",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return [], False

    candidates = []
    total_momentum = 0.0
    valid_count = 0

    for ticker in filtered_tickers:
        try:
            valid_close = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                DUAL_MOM_LOOKBACK_DAYS,
                field_name="close",
                min_rows=5,
            )
            if valid_close is None or valid_close.size < 2:
                continue

            momentum = (valid_close[-1] / valid_close[0] - 1) * 100

            total_momentum += momentum
            valid_count += 1

            # Absolute momentum filter
            if momentum > DUAL_MOM_ABSOLUTE_THRESHOLD:
                candidates.append((ticker, momentum))

        except Exception:
            continue

    # Calculate market momentum (average of all stocks)
    market_momentum = total_momentum / valid_count if valid_count > 0 else 0

    # Risk-off if market momentum is negative
    is_risk_on = market_momentum > 0

    if not is_risk_on:
        print(f"   ⚠️ Dual Momentum: RISK-OFF mode (market momentum: {market_momentum:.1f}%)")
        return [], False

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in candidates[:top_n]]

        print(f"   📊 Dual Momentum: RISK-ON, selected {len(selected)} stocks")
        print(f"      Market momentum: {market_momentum:+.1f}%")
        for t, mom in candidates[:min(5, len(candidates))]:
            print(f"      {t}: {mom:+.1f}%")

        return selected, True

    print(f"   ❌ Dual Momentum: No candidates with positive absolute momentum")
    return [], True


# ============================================
# 4. TREND FOLLOWING WITH ATR TRAILING STOP
# ============================================

class TrendFollowingATR:
    """
    Trend Following Strategy with ATR-based trailing stops.
    Tracks positions and their trailing stops.
    """

    def __init__(self):
        self.positions = {}  # ticker -> {'entry_price': float, 'peak_price': float, 'atr': float}

    def calculate_atr(self, ticker_data: pd.DataFrame, period: int = None) -> float:
        """Calculate Average True Range."""
        if period is None:
            period = TREND_ATR_PERIOD

        if len(ticker_data) < period + 5:
            return 0.0

        high_low = ticker_data['High'] - ticker_data['Low']
        high_close = abs(ticker_data['High'] - ticker_data['Close'].shift(1))
        low_close = abs(ticker_data['Low'] - ticker_data['Close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0.0

    def is_breakout(self, ticker_data: pd.DataFrame, current_price: float) -> bool:
        """Check if current price is a breakout above N-day high.
        
        Uses the highest CLOSE (not High) of the prior N days as the breakout level.
        This is more appropriate for end-of-day backtesting where we only have
        close prices to act on.
        """
        if len(ticker_data) < TREND_ATR_ENTRY_BREAKOUT:
            return False

        # Use close prices for breakout detection - more realistic for EOD signals
        close_n_days = ticker_data['Close'].iloc[-TREND_ATR_ENTRY_BREAKOUT:-1].max()
        return current_price > close_n_days

    def check_trailing_stop(self, ticker: str, current_price: float) -> bool:
        """Check if trailing stop is hit. Returns True if should sell."""
        if ticker not in self.positions:
            return False

        pos = self.positions[ticker]

        if current_price > pos['peak_price']:
            pos['peak_price'] = current_price

        stop_level = pos['peak_price'] - (pos['atr'] * TREND_ATR_TRAILING_MULT)
        return current_price < stop_level

    def add_position(self, ticker: str, entry_price: float, atr: float):
        """Add a new position."""
        self.positions[ticker] = {
            'entry_price': entry_price,
            'peak_price': entry_price,
            'atr': atr,
        }

    def remove_position(self, ticker: str):
        """Remove a position."""
        if ticker in self.positions:
            del self.positions[ticker]

    def get_positions(self) -> Dict:
        """Get current positions."""
        return self.positions.copy()


def _get_cached_ohlc_frame(
    price_history_cache,
    ticker: str,
    current_date: datetime,
    min_rows: int,
) -> Optional[pd.DataFrame]:
    close = get_cached_history_up_to(
        price_history_cache, ticker, current_date, field_name="close", min_rows=min_rows
    )
    if close is None or len(close) < min_rows:
        return None

    open_prices = get_cached_history_up_to(
        price_history_cache, ticker, current_date, field_name="open", min_rows=min_rows
    )
    high_prices = get_cached_history_up_to(
        price_history_cache, ticker, current_date, field_name="high", min_rows=min_rows
    )
    low_prices = get_cached_history_up_to(
        price_history_cache, ticker, current_date, field_name="low", min_rows=min_rows
    )

    if open_prices is None or high_prices is None or low_prices is None:
        return None

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close,
        }
    )


# Global instance for state persistence
_trend_atr_instance = None

def get_trend_atr_instance() -> TrendFollowingATR:
    """Get or create the global trend ATR instance."""
    global _trend_atr_instance
    if _trend_atr_instance is None:
        _trend_atr_instance = TrendFollowingATR()
    return _trend_atr_instance

def reset_trend_atr_state():
    """Reset the global trend ATR instance."""
    global _trend_atr_instance
    _trend_atr_instance = None


def select_trend_following_atr_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    Trend Following with ATR Trailing Stop Strategy:
    - Identify uptrends using moving averages
    - Enter on breakout above recent high
    - Use ATR-based trailing stop for risk management

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select

    Returns:
        List of selected ticker symbols
    """
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Trend Following ATR",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []

    trend_tracker = get_trend_atr_instance()
    stocks_to_buy = []
    stocks_to_sell = []
    breakout_candidates = []
    positive_momentum_count = 0
    negative_momentum_count = 0
    breakout_signal_count = 0
    insufficient_history_count = 0
    selection_error_count = 0
    selection_error_sample = None

    for ticker in filtered_tickers:
        try:
            close_history = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="close",
                min_rows=1,
            )
            if close_history is None or close_history.size == 0:
                continue
            close_history = np.asarray(close_history, dtype=float)

            current_price = close_history[-1]

            if trend_tracker.check_trailing_stop(ticker, current_price):
                stocks_to_sell.append(ticker)
                print(f"      🛑 {ticker}: Trailing stop hit @ ${current_price:.2f}")
        except Exception:
            continue

    # Remove sold positions
    for ticker in stocks_to_sell:
        trend_tracker.remove_position(ticker)

    # Find new breakout candidates
    for ticker in filtered_tickers:
        try:
            # Skip if already in position
            if ticker in trend_tracker.positions:
                continue

            data = _get_cached_ohlc_frame(
                price_history_cache,
                ticker,
                current_date,
                TREND_ATR_ENTRY_BREAKOUT + 10,
            )
            if data is None:
                insufficient_history_count += 1
                continue
            if len(data) < TREND_ATR_ENTRY_BREAKOUT + 10:
                insufficient_history_count += 1
                continue

            current_price = data['Close'].iloc[-1]

            data_3m = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                TREND_ATR_LOOKBACK_DAYS,
                field_name="close",
                min_rows=5,
            )
            if data_3m is None or data_3m.size < 2:
                insufficient_history_count += 1
                continue
            data_3m = np.asarray(data_3m, dtype=float)

            momentum_3m = (data_3m[-1] / data_3m[0] - 1) * 100

            # Skip if momentum is negative
            if momentum_3m <= 0:
                negative_momentum_count += 1
                continue
            positive_momentum_count += 1

            # Check for breakout
            if trend_tracker.is_breakout(data, current_price):
                breakout_signal_count += 1
                atr = trend_tracker.calculate_atr(data)
                if atr > 0:
                    breakout_candidates.append((ticker, momentum_3m, current_price, atr))

        except Exception as exc:
            selection_error_count += 1
            if selection_error_sample is None:
                selection_error_sample = f"{ticker}: {type(exc).__name__}: {exc}"
            continue

    # Select top breakouts by momentum
    if breakout_candidates:
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)

        # Limit new entries based on available slots
        current_positions = len(trend_tracker.positions)
        available_slots = max(0, top_n - current_positions)

        for ticker, momentum, price, atr in breakout_candidates[:available_slots]:
            stocks_to_buy.append(ticker)
            trend_tracker.add_position(ticker, price, atr)
            print(f"      🚀 {ticker}: Breakout entry @ ${price:.2f}, ATR=${atr:.2f}, 3M={momentum:+.1f}%")

    print(
        "   📈 Trend Following ATR: "
        f"{len(stocks_to_buy)} buys, {len(stocks_to_sell)} sells, {len(trend_tracker.positions)} positions "
        f"(filtered={len(filtered_tickers)}, pos_mom={positive_momentum_count}, neg_mom={negative_momentum_count}, "
        f"breakouts={breakout_signal_count}, qualified={len(breakout_candidates)}, "
        f"insufficient={insufficient_history_count}, errors={selection_error_count})"
    )
    if selection_error_sample:
        print(f"      ⚠️ Trend ATR sample error: {selection_error_sample}")
    if not stocks_to_buy and not stocks_to_sell and not trend_tracker.positions:
        if positive_momentum_count == 0:
            print("      ℹ️ No Trend ATR entries yet. No filtered tickers had positive 3M momentum.")
        elif breakout_signal_count == 0:
            print(
                "      ℹ️ No Trend ATR entries yet. Positive-momentum names were found, "
                "but none broke above the breakout threshold."
            )
        elif not breakout_candidates:
            print("      ℹ️ No Trend ATR entries yet. Breakouts were found, but ATR validation rejected them.")

    return stocks_to_buy, stocks_to_sell


def get_trend_following_current_stocks() -> List[str]:
    """Get list of current trend following positions."""
    trend_tracker = get_trend_atr_instance()
    return list(trend_tracker.positions.keys())


def select_trend_breakout_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = PORTFOLIO_SIZE,
    price_history_cache=None,
) -> List[str]:
    """
    Trend Breakout Strategy:
    - Same entry logic as Trend Following ATR (breakout above recent high)
    - Uses smart_rebalance for position management (no ATR selling)

    Returns:
        List of selected tickers (buy list only)
    """
    from performance_filters import filter_tickers_by_performance
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Trend Breakout",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered_tickers)
    if current_date is None:
        return []

    # Use TrendFollowingATR from this module (not a separate module)
    trend_tracker = TrendFollowingATR()

    breakout_candidates = []

    for ticker in filtered_tickers:
        try:
            data = _get_cached_ohlc_frame(
                price_history_cache,
                ticker,
                current_date,
                TREND_ATR_ENTRY_BREAKOUT + 10,
            )
            if data is None:
                continue
            if len(data) < TREND_ATR_ENTRY_BREAKOUT + 10:
                continue

            current_price = data['Close'].iloc[-1]

            data_3m = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                TREND_ATR_LOOKBACK_DAYS,
                field_name="close",
                min_rows=5,
            )
            if data_3m is None or data_3m.size < 2:
                continue

            momentum_3m = (data_3m[-1] / data_3m[0] - 1) * 100

            if momentum_3m <= 0:
                continue

            if trend_tracker.is_breakout(data, current_price):
                atr = trend_tracker.calculate_atr(data)
                if atr > 0:
                    breakout_candidates.append((ticker, momentum_3m))

        except Exception:
            continue

    stocks_to_buy = []
    if breakout_candidates:
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        stocks_to_buy = [ticker for ticker, _ in breakout_candidates[:top_n]]

    return stocks_to_buy
