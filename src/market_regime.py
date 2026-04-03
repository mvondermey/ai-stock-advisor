"""
Shared market regime helpers.

These helpers are backward-looking only, so they can be used safely by both
market-up filters and inverse hedge logic during backtests and live trading.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_to_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    current_ts = pd.Timestamp(ts)
    if current_ts.tz is None:
        return current_ts.tz_localize("UTC")
    return current_ts.tz_convert("UTC")


def _prepare_price_data(ticker_data: pd.DataFrame) -> pd.DataFrame:
    prepared = ticker_data.copy()
    if prepared.index.duplicated().any():
        prepared = prepared[~prepared.index.duplicated(keep="last")]

    if prepared.index.tz is None:
        prepared.index = prepared.index.tz_localize("UTC")
    else:
        prepared.index = prepared.index.tz_convert("UTC")

    return prepared.sort_index()


def calculate_trailing_return(
    ticker_data: pd.DataFrame,
    current_date: datetime,
    lookback_days: int,
) -> Optional[float]:
    """Calculate realized trailing return in percent over the past N calendar days."""
    try:
        prepared = _prepare_price_data(ticker_data)
        current_ts = _normalize_to_utc(current_date)

        data_until_now = prepared[prepared.index <= current_ts]
        if data_until_now.empty:
            return None

        close_now = data_until_now["Close"].dropna()
        if close_now.empty:
            return None

        current_price = close_now.iloc[-1]

        start_ts = current_ts - timedelta(days=lookback_days)
        lookback_window = data_until_now[data_until_now.index >= start_ts]
        lookback_close = lookback_window["Close"].dropna()
        if len(lookback_close) < 2:
            return None

        start_price = lookback_close.iloc[0]
        if start_price <= 0:
            return None

        return ((current_price - start_price) / start_price) * 100
    except Exception:
        return None


def get_trailing_market_regime(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    lookback_days: int = 5,
    threshold_pct: float = 0.0,
    market_ticker: str = "SPY",
) -> Tuple[Optional[float], bool, str]:
    """
    Return trailing market return, up/down state, and proxy label.

    Uses SPY as the primary market proxy and falls back to the equal-weighted
    average trailing return across available tickers when SPY is unavailable.
    """
    market_data = ticker_data_grouped.get(market_ticker)
    if market_data is not None and len(market_data) > 0:
        market_return = calculate_trailing_return(market_data, current_date, lookback_days)
        if market_return is not None:
            return market_return, market_return > threshold_pct, market_ticker

    returns = []
    for data in ticker_data_grouped.values():
        if data is None or len(data) == 0:
            continue
        trailing_return = calculate_trailing_return(data, current_date, lookback_days)
        if trailing_return is not None:
            returns.append(trailing_return)

    if returns:
        market_return = float(np.mean(returns))
        return market_return, market_return > threshold_pct, "equal_weighted_fallback"

    return None, True, "unavailable"
