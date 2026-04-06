"""
Multi-Horizon Intraday Strategy
Uses the same long_term / medium_term / short_term horizon labels as the
existing ensemble, but powers medium_term and short_term with cached hourly
data when available.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    INVERSE_ETFS,
    MULTI_TIMEFRAMES,
    MULTI_TIMEFRAME_LOOKBACK,
    MULTI_TIMEFRAME_MIN_CONSENSUS,
    MULTI_TIMEFRAME_WEIGHTS,
    NUM_PROCESSES,
    PORTFOLIO_SIZE,
)

_HOURLY_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
_INTRADAY_SELECTION_CONTEXT: Dict[str, object] = {}


def _load_hourly_data_cached(ticker: str) -> Optional[pd.DataFrame]:
    """Load cached hourly data once per ticker from the shared data cache."""
    if ticker in _HOURLY_CACHE:
        return _HOURLY_CACHE[ticker]

    try:
        from data_utils import _RESOLVED_DATA_CACHE_DIR

        cache_file = Path(_RESOLVED_DATA_CACHE_DIR) / f"{ticker}.csv"
        if not cache_file.exists():
            _HOURLY_CACHE[ticker] = None
            return None

        hourly_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if hourly_df.empty:
            _HOURLY_CACHE[ticker] = None
            return None

        if hourly_df.index.tz is None:
            hourly_df.index = hourly_df.index.tz_localize("UTC")
        else:
            hourly_df.index = hourly_df.index.tz_convert("UTC")

        _HOURLY_CACHE[ticker] = hourly_df.sort_index()
        return _HOURLY_CACHE[ticker]
    except Exception:
        _HOURLY_CACHE[ticker] = None
        return None


def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resample_hourly_to_4h(hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bars into 4-hour bars."""
    agg_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in hourly_data.columns:
        agg_map["Volume"] = "sum"

    resampled = hourly_data.resample("4h").agg(agg_map)
    return resampled.dropna(subset=["Close"])


def _clean_close_series(data: pd.DataFrame) -> pd.Series:
    """Return a NaN-free close series for safe window calculations."""
    if "Close" not in data.columns:
        return pd.Series(dtype=float)
    return data["Close"].dropna()


def calculate_daily_momentum(data: pd.DataFrame, current_date: datetime = None) -> float:
    """Calculate long-term daily momentum signal using calendar days."""
    close = _clean_close_series(data)
    if len(close) < 50:
        return 0.0

    if current_date is None:
        current_date = close.index.max()

    data = data[data.index <= current_date]
    close = _clean_close_series(data)
    if len(close) < 2:
        return 0.0

    start_1y = current_date - timedelta(days=365)
    data_1y = data[data.index >= start_1y]
    close_1y = _clean_close_series(data_1y)
    if len(close_1y) >= 50:
        momentum_1y = (close_1y.iloc[-1] / close_1y.iloc[0] - 1) * 100
    else:
        momentum_1y = (close.iloc[-1] / close.iloc[0] - 1) * 100

    start_6m = current_date - timedelta(days=180)
    data_6m = data[data.index >= start_6m]
    close_6m = _clean_close_series(data_6m)
    if len(close_6m) >= 25:
        momentum_6m = (close_6m.iloc[-1] / close_6m.iloc[0] - 1) * 100
    else:
        momentum_6m = momentum_1y

    start_3m = current_date - timedelta(days=90)
    data_3m = data[data.index >= start_3m]
    close_3m = _clean_close_series(data_3m)
    if len(close_3m) >= 10:
        momentum_3m = (close_3m.iloc[-1] / close_3m.iloc[0] - 1) * 100
    else:
        momentum_3m = momentum_1y

    return momentum_1y * 0.5 + momentum_6m * 0.3 + momentum_3m * 0.2


def calculate_medium_term_daily_fallback(data: pd.DataFrame) -> float:
    """Match the existing daily-derived medium-term signal when hourly data is unavailable."""
    close = _clean_close_series(data)
    if len(close) < 20:
        return 0.0

    momentum_30d = (close.iloc[-1] / close.iloc[0] - 1) * 100
    recent_10d = close.tail(10)
    prev_10d = close.tail(20).head(10)
    if len(prev_10d) == 0:
        return 0.0
    recent_avg = recent_10d.mean()
    prev_avg = prev_10d.mean()
    if pd.isna(prev_avg) or prev_avg == 0:
        return 0.0
    trend_signal = ((recent_avg / prev_avg) - 1) * 100

    returns = close.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    vol_adjusted = momentum_30d / (volatility + 1)
    return vol_adjusted * 0.7 + trend_signal * 0.3


def calculate_short_term_daily_fallback(data: pd.DataFrame) -> float:
    """Match the existing daily-derived short-term signal when hourly data is unavailable."""
    close = _clean_close_series(data)
    if len(close) < 5:
        return 0.0

    momentum_7d = (close.iloc[-1] / close.iloc[0] - 1) * 100
    recent_3d = close.tail(3)
    price_change = (recent_3d.iloc[-1] / recent_3d.iloc[0] - 1) * 100

    if "Volume" in data.columns and len(data) >= 5:
        recent_vol = data["Volume"].iloc[-5:].mean()
        avg_vol = data["Volume"].mean()
        volume_factor = min(recent_vol / (avg_vol + 1), 2.0)
    else:
        volume_factor = 1.0

    return (momentum_7d * 0.5 + price_change * 0.3) * volume_factor


def calculate_medium_term_intraday_momentum(data: pd.DataFrame) -> float:
    """Calculate medium-term momentum on 4-hour bars."""
    close = _clean_close_series(data)
    if len(close) < 20:
        return 0.0

    momentum = (close.iloc[-1] / close.iloc[0] - 1) * 100

    if len(close) >= 20:
        recent_bars = close.tail(10)
        prev_bars = close.tail(20).head(10)
        recent_avg = recent_bars.mean()
        prev_avg = prev_bars.mean()
        trend_signal = ((recent_avg / prev_avg) - 1) * 100 if prev_avg else 0.0
    else:
        trend_signal = momentum

    returns = close.pct_change().dropna()
    volatility = returns.std() * np.sqrt(max(len(data), 1)) * 100
    vol_adjusted = momentum / (volatility + 1)
    return vol_adjusted * 0.7 + trend_signal * 0.3


def calculate_short_term_intraday_momentum(data: pd.DataFrame) -> float:
    """Calculate short-term momentum on 1-hour bars."""
    close = _clean_close_series(data)
    if len(close) < 12:
        return 0.0

    lookback_window = close.tail(min(len(close), 35))  # Roughly 5 trading days of hourly bars
    if len(lookback_window) < 2:
        return 0.0
    momentum = (lookback_window.iloc[-1] / lookback_window.iloc[0] - 1) * 100

    recent_window = close.tail(min(len(close), 6))
    if len(recent_window) < 2:
        return 0.0
    recent_change = (recent_window.iloc[-1] / recent_window.iloc[0] - 1) * 100

    if "Volume" in data.columns and len(data) >= 12:
        recent_vol = data["Volume"].iloc[-12:].mean()
        avg_vol = data["Volume"].mean()
        volume_factor = min(recent_vol / (avg_vol + 1), 2.0)
    else:
        volume_factor = 1.0

    return (momentum * 0.5 + recent_change * 0.3) * volume_factor


def calculate_multi_timeframe_intraday_signals(
    ticker: str,
    daily_data: pd.DataFrame,
    current_date: datetime,
    timeframes: List[str] = None,
) -> Dict[str, float]:
    """Calculate long-term from daily data and medium/short horizons from hourly data."""
    if timeframes is None:
        timeframes = MULTI_TIMEFRAMES

    signals: Dict[str, float] = {}
    current_ts = _to_utc_timestamp(current_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    hourly_data = _load_hourly_data_cached(ticker)

    for timeframe in timeframes:
        lookback_days = MULTI_TIMEFRAME_LOOKBACK[timeframe]
        start_date = current_date - timedelta(days=lookback_days)
        daily_slice = daily_data[daily_data.index >= start_date]

        if timeframe == "long_term":
            signals[timeframe] = calculate_daily_momentum(daily_slice, current_date) if len(daily_slice) >= 10 else 0.0
            continue

        if hourly_data is None or hourly_data.empty:
            if timeframe == "medium_term":
                signals[timeframe] = calculate_medium_term_daily_fallback(daily_slice)
            elif timeframe == "short_term":
                signals[timeframe] = calculate_short_term_daily_fallback(daily_slice)
            else:
                signals[timeframe] = 0.0
            continue

        start_ts = current_ts - timedelta(days=lookback_days)
        hourly_slice = hourly_data[(hourly_data.index >= start_ts) & (hourly_data.index <= current_ts)].copy()
        if hourly_slice.empty:
            if timeframe == "medium_term":
                signals[timeframe] = calculate_medium_term_daily_fallback(daily_slice)
            elif timeframe == "short_term":
                signals[timeframe] = calculate_short_term_daily_fallback(daily_slice)
            else:
                signals[timeframe] = 0.0
            continue

        if timeframe == "medium_term":
            resampled_4h = _resample_hourly_to_4h(hourly_slice)
            signals[timeframe] = calculate_medium_term_intraday_momentum(resampled_4h)
        elif timeframe == "short_term":
            signals[timeframe] = calculate_short_term_intraday_momentum(hourly_slice)
        else:
            signals[timeframe] = 0.0

    return signals


def calculate_ensemble_score(signals: Dict[str, float]) -> Tuple[float, bool]:
    ensemble_score = 0.0
    consensus_count = 0

    for timeframe, signal in signals.items():
        weight = MULTI_TIMEFRAME_WEIGHTS[timeframe]
        ensemble_score += signal * weight
        if signal > 0:
            consensus_count += 1

    has_consensus = consensus_count >= MULTI_TIMEFRAME_MIN_CONSENSUS
    return ensemble_score, has_consensus


def _init_multi_timeframe_intraday_worker(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> None:
    """Expose shared read-only selection context to worker processes."""
    global _INTRADAY_SELECTION_CONTEXT
    _INTRADAY_SELECTION_CONTEXT = {
        "ticker_data_grouped": ticker_data_grouped,
        "current_date": current_date,
    }


def _score_multi_timeframe_intraday_ticker_worker(
    ticker: str,
) -> Tuple[str, Optional[Tuple[str, float, Dict[str, float]]], Optional[str]]:
    """Score one ticker for the intraday ensemble in a worker process."""
    context = _INTRADAY_SELECTION_CONTEXT
    ticker_data_grouped = context.get("ticker_data_grouped") or {}
    current_date = context.get("current_date")
    ticker_data = ticker_data_grouped.get(ticker)
    if ticker_data is None or ticker_data.empty or current_date is None:
        return ticker, None, None

    try:
        signals = calculate_multi_timeframe_intraday_signals(ticker, ticker_data, current_date)
        ensemble_score, has_consensus = calculate_ensemble_score(signals)
        if has_consensus:
            return ticker, (ticker, ensemble_score, signals), None
        return ticker, None, None
    except Exception as exc:
        return ticker, None, str(exc)


def select_multi_timeframe_intraday_stocks(
    initial_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = PORTFOLIO_SIZE,
    verbose: bool = True,
) -> List[str]:
    """Select stocks using intraday-enhanced multi-timeframe ensemble signals."""
    tickers_to_use = [ticker for ticker in initial_tickers if ticker not in INVERSE_ETFS]
    stock_scores = []
    error_count = 0
    n_workers = max(1, min(NUM_PROCESSES, len(tickers_to_use))) if tickers_to_use else 1

    if n_workers > 1 and len(tickers_to_use) >= max(32, n_workers * 2):
        if verbose:
            print(f"   🚀 Multi-Horizon Intraday: Scoring {len(tickers_to_use)} tickers with {n_workers} processes")
        with Pool(
            processes=n_workers,
            initializer=_init_multi_timeframe_intraday_worker,
            initargs=(ticker_data_grouped, current_date),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(_score_multi_timeframe_intraday_ticker_worker, tickers_to_use),
                total=len(tickers_to_use),
                desc="   Multi-Horizon Intraday scoring",
                ncols=100,
                unit="ticker",
            ))
        for ticker, scored_item, error_msg in results:
            if scored_item is not None:
                stock_scores.append(scored_item)
            elif error_msg:
                error_count += 1
                if verbose and error_count <= 5:
                    print(f"   ⚠️ Multi-Horizon Intraday ticker error for {ticker}: {error_msg}")
    else:
        for ticker in tickers_to_use:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or ticker_data.empty:
                continue

            try:
                signals = calculate_multi_timeframe_intraday_signals(ticker, ticker_data, current_date)
                ensemble_score, has_consensus = calculate_ensemble_score(signals)
                if has_consensus:
                    stock_scores.append((ticker, ensemble_score, signals))
            except Exception as exc:
                error_count += 1
                if verbose and error_count <= 5:
                    print(f"   ⚠️ Multi-Horizon Intraday ticker error for {ticker}: {exc}")
                continue

    stock_scores.sort(key=lambda item: item[1], reverse=True)
    selected_stocks = [ticker for ticker, _, _ in stock_scores[:top_n]]

    if verbose:
        print(f"   📊 Multi-Horizon Intraday: {len(stock_scores)} candidates with consensus")
        if error_count:
            print(f"   ⚠️ Multi-Horizon Intraday skipped {error_count} ticker(s) due to data errors")
        if len(selected_stocks) > 5:
            print(f"   🎯 Selected: {selected_stocks[:5]}...")
        else:
            print(f"   🎯 Selected: {selected_stocks}")

    return selected_stocks
