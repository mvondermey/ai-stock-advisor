"""
Multi-Horizon Intraday Strategy
Uses the same long_term / medium_term / short_term horizon labels as the
existing ensemble, but powers medium_term and short_term with cached hourly
data when available.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import get_context
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from strategy_disk_cache import load_joblib_cache, save_joblib_cache, universe_signature_from_frames
from strategy_cache_adapter import (
    ensure_hourly_history_cache,
    ensure_price_history_cache,
    get_cached_hourly_frame_between,
    resolve_cache_current_date,
)

from config import (
    INVERSE_ETFS,
    MULTI_TIMEFRAMES,
    MULTI_TIMEFRAME_LOOKBACK,
    MULTI_TIMEFRAME_MIN_CONSENSUS,
    MULTI_TIMEFRAME_WEIGHTS,
    NUM_PROCESSES,
    PORTFOLIO_SIZE,
)

_INTRADAY_SELECTION_CONTEXT: Dict[str, object] = {}
_INTRADAY_DAILY_CACHE_MEMORY: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}


def _timestamp_ns(value: datetime) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.tz_localize(None).value)


def _build_intraday_daily_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    iterator = tqdm(
        tickers,
        total=len(tickers),
        desc="   Multi-Horizon Intraday cache build",
        ncols=100,
        unit="ticker",
    )
    for ticker in iterator:
        ticker_data = ticker_data_grouped.get(ticker)
        if ticker_data is None or ticker_data.empty or "Close" not in ticker_data.columns:
            continue

        close_series = pd.to_numeric(ticker_data["Close"], errors="coerce")
        valid_mask = close_series.notna()
        if int(valid_mask.sum()) < 2:
            continue

        filtered_close = close_series.loc[valid_mask]
        date_index = pd.DatetimeIndex(filtered_close.index)
        if date_index.tz is not None:
            date_index = date_index.tz_convert("UTC").tz_localize(None)

        volume_values = None
        if "Volume" in ticker_data.columns:
            volume_series = pd.to_numeric(ticker_data.loc[valid_mask, "Volume"], errors="coerce").fillna(0.0)
            volume_values = volume_series.to_numpy(dtype=float, copy=True)

        cache[ticker] = {
            "date_ns": date_index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=True),
            "close": filtered_close.to_numpy(dtype=float, copy=True),
            "volume": volume_values,
        }

    return cache


def build_multi_timeframe_intraday_daily_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    sorted_tickers = sorted(tickers)
    cache_key_parts = {
        "tickers": sorted_tickers,
        "universe_signature": universe_signature_from_frames(ticker_data_grouped, sorted_tickers),
    }
    cache_key = str(cache_key_parts)
    memory_cached = _INTRADAY_DAILY_CACHE_MEMORY.get(cache_key)
    if memory_cached is not None:
        return memory_cached

    disk_cached = load_joblib_cache(
        "multi_timeframe_intraday/daily_cache",
        cache_key_parts,
        filename="daily_cache.joblib",
    )
    if isinstance(disk_cached, dict):
        _INTRADAY_DAILY_CACHE_MEMORY[cache_key] = disk_cached
        return disk_cached

    built_cache = _build_intraday_daily_cache(ticker_data_grouped, tickers)
    _INTRADAY_DAILY_CACHE_MEMORY[cache_key] = built_cache
    save_joblib_cache(
        "multi_timeframe_intraday/daily_cache",
        cache_key_parts,
        built_cache,
        filename="daily_cache.joblib",
    )
    return built_cache


def _daily_slice_from_cache(
    cache_entry: Dict[str, np.ndarray],
    current_date: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    date_ns = cache_entry["date_ns"]
    end_ns = _timestamp_ns(current_date)
    start_ns = _timestamp_ns(current_date - timedelta(days=lookback_days))
    start_idx = int(np.searchsorted(date_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    if end_idx <= start_idx:
        return pd.DataFrame()

    data = {"Close": cache_entry["close"][start_idx:end_idx]}
    volume_values = cache_entry.get("volume")
    if volume_values is not None:
        data["Volume"] = volume_values[start_idx:end_idx]

    index = pd.to_datetime(date_ns[start_idx:end_idx], utc=True)
    return pd.DataFrame(data, index=index)


def _get_daily_cache_entry_from_price_history(price_history_cache, ticker: str) -> Optional[Dict[str, np.ndarray]]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    close_values = price_history_cache.close_by_ticker.get(ticker)
    if date_ns is None or close_values is None or date_ns.size < 2 or close_values.size < 2:
        return None
    return {
        "date_ns": date_ns,
        "close": close_values,
        "volume": price_history_cache.volume_by_ticker.get(ticker),
    }


def _calculate_multi_timeframe_intraday_signals_from_cache(
    ticker: str,
    cache_entry: Dict[str, np.ndarray],
    current_date: datetime,
    timeframes: List[str] = None,
    hourly_history_cache=None,
) -> Dict[str, float]:
    if timeframes is None:
        timeframes = MULTI_TIMEFRAMES

    signals: Dict[str, float] = {}
    current_ts = _to_utc_timestamp(current_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    max_lookback_days = max(
        (MULTI_TIMEFRAME_LOOKBACK[timeframe] for timeframe in timeframes if timeframe != "long_term"),
        default=0,
    )
    hourly_data = _load_hourly_data_cached(
        ticker,
        current_ts.to_pydatetime() - timedelta(days=max_lookback_days),
        current_ts.to_pydatetime(),
        hourly_history_cache=hourly_history_cache,
    )

    for timeframe in timeframes:
        lookback_days = MULTI_TIMEFRAME_LOOKBACK[timeframe]
        daily_slice = _daily_slice_from_cache(cache_entry, current_date, lookback_days)

        if timeframe == "long_term":
            signals[timeframe] = calculate_daily_momentum(daily_slice, current_date) if len(daily_slice) >= 10 else 0.0
            continue

        if hourly_data is None or hourly_data.empty:
            signals[timeframe] = 0.0
            continue

        start_ts = current_ts - timedelta(days=lookback_days)
        hourly_slice = hourly_data[(hourly_data.index >= start_ts) & (hourly_data.index <= current_ts)].copy()
        if hourly_slice.empty:
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


def _load_hourly_data_cached(
    ticker: str,
    start: datetime,
    end: datetime,
    hourly_history_cache=None,
) -> Optional[pd.DataFrame]:
    """Load cached hourly data from the shared hourly history cache."""
    try:
        hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
        hourly_df = get_cached_hourly_frame_between(
            hourly_history_cache,
            ticker,
            start,
            end,
            field_names=("open", "high", "low", "close", "volume"),
            min_rows=2,
        )
        if hourly_df is None or hourly_df.empty:
            return None
        return hourly_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    except Exception:
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
    else:
        current_date = _to_utc_timestamp(current_date)

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
    hourly_history_cache=None,
) -> Dict[str, float]:
    """Calculate long-term from daily data and medium/short horizons from hourly data."""
    if timeframes is None:
        timeframes = MULTI_TIMEFRAMES

    signals: Dict[str, float] = {}
    current_ts = _to_utc_timestamp(current_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    max_lookback_days = max(
        (MULTI_TIMEFRAME_LOOKBACK[timeframe] for timeframe in timeframes if timeframe != "long_term"),
        default=0,
    )
    hourly_data = _load_hourly_data_cached(
        ticker,
        current_ts.to_pydatetime() - timedelta(days=max_lookback_days),
        current_ts.to_pydatetime(),
        hourly_history_cache=hourly_history_cache,
    )

    for timeframe in timeframes:
        lookback_days = MULTI_TIMEFRAME_LOOKBACK[timeframe]
        start_date = current_ts - timedelta(days=lookback_days)
        daily_slice = daily_data[daily_data.index >= start_date]

        if timeframe == "long_term":
            signals[timeframe] = calculate_daily_momentum(daily_slice, current_date) if len(daily_slice) >= 10 else 0.0
            continue

        if hourly_data is None or hourly_data.empty:
            signals[timeframe] = 0.0
            continue

        start_ts = current_ts - timedelta(days=lookback_days)
        hourly_slice = hourly_data[(hourly_data.index >= start_ts) & (hourly_data.index <= current_ts)].copy()
        if hourly_slice.empty:
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
    daily_cache: Optional[Dict[str, Dict[str, np.ndarray]]],
    current_date: datetime,
    price_history_cache=None,
    hourly_history_cache=None,
) -> None:
    global _INTRADAY_SELECTION_CONTEXT
    _INTRADAY_SELECTION_CONTEXT = {
        "daily_cache": daily_cache,
        "current_date": current_date,
        "price_history_cache": price_history_cache,
        "hourly_history_cache": hourly_history_cache,
    }


def _score_multi_timeframe_intraday_ticker_worker(
    ticker: str,
) -> Tuple[str, Optional[Tuple[str, float, Dict[str, float]]], Optional[str]]:
    context = _INTRADAY_SELECTION_CONTEXT
    daily_cache = context.get("daily_cache") or {}
    current_date = context.get("current_date")
    price_history_cache = context.get("price_history_cache")
    hourly_history_cache = context.get("hourly_history_cache")
    cache_entry = daily_cache.get(ticker)
    if cache_entry is None and price_history_cache is not None:
        cache_entry = _get_daily_cache_entry_from_price_history(price_history_cache, ticker)
    if cache_entry is None or current_date is None:
        return ticker, None, None

    try:
        signals = _calculate_multi_timeframe_intraday_signals_from_cache(
            ticker,
            cache_entry,
            current_date,
            hourly_history_cache=hourly_history_cache,
        )
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
    daily_cache: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    price_history_cache=None,
    hourly_history_cache=None,
) -> List[str]:
    """Select stocks using intraday-enhanced multi-timeframe ensemble signals."""
    tickers_to_use = [ticker for ticker in initial_tickers if ticker not in INVERSE_ETFS]
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
    current_date = resolve_cache_current_date(price_history_cache, current_date, tickers_to_use)
    if current_date is None:
        return []
    if daily_cache is None:
        cached_tickers = [
            ticker for ticker in tickers_to_use
            if _get_daily_cache_entry_from_price_history(price_history_cache, ticker) is not None
        ]
    else:
        cached_tickers = [ticker for ticker in tickers_to_use if ticker in daily_cache]
    stock_scores = []
    error_count = 0
    n_workers = max(1, min(NUM_PROCESSES, len(cached_tickers))) if cached_tickers else 1

    if n_workers > 1 and len(cached_tickers) >= max(32, n_workers * 2):
        n_workers = min(n_workers, 4)
        if os.name != "nt":
            if verbose:
                print(f"   🚀 Multi-Horizon Intraday: Scoring {len(cached_tickers)} tickers with {n_workers} fork workers")
            chunksize = max(1, len(cached_tickers) // (n_workers * 4))
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=_init_multi_timeframe_intraday_worker,
                initargs=(daily_cache, current_date, price_history_cache, hourly_history_cache),
            ) as pool:
                results = pool.imap_unordered(
                    _score_multi_timeframe_intraday_ticker_worker,
                    cached_tickers,
                    chunksize=chunksize,
                )
                for ticker, scored_item, error_msg in tqdm(
                    results,
                    total=len(cached_tickers),
                    desc="   Multi-Horizon Intraday scoring",
                    ncols=100,
                    unit="ticker",
                ):
                    if scored_item is not None:
                        stock_scores.append(scored_item)
                    elif error_msg:
                        error_count += 1
                        if verbose and error_count <= 5:
                            print(f"   ⚠️ Multi-Horizon Intraday ticker error for {ticker}: {error_msg}")
        else:
            if verbose:
                print(f"   🚀 Multi-Horizon Intraday: Scoring {len(cached_tickers)} tickers with {n_workers} threads")
            _init_multi_timeframe_intraday_worker(
                daily_cache,
                current_date,
                price_history_cache=price_history_cache,
                hourly_history_cache=hourly_history_cache,
            )
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = executor.map(_score_multi_timeframe_intraday_ticker_worker, cached_tickers)
                for ticker, scored_item, error_msg in tqdm(
                    results,
                    total=len(cached_tickers),
                    desc="   Multi-Horizon Intraday scoring",
                    ncols=100,
                    unit="ticker",
                ):
                    if scored_item is not None:
                        stock_scores.append(scored_item)
                    elif error_msg:
                        error_count += 1
                        if verbose and error_count <= 5:
                            print(f"   ⚠️ Multi-Horizon Intraday ticker error for {ticker}: {error_msg}")
    else:
        for ticker in cached_tickers:
            cache_entry = (
                daily_cache.get(ticker)
                if daily_cache is not None
                else _get_daily_cache_entry_from_price_history(price_history_cache, ticker)
            )
            if cache_entry is None:
                continue

            try:
                signals = _calculate_multi_timeframe_intraday_signals_from_cache(
                    ticker,
                    cache_entry,
                    current_date,
                    hourly_history_cache=hourly_history_cache,
                )
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
