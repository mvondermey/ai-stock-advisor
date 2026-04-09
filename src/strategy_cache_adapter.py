from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from parallel_backtest import PriceHistoryCache, build_price_history_cache


def _normalize_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def ensure_price_history_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    price_history_cache: Optional[PriceHistoryCache] = None,
) -> PriceHistoryCache:
    if price_history_cache is not None:
        return price_history_cache
    return build_price_history_cache(ticker_data_grouped)


def resolve_cache_current_date(
    price_history_cache: PriceHistoryCache,
    current_date: Optional[datetime],
    tickers: Optional[Iterable[str]] = None,
) -> Optional[datetime]:
    if current_date is not None:
        return _normalize_timestamp(current_date).to_pydatetime()

    candidate_dates = []
    ticker_iter = tickers if tickers is not None else price_history_cache.date_ns_by_ticker.keys()
    for ticker in ticker_iter:
        date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
        if date_ns is None or date_ns.size == 0:
            continue
        candidate_dates.append(int(date_ns[-1]))

    if not candidate_dates:
        return None

    latest_ns = max(candidate_dates)
    return pd.Timestamp(latest_ns, unit="ns").to_pydatetime()


def _field_map(price_history_cache: PriceHistoryCache, field_name: str) -> Dict[str, np.ndarray]:
    field_attr = f"{field_name}_by_ticker"
    if not hasattr(price_history_cache, field_attr):
        raise AttributeError(f"PriceHistoryCache has no field map '{field_attr}'")
    return getattr(price_history_cache, field_attr)


def get_cached_values_between(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    field_name: str = "close",
    min_rows: int = 2,
) -> Optional[np.ndarray]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    values = _field_map(price_history_cache, field_name).get(ticker)
    if date_ns is None or values is None or date_ns.size == 0 or values.size == 0:
        return None

    start_ns = int(_normalize_timestamp(start_date).value)
    end_ns = int(_normalize_timestamp(end_date).value)
    start_idx = int(np.searchsorted(date_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    window = values[start_idx:end_idx]
    if window.size < max(2, min_rows):
        return None

    if np.issubdtype(window.dtype, np.floating):
        window = window[~np.isnan(window)]
        if window.size < max(2, min_rows):
            return None

    return np.asarray(window, dtype=float)


def get_cached_window(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    current_date: datetime,
    period_days: int,
    field_name: str = "close",
    min_rows: int = 2,
) -> Optional[np.ndarray]:
    return get_cached_values_between(
        price_history_cache,
        ticker,
        _normalize_timestamp(current_date).to_pydatetime() - timedelta(days=period_days),
        current_date,
        field_name=field_name,
        min_rows=min_rows,
    )


def get_cached_history_up_to(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    current_date: datetime,
    field_name: str = "close",
    min_rows: int = 2,
) -> Optional[np.ndarray]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    values = _field_map(price_history_cache, field_name).get(ticker)
    if date_ns is None or values is None or date_ns.size == 0 or values.size == 0:
        return None

    end_ns = int(_normalize_timestamp(current_date).value)
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    history = values[:end_idx]
    if history.size < max(2, min_rows):
        return None

    if np.issubdtype(history.dtype, np.floating):
        history = history[~np.isnan(history)]
        if history.size < max(2, min_rows):
            return None

    return np.asarray(history, dtype=float)


def get_cached_mean_volume(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    current_date: Optional[datetime] = None,
) -> Optional[float]:
    if current_date is None:
        volume_history = _field_map(price_history_cache, "volume").get(ticker)
        if volume_history is None or volume_history.size == 0:
            return None
        valid = volume_history[~np.isnan(volume_history)]
        if valid.size == 0:
            return None
        return float(np.mean(valid))

    volume_history = get_cached_history_up_to(
        price_history_cache,
        ticker,
        current_date,
        field_name="volume",
        min_rows=1,
    )
    if volume_history is None or volume_history.size == 0:
        return None
    return float(np.mean(volume_history))


def get_cached_frame_between(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    field_names: Sequence[str],
    min_rows: int = 2,
) -> Optional[pd.DataFrame]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    if date_ns is None or date_ns.size == 0:
        return None

    start_ns = int(_normalize_timestamp(start_date).value)
    end_ns = int(_normalize_timestamp(end_date).value)
    start_idx = int(np.searchsorted(date_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    if end_idx - start_idx < min_rows:
        return None

    data = {}
    for field_name in field_names:
        values = _field_map(price_history_cache, field_name).get(ticker)
        if values is None or values.size == 0:
            return None
        data[field_name] = values[start_idx:end_idx]

    frame = pd.DataFrame(data)
    if frame.empty:
        return None

    frame = frame.dropna()
    if len(frame) < min_rows:
        return None
    return frame.reset_index(drop=True)


def get_cached_frame_up_to(
    price_history_cache: PriceHistoryCache,
    ticker: str,
    current_date: datetime,
    field_names: Sequence[str],
    min_rows: int = 2,
) -> Optional[pd.DataFrame]:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    if date_ns is None or date_ns.size == 0:
        return None

    end_ns = int(_normalize_timestamp(current_date).value)
    end_idx = int(np.searchsorted(date_ns, end_ns, side="right"))
    if end_idx < min_rows:
        return None

    data = {}
    for field_name in field_names:
        values = _field_map(price_history_cache, field_name).get(ticker)
        if values is None or values.size == 0:
            return None
        data[field_name] = values[:end_idx]

    frame = pd.DataFrame(data)
    if frame.empty:
        return None

    frame = frame.dropna()
    if len(frame) < min_rows:
        return None
    return frame.reset_index(drop=True)
