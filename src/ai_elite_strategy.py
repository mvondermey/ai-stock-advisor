"""
AI Elite Strategy: ML-powered scoring of elite stock candidates

Uses machine learning to learn optimal scoring from:
- 6M momentum
- Volatility (risk)
- Volume (liquidity)
- Dip score (1Y/3M ratio)
- 1Y performance
- 3M performance

The ML model learns which combinations of these features predict future outperformance,
discovering non-linear relationships that fixed formulas miss.
"""

import json
import hashlib
import tempfile

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from multiprocessing import get_context
import os
import time
from pathlib import Path

import joblib
from tqdm import tqdm

from model_training_safety import restore_native_model_artifacts
from strategy_cache_adapter import (
    ensure_hourly_history_cache,
    ensure_price_history_cache,
    get_cached_hourly_frame_between,
    get_cached_hourly_frame_up_to,
)
from strategy_disk_cache import get_cache_dir, universe_signature_from_frames

_AI_ELITE_SHARED_MODEL_CACHE: Dict[Tuple[str, Optional[str]], any] = {}
_AI_ELITE_PREDICT_CONTEXT: Dict[str, object] = {}
_AI_ELITE_FILE_CACHE_CONTEXT_PATH: Optional[str] = None
_AI_ELITE_FILE_CACHE_CONTEXT: Optional[Dict[str, object]] = None

AI_ELITE_FEATURE_NAMES: Tuple[str, ...] = (
    'perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
    'overnight_gap', 'intraday_range', 'last_hour_momentum',
    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
    'volume_ratio', 'rsi_14',
    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd',
    'mom_5d', 'mom_10d', 'mom_20d', 'mom_40d',
    'volatility_10d', 'volatility_20d', 'drawdown_20d', 'atr_pct_14d', 'volume_ratio_20_60',
    'price_vs_sg_short', 'price_vs_sg_long',
    'sg_short_slope', 'sg_long_slope',
    'sg_short_curvature', 'sg_long_curvature',
    'sg_trend_spread', 'sg_trend_stability', 'sg_long_stability',
    'sg_regression_slope', 'sg_residual_vol_20d',
    'market_return_5d', 'market_return_20d', 'market_return_60d',
    'market_volatility_20d', 'market_breadth_20d', 'market_breadth_60d',
    'rel_strength_20d', 'rel_strength_60d',
)
AI_ELITE_INTRADAY_FEATURE_NAMES: Tuple[str, ...] = (
    'overnight_gap',
    'intraday_range',
    'last_hour_momentum',
)
AI_ELITE_DAILY_FEATURE_NAMES: Tuple[str, ...] = tuple(
    feature_name for feature_name in AI_ELITE_FEATURE_NAMES
    if feature_name not in AI_ELITE_INTRADAY_FEATURE_NAMES
)
AI_ELITE_CACHE_NUMERIC_DTYPE = "float64"
AI_ELITE_MARKET_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    'market_return_5d',
    'market_return_20d',
    'market_return_60d',
    'market_volatility_20d',
    'market_breadth_20d',
    'market_breadth_60d',
)


def _build_feature_frame(features: Dict[str, float], feature_cols: List[str]) -> pd.DataFrame:
    """Build a single-row DataFrame so sklearn/lightgbm keep feature-name alignment."""
    return pd.DataFrame(
        [[features.get(col, 0.0) for col in feature_cols]],
        columns=feature_cols,
        dtype=np.float64,
    )


def _create_prediction_timing_stats() -> Dict[str, object]:
    return {
        "lock": Lock(),
        "hourly_seconds": 0.0,
        "hourly_count": 0,
        "feature_seconds": 0.0,
        "feature_count": 0,
        "model_seconds": 0.0,
        "model_count": 0,
    }


def _record_prediction_timing(stats, phase: str, elapsed_seconds: float) -> None:
    if stats is None:
        return
    with stats["lock"]:
        stats[f"{phase}_seconds"] += float(elapsed_seconds)
        stats[f"{phase}_count"] += 1


def _print_prediction_timing(label: str, stats) -> None:
    if stats is None:
        return

    phase_labels = (
        ("hourly", "hourly load"),
        ("feature", "feature extract"),
        ("model", "model predict"),
    )
    segments = []
    for phase_key, phase_label in phase_labels:
        count = int(stats[f"{phase_key}_count"])
        if count <= 0:
            continue
        total_seconds = float(stats[f"{phase_key}_seconds"])
        avg_ms = (total_seconds / count) * 1000.0
        segments.append(f"{phase_label}={total_seconds:.1f}s ({avg_ms:.1f}ms x {count})")

    if segments:
        print(f"   ⏱️ {label} timing: " + ", ".join(segments))


def _normalize_ai_elite_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _normalize_ai_elite_current_date(data: pd.DataFrame, current_date: datetime) -> pd.Timestamp:
    current_ts = _normalize_ai_elite_timestamp(current_date)
    if data.index.tz is None:
        data_index = data.index.tz_localize('UTC')
    else:
        data_index = data.index.tz_convert('UTC')
    return current_ts.normalize() if (data_index.normalize() == current_ts.normalize()).any() else current_ts


def _ai_elite_safe_pct_change(current_value: float, base_value: float) -> float:
    if base_value <= 0:
        return 0.0
    return ((current_value - base_value) / base_value) * 100.0


def _ai_elite_odd_window(length: int, cap: int) -> int:
    if length < 3:
        return 0
    window = min(length, cap)
    if window % 2 == 0:
        window -= 1
    return window if window >= 3 else 0


def _ai_elite_pct_change_array(values: np.ndarray, periods: int) -> np.ndarray:
    result = np.zeros(values.shape, dtype=float)
    if periods <= 0 or values.size <= periods:
        return result
    base = values[:-periods]
    current = values[periods:]
    valid = base > 0
    if np.any(valid):
        result_slice = np.zeros(current.shape, dtype=float)
        result_slice[valid] = ((current[valid] - base[valid]) / base[valid]) * 100.0
        result[periods:] = result_slice
    return result


def _calculate_ai_elite_market_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    current_ts = _normalize_ai_elite_timestamp(current_date).normalize()
    universe_returns_20d: List[float] = []
    universe_returns_60d: List[float] = []
    proxy_series: List[pd.Series] = []

    for data in ticker_data_grouped.values():
        if data is None or len(data) == 0 or 'Close' not in data.columns:
            continue
        try:
            data_ts = data
            if data_ts.index.tz is None:
                data_ts = data_ts.copy()
                data_ts.index = data_ts.index.tz_localize('UTC')
            else:
                data_ts = data_ts.copy()
                data_ts.index = data_ts.index.tz_convert('UTC')

            hist = data_ts.loc[:current_ts]
            close = pd.to_numeric(hist['Close'], errors='coerce').dropna()
            if len(close) < 61:
                continue

            proxy_series.append(close)
            universe_returns_20d.append(_ai_elite_safe_pct_change(float(close.iloc[-1]), float(close.iloc[-21])))
            universe_returns_60d.append(_ai_elite_safe_pct_change(float(close.iloc[-1]), float(close.iloc[-61])))
        except Exception:
            continue

    if not proxy_series:
        return {
            'market_return_5d': 0.0,
            'market_return_20d': 0.0,
            'market_return_60d': 0.0,
            'market_volatility_20d': 0.0,
            'market_breadth_20d': 0.5,
            'market_breadth_60d': 0.5,
        }

    min_len = min(len(series) for series in proxy_series)
    aligned_proxy = np.vstack([series.iloc[-min_len:].to_numpy(dtype=float, copy=True) for series in proxy_series])
    market_proxy = np.nanmean(aligned_proxy, axis=0)
    market_proxy_series = pd.Series(market_proxy)
    market_returns = market_proxy_series.pct_change().dropna()

    market_return_5d = 0.0
    if min_len >= 6 and market_proxy[-6] > 0:
        market_return_5d = _ai_elite_safe_pct_change(float(market_proxy[-1]), float(market_proxy[-6]))

    market_return_20d = 0.0
    if min_len >= 21 and market_proxy[-21] > 0:
        market_return_20d = _ai_elite_safe_pct_change(float(market_proxy[-1]), float(market_proxy[-21]))

    market_return_60d = 0.0
    if min_len >= 61 and market_proxy[-61] > 0:
        market_return_60d = _ai_elite_safe_pct_change(float(market_proxy[-1]), float(market_proxy[-61]))

    market_volatility_20d = 0.0
    if len(market_returns) >= 20:
        market_volatility_20d = float(np.std(market_returns.iloc[-20:].to_numpy(dtype=float, copy=True)) * np.sqrt(252) * 100.0)

    breadth_20 = np.asarray(universe_returns_20d[:250], dtype=float)
    breadth_60 = np.asarray(universe_returns_60d[:250], dtype=float)
    market_breadth_20d = float(np.mean(breadth_20 > 0)) if breadth_20.size > 0 else 0.5
    market_breadth_60d = float(np.mean(breadth_60 > 0)) if breadth_60.size > 0 else 0.5

    return {
        'market_return_5d': market_return_5d,
        'market_return_20d': market_return_20d,
        'market_return_60d': market_return_60d,
        'market_volatility_20d': market_volatility_20d,
        'market_breadth_20d': market_breadth_20d,
        'market_breadth_60d': market_breadth_60d,
    }


def precompute_ai_elite_market_context_map(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    context_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    start_ts = _normalize_ai_elite_timestamp(start_date).normalize()
    end_ts = _normalize_ai_elite_timestamp(end_date).normalize()
    total_days = max(0, (end_ts - start_ts).days) + 1
    current_date = start_ts
    for _ in tqdm(
        range(total_days),
        desc="AI Elite market context",
        total=total_days,
        ncols=100,
    ):
        context_map[current_date] = _calculate_ai_elite_market_context(ticker_data_grouped, current_date)
        current_date += timedelta(days=1)
    return context_map


def _build_ai_elite_savgol_feature_arrays(
    log_close: np.ndarray,
) -> Dict[str, np.ndarray]:
    size = log_close.size
    arrays = {
        'price_vs_sg_short': np.zeros(size, dtype=float),
        'price_vs_sg_long': np.zeros(size, dtype=float),
        'sg_short_slope': np.zeros(size, dtype=float),
        'sg_long_slope': np.zeros(size, dtype=float),
        'sg_short_curvature': np.zeros(size, dtype=float),
        'sg_long_curvature': np.zeros(size, dtype=float),
        'sg_trend_spread': np.zeros(size, dtype=float),
        'sg_trend_stability': np.zeros(size, dtype=float),
        'sg_long_stability': np.zeros(size, dtype=float),
        'sg_regression_slope': np.zeros(size, dtype=float),
        'sg_residual_vol_20d': np.zeros(size, dtype=float),
    }
    if size < 20:
        return arrays

    for idx in range(19, size):
        prefix = log_close[:idx + 1]
        short_window = _ai_elite_odd_window(idx + 1, 9)
        long_window = _ai_elite_odd_window(idx + 1, 21)
        if short_window == 0 or long_window == 0:
            continue

        short_poly = 2 if short_window >= 5 else 1
        long_poly = 3 if long_window >= 7 else 2
        if short_window <= short_poly or long_window <= long_poly:
            continue

        sg_short = savgol_filter(prefix, window_length=short_window, polyorder=short_poly, mode='interp')
        sg_long = savgol_filter(prefix, window_length=long_window, polyorder=long_poly, mode='interp')

        arrays['price_vs_sg_short'][idx] = (prefix[-1] - sg_short[-1]) * 100.0
        arrays['price_vs_sg_long'][idx] = (prefix[-1] - sg_long[-1]) * 100.0
        arrays['sg_trend_spread'][idx] = (sg_short[-1] - sg_long[-1]) * 100.0
        if len(sg_short) >= 2:
            arrays['sg_short_slope'][idx] = (sg_short[-1] - sg_short[-2]) * 100.0
        if len(sg_long) >= 2:
            arrays['sg_long_slope'][idx] = (sg_long[-1] - sg_long[-2]) * 100.0
        if len(sg_short) >= 3:
            arrays['sg_short_curvature'][idx] = (sg_short[-1] - 2 * sg_short[-2] + sg_short[-3]) * 10000.0
        if len(sg_long) >= 3:
            arrays['sg_long_curvature'][idx] = (sg_long[-1] - 2 * sg_long[-2] + sg_long[-3]) * 10000.0

        short_tail = sg_short[-10:] if len(sg_short) >= 10 else sg_short
        long_tail = sg_long[-10:] if len(sg_long) >= 10 else sg_long
        short_diffs = np.diff(short_tail)
        long_diffs = np.diff(long_tail)
        arrays['sg_trend_stability'][idx] = float(np.mean(short_diffs > 0)) if short_diffs.size > 0 else 0.5
        arrays['sg_long_stability'][idx] = float(np.mean(long_diffs > 0)) if long_diffs.size > 0 else 0.5

        reg_len = min(10, len(sg_long))
        if reg_len >= 2:
            x = np.arange(reg_len, dtype=float)
            arrays['sg_regression_slope'][idx] = float(np.polyfit(x, sg_long[-reg_len:], 1)[0] * 100.0)

        residual = prefix - sg_long
        if residual.size >= 20:
            arrays['sg_residual_vol_20d'][idx] = float(np.std(residual[-20:]) * 100.0)

    return arrays


def _ai_elite_datetime_from_ordinal(ordinal: int, reference_date: datetime) -> datetime:
    current_date = datetime.fromordinal(ordinal)
    if reference_date.tzinfo is not None:
        current_date = current_date.replace(tzinfo=reference_date.tzinfo)
    return current_date


def _ai_elite_context_metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "context.json"


def _save_ai_elite_context_metadata(context: Dict[str, object]) -> None:
    temp_dir = context.get("temp_dir")
    if not temp_dir:
        return

    metadata = {
        "all_tickers": list(context.get("all_tickers") or []),
        "start_ordinal": int(context.get("start_ordinal", 0)),
        "n_dates": int(context.get("n_dates", 0)),
        "daily_feature_path": str(context.get("daily_feature_path", "")),
        "intraday_feature_path": str(context.get("intraday_feature_path", "")),
        "forward_path": str(context.get("forward_path", "")),
        "valid_mask_path": str(context.get("valid_mask_path", "")),
        "intraday_coverage_path": str(context.get("intraday_coverage_path", "")),
        "date_index_path": str(context.get("date_index_path", "")),
        "ticker_index_path": str(context.get("ticker_index_path", "")),
        "forward_days": int(context.get("forward_days", 0)),
        "latest_allowed_ordinal": int(context.get("latest_allowed_ordinal", 0)),
        "daily_signature": str(context.get("daily_signature", "")),
        "hourly_signature": str(context.get("hourly_signature", "")),
        "persistent": bool(context.get("persistent", True)),
        "daily_feature_names": list(context.get("daily_feature_names") or AI_ELITE_DAILY_FEATURE_NAMES),
        "intraday_feature_names": list(context.get("intraday_feature_names") or AI_ELITE_INTRADAY_FEATURE_NAMES),
        "feature_names": list(context.get("feature_names") or AI_ELITE_FEATURE_NAMES),
        "numeric_dtype": str(context.get("numeric_dtype") or AI_ELITE_CACHE_NUMERIC_DTYPE),
        "temp_dir": str(temp_dir),
    }
    _ai_elite_context_metadata_path(Path(str(temp_dir))).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_ai_elite_context_metadata(cache_dir: Path) -> Optional[Dict[str, object]]:
    metadata_path = _ai_elite_context_metadata_path(cache_dir)
    if not metadata_path.exists():
        return None
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None
    return loaded


def _valid_ai_elite_tickers(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
) -> List[str]:
    valid_tickers: List[str] = []
    for ticker in all_tickers:
        ticker_df = ticker_data_grouped.get(ticker)
        if ticker_df is None or len(ticker_df) == 0 or 'Close' not in ticker_df.columns:
            continue
        valid_tickers.append(ticker)
    return valid_tickers


def _ai_elite_hourly_universe_signature(tickers: List[str]) -> str:
    try:
        from data_utils import _RESOLVED_DATA_CACHE_DIR
    except Exception:
        return "missing"

    signature_payload = []
    for ticker in sorted(tickers):
        cache_file = _RESOLVED_DATA_CACHE_DIR / f"{ticker}.csv"
        if not cache_file.exists():
            signature_payload.append({"ticker": ticker, "exists": False})
            continue
        try:
            stat_result = cache_file.stat()
            signature_payload.append(
                {
                    "ticker": ticker,
                    "exists": True,
                    "mtime_ns": int(stat_result.st_mtime_ns),
                    "size": int(stat_result.st_size),
                }
            )
        except OSError:
            signature_payload.append({"ticker": ticker, "exists": False, "error": True})

    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ai_elite_context_cache_key(
    valid_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    daily_signature: str,
    hourly_signature: str,
) -> Dict[str, object]:
    start_ts = _normalize_ai_elite_timestamp(train_start_date)
    end_ts = _normalize_ai_elite_timestamp(train_end_date)
    return {
        "tickers": list(valid_tickers),
        "train_start": start_ts.isoformat(),
        "train_end": end_ts.isoformat(),
        "forward_days": int(forward_days),
        "daily_signature": daily_signature,
        "hourly_signature": hourly_signature,
        "feature_names": list(AI_ELITE_FEATURE_NAMES),
        "numeric_dtype": AI_ELITE_CACHE_NUMERIC_DTYPE,
    }


def _init_ai_elite_file_cache_worker(
    context_path: Optional[str] = None,
    context_data: Optional[Dict[str, object]] = None,
) -> None:
    global _AI_ELITE_FILE_CACHE_CONTEXT_PATH, _AI_ELITE_FILE_CACHE_CONTEXT
    _AI_ELITE_FILE_CACHE_CONTEXT_PATH = context_path
    _AI_ELITE_FILE_CACHE_CONTEXT = context_data


def _ensure_ai_elite_feature_cache_context(
    context: Optional[Dict[str, object]],
    mmap_mode: str = "r",
) -> Optional[Dict[str, object]]:
    if not isinstance(context, dict):
        return None

    if "_daily_feature_array" not in context and context.get("daily_feature_path"):
        context["_daily_feature_array"] = np.load(str(context["daily_feature_path"]), mmap_mode=mmap_mode)
    if "_intraday_feature_array" not in context and context.get("intraday_feature_path"):
        context["_intraday_feature_array"] = np.load(str(context["intraday_feature_path"]), mmap_mode=mmap_mode)
    if "_forward_array" not in context and context.get("forward_path"):
        context["_forward_array"] = np.load(str(context["forward_path"]), mmap_mode=mmap_mode)
    if "_valid_mask_array" not in context and context.get("valid_mask_path"):
        context["_valid_mask_array"] = np.load(str(context["valid_mask_path"]), mmap_mode=mmap_mode)
    if "_intraday_coverage_array" not in context and context.get("intraday_coverage_path"):
        context["_intraday_coverage_array"] = np.load(str(context["intraday_coverage_path"]), mmap_mode=mmap_mode)
    if "_date_index_array" not in context and context.get("date_index_path"):
        context["_date_index_array"] = np.load(str(context["date_index_path"]), mmap_mode=mmap_mode)
    if "ticker_to_idx" not in context:
        context["ticker_to_idx"] = {
            ticker: idx for idx, ticker in enumerate(context.get("all_tickers") or [])
        }
    return context


def _get_ai_elite_file_cache_context(mmap_mode: str = "r+") -> Dict[str, object]:
    global _AI_ELITE_FILE_CACHE_CONTEXT
    if _AI_ELITE_FILE_CACHE_CONTEXT is None:
        if not _AI_ELITE_FILE_CACHE_CONTEXT_PATH:
            raise ValueError("AI Elite file cache context is not initialized")
        _AI_ELITE_FILE_CACHE_CONTEXT = joblib.load(_AI_ELITE_FILE_CACHE_CONTEXT_PATH)

    ensured = _ensure_ai_elite_feature_cache_context(_AI_ELITE_FILE_CACHE_CONTEXT, mmap_mode=mmap_mode)
    if ensured is None:
        raise ValueError("AI Elite file cache context could not be loaded")
    return ensured


def _ai_elite_daily_vector_from_features(features: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [features.get(feature_name, 0.0) for feature_name in AI_ELITE_DAILY_FEATURE_NAMES],
        dtype=np.float64,
    )


def _ai_elite_intraday_vector_from_features(features: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [features.get(feature_name, 0.0) for feature_name in AI_ELITE_INTRADAY_FEATURE_NAMES],
        dtype=np.float64,
    )


def _ai_elite_feature_dict_from_vectors(
    daily_vector: np.ndarray,
    intraday_vector: np.ndarray,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for idx, feature_name in enumerate(AI_ELITE_DAILY_FEATURE_NAMES):
        features[feature_name] = float(daily_vector[idx])
    for idx, feature_name in enumerate(AI_ELITE_INTRADAY_FEATURE_NAMES):
        features[feature_name] = float(intraday_vector[idx])
    return features


def clone_ai_elite_feature_cache_context(
    context: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not isinstance(context, dict):
        return None
    return {
        key: value
        for key, value in context.items()
        if not str(key).startswith("_")
    }


def _init_ai_elite_predict_worker(context: Dict[str, object]) -> None:
    global _AI_ELITE_PREDICT_CONTEXT
    _AI_ELITE_PREDICT_CONTEXT = context


def _predict_ticker_from_context_worker(ticker: str):
    context = _AI_ELITE_PREDICT_CONTEXT
    ticker_data_grouped = context.get("ticker_data_grouped") or {}
    per_ticker_models = context.get("per_ticker_models") or {}
    shared_base_model = context.get("shared_base_model")
    ticker_model = per_ticker_models.get(ticker, shared_base_model) if isinstance(per_ticker_models, dict) else None
    return _predict_ticker_worker(
        (
            ticker,
            ticker_data_grouped.get(ticker),
            context.get("current_date"),
            ticker_model,
            context.get("price_history_cache"),
            context.get("hourly_history_cache"),
            context.get("feature_cache_context"),
            context.get("market_context_map"),
            None,
        )
    )


def _predict_ticker_ensemble_from_context_worker(ticker: str):
    context = _AI_ELITE_PREDICT_CONTEXT
    ticker_data_grouped = context.get("ticker_data_grouped") or {}
    per_ticker_models = context.get("per_ticker_models") or {}
    ticker_model = per_ticker_models.get(ticker) if isinstance(per_ticker_models, dict) else None
    return _predict_ticker_ensemble_worker(
        (
            ticker,
            ticker_data_grouped.get(ticker),
            context.get("current_date"),
            ticker_model,
            context.get("price_history_cache"),
            context.get("hourly_history_cache"),
            context.get("feature_cache_context"),
            context.get("market_context_map"),
            None,
        )
    )


def _predict_ticker_rank_ensemble_from_context_worker(ticker: str):
    context = _AI_ELITE_PREDICT_CONTEXT
    ticker_data_grouped = context.get("ticker_data_grouped") or {}
    per_ticker_models = context.get("per_ticker_models") or {}
    ticker_model = per_ticker_models.get(ticker) if isinstance(per_ticker_models, dict) else None
    return _predict_ticker_rank_ensemble_worker(
        (
            ticker,
            ticker_data_grouped.get(ticker),
            context.get("current_date"),
            ticker_model,
            context.get("selected_model_names") or [],
            context.get("price_history_cache"),
            context.get("hourly_history_cache"),
            context.get("feature_cache_context"),
            context.get("market_context_map"),
            None,
        )
    )


def select_ai_elite_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None,
    shared_model_path: Optional[str] = None,
    shared_model_token: Optional[str] = None,
    max_prediction_workers: Optional[int] = None,
    price_history_cache=None,
    hourly_history_cache=None,
    feature_cache_context=None,
    market_context_map=None,
) -> List[str]:
    """
    AI Elite Strategy: ML-based scoring of momentum + dip opportunities

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        per_ticker_models: Dict of ticker -> trained model
        shared_model_path: Optional path to a shared-base model for inference-only execution
        shared_model_token: Optional cache-busting token for worker-side model reloads
        max_prediction_workers: Optional override for prediction thread count

    Returns:
        List of selected ticker symbols
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    if per_ticker_models is None and shared_model_path:
        shared_model = _load_shared_ai_elite_model_for_inference(shared_model_path, shared_model_token)
        if shared_model is None:
            print(f"   ⚠️ AI Elite: No shared model available for inference at {shared_model_path}")
            return []
        per_ticker_models = {"_shared_base": shared_model}

    if not per_ticker_models or not isinstance(per_ticker_models, dict):
        print("   ⚠️ AI Elite: No AI models available, returning empty")
        return []

    shared_base = per_ticker_models.get('_shared_base')
    if not shared_base and not any(k != '_shared_base' for k in per_ticker_models):
        print("   ⚠️ AI Elite: No valid AI models found, returning empty")
        return []

    hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
    if market_context_map is None:
        market_context_map = precompute_ai_elite_market_context_map(
            ticker_data_grouped,
            current_date,
            current_date,
        )

    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite"
    )

    print(f"   🤖 AI Elite: Analyzing {len(filtered_tickers)} tickers with ML scoring (filtered from {len(all_tickers)})")

    # Use threads here instead of another process pool to avoid nested multiprocessing issues.
    from config import NUM_PROCESSES, PARALLEL_THRESHOLD

    start_time = time.time()
    timing_stats = _create_prediction_timing_stats()

    ai_scores = {}
    candidate_payloads = {}
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'no_model': 0, 'exception': 0}

    def _record_prediction_result(ticker_result, score, status, payload=None):
        if status == 'success':
            ai_scores[ticker_result] = score
            if payload is not None:
                candidate_payloads[ticker_result] = payload
        else:
            fail_reasons[status] += 1
            if status == 'no_model':
                ai_scores[ticker_result] = 0.0
                if payload is not None:
                    candidate_payloads[ticker_result] = payload

    def _run_sequential_prediction():
        for args in tqdm(
            predict_args,
            total=len(predict_args),
            desc="AI Elite prediction",
            unit="ticker",
        ):
            ticker_result, score, status, payload = _predict_ticker_worker(args)
            _record_prediction_result(ticker_result, score, status, payload)

    predict_args = []
    shared_base_model = per_ticker_models.get('_shared_base') if isinstance(per_ticker_models, dict) else None
    for ticker in filtered_tickers:
        ticker_data = ticker_data_grouped.get(ticker)
        ticker_model = None
        if isinstance(per_ticker_models, dict):
            ticker_model = per_ticker_models.get(ticker, shared_base_model)
        predict_args.append((
            ticker,
            ticker_data,
            current_date,
            ticker_model,
            price_history_cache,
            hourly_history_cache,
            feature_cache_context,
            market_context_map,
            timing_stats,
        ))

    configured_workers = max_prediction_workers if max_prediction_workers is not None else NUM_PROCESSES
    n_workers = min(max(1, configured_workers), len(predict_args)) if predict_args else 1
    use_parallel_prediction = n_workers > 1 and len(predict_args) >= PARALLEL_THRESHOLD

    if use_parallel_prediction:
        if os.name != "nt":
            print(f"   🚀 AI Elite: Predicting with {n_workers} fork workers")
            predict_context = {
                "ticker_data_grouped": ticker_data_grouped,
                "per_ticker_models": per_ticker_models,
                "shared_base_model": shared_base_model,
                "current_date": current_date,
                "price_history_cache": price_history_cache,
                "hourly_history_cache": hourly_history_cache,
                "feature_cache_context": feature_cache_context,
                "market_context_map": market_context_map,
            }
            chunksize = max(1, len(filtered_tickers) // (n_workers * 4))
            try:
                with get_context("fork").Pool(
                    processes=n_workers,
                    initializer=_init_ai_elite_predict_worker,
                    initargs=(predict_context,),
                ) as pool:
                    results = pool.imap_unordered(
                        _predict_ticker_from_context_worker,
                        filtered_tickers,
                        chunksize=chunksize,
                    )
                    for ticker_result, score, status, payload in tqdm(
                        results,
                        total=len(filtered_tickers),
                        desc="AI Elite prediction",
                        unit="ticker",
                    ):
                        _record_prediction_result(ticker_result, score, status, payload)
            except Exception as e:
                print(f"   ⚠️ AI Elite: Forked prediction failed ({type(e).__name__}: {e})")
                return []
        else:
            print(f"   🧵 AI Elite: Predicting with {n_workers} threads")
            try:
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ai-elite-predict") as executor:
                    futures = [executor.submit(_predict_ticker_worker, args) for args in predict_args]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="AI Elite prediction",
                        unit="ticker",
                    ):
                        ticker_result, score, status, payload = future.result()
                        _record_prediction_result(ticker_result, score, status, payload)
            except Exception as e:
                print(f"   ⚠️ AI Elite: Threaded prediction failed ({type(e).__name__}: {e})")
                return []
    else:
        _run_sequential_prediction()

    elapsed = time.time() - start_time
    print(f"   📊 AI Elite: Predicted {len(ai_scores)} tickers ({elapsed:.1f}s)")
    _print_prediction_timing("AI Elite", timing_stats)

    if not ai_scores:
        print(f"   ⚠️ AI Elite: No predictions found")
        print(f"   🔍 AI Elite: Fail reasons: {fail_reasons}")
        return []

    # Create candidates DataFrame from successful predictions
    feature_cols = list(AI_ELITE_FEATURE_NAMES)

    # Reuse features computed during prediction to avoid a second hourly load pass.
    candidates = []
    debug_count = 0
    for ticker, ai_score in ai_scores.items():
        try:
            daily_data = ticker_data_grouped[ticker]
            payload = candidate_payloads.get(ticker)
            if payload is None:
                continue
            features = dict(payload["features"])

            features['ticker'] = ticker
            features['ai_score'] = ai_score
            candidates.append(features)

            # Debug first 3 tickers
            if debug_count < 3:
                has_hourly = bool(payload.get("has_hourly_data"))
                print(f"   🔍 AI Elite DEBUG {ticker}: daily={len(daily_data)} rows, hourly={'yes' if has_hourly else 'no'}")
                debug_count += 1
        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    candidates_df = pd.DataFrame(candidates)

    # Debug-only ranks
    candidates_df['momentum_rank'] = candidates_df['perf_3m'].rank(pct=True)
    candidates_df['risk_adj_mom_rank'] = candidates_df['risk_adj_mom_3m'].rank(pct=True)

    # Pure AI scoring - the model predicts expected forward return directly
    candidates_df['final_score'] = candidates_df['ai_score']

    # Sort by final hybrid score
    candidates_df = candidates_df.sort_values('final_score', ascending=False)

    # Debug: show top candidates with momentum rank
    print(f"   ✅ AI Elite: Found {len(candidates_df)} candidates")
    print(f"   📊 AI Elite: Scoring = ML regression (predicts forward return)")
    for i, row in candidates_df.head(5).iterrows():
        print(f"      {i+1}. {row['ticker']}: PredReturn={row['final_score']:+.2f}% (RiskAdjMom={row['risk_adj_mom_rank']:.3f}), "
              f"3M={row['perf_3m']:+.1f}%, Vol={row['volatility']:.1f}%, RiskAdj={row['risk_adj_mom_3m']:.2f})")

    # Return top N tickers by final hybrid score
    selected = candidates_df.head(top_n)['ticker'].tolist()
    return selected


def _get_ai_elite_daily_frame_from_price_cache(
    ticker: str,
    current_date: datetime,
    price_history_cache,
) -> Optional[pd.DataFrame]:
    if price_history_cache is None:
        return None

    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    close_values = price_history_cache.close_by_ticker.get(ticker)
    if date_ns is None or close_values is None or date_ns.size < 20 or close_values.size < 20:
        return None

    current_ts = pd.Timestamp(current_date)
    if current_ts.tz is None:
        current_ts = current_ts.tz_localize('UTC')
    else:
        current_ts = current_ts.tz_convert('UTC')

    end_idx = int(np.searchsorted(date_ns, current_ts.tz_localize(None).value, side='right'))
    if end_idx < 20:
        return None

    data = {
        'Close': np.asarray(close_values[:end_idx], dtype=float),
    }

    high_values = getattr(price_history_cache, 'high_by_ticker', {}).get(ticker)
    if high_values is not None and high_values.size >= end_idx:
        data['High'] = np.asarray(high_values[:end_idx], dtype=float)

    low_values = getattr(price_history_cache, 'low_by_ticker', {}).get(ticker)
    if low_values is not None and low_values.size >= end_idx:
        data['Low'] = np.asarray(low_values[:end_idx], dtype=float)

    volume_values = price_history_cache.volume_by_ticker.get(ticker)
    if volume_values is not None and volume_values.size >= end_idx:
        data['Volume'] = np.asarray(volume_values[:end_idx], dtype=float)

    daily_frame = pd.DataFrame(
        data,
        index=pd.to_datetime(date_ns[:end_idx], unit='ns', utc=True),
    )
    daily_frame = daily_frame.dropna(subset=['Close'])
    return daily_frame if len(daily_frame) >= 20 else None


def _prepare_ai_elite_daily_context(
    daily_data: Optional[pd.DataFrame],
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    if daily_data is None or len(daily_data) == 0:
        return None

    if daily_data.index.duplicated().any():
        daily_data = daily_data[~daily_data.index.duplicated(keep='last')]
    daily_data = daily_data.sort_index()

    if daily_data.index.tz is None:
        daily_data = daily_data.copy()
        daily_data.index = daily_data.index.tz_localize('UTC')
    else:
        daily_data = daily_data.copy()
        daily_data.index = daily_data.index.tz_convert('UTC')

    daily_data = daily_data.dropna(subset=['Close'])
    if len(daily_data) < 20:
        return None

    close_values = pd.to_numeric(daily_data['Close'], errors='coerce').to_numpy(dtype=float, copy=True)
    valid_close = ~np.isnan(close_values)
    if int(valid_close.sum()) < 20:
        return None

    aligned_index = daily_data.index[valid_close]
    aligned_close = close_values[valid_close]
    high_values = (
        pd.to_numeric(daily_data['High'], errors='coerce').ffill().fillna(daily_data['Close']).to_numpy(dtype=float, copy=True)[valid_close]
        if 'High' in daily_data.columns
        else aligned_close.copy()
    )
    low_values = (
        pd.to_numeric(daily_data['Low'], errors='coerce').ffill().fillna(daily_data['Close']).to_numpy(dtype=float, copy=True)[valid_close]
        if 'Low' in daily_data.columns
        else aligned_close.copy()
    )
    volume_values = (
        pd.to_numeric(daily_data['Volume'], errors='coerce').fillna(0.0).to_numpy(dtype=float, copy=True)[valid_close]
        if 'Volume' in daily_data.columns
        else np.zeros(aligned_close.shape, dtype=float)
    )

    context: Dict[str, np.ndarray] = {
        'date_ns': aligned_index.to_numpy(dtype='datetime64[ns]').astype(np.int64, copy=True),
        'close': aligned_close,
        'high': high_values,
        'low': low_values,
        'volume': volume_values,
    }

    close_series = pd.Series(aligned_close)
    date_ns = context['date_ns']
    sample_idx = np.arange(aligned_close.size, dtype=np.int64)
    day_ns = 24 * 60 * 60 * 1_000_000_000

    def _perf_series(calendar_days: int) -> np.ndarray:
        start_ns = date_ns - (calendar_days * day_ns)
        start_idx = np.searchsorted(date_ns, start_ns, side='left')
        counts = sample_idx - start_idx + 1
        perf = np.zeros(aligned_close.shape, dtype=float)
        valid = counts >= 5
        if np.any(valid):
            base_prices = aligned_close[start_idx[valid]]
            current_prices = aligned_close[valid]
            valid_base = base_prices > 0
            perf_vals = np.zeros(base_prices.shape, dtype=float)
            perf_vals[valid_base] = ((current_prices[valid_base] - base_prices[valid_base]) / base_prices[valid_base]) * 100
            perf[valid] = perf_vals
        return perf

    perf_5d = _perf_series(5)
    perf_20d = _perf_series(20)
    perf_3m = _perf_series(90)
    perf_6m = _perf_series(180)
    perf_1y = _perf_series(365)

    daily_returns = close_series.pct_change()
    returns_array = daily_returns.to_numpy(dtype=float, copy=True)[1:]
    daily_vol_pct = (daily_returns.rolling(window=20, min_periods=10).std() * 100).to_numpy(dtype=float, copy=True)
    volatility = daily_vol_pct * np.sqrt(252)
    volatility_floor = np.maximum(np.nan_to_num(volatility, nan=0.0), 5.0)

    gains = daily_returns.clip(lower=0.0)
    losses = (-daily_returns).clip(lower=0.0)
    avg_gain_14 = gains.rolling(window=14, min_periods=14).mean()
    avg_loss_14 = losses.rolling(window=14, min_periods=14).mean()
    rsi_14 = np.full(aligned_close.shape, 50.0, dtype=float)
    avg_gain_arr = avg_gain_14.to_numpy(dtype=float, copy=True)
    avg_loss_arr = avg_loss_14.to_numpy(dtype=float, copy=True)
    positive_loss = avg_loss_arr > 0
    if np.any(positive_loss):
        rs = avg_gain_arr[positive_loss] / avg_loss_arr[positive_loss]
        rsi_14[positive_loss] = 100.0 - (100.0 / (1.0 + rs))
    zero_loss_positive_gain = (~positive_loss) & (avg_gain_arr > 0)
    rsi_14[zero_loss_positive_gain] = 100.0

    sma20 = close_series.rolling(window=20, min_periods=20).mean().to_numpy(dtype=float, copy=True)
    std20 = close_series.rolling(window=20, min_periods=20).std().to_numpy(dtype=float, copy=True)
    sma50 = close_series.rolling(window=50, min_periods=50).mean().to_numpy(dtype=float, copy=True)

    bollinger_position = np.full(aligned_close.shape, 0.5, dtype=float)
    sma20_distance = np.zeros(aligned_close.shape, dtype=float)
    sma50_distance = np.zeros(aligned_close.shape, dtype=float)
    valid_boll = (~np.isnan(sma20)) & (~np.isnan(std20)) & (std20 > 0) & (sma20 > 0)
    if np.any(valid_boll):
        upper_band = sma20[valid_boll] + 2 * std20[valid_boll]
        lower_band = sma20[valid_boll] - 2 * std20[valid_boll]
        pos = (aligned_close[valid_boll] - lower_band) / (upper_band - lower_band)
        bollinger_position[valid_boll] = np.clip(pos, 0.0, 1.0)
        sma20_distance[valid_boll] = ((aligned_close[valid_boll] - sma20[valid_boll]) / sma20[valid_boll]) * 100
    valid_sma50 = (~np.isnan(sma50)) & (sma50 > 0)
    if np.any(valid_sma50):
        sma50_distance[valid_sma50] = ((aligned_close[valid_sma50] - sma50[valid_sma50]) / sma50[valid_sma50]) * 100

    macd_values = np.full(aligned_close.shape, np.nan, dtype=float)
    if aligned_close.size >= 26:
        for idx in range(25, aligned_close.size):
            close_slice_26 = pd.Series(aligned_close[max(0, idx - 25):idx + 1])
            ema12 = close_slice_26.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = close_slice_26.ewm(span=26, adjust=False).mean().iloc[-1]
            macd_line = ema12 - ema26

            close_slice_35 = pd.Series(aligned_close[max(0, idx - 34):idx + 1])
            macd_series = (
                close_slice_35.ewm(span=12, adjust=False).mean()
                - close_slice_35.ewm(span=26, adjust=False).mean()
            )
            signal_line = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
            macd_values[idx] = macd_line - signal_line
        macd_values = np.nan_to_num(macd_values, nan=0.0)

    mom_5d = _ai_elite_pct_change_array(aligned_close, 5)
    mom_10d = _ai_elite_pct_change_array(aligned_close, 10)
    mom_20d = _ai_elite_pct_change_array(aligned_close, 20)
    mom_40d = _ai_elite_pct_change_array(aligned_close, 40)
    mom_60d = _ai_elite_pct_change_array(aligned_close, 60)

    volatility_10d = np.zeros(aligned_close.shape, dtype=float)
    volatility_20d = np.zeros(aligned_close.shape, dtype=float)
    for idx in range(10, aligned_close.size):
        volatility_10d[idx] = float(np.std(returns_array[idx - 10:idx]) * np.sqrt(252) * 100.0)
    for idx in range(20, aligned_close.size):
        volatility_20d[idx] = float(np.std(returns_array[idx - 20:idx]) * np.sqrt(252) * 100.0)

    drawdown_20d = np.zeros(aligned_close.shape, dtype=float)
    atr_pct_14d = np.zeros(aligned_close.shape, dtype=float)
    prefix_volume_mean = np.cumsum(volume_values, dtype=float) / np.arange(1, aligned_close.size + 1, dtype=float)
    recent_volume_20 = np.zeros(aligned_close.shape, dtype=float)
    volume_ratio_20_60 = np.ones(aligned_close.shape, dtype=float)
    for idx in range(aligned_close.size):
        window_start = max(0, idx - 19)
        recent_close = aligned_close[window_start:idx + 1]
        if recent_close.size > 0:
            drawdown_20d[idx] = _ai_elite_safe_pct_change(float(aligned_close[idx]), float(np.max(recent_close)))
        recent_volume = volume_values[window_start:idx + 1]
        if recent_volume.size > 0:
            recent_volume_20[idx] = float(np.mean(recent_volume))
            if prefix_volume_mean[idx] > 0:
                volume_ratio_20_60[idx] = recent_volume_20[idx] / prefix_volume_mean[idx]
        if idx >= 13 and aligned_close[idx] > 0:
            atr_14 = float(np.mean(high_values[idx - 13:idx + 1] - low_values[idx - 13:idx + 1]))
            atr_pct_14d[idx] = (atr_14 / aligned_close[idx]) * 100.0

    market_return_5d = np.zeros(aligned_close.shape, dtype=float)
    market_return_20d = np.zeros(aligned_close.shape, dtype=float)
    market_return_60d = np.zeros(aligned_close.shape, dtype=float)
    market_volatility_20d = np.zeros(aligned_close.shape, dtype=float)
    market_breadth_20d = np.full(aligned_close.shape, 0.5, dtype=float)
    market_breadth_60d = np.full(aligned_close.shape, 0.5, dtype=float)
    if market_context_map:
        normalized_dates = pd.to_datetime(date_ns, unit='ns', utc=True).normalize()
        for idx, normalized_date in enumerate(normalized_dates):
            market_context = market_context_map.get(normalized_date)
            if not market_context:
                continue
            market_return_5d[idx] = float(market_context.get('market_return_5d', 0.0))
            market_return_20d[idx] = float(market_context.get('market_return_20d', 0.0))
            market_return_60d[idx] = float(market_context.get('market_return_60d', 0.0))
            market_volatility_20d[idx] = float(market_context.get('market_volatility_20d', 0.0))
            market_breadth_20d[idx] = float(market_context.get('market_breadth_20d', 0.5))
            market_breadth_60d[idx] = float(market_context.get('market_breadth_60d', 0.5))

    savgol_arrays = _build_ai_elite_savgol_feature_arrays(np.log(np.clip(aligned_close, 1e-12, None)))

    context['perf_5d'] = perf_5d
    context['perf_20d'] = perf_20d
    context['perf_3m'] = perf_3m
    context['perf_6m'] = perf_6m
    context['perf_1y'] = perf_1y
    context['mom_5d'] = mom_5d
    context['mom_10d'] = mom_10d
    context['mom_20d'] = mom_20d
    context['mom_40d'] = mom_40d
    context['mom_60d'] = mom_60d
    context['daily_vol_pct'] = np.nan_to_num(daily_vol_pct, nan=0.0)
    context['volatility'] = np.nan_to_num(volatility, nan=0.0)
    context['volatility_10d'] = volatility_10d
    context['volatility_20d'] = volatility_20d
    context['risk_adj_mom_3m'] = perf_3m / np.sqrt(volatility_floor)
    context['risk_adj_score'] = np.where(
        np.nan_to_num(daily_vol_pct, nan=0.0) > 0,
        perf_1y / (np.sqrt(np.nan_to_num(daily_vol_pct, nan=0.0)) + 0.001),
        0.0,
    )
    context['dip_score'] = perf_1y - perf_3m
    context['mom_accel'] = perf_3m - perf_6m
    context['vol_sweet_spot'] = ((np.nan_to_num(volatility, nan=0.0) >= 20.0) & (np.nan_to_num(volatility, nan=0.0) <= 40.0)).astype(float)
    context['rsi_14'] = rsi_14
    context['sma20'] = sma20
    context['std20'] = std20
    context['sma50'] = sma50
    context['bollinger_position'] = bollinger_position
    context['sma20_distance'] = sma20_distance
    context['sma50_distance'] = sma50_distance
    context['macd'] = macd_values
    context['short_term_reversal'] = perf_5d - perf_20d
    context['drawdown_20d'] = drawdown_20d
    context['atr_pct_14d'] = atr_pct_14d
    context['volume_ratio_20_60'] = volume_ratio_20_60
    context['market_return_5d'] = market_return_5d
    context['market_return_20d'] = market_return_20d
    context['market_return_60d'] = market_return_60d
    context['market_volatility_20d'] = market_volatility_20d
    context['market_breadth_20d'] = market_breadth_20d
    context['market_breadth_60d'] = market_breadth_60d
    context['rel_strength_20d'] = mom_20d - market_return_20d
    context['rel_strength_60d'] = mom_60d - market_return_60d
    context.update(savgol_arrays)

    volume_series = pd.Series(volume_values)
    avg_volume_30 = volume_series.rolling(window=30, min_periods=1).mean().to_numpy(dtype=float, copy=True)
    recent_vol_20 = volume_series.rolling(window=20, min_periods=20).mean().to_numpy(dtype=float, copy=True)
    prior_vol_20 = pd.Series(recent_vol_20).shift(20).to_numpy(dtype=float, copy=True)
    recent_vol_5 = volume_series.rolling(window=5, min_periods=5).mean().to_numpy(dtype=float, copy=True)

    volume_ratio = np.ones(volume_values.shape, dtype=float)
    valid_ratio = (~np.isnan(prior_vol_20)) & (prior_vol_20 > 0) & (~np.isnan(recent_vol_20))
    if np.any(valid_ratio):
        volume_ratio[valid_ratio] = recent_vol_20[valid_ratio] / prior_vol_20[valid_ratio]

    volume_sentiment = np.zeros(volume_values.shape, dtype=float)
    valid_sentiment = (~np.isnan(recent_vol_5)) & (~np.isnan(recent_vol_20)) & (recent_vol_20 > 0)
    if np.any(valid_sentiment):
        vol_surge = (recent_vol_5[valid_sentiment] / recent_vol_20[valid_sentiment]) - 1.0
        price_direction = np.sign(perf_5d[valid_sentiment])
        volume_sentiment[valid_sentiment] = vol_surge * price_direction

    context['avg_volume'] = np.nan_to_num(avg_volume_30, nan=0.0)
    context['volume_ratio'] = volume_ratio
    context['volume_sentiment'] = volume_sentiment

    return context


def _prepare_ai_elite_hourly_context(hourly_data: Optional[pd.DataFrame]) -> Optional[Dict[str, np.ndarray]]:
    if hourly_data is None or len(hourly_data) == 0:
        return None

    if hourly_data.index.duplicated().any():
        hourly_data = hourly_data[~hourly_data.index.duplicated(keep='last')]
    hourly_data = hourly_data.sort_index()

    if hourly_data.index.tz is None:
        hourly_data = hourly_data.copy()
        hourly_data.index = hourly_data.index.tz_localize('UTC')
    else:
        hourly_data = hourly_data.copy()
        hourly_data.index = hourly_data.index.tz_convert('UTC')

    close_values = pd.to_numeric(hourly_data['Close'], errors='coerce').to_numpy(dtype=float, copy=True)
    valid_close = ~np.isnan(close_values)
    if int(valid_close.sum()) < 2:
        return None

    open_values = pd.to_numeric(hourly_data['Open'], errors='coerce').to_numpy(dtype=float, copy=True)[valid_close]
    high_values = pd.to_numeric(hourly_data['High'], errors='coerce').to_numpy(dtype=float, copy=True)[valid_close]
    low_values = pd.to_numeric(hourly_data['Low'], errors='coerce').to_numpy(dtype=float, copy=True)[valid_close]
    close_values = close_values[valid_close]
    aligned_index = hourly_data.index[valid_close]
    date_ns = aligned_index.to_numpy(dtype='datetime64[ns]').astype(np.int64, copy=True)
    session_ns = aligned_index.normalize().to_numpy(dtype='datetime64[ns]').astype(np.int64, copy=True)

    session_date_ns, session_start_idx, session_counts = np.unique(
        session_ns,
        return_index=True,
        return_counts=True,
    )
    session_end_idx = session_start_idx + session_counts
    prev_close = np.full(session_date_ns.shape, np.nan, dtype=float)
    if session_date_ns.size > 1:
        prev_close[1:] = close_values[session_end_idx[:-1] - 1]

    return {
        'date_ns': date_ns,
        'session_date_ns': session_date_ns,
        'session_start_idx': session_start_idx.astype(np.int64, copy=False),
        'session_end_idx': session_end_idx.astype(np.int64, copy=False),
        'session_prev_close': prev_close,
        'open': open_values,
        'high': high_values,
        'low': low_values,
        'close': close_values,
    }


def _lookup_ai_elite_daily_features(
    daily_context: Optional[Dict[str, np.ndarray]],
    current_ts: pd.Timestamp,
) -> Optional[Dict[str, float]]:
    if daily_context is None:
        return None

    date_ns = daily_context.get('date_ns')
    close_values = daily_context.get('close')
    if date_ns is None or close_values is None or len(date_ns) < 20 or len(close_values) < 20:
        return None

    current_ns = current_ts.tz_localize(None).value
    end_idx = int(np.searchsorted(date_ns, current_ns, side='right'))
    if end_idx < 20:
        return None

    latest_price = float(close_values[end_idx - 1])
    if latest_price <= 0:
        return None

    def _lookup(context_key: str, default: float) -> float:
        values = daily_context.get(context_key)
        if values is None or len(values) < end_idx:
            return default
        value = values[end_idx - 1]
        if np.isnan(value):
            return default
        return float(value)

    return {
        'perf_3m': _lookup('perf_3m', 0.0),
        'perf_6m': _lookup('perf_6m', 0.0),
        'perf_1y': _lookup('perf_1y', 0.0),
        'volatility': _lookup('volatility', 0.0),
        'avg_volume': _lookup('avg_volume', 0.0),
        'risk_adj_score': _lookup('risk_adj_score', 0.0),
        'dip_score': _lookup('dip_score', 0.0),
        'mom_accel': _lookup('mom_accel', 0.0),
        'vol_sweet_spot': _lookup('vol_sweet_spot', 0.0),
        'volume_ratio': _lookup('volume_ratio', 1.0),
        'rsi_14': _lookup('rsi_14', 50.0),
        'short_term_reversal': _lookup('short_term_reversal', 0.0),
        'volume_sentiment': _lookup('volume_sentiment', 0.0),
        'risk_adj_mom_3m': _lookup('risk_adj_mom_3m', 0.0),
        'bollinger_position': _lookup('bollinger_position', 0.5),
        'sma20_distance': _lookup('sma20_distance', 0.0),
        'sma50_distance': _lookup('sma50_distance', 0.0),
        'macd': _lookup('macd', 0.0),
        'mom_5d': _lookup('mom_5d', 0.0),
        'mom_10d': _lookup('mom_10d', 0.0),
        'mom_20d': _lookup('mom_20d', 0.0),
        'mom_40d': _lookup('mom_40d', 0.0),
        'volatility_10d': _lookup('volatility_10d', 0.0),
        'volatility_20d': _lookup('volatility_20d', 0.0),
        'drawdown_20d': _lookup('drawdown_20d', 0.0),
        'atr_pct_14d': _lookup('atr_pct_14d', 0.0),
        'volume_ratio_20_60': _lookup('volume_ratio_20_60', 1.0),
        'price_vs_sg_short': _lookup('price_vs_sg_short', 0.0),
        'price_vs_sg_long': _lookup('price_vs_sg_long', 0.0),
        'sg_short_slope': _lookup('sg_short_slope', 0.0),
        'sg_long_slope': _lookup('sg_long_slope', 0.0),
        'sg_short_curvature': _lookup('sg_short_curvature', 0.0),
        'sg_long_curvature': _lookup('sg_long_curvature', 0.0),
        'sg_trend_spread': _lookup('sg_trend_spread', 0.0),
        'sg_trend_stability': _lookup('sg_trend_stability', 0.5),
        'sg_long_stability': _lookup('sg_long_stability', 0.5),
        'sg_regression_slope': _lookup('sg_regression_slope', 0.0),
        'sg_residual_vol_20d': _lookup('sg_residual_vol_20d', 0.0),
        'market_return_5d': _lookup('market_return_5d', 0.0),
        'market_return_20d': _lookup('market_return_20d', 0.0),
        'market_return_60d': _lookup('market_return_60d', 0.0),
        'market_volatility_20d': _lookup('market_volatility_20d', 0.0),
        'market_breadth_20d': _lookup('market_breadth_20d', 0.5),
        'market_breadth_60d': _lookup('market_breadth_60d', 0.5),
        'rel_strength_20d': _lookup('rel_strength_20d', 0.0),
        'rel_strength_60d': _lookup('rel_strength_60d', 0.0),
    }


def _lookup_ai_elite_intraday_features(
    hourly_context: Optional[Dict[str, np.ndarray]],
    current_ts: pd.Timestamp,
) -> Tuple[Dict[str, float], bool]:
    intraday_features = {
        'overnight_gap': 0.0,
        'intraday_range': 0.0,
        'last_hour_momentum': 0.0,
    }
    if hourly_context is None:
        return intraday_features, False

    try:
        sample_session_ns = current_ts.normalize().tz_localize(None).value
        session_date_ns = hourly_context.get('session_date_ns')
        session_start_idx = hourly_context.get('session_start_idx')
        session_end_idx = hourly_context.get('session_end_idx')
        session_prev_close = hourly_context.get('session_prev_close')
        hourly_date_ns = hourly_context.get('date_ns')
        open_values = hourly_context.get('open')
        high_values = hourly_context.get('high')
        low_values = hourly_context.get('low')
        close_hourly = hourly_context.get('close')

        if (
            session_date_ns is None
            or session_start_idx is None
            or session_end_idx is None
            or hourly_date_ns is None
            or open_values is None
            or high_values is None
            or low_values is None
            or close_hourly is None
        ):
            return intraday_features, False

        session_pos = int(np.searchsorted(session_date_ns, sample_session_ns, side='left'))
        if session_pos >= len(session_date_ns) or session_date_ns[session_pos] != sample_session_ns:
            return intraday_features, False

        start_idx = int(session_start_idx[session_pos])
        end_slice = int(session_end_idx[session_pos])
        if end_slice <= start_idx:
            return intraday_features, False

        first_open = float(open_values[start_idx])
        if first_open > 0:
            intraday_features['intraday_range'] = float(
                (np.max(high_values[start_idx:end_slice]) - np.min(low_values[start_idx:end_slice]))
                / first_open
                * 100
            )

        if end_slice - start_idx >= 2:
            prev_hour_close = float(close_hourly[end_slice - 2])
            if prev_hour_close > 0:
                intraday_features['last_hour_momentum'] = float(
                    (float(close_hourly[end_slice - 1]) - prev_hour_close)
                    / prev_hour_close
                    * 100
                )

        if session_prev_close is not None:
            prev_close = float(session_prev_close[session_pos])
            if not np.isnan(prev_close) and prev_close > 0 and first_open > 0:
                intraday_features['overnight_gap'] = float((first_open - prev_close) / prev_close * 100)

        return intraday_features, True
    except Exception:
        return intraday_features, False


def _build_ai_elite_feature_row(
    current_ts: pd.Timestamp,
    daily_context: Optional[Dict[str, np.ndarray]],
    hourly_context: Optional[Dict[str, np.ndarray]],
) -> Tuple[Optional[Dict[str, float]], bool]:
    daily_features = _lookup_ai_elite_daily_features(daily_context, current_ts)
    if daily_features is None:
        return None, False
    intraday_features, intraday_covered = _lookup_ai_elite_intraday_features(hourly_context, current_ts)
    return {**daily_features, **intraday_features}, intraday_covered


def _extract_features(
    ticker: str,
    hourly_data: Optional[pd.DataFrame],
    current_date: datetime,
    daily_data: Optional[pd.DataFrame] = None,
    price_history_cache=None,
    daily_context: Optional[Dict[str, np.ndarray]] = None,
    hourly_context: Optional[Dict[str, np.ndarray]] = None,
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
) -> Optional[Dict]:
    """
    Extract ML features using BOTH data sources:
    - hourly_data: intraday features (overnight gap, intraday range, last-hour momentum)
    - daily_data:  daily features (3m/6m/1y performance, volatility, volume)

    If hourly_data is None, intraday features default to 0.
    daily_data is required - returns None if missing.
    """
    try:
        current_ts = _normalize_ai_elite_timestamp(current_date)

        # ------------------------------------------------------------------ #
        # DAILY FEATURES  (3m / 6m / 1y performance, volatility, volume)     #
        # ------------------------------------------------------------------ #
        if daily_context is None:
            cached_daily_data = _get_ai_elite_daily_frame_from_price_cache(
                ticker,
                current_date,
                price_history_cache,
            )
            if cached_daily_data is not None:
                daily_data = cached_daily_data
                daily_context = _prepare_ai_elite_daily_context(daily_data, market_context_map=market_context_map)
            else:
                daily_context = _prepare_ai_elite_daily_context(daily_data, market_context_map=market_context_map)

        if daily_context is None:
            return None

        if hourly_context is None and hourly_data is not None and len(hourly_data) > 0:
            hourly_context = _prepare_ai_elite_hourly_context(hourly_data)

        feature_row, _ = _build_ai_elite_feature_row(
            current_ts=current_ts,
            daily_context=daily_context,
            hourly_context=hourly_context,
        )
        return feature_row

    except Exception as e:
        if not hasattr(_extract_features, '_err_logged') or _extract_features._err_logged < 3:
            print(f"   ❌ FEAT EXCEPTION {ticker}: {type(e).__name__}: {e}")
            if not hasattr(_extract_features, '_err_logged'):
                _extract_features._err_logged = 0
            _extract_features._err_logged += 1
        return None


def _ai_elite_context_files_exist(context: Optional[Dict[str, object]]) -> bool:
    if not isinstance(context, dict):
        return False
    if str(context.get("numeric_dtype") or "") != AI_ELITE_CACHE_NUMERIC_DTYPE:
        return False
    required_paths = (
        "daily_feature_path",
        "intraday_feature_path",
        "forward_path",
        "valid_mask_path",
        "intraday_coverage_path",
        "date_index_path",
    )
    for key in required_paths:
        path_value = context.get(key)
        if not path_value or not Path(str(path_value)).exists():
            return False
    return True


def _load_exact_ai_elite_training_context(cache_dir: Path) -> Optional[Dict[str, object]]:
    loaded = _load_ai_elite_context_metadata(cache_dir)
    if not _ai_elite_context_files_exist(loaded):
        return None
    return loaded


def precompute_ai_elite_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    existing_context: Optional[Dict[str, object]] = None,
    cache_start_date: Optional[datetime] = None,
    cache_end_date: Optional[datetime] = None,
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
    price_history_cache=None,
    hourly_history_cache=None,
) -> Dict[str, object]:
    from parallel_backtest import build_hourly_history_cache, preload_hourly_tickers_parallel
    from config import NUM_PROCESSES

    valid_tickers = _valid_ai_elite_tickers(ticker_data_grouped, all_tickers)
    cache_start = cache_start_date if cache_start_date is not None else train_start_date
    cache_end = cache_end_date if cache_end_date is not None else train_end_date
    start_key = _normalize_ai_elite_timestamp(cache_start)
    end_key = _normalize_ai_elite_timestamp(cache_end)
    requested_start_ordinal = start_key.to_pydatetime().date().toordinal()
    requested_end_ordinal = end_key.to_pydatetime().date().toordinal()
    requested_latest_allowed_ordinal = _normalize_ai_elite_timestamp(train_end_date).to_pydatetime().date().toordinal()
    daily_signature = universe_signature_from_frames(ticker_data_grouped, valid_tickers)
    hourly_signature = _ai_elite_hourly_universe_signature(valid_tickers)
    cache_dir = get_cache_dir(
        "ai_elite/context",
        _ai_elite_context_cache_key(
            valid_tickers=valid_tickers,
            train_start_date=cache_start,
            train_end_date=cache_end,
            forward_days=forward_days,
            daily_signature=daily_signature,
            hourly_signature=hourly_signature,
        ),
    )

    if hourly_history_cache is None:
        hourly_history_cache = build_hourly_history_cache()
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    n_workers = max(1, min(NUM_PROCESSES, len(valid_tickers))) if valid_tickers else 1
    if valid_tickers:
        print(
            f"   ⚡ AI Elite: Preloading hourly caches for {len(valid_tickers)} tickers "
            f"({n_workers} workers)..."
        )
        preload_start = time.perf_counter()
        loaded_count = preload_hourly_tickers_parallel(
            hourly_history_cache,
            valid_tickers,
            n_workers=n_workers,
            show_progress=True,
        )
        preload_elapsed = time.perf_counter() - preload_start
        print(
            f"   ✅ AI Elite: Hourly preload ready "
            f"({loaded_count} loaded, {preload_elapsed:.1f}s)"
        )

    def _ensure_market_context_map() -> Dict[pd.Timestamp, Dict[str, float]]:
        nonlocal market_context_map
        if market_context_map is None:
            print(
                f"   🌐 AI Elite: Precomputing market context "
                f"({cache_start.date()} to {cache_end.date()})..."
            )
            market_context_start = time.perf_counter()
            market_context_map = precompute_ai_elite_market_context_map(
                ticker_data_grouped,
                cache_start,
                cache_end,
            )
            market_context_elapsed = time.perf_counter() - market_context_start
            print(
                f"   ✅ AI Elite: Market context ready "
                f"({len(market_context_map)} days, {market_context_elapsed:.1f}s)"
            )
        return market_context_map

    exact_cached = _load_exact_ai_elite_training_context(cache_dir)
    if exact_cached is not None:
        exact_cached = dict(exact_cached)
        exact_cached["all_tickers"] = valid_tickers
        exact_cached["daily_signature"] = daily_signature
        exact_cached["hourly_signature"] = hourly_signature
        if requested_latest_allowed_ordinal > int(exact_cached.get("latest_allowed_ordinal", 0)):
            return _refresh_existing_ai_elite_training_context(
                ticker_data_grouped=ticker_data_grouped,
                valid_tickers=valid_tickers,
                train_end_date=train_end_date,
                forward_days=forward_days,
                context=exact_cached,
            )
        print(
            f"   ♻️ AI Elite: Reusing file cache "
            f"({len(valid_tickers)} tickers, {int(exact_cached.get('n_dates', 0))} cached dates)"
        )
        return exact_cached

    cache_context = existing_context if isinstance(existing_context, dict) else None
    cache_is_compatible = False
    if cache_context and _ai_elite_context_files_exist(cache_context):
        cached_tickers = list(cache_context.get("all_tickers") or [])
        cached_start_ordinal = int(cache_context.get("start_ordinal", 0))
        cached_n_dates = int(cache_context.get("n_dates", 0))
        cached_end_ordinal = cached_start_ordinal + cached_n_dates - 1
        cache_is_compatible = (
            cached_tickers == valid_tickers
            and int(cache_context.get("forward_days", -1)) == int(forward_days)
            and cached_start_ordinal <= requested_start_ordinal
            and str(cache_context.get("daily_signature", "")) == daily_signature
            and str(cache_context.get("hourly_signature", "")) == hourly_signature
        )
        if cache_is_compatible and cached_end_ordinal >= requested_end_ordinal:
            if requested_latest_allowed_ordinal > int(cache_context.get("latest_allowed_ordinal", cached_end_ordinal)):
                return _refresh_existing_ai_elite_training_context(
                    ticker_data_grouped=ticker_data_grouped,
                    valid_tickers=valid_tickers,
                    train_end_date=train_end_date,
                    forward_days=forward_days,
                    context=cache_context,
                )
            print(
                f"   ♻️ AI Elite: Reusing in-memory file cache "
                f"({len(valid_tickers)} tickers, {cached_n_dates} cached dates)"
            )
            return cache_context

    if cache_context and cache_is_compatible:
        return _extend_ai_elite_training_context(
            ticker_data_grouped=ticker_data_grouped,
            valid_tickers=valid_tickers,
            train_end_date=train_end_date,
            forward_days=forward_days,
            context=cache_context,
            market_context_map=_ensure_market_context_map(),
            price_history_cache=price_history_cache,
            hourly_history_cache=hourly_history_cache,
        )

    return _build_ai_elite_training_context(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        train_start_date=cache_start,
        train_end_date=cache_end,
        latest_allowed_date=train_end_date,
        forward_days=forward_days,
        cache_dir=cache_dir,
        daily_signature=daily_signature,
        hourly_signature=hourly_signature,
        market_context_map=_ensure_market_context_map(),
        price_history_cache=price_history_cache,
        hourly_history_cache=hourly_history_cache,
    )


def _build_ai_elite_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    latest_allowed_date: datetime,
    forward_days: int,
    cache_dir: Path,
    daily_signature: str,
    hourly_signature: str,
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
    price_history_cache=None,
    hourly_history_cache=None,
) -> Dict[str, object]:
    start_ordinal = _normalize_ai_elite_timestamp(train_start_date).to_pydatetime().date().toordinal()
    end_ordinal = _normalize_ai_elite_timestamp(train_end_date).to_pydatetime().date().toordinal()
    n_dates = max(1, int(end_ordinal - start_ordinal) + 1)
    daily_feature_path = cache_dir / "daily_features.npy"
    intraday_feature_path = cache_dir / "intraday_features.npy"
    forward_path = cache_dir / "forward_returns.npy"
    valid_mask_path = cache_dir / "valid_mask.npy"
    intraday_coverage_path = cache_dir / "intraday_coverage.npy"
    date_index_path = cache_dir / "date_ordinals.npy"
    ticker_index_path = cache_dir / "ticker_index.json"

    daily_feature_array = np.lib.format.open_memmap(
        daily_feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), n_dates, len(AI_ELITE_DAILY_FEATURE_NAMES)),
    )
    intraday_feature_array = np.lib.format.open_memmap(
        intraday_feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), n_dates, len(AI_ELITE_INTRADAY_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), n_dates),
    )
    valid_mask_array = np.lib.format.open_memmap(
        valid_mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(valid_tickers), n_dates),
    )
    intraday_coverage_array = np.lib.format.open_memmap(
        intraday_coverage_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(valid_tickers), n_dates),
    )

    daily_feature_array.fill(np.nan)
    intraday_feature_array.fill(0.0)
    forward_array.fill(np.nan)
    valid_mask_array.fill(False)
    intraday_coverage_array.fill(False)
    np.save(date_index_path, np.arange(start_ordinal, start_ordinal + n_dates, dtype=np.int32))
    ticker_index_path.write_text(
        json.dumps({ticker: idx for idx, ticker in enumerate(valid_tickers)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"   🗂️ AI Elite: Building file cache ({len(valid_tickers)} tickers, {n_dates} dates)...")
    _populate_ai_elite_ticker_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        start_ordinal=start_ordinal,
        fill_start_ordinal=start_ordinal,
        fill_end_ordinal=end_ordinal,
        forward_days=forward_days,
        latest_allowed_date=latest_allowed_date,
        daily_feature_path=daily_feature_path,
        intraday_feature_path=intraday_feature_path,
        forward_path=forward_path,
        valid_mask_path=valid_mask_path,
        intraday_coverage_path=intraday_coverage_path,
        market_context_map=market_context_map,
        price_history_cache=price_history_cache,
        hourly_history_cache=hourly_history_cache,
        progress_label="build",
    )

    daily_feature_array.flush()
    intraday_feature_array.flush()
    forward_array.flush()
    valid_mask_array.flush()
    intraday_coverage_array.flush()

    context = {
        "all_tickers": valid_tickers,
        "start_ordinal": start_ordinal,
        "n_dates": n_dates,
        "daily_feature_path": str(daily_feature_path),
        "intraday_feature_path": str(intraday_feature_path),
        "forward_path": str(forward_path),
        "valid_mask_path": str(valid_mask_path),
        "intraday_coverage_path": str(intraday_coverage_path),
        "date_index_path": str(date_index_path),
        "ticker_index_path": str(ticker_index_path),
        "forward_days": int(forward_days),
        "latest_allowed_ordinal": _normalize_ai_elite_timestamp(latest_allowed_date).to_pydatetime().date().toordinal(),
        "persistent": True,
        "daily_feature_names": list(AI_ELITE_DAILY_FEATURE_NAMES),
        "intraday_feature_names": list(AI_ELITE_INTRADAY_FEATURE_NAMES),
        "feature_names": list(AI_ELITE_FEATURE_NAMES),
        "numeric_dtype": AI_ELITE_CACHE_NUMERIC_DTYPE,
        "temp_dir": str(cache_dir),
        "daily_signature": daily_signature,
        "hourly_signature": hourly_signature,
    }
    _save_ai_elite_context_metadata(context)
    print("   💾 AI Elite: File cache ready")
    return context


def _extend_ai_elite_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    train_end_date: datetime,
    forward_days: int,
    context: Dict[str, object],
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
    price_history_cache=None,
    hourly_history_cache=None,
) -> Dict[str, object]:
    start_ordinal = int(context["start_ordinal"])
    old_n_dates = int(context["n_dates"])
    old_end_ordinal = start_ordinal + old_n_dates - 1
    requested_end_ordinal = _normalize_ai_elite_timestamp(train_end_date).to_pydatetime().date().toordinal()
    if requested_end_ordinal <= old_end_ordinal:
        return context

    new_n_dates = requested_end_ordinal - start_ordinal + 1
    cache_dir = Path(str(context["temp_dir"]))
    old_daily_path = Path(str(context["daily_feature_path"]))
    old_intraday_path = Path(str(context["intraday_feature_path"]))
    old_forward_path = Path(str(context["forward_path"]))
    old_valid_mask_path = Path(str(context["valid_mask_path"]))
    old_intraday_coverage_path = Path(str(context["intraday_coverage_path"]))
    old_date_index_path = Path(str(context["date_index_path"]))

    daily_feature_path = cache_dir / f"daily_features_{requested_end_ordinal}.npy"
    intraday_feature_path = cache_dir / f"intraday_features_{requested_end_ordinal}.npy"
    forward_path = cache_dir / f"forward_returns_{requested_end_ordinal}.npy"
    valid_mask_path = cache_dir / f"valid_mask_{requested_end_ordinal}.npy"
    intraday_coverage_path = cache_dir / f"intraday_coverage_{requested_end_ordinal}.npy"
    date_index_path = cache_dir / f"date_ordinals_{requested_end_ordinal}.npy"

    old_daily_array = np.load(str(old_daily_path), mmap_mode="r")
    old_intraday_array = np.load(str(old_intraday_path), mmap_mode="r")
    old_forward_array = np.load(str(old_forward_path), mmap_mode="r")
    old_valid_mask_array = np.load(str(old_valid_mask_path), mmap_mode="r")
    old_intraday_coverage_array = np.load(str(old_intraday_coverage_path), mmap_mode="r")

    daily_feature_array = np.lib.format.open_memmap(
        daily_feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), new_n_dates, len(AI_ELITE_DAILY_FEATURE_NAMES)),
    )
    intraday_feature_array = np.lib.format.open_memmap(
        intraday_feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), new_n_dates, len(AI_ELITE_INTRADAY_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), new_n_dates),
    )
    valid_mask_array = np.lib.format.open_memmap(
        valid_mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(valid_tickers), new_n_dates),
    )
    intraday_coverage_array = np.lib.format.open_memmap(
        intraday_coverage_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(valid_tickers), new_n_dates),
    )

    daily_feature_array.fill(np.nan)
    intraday_feature_array.fill(0.0)
    forward_array.fill(np.nan)
    valid_mask_array.fill(False)
    intraday_coverage_array.fill(False)

    daily_feature_array[:, :old_n_dates] = old_daily_array
    intraday_feature_array[:, :old_n_dates] = old_intraday_array
    forward_array[:, :old_n_dates] = old_forward_array
    valid_mask_array[:, :old_n_dates] = old_valid_mask_array
    intraday_coverage_array[:, :old_n_dates] = old_intraday_coverage_array
    np.save(date_index_path, np.arange(start_ordinal, start_ordinal + new_n_dates, dtype=np.int32))

    print(
        f"   ♻️ AI Elite: Extending file cache by {requested_end_ordinal - old_end_ordinal} day(s) "
        f"to {new_n_dates} total cached dates"
    )
    _populate_ai_elite_ticker_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        start_ordinal=start_ordinal,
        fill_start_ordinal=old_end_ordinal + 1,
        fill_end_ordinal=requested_end_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
        daily_feature_path=daily_feature_path,
        intraday_feature_path=intraday_feature_path,
        forward_path=forward_path,
        valid_mask_path=valid_mask_path,
        intraday_coverage_path=intraday_coverage_path,
        market_context_map=market_context_map,
        price_history_cache=price_history_cache,
        hourly_history_cache=hourly_history_cache,
        progress_label="append",
    )
    _refresh_ai_elite_forward_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        forward_path=forward_path,
        start_ordinal=start_ordinal,
        refresh_start_ordinal=max(start_ordinal, old_end_ordinal - forward_days + 1),
        refresh_end_ordinal=requested_end_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
    )

    daily_feature_array.flush()
    intraday_feature_array.flush()
    forward_array.flush()
    valid_mask_array.flush()
    intraday_coverage_array.flush()

    del old_daily_array
    del old_intraday_array
    del old_forward_array
    del old_valid_mask_array
    del old_intraday_coverage_array
    for old_path in (
        old_daily_path,
        old_intraday_path,
        old_forward_path,
        old_valid_mask_path,
        old_intraday_coverage_path,
        old_date_index_path,
    ):
        try:
            old_path.unlink(missing_ok=True)
        except Exception:
            pass

    updated_context = dict(context)
    updated_context.update(
        {
            "n_dates": new_n_dates,
            "daily_feature_path": str(daily_feature_path),
            "intraday_feature_path": str(intraday_feature_path),
            "forward_path": str(forward_path),
            "valid_mask_path": str(valid_mask_path),
            "intraday_coverage_path": str(intraday_coverage_path),
            "date_index_path": str(date_index_path),
            "latest_allowed_ordinal": requested_end_ordinal,
        }
    )
    _save_ai_elite_context_metadata(updated_context)
    print("   💾 AI Elite: File cache ready")
    return updated_context


def _refresh_existing_ai_elite_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    train_end_date: datetime,
    forward_days: int,
    context: Dict[str, object],
) -> Dict[str, object]:
    start_ordinal = int(context["start_ordinal"])
    latest_allowed_ordinal = _normalize_ai_elite_timestamp(train_end_date).to_pydatetime().date().toordinal()
    old_latest_allowed_ordinal = int(context.get("latest_allowed_ordinal", start_ordinal - 1))
    if latest_allowed_ordinal <= old_latest_allowed_ordinal:
        return context

    print(
        f"   ♻️ AI Elite: Refreshing forward labels by "
        f"{latest_allowed_ordinal - old_latest_allowed_ordinal} day(s)"
    )
    _refresh_ai_elite_forward_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        forward_path=Path(str(context["forward_path"])),
        start_ordinal=start_ordinal,
        refresh_start_ordinal=max(start_ordinal, old_latest_allowed_ordinal - forward_days + 1),
        refresh_end_ordinal=latest_allowed_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
    )
    updated_context = dict(context)
    updated_context["latest_allowed_ordinal"] = latest_allowed_ordinal
    _save_ai_elite_context_metadata(updated_context)
    return updated_context


def _build_ai_elite_cache_rows(args) -> None:
    ticker_idx, ticker = args
    context = _get_ai_elite_file_cache_context(mmap_mode="r+")
    latest_allowed_date = context["latest_allowed_date"]
    price_history_cache = context.get("price_history_cache")
    forward_source_data = None
    daily_data = _get_ai_elite_daily_frame_from_price_cache(
        ticker,
        latest_allowed_date,
        price_history_cache,
    )
    if daily_data is None:
        ticker_df = context["ticker_data_grouped"].get(ticker)
        if ticker_df is None or len(ticker_df) == 0:
            return
        daily_data = ticker_df.loc[:latest_allowed_date]
        if daily_data is None or len(daily_data) == 0:
            return
        forward_source_data = daily_data
    else:
        forward_source_data = daily_data

    daily_context = _prepare_ai_elite_daily_context(
        daily_data,
        market_context_map=context.get("market_context_map"),
    )
    hourly_data = get_cached_hourly_frame_up_to(
        ensure_hourly_history_cache(context.get("hourly_history_cache")),
        ticker,
        latest_allowed_date,
        field_names=("open", "high", "low", "close", "volume"),
        min_rows=2,
    )
    if hourly_data is not None and not hourly_data.empty:
        hourly_data = hourly_data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    hourly_context = _prepare_ai_elite_hourly_context(hourly_data)

    daily_feature_array = context["_daily_feature_array"]
    intraday_feature_array = context["_intraday_feature_array"]
    forward_array = context["_forward_array"]
    valid_mask_array = context["_valid_mask_array"]
    intraday_coverage_array = context["_intraday_coverage_array"]
    start_ordinal = int(context["start_ordinal"])
    fill_start_ordinal = int(context["fill_start_ordinal"])
    fill_end_ordinal = int(context["fill_end_ordinal"])
    forward_days = int(context["forward_days"])

    for ordinal in range(fill_start_ordinal, fill_end_ordinal + 1):
        current_date = _ai_elite_datetime_from_ordinal(ordinal, context["latest_allowed_date"])
        current_ts = _normalize_ai_elite_timestamp(current_date)
        date_idx = ordinal - start_ordinal
        feature_row, intraday_covered = _build_ai_elite_feature_row(
            current_ts=current_ts,
            daily_context=daily_context,
            hourly_context=hourly_context,
        )
        if feature_row is not None:
            daily_feature_array[ticker_idx, date_idx] = _ai_elite_daily_vector_from_features(feature_row)
            intraday_feature_array[ticker_idx, date_idx] = _ai_elite_intraday_vector_from_features(feature_row)
            valid_mask_array[ticker_idx, date_idx] = True
            intraday_coverage_array[ticker_idx, date_idx] = bool(intraday_covered)
        else:
            valid_mask_array[ticker_idx, date_idx] = False
            intraday_coverage_array[ticker_idx, date_idx] = False

        forward_ret = _calculate_forward_return(forward_source_data, current_date, forward_days)
        if forward_ret is not None:
            forward_array[ticker_idx, date_idx] = float(forward_ret)
        else:
            forward_array[ticker_idx, date_idx] = np.nan


def _refresh_ai_elite_forward_rows(args) -> None:
    ticker_idx, ticker = args
    context = _get_ai_elite_file_cache_context(mmap_mode="r+")
    ticker_df = context["ticker_data_grouped"].get(ticker)
    if ticker_df is None or len(ticker_df) == 0:
        return

    forward_array = context["_forward_array"]
    start_ordinal = int(context["start_ordinal"])
    refresh_start_ordinal = int(context["refresh_start_ordinal"])
    refresh_end_ordinal = int(context["refresh_end_ordinal"])
    forward_days = int(context["forward_days"])
    latest_allowed_date = context["latest_allowed_date"]

    for ordinal in range(refresh_start_ordinal, refresh_end_ordinal + 1):
        current_date = _ai_elite_datetime_from_ordinal(ordinal, latest_allowed_date)
        date_idx = ordinal - start_ordinal
        forward_ret = _calculate_forward_return(ticker_df, current_date, forward_days)
        if forward_ret is not None:
            forward_array[ticker_idx, date_idx] = float(forward_ret)
        else:
            forward_array[ticker_idx, date_idx] = np.nan


def _populate_ai_elite_ticker_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    start_ordinal: int,
    fill_start_ordinal: int,
    fill_end_ordinal: int,
    forward_days: int,
    latest_allowed_date: datetime,
    daily_feature_path: Path,
    intraday_feature_path: Path,
    forward_path: Path,
    valid_mask_path: Path,
    intraday_coverage_path: Path,
    market_context_map: Optional[Dict[pd.Timestamp, Dict[str, float]]],
    price_history_cache,
    hourly_history_cache,
    progress_label: str,
) -> None:
    if fill_end_ordinal < fill_start_ordinal:
        return

    from config import NUM_PROCESSES

    worker_args = list(enumerate(valid_tickers))
    n_workers = max(1, min(NUM_PROCESSES, len(worker_args))) if worker_args else 1
    cache_context = {
        "ticker_data_grouped": ticker_data_grouped,
        "price_history_cache": price_history_cache,
        "hourly_history_cache": hourly_history_cache,
        "daily_feature_path": str(daily_feature_path),
        "intraday_feature_path": str(intraday_feature_path),
        "forward_path": str(forward_path),
        "valid_mask_path": str(valid_mask_path),
        "intraday_coverage_path": str(intraday_coverage_path),
        "market_context_map": market_context_map or {},
        "start_ordinal": int(start_ordinal),
        "fill_start_ordinal": int(fill_start_ordinal),
        "fill_end_ordinal": int(fill_end_ordinal),
        "forward_days": int(forward_days),
        "latest_allowed_date": latest_allowed_date,
    }
    temp_context_path: Optional[str] = None
    chunksize = max(1, len(worker_args) // (n_workers * 4))
    try:
        if os.name != "nt":
            global _AI_ELITE_FILE_CACHE_CONTEXT_PATH, _AI_ELITE_FILE_CACHE_CONTEXT
            _AI_ELITE_FILE_CACHE_CONTEXT_PATH = None
            _AI_ELITE_FILE_CACHE_CONTEXT = cache_context
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=_init_ai_elite_file_cache_worker,
                initargs=(None, cache_context),
            ) as pool:
                result_iter = pool.imap_unordered(
                    _build_ai_elite_cache_rows,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    result_iter,
                    total=len(worker_args),
                    desc=f"   AI Elite cache {progress_label}",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
        else:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                temp_context_path = temp_context_file.name
            joblib.dump(cache_context, temp_context_path)
            with get_context("spawn").Pool(
                processes=n_workers,
                initializer=_init_ai_elite_file_cache_worker,
                initargs=(temp_context_path,),
            ) as pool:
                result_iter = pool.imap_unordered(
                    _build_ai_elite_cache_rows,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    result_iter,
                    total=len(worker_args),
                    desc=f"   AI Elite cache {progress_label}",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
    finally:
        _AI_ELITE_FILE_CACHE_CONTEXT_PATH = None
        _AI_ELITE_FILE_CACHE_CONTEXT = None
        if temp_context_path:
            try:
                os.unlink(temp_context_path)
            except OSError:
                pass


def _refresh_ai_elite_forward_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    forward_path: Path,
    start_ordinal: int,
    refresh_start_ordinal: int,
    refresh_end_ordinal: int,
    forward_days: int,
    latest_allowed_date: datetime,
) -> None:
    if refresh_end_ordinal < refresh_start_ordinal:
        return

    from config import NUM_PROCESSES

    worker_args = list(enumerate(valid_tickers))
    n_workers = max(1, min(NUM_PROCESSES, len(worker_args))) if worker_args else 1
    cache_context = {
        "ticker_data_grouped": ticker_data_grouped,
        "forward_path": str(forward_path),
        "start_ordinal": int(start_ordinal),
        "refresh_start_ordinal": int(refresh_start_ordinal),
        "refresh_end_ordinal": int(refresh_end_ordinal),
        "forward_days": int(forward_days),
        "latest_allowed_date": latest_allowed_date,
    }
    temp_context_path: Optional[str] = None
    chunksize = max(1, len(worker_args) // (n_workers * 4))
    try:
        if os.name != "nt":
            global _AI_ELITE_FILE_CACHE_CONTEXT_PATH, _AI_ELITE_FILE_CACHE_CONTEXT
            _AI_ELITE_FILE_CACHE_CONTEXT_PATH = None
            _AI_ELITE_FILE_CACHE_CONTEXT = cache_context
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=_init_ai_elite_file_cache_worker,
                initargs=(None, cache_context),
            ) as pool:
                result_iter = pool.imap_unordered(
                    _refresh_ai_elite_forward_rows,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    result_iter,
                    total=len(worker_args),
                    desc="   AI Elite cache refresh",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
        else:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                temp_context_path = temp_context_file.name
            joblib.dump(cache_context, temp_context_path)
            with get_context("spawn").Pool(
                processes=n_workers,
                initializer=_init_ai_elite_file_cache_worker,
                initargs=(temp_context_path,),
            ) as pool:
                result_iter = pool.imap_unordered(
                    _refresh_ai_elite_forward_rows,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    result_iter,
                    total=len(worker_args),
                    desc="   AI Elite cache refresh",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
    finally:
        _AI_ELITE_FILE_CACHE_CONTEXT_PATH = None
        _AI_ELITE_FILE_CACHE_CONTEXT = None
        if temp_context_path:
            try:
                os.unlink(temp_context_path)
            except OSError:
                pass


def get_ai_elite_cached_feature_payload(
    feature_cache_context: Optional[Dict[str, object]],
    ticker: str,
    current_date: datetime,
    require_intraday_coverage: bool = False,
) -> Optional[Dict[str, object]]:
    context = _ensure_ai_elite_feature_cache_context(feature_cache_context, mmap_mode="r")
    if context is None:
        return None

    ticker_to_idx = context.get("ticker_to_idx") or {}
    ticker_idx = ticker_to_idx.get(ticker)
    if ticker_idx is None:
        return None

    current_ordinal = _normalize_ai_elite_timestamp(current_date).to_pydatetime().date().toordinal()
    start_ordinal = int(context.get("start_ordinal", 0))
    n_dates = int(context.get("n_dates", 0))
    date_idx = current_ordinal - start_ordinal
    if date_idx < 0 or date_idx >= n_dates:
        return None

    valid_mask_array = context.get("_valid_mask_array")
    if valid_mask_array is None or not bool(valid_mask_array[ticker_idx, date_idx]):
        return None

    intraday_coverage_array = context.get("_intraday_coverage_array")
    has_hourly_data = bool(intraday_coverage_array[ticker_idx, date_idx]) if intraday_coverage_array is not None else False
    if require_intraday_coverage and not has_hourly_data:
        return None

    daily_feature_array = context.get("_daily_feature_array")
    intraday_feature_array = context.get("_intraday_feature_array")
    if daily_feature_array is None or intraday_feature_array is None:
        return None

    features = _ai_elite_feature_dict_from_vectors(
        np.asarray(daily_feature_array[ticker_idx, date_idx], dtype=np.float64),
        np.asarray(intraday_feature_array[ticker_idx, date_idx], dtype=np.float64),
    )
    return {
        "features": features,
        "has_hourly_data": has_hourly_data,
    }


def _load_hourly_data_direct(
    ticker: str,
    start: datetime,
    end: datetime,
    hourly_history_cache=None,
) -> Optional[pd.DataFrame]:
    """
    Load cached 1-hour data without converting to daily.
    Uses the same cache as load_prices but stops before daily conversion.
    """
    try:
        hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
        result = get_cached_hourly_frame_between(
            hourly_history_cache,
            ticker,
            start,
            end,
            field_names=("open", "high", "low", "close", "volume"),
            min_rows=120,
        )
        if result is None or result.empty:
            return None
        return result.rename(
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


def _load_or_create_model(model_path: Optional[str] = None):
    """Load existing ML model from disk. Returns None if no model exists."""
    if model_path and os.path.exists(model_path):
        try:
            import joblib

            try:
                model_data = joblib.load(model_path)
            except Exception:
                import pickle

                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            model_data = restore_native_model_artifacts(model_data, model_path)
            # Handle both old format (direct model) and new format (model + metadata)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                metadata = model_data.get('metadata', {})
                info_parts = []
                if 'trained' in metadata:
                    info_parts.append(f"trained {metadata['trained'][:10]}")
                elif 'updated' in metadata:
                    info_parts.append(f"updated {metadata['updated'][:10]}")
                if 'train_start' in metadata and 'train_end' in metadata:
                    info_parts.append(f"data {metadata['train_start'][:10]} to {metadata['train_end'][:10]}")
                info_str = f" ({', '.join(info_parts)})" if info_parts else ""
                print(f"   ✅ AI Elite: Loaded ML model from {model_path}{info_str}")
            else:
                model = model_data
                print(f"   ✅ AI Elite: Loaded ML model from {model_path} (legacy format)")
            return model
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to load model: {e}")

    print(f"   ⚠️ AI Elite: No saved model found at {model_path}. Run backtesting first to train.")
    return None


def _load_shared_ai_elite_model_for_inference(
    model_path: str,
    model_token: Optional[str] = None,
):
    """Load and cache the shared AI Elite model for inference-only worker execution."""
    cache_key = (str(model_path), model_token)
    cached_model = _AI_ELITE_SHARED_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    model = _load_or_create_model(model_path)
    if model is None:
        return None

    _AI_ELITE_SHARED_MODEL_CACHE[cache_key] = model
    stale_keys = [key for key in _AI_ELITE_SHARED_MODEL_CACHE.keys() if key[0] == str(model_path) and key != cache_key]
    for stale_key in stale_keys:
        _AI_ELITE_SHARED_MODEL_CACHE.pop(stale_key, None)
    return model


# NOTE: train_ai_elite_model was removed - training now happens in ai_elite_strategy_per_ticker.py
# via train_shared_base_model(), called from shared_strategies.select_ai_elite_with_training()


def _calculate_market_return(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    forward_days: int,
    market_ticker: str = 'SPY'  # Use SPY as market proxy
) -> Optional[float]:
    """
    Calculate market return for the same period.
    Uses SPY as market proxy, falls back to equal-weighted average of available stocks.
    """
    try:
        # Try SPY first
        if market_ticker in ticker_data_grouped:
            market_data = ticker_data_grouped[market_ticker]
            market_return = _calculate_forward_return(market_data, current_date, forward_days)
            if market_return is not None:
                return market_return

        # Fallback: equal-weighted average of all available stocks
        returns = []
        for ticker, data in ticker_data_grouped.items():
            if len(data) > 0:
                ret = _calculate_forward_return(data, current_date, forward_days)
                if ret is not None:
                    returns.append(ret)

        if returns:
            return np.mean(returns)
        return None

    except Exception as e:
        return None


def _calculate_forward_return(
    ticker_data: pd.DataFrame,
    current_date: datetime,
    forward_days: int
) -> Optional[float]:
    """
    Calculate forward return for a stock over next N days.

    Args:
        ticker_data: Historical price data for ticker
        current_date: Current date
        forward_days: Number of days to look ahead

    Returns:
        Forward return percentage or None if insufficient data
    """
    try:
        # ✅ FIX: Deduplicate index (hourly data combined creates duplicates)
        if ticker_data.index.duplicated().any():
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]

        # Convert current_date to pandas Timestamp - ensure UTC-aware
        current_date_tz = pd.Timestamp(current_date)
        if current_date_tz.tz is None:
            current_date_tz = current_date_tz.tz_localize('UTC')
        elif str(current_date_tz.tz) != 'UTC':
            current_date_tz = current_date_tz.tz_convert('UTC')

        # Ensure index is also UTC-aware
        if ticker_data.index.tz is None:
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_localize('UTC')
        elif str(ticker_data.index.tz) != 'UTC':
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_convert('UTC')

        # Get current price
        current_data = ticker_data[ticker_data.index <= current_date_tz]
        if len(current_data) == 0:
            return None
        close_col = current_data['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        current_price = close_col.iloc[-1]

        # Get future price
        future_date = current_date_tz + timedelta(days=forward_days)
        future_data = ticker_data[(ticker_data.index > current_date_tz) &
                                  (ticker_data.index <= future_date)]

        if len(future_data) == 0:
            return None

        future_close = future_data['Close']
        if isinstance(future_close, pd.DataFrame):
            future_close = future_close.iloc[:, 0]
        future_price = future_close.iloc[-1]

        # Calculate return
        forward_return = ((future_price - current_price) / current_price) * 100

        return forward_return

    except Exception as e:
        return None


def _predict_ticker_worker(args):
    """Predict score for one ticker and return reusable feature payload."""
    (
        ticker,
        ticker_data,
        current_date,
        ticker_model,
        price_history_cache,
        hourly_history_cache,
        feature_cache_context,
        market_context_map,
        timing_stats,
    ) = args

    try:
        if ticker_data is None or len(ticker_data) == 0:
            return ticker, None, 'empty', None

        feature_start = time.perf_counter()
        cached_payload = get_ai_elite_cached_feature_payload(
            feature_cache_context,
            ticker,
            current_date,
            require_intraday_coverage=True,
        )
        _record_prediction_timing(timing_stats, "feature", time.perf_counter() - feature_start)
        if cached_payload is not None:
            features = dict(cached_payload["features"])
            payload = {
                "features": features,
                "has_hourly_data": bool(cached_payload.get("has_hourly_data")),
            }
        else:
            hourly_start = time.perf_counter()
            hourly_data = _load_hourly_data_direct(
                ticker,
                current_date - timedelta(days=30),
                current_date + timedelta(days=5),
                hourly_history_cache=hourly_history_cache,
            )
            _record_prediction_timing(timing_stats, "hourly", time.perf_counter() - hourly_start)

            feature_start = time.perf_counter()
            features = _extract_features(
                ticker,
                hourly_data,
                current_date,
                daily_data=ticker_data,
                price_history_cache=price_history_cache,
                market_context_map=market_context_map,
            )
            _record_prediction_timing(timing_stats, "feature", time.perf_counter() - feature_start)
            if features is None:
                return ticker, None, 'features_none', None

            payload = {
                "features": dict(features),
                "has_hourly_data": hourly_data is not None and len(hourly_data) > 0,
            }

        # Get model and predict
        if ticker_model is None:
            return ticker, 0.0, 'no_model', payload

        # Extract the actual model from ensemble dict
        if isinstance(ticker_model, dict):
            actual_model = ticker_model.get('best_model')
            if actual_model is None:
                return ticker, 0.0, 'no_model', payload
        else:
            actual_model = ticker_model

        from ai_elite_strategy_per_ticker import FEATURE_COLS
        feature_values = _build_feature_frame(features, FEATURE_COLS)
        model_start = time.perf_counter()
        ai_score = actual_model.predict(feature_values)[0]
        _record_prediction_timing(timing_stats, "model", time.perf_counter() - model_start)

        return ticker, ai_score, 'success', payload
    except Exception as e:
        import traceback
        print(f"   ⚠️ AI Elite worker exception for {ticker}: {e}")
        traceback.print_exc()
        return ticker, None, 'exception', None


def _predict_ticker_ensemble_worker(args):
    """Predict using the same single best model path as regular AI Elite."""
    ticker, score, status, _payload = _predict_ticker_worker(args)
    return ticker, score, status


def select_ai_elite_ensemble_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None,
    price_history_cache=None,
    hourly_history_cache=None,
    feature_cache_context=None,
    market_context_map=None,
) -> List[str]:
    """
    AI Elite Ensemble Strategy: Weighted average of top 3 positive-R² models.

    Must run AFTER regular AI Elite so models are already trained.
    """
    if current_date is None:
        latest_dates = [
            ticker_data_grouped[t].index.max()
            for t in all_tickers
            if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0
        ]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
    if market_context_map is None:
        market_context_map = precompute_ai_elite_market_context_map(
            ticker_data_grouped,
            current_date,
            current_date,
        )

    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite Ensemble"
    )

    print(f"   🤖 AI Elite Ensemble: Analyzing {len(filtered_tickers)} tickers (best-model scoring)")

    from config import NUM_PROCESSES, PARALLEL_THRESHOLD

    start_time = time.time()
    timing_stats = _create_prediction_timing_stats()

    ai_scores = {}
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'no_model': 0, 'exception': 0}

    def _record_result(ticker_result, score, status):
        if status == 'success':
            ai_scores[ticker_result] = score
        else:
            fail_reasons[status] += 1

    def _run_sequential():
        for args in tqdm(
            predict_args,
            total=len(predict_args),
            desc="AI Elite Ensemble prediction",
            unit="ticker",
        ):
            ticker_result, score, status = _predict_ticker_ensemble_worker(args)
            _record_result(ticker_result, score, status)

    predict_args = []
    for ticker in filtered_tickers:
        ticker_data = ticker_data_grouped.get(ticker)
        ticker_model = per_ticker_models.get(ticker) if per_ticker_models else None
        predict_args.append((
            ticker,
            ticker_data,
            current_date,
            ticker_model,
            price_history_cache,
            hourly_history_cache,
            feature_cache_context,
            market_context_map,
            timing_stats,
        ))

    n_workers = min(max(1, NUM_PROCESSES), len(predict_args)) if predict_args else 1
    use_parallel = n_workers > 1 and len(predict_args) >= PARALLEL_THRESHOLD

    if use_parallel:
        if os.name != "nt":
            print(f"   🚀 AI Elite Ensemble: Predicting with {n_workers} fork workers")
            predict_context = {
                "ticker_data_grouped": ticker_data_grouped,
                "per_ticker_models": per_ticker_models,
                "current_date": current_date,
                "price_history_cache": price_history_cache,
                "hourly_history_cache": hourly_history_cache,
                "feature_cache_context": feature_cache_context,
                "market_context_map": market_context_map,
            }
            chunksize = max(1, len(filtered_tickers) // (n_workers * 4))
            try:
                with get_context("fork").Pool(
                    processes=n_workers,
                    initializer=_init_ai_elite_predict_worker,
                    initargs=(predict_context,),
                ) as pool:
                    results = pool.imap_unordered(
                        _predict_ticker_ensemble_from_context_worker,
                        filtered_tickers,
                        chunksize=chunksize,
                    )
                    for ticker_result, score, status in tqdm(
                        results,
                        total=len(filtered_tickers),
                        desc="AI Elite Ensemble prediction",
                        unit="ticker",
                    ):
                        _record_result(ticker_result, score, status)
            except Exception as e:
                print(f"   ⚠️ AI Elite Ensemble: Forked prediction failed ({type(e).__name__}: {e})")
                return []
        else:
            print(f"   🧵 AI Elite Ensemble: Predicting with {n_workers} threads")
            try:
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ai-ensemble") as executor:
                    futures = [executor.submit(_predict_ticker_ensemble_worker, args) for args in predict_args]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="AI Elite Ensemble prediction",
                        unit="ticker",
                    ):
                        ticker_result, score, status = future.result()
                        _record_result(ticker_result, score, status)
            except Exception as e:
                print(f"   ⚠️ AI Elite Ensemble: Threaded prediction failed ({type(e).__name__}: {e})")
                return []
    else:
        _run_sequential()

    elapsed = time.time() - start_time
    print(f"   📊 AI Elite Ensemble: Predicted {len(ai_scores)} tickers ({elapsed:.1f}s)")
    _print_prediction_timing("AI Elite Ensemble", timing_stats)

    if not ai_scores:
        print(f"   ⚠️ AI Elite Ensemble: No predictions")
        print(f"   🔍 AI Elite Ensemble: Fail reasons: {fail_reasons}")
        return []

    # Sort by ensemble score, return top N
    sorted_tickers = sorted(ai_scores.items(), key=lambda x: x[1], reverse=True)

    # Debug top 5
    print(f"   📊 AI Elite Ensemble: Top 5 by best-model score:")
    for ticker, score in sorted_tickers[:5]:
        print(f"      {ticker}: {score:+.2f}")

    selected = [t for t, _ in sorted_tickers[:top_n]]
    return selected


def _get_top_positive_ai_elite_models(model_bundle, top_k: int = 3):
    """Return the top positive-R² ensemble members from one saved AI Elite bundle."""
    if not isinstance(model_bundle, dict):
        return []

    all_models = model_bundle.get('all_models') or {}
    all_scores = model_bundle.get('all_scores') or {}
    positive_models = [
        (name, all_scores[name])
        for name in all_models
        if all_scores.get(name, -999) > 0
    ]
    positive_models.sort(key=lambda x: x[1], reverse=True)
    return positive_models[:top_k]


def _get_representative_ai_elite_model(per_ticker_models):
    """Find one representative shared AI Elite model bundle."""
    if not isinstance(per_ticker_models, dict):
        return None

    shared_base = per_ticker_models.get('_shared_base')
    if isinstance(shared_base, dict):
        return shared_base

    for model_bundle in per_ticker_models.values():
        if isinstance(model_bundle, dict):
            return model_bundle

    return None


def _predict_ticker_rank_ensemble_worker(args):
    """Predict using the same single best model path as regular AI Elite."""
    (
        ticker,
        ticker_data,
        current_date,
        ticker_model,
        selected_model_names,
        price_history_cache,
        hourly_history_cache,
        feature_cache_context,
        market_context_map,
        timing_stats,
    ) = args

    ticker, score, status, _payload = _predict_ticker_worker(
        (
            ticker,
            ticker_data,
            current_date,
            ticker_model,
            price_history_cache,
            hourly_history_cache,
            feature_cache_context,
            market_context_map,
            timing_stats,
        )
    )
    return ticker, score, status


def select_ai_elite_rank_ensemble_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None,
    price_history_cache=None,
    hourly_history_cache=None,
    feature_cache_context=None,
    market_context_map=None,
) -> List[str]:
    """
    AI Elite Rank Ensemble Strategy: weighted rank average of top 3 positive-R² models.

    Must run AFTER regular AI Elite so models are already trained.
    """
    if current_date is None:
        latest_dates = [
            ticker_data_grouped[t].index.max()
            for t in all_tickers
            if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0
        ]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    hourly_history_cache = ensure_hourly_history_cache(hourly_history_cache)
    if market_context_map is None:
        market_context_map = precompute_ai_elite_market_context_map(
            ticker_data_grouped,
            current_date,
            current_date,
        )

    print("   🤖 AI Elite Rank Ensemble: Using best-model scoring")

    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite Rank Ensemble"
    )

    print(f"   🤖 AI Elite Rank Ensemble: Analyzing {len(filtered_tickers)} tickers (best-model rank)")

    from config import NUM_PROCESSES, PARALLEL_THRESHOLD

    start_time = time.time()
    timing_stats = _create_prediction_timing_stats()

    per_ticker_predictions = {}
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'no_model': 0, 'exception': 0}

    def _record_result(ticker_result, score_map, status):
        if status == 'success':
            per_ticker_predictions[ticker_result] = score_map
        else:
            fail_reasons[status] += 1

    def _run_sequential():
        for args in predict_args:
            ticker_result, score, status = _predict_ticker_rank_ensemble_worker(args)
            _record_result(ticker_result, score, status)

    predict_args = []
    for ticker in filtered_tickers:
        ticker_data = ticker_data_grouped.get(ticker)
        ticker_model = per_ticker_models.get(ticker) if per_ticker_models else None
        predict_args.append((
            ticker,
            ticker_data,
            current_date,
            ticker_model,
            None,
            price_history_cache,
            hourly_history_cache,
            feature_cache_context,
            market_context_map,
            timing_stats,
        ))

    n_workers = min(max(1, NUM_PROCESSES), len(predict_args)) if predict_args else 1
    use_parallel = n_workers > 1 and len(predict_args) >= PARALLEL_THRESHOLD

    if use_parallel:
        if os.name != "nt":
            print(f"   🚀 AI Elite Rank Ensemble: Predicting with {n_workers} fork workers")
            predict_context = {
                "ticker_data_grouped": ticker_data_grouped,
                "per_ticker_models": per_ticker_models,
                "selected_model_names": None,
                "current_date": current_date,
                "price_history_cache": price_history_cache,
                "hourly_history_cache": hourly_history_cache,
                "feature_cache_context": feature_cache_context,
                "market_context_map": market_context_map,
            }
            chunksize = max(1, len(filtered_tickers) // (n_workers * 4))
            try:
                with get_context("fork").Pool(
                    processes=n_workers,
                    initializer=_init_ai_elite_predict_worker,
                    initargs=(predict_context,),
                ) as pool:
                    results = pool.imap_unordered(
                        _predict_ticker_rank_ensemble_from_context_worker,
                        filtered_tickers,
                        chunksize=chunksize,
                    )
                    for ticker_result, score, status in tqdm(
                        results,
                        total=len(filtered_tickers),
                        desc="AI Elite Rank Ensemble prediction",
                        unit="ticker",
                    ):
                        _record_result(ticker_result, score, status)
            except Exception as e:
                print(f"   ⚠️ AI Elite Rank Ensemble: Forked prediction failed ({type(e).__name__}: {e})")
                return []
        else:
            print(f"   🧵 AI Elite Rank Ensemble: Predicting with {n_workers} threads")
            try:
                with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="ai-rank-ensemble") as executor:
                    futures = [executor.submit(_predict_ticker_rank_ensemble_worker, args) for args in predict_args]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="AI Elite Rank Ensemble prediction",
                        unit="ticker",
                    ):
                        ticker_result, score, status = future.result()
                        _record_result(ticker_result, score, status)
            except Exception as e:
                print(f"   ⚠️ AI Elite Rank Ensemble: Threaded prediction failed ({type(e).__name__}: {e})")
                return []
    else:
        _run_sequential()

    elapsed = time.time() - start_time
    print(f"   📊 AI Elite Rank Ensemble: Predicted {len(per_ticker_predictions)} tickers ({elapsed:.1f}s)")
    _print_prediction_timing("AI Elite Rank Ensemble", timing_stats)

    if not per_ticker_predictions:
        print("   ⚠️ AI Elite Rank Ensemble: No predictions")
        print(f"   🔍 AI Elite Rank Ensemble: Fail reasons: {fail_reasons}")
        return []

    sorted_scores = pd.Series(per_ticker_predictions, dtype=float).sort_values(ascending=False)
    if sorted_scores.empty:
        print("   ⚠️ AI Elite Rank Ensemble: No rankable predictions")
        print(f"   🔍 AI Elite Rank Ensemble: Fail reasons: {fail_reasons}")
        return []

    print("   📊 AI Elite Rank Ensemble: Top 5 by best-model score:")
    for ticker, score in sorted_scores.head(5).items():
        print(f"      {ticker}: {score:.4f}")

    return sorted_scores.head(top_n).index.tolist()
