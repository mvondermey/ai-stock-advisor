"""
Savitzky-Golay Trend Strategy.

This version keeps the pooled base-model architecture but strengthens the
signal with richer medium-horizon features, explicit market context, and
more conservative selection logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from tqdm import tqdm
import config

from model_training_safety import (
    catboost_has_trained_trees,
    configure_catboost_cpu_continuation,
    ensure_catboost_cpu_metadata,
    release_runtime_memory,
    reset_legacy_catboost_member,
    restore_native_model_artifacts,
    save_native_model_artifacts,
)
from config import (
    SAVGOL_TREND_FALLBACK_TO_MOMENTUM,
    SAVGOL_TREND_FORWARD_DAYS,
    SAVGOL_TREND_HOLD_MARGIN,
    SAVGOL_TREND_LOOKBACK_DAYS,
    SAVGOL_TREND_MAX_WORKERS,
    SAVGOL_TREND_MIN_MODEL_SPEARMAN,
    SAVGOL_TREND_MIN_PREDICTED_EDGE,
    SAVGOL_TREND_MIN_SAMPLES,
    SAVGOL_TREND_MIN_SCORE_SPREAD,
    SAVGOL_TREND_RETRAIN_DAYS,
    XGBOOST_USE_GPU,
)


MODEL_SAVE_DIR = Path("logs/models")
SAVGOL_TREND_MODEL_PATH = MODEL_SAVE_DIR / "savgol_trend_model.joblib"


def _normalize_current_date(data: pd.DataFrame, current_date: datetime) -> datetime:
    """Align naive dates to the DataFrame timezone when needed."""
    if hasattr(data.index, "tz") and data.index.tz is not None and current_date.tzinfo is None:
        return current_date.replace(tzinfo=data.index.tz)
    return current_date


def _timestamp_ns(value: datetime | pd.Timestamp) -> int:
    """Convert timestamps to comparable UTC-naive nanoseconds."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


def _odd_window(length: int, target: int, minimum: int = 5) -> int:
    """Return a valid odd Savitzky-Golay window for the available history."""
    window = min(length, target)
    if window % 2 == 0:
        window -= 1
    return window if window >= minimum else 0


def _safe_pct_change(end_value: float, start_value: float) -> float:
    if start_value <= 0:
        return 0.0
    return (end_value / start_value - 1.0) * 100.0


def _series_pct_change(series: pd.Series, periods: int) -> float:
    """Return trailing percent change over a fixed number of observations."""
    if len(series) <= periods:
        return 0.0
    end_value = float(series.iloc[-1])
    start_value = float(series.iloc[-periods - 1])
    return _safe_pct_change(end_value, start_value)


def _series_volatility_pct(series: pd.Series, periods: int) -> float:
    """Return annualized volatility over a trailing window."""
    if len(series) <= periods:
        return 0.0
    returns = series.pct_change().dropna().tail(periods)
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns) * np.sqrt(252) * 100.0)


def _calculate_market_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    """Build lightweight market context features for the current date."""
    proxy_candidates = ("SPY", "QQQ", "VTI", "DIA")
    proxy_series: Optional[pd.Series] = None

    for ticker in proxy_candidates:
        data = ticker_data_grouped.get(ticker)
        if data is None or "Close" not in data.columns:
            continue
        current_date_norm = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date_norm]
        close = hist["Close"].dropna()
        if len(close) >= 61:
            proxy_series = close
            break

    if proxy_series is not None:
        market_return_5d = _series_pct_change(proxy_series, 5)
        market_return_20d = _series_pct_change(proxy_series, 20)
        market_return_60d = _series_pct_change(proxy_series, 60)
        market_volatility_20d = _series_volatility_pct(proxy_series, 20)
    else:
        market_return_5d = 0.0
        market_return_20d = 0.0
        market_return_60d = 0.0
        market_volatility_20d = 0.0

    breadth_20d_values: List[float] = []
    breadth_60d_values: List[float] = []
    for data in ticker_data_grouped.values():
        if data is None or "Close" not in data.columns:
            continue
        current_date_norm = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date_norm]
        close = hist["Close"].dropna()
        if len(close) >= 21:
            breadth_20d_values.append(_series_pct_change(close, 20))
        if len(close) >= 61:
            breadth_60d_values.append(_series_pct_change(close, 60))
        if len(breadth_20d_values) >= 250 and len(breadth_60d_values) >= 250:
            break

    breadth_20d = float(np.mean(np.array(breadth_20d_values) > 0)) if breadth_20d_values else 0.5
    breadth_60d = float(np.mean(np.array(breadth_60d_values) > 0)) if breadth_60d_values else 0.5

    return {
        "market_return_5d": market_return_5d,
        "market_return_20d": market_return_20d,
        "market_return_60d": market_return_60d,
        "market_volatility_20d": market_volatility_20d,
        "market_breadth_20d": breadth_20d,
        "market_breadth_60d": breadth_60d,
    }


def _market_context_key(current_date: datetime) -> pd.Timestamp:
    """Normalize dates to a timezone-agnostic daily key for cached lookups."""
    return pd.Timestamp(current_date).tz_localize(None).normalize()


def _precompute_market_context_map(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    sample_dates: Optional[List[datetime]] = None,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """Build daily market context once per training run."""
    del forward_days  # SavGol now trains on raw forward return like AI Elite.
    market_context_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    missing_dates: List[datetime] = []
    seen_dates: set[pd.Timestamp] = set()

    if sample_dates is None:
        current_date = train_start_date
        while current_date <= train_end_date:
            cache_key = _market_context_key(current_date)
            if cache_key not in seen_dates:
                seen_dates.add(cache_key)
                missing_dates.append(current_date)
            current_date += timedelta(days=1)
    else:
        for current_date in sample_dates:
            if current_date < train_start_date or current_date > train_end_date:
                continue
            cache_key = _market_context_key(current_date)
            if cache_key not in seen_dates:
                seen_dates.add(cache_key)
                missing_dates.append(current_date)

    if missing_dates:
        with tqdm(
            total=len(missing_dates),
            desc="   SavGol Trend market context",
            ncols=100,
            unit="day",
        ) as pbar:
            for current_date in missing_dates:
                cache_key = _market_context_key(current_date)
                market_context_map[cache_key] = _calculate_market_context(ticker_data_grouped, current_date)
                pbar.update(1)

    return market_context_map


def _prepare_label(raw_label: float) -> float:
    """Clip extreme labels using AI-Elite-style forward-return bounds."""
    return float(np.clip(raw_label, -100.0, 200.0))


def _calculate_savgol_features_from_history(
    close_history: pd.Series,
    high_history: np.ndarray,
    low_history: np.ndarray,
    volume_history: np.ndarray,
    lookback_days: int,
    market_context: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """Shared SavGol feature logic for both direct and index-based collection."""
    if len(close_history) < lookback_days:
        return None

    close = np.asarray(close_history.iloc[-lookback_days:], dtype=float)
    if len(close) < lookback_days or close[-1] <= 0:
        return None

    volume = np.asarray(volume_history[-lookback_days:], dtype=float)
    high = np.asarray(high_history[-lookback_days:], dtype=float)
    low = np.asarray(low_history[-lookback_days:], dtype=float)

    log_close = np.log(np.clip(close, 1e-12, None))
    returns = np.diff(close) / np.clip(close[:-1], 1e-12, None)

    short_window = _odd_window(len(log_close), 9)
    long_window = _odd_window(len(log_close), 21)
    if short_window == 0 or long_window == 0:
        return None

    short_poly = 2 if short_window >= 5 else 1
    long_poly = 3 if long_window >= 7 else 2
    if short_window <= short_poly or long_window <= long_poly:
        return None

    sg_short = savgol_filter(log_close, window_length=short_window, polyorder=short_poly, mode="interp")
    sg_long = savgol_filter(log_close, window_length=long_window, polyorder=long_poly, mode="interp")

    short_slope = (sg_short[-1] - sg_short[-2]) * 100.0 if len(sg_short) >= 2 else 0.0
    long_slope = (sg_long[-1] - sg_long[-2]) * 100.0 if len(sg_long) >= 2 else 0.0
    short_curvature = (
        (sg_short[-1] - 2 * sg_short[-2] + sg_short[-3]) * 10000.0
        if len(sg_short) >= 3
        else 0.0
    )
    long_curvature = (
        (sg_long[-1] - 2 * sg_long[-2] + sg_long[-3]) * 10000.0
        if len(sg_long) >= 3
        else 0.0
    )

    short_diffs = np.diff(sg_short[-10:]) if len(sg_short) >= 10 else np.diff(sg_short)
    long_diffs = np.diff(sg_long[-10:]) if len(sg_long) >= 10 else np.diff(sg_long)
    sg_trend_stability = float(np.mean(short_diffs > 0)) if len(short_diffs) > 0 else 0.5
    sg_long_stability = float(np.mean(long_diffs > 0)) if len(long_diffs) > 0 else 0.5

    x = np.arange(min(10, len(sg_long)))
    sg_regression_slope = (
        float(np.polyfit(x, sg_long[-len(x):], 1)[0] * 100.0)
        if len(x) >= 2
        else 0.0
    )

    residual = log_close - sg_long
    residual_vol_20d = float(np.std(residual[-20:]) * 100.0) if len(residual) >= 20 else 0.0

    high_20 = np.max(close[-20:])
    atr_14 = float(np.mean(high[-14:] - low[-14:])) if len(high) >= 14 and len(low) >= 14 else 0.0
    avg_vol_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
    avg_vol_60 = float(np.mean(volume)) if len(volume) > 0 else 1.0

    perf_5d = _series_pct_change(close_history, 5)
    perf_20d = _series_pct_change(close_history, 20)
    perf_3m = _series_pct_change(close_history, 63)
    perf_6m = _series_pct_change(close_history, 126)
    perf_1y = _series_pct_change(close_history, 252)
    daily_volatility = _series_volatility_pct(close_history, 20)
    risk_adj_mom_3m = perf_3m / (np.sqrt(max(daily_volatility, 5.0)) if daily_volatility > 0 else np.sqrt(5.0))
    dip_score = perf_1y - perf_3m
    mom_accel = perf_3m - perf_6m

    rsi_14 = 50.0
    if len(returns) >= 14:
        gains = np.clip(returns, 0.0, None)
        losses = np.clip(-returns, 0.0, None)
        avg_gain = float(np.mean(gains[-14:]))
        avg_loss = float(np.mean(losses[-14:]))
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain > 0:
            rsi_14 = 100.0

    bollinger_position = 0.5
    sma20_distance = 0.0
    sma50_distance = 0.0
    if len(close_history) >= 20:
        sma20 = float(close_history.tail(20).mean())
        std20 = float(close_history.tail(20).std())
        if sma20 > 0 and std20 > 0:
            upper_band = sma20 + 2.0 * std20
            lower_band = sma20 - 2.0 * std20
            if upper_band > lower_band:
                bollinger_position = float(np.clip((close[-1] - lower_band) / (upper_band - lower_band), 0.0, 1.0))
            sma20_distance = _safe_pct_change(close[-1], sma20)
    if len(close_history) >= 50:
        sma50 = float(close_history.tail(50).mean())
        if sma50 > 0:
            sma50_distance = _safe_pct_change(close[-1], sma50)

    macd = 0.0
    if len(close_history) >= 35:
        ema12 = close_history.ewm(span=12, adjust=False).mean()
        ema26 = close_history.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd = float(macd_line.iloc[-1] - signal_line.iloc[-1])

    context = market_context or {}
    market_return_20d = float(context.get("market_return_20d", 0.0))
    market_return_60d = float(context.get("market_return_60d", 0.0))

    return {
        "mom_5d": perf_5d,
        "mom_10d": _series_pct_change(close_history, 10),
        "mom_20d": perf_20d,
        "mom_40d": _series_pct_change(close_history, 40),
        "perf_3m": perf_3m,
        "perf_6m": perf_6m,
        "perf_1y": perf_1y,
        "risk_adj_mom_3m": risk_adj_mom_3m,
        "dip_score": dip_score,
        "mom_accel": mom_accel,
        "volatility_10d": float(np.std(returns[-10:]) * np.sqrt(252) * 100.0) if len(returns) >= 10 else 0.0,
        "volatility_20d": float(np.std(returns[-20:]) * np.sqrt(252) * 100.0) if len(returns) >= 20 else 0.0,
        "drawdown_20d": _safe_pct_change(close[-1], high_20),
        "atr_pct_14d": (atr_14 / close[-1]) * 100.0 if close[-1] > 0 else 0.0,
        "volume_ratio_20_60": avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else 1.0,
        "rsi_14": rsi_14,
        "bollinger_position": bollinger_position,
        "sma20_distance": sma20_distance,
        "sma50_distance": sma50_distance,
        "macd": macd,
        "price_vs_sg_short": (log_close[-1] - sg_short[-1]) * 100.0,
        "price_vs_sg_long": (log_close[-1] - sg_long[-1]) * 100.0,
        "sg_short_slope": short_slope,
        "sg_long_slope": long_slope,
        "sg_short_curvature": short_curvature,
        "sg_long_curvature": long_curvature,
        "sg_trend_spread": (sg_short[-1] - sg_long[-1]) * 100.0,
        "sg_trend_stability": sg_trend_stability,
        "sg_long_stability": sg_long_stability,
        "sg_regression_slope": sg_regression_slope,
        "sg_residual_vol_20d": residual_vol_20d,
        "market_return_5d": float(context.get("market_return_5d", 0.0)),
        "market_return_20d": market_return_20d,
        "market_return_60d": market_return_60d,
        "market_volatility_20d": float(context.get("market_volatility_20d", 0.0)),
        "market_breadth_20d": float(context.get("market_breadth_20d", 0.5)),
        "market_breadth_60d": float(context.get("market_breadth_60d", 0.5)),
        "rel_strength_20d": perf_20d - market_return_20d,
        "rel_strength_60d": _series_pct_change(close_history, 60) - market_return_60d,
    }


def calculate_savgol_features(
    ticker: str,
    data: pd.DataFrame,
    current_date: datetime,
    lookback_days: int = SAVGOL_TREND_LOOKBACK_DAYS,
    market_context: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """Calculate local polynomial trend and medium-horizon context features."""
    del ticker  # Still pooled: no ticker ID is passed as a model feature.

    try:
        current_date = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date]
        if len(hist) < lookback_days:
            return None

        close_history = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(close_history) < lookback_days:
            return None

        aligned_hist = hist.reindex(close_history.index)
        volume_history = (
            pd.to_numeric(aligned_hist["Volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
            if "Volume" in aligned_hist.columns
            else np.ones(len(close_history), dtype=float)
        )
        high_history = (
            pd.to_numeric(aligned_hist["High"], errors="coerce").ffill().fillna(close_history).to_numpy(dtype=float, copy=True)
            if "High" in aligned_hist.columns
            else close_history.to_numpy(dtype=float, copy=True)
        )
        low_history = (
            pd.to_numeric(aligned_hist["Low"], errors="coerce").ffill().fillna(close_history).to_numpy(dtype=float, copy=True)
            if "Low" in aligned_hist.columns
            else close_history.to_numpy(dtype=float, copy=True)
        )

        return _calculate_savgol_features_from_history(
            close_history=close_history,
            high_history=high_history,
            low_history=low_history,
            volume_history=volume_history,
            lookback_days=lookback_days,
            market_context=market_context,
        )
    except Exception:
        return None


def _calculate_forward_return_from_close(
    close_values: np.ndarray,
    current_idx: int,
    forward_days: int = SAVGOL_TREND_FORWARD_DAYS,
) -> Optional[float]:
    """Calculate forward return directly from aligned close arrays."""
    min_future_rows = max(2, forward_days // 2)
    if current_idx < 0 or current_idx >= len(close_values):
        return None
    if (len(close_values) - current_idx) < min_future_rows:
        return None

    current_price = float(close_values[current_idx])
    future_idx = min(current_idx + forward_days, len(close_values) - 1)
    future_price = float(close_values[future_idx])
    if current_price > 0 and future_price > 0:
        return (future_price / current_price - 1.0) * 100.0
    return None


def calculate_forward_return(
    data: pd.DataFrame,
    current_date: datetime,
    forward_days: int = SAVGOL_TREND_FORWARD_DAYS,
) -> Optional[float]:
    """Calculate the realized forward return used as the training label."""
    try:
        current_date = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date]
        if len(hist) == 0:
            return None

        current_price = hist["Close"].iloc[-1]
        future_date = current_date + timedelta(days=forward_days + 7)
        future_data = data.loc[current_date:future_date]
        if len(future_data) < max(2, forward_days // 2):
            return None

        future_idx = min(forward_days, len(future_data) - 1)
        future_price = future_data["Close"].iloc[future_idx]
        if current_price > 0 and future_price > 0:
            return (future_price / current_price - 1.0) * 100.0
        return None
    except Exception:
        return None


def collect_savgol_ticker_training_data(
    ticker: str,
    ticker_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    lookback_days: int,
    forward_days: int,
    market_context_map: Dict[pd.Timestamp, Dict[str, float]],
) -> List[Tuple[pd.Timestamp, Dict[str, float], float]]:
    """Collect training samples for a single ticker."""
    if ticker_data is None or len(ticker_data) == 0:
        return []

    try:
        close_series = pd.to_numeric(ticker_data["Close"], errors="coerce").dropna()
    except Exception:
        return []
    if len(close_series) == 0:
        return []

    aligned_frame = ticker_data.reindex(close_series.index)
    close_values = close_series.to_numpy(dtype=float, copy=True)
    volume_values = (
        pd.to_numeric(aligned_frame["Volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=True)
        if "Volume" in aligned_frame.columns
        else np.ones(len(close_series), dtype=float)
    )
    high_values = (
        pd.to_numeric(aligned_frame["High"], errors="coerce").ffill().fillna(close_series).to_numpy(dtype=float, copy=True)
        if "High" in aligned_frame.columns
        else close_values.copy()
    )
    low_values = (
        pd.to_numeric(aligned_frame["Low"], errors="coerce").ffill().fillna(close_series).to_numpy(dtype=float, copy=True)
        if "Low" in aligned_frame.columns
        else close_values.copy()
    )

    date_index = pd.DatetimeIndex(close_series.index)
    if date_index.tz is not None:
        normalized_index = date_index.tz_convert("UTC").tz_localize(None)
    else:
        normalized_index = date_index
    date_ns = normalized_index.asi8
    start_ns = _timestamp_ns(train_start_date)
    end_ns = _timestamp_ns(train_end_date)
    min_future_rows = max(2, forward_days // 2)
    start_pos = max(lookback_days - 1, int(np.searchsorted(date_ns, start_ns, side="left")))
    end_pos = min(
        int(np.searchsorted(date_ns, end_ns, side="right")) - 1,
        len(close_series) - min_future_rows,
    )
    if start_pos > end_pos:
        return []

    history_limit = max(lookback_days, 260)
    samples: List[Tuple[pd.Timestamp, Dict[str, float], float]] = []
    for pos in range(start_pos, end_pos + 1):
        try:
            sample_key = pd.Timestamp(close_series.index[pos])
            cache_key = _market_context_key(sample_key)
            market_context = market_context_map.get(cache_key, {})
            history_start = max(0, pos - history_limit + 1)
            close_history = close_series.iloc[history_start : pos + 1]
            features = _calculate_savgol_features_from_history(
                close_history=close_history,
                high_history=high_values[history_start : pos + 1],
                low_history=low_values[history_start : pos + 1],
                volume_history=volume_values[history_start : pos + 1],
                lookback_days=lookback_days,
                market_context=market_context,
            )
            if features is None:
                continue

            forward_ret = _calculate_forward_return_from_close(
                close_values,
                pos,
                forward_days=forward_days,
            )
            if forward_ret is None:
                continue

            target = _prepare_label(float(forward_ret))
            samples.append((sample_key, features, target))
        except Exception:
            continue

    return samples


def _collect_savgol_data_worker(args):
    """Top-level worker for multiprocessing Pool - collects SavGol training data for one ticker."""
    (
        ticker,
        ticker_data,
        train_start_date,
        train_end_date,
        lookback_days,
        forward_days,
        market_context_map,
    ) = args
    samples = collect_savgol_ticker_training_data(
        ticker=ticker,
        ticker_data=ticker_data,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        lookback_days=lookback_days,
        forward_days=forward_days,
        market_context_map=market_context_map,
    )
    return ticker, samples


def _fresh_model(name: str, device: str):
    """Create a new model instance for the requested backend."""
    import xgboost as xgb
    import lightgbm as lgb

    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            tree_method="hist",
            device=device,
            verbosity=0,
            n_jobs=-1,
        )
    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    if name == "CatBoost":
        import catboost as cb

        return cb.CatBoostRegressor(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="CPU",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            thread_count=1,
        )
    raise ValueError(f"Unknown model backend: {name}")


def _build_model_set():
    """Build the default ensemble used for training and incremental continuation."""
    device = "cuda" if XGBOOST_USE_GPU else "cpu"
    models = {
        "XGBoost": _fresh_model("XGBoost", device),
        "LightGBM": _fresh_model("LightGBM", device),
    }
    try:
        models["CatBoost"] = _fresh_model("CatBoost", device)
    except Exception:
        pass
    return models, device


class SavgolTrendStrategy:
    """Pooled ML strategy driven by Savitzky-Golay and market-context features."""

    def __init__(
        self,
        retrain_days: int = SAVGOL_TREND_RETRAIN_DAYS,
        min_samples: int = SAVGOL_TREND_MIN_SAMPLES,
        lookback_days: int = SAVGOL_TREND_LOOKBACK_DAYS,
        forward_days: int = SAVGOL_TREND_FORWARD_DAYS,
    ):
        self.retrain_days = retrain_days
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        self.forward_days = forward_days
        self.model = None
        self.all_models: Optional[Dict[str, object]] = None
        self.all_scores: Optional[Dict[str, float]] = None
        self.best_name: Optional[str] = None
        self.feature_cols: List[str] = []
        self.day_count = 0
        self.last_train_day = 0

    def should_retrain(self) -> bool:
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days

    def increment_day(self):
        self.day_count += 1

    def train_model(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        business_days: List[datetime],
        current_day_idx: int,
    ) -> bool:
        """Train the pooled model on historical ticker samples only."""
        if not business_days:
            return False

        try:
            if current_day_idx >= len(business_days):
                current_day_idx = len(business_days) - 1
            current_date = business_days[current_day_idx] if current_day_idx >= 0 else business_days[0]
            train_start = current_date - timedelta(days=self.lookback_days)

            import time as _time

            all_tickers = list(ticker_data_grouped.keys())
            n_workers = max(1, min(SAVGOL_TREND_MAX_WORKERS, len(all_tickers)))

            cache_start = train_start
            cache_end = current_date
            market_context_dates = [day for day in business_days if cache_start <= day <= cache_end]
            market_context_map = _precompute_market_context_map(
                ticker_data_grouped=ticker_data_grouped,
                train_start_date=cache_start,
                train_end_date=cache_end,
                forward_days=self.forward_days,
                sample_dates=market_context_dates,
            )

            print(f"   📊 SavGol Trend: Collecting data from {len(all_tickers)} tickers ({n_workers} workers, {self.lookback_days}d lookback)...")
            _start_time = _time.time()

            collect_args = [
                (
                    t,
                    ticker_data_grouped.get(t),
                    train_start,
                    current_date,
                    self.lookback_days,
                    self.forward_days,
                    market_context_map,
                )
                for t in all_tickers
            ]

            samples: List[Tuple[pd.Timestamp, Dict[str, float], float]] = []
            ticker_samples_map = {}

            results = []
            # ThreadPoolExecutor avoids the DataFrame pickling overhead of process workers.
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="savgol-collect") as executor:
                futures = [
                    executor.submit(_collect_savgol_data_worker, args)
                    for args in collect_args
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="   SavGol Trend collection",
                    ncols=100,
                    unit="ticker",
                ):
                    results.append(future.result())

            for ticker, ticker_samples in results:
                if ticker_samples:
                    samples.extend(ticker_samples)
                    ticker_samples_map[ticker] = ticker_samples

            _elapsed = _time.time() - _start_time
            print(f"   📊 SavGol Trend: Collected {len(samples)} samples from {len(ticker_samples_map)} tickers ({_elapsed:.1f}s)")

            if len(samples) < self.min_samples:
                print(f"   ⚠️ SavGol Trend: Not enough samples ({len(samples)} < {self.min_samples})")
                return False

            samples.sort(key=lambda row: row[0])
            split_idx = max(int(len(samples) * 0.8), 1)
            split_idx = min(split_idx, len(samples) - 1)
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]

            self.feature_cols = list(train_samples[0][1].keys())

            X_train = (
                pd.DataFrame([row[1] for row in train_samples], columns=self.feature_cols)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            y_train = np.array([row[2] for row in train_samples], dtype=float)
            X_val = (
                pd.DataFrame([row[1] for row in val_samples], columns=self.feature_cols)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            y_val = np.array([row[2] for row in val_samples], dtype=float)

            import time
            import warnings

            models, device = _build_model_set()
            has_existing = self.all_models is not None and len(self.all_models) > 0
            if has_existing:
                models = dict(self.all_models)
                for name in ("XGBoost", "LightGBM", "CatBoost"):
                    if name not in models:
                        try:
                            models[name] = _fresh_model(name, device)
                        except Exception:
                            continue

            print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")

            trained_models: Dict[str, object] = {}
            model_scores: Dict[str, float] = {}

            for name, model in models.items():
                print(f"      🔄 {name}: Training started...", end=" ", flush=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        incremental_failed = False
                        train_start = time.perf_counter()

                        if has_existing:
                            try:
                                if name == "XGBoost":
                                    model.fit(X_train, y_train, xgb_model=model.get_booster())
                                elif name == "LightGBM":
                                    model.fit(X_train, y_train, init_model=model.booster_)
                                elif name == "CatBoost":
                                    if catboost_has_trained_trees(model):
                                        configure_catboost_cpu_continuation(model)
                                        model.fit(X_train, y_train, init_model=model)
                                    else:
                                        incremental_failed = True
                                else:
                                    model.fit(X_train, y_train)

                                quick_pred = np.asarray(model.predict(X_val.iloc[:min(len(X_val), 100)]), dtype=float)
                                if (
                                    np.any(np.isnan(quick_pred))
                                    or np.any(np.isinf(quick_pred))
                                    or (len(quick_pred) > 0 and np.max(np.abs(quick_pred)) > 1e10)
                                ):
                                    incremental_failed = True
                            except Exception:
                                incremental_failed = True

                        if not has_existing or incremental_failed:
                            model = _fresh_model(name, device)
                            model.fit(X_train, y_train)

                    train_elapsed = time.perf_counter() - train_start

                    val_pred = np.asarray(model.predict(X_val), dtype=float)
                    if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
                        continue

                    score = spearmanr(y_val, val_pred).correlation if len(y_val) > 1 else 0.0
                    if score is None or np.isnan(score):
                        score = 0.0

                    trained_models[name] = model
                    model_scores[name] = float(score)
                    mode = "incremental" if has_existing and not incremental_failed else "fresh"
                    print(f"spearman = {score:.3f} ({mode}, {train_elapsed:.1f}s)")
                except Exception as exc:
                    print(f"failed: {exc}")

            if not trained_models:
                print("   ⚠️ SavGol Trend: No models trained successfully")
                return False

            self.best_name = max(model_scores, key=model_scores.get)
            self.model = trained_models[self.best_name]
            self.all_models = trained_models
            self.all_scores = model_scores
            self.last_train_day = self.day_count
            save_succeeded = self.save_model(
                current_date=current_date,
                train_start=train_samples[0][0],
                train_end=train_samples[-1][0],
            )
            if save_succeeded:
                print(
                    f"   ✅ SavGol Trend: Saved {len(trained_models)} models. Best = {self.best_name} "
                    f"(spearman {model_scores[self.best_name]:.3f})"
                )
            else:
                print(
                    f"   ⚠️ SavGol Trend: Trained {len(trained_models)} models, but save failed. "
                    f"Best = {self.best_name} (spearman {model_scores[self.best_name]:.3f})"
                )
            return True
        finally:
            release_runtime_memory()

    def save_model(
        self,
        current_date: Optional[datetime] = None,
        train_start: Optional[pd.Timestamp] = None,
        train_end: Optional[pd.Timestamp] = None,
    ) -> bool:
        """Persist model and training state for later reuse."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            metadata = {}
            if current_date is not None:
                metadata["updated"] = pd.Timestamp(current_date).isoformat()
            if train_start is not None:
                metadata["train_start"] = pd.Timestamp(train_start).isoformat()
            if train_end is not None:
                metadata["train_end"] = pd.Timestamp(train_end).isoformat()
            if self.best_name is not None:
                metadata["best_model"] = self.best_name
            if self.all_scores:
                metadata["all_scores"] = self.all_scores
            payload = ensure_catboost_cpu_metadata(
                {
                    "all_models": self.all_models,
                    "all_scores": self.all_scores,
                    "model": self.model,
                    "best_name": self.best_name,
                    "feature_cols": self.feature_cols,
                    "day_count": self.day_count,
                    "last_train_day": self.last_train_day,
                    "retrain_days": self.retrain_days,
                    "min_samples": self.min_samples,
                    "lookback_days": self.lookback_days,
                    "forward_days": self.forward_days,
                    "metadata": metadata,
                }
            )
            joblib.dump(payload, SAVGOL_TREND_MODEL_PATH)
            save_native_model_artifacts(payload, SAVGOL_TREND_MODEL_PATH)
            print(f"   💾 SavGol Trend: Saved model to {SAVGOL_TREND_MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"   ⚠️ SavGol Trend: Failed to save model: {exc}")
            return False

    def load_model(self) -> bool:
        """Load a previously trained model if available."""
        try:
            if not SAVGOL_TREND_MODEL_PATH.exists():
                return False

            saved = joblib.load(SAVGOL_TREND_MODEL_PATH)
            saved = restore_native_model_artifacts(saved, SAVGOL_TREND_MODEL_PATH)
            if isinstance(saved, dict):
                saved, reset_catboost = reset_legacy_catboost_member(saved)
                if reset_catboost:
                    print("   ♻️ SavGol Trend: Resetting saved CatBoost member for a clean CPU incremental restart")
            if isinstance(saved, dict) and "all_models" in saved:
                self.all_models = saved.get("all_models")
                self.all_scores = saved.get("all_scores")
                self.model = saved.get("model")
                self.best_name = saved.get("best_name")
                self.feature_cols = saved.get("feature_cols", [])
                self.day_count = saved.get("day_count", 0)
                self.last_train_day = saved.get("last_train_day", 0)
            else:
                self.model = saved.get("model") if isinstance(saved, dict) else saved
                self.best_name = saved.get("best_name") if isinstance(saved, dict) else None
                self.feature_cols = saved.get("feature_cols", []) if isinstance(saved, dict) else []
                self.day_count = saved.get("day_count", 0) if isinstance(saved, dict) else 0
                self.last_train_day = saved.get("last_train_day", 0) if isinstance(saved, dict) else 0
            print(
                f"   📂 SavGol Trend: Loaded {self.best_name or 'model'} "
                f"(day_count={self.day_count})"
            )
            return self.model is not None
        except Exception as exc:
            print(f"   ⚠️ SavGol Trend: Failed to load model: {exc}")
            return False

    def release_model_artifacts(self):
        """Drop heavy fitted models while preserving lightweight state."""
        self.model = None
        self.all_models = None

    def predict_returns(
        self,
        tickers: List[str],
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict scores for the current candidate universe with optimized parallel processing."""
        if self.model is None or not self.feature_cols:
            return []

        market_context = _calculate_market_context(ticker_data_grouped, current_date)
        
        # Process tickers in parallel batches for much faster performance
        def process_single_ticker(ticker):
            data = ticker_data_grouped.get(ticker)
            if data is None:
                return None

            features = calculate_savgol_features(
                ticker,
                data,
                current_date,
                lookback_days=self.lookback_days,
                market_context=market_context,
            )
            if features is None:
                return None

            return ticker, features

        # Use ThreadPoolExecutor for parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        # Limit threads to avoid overwhelming the system
        max_workers = min(32, (os.cpu_count() or 1) + 4, len(tickers))
        
        valid_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {executor.submit(process_single_ticker, ticker): ticker for ticker in tickers}
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                result = future.result()
                if result is not None:
                    valid_results.append(result)
        
        # Batch predict all valid results at once for massive speedup
        if not valid_results:
            return []
            
        # Create batch DataFrame for all valid tickers
        batch_data = []
        ticker_map = []
        for ticker, features in valid_results:
            row = [features.get(col, 0.0) for col in self.feature_cols]
            batch_data.append(row)
            ticker_map.append((ticker, features))
        
        if not batch_data:
            return []
            
        batch_df = pd.DataFrame(batch_data, columns=self.feature_cols)
        
        # Batch prediction - much faster than individual predictions
        try:
            batch_predictions = self.model.predict(batch_df)
            predictions = [
                (ticker_map[i][0], float(batch_predictions[i]), ticker_map[i][1])
                for i in range(len(batch_predictions))
            ]
        except Exception:
            return []

        return predictions

def select_savgol_trend_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    model: SavgolTrendStrategy,
    current_holdings: Optional[List[str]] = None,
) -> List[str]:
    """Score the current universe with an already-trained SavGol model."""
    predictions = model.predict_returns(all_tickers, ticker_data_grouped, current_date)
    if not predictions:
        return []

    current_holdings = current_holdings or []
    predictions.sort(key=lambda item: item[1], reverse=True)

    best_model_score = float(model.all_scores.get(model.best_name, 0.0)) if model.all_scores and model.best_name else 0.0
    top_score = float(predictions[0][1])
    cutoff_score = float(predictions[min(top_n - 1, len(predictions) - 1)][1])
    score_spread = top_score - cutoff_score

    adjusted_predictions = []
    for ticker, pred, features in predictions:
        adjusted_pred = pred + (SAVGOL_TREND_HOLD_MARGIN if ticker in current_holdings else 0.0)
        adjusted_predictions.append((ticker, adjusted_pred, pred, features))
    adjusted_predictions.sort(key=lambda item: item[1], reverse=True)

    low_confidence = (
        best_model_score < SAVGOL_TREND_MIN_MODEL_SPEARMAN
        or top_score < SAVGOL_TREND_MIN_PREDICTED_EDGE
        or score_spread < SAVGOL_TREND_MIN_SCORE_SPREAD
    )

    if low_confidence:
        print(
            "   ⚠️ SavGol Trend: Low confidence "
            f"(spearman={best_model_score:.3f}, top={top_score:+.3f}, spread={score_spread:.3f})"
        )
        if current_holdings:
            return []
        return []

    selected = [ticker for ticker, _, _, _ in adjusted_predictions[:top_n]]
    print(
        f"   ✅ SavGol Trend: Selected {len(selected)} tickers "
        f"(spearman={best_model_score:.3f}, top={top_score:+.3f}, spread={score_spread:.3f})"
    )
    return selected


def select_savgol_trend_stocks_with_training(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    model: SavgolTrendStrategy,
    business_days: List[datetime],
    current_day_idx: int,
    current_holdings: Optional[List[str]] = None,
    force_train: bool = False,
) -> List[str]:
    """AI-Elite-style SavGol wrapper backed by the on-disk checkpoint."""
    walk_forward_retraining_enabled = bool(getattr(config, "ENABLE_WALK_FORWARD_RETRAINING", True))
    if not walk_forward_retraining_enabled and not force_train:
        if model.model is None:
            if model.load_model() and model.model is not None:
                print("   ⏭️ SavGol Trend: Walk-forward retraining disabled, using loaded model")
            else:
                print("   ⚠️ SavGol Trend: Walk-forward retraining disabled and no saved model loaded")
                return []
    else:
        if model.model is None:
            model.load_model()
        if force_train or model.should_retrain():
            model.train_model(ticker_data_grouped, business_days, current_day_idx)

    return select_savgol_trend_stocks(
        all_tickers,
        ticker_data_grouped,
        current_date,
        top_n,
        model,
        current_holdings=current_holdings,
    )
