"""
AI rebalance model for Voting Ensemble-style strategies.

Ticker selection remains separate: this module only decides whether replacing
an existing holding with the best available candidate is worth it.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_elite_strategy_per_ticker import (
    _catboost_has_trained_trees,
    _configure_existing_model_backend,
    _configure_catboost_continuation,
    _format_training_plan,
    _fresh_ensemble_model,
    _model_backend,
    _model_worker_mode,
    _order_models_for_training,
    _predictions_are_unstable,
    _set_model_parallelism,
)
from model_training_safety import (
    release_runtime_memory,
    reset_legacy_catboost_member,
    restore_native_model_artifacts,
    save_native_model_artifacts,
)
from strategy_disk_cache import get_cache_dir


MODEL_SAVE_DIR = Path("logs/models")
AI_REBALANCE_MODEL_PATH = MODEL_SAVE_DIR / "ai_rebalance.joblib"
VOTING_AI_REBALANCE_MODEL_PATH = AI_REBALANCE_MODEL_PATH
_REBALANCE_COLLECTION_CONTEXT: Dict[str, object] = {}
_REBALANCE_TEMP_DIRS: set[str] = set()
_AI_REBALANCE_CACHE_CONTEXT_PATH: Optional[str] = None
_AI_REBALANCE_CACHE_CONTEXT: Optional[Dict[str, object]] = None
_AI_REBALANCE_TRAIN_CONTEXT_PATH: Optional[str] = None
_AI_REBALANCE_TRAIN_CONTEXT: Optional[Dict[str, object]] = None
AI_REBALANCE_CACHE_NUMERIC_DTYPE = "float64"
TICKER_FEATURE_NAMES = [
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_10d",
    "drawdown_20d",
    "sma20_dist",
]
MARKET_FEATURE_NAMES = [
    "market_return_5d",
    "market_return_20d",
    "market_vol_10d",
    "market_above_sma20",
    "market_is_up",
]

FEATURE_COLS = [
    "held_rank",
    "candidate_rank",
    "rank_gap",
    "held_in_target",
    "held_in_buffer",
    "candidate_in_target",
    "candidate_in_buffer",
    "current_holding_count",
    "transaction_cost",
    "held_ret_3d",
    "held_ret_5d",
    "held_ret_10d",
    "held_ret_20d",
    "held_vol_10d",
    "held_drawdown_20d",
    "held_sma20_dist",
    "candidate_ret_3d",
    "candidate_ret_5d",
    "candidate_ret_10d",
    "candidate_ret_20d",
    "candidate_vol_10d",
    "candidate_drawdown_20d",
    "candidate_sma20_dist",
    "diff_ret_3d",
    "diff_ret_5d",
    "diff_ret_10d",
    "diff_ret_20d",
    "diff_vol_10d",
    "diff_drawdown_20d",
    "diff_sma20_dist",
    "market_return_5d",
    "market_return_20d",
    "market_vol_10d",
    "market_above_sma20",
    "market_is_up",
]


def _default_ticker_features() -> Dict[str, float]:
    return {name: 0.0 for name in TICKER_FEATURE_NAMES}


def _default_market_features() -> Dict[str, float]:
    return {name: 0.0 for name in MARKET_FEATURE_NAMES}


def _ticker_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in TICKER_FEATURE_NAMES], dtype=np.float64)


def _market_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in MARKET_FEATURE_NAMES], dtype=np.float64)


def _rebalance_cache_key(current_date: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(current_date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _series_up_to(ticker_df: pd.DataFrame, current_date: datetime) -> pd.Series:
    close = ticker_df.get("Close")
    if close is None:
        return pd.Series(dtype=float)
    return close.loc[:current_date].dropna()


def _safe_return(close: pd.Series, lookback: int) -> float:
    if len(close) <= lookback:
        return 0.0
    start = float(close.iloc[-(lookback + 1)])
    end = float(close.iloc[-1])
    if start <= 0:
        return 0.0
    return (end / start) - 1.0


def _safe_vol(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback + 1:
        return 0.0
    returns = close.tail(lookback + 1).pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(252))


def _safe_drawdown(close: pd.Series, lookback: int) -> float:
    if len(close) < 2:
        return 0.0
    window = np.asarray(close.tail(min(len(close), lookback)), dtype=float)
    if len(window) == 0:
        return 0.0
    peaks = np.maximum.accumulate(window)
    drawdowns = np.where(peaks > 0, (peaks - window) / peaks, 0.0)
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def _safe_sma_distance(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback:
        return 0.0
    sma = float(close.tail(lookback).mean())
    last = float(close.iloc[-1])
    if sma == 0:
        return 0.0
    return (last / sma) - 1.0


def _ticker_features(
    ticker: str,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    ticker_df = ticker_data_grouped.get(ticker)
    return _ticker_features_from_df(ticker_df, current_date)


def _ticker_features_from_df(
    ticker_df: Optional[pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    if ticker_df is None or len(ticker_df) == 0:
        return _default_ticker_features()

    close = _series_up_to(ticker_df, current_date)
    if len(close) == 0:
        return _default_ticker_features()

    return {
        "ret_3d": _safe_return(close, 3),
        "ret_5d": _safe_return(close, 5),
        "ret_10d": _safe_return(close, 10),
        "ret_20d": _safe_return(close, 20),
        "vol_10d": _safe_vol(close, 10),
        "drawdown_20d": _safe_drawdown(close, 20),
        "sma20_dist": _safe_sma_distance(close, 20),
    }


def _market_features(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    market_df = ticker_data_grouped.get("SPY")
    if market_df is None or len(market_df) == 0:
        market_df = ticker_data_grouped.get("QQQ")

    return _market_features_from_df(market_df, current_date)


def _market_features_from_df(
    market_df: Optional[pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    if market_df is None or len(market_df) == 0:
        return _default_market_features()

    close = _series_up_to(market_df, current_date)
    if len(close) == 0:
        return _default_market_features()

    market_return_5d = _safe_return(close, 5)
    market_return_20d = _safe_return(close, 20)
    market_above_sma20 = 1.0 if _safe_sma_distance(close, 20) > 0 else 0.0

    return {
        "market_return_5d": market_return_5d,
        "market_return_20d": market_return_20d,
        "market_vol_10d": _safe_vol(close, 10),
        "market_above_sma20": market_above_sma20,
        "market_is_up": 1.0 if market_return_5d > 0 else 0.0,
    }


def _calculate_forward_return(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    ticker: str,
    current_date: datetime,
    forward_days: int,
    latest_allowed_date: datetime,
) -> Optional[float]:
    ticker_df = ticker_data_grouped.get(ticker)
    return _calculate_forward_return_from_df(
        ticker_df=ticker_df,
        current_date=current_date,
        forward_days=forward_days,
        latest_allowed_date=latest_allowed_date,
    )


def _calculate_forward_return_from_df(
    ticker_df: Optional[pd.DataFrame],
    current_date: datetime,
    forward_days: int,
    latest_allowed_date: datetime,
) -> Optional[float]:
    if ticker_df is None or len(ticker_df) == 0:
        return None

    close = ticker_df.get("Close")
    if close is None:
        return None

    history = close.loc[:current_date].dropna()
    future = close.loc[(close.index > current_date) & (close.index <= latest_allowed_date)].dropna()
    if len(history) == 0 or len(future) < forward_days:
        return None

    start_price = float(history.iloc[-1])
    end_price = float(future.iloc[forward_days - 1])
    if start_price <= 0:
        return None
    return (end_price / start_price) - 1.0


def _build_rebalance_feature_row_from_features(
    held_ticker: str,
    candidate_ticker: str,
    ranked_candidates: List[str],
    current_holdings: List[str],
    held_features: Dict[str, float],
    candidate_features: Dict[str, float],
    market_features: Dict[str, float],
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
) -> Dict[str, float]:
    rank_lookup = {ticker: idx + 1 for idx, ticker in enumerate(ranked_candidates)}
    fallback_rank = len(ranked_candidates) + 1
    held_rank = float(rank_lookup.get(held_ticker, fallback_rank))
    candidate_rank = float(rank_lookup.get(candidate_ticker, fallback_rank))

    row = {
        "held_rank": held_rank,
        "candidate_rank": candidate_rank,
        "rank_gap": candidate_rank - held_rank,
        "held_in_target": 1.0 if held_ticker in ranked_candidates[:portfolio_size] else 0.0,
        "held_in_buffer": 1.0 if held_ticker in ranked_candidates[:buffer_size] else 0.0,
        "candidate_in_target": 1.0 if candidate_ticker in ranked_candidates[:portfolio_size] else 0.0,
        "candidate_in_buffer": 1.0 if candidate_ticker in ranked_candidates[:buffer_size] else 0.0,
        "current_holding_count": float(len(current_holdings)),
        "transaction_cost": float(transaction_cost),
        "held_ret_3d": held_features["ret_3d"],
        "held_ret_5d": held_features["ret_5d"],
        "held_ret_10d": held_features["ret_10d"],
        "held_ret_20d": held_features["ret_20d"],
        "held_vol_10d": held_features["vol_10d"],
        "held_drawdown_20d": held_features["drawdown_20d"],
        "held_sma20_dist": held_features["sma20_dist"],
        "candidate_ret_3d": candidate_features["ret_3d"],
        "candidate_ret_5d": candidate_features["ret_5d"],
        "candidate_ret_10d": candidate_features["ret_10d"],
        "candidate_ret_20d": candidate_features["ret_20d"],
        "candidate_vol_10d": candidate_features["vol_10d"],
        "candidate_drawdown_20d": candidate_features["drawdown_20d"],
        "candidate_sma20_dist": candidate_features["sma20_dist"],
        "diff_ret_3d": candidate_features["ret_3d"] - held_features["ret_3d"],
        "diff_ret_5d": candidate_features["ret_5d"] - held_features["ret_5d"],
        "diff_ret_10d": candidate_features["ret_10d"] - held_features["ret_10d"],
        "diff_ret_20d": candidate_features["ret_20d"] - held_features["ret_20d"],
        "diff_vol_10d": candidate_features["vol_10d"] - held_features["vol_10d"],
        "diff_drawdown_20d": candidate_features["drawdown_20d"] - held_features["drawdown_20d"],
        "diff_sma20_dist": candidate_features["sma20_dist"] - held_features["sma20_dist"],
    }
    row.update(market_features)
    return row


def _build_rebalance_feature_row_from_vectors(
    held_ticker: str,
    candidate_ticker: str,
    ranked_candidates: List[str],
    current_holdings: List[str],
    held_vector: np.ndarray,
    candidate_vector: np.ndarray,
    market_vector: np.ndarray,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
) -> Dict[str, float]:
    rank_lookup = {ticker: idx + 1 for idx, ticker in enumerate(ranked_candidates)}
    fallback_rank = len(ranked_candidates) + 1
    held_rank = float(rank_lookup.get(held_ticker, fallback_rank))
    candidate_rank = float(rank_lookup.get(candidate_ticker, fallback_rank))

    held_ret_3d, held_ret_5d, held_ret_10d, held_ret_20d, held_vol_10d, held_drawdown_20d, held_sma20_dist = map(float, held_vector)
    candidate_ret_3d, candidate_ret_5d, candidate_ret_10d, candidate_ret_20d, candidate_vol_10d, candidate_drawdown_20d, candidate_sma20_dist = map(float, candidate_vector)
    market_return_5d, market_return_20d, market_vol_10d, market_above_sma20, market_is_up = map(float, market_vector)

    return {
        "held_rank": held_rank,
        "candidate_rank": candidate_rank,
        "rank_gap": candidate_rank - held_rank,
        "held_in_target": 1.0 if held_ticker in ranked_candidates[:portfolio_size] else 0.0,
        "held_in_buffer": 1.0 if held_ticker in ranked_candidates[:buffer_size] else 0.0,
        "candidate_in_target": 1.0 if candidate_ticker in ranked_candidates[:portfolio_size] else 0.0,
        "candidate_in_buffer": 1.0 if candidate_ticker in ranked_candidates[:buffer_size] else 0.0,
        "current_holding_count": float(len(current_holdings)),
        "transaction_cost": float(transaction_cost),
        "held_ret_3d": held_ret_3d,
        "held_ret_5d": held_ret_5d,
        "held_ret_10d": held_ret_10d,
        "held_ret_20d": held_ret_20d,
        "held_vol_10d": held_vol_10d,
        "held_drawdown_20d": held_drawdown_20d,
        "held_sma20_dist": held_sma20_dist,
        "candidate_ret_3d": candidate_ret_3d,
        "candidate_ret_5d": candidate_ret_5d,
        "candidate_ret_10d": candidate_ret_10d,
        "candidate_ret_20d": candidate_ret_20d,
        "candidate_vol_10d": candidate_vol_10d,
        "candidate_drawdown_20d": candidate_drawdown_20d,
        "candidate_sma20_dist": candidate_sma20_dist,
        "diff_ret_3d": candidate_ret_3d - held_ret_3d,
        "diff_ret_5d": candidate_ret_5d - held_ret_5d,
        "diff_ret_10d": candidate_ret_10d - held_ret_10d,
        "diff_ret_20d": candidate_ret_20d - held_ret_20d,
        "diff_vol_10d": candidate_vol_10d - held_vol_10d,
        "diff_drawdown_20d": candidate_drawdown_20d - held_drawdown_20d,
        "diff_sma20_dist": candidate_sma20_dist - held_sma20_dist,
        "market_return_5d": market_return_5d,
        "market_return_20d": market_return_20d,
        "market_vol_10d": market_vol_10d,
        "market_above_sma20": market_above_sma20,
        "market_is_up": market_is_up,
    }


def build_rebalance_feature_row(
    held_ticker: str,
    candidate_ticker: str,
    ranked_candidates: List[str],
    current_holdings: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
) -> Dict[str, float]:
    held_features = _ticker_features(held_ticker, ticker_data_grouped, current_date)
    candidate_features = _ticker_features(candidate_ticker, ticker_data_grouped, current_date)
    market_features = _market_features(ticker_data_grouped, current_date)
    return _build_rebalance_feature_row_from_features(
        held_ticker=held_ticker,
        candidate_ticker=candidate_ticker,
        ranked_candidates=ranked_candidates,
        current_holdings=current_holdings,
        held_features=held_features,
        candidate_features=candidate_features,
        market_features=market_features,
        transaction_cost=transaction_cost,
        portfolio_size=portfolio_size,
        buffer_size=buffer_size,
    )


def collect_rebalance_training_samples_from_state(
    current_holdings: List[str],
    ranked_candidates: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    sample_date: datetime,
    latest_allowed_date: datetime,
    forward_days: int,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
    cache_context: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """
    Build supervised rebalance samples from a real historical portfolio state.

    This keeps AI Elite untouched: the rebalance model learns from historical
    AI Elite candidate lists and the actual holdings that the AI-rebalance
    overlay was managing on each date.
    """
    if not current_holdings or not ranked_candidates:
        return []

    sample_ts = pd.Timestamp(sample_date)
    latest_allowed_ts = pd.Timestamp(latest_allowed_date)
    if sample_ts.tzinfo is None and latest_allowed_ts.tzinfo is not None:
        sample_ts = sample_ts.tz_localize(latest_allowed_ts.tzinfo)
    elif sample_ts.tzinfo is not None and latest_allowed_ts.tzinfo is None:
        latest_allowed_ts = latest_allowed_ts.tz_localize(sample_ts.tzinfo)
    if latest_allowed_ts <= sample_ts:
        return []

    valid_holdings = [
        ticker
        for ticker in current_holdings
        if ticker_data_grouped.get(ticker) is not None and len(ticker_data_grouped.get(ticker)) > 0
    ]
    if not valid_holdings:
        return []

    held_set = set(valid_holdings)
    replacement_candidates = [
        ticker
        for ticker in ranked_candidates
        if ticker not in held_set
        and ticker_data_grouped.get(ticker) is not None
        and len(ticker_data_grouped.get(ticker)) > 0
    ]
    if not replacement_candidates:
        return []

    ticker_to_idx = {}
    feature_array = None
    forward_array = None
    market_array = None
    start_ordinal = None
    n_dates = None
    date_idx = None
    if cache_context:
        try:
            ticker_to_idx = {
                ticker: idx for idx, ticker in enumerate(cache_context.get("all_tickers") or [])
            }
            start_ordinal = int(cache_context.get("start_ordinal", 0))
            n_dates = int(cache_context.get("n_dates", 0))
            date_idx = _rebalance_cache_key(sample_date).toordinal() - start_ordinal
            if 0 <= date_idx < n_dates:
                feature_array = np.load(str(cache_context["feature_path"]), mmap_mode="r")
                forward_array = np.load(str(cache_context["forward_path"]), mmap_mode="r")
                market_array = np.load(str(cache_context["market_path"]), mmap_mode="r")
            else:
                date_idx = None
        except Exception:
            ticker_to_idx = {}
            feature_array = None
            forward_array = None
            market_array = None
            date_idx = None

    samples: List[Dict[str, object]] = []
    for held_ticker in valid_holdings:
        held_idx = ticker_to_idx.get(held_ticker)
        if forward_array is not None and held_idx is not None and date_idx is not None:
            keep_return = float(forward_array[held_idx, date_idx])
            if np.isnan(keep_return):
                keep_return = None
        else:
            keep_return = _calculate_forward_return(
                ticker_data_grouped=ticker_data_grouped,
                ticker=held_ticker,
                current_date=sample_date,
                forward_days=forward_days,
                latest_allowed_date=latest_allowed_date,
            )
        if keep_return is None:
            continue

        for candidate_ticker in replacement_candidates:
            candidate_idx = ticker_to_idx.get(candidate_ticker)
            if forward_array is not None and candidate_idx is not None and date_idx is not None:
                replace_return = float(forward_array[candidate_idx, date_idx])
                if np.isnan(replace_return):
                    replace_return = None
            else:
                replace_return = _calculate_forward_return(
                    ticker_data_grouped=ticker_data_grouped,
                    ticker=candidate_ticker,
                    current_date=sample_date,
                    forward_days=forward_days,
                    latest_allowed_date=latest_allowed_date,
                )
            if replace_return is None:
                continue

            if (
                feature_array is not None
                and market_array is not None
                and held_idx is not None
                and candidate_idx is not None
                and date_idx is not None
            ):
                row = _build_rebalance_feature_row_from_vectors(
                    held_ticker=held_ticker,
                    candidate_ticker=candidate_ticker,
                    ranked_candidates=ranked_candidates,
                    current_holdings=valid_holdings,
                    held_vector=np.asarray(feature_array[held_idx, date_idx], dtype=np.float64),
                    candidate_vector=np.asarray(feature_array[candidate_idx, date_idx], dtype=np.float64),
                    market_vector=np.asarray(market_array[date_idx], dtype=np.float64),
                    transaction_cost=transaction_cost,
                    portfolio_size=portfolio_size,
                    buffer_size=buffer_size,
                )
            else:
                row = build_rebalance_feature_row(
                    held_ticker=held_ticker,
                    candidate_ticker=candidate_ticker,
                    ranked_candidates=ranked_candidates,
                    current_holdings=valid_holdings,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=sample_date,
                    transaction_cost=transaction_cost,
                    portfolio_size=portfolio_size,
                    buffer_size=buffer_size,
                )
            switch_advantage = replace_return - keep_return - (2.0 * transaction_cost)
            row.update({
                "label": switch_advantage,
                "held_ticker": held_ticker,
                "candidate_ticker": candidate_ticker,
                "sample_date": sample_date,
            })
            samples.append(row)

    return samples


def _valid_rebalance_tickers(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
) -> List[str]:
    return [
        ticker
        for ticker in all_tickers
        if ticker_data_grouped.get(ticker) is not None and len(ticker_data_grouped.get(ticker)) > 0
    ]


def _rebalance_context_cache_key(
    valid_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
) -> Dict[str, object]:
    return {
        "tickers": list(valid_tickers),
        "train_start_date": pd.Timestamp(train_start_date).isoformat(),
        "train_end_date": pd.Timestamp(train_end_date).isoformat(),
        "forward_days": int(forward_days),
        "numeric_dtype": AI_REBALANCE_CACHE_NUMERIC_DTYPE,
    }


def _cleanup_rebalance_temp_dirs_at_exit() -> None:
    for temp_dir in list(_REBALANCE_TEMP_DIRS):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        finally:
            _REBALANCE_TEMP_DIRS.discard(temp_dir)


atexit.register(_cleanup_rebalance_temp_dirs_at_exit)


def _get_ai_rebalance_train_context() -> Dict[str, object]:
    global _AI_REBALANCE_TRAIN_CONTEXT
    if _AI_REBALANCE_TRAIN_CONTEXT is None:
        if not _AI_REBALANCE_TRAIN_CONTEXT_PATH:
            raise ValueError("AI Rebalance training context is not initialized")
        _AI_REBALANCE_TRAIN_CONTEXT = joblib.load(_AI_REBALANCE_TRAIN_CONTEXT_PATH)
    return _AI_REBALANCE_TRAIN_CONTEXT


def _init_ai_rebalance_train_worker(
    context_path: Optional[str] = None,
    context_data: Optional[Dict[str, object]] = None,
) -> None:
    global _AI_REBALANCE_TRAIN_CONTEXT_PATH, _AI_REBALANCE_TRAIN_CONTEXT
    _AI_REBALANCE_TRAIN_CONTEXT_PATH = context_path
    _AI_REBALANCE_TRAIN_CONTEXT = context_data


def _train_single_ai_rebalance_model(
    name: str,
    n_jobs_limit: int = 4,
) -> Tuple[bool, Optional[object], Optional[float], float, str, str]:
    import time
    import warnings
    from sklearn.metrics import r2_score

    start_time = time.time()
    context = _get_ai_rebalance_train_context()
    X_train = context["X_train"]
    y_train = context["y_train"]
    X_val = context["X_val"]
    y_val = context["y_val"]
    device = context["device"]
    has_existing = bool(context["has_existing"])
    existing_model = context["models"].get(name) if has_existing else None

    if len(X_train) < 10:
        return (False, None, None, 0.0, "skipped", "insufficient training data")
    if np.all(y_train == y_train[0]):
        return (False, None, None, 0.0, "skipped", "constant target")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            incremental_failed = False
            used_incremental = False
            incremental_error = None

            model = existing_model
            if has_existing and model is not None:
                _set_model_parallelism(model, n_jobs_limit)
                try:
                    if name == "XGBoost":
                        used_incremental = True
                        model.fit(X_train, y_train, xgb_model=model.get_booster())
                    elif name == "LightGBM":
                        used_incremental = True
                        model.fit(X_train, y_train, init_model=model.booster_)
                    elif name == "CatBoost":
                        used_incremental = True
                        if _catboost_has_trained_trees(model):
                            _configure_catboost_continuation(
                                model,
                                thread_count=n_jobs_limit if n_jobs_limit > 0 else (os.cpu_count() or 1),
                            )
                            model.fit(X_train, y_train, init_model=model)
                        else:
                            incremental_error = "no saved trees yet"
                            incremental_failed = True
                    elif name in ("SGDRegressor-L2", "SGDRegressor-ElasticNet"):
                        used_incremental = True
                        model.partial_fit(X_train, y_train)
                    if used_incremental:
                        quick_pred = model.predict(X_val[: min(100, len(X_val))])
                        if _predictions_are_unstable(quick_pred):
                            incremental_error = "incremental predictions unstable"
                            incremental_failed = True
                except Exception as exc:
                    incremental_error = str(exc)
                    incremental_failed = True

            if not used_incremental or incremental_failed:
                model = _fresh_ensemble_model(name, device, n_jobs=n_jobs_limit)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            if _predictions_are_unstable(y_pred):
                return (False, None, None, time.time() - start_time, "failed", "unstable predictions")

            score = r2_score(y_val, y_pred)
            if used_incremental and (score < -10 or score > 1 or np.isnan(score) or np.isinf(score)):
                model = _fresh_ensemble_model(name, device, n_jobs=n_jobs_limit)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                if _predictions_are_unstable(y_pred):
                    return (False, None, None, time.time() - start_time, "failed", "unstable after fresh retrain")
                score = r2_score(y_val, y_pred)

            if score < -10 or score > 1 or np.isnan(score) or np.isinf(score):
                return (False, None, None, time.time() - start_time, "failed", f"invalid validation score {score:.3f}")

            elapsed = time.time() - start_time
            status = "incremental" if used_incremental and not incremental_failed else "fresh"
            detail = "ok"
            if incremental_failed and incremental_error:
                detail = f"incremental fallback: {incremental_error}"
            return (True, model, score, elapsed, status, detail)
    except Exception as exc:
        return (False, None, None, time.time() - start_time, "failed", str(exc))


def precompute_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    existing_context: Optional[Dict[str, object]] = None,
    cache_start_date: Optional[datetime] = None,
    cache_end_date: Optional[datetime] = None,
) -> Dict[str, object]:
    """Build or extend file-backed numeric caches so workers mmap shared readonly data."""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    market_df = ticker_data_grouped.get("SPY")
    market_symbol = "SPY"
    if market_df is None or len(market_df) == 0:
        market_df = ticker_data_grouped.get("QQQ")
        market_symbol = "QQQ"

    valid_tickers = _valid_rebalance_tickers(ticker_data_grouped, all_tickers)
    cache_start = cache_start_date if cache_start_date is not None else train_start_date
    cache_end = cache_end_date if cache_end_date is not None else train_end_date
    start_key = _rebalance_cache_key(cache_start)
    end_key = _rebalance_cache_key(cache_end)
    requested_start_ordinal = start_key.toordinal()
    requested_end_ordinal = end_key.toordinal()
    requested_latest_allowed_ordinal = _rebalance_cache_key(train_end_date).toordinal()
    print(
        f"   🗂️ AI Rebalance: Preparing global feature cache "
        f"({len(valid_tickers)} tickers, {cache_start.date()} to {cache_end.date()}, "
        f"market={market_symbol if market_df is not None and len(market_df) > 0 else 'fallback-none'})..."
    )

    cache_context = existing_context if isinstance(existing_context, dict) else None
    cache_is_compatible = False
    if cache_context:
        cached_tickers = list(cache_context.get("all_tickers") or [])
        cached_start_ordinal = int(cache_context.get("start_ordinal", 0))
        cached_n_dates = int(cache_context.get("n_dates", 0))
        cached_end_ordinal = cached_start_ordinal + cached_n_dates - 1
        cached_latest_allowed_ordinal = int(cache_context.get("latest_allowed_ordinal", cached_end_ordinal))
        cache_is_compatible = (
            cached_tickers == valid_tickers
            and int(cache_context.get("forward_days", -1)) == int(forward_days)
            and cached_start_ordinal <= requested_start_ordinal
            and Path(str(cache_context.get("feature_path", ""))).exists()
            and Path(str(cache_context.get("forward_path", ""))).exists()
            and Path(str(cache_context.get("market_path", ""))).exists()
        )
        if cache_is_compatible and cached_end_ordinal >= requested_end_ordinal:
            if requested_latest_allowed_ordinal > cached_latest_allowed_ordinal:
                return _refresh_existing_rebalance_training_context(
                    ticker_data_grouped=ticker_data_grouped,
                    valid_tickers=valid_tickers,
                    train_end_date=train_end_date,
                    forward_days=forward_days,
                    context=cache_context,
                )
            print(
                f"   ♻️ AI Rebalance: Reusing file cache "
                f"({len(valid_tickers)} tickers, {cached_n_dates} cached dates)"
            )
            return cache_context
        if not cache_is_compatible:
            cleanup_rebalance_training_context(cache_context)
            cache_context = None

    if cache_context:
        return _extend_rebalance_training_context(
            ticker_data_grouped=ticker_data_grouped,
            market_df=market_df,
            valid_tickers=valid_tickers,
            train_end_date=train_end_date,
            forward_days=forward_days,
            context=cache_context,
        )

    return _build_rebalance_training_context(
        ticker_data_grouped=ticker_data_grouped,
        market_df=market_df,
        valid_tickers=valid_tickers,
        train_start_date=cache_start,
        train_end_date=cache_end,
        latest_allowed_date=train_end_date,
        forward_days=forward_days,
    )


def _build_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame],
    valid_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    latest_allowed_date: datetime,
    forward_days: int,
) -> Dict[str, object]:
    start_key = _rebalance_cache_key(train_start_date)
    end_key = _rebalance_cache_key(train_end_date)
    start_ordinal = start_key.toordinal()
    n_dates = max(1, int((end_key - start_key).days) + 1)
    temp_dir = get_cache_dir(
        "ai_rebalance/context",
        _rebalance_context_cache_key(valid_tickers, train_start_date, train_end_date, forward_days),
    )
    feature_path = temp_dir / "ticker_features.npy"
    forward_path = temp_dir / "forward_returns.npy"
    market_path = temp_dir / "market_features.npy"

    feature_array = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), n_dates, len(TICKER_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), n_dates),
    )
    market_array = np.lib.format.open_memmap(
        market_path,
        mode="w+",
        dtype=np.float64,
        shape=(n_dates, len(MARKET_FEATURE_NAMES)),
    )
    forward_array.fill(np.nan)

    print(f"   🗂️ AI Rebalance: Building file cache ({len(valid_tickers)} tickers, {n_dates} dates)...")
    print("   🌐 AI Rebalance: Populating market feature cache...")
    market_cache_start = time.perf_counter()
    _populate_rebalance_market_cache(
        market_array=market_array,
        market_df=market_df,
        start_ordinal=start_ordinal,
        fill_start_ordinal=start_ordinal,
        fill_end_ordinal=start_ordinal + n_dates - 1,
        reference_date=train_end_date,
    )
    market_cache_elapsed = time.perf_counter() - market_cache_start
    print(f"   ✅ AI Rebalance: Market feature cache ready ({market_cache_elapsed:.1f}s)")
    _populate_rebalance_ticker_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        feature_array=feature_array,
        forward_array=forward_array,
        start_ordinal=start_ordinal,
        fill_start_ordinal=start_ordinal,
        fill_end_ordinal=start_ordinal + n_dates - 1,
        forward_days=forward_days,
        latest_allowed_date=latest_allowed_date,
        progress_label="build",
    )

    feature_array.flush()
    forward_array.flush()
    market_array.flush()
    print("   💾 AI Rebalance: File cache ready")
    return {
        "all_tickers": valid_tickers,
        "start_ordinal": start_ordinal,
        "n_dates": n_dates,
        "feature_path": str(feature_path),
        "forward_path": str(forward_path),
        "market_path": str(market_path),
        "temp_dir": str(temp_dir),
        "forward_days": int(forward_days),
        "latest_allowed_ordinal": _rebalance_cache_key(latest_allowed_date).toordinal(),
        "persistent": True,
    }


def _extend_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame],
    valid_tickers: List[str],
    train_end_date: datetime,
    forward_days: int,
    context: Dict[str, object],
) -> Dict[str, object]:
    start_ordinal = int(context["start_ordinal"])
    old_n_dates = int(context["n_dates"])
    old_end_ordinal = start_ordinal + old_n_dates - 1
    requested_end_ordinal = _rebalance_cache_key(train_end_date).toordinal()
    if requested_end_ordinal <= old_end_ordinal:
        return context

    new_n_dates = requested_end_ordinal - start_ordinal + 1
    temp_dir = Path(str(context["temp_dir"]))
    old_feature_path = Path(str(context["feature_path"]))
    old_forward_path = Path(str(context["forward_path"]))
    old_market_path = Path(str(context["market_path"]))
    feature_path = temp_dir / f"ticker_features_{requested_end_ordinal}.npy"
    forward_path = temp_dir / f"forward_returns_{requested_end_ordinal}.npy"
    market_path = temp_dir / f"market_features_{requested_end_ordinal}.npy"

    old_feature_array = np.load(str(old_feature_path), mmap_mode="r")
    old_forward_array = np.load(str(old_forward_path), mmap_mode="r")
    old_market_array = np.load(str(old_market_path), mmap_mode="r")

    feature_array = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), new_n_dates, len(TICKER_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(valid_tickers), new_n_dates),
    )
    market_array = np.lib.format.open_memmap(
        market_path,
        mode="w+",
        dtype=np.float64,
        shape=(new_n_dates, len(MARKET_FEATURE_NAMES)),
    )
    forward_array.fill(np.nan)
    feature_array[:, :old_n_dates] = old_feature_array
    forward_array[:, :old_n_dates] = old_forward_array
    market_array[:old_n_dates] = old_market_array

    append_days = requested_end_ordinal - old_end_ordinal
    print(
        f"   ♻️ AI Rebalance: Extending file cache by {append_days} day(s) "
        f"to {new_n_dates} total cached dates"
    )
    print("   🌐 AI Rebalance: Extending market feature cache...")
    market_cache_start = time.perf_counter()
    _populate_rebalance_market_cache(
        market_array=market_array,
        market_df=market_df,
        start_ordinal=start_ordinal,
        fill_start_ordinal=old_end_ordinal + 1,
        fill_end_ordinal=requested_end_ordinal,
        reference_date=train_end_date,
    )
    market_cache_elapsed = time.perf_counter() - market_cache_start
    print(f"   ✅ AI Rebalance: Market feature cache ready ({market_cache_elapsed:.1f}s)")
    forward_refresh_start = max(start_ordinal, old_end_ordinal - forward_days + 1)
    _populate_rebalance_ticker_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        feature_array=feature_array,
        forward_array=forward_array,
        start_ordinal=start_ordinal,
        fill_start_ordinal=old_end_ordinal + 1,
        fill_end_ordinal=requested_end_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
        progress_label="append",
    )
    _refresh_rebalance_forward_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        forward_array=forward_array,
        start_ordinal=start_ordinal,
        refresh_start_ordinal=forward_refresh_start,
        refresh_end_ordinal=requested_end_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
    )

    feature_array.flush()
    forward_array.flush()
    market_array.flush()

    del old_feature_array
    del old_forward_array
    del old_market_array
    for old_path in (old_feature_path, old_forward_path, old_market_path):
        try:
            old_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("   💾 AI Rebalance: File cache ready")
    return {
        "all_tickers": valid_tickers,
        "start_ordinal": start_ordinal,
        "n_dates": new_n_dates,
        "feature_path": str(feature_path),
        "forward_path": str(forward_path),
        "market_path": str(market_path),
        "temp_dir": str(temp_dir),
        "forward_days": int(forward_days),
        "latest_allowed_ordinal": _rebalance_cache_key(train_end_date).toordinal(),
    }


def _refresh_existing_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    train_end_date: datetime,
    forward_days: int,
    context: Dict[str, object],
) -> Dict[str, object]:
    start_ordinal = int(context["start_ordinal"])
    latest_allowed_ordinal = _rebalance_cache_key(train_end_date).toordinal()
    old_latest_allowed_ordinal = int(context.get("latest_allowed_ordinal", start_ordinal - 1))
    if latest_allowed_ordinal <= old_latest_allowed_ordinal:
        return context

    forward_array = np.load(str(context["forward_path"]), mmap_mode="r+")
    refresh_start_ordinal = max(start_ordinal, old_latest_allowed_ordinal - forward_days + 1)
    print(
        f"   ♻️ AI Rebalance: Refreshing forward labels by "
        f"{latest_allowed_ordinal - old_latest_allowed_ordinal} day(s)"
    )
    _refresh_rebalance_forward_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        forward_array=forward_array,
        start_ordinal=start_ordinal,
        refresh_start_ordinal=refresh_start_ordinal,
        refresh_end_ordinal=latest_allowed_ordinal,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
    )
    forward_array.flush()
    del forward_array
    updated_context = dict(context)
    updated_context["latest_allowed_ordinal"] = latest_allowed_ordinal
    return updated_context


def _rebalance_datetime_from_ordinal(ordinal: int, reference_date: datetime) -> datetime:
    current_date = datetime.fromordinal(ordinal)
    if reference_date.tzinfo is not None:
        current_date = current_date.replace(tzinfo=reference_date.tzinfo)
    return current_date


def _populate_rebalance_market_cache(
    market_array,
    market_df: Optional[pd.DataFrame],
    start_ordinal: int,
    fill_start_ordinal: int,
    fill_end_ordinal: int,
    reference_date: datetime,
) -> None:
    if fill_end_ordinal < fill_start_ordinal:
        return
    for ordinal in range(fill_start_ordinal, fill_end_ordinal + 1):
        current_date = _rebalance_datetime_from_ordinal(ordinal, reference_date)
        date_idx = ordinal - start_ordinal
        market_array[date_idx] = _market_feature_vector(_market_features_from_df(market_df, current_date))


def _get_ai_rebalance_cache_context() -> Dict[str, object]:
    global _AI_REBALANCE_CACHE_CONTEXT
    if _AI_REBALANCE_CACHE_CONTEXT is None:
        if not _AI_REBALANCE_CACHE_CONTEXT_PATH:
            raise ValueError("AI Rebalance cache context is not initialized")
        _AI_REBALANCE_CACHE_CONTEXT = joblib.load(_AI_REBALANCE_CACHE_CONTEXT_PATH)

    if "_feature_array" not in _AI_REBALANCE_CACHE_CONTEXT and _AI_REBALANCE_CACHE_CONTEXT.get("feature_path"):
        _AI_REBALANCE_CACHE_CONTEXT["_feature_array"] = np.load(
            str(_AI_REBALANCE_CACHE_CONTEXT["feature_path"]),
            mmap_mode="r+",
        )
    if "_forward_array" not in _AI_REBALANCE_CACHE_CONTEXT and _AI_REBALANCE_CACHE_CONTEXT.get("forward_path"):
        _AI_REBALANCE_CACHE_CONTEXT["_forward_array"] = np.load(
            str(_AI_REBALANCE_CACHE_CONTEXT["forward_path"]),
            mmap_mode="r+",
        )
    return _AI_REBALANCE_CACHE_CONTEXT


def ensure_ai_rebalance_feature_cache_context(
    context: Optional[Dict[str, object]],
    mmap_mode: str = "r",
) -> Optional[Dict[str, object]]:
    if not isinstance(context, dict):
        return None
    if "_feature_array" not in context and context.get("feature_path"):
        context["_feature_array"] = np.load(str(context["feature_path"]), mmap_mode=mmap_mode)
    if "_forward_array" not in context and context.get("forward_path"):
        context["_forward_array"] = np.load(str(context["forward_path"]), mmap_mode=mmap_mode)
    if "_market_array" not in context and context.get("market_path"):
        context["_market_array"] = np.load(str(context["market_path"]), mmap_mode=mmap_mode)
    if "ticker_to_idx" not in context:
        context["ticker_to_idx"] = {
            ticker: idx for idx, ticker in enumerate(context.get("all_tickers") or [])
        }
    return context


def clone_ai_rebalance_feature_cache_context(
    context: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not isinstance(context, dict):
        return None
    return {
        key: value
        for key, value in context.items()
        if not str(key).startswith("_")
    }


def get_ai_rebalance_cached_feature_row(
    held_ticker: str,
    candidate_ticker: str,
    ranked_candidates: List[str],
    current_holdings: List[str],
    current_date: datetime,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
    cache_context: Optional[Dict[str, object]],
) -> Optional[Dict[str, float]]:
    context = ensure_ai_rebalance_feature_cache_context(cache_context, mmap_mode="r")
    if context is None:
        return None

    ticker_to_idx = context.get("ticker_to_idx") or {}
    held_idx = ticker_to_idx.get(held_ticker)
    candidate_idx = ticker_to_idx.get(candidate_ticker)
    if held_idx is None or candidate_idx is None:
        return None

    start_ordinal = int(context.get("start_ordinal", 0))
    n_dates = int(context.get("n_dates", 0))
    date_idx = _rebalance_cache_key(current_date).toordinal() - start_ordinal
    if date_idx < 0 or date_idx >= n_dates:
        return None

    feature_array = context.get("_feature_array")
    market_array = context.get("_market_array")
    if feature_array is None or market_array is None:
        return None

    return _build_rebalance_feature_row_from_vectors(
        held_ticker=held_ticker,
        candidate_ticker=candidate_ticker,
        ranked_candidates=ranked_candidates,
        current_holdings=current_holdings,
        held_vector=np.asarray(feature_array[held_idx, date_idx], dtype=np.float64),
        candidate_vector=np.asarray(feature_array[candidate_idx, date_idx], dtype=np.float64),
        market_vector=np.asarray(market_array[date_idx], dtype=np.float64),
        transaction_cost=transaction_cost,
        portfolio_size=portfolio_size,
        buffer_size=buffer_size,
    )


def _init_ai_rebalance_cache_worker(
    context_path: Optional[str] = None,
    context_data: Optional[Dict[str, object]] = None,
) -> None:
    global _AI_REBALANCE_CACHE_CONTEXT_PATH, _AI_REBALANCE_CACHE_CONTEXT
    _AI_REBALANCE_CACHE_CONTEXT_PATH = context_path
    _AI_REBALANCE_CACHE_CONTEXT = context_data


def _build_ticker_cache_row(args) -> None:
    ticker_idx, ticker = args
    context = _get_ai_rebalance_cache_context()
    ticker_df = context["ticker_data_grouped"].get(ticker)
    feature_array = context["_feature_array"]
    forward_array = context["_forward_array"]
    fill_start_ordinal = int(context["fill_start_ordinal"])
    fill_end_ordinal = int(context["fill_end_ordinal"])
    start_ordinal = int(context["start_ordinal"])
    forward_days = int(context["forward_days"])
    latest_allowed_date = context["latest_allowed_date"]

    for ordinal in range(fill_start_ordinal, fill_end_ordinal + 1):
        current_date = _rebalance_datetime_from_ordinal(ordinal, latest_allowed_date)
        date_idx = ordinal - start_ordinal
        feature_array[ticker_idx, date_idx] = _ticker_feature_vector(
            _ticker_features_from_df(ticker_df, current_date)
        )
        forward_ret = _calculate_forward_return_from_df(
            ticker_df=ticker_df,
            current_date=current_date,
            forward_days=forward_days,
            latest_allowed_date=latest_allowed_date,
        )
        if forward_ret is not None:
            forward_array[ticker_idx, date_idx] = float(forward_ret)
        else:
            forward_array[ticker_idx, date_idx] = np.nan


def _refresh_ticker_forward_row(args) -> None:
    ticker_idx, ticker = args
    context = _get_ai_rebalance_cache_context()
    ticker_df = context["ticker_data_grouped"].get(ticker)
    forward_array = context["_forward_array"]
    refresh_start_ordinal = int(context["refresh_start_ordinal"])
    refresh_end_ordinal = int(context["refresh_end_ordinal"])
    start_ordinal = int(context["start_ordinal"])
    forward_days = int(context["forward_days"])
    latest_allowed_date = context["latest_allowed_date"]

    for ordinal in range(refresh_start_ordinal, refresh_end_ordinal + 1):
        current_date = _rebalance_datetime_from_ordinal(ordinal, latest_allowed_date)
        date_idx = ordinal - start_ordinal
        forward_ret = _calculate_forward_return_from_df(
            ticker_df=ticker_df,
            current_date=current_date,
            forward_days=forward_days,
            latest_allowed_date=latest_allowed_date,
        )
        if forward_ret is not None:
            forward_array[ticker_idx, date_idx] = float(forward_ret)
        else:
            forward_array[ticker_idx, date_idx] = np.nan


def _populate_rebalance_ticker_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    feature_array,
    forward_array,
    start_ordinal: int,
    fill_start_ordinal: int,
    fill_end_ordinal: int,
    forward_days: int,
    latest_allowed_date: datetime,
    progress_label: str,
) -> None:
    if fill_end_ordinal < fill_start_ordinal:
        return
    from config import NUM_PROCESSES

    worker_args = list(enumerate(valid_tickers))
    n_workers = max(1, min(NUM_PROCESSES, len(worker_args))) if worker_args else 1
    cache_context = {
        "ticker_data_grouped": ticker_data_grouped,
        "feature_path": str(feature_array.filename),
        "forward_path": str(forward_array.filename),
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
            global _AI_REBALANCE_CACHE_CONTEXT_PATH, _AI_REBALANCE_CACHE_CONTEXT
            _AI_REBALANCE_CACHE_CONTEXT_PATH = None
            _AI_REBALANCE_CACHE_CONTEXT = cache_context
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=_init_ai_rebalance_cache_worker,
                initargs=(None, cache_context),
            ) as pool:
                results = pool.imap_unordered(
                    _build_ticker_cache_row,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    results,
                    total=len(worker_args),
                    desc=f"   AI Rebalance cache {progress_label}",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
        else:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                temp_context_path = temp_context_file.name
            joblib.dump(cache_context, temp_context_path)
            with Pool(
                processes=n_workers,
                initializer=_init_ai_rebalance_cache_worker,
                initargs=(temp_context_path,),
            ) as pool:
                results = pool.imap_unordered(
                    _build_ticker_cache_row,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    results,
                    total=len(worker_args),
                    desc=f"   AI Rebalance cache {progress_label}",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
    finally:
        _AI_REBALANCE_CACHE_CONTEXT_PATH = None
        _AI_REBALANCE_CACHE_CONTEXT = None
        if temp_context_path:
            try:
                os.unlink(temp_context_path)
            except OSError:
                pass


def _refresh_rebalance_forward_cache(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    valid_tickers: List[str],
    forward_array,
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
        "forward_path": str(forward_array.filename),
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
            global _AI_REBALANCE_CACHE_CONTEXT_PATH, _AI_REBALANCE_CACHE_CONTEXT
            _AI_REBALANCE_CACHE_CONTEXT_PATH = None
            _AI_REBALANCE_CACHE_CONTEXT = cache_context
            with get_context("fork").Pool(
                processes=n_workers,
                initializer=_init_ai_rebalance_cache_worker,
                initargs=(None, cache_context),
            ) as pool:
                results = pool.imap_unordered(
                    _refresh_ticker_forward_row,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    results,
                    total=len(worker_args),
                    desc="   AI Rebalance cache refresh",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
        else:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                temp_context_path = temp_context_file.name
            joblib.dump(cache_context, temp_context_path)
            with Pool(
                processes=n_workers,
                initializer=_init_ai_rebalance_cache_worker,
                initargs=(temp_context_path,),
            ) as pool:
                results = pool.imap_unordered(
                    _refresh_ticker_forward_row,
                    worker_args,
                    chunksize=chunksize,
                )
                for _ in tqdm(
                    results,
                    total=len(worker_args),
                    desc="   AI Rebalance cache refresh",
                    ncols=100,
                    unit="ticker",
                ):
                    pass
    finally:
        _AI_REBALANCE_CACHE_CONTEXT_PATH = None
        _AI_REBALANCE_CACHE_CONTEXT = None
        if temp_context_path:
            try:
                os.unlink(temp_context_path)
            except OSError:
                pass


def init_rebalance_collection_worker(context: Dict[str, object]) -> None:
    """Load file-backed readonly caches once per worker process."""
    global _REBALANCE_COLLECTION_CONTEXT
    _REBALANCE_COLLECTION_CONTEXT = {
        "all_tickers": list(context.get("all_tickers") or []),
        "ticker_to_idx": {ticker: idx for idx, ticker in enumerate(context.get("all_tickers") or [])},
        "start_ordinal": int(context.get("start_ordinal", 0)),
        "n_dates": int(context.get("n_dates", 0)),
        "feature_array": np.load(str(context["feature_path"]), mmap_mode="r"),
        "forward_array": np.load(str(context["forward_path"]), mmap_mode="r"),
        "market_array": np.load(str(context["market_path"]), mmap_mode="r"),
    }


def cleanup_rebalance_training_context(context: Optional[Dict[str, object]]) -> None:
    """Delete the persistent temp cache directory when it is no longer needed."""
    if not context:
        return
    if context.get("persistent"):
        return
    temp_dir = context.get("temp_dir")
    if not temp_dir:
        return
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    finally:
        _REBALANCE_TEMP_DIRS.discard(str(temp_dir))


def collect_rebalance_ticker_training_data(
    ticker: str,
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
    ticker_to_idx: Dict[str, int],
    feature_array,
    forward_array,
    market_array,
    start_ordinal: int,
    n_dates: int,
) -> List[Dict[str, object]]:
    """Collect training samples for a single ticker from file-backed caches."""
    ticker_idx = ticker_to_idx.get(ticker)
    if ticker_idx is None:
        return []

    if train_start_date.tzinfo is None:
        train_start_date = train_start_date.replace(tzinfo=timezone.utc)
    if train_end_date.tzinfo is None:
        train_end_date = train_end_date.replace(tzinfo=timezone.utc)

    samples = []
    current_date = train_start_date

    other_tickers = [t for t in all_tickers if t != ticker and t in ticker_to_idx]
    if not other_tickers:
        return []

    while current_date <= train_end_date:
        try:
            date_idx = _rebalance_cache_key(current_date).toordinal() - start_ordinal
            if date_idx < 0 or date_idx >= n_dates:
                current_date += timedelta(days=1)
                continue
            keep_return = float(forward_array[ticker_idx, date_idx])
            if np.isnan(keep_return):
                current_date += timedelta(days=1)
                continue

            candidate_idx = hash((ticker, current_date.toordinal())) % len(other_tickers)
            candidate_ticker = other_tickers[candidate_idx]
            candidate_ticker_idx = ticker_to_idx.get(candidate_ticker)
            if candidate_ticker_idx is None:
                current_date += timedelta(days=1)
                continue

            replace_return = float(forward_array[candidate_ticker_idx, date_idx])
            if np.isnan(replace_return):
                current_date += timedelta(days=1)
                continue

            simulated_holdings = [ticker] + other_tickers[:portfolio_size - 1]
            simulated_candidates = other_tickers[:buffer_size]

            row = _build_rebalance_feature_row_from_vectors(
                held_ticker=ticker,
                candidate_ticker=candidate_ticker,
                ranked_candidates=simulated_candidates,
                current_holdings=simulated_holdings,
                held_vector=np.asarray(feature_array[ticker_idx, date_idx], dtype=np.float64),
                candidate_vector=np.asarray(feature_array[candidate_ticker_idx, date_idx], dtype=np.float64),
                market_vector=np.asarray(market_array[date_idx], dtype=np.float64),
                transaction_cost=transaction_cost,
                portfolio_size=portfolio_size,
                buffer_size=buffer_size,
            )
            switch_advantage = replace_return - keep_return - (2.0 * transaction_cost)
            row.update({
                "label": switch_advantage,
                "held_ticker": ticker,
                "candidate_ticker": candidate_ticker,
                "sample_date": current_date,
            })
            samples.append(row)
        except Exception:
            pass

        current_date += timedelta(days=1)

    return samples


def _collect_rebalance_data_worker(args):
    """Top-level worker for multiprocessing Pool - collects rebalance training data for one ticker."""
    (
        ticker,
        all_tickers,
        train_start,
        train_end,
        forward_days,
        transaction_cost,
        portfolio_size,
        buffer_size,
    ) = args
    context = _REBALANCE_COLLECTION_CONTEXT
    samples = collect_rebalance_ticker_training_data(
        ticker=ticker,
        all_tickers=all_tickers,
        train_start_date=train_start,
        train_end_date=train_end,
        forward_days=forward_days,
        transaction_cost=transaction_cost,
        portfolio_size=portfolio_size,
        buffer_size=buffer_size,
        ticker_to_idx=context.get("ticker_to_idx", {}),
        feature_array=context.get("feature_array"),
        forward_array=context.get("forward_array"),
        market_array=context.get("market_array"),
        start_ordinal=int(context.get("start_ordinal", 0)),
        n_dates=int(context.get("n_dates", 0)),
    )
    return ticker, samples


def _save_model(model, path: str, metadata: dict = None):
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.exists():
            backup_path = path_obj.with_suffix(path_obj.suffix + ".backup")
            import shutil

            shutil.copy2(path_obj, backup_path)
        payload = {
            "model": model,
            "metadata": metadata or {},
        }
        joblib.dump(payload, path_obj)
        save_native_model_artifacts(model, path_obj)
    except Exception as exc:
        print(f"   ⚠️ AI Rebalance: Failed to save model to {path}: {exc}")


def load_ai_rebalance_model(path: str | Path = AI_REBALANCE_MODEL_PATH):
    path_obj = Path(path)
    if not path_obj.exists():
        return None

    try:
        loaded = joblib.load(path_obj)
        metadata = {}
        if isinstance(loaded, dict) and "model" in loaded:
            metadata = dict(loaded.get("metadata") or {})
            loaded_model = loaded["model"]
        else:
            loaded_model = loaded
        loaded_model = restore_native_model_artifacts(loaded_model, path_obj)
        loaded_model, reset_catboost = reset_legacy_catboost_member(loaded_model)
        if reset_catboost:
            print("   ♻️ AI Rebalance: Resetting saved CatBoost member for clean CPU continuation")
        if metadata.get("trained"):
            print(f"   ✅ AI Rebalance: Loaded model from {path_obj} (trained {metadata['trained'][:10]})")
        else:
            print(f"   ✅ AI Rebalance: Loaded model from {path_obj}")
        return loaded_model
    except Exception as exc:
        print(f"   ⚠️ AI Rebalance: Failed to load model: {exc}")
        return None


def train_ai_rebalance_model(
    all_training_data: List[Dict[str, object]],
    save_path: str | Path = AI_REBALANCE_MODEL_PATH,
    existing_model=None,
    train_start: Optional[datetime] = None,
    train_end: Optional[datetime] = None,
):
    try:
        from config import AI_REBALANCE_MIN_SAMPLES, XGBOOST_USE_GPU
        from sklearn.model_selection import train_test_split
        import time

        if len(all_training_data) < AI_REBALANCE_MIN_SAMPLES:
            print(
                f"   ⚠️ AI Rebalance: Insufficient training data "
                f"({len(all_training_data)} samples, need {AI_REBALANCE_MIN_SAMPLES})"
            )
            return existing_model, None

        train_df = pd.DataFrame(all_training_data)
        X = train_df[FEATURE_COLS].fillna(0.0)
        y = train_df["label"].astype(float).values
        if len(X) < AI_REBALANCE_MIN_SAMPLES or np.all(y == y[0]):
            print("   ⚠️ AI Rebalance: Training target not usable, keeping existing model")
            return existing_model, None

        has_existing = existing_model is not None and isinstance(existing_model, dict) and "all_models" in existing_model
        device = "cuda" if XGBOOST_USE_GPU else "cpu"

        status_msg = "Continuing" if has_existing else "Training NEW"
        print(f"   📊 AI Rebalance: {status_msg} training on {len(X)} samples...")

        if has_existing:
            models = dict(existing_model.get("all_models") or {})
            for model_name in (
                "CatBoost",
                "RandomForest",
                "ExtraTrees",
                "Ridge",
                "ElasticNet",
                "SGDRegressor-L2",
                "SGDRegressor-ElasticNet",
            ):
                if model_name not in models:
                    try:
                        models[model_name] = _fresh_ensemble_model(model_name, device)
                    except ImportError:
                        pass
            for model_name, model in models.items():
                _configure_existing_model_backend(model_name, model, device)
            models = _order_models_for_training(models)
            print(
                f"   🚀 AI Rebalance: Incremental training {len(models)} models "
                f"({_format_training_plan(list(models.keys()), device)})"
            )
        else:
            models = {}
            try:
                models["CatBoost"] = _fresh_ensemble_model("CatBoost", device)
            except ImportError:
                pass
            models["XGBoost"] = _fresh_ensemble_model("XGBoost", device)
            models["LightGBM"] = _fresh_ensemble_model("LightGBM", device)
            models["RandomForest"] = _fresh_ensemble_model("RandomForest", device)
            models["ExtraTrees"] = _fresh_ensemble_model("ExtraTrees", device)
            models["Ridge"] = _fresh_ensemble_model("Ridge", device)
            models["ElasticNet"] = _fresh_ensemble_model("ElasticNet", device)
            models["SGDRegressor-L2"] = _fresh_ensemble_model("SGDRegressor-L2", device)
            models["SGDRegressor-ElasticNet"] = _fresh_ensemble_model("SGDRegressor-ElasticNet", device)
            models = _order_models_for_training(models)
            print(
                f"   🚀 AI Rebalance: Fresh training {list(models.keys())} "
                f"({_format_training_plan(list(models.keys()), device)})"
            )

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")

        trained_models = []
        model_scores = []
        model_names = []

        model_names_to_train = list(models.keys())
        X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        X_val_np = X_val.values if hasattr(X_val, "values") else X_val
        y_train_np = np.asarray(y_train)
        y_val_np = np.asarray(y_val)

        spawn_model_names: List[str] = []
        parallel_model_names = list(model_names_to_train)
        for name in list(parallel_model_names):
            if _model_worker_mode(name, device) == "spawn":
                parallel_model_names.remove(name)
                spawn_model_names.append(name)

        total_cores = os.cpu_count() or 8
        n_train_workers = max(1, min(3, len(parallel_model_names), total_cores)) if parallel_model_names else 1
        n_jobs_per_model = max(1, total_cores // n_train_workers)

        print(
            f"   🚀 AI Rebalance: Training {len(model_names_to_train)} models in parallel "
            f"({n_train_workers} fork workers, {n_jobs_per_model} threads/model; "
            f"{len(spawn_model_names)} spawn models)..."
        )
        if spawn_model_names:
            print("   ℹ️ AI Rebalance: GPU models use spawned workers; CPU models stay on fork workers")
            for name in spawn_model_names:
                print(f"      🔄 {name}: queued (spawn, backend={_model_backend(name, device)})")

        global _AI_REBALANCE_TRAIN_CONTEXT_PATH, _AI_REBALANCE_TRAIN_CONTEXT
        _AI_REBALANCE_TRAIN_CONTEXT = {
            "models": models,
            "X_train": X_train_np,
            "y_train": y_train_np,
            "X_val": X_val_np,
            "y_val": y_val_np,
            "device": device,
            "has_existing": has_existing,
        }
        train_context_path = None
        try:
            pending: Dict[str, object] = {}
            fork_pool = None
            spawn_pool = None
            try:
                if os.name != "nt" and parallel_model_names and n_train_workers > 1:
                    fork_pool = get_context("fork").Pool(
                        processes=n_train_workers,
                        initializer=_init_ai_rebalance_train_worker,
                        initargs=(None, _AI_REBALANCE_TRAIN_CONTEXT),
                    )
                    for name in parallel_model_names:
                        print(f"      🔄 {name}: queued (fork, backend={_model_backend(name, device)})")
                        pending[name] = fork_pool.apply_async(_train_single_ai_rebalance_model, (name, n_jobs_per_model))
                else:
                    for name in parallel_model_names:
                        print(f"      🔄 {name}: starting (main-process, backend={_model_backend(name, device)})")
                        ok, model, score, elapsed, status, detail = _train_single_ai_rebalance_model(name, n_jobs_limit=-1)
                        if ok:
                            print(f"      🔄 {name}: R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                            trained_models.append(model)
                            model_scores.append(score)
                            model_names.append(name)
                        else:
                            print(f"      ⚠️ {name}: {detail} ({status}, {elapsed:.1f}s)")

                if spawn_model_names:
                    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                        train_context_path = temp_context_file.name
                    joblib.dump(_AI_REBALANCE_TRAIN_CONTEXT, train_context_path)
                    _AI_REBALANCE_TRAIN_CONTEXT_PATH = train_context_path
                    spawn_pool = get_context("spawn").Pool(
                        processes=len(spawn_model_names),
                        initializer=_init_ai_rebalance_train_worker,
                        initargs=(train_context_path, None),
                    )
                    for name in spawn_model_names:
                        pending[name] = spawn_pool.apply_async(_train_single_ai_rebalance_model, (name, 1))

                while pending:
                    ready_names = [name for name, handle in pending.items() if handle.ready()]
                    if not ready_names:
                        time.sleep(0.5)
                        continue
                    for name in ready_names:
                        result_handle = pending.pop(name)
                        try:
                            ok, model, score, elapsed, status, detail = result_handle.get(timeout=0)
                            if ok:
                                print(f"      🔄 {name}: R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                                trained_models.append(model)
                                model_scores.append(score)
                                model_names.append(name)
                            else:
                                print(f"      ⚠️ {name}: {detail} ({status}, {elapsed:.1f}s)")
                        except Exception as exc:
                            print(f"      ⚠️ {name}: Training error: {exc}")
            finally:
                if fork_pool is not None:
                    fork_pool.close()
                    fork_pool.join()
                if spawn_pool is not None:
                    spawn_pool.close()
                    spawn_pool.join()
        finally:
            _AI_REBALANCE_TRAIN_CONTEXT_PATH = None
            _AI_REBALANCE_TRAIN_CONTEXT = None
            if train_context_path is not None:
                try:
                    os.unlink(train_context_path)
                except OSError as exc:
                    print(f"   ⚠️ AI Rebalance: Failed to remove train context: {exc}")

        if not trained_models:
            print("   ⚠️ AI Rebalance: No models trained successfully")
            return existing_model, None

        best_idx = max(range(len(model_scores)), key=lambda idx: model_scores[idx])
        best_name = model_names[best_idx]
        best_score = model_scores[best_idx]
        best_model = trained_models[best_idx]

        model_dict = {
            "all_models": dict(zip(model_names, trained_models)),
            "all_scores": dict(zip(model_names, model_scores)),
            "best_model": best_model,
            "best_name": best_name,
            "best_score": best_score,
            "feature_cols": FEATURE_COLS,
        }

        if save_path:
            metadata = {
                "trained": datetime.now(timezone.utc).isoformat(),
                "best_model": best_name,
                "best_r2": best_score,
                "all_scores": dict(zip(model_names, model_scores)),
                "catboost_backend": "cpu",
            }
            if train_start and train_end:
                metadata["train_start"] = train_start.isoformat()
                metadata["train_end"] = train_end.isoformat()
            _save_model(model_dict, str(save_path), metadata)

        print(f"   ✅ AI Rebalance: Best model = {best_name} (R² {best_score:.3f})")
        return model_dict, best_score
    finally:
        release_runtime_memory()


def load_voting_ai_rebalance_model(
    path: str | Path = VOTING_AI_REBALANCE_MODEL_PATH,
):
    """Compatibility wrapper for the voting-ensemble AI rebalance checkpoint."""
    return load_ai_rebalance_model(path=path)


def train_voting_ai_rebalance_model(
    all_training_data: List[Dict[str, object]],
    save_path: str | Path = VOTING_AI_REBALANCE_MODEL_PATH,
    existing_model=None,
    train_start: Optional[datetime] = None,
    train_end: Optional[datetime] = None,
):
    """Compatibility wrapper that preserves the voting-ensemble naming used by backtesting."""
    return train_ai_rebalance_model(
        all_training_data=all_training_data,
        save_path=save_path,
        existing_model=existing_model,
        train_start=train_start,
        train_end=train_end,
    )


def choose_ai_rebalance_ranked_candidates(
    current_holdings: List[str],
    ranked_candidates: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    model_bundle,
    transaction_cost: float,
    portfolio_size: int,
    buffer_size: int,
    min_predicted_edge: float,
    cache_context: Optional[Dict[str, object]] = None,
) -> Tuple[List[str], List[Tuple[str, str, float, str]]]:
    if not current_holdings:
        return list(ranked_candidates), []
    if not model_bundle or not isinstance(model_bundle, dict):
        return list(ranked_candidates), []

    best_model = model_bundle.get("best_model")
    feature_cols = model_bundle.get("feature_cols") or FEATURE_COLS
    if best_model is None:
        return list(ranked_candidates), []

    remaining_candidates = [ticker for ticker in ranked_candidates if ticker not in set(current_holdings)]
    if not remaining_candidates:
        return list(ranked_candidates), []

    keepers: List[str] = []
    reserved_replacements: List[str] = []
    decision_logs: List[Tuple[str, str, float, str]] = []

    for held_ticker in current_holdings:
        candidate_ticker = next(
            (ticker for ticker in remaining_candidates if ticker not in reserved_replacements),
            None,
        )
        if candidate_ticker is None:
            keepers.append(held_ticker)
            continue

        row = get_ai_rebalance_cached_feature_row(
            held_ticker=held_ticker,
            candidate_ticker=candidate_ticker,
            ranked_candidates=ranked_candidates,
            current_holdings=current_holdings,
            current_date=current_date,
            transaction_cost=transaction_cost,
            portfolio_size=portfolio_size,
            buffer_size=buffer_size,
            cache_context=cache_context,
        )
        if row is None:
            row = build_rebalance_feature_row(
                held_ticker=held_ticker,
                candidate_ticker=candidate_ticker,
                ranked_candidates=ranked_candidates,
                current_holdings=current_holdings,
                ticker_data_grouped=ticker_data_grouped,
                current_date=current_date,
                transaction_cost=transaction_cost,
                portfolio_size=portfolio_size,
                buffer_size=buffer_size,
            )
        X = pd.DataFrame([[row.get(col, 0.0) for col in feature_cols]], columns=feature_cols)

        try:
            predicted_edge = float(best_model.predict(X)[0])
        except Exception as exc:
            print(f"   ⚠️ AI Rebalance: Prediction failed for {held_ticker}: {exc}")
            keepers.append(held_ticker)
            continue

        action = "replace" if predicted_edge > min_predicted_edge else "keep"
        decision_logs.append((held_ticker, candidate_ticker, predicted_edge, action))
        if action == "replace":
            reserved_replacements.append(candidate_ticker)
        else:
            keepers.append(held_ticker)

    remaining_order = [
        ticker for ticker in ranked_candidates
        if ticker not in keepers and ticker not in reserved_replacements
    ]
    ai_ranked_candidates = keepers + reserved_replacements + remaining_order
    return ai_ranked_candidates, decision_logs
