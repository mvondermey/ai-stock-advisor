"""
AI rebalance model for Voting Ensemble-style strategies.

Ticker selection remains separate: this module only decides whether replacing
an existing holding with the best available candidate is worth it.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_elite_strategy_per_ticker import (
    _catboost_has_trained_trees,
    _configure_catboost_continuation,
    _fresh_ensemble_model,
    _order_models_for_training,
    _predictions_are_unstable,
)
from model_training_safety import (
    release_runtime_memory,
    reset_legacy_catboost_member,
    restore_native_model_artifacts,
    save_native_model_artifacts,
)


MODEL_SAVE_DIR = Path("logs/models")
VOTING_AI_REBALANCE_MODEL_PATH = MODEL_SAVE_DIR / "voting_ensemble_ai_rebalance.joblib"
_REBALANCE_COLLECTION_CONTEXT: Dict[str, object] = {}
_REBALANCE_TEMP_DIRS: set[str] = set()
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
    return np.asarray([float(features.get(name, 0.0)) for name in TICKER_FEATURE_NAMES], dtype=np.float32)


def _market_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in MARKET_FEATURE_NAMES], dtype=np.float32)


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


def _valid_rebalance_tickers(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
) -> List[str]:
    return [
        ticker
        for ticker in all_tickers
        if ticker_data_grouped.get(ticker) is not None and len(ticker_data_grouped.get(ticker)) > 0
    ]


def _cleanup_rebalance_temp_dirs_at_exit() -> None:
    for temp_dir in list(_REBALANCE_TEMP_DIRS):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        finally:
            _REBALANCE_TEMP_DIRS.discard(temp_dir)


atexit.register(_cleanup_rebalance_temp_dirs_at_exit)


def precompute_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    existing_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build or extend file-backed numeric caches so workers mmap shared readonly data."""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    market_df = ticker_data_grouped.get("SPY")
    if market_df is None or len(market_df) == 0:
        market_df = ticker_data_grouped.get("QQQ")

    valid_tickers = _valid_rebalance_tickers(ticker_data_grouped, all_tickers)
    start_key = _rebalance_cache_key(train_start_date)
    end_key = _rebalance_cache_key(train_end_date)
    requested_start_ordinal = start_key.toordinal()
    requested_end_ordinal = end_key.toordinal()

    cache_context = existing_context if isinstance(existing_context, dict) else None
    cache_is_compatible = False
    if cache_context:
        cached_tickers = list(cache_context.get("all_tickers") or [])
        cached_start_ordinal = int(cache_context.get("start_ordinal", 0))
        cached_n_dates = int(cache_context.get("n_dates", 0))
        cached_end_ordinal = cached_start_ordinal + cached_n_dates - 1
        cache_is_compatible = (
            cached_tickers == valid_tickers
            and int(cache_context.get("forward_days", -1)) == int(forward_days)
            and cached_start_ordinal <= requested_start_ordinal
            and Path(str(cache_context.get("feature_path", ""))).exists()
            and Path(str(cache_context.get("forward_path", ""))).exists()
            and Path(str(cache_context.get("market_path", ""))).exists()
        )
        if cache_is_compatible and cached_end_ordinal >= requested_end_ordinal:
            print(
                f"   ♻️ Voting AI Reb: Reusing file cache "
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
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        forward_days=forward_days,
    )


def _build_rebalance_training_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame],
    valid_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
) -> Dict[str, object]:
    start_key = _rebalance_cache_key(train_start_date)
    end_key = _rebalance_cache_key(train_end_date)
    start_ordinal = start_key.toordinal()
    n_dates = max(1, int((end_key - start_key).days) + 1)
    temp_dir = Path(tempfile.mkdtemp(prefix="voting_ai_reb_cache_", dir=str(MODEL_SAVE_DIR)))
    _REBALANCE_TEMP_DIRS.add(str(temp_dir))
    feature_path = temp_dir / "ticker_features.npy"
    forward_path = temp_dir / "forward_returns.npy"
    market_path = temp_dir / "market_features.npy"

    feature_array = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(valid_tickers), n_dates, len(TICKER_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(valid_tickers), n_dates),
    )
    market_array = np.lib.format.open_memmap(
        market_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_dates, len(MARKET_FEATURE_NAMES)),
    )
    forward_array.fill(np.nan)

    print(f"   🗂️ Voting AI Reb: Building file cache ({len(valid_tickers)} tickers, {n_dates} dates)...")
    _populate_rebalance_market_cache(
        market_array=market_array,
        market_df=market_df,
        start_ordinal=start_ordinal,
        fill_start_ordinal=start_ordinal,
        fill_end_ordinal=start_ordinal + n_dates - 1,
        reference_date=train_end_date,
    )
    _populate_rebalance_ticker_cache(
        ticker_data_grouped=ticker_data_grouped,
        valid_tickers=valid_tickers,
        feature_array=feature_array,
        forward_array=forward_array,
        start_ordinal=start_ordinal,
        fill_start_ordinal=start_ordinal,
        fill_end_ordinal=start_ordinal + n_dates - 1,
        forward_days=forward_days,
        latest_allowed_date=train_end_date,
        progress_label="build",
    )

    feature_array.flush()
    forward_array.flush()
    market_array.flush()
    print("   💾 Voting AI Reb: File cache ready")
    return {
        "all_tickers": valid_tickers,
        "start_ordinal": start_ordinal,
        "n_dates": n_dates,
        "feature_path": str(feature_path),
        "forward_path": str(forward_path),
        "market_path": str(market_path),
        "temp_dir": str(temp_dir),
        "forward_days": int(forward_days),
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
        dtype=np.float32,
        shape=(len(valid_tickers), new_n_dates, len(TICKER_FEATURE_NAMES)),
    )
    forward_array = np.lib.format.open_memmap(
        forward_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(valid_tickers), new_n_dates),
    )
    market_array = np.lib.format.open_memmap(
        market_path,
        mode="w+",
        dtype=np.float32,
        shape=(new_n_dates, len(MARKET_FEATURE_NAMES)),
    )
    forward_array.fill(np.nan)
    feature_array[:, :old_n_dates] = old_feature_array
    forward_array[:, :old_n_dates] = old_forward_array
    market_array[:old_n_dates] = old_market_array

    append_days = requested_end_ordinal - old_end_ordinal
    print(
        f"   ♻️ Voting AI Reb: Extending file cache by {append_days} day(s) "
        f"to {new_n_dates} total cached dates"
    )
    _populate_rebalance_market_cache(
        market_array=market_array,
        market_df=market_df,
        start_ordinal=start_ordinal,
        fill_start_ordinal=old_end_ordinal + 1,
        fill_end_ordinal=requested_end_ordinal,
        reference_date=train_end_date,
    )
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

    print("   💾 Voting AI Reb: File cache ready")
    return {
        "all_tickers": valid_tickers,
        "start_ordinal": start_ordinal,
        "n_dates": new_n_dates,
        "feature_path": str(feature_path),
        "forward_path": str(forward_path),
        "market_path": str(market_path),
        "temp_dir": str(temp_dir),
        "forward_days": int(forward_days),
    }


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
    iterator = tqdm(
        enumerate(valid_tickers),
        total=len(valid_tickers),
        desc=f"   Voting AI Reb cache {progress_label}",
        ncols=100,
        unit="ticker",
    )
    for ticker_idx, ticker in iterator:
        ticker_df = ticker_data_grouped.get(ticker)
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
                forward_array[ticker_idx, date_idx] = np.float32(forward_ret)
            else:
                forward_array[ticker_idx, date_idx] = np.nan


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
    iterator = tqdm(
        enumerate(valid_tickers),
        total=len(valid_tickers),
        desc="   Voting AI Reb cache refresh",
        ncols=100,
        unit="ticker",
    )
    for ticker_idx, ticker in iterator:
        ticker_df = ticker_data_grouped.get(ticker)
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
                forward_array[ticker_idx, date_idx] = np.float32(forward_ret)
            else:
                forward_array[ticker_idx, date_idx] = np.nan


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
                held_vector=np.asarray(feature_array[ticker_idx, date_idx], dtype=np.float32),
                candidate_vector=np.asarray(feature_array[candidate_ticker_idx, date_idx], dtype=np.float32),
                market_vector=np.asarray(market_array[date_idx], dtype=np.float32),
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
        print(f"   ⚠️ Voting AI Reb: Failed to save model to {path}: {exc}")


def load_voting_ai_rebalance_model(path: str | Path = VOTING_AI_REBALANCE_MODEL_PATH):
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
            print("   ♻️ Voting AI Reb: Resetting saved CatBoost member for clean CPU continuation")
        if metadata.get("trained"):
            print(f"   ✅ Voting AI Reb: Loaded model from {path_obj} (trained {metadata['trained'][:10]})")
        else:
            print(f"   ✅ Voting AI Reb: Loaded model from {path_obj}")
        return loaded_model
    except Exception as exc:
        print(f"   ⚠️ Voting AI Reb: Failed to load model: {exc}")
        return None


def train_voting_ai_rebalance_model(
    all_training_data: List[Dict[str, object]],
    save_path: str | Path = VOTING_AI_REBALANCE_MODEL_PATH,
    existing_model=None,
    train_start: Optional[datetime] = None,
    train_end: Optional[datetime] = None,
):
    try:
        from config import AI_VOTING_REBALANCE_MIN_SAMPLES, XGBOOST_USE_GPU
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        import time
        import warnings
        import xgboost as xgb

        if len(all_training_data) < AI_VOTING_REBALANCE_MIN_SAMPLES:
            print(
                f"   ⚠️ Voting AI Reb: Insufficient training data "
                f"({len(all_training_data)} samples, need {AI_VOTING_REBALANCE_MIN_SAMPLES})"
            )
            return existing_model, None

        train_df = pd.DataFrame(all_training_data)
        X = train_df[FEATURE_COLS].fillna(0.0)
        y = train_df["label"].astype(float).values
        if len(X) < AI_VOTING_REBALANCE_MIN_SAMPLES or np.all(y == y[0]):
            print("   ⚠️ Voting AI Reb: Training target not usable, keeping existing model")
            return existing_model, None

        has_existing = existing_model is not None and isinstance(existing_model, dict) and "all_models" in existing_model
        device = "cuda" if XGBOOST_USE_GPU else "cpu"

        status_msg = "Continuing" if has_existing else "Training NEW"
        print(f"   📊 Voting AI Reb: {status_msg} training on {len(X)} samples...")

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
            models = _order_models_for_training(models)
            print(
                f"   🚀 Voting AI Reb: Incremental training {len(models)} models "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu, SGD=cpu)"
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
                f"   🚀 Voting AI Reb: Fresh training {list(models.keys())} "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu, SGD=cpu)"
            )

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        trained_models = []
        model_scores = []
        model_names = []

        for name, model in models.items():
            try:
                print(f"      🔄 {name}: Training...", end=" ", flush=True)
                start_time = time.time()
                if len(X_train) < 10:
                    print(f"skipped (insufficient data: {len(X_train)} samples)")
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    incremental_failed = False
                    used_incremental = False

                    if has_existing:
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
                                    _configure_catboost_continuation(model)
                                    model.fit(X_train, y_train, init_model=model)
                                else:
                                    print("(no saved trees yet, training fresh)...", end=" ", flush=True)
                                    incremental_failed = True
                            elif name in ("SGDRegressor-L2", "SGDRegressor-ElasticNet"):
                                used_incremental = True
                                model.partial_fit(X_train, y_train)
                            if used_incremental:
                                quick_pred = model.predict(X_val[: min(100, len(X_val))])
                                if _predictions_are_unstable(quick_pred):
                                    print("(incremental unstable, retraining fresh)...", end=" ", flush=True)
                                    incremental_failed = True
                        except Exception as exc:
                            print(f"(incremental failed: {exc}, retraining fresh)...", end=" ", flush=True)
                            incremental_failed = True

                    if not used_incremental or incremental_failed:
                        model = _fresh_ensemble_model(name, device)
                        model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    if _predictions_are_unstable(y_pred):
                        print("failed: unstable predictions")
                        continue

                    score = r2_score(y_val, y_pred)
                    if used_incremental and (score < -10 or score > 1 or np.isnan(score) or np.isinf(score)):
                        print(f"⚠️ retraining fresh after invalid R² {score:.3f}...", end=" ", flush=True)
                        model = _fresh_ensemble_model(name, device)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        if _predictions_are_unstable(y_pred):
                            print("failed: unstable after fresh retrain")
                            continue
                        score = r2_score(y_val, y_pred)

                    if score < -10 or score > 1 or np.isnan(score) or np.isinf(score):
                        print(f"failed: invalid validation score {score:.3f}")
                        continue

                elapsed = time.time() - start_time
                status = "incremental" if used_incremental and not incremental_failed else "fresh"
                print(f"R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                trained_models.append(model)
                model_scores.append(score)
                model_names.append(name)
            except Exception as exc:
                print(f"failed: {exc}")

        if not trained_models:
            print("   ⚠️ Voting AI Reb: No models trained successfully")
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

        print(f"   ✅ Voting AI Reb: Best model = {best_name} (R² {best_score:.3f})")
        return model_dict, best_score
    finally:
        release_runtime_memory()


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
            print(f"   ⚠️ Voting AI Reb: Prediction failed for {held_ticker}: {exc}")
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
