"""
Persistent Momentum Core (PMC) Strategy
=======================================

Designed to beat the best existing momentum strategies (Dynamic BH 1Y,
Risk-Adj Mom 6M, Mom-Vol Hybrid 6M) by stacking their winning traits while
avoiding the failure modes of short-horizon and contrarian strategies.

Winning traits stacked:
  1. Long 1Y momentum core (50% weight) -- Dynamic BH 1Y pattern
  2. 6M risk-adjusted component (35% weight) -- Risk-Adj Mom 6M pattern
  3. 3M confirmation (15% weight) -- avoid buying dying names
  4. Inverse-volatility scoring -- penalize high-vol noise
  5. Rank-persistence bonus -- reward stocks repeatedly in top-50 by 1Y
  6. Hard reject on rank collapse -- exit broken leaders early (CELH/AZTA pattern)

The selection function returns a deterministic, sorted list of tickers.
Position sizing and monthly rebalancing are handled by the caller
(`backtesting.py`) so this module stays purely signal-generation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_cache_adapter import (
    ensure_price_history_cache,
    resolve_cache_current_date,
)


# ---------------------------------------------------------------------------
# Config constants (defaults; overridden from config.py when available)
# ---------------------------------------------------------------------------
PMC_MIN_DATA_DAYS = 253            # ~1 trading year
PMC_MIN_YEAR_WINDOW_ROWS = 200     # min rows in trailing-year window
PMC_MIN_PRICE = 5.0                # minimum price to avoid low-quality tickers
PMC_WEIGHT_1Y = 0.50
PMC_WEIGHT_6M = 0.35
PMC_WEIGHT_3M = 0.15
PMC_VOL_FLOOR = 0.15               # min annualized vol used in ratio (avoid div/0)
PMC_RANK_PERSISTENCE_WINDOW = 30   # days
PMC_RANK_PERSISTENCE_TOPN = 50     # must be in top-N by 1Y to count
PMC_RANK_COLLAPSE_WINDOW = 5       # days
PMC_RANK_COLLAPSE_DROP = 30        # reject if rank fell > this many places in window
PMC_REQUIRE_POSITIVE_3M = True     # hard reject if 3M momentum negative
PMC_MIN_1Y_PERFORMANCE = 0.10      # +10% 1Y minimum (aligns with perf filter)
PMC_MIN_6M_PERFORMANCE = 0.05      # +5% 6M minimum
PMC_MIN_3M_PERFORMANCE = 0.025     # +2.5% 3M minimum


def _load_config_overrides() -> None:
    """Load PMC_* overrides from config.py if they exist."""
    global PMC_MIN_DATA_DAYS, PMC_MIN_YEAR_WINDOW_ROWS, PMC_MIN_PRICE
    global PMC_WEIGHT_1Y, PMC_WEIGHT_6M, PMC_WEIGHT_3M
    global PMC_VOL_FLOOR
    global PMC_RANK_PERSISTENCE_WINDOW, PMC_RANK_PERSISTENCE_TOPN
    global PMC_RANK_COLLAPSE_WINDOW, PMC_RANK_COLLAPSE_DROP
    global PMC_REQUIRE_POSITIVE_3M
    global PMC_MIN_1Y_PERFORMANCE, PMC_MIN_6M_PERFORMANCE, PMC_MIN_3M_PERFORMANCE
    try:
        import config
    except ImportError:
        return
    for name in (
        'PMC_MIN_DATA_DAYS', 'PMC_MIN_YEAR_WINDOW_ROWS', 'PMC_MIN_PRICE',
        'PMC_WEIGHT_1Y', 'PMC_WEIGHT_6M', 'PMC_WEIGHT_3M',
        'PMC_VOL_FLOOR',
        'PMC_RANK_PERSISTENCE_WINDOW', 'PMC_RANK_PERSISTENCE_TOPN',
        'PMC_RANK_COLLAPSE_WINDOW', 'PMC_RANK_COLLAPSE_DROP',
        'PMC_REQUIRE_POSITIVE_3M',
        'PMC_MIN_1Y_PERFORMANCE', 'PMC_MIN_6M_PERFORMANCE', 'PMC_MIN_3M_PERFORMANCE',
    ):
        if hasattr(config, name):
            globals()[name] = getattr(config, name)


def _annualized_return(start_price: float, end_price: float, days: int) -> float:
    """Annualize a total return over `days` trading days."""
    if start_price <= 0 or end_price <= 0 or days <= 0:
        return 0.0
    total_return = (end_price / start_price) - 1.0
    if total_return <= -0.999:
        return -0.999
    return (1.0 + total_return) ** (252.0 / days) - 1.0


def _normalize_timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _build_daily_close_cache(
    price_history_cache,
    universe: List[str],
    current_date: datetime,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build per-ticker daily close arrays from the shared intraday cache.

    Returns ticker -> (daily_date_ns, daily_close_values), where dates are
    normalized to daily midnight UTC and values are the last valid close of each day.
    """
    current_ns = int(_normalize_timestamp(current_date).value)
    daily_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for ticker in universe:
        date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
        close_values = price_history_cache.close_by_ticker.get(ticker)
        if date_ns is None or close_values is None or date_ns.size == 0 or close_values.size == 0:
            continue

        end_idx = int(np.searchsorted(date_ns, current_ns, side="right"))
        if end_idx < 2:
            continue

        ticker_dates = date_ns[:end_idx]
        ticker_closes = np.asarray(close_values[:end_idx], dtype=float)
        valid_mask = ~np.isnan(ticker_closes)
        if not np.any(valid_mask):
            continue

        series = pd.Series(
            ticker_closes[valid_mask],
            index=pd.to_datetime(ticker_dates[valid_mask], unit="ns", utc=True),
        )
        daily_series = series.resample("1D").last().dropna()
        if len(daily_series) < 2:
            continue

        daily_index = daily_series.index.tz_convert("UTC").tz_localize(None)
        daily_cache[ticker] = (
            daily_index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=True),
            daily_series.to_numpy(dtype=float, copy=True),
        )

    return daily_cache


def _select_reference_trading_dates(
    daily_close_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    universe: List[str],
    window: int,
) -> np.ndarray:
    """
    Choose a reference trading calendar for persistence scoring.

    Prefer broad-market ETFs when available; otherwise fall back to the ticker
    with the longest daily history in the current universe.
    """
    for preferred in ("SPY", "QQQ", "IWM", "DIA"):
        if preferred in daily_close_cache:
            dates, _ = daily_close_cache[preferred]
            return dates[-window:] if len(dates) >= window else dates

    best_ticker = None
    best_len = 0
    for ticker in universe:
        cached = daily_close_cache.get(ticker)
        if cached is None:
            continue
        dates, _ = cached
        if len(dates) > best_len:
            best_ticker = ticker
            best_len = len(dates)

    if best_ticker is None:
        return np.asarray([], dtype=np.int64)

    dates, _ = daily_close_cache[best_ticker]
    return dates[-window:] if len(dates) >= window else dates


def _compute_rank_persistence(
    ranks_series: np.ndarray,
    top_n: int,
    window: int,
) -> int:
    """Count how many of the last `window` days this ticker was ranked <= top_n."""
    if ranks_series is None or len(ranks_series) == 0:
        return 0
    tail = ranks_series[-window:]
    return int(np.sum((tail > 0) & (tail <= top_n)))


def _precompute_daily_1y_ranks(
    daily_close_cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    universe: List[str],
    trading_dates: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    For each reference trading day, compute the 1Y return rank of
    every ticker. Returns a dict ticker -> np.ndarray of ranks (1 = best).

    Tickers with insufficient data on a given day get rank 0 (ignored).
    """
    returns_by_date: List[Dict[str, float]] = []

    for trading_date_ns in trading_dates:
        returns_today: Dict[str, float] = {}
        for ticker in universe:
            cached = daily_close_cache.get(ticker)
            if cached is None:
                continue

            ticker_dates, ticker_closes = cached
            end_idx = int(np.searchsorted(ticker_dates, int(trading_date_ns), side="right"))
            if end_idx < 253:
                continue

            start = float(ticker_closes[end_idx - 253])
            end = float(ticker_closes[end_idx - 1])
            if start <= 0 or end <= 0 or np.isnan(start) or np.isnan(end):
                continue
            returns_today[ticker] = (end / start) - 1.0
        returns_by_date.append(returns_today)

    # Build ranks: for each date, sort by return desc and assign rank
    ranks_by_ticker: Dict[str, List[int]] = {t: [] for t in universe}
    for returns_today in returns_by_date:
        if not returns_today:
            for t in universe:
                ranks_by_ticker[t].append(0)
            continue
        sorted_tickers = sorted(returns_today.items(), key=lambda kv: kv[1], reverse=True)
        rank_map = {tk: i + 1 for i, (tk, _) in enumerate(sorted_tickers)}
        for t in universe:
            ranks_by_ticker[t].append(rank_map.get(t, 0))

    return {t: np.asarray(r, dtype=int) for t, r in ranks_by_ticker.items()}


def select_persistent_momentum_core_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """
    Select the top_n stocks by Persistent Momentum Core score.

    Args:
        all_tickers: Pre-filtered universe (top-2000 by 1Y performance) -- used directly.
        ticker_data_grouped: Dict of ticker -> OHLCV DataFrame.
        current_date: Analysis date (UTC).
        top_n: Number of stocks to select.
        price_history_cache: Optional shared price cache.

    Returns:
        Deterministic, sorted list of ticker symbols (highest PMC score first).
    """
    _load_config_overrides()

    from config import INVERSE_ETFS

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    universe = [t for t in all_tickers if t not in INVERSE_ETFS]
    current_date = resolve_cache_current_date(price_history_cache, current_date, universe)
    if current_date is None:
        print("   ⚠️ PMC: Could not resolve current_date -- empty selection")
        return []

    print(f"   🔄 Persistent Momentum Core: Scanning {len(universe)} tickers @ "
          f"{pd.Timestamp(current_date).strftime('%Y-%m-%d')}")

    daily_close_cache = _build_daily_close_cache(price_history_cache, universe, current_date)

    reference_window = max(PMC_RANK_PERSISTENCE_WINDOW, PMC_RANK_COLLAPSE_WINDOW + 1)
    trading_dates = _select_reference_trading_dates(daily_close_cache, universe, window=reference_window)
    if trading_dates.size == 0:
        print("   ⚠️ PMC: Could not derive reference trading dates -- empty selection")
        return []

    # --- Precompute rolling 1Y ranks over the persistence window ---
    ranks_by_ticker = _precompute_daily_1y_ranks(
        daily_close_cache,
        universe,
        trading_dates,
    )

    candidates = []
    skipped_insufficient = 0
    skipped_price_low = 0
    skipped_perf_1y = 0
    skipped_perf_6m = 0
    skipped_perf_3m = 0
    skipped_rank_collapse = 0

    for ticker in universe:
        try:
            cached = daily_close_cache.get(ticker)
            if cached is None:
                skipped_insufficient += 1
                continue

            _, closes = cached
            closes = np.asarray(closes, dtype=float)
            latest = float(closes[-1])
            if np.isnan(latest) or latest < PMC_MIN_PRICE:
                skipped_price_low += 1
                continue

            # Horizon prices
            n = len(closes)
            if n < PMC_MIN_DATA_DAYS:
                skipped_insufficient += 1
                continue

            lookback_1y = 252
            lookback_6m = 126
            lookback_3m = 63

            p_1y = float(closes[-lookback_1y - 1]) if lookback_1y + 1 <= n else float("nan")
            p_6m = float(closes[-lookback_6m - 1]) if lookback_6m + 1 <= n else float("nan")
            p_3m = float(closes[-lookback_3m - 1]) if lookback_3m + 1 <= n else float("nan")

            if any(np.isnan(x) or x <= 0 for x in (p_1y, p_6m, p_3m)):
                skipped_insufficient += 1
                continue

            total_1y = (latest - p_1y) / p_1y
            total_6m = (latest - p_6m) / p_6m
            total_3m = (latest - p_3m) / p_3m

            # Threshold gates (replicate the performance filter inline for clarity)
            if total_1y < PMC_MIN_1Y_PERFORMANCE:
                skipped_perf_1y += 1
                continue
            if total_6m < PMC_MIN_6M_PERFORMANCE:
                skipped_perf_6m += 1
                continue
            if total_3m < PMC_MIN_3M_PERFORMANCE:
                skipped_perf_3m += 1
                continue

            ann_1y = _annualized_return(p_1y, latest, lookback_1y)
            ann_6m = _annualized_return(p_6m, latest, lookback_6m)
            ann_3m = _annualized_return(p_3m, latest, lookback_3m)

            # --- Volatility (20-trading-day realized, annualized) ---
            if n >= 21:
                recent_returns = np.diff(closes[-21:]) / closes[-21:-1]
                recent_returns = recent_returns[~np.isnan(recent_returns)]
                if len(recent_returns) >= 5:
                    vol_20d = float(np.std(recent_returns, ddof=1)) * np.sqrt(252.0)
                else:
                    vol_20d = PMC_VOL_FLOOR
            else:
                vol_20d = PMC_VOL_FLOOR
            vol_denom = max(vol_20d, PMC_VOL_FLOOR)

            # --- Composite annualized momentum ---
            composite_mom = (
                PMC_WEIGHT_1Y * ann_1y
                + PMC_WEIGHT_6M * ann_6m
                + PMC_WEIGHT_3M * ann_3m
            )

            # --- Risk-adjusted core score ---
            risk_adj = composite_mom / vol_denom

            # --- Rank persistence bonus ---
            ranks_arr = ranks_by_ticker.get(ticker, np.array([], dtype=int))
            persistence = _compute_rank_persistence(
                ranks_arr, PMC_RANK_PERSISTENCE_TOPN, PMC_RANK_PERSISTENCE_WINDOW
            )
            # Up to 2x multiplier when ticker was in top-N every day of window
            persistence_mult = 1.0 + (persistence / max(PMC_RANK_PERSISTENCE_WINDOW, 1))

            # --- Rank collapse hard reject ---
            if len(ranks_arr) >= PMC_RANK_COLLAPSE_WINDOW + 1:
                recent_rank = int(ranks_arr[-1])
                past_rank = int(ranks_arr[-1 - PMC_RANK_COLLAPSE_WINDOW])
                if recent_rank > 0 and past_rank > 0:
                    # Positive drop means ticker fell (rank number went UP)
                    rank_drop = recent_rank - past_rank
                    if rank_drop > PMC_RANK_COLLAPSE_DROP:
                        skipped_rank_collapse += 1
                        continue

            # --- Penalty if 3M momentum negative (belt-and-suspenders w/ gate) ---
            three_m_penalty = 1.0 if total_3m > 0 else 0.5

            score = risk_adj * persistence_mult * three_m_penalty

            candidates.append({
                'ticker': ticker,
                'score': score,
                'ann_1y': ann_1y,
                'ann_6m': ann_6m,
                'ann_3m': ann_3m,
                'vol_20d': vol_20d,
                'persistence': persistence,
                'composite_mom': composite_mom,
            })

        except Exception as e:
            print(f"   ⚠️ PMC error processing {ticker}: {e}")
            continue

    # Sort deterministically: primary = score desc, tiebreak = ticker asc
    candidates.sort(key=lambda c: (-c['score'], c['ticker']))

    print(
        f"   📊 PMC: {len(candidates)} candidates from {len(universe)} | "
        f"Skipped: insufficient={skipped_insufficient}, price_low={skipped_price_low}, "
        f"perf_1y={skipped_perf_1y}, perf_6m={skipped_perf_6m}, perf_3m={skipped_perf_3m}, "
        f"rank_collapse={skipped_rank_collapse}"
    )

    selected = candidates[:top_n]
    if selected:
        print(f"   🎯 PMC selected {len(selected)} stocks:")
        for c in selected:
            print(
                f"      {c['ticker']}: score={c['score']:.3f}, "
                f"1Y={c['ann_1y']*100:+.1f}%, 6M={c['ann_6m']*100:+.1f}%, "
                f"3M={c['ann_3m']*100:+.1f}%, vol={c['vol_20d']*100:.1f}%, "
                f"persist={c['persistence']}/{PMC_RANK_PERSISTENCE_WINDOW}"
            )

    return [c['ticker'] for c in selected]
