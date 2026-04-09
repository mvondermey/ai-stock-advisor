"""
Risk-Adj Mom 1M + Vol-Sweet Strategy

Goal: Combine best Sharpe (1M momentum) with volatility sweet spot filtering.
Based on Day 51 analysis: RiskAdj 1M Mth had best Sharpe (Return=40.8%, StdDev=21.8%)

Steps per ticker:
1. Base score = 1-Month Risk-Adjusted Momentum = return_1m / sqrt(volatility_pct)
2. Volatility sweet spot filter: accept only annualised daily volatility between 15% and 30%.
   • If outside, down-weight score linearly to zero at 8% or 50%.
3. Rebalance: daily

Return top N tickers by final_score.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_history_up_to,
    resolve_cache_current_date,
)

# Constants
PERF_WINDOW = 21  # ~1 month trading days
VOL_SWEET_MIN = 15.0  # %
VOL_SWEET_MAX = 30.0  # %
VOL_CUTOFF_LOW = 8.0  # below this score -> 0
VOL_CUTOFF_HIGH = 50.0  # above this score -> 0


def _risk_adj_return(basic_ret: float, vol_pct: float) -> float:
    return basic_ret / (np.sqrt(max(vol_pct, 1e-3)))


def _vol_factor(vol_pct: float) -> float:
    if VOL_SWEET_MIN <= vol_pct <= VOL_SWEET_MAX:
        return 1.0  # perfect sweet spot
    if vol_pct < VOL_SWEET_MIN:
        if vol_pct <= VOL_CUTOFF_LOW:
            return 0.0
        return (vol_pct - VOL_CUTOFF_LOW) / (VOL_SWEET_MIN - VOL_CUTOFF_LOW)
    if vol_pct >= VOL_CUTOFF_HIGH:
        return 0.0
    return (VOL_CUTOFF_HIGH - vol_pct) / (VOL_CUTOFF_HIGH - VOL_SWEET_MAX)


def select_risk_adj_mom_1m_vol_sweet_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    verbose: bool = True,
    price_history_cache=None,
) -> List[str]:
    from performance_filters import filter_tickers_by_performance

    # Min score for annualized vol (much lower than daily vol threshold)
    MIN_SCORE = 0.5  # With annualized vol ~30-40%, scores are ~1-5 range

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    filtered = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "RiskAdj1MVol",
        price_history_cache=price_history_cache,
    )
    current_date = resolve_cache_current_date(price_history_cache, current_date, filtered)
    if current_date is None:
        return []
    if verbose:
        print(f"   📊 1M VolSweet: analyzing {len(filtered)} tickers")

    candidates = []
    for tkr in filtered:
        try:
            close = get_cached_history_up_to(
                price_history_cache,
                tkr,
                current_date,
                field_name="close",
                min_rows=60,
            )
            if close is None or len(close) < 60:
                continue

            # 1-month return
            latest = close[-1]
            start = close[-PERF_WINDOW] if len(close) > PERF_WINDOW else close[0]
            basic_ret = (latest / start - 1) * 100

            # Annualized vol (daily std * sqrt(252) * 100)
            daily_returns = np.diff(close) / close[:-1]
            if daily_returns.size == 0:
                continue
            daily_vol = float(np.std(daily_returns, ddof=1))
            vol_pct = daily_vol * np.sqrt(252) * 100  # annualized %
            if vol_pct <= 0:
                continue

            base_score = _risk_adj_return(basic_ret, vol_pct)
            if base_score < MIN_SCORE:
                continue

            # Vol factor
            vf = _vol_factor(vol_pct)
            if vf == 0:
                continue
            score = base_score * vf

            candidates.append((tkr, score, basic_ret, vol_pct))
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error processing {tkr}: {e}")
            continue

    if not candidates:
        if verbose:
            print("   ⚠️ 1M VolSweet: no candidates")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [c[0] for c in candidates[:top_n]]

    if verbose:
        print(f"   ✅ 1M VolSweet: selected {selected}")
    return selected
