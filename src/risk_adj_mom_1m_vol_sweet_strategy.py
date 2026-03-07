"""
Risk-Adj Mom 1M + Vol-Sweet Strategy

Goal: Combine best Sharpe (1M momentum) with volatility sweet spot filtering.
Based on Day 51 analysis: RiskAdj 1M Mth had best Sharpe (Return=40.8%, StdDev=21.8%)

Steps per ticker:
1. Base score = 1-Month Risk-Adjusted Momentum = return_1m / sqrt(volatility_pct)
2. Volatility sweet spot filter: accept only annualised daily volatility between 15% and 30%.
   • If outside, down-weight score linearly to zero at 8% or 50%.
3. Rebalance: monthly (start of month only)

Return top N tickers by final_score.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

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
) -> List[str]:
    from performance_filters import filter_tickers_by_performance

    # Min score for annualized vol (much lower than daily vol threshold)
    MIN_SCORE = 0.5  # With annualized vol ~30-40%, scores are ~1-5 range

    filtered = filter_tickers_by_performance(all_tickers, ticker_data_grouped, current_date, "RiskAdj1MVol")
    print(f"   📊 1M VolSweet: analyzing {len(filtered)} tickers")

    candidates = []
    for tkr in filtered:
        try:
            df = ticker_data_grouped.get(tkr)
            if df is None or len(df) < 60:
                continue
            close = df["Close"].dropna()
            if len(close) < 21:
                continue

            # 1-month return
            latest = close.iloc[-1]
            start = close.iloc[-PERF_WINDOW] if len(close) > PERF_WINDOW else close.iloc[0]
            basic_ret = (latest / start - 1) * 100

            # Annualized vol (daily std * sqrt(252) * 100)
            daily_vol = close.pct_change().dropna().std()
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
        except Exception:
            continue

    if not candidates:
        print("   ⚠️ 1M VolSweet: no candidates")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [c[0] for c in candidates[:top_n]]

    print(f"   ✅ 1M VolSweet: selected {selected}")
    return selected
