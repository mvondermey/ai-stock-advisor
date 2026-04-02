"""
Vol-Sweet Momentum (VSM) Strategy

Goal: beat existing Risk-Adj Mom 3M by lowering volatility without sacrificing much return.
Steps per ticker:
1. Base score = 3-Month Risk-Adjusted Momentum = return_3m / sqrt(volatility_pct)
2. Volatility sweet spot filter: accept only annualised daily volatility between 20% and 35%.
   • If outside, down-weight score linearly to zero at 10% or 60%.
3. Sentiment adjustment (price-derived) – reuse calculate_sentiment_score from risk_adj_mom_3m_sentiment_strategy.
   final_score = base_score * (1 + 0.20 * sentiment)   # ±20 % adjustment.

Return top N tickers by final_score.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from risk_adj_mom_3m_sentiment_strategy import calculate_sentiment_score

# Constants
PERF_WINDOW = 90  # days
VOL_SWEET_MIN = 20.0  # %
VOL_SWEET_MAX = 35.0  # %
VOL_CUTOFF_LOW = 10.0  # below this score -> 0
VOL_CUTOFF_HIGH = 60.0  # above this score -> 0
SENTIMENT_WEIGHT = 0.20


def _risk_adj_return(basic_ret: float, vol_pct: float) -> float:
    return basic_ret / (np.sqrt(max(vol_pct, 1e-3)))


def _vol_factor(vol_pct: float) -> float:
    if VOL_SWEET_MIN <= vol_pct <= VOL_SWEET_MAX:
        return 1.0  # perfect sweet spot
    if vol_pct < VOL_SWEET_MIN:
        if vol_pct <= VOL_CUTOFF_LOW:
            return 0.0
        # scale up from 0-1 between cutoff and sweet min
        return (vol_pct - VOL_CUTOFF_LOW) / (VOL_SWEET_MIN - VOL_CUTOFF_LOW)
    # vol_pct > VOL_SWEET_MAX
    if vol_pct >= VOL_CUTOFF_HIGH:
        return 0.0
    return (VOL_CUTOFF_HIGH - vol_pct) / (VOL_CUTOFF_HIGH - VOL_SWEET_MAX)


def select_vol_sweet_mom_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    verbose: bool = True,
) -> List[str]:
    from performance_filters import filter_tickers_by_performance

    # Min score for annualized vol (much lower than daily vol threshold)
    MIN_SCORE = 0.5  # With annualized vol ~30-40%, scores are ~1-5 range

    filtered = filter_tickers_by_performance(all_tickers, ticker_data_grouped, current_date, "VolSweetMom")
    if verbose:
        print(f"   📊 VSM: analysing {len(filtered)} tickers")

    candidates = []
    for tkr in filtered:
        try:
            df = ticker_data_grouped.get(tkr)
            if df is None or len(df) < 10:  # Reduced from 100
                continue
            close = df["Close"].dropna()
            if len(close) < 30:
                continue

            # 3-month return
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

            # Sentiment
            sent = calculate_sentiment_score(df, current_date)
            score *= (1 + SENTIMENT_WEIGHT * sent)

            candidates.append((tkr, score, basic_ret, vol_pct, sent))
        except Exception:
            continue

    if not candidates:
        if verbose:
            print("   ⚠️ VSM: no candidates")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [c[0] for c in candidates[:top_n]]

    if verbose:
        print(f"   ✅ VSM: selected {selected}")
    return selected
