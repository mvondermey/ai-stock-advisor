"""
Risk-Adjusted Momentum 3M Strategy
Identical to Risk-Adj Mom but uses 3-month (90-day) performance window instead of 1-year.
score = return_3m / sqrt(volatility)
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime


def select_risk_adj_mom_3m_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
) -> List[str]:
    from performance_filters import filter_tickers_by_performance
    from config import (
        RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION,
        RISK_ADJ_MOM_MIN_CONFIRMATIONS,
        RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION,
        RISK_ADJ_MOM_VOLUME_WINDOW,
        RISK_ADJ_MOM_VOLUME_MULTIPLIER,
        RISK_ADJ_MOM_MIN_SCORE,
        INVERSE_ETFS,
    )

    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Risk-Adj Mom 3M"
    )

    PERF_WINDOW = 90  # 3 months

    candidates = []
    print(f"   📊 Risk-Adj Mom 3M: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            data = ticker_data_grouped[ticker]
            if data is None or len(data) == 0:
                continue

            close = data['Close'].dropna()
            n = len(close)
            if n < 30:
                continue

            latest_price = close.iloc[-1]
            if latest_price <= 0:
                continue

            perf_window = min(PERF_WINDOW, n - 1)
            if perf_window < 30:
                continue

            start_price = close.iloc[-perf_window]
            if start_price <= 0:
                continue

            basic_return = (latest_price - start_price) / start_price * 100

            daily_returns = close.pct_change().dropna()
            if len(daily_returns) < 20:
                continue

            volatility_pct = daily_returns.std() * 100
            if volatility_pct <= 0:
                continue

            score = basic_return / (volatility_pct ** 0.5 + 0.001)

            if score <= RISK_ADJ_MOM_MIN_SCORE:
                continue

            # Momentum confirmation
            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                confirmations = 0
                for days in [30, 60, 90]:
                    lookback = min(days, n - 1)
                    p = close.iloc[-lookback]
                    if p > 0 and (latest_price - p) / p > 0:
                        confirmations += 1
                if confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    continue

            # Volume confirmation
            if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                vol_series = data['Volume'].dropna()
                if len(vol_series) >= RISK_ADJ_MOM_VOLUME_WINDOW + 10:
                    recent_vol = vol_series.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
                    avg_vol = vol_series.iloc[:-RISK_ADJ_MOM_VOLUME_WINDOW].mean()
                    if avg_vol > 0 and recent_vol < avg_vol * RISK_ADJ_MOM_VOLUME_MULTIPLIER:
                        continue

            candidates.append((ticker, score, basic_return, volatility_pct))

        except Exception:
            continue

    if not candidates:
        print(f"   ⚠️ Risk-Adj Mom 3M: No candidates found")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, s, r, v in candidates[:top_n]]

    print(f"   ✅ Risk-Adj Mom 3M: Found {len(candidates)} candidates, selected {len(selected)}")
    for t, s, r, v in candidates[:top_n]:
        print(f"      {t}: score={s:.2f}, return={r:.1f}%, vol={v:.1f}%")

    return selected
