"""
Risk-Adjusted Momentum 6M Strategy
Identical to Risk-Adj Mom but uses 6-month (180-day) performance window instead of 1-year.
score = return_6m / sqrt(volatility)
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime


def select_risk_adj_mom_6m_stocks(
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
        all_tickers, ticker_data_grouped, current_date, "Risk-Adj Mom 6M"
    )

    PERF_WINDOW = 180  # 6 months instead of 365

    candidates = []
    print(f"   📊 Risk-Adj Mom 6M: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            data = ticker_data_grouped[ticker]
            if data is None or len(data) == 0:
                continue

            close = data['Close'].dropna()
            n = len(close)
            if n < 60:
                continue

            latest_price = close.iloc[-1]
            if latest_price <= 0:
                continue

            # 6M performance window
            perf_window = min(PERF_WINDOW, n - 1)
            if perf_window < 60:
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

            # Momentum confirmation (same as Risk-Adj Mom)
            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                confirmations = 0
                for days in [90, 180]:
                    lookback = min(days, n - 1)
                    p = close.iloc[-lookback]
                    if p > 0 and (latest_price - p) / p > 0:
                        confirmations += 1
                if confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    continue

            # Volume confirmation (same as Risk-Adj Mom)
            if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                vol_series = data['Volume'].dropna()
                if len(vol_series) >= RISK_ADJ_MOM_VOLUME_WINDOW + 10:
                    recent_vol = vol_series.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
                    avg_vol = vol_series.iloc[:-RISK_ADJ_MOM_VOLUME_WINDOW].mean()
                    if avg_vol > 0 and recent_vol < avg_vol * RISK_ADJ_MOM_VOLUME_MULTIPLIER:
                        continue

            candidates.append((ticker, score, basic_return, volatility_pct))

        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    if not candidates:
        print(f"   ⚠️ Risk-Adj Mom 6M: No candidates found")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, s, r, v in candidates[:top_n]]

    print(f"   ✅ Risk-Adj Mom 6M: Found {len(candidates)} candidates, selected {len(selected)}")
    for t, s, r, v in candidates[:top_n]:
        print(f"      {t}: score={s:.2f}, return={r:.1f}%, vol={v:.1f}%")

    return selected
