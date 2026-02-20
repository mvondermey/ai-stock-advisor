"""
Elite Risk Strategy: Combines Risk-Adj Mom + Elite Hybrid

Best of both worlds:
- Risk-Adj Mom: rigorous return/vol^0.5 scoring, multi-timeframe momentum confirmation
- Elite Hybrid: dip detection (strong 1Y, weak 3M), volatility sweet-spot, volume quality

Scoring formula:
  base_score   = 1Y_return / (daily_vol^0.5)        [Risk-Adj Mom core]
  dip_bonus    = 1.0 - 1.5x  based on 1Y/3M spread  [Elite Hybrid innovation]
  vol_bonus    = 0.9 - 1.1x  for 20-40% vol range   [Elite Hybrid innovation]
  vol_bonus_h  = 1.0 - 1.15x based on avg volume     [Elite Hybrid innovation]
  elite_score  = base_score * dip_bonus * vol_bonus * vol_bonus_h
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone


def select_elite_risk_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10
) -> List[str]:
    """
    Elite Risk Strategy: Risk-Adj Mom base + Elite Hybrid dip/vol bonuses.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                        for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []

    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)

    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Elite Risk"
    )

    from config import (
        RISK_ADJ_MOM_PERFORMANCE_WINDOW,
        RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION,
        RISK_ADJ_MOM_MIN_CONFIRMATIONS,
        RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION,
        RISK_ADJ_MOM_VOLUME_WINDOW,
        RISK_ADJ_MOM_VOLUME_MULTIPLIER,
    )

    candidates = []
    print(f"   🏆 Elite Risk: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) == 0:
                continue

            # --- Normalise current_date to data timezone ---
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)
            elif current_ts.tz is not None:
                current_ts = current_ts.replace(tzinfo=None)

            data = ticker_data[ticker_data.index <= current_ts]
            close = data['Close'].dropna()
            n = len(close)

            if n < 60:
                continue

            latest_price = close.iloc[-1]
            if latest_price <= 0:
                continue

            # ---------------------------------------------------------------- #
            # PART 1: Risk-Adj Mom base score  (return / vol^0.5)              #
            # ---------------------------------------------------------------- #
            perf_window = min(RISK_ADJ_MOM_PERFORMANCE_WINDOW, n - 1)
            if perf_window < 60:
                continue

            start_price = close.iloc[-perf_window]
            if start_price <= 0:
                continue

            basic_return = (latest_price - start_price) / start_price * 100

            daily_returns = close.pct_change().dropna()
            if len(daily_returns) < 20:
                continue

            volatility_pct = daily_returns.std() * 100  # daily vol %
            if volatility_pct <= 0:
                continue

            base_score = basic_return / (volatility_pct ** 0.5)

            if base_score <= 0:
                continue

            # ---------------------------------------------------------------- #
            # PART 2: Multi-timeframe momentum confirmation (Risk-Adj Mom)      #
            # ---------------------------------------------------------------- #
            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                confirmations = 0
                for days in [90, 180, 365]:
                    lookback = min(days, n - 1)
                    p = close.iloc[-lookback]
                    if p > 0 and (latest_price - p) / p > 0:
                        confirmations += 1
                if confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    continue

            # ---------------------------------------------------------------- #
            # PART 3: Volume confirmation (Risk-Adj Mom)                        #
            # ---------------------------------------------------------------- #
            if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                vol_series = data['Volume'].dropna()
                if len(vol_series) >= RISK_ADJ_MOM_VOLUME_WINDOW + 10:
                    recent_vol = vol_series.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
                    avg_vol = vol_series.iloc[:-RISK_ADJ_MOM_VOLUME_WINDOW].mean()
                    if avg_vol > 0 and recent_vol < avg_vol * RISK_ADJ_MOM_VOLUME_MULTIPLIER:
                        continue

            # ---------------------------------------------------------------- #
            # PART 4: Elite Hybrid bonuses                                      #
            # ---------------------------------------------------------------- #
            volatility_ann = daily_returns.std() * (252 ** 0.5)  # annualised

            # 3M and 1Y performance
            lookback_3m = min(63, n - 1)
            lookback_1y = min(252, n - 1)
            perf_3m = (latest_price / close.iloc[-lookback_3m] - 1) * 100 if lookback_3m >= 10 else 0.0
            perf_1y = (latest_price / close.iloc[-lookback_1y] - 1) * 100 if lookback_1y >= 60 else 0.0

            # Dip bonus: strong 1Y but weak 3M = buy-the-dip opportunity
            if perf_1y > 30 and perf_3m < 5:
                dip_bonus = 1.5
            elif perf_1y > 20 and perf_3m < 10 and perf_3m >= 0:
                dip_bonus = 1.3
            elif perf_1y > 10 and perf_3m < 0:
                dip_bonus = 1.2
            else:
                # Mild dip ratio bonus
                if perf_3m > 0 and perf_1y > 0:
                    dip_ratio = max(min((perf_1y - perf_3m) / perf_3m, 3.0), 0.0)
                else:
                    dip_ratio = 0.0
                dip_bonus = 1.0 + dip_ratio * 0.05

            # Volatility sweet-spot bonus (20-40% annualised)
            if 0.20 <= volatility_ann <= 0.40:
                vol_bonus = 1.10
            elif volatility_ann < 0.15:
                vol_bonus = 0.95
            elif volatility_ann > 0.60:
                vol_bonus = 0.90
            else:
                vol_bonus = 1.0

            # Volume quality bonus
            avg_volume = data['Volume'].tail(30).mean() if 'Volume' in data.columns else 0
            if avg_volume > 5_000_000:
                vol_h_bonus = 1.15
            elif avg_volume > 2_000_000:
                vol_h_bonus = 1.10
            elif avg_volume > 1_000_000:
                vol_h_bonus = 1.05
            else:
                vol_h_bonus = 1.0

            elite_score = base_score * dip_bonus * vol_bonus * vol_h_bonus

            candidates.append({
                'ticker':       ticker,
                'elite_score':  elite_score,
                'base_score':   base_score,
                'return_pct':   basic_return,
                'volatility':   volatility_ann * 100,
                'perf_3m':      perf_3m,
                'perf_1y':      perf_1y,
                'dip_bonus':    dip_bonus,
            })

        except Exception:
            continue

    if not candidates:
        print(f"   ⚠️ Elite Risk: No candidates found")
        return []

    candidates.sort(key=lambda x: x['elite_score'], reverse=True)

    print(f"   ✅ Elite Risk: Found {len(candidates)} candidates")
    for i, c in enumerate(candidates[:5], 1):
        print(f"      {i}. {c['ticker']}: Score={c['elite_score']:.2f} (base={c['base_score']:.2f}, "
              f"dip={c['dip_bonus']:.2f}x), 1Y={c['perf_1y']:+.1f}%, 3M={c['perf_3m']:+.1f}%, "
              f"Vol={c['volatility']:.1f}%")

    selected = [c['ticker'] for c in candidates[:top_n]]
    print(f"   🎯 Elite Risk selected: {selected}")
    return selected
