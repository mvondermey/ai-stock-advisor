"""
Ultimate Strategy V2 - Multi-Timeframe Hybrid for Maximum Performance

Analysis of Multi-TF Ensemble success (+62.6%):
- Multi-timeframe consensus (2+ timeframes must agree)
- Long-term bias (1Y momentum weighted 60%)
- Medium-term confirmation (30-day momentum, 30% weight)
- Short-term timing (7-day momentum, 10% weight)
- No strict volatility cutoffs - allows higher-vol winners
- Volume confirmation on short-term signals

Key changes from V1:
1. Removed strict volatility sweet spot - use soft penalty instead
2. Added multi-timeframe scoring like Multi-TF Ensemble
3. Increased long-term (1Y) momentum weight
4. Added consensus requirement (2+ timeframes positive)
5. Simplified scoring - fewer factors, stronger signals
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Multi-timeframe weights (balanced for current winners)
WEIGHT_LONG = 0.30   # 1Y momentum (long-term trend) - reduced
WEIGHT_MEDIUM = 0.35 # 3M momentum (medium-term confirmation)
WEIGHT_SHORT = 0.35  # 1M momentum (short-term timing) - increased

# Consensus requirement
MIN_POSITIVE_TIMEFRAMES = 2  # At least 2 timeframes must be positive
REQUIRE_POSITIVE_1M = True   # Must have positive 1M momentum

# Volatility parameters - soft penalty, not hard cutoff
VOL_OPTIMAL = 25.0   # Optimal volatility %
VOL_PENALTY_SCALE = 0.02  # Penalty per % deviation from optimal

# Filters
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 100000
MIN_HISTORY_DAYS = 252  # Need 1 year of data


def _calculate_momentum_calendar(df: pd.DataFrame, current_date: datetime, days: int) -> float:
    """Calculate momentum over calendar days."""
    if current_date is None:
        current_date = df.index.max()

    # Filter to current date
    data = df[df.index <= current_date]
    if len(data) < 10:
        return 0.0

    close = data['Close'].dropna()
    if len(close) < 10:
        return 0.0

    start_date = current_date - timedelta(days=days)
    start_data = close[close.index >= start_date]

    if len(start_data) < 5:
        return 0.0

    return (close.iloc[-1] / start_data.iloc[0] - 1) * 100


def _calculate_volatility(close: pd.Series) -> float:
    """Calculate annualized volatility percentage."""
    daily_returns = close.pct_change().dropna()
    if len(daily_returns) < 20:
        return 0.0
    return daily_returns.std() * np.sqrt(252) * 100


def _calculate_volume_factor(df: pd.DataFrame) -> float:
    """Calculate volume confirmation factor (0.5 to 1.5)."""
    if 'Volume' not in df.columns or len(df) < 30:
        return 1.0

    volume = df['Volume'].dropna()
    if len(volume) < 30:
        return 1.0

    recent_vol = volume.iloc[-5:].mean()
    avg_vol = volume.iloc[-30:].mean()

    if avg_vol <= 0:
        return 1.0

    ratio = recent_vol / avg_vol
    return np.clip(ratio, 0.5, 1.5)


def _calculate_trend_bonus(close: pd.Series) -> float:
    """Calculate trend bonus based on MA alignment (0 to 0.2)."""
    if len(close) < 50:
        return 0.0

    current = close.iloc[-1]
    ma_10 = close.iloc[-10:].mean()
    ma_20 = close.iloc[-20:].mean()
    ma_50 = close.iloc[-50:].mean()

    bonus = 0.0

    # Price above all MAs = strong trend
    if current > ma_10 > ma_20 > ma_50:
        bonus = 0.20
    elif current > ma_20 > ma_50:
        bonus = 0.10
    elif current > ma_50:
        bonus = 0.05

    return bonus


def select_ultimate_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
) -> List[str]:
    """
    Select stocks using Ultimate Strategy V2.

    Multi-timeframe momentum with consensus requirement.
    """
    from performance_filters import filter_tickers_by_performance

    # Pre-filter tickers
    filtered = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Ultimate"
    )
    print(f"   📊 Ultimate V2: analyzing {len(filtered)} tickers")

    candidates = []

    for ticker in filtered:
        try:
            df = ticker_data_grouped.get(ticker)
            if df is None or len(df) < MIN_HISTORY_DAYS:
                continue

            close = df["Close"].dropna()
            if len(close) < MIN_HISTORY_DAYS:
                continue

            # Basic filters
            current_price = close.iloc[-1]
            if current_price < MIN_PRICE:
                continue

            if 'Volume' in df.columns:
                avg_volume = df['Volume'].iloc[-20:].mean()
                if avg_volume < MIN_AVG_VOLUME:
                    continue

            # Calculate multi-timeframe momentum (calendar days)
            mom_1y = _calculate_momentum_calendar(df, current_date, 365)
            mom_3m = _calculate_momentum_calendar(df, current_date, 90)
            mom_1m = _calculate_momentum_calendar(df, current_date, 30)

            # CRITICAL: Require positive 1M momentum (we want current winners)
            if REQUIRE_POSITIVE_1M and mom_1m <= 0:
                continue

            # Count positive timeframes for consensus
            positive_count = sum([
                1 if mom_1y > 0 else 0,
                1 if mom_3m > 0 else 0,
                1 if mom_1m > 0 else 0,
            ])

            # Skip if no consensus (less than 2 positive timeframes)
            if positive_count < MIN_POSITIVE_TIMEFRAMES:
                continue

            # Calculate volatility
            vol_pct = _calculate_volatility(close)
            if vol_pct <= 0:
                continue

            # Soft volatility adjustment (prefer ~25% vol, but don't exclude)
            vol_deviation = abs(vol_pct - VOL_OPTIMAL)
            vol_factor = max(0.5, 1.0 - vol_deviation * VOL_PENALTY_SCALE)

            # Calculate weighted momentum score
            raw_score = (
                WEIGHT_LONG * mom_1y +
                WEIGHT_MEDIUM * mom_3m +
                WEIGHT_SHORT * mom_1m
            )

            # Risk-adjust by volatility (Sharpe-like)
            risk_adj_score = raw_score / np.sqrt(max(vol_pct, 10.0))

            # Apply volatility factor
            score = risk_adj_score * vol_factor

            # Volume confirmation (boost if recent volume is high)
            volume_factor = _calculate_volume_factor(df)
            score *= volume_factor

            # Trend bonus (up to +20% for perfect MA alignment)
            trend_bonus = _calculate_trend_bonus(close)
            score *= (1 + trend_bonus)

            # Acceleration bonus: if 1M > 3M/3, momentum is accelerating
            if mom_1m > 0 and mom_3m > 0 and mom_1m > mom_3m / 3:
                score *= 1.10  # 10% bonus

            candidates.append({
                'ticker': ticker,
                'score': score,
                'mom_1y': mom_1y,
                'mom_3m': mom_3m,
                'mom_1m': mom_1m,
                'vol': vol_pct,
                'consensus': positive_count,
            })

        except Exception as e:
            continue

    if not candidates:
        print("   ⚠️ Ultimate V2: no candidates found")
        return []

    # Sort by score and select top N
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected = candidates[:top_n]

    # Print selected stocks
    print(f"   🎯 Ultimate V2: Selected {len(selected)} stocks:")
    for i, c in enumerate(selected[:5], 1):
        print(f"      {i}. {c['ticker']}: score={c['score']:.2f}, "
              f"1Y={c['mom_1y']:+.1f}%, 3M={c['mom_3m']:+.1f}%, 1M={c['mom_1m']:+.1f}%, "
              f"vol={c['vol']:.1f}%")

    return [c['ticker'] for c in selected]


def get_strategy_description() -> str:
    """Return strategy description for logging."""
    return (
        f"Ultimate V2: Multi-timeframe momentum (1Y:{WEIGHT_LONG:.0%}, 3M:{WEIGHT_MEDIUM:.0%}, "
        f"1M:{WEIGHT_SHORT:.0%}) with {MIN_POSITIVE_TIMEFRAMES}+ timeframe consensus, "
        f"soft volatility adjustment, volume confirmation, and trend bonus."
    )
