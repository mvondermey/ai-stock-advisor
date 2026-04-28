"""
Ultimate Strategy V5 - Adaptive Multi-Asset with Market Regime Detection

Key insight from backtests:
- Inv ETF Hedge: +289% (best in down market)
- VolSweet Mom: +25% (good stock picker)
- Sector Rotation: +17.5% (solid ETF rotation)
- Holding both longs AND shorts creates drag!

V5 Solution: DYNAMIC allocation based on market regime
- Market UP (SPY 1M > 0%): 70% stocks, 30% sectors, 0% inverse
- Market DOWN (SPY 1M < 0%): 30% stocks, 20% sectors, 50% inverse

This way we:
1. Ride momentum in up markets (like VolSweet Mom)
2. Protect capital in down markets (like Inv ETF Hedge)
3. Never hold contradicting positions
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from strategy_cache_adapter import ensure_price_history_cache

# Multi-timeframe weights
WEIGHT_LONG = 0.20   # 1Y momentum (trend confirmation only)
WEIGHT_MEDIUM = 0.50 # 3M momentum (primary signal)
WEIGHT_SHORT = 0.20  # 1M momentum (timing)
WEIGHT_PULLBACK = 0.10  # Hourly pullback bonus

# Consensus requirement
MIN_POSITIVE_TIMEFRAMES = 2  # At least 2 timeframes must be positive
REQUIRE_POSITIVE_1M = True   # Must have positive 1M momentum

# Volatility parameters - STRICTER sweet spot (15-30% like winning strategies)
VOL_MIN = 10.0       # Minimum volatility (avoid dead stocks)
VOL_MAX = 40.0       # Maximum volatility (hard cutoff - no crypto/meme stocks)
VOL_OPTIMAL = 20.0   # Optimal volatility % (lower for stability)
VOL_PENALTY_SCALE = 0.05  # Stronger penalty per % deviation from optimal

# Filters
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 100000
MIN_HISTORY_DAYS = 252  # Need 1 year of data

# V5: Dynamic allocation based on market conditions
# When market UP: 70% stocks, 30% sectors, 0% inverse
# When market DOWN: 30% stocks, 20% sectors, 50% inverse
MARKET_UP_THRESHOLD = 0.0  # SPY 1M return threshold


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


def _calculate_hourly_rsi(close: pd.Series, period: int = 14) -> float:
    """Calculate RSI on hourly data."""
    if len(close) < period + 1:
        return 50.0  # Neutral

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    if avg_loss.iloc[-1] == 0:
        return 100.0

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _calculate_pullback_score(df: pd.DataFrame, current_date: datetime = None) -> Tuple[float, str]:
    """
    Calculate pullback score using 1H data.

    Returns:
        (score, reason): score from 0 to 1, higher = better pullback entry
        - 1.0 = Strong pullback in uptrend (best entry)
        - 0.5 = Neutral (no clear signal)
        - 0.0 = Extended/overbought (avoid entry)
    """
    if current_date is None:
        current_date = df.index.max()

    # Filter to current date
    data = df[df.index <= current_date]
    if len(data) < 50:
        return 0.5, "insufficient_data"

    close = data['Close'].dropna()
    if len(close) < 50:
        return 0.5, "insufficient_close"

    # Calculate hourly RSI (last 14 bars = ~2 trading days of hourly data)
    rsi = _calculate_hourly_rsi(close, period=14)

    # Calculate short-term momentum (last 24 hours = ~24 bars)
    if len(close) >= 24:
        mom_24h = (close.iloc[-1] / close.iloc[-24] - 1) * 100
    else:
        mom_24h = 0.0

    # Calculate 5-day momentum (last 5 days = ~35 hourly bars)
    if len(close) >= 35:
        mom_5d = (close.iloc[-1] / close.iloc[-35] - 1) * 100
    else:
        mom_5d = 0.0

    # Calculate distance from 20-period MA (hourly)
    if len(close) >= 20:
        ma_20 = close.iloc[-20:].mean()
        dist_from_ma = (close.iloc[-1] / ma_20 - 1) * 100
    else:
        dist_from_ma = 0.0

    # Scoring logic:
    # Best entry: RSI oversold (< 40), short-term dip (mom_24h < 0), but 5d positive
    # Worst entry: RSI overbought (> 70), extended above MA

    score = 0.5  # Start neutral
    reason = "neutral"

    # RSI-based scoring
    if rsi < 30:
        score += 0.25
        reason = "rsi_oversold"
    elif rsi < 40:
        score += 0.15
        reason = "rsi_low"
    elif rsi > 70:
        score -= 0.20
        reason = "rsi_overbought"
    elif rsi > 60:
        score -= 0.10
        reason = "rsi_high"

    # Pullback detection: short-term down, medium-term up
    if mom_24h < -1.0 and mom_5d > 0:
        score += 0.20
        reason = "pullback_in_uptrend"
    elif mom_24h < -2.0 and mom_5d > 2.0:
        score += 0.30
        reason = "strong_pullback_in_uptrend"
    elif mom_24h > 3.0:
        score -= 0.15
        reason = "extended_short_term"

    # Distance from MA scoring
    if -3.0 < dist_from_ma < 0:
        score += 0.10
        reason = "near_ma_support"
    elif dist_from_ma < -5.0:
        score += 0.05  # Deeper pullback, riskier
        reason = "deep_pullback"
    elif dist_from_ma > 5.0:
        score -= 0.10
        reason = "extended_above_ma"

    # Clamp score to [0, 1]
    score = np.clip(score, 0.0, 1.0)

    return score, reason


def _get_market_regime(ticker_data_grouped: Dict[str, pd.DataFrame], current_date: datetime) -> Tuple[str, float]:
    """
    Detect market regime based on SPY momentum.

    Returns:
        (regime, spy_return): 'up' or 'down', and the 1M SPY return
    """
    spy_tickers = ['SPY', 'QQQ']  # Fallback to QQQ if SPY not available

    for ticker in spy_tickers:
        if ticker in ticker_data_grouped:
            try:
                df = ticker_data_grouped[ticker]
                current_ts = pd.Timestamp(current_date)
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(df.index.tz)

                df_filtered = df[df.index <= current_ts]
                if len(df_filtered) >= 20:
                    # Calculate 1-month (21 trading days) return
                    start_date = current_ts - timedelta(days=30)
                    df_period = df_filtered[df_filtered.index >= start_date]
                    if len(df_period) >= 5:
                        spy_return = (df_period['Close'].iloc[-1] / df_period['Close'].iloc[0] - 1) * 100
                        regime = 'up' if spy_return >= MARKET_UP_THRESHOLD else 'down'
                        return regime, spy_return
            except:
                continue

    return 'up', 0.0  # Default to up market if can't determine


def _get_sector_etfs(ticker_data_grouped: Dict[str, pd.DataFrame], current_date: datetime, top_n: int = 3, verbose: bool = False) -> List[str]:
    """Get top performing sector ETFs."""
    from datetime import timedelta

    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB', 'GDX', 'USO', 'TLT']

    etf_performance = []
    for etf in sector_etfs:
        if etf in ticker_data_grouped:
            try:
                df = ticker_data_grouped[etf]
                current_ts = pd.Timestamp(current_date)
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(df.index.tz)

                df_filtered = df[df.index <= current_ts]
                if len(df_filtered) >= 20:
                    start_date = current_ts - timedelta(days=60)
                    df_period = df_filtered[df_filtered.index >= start_date]
                    if len(df_period) >= 10:
                        perf = (df_period['Close'].iloc[-1] / df_period['Close'].iloc[0] - 1) * 100
                        etf_performance.append((etf, perf))
            except:
                continue

    etf_performance.sort(key=lambda x: x[1], reverse=True)
    selected = [etf for etf, _ in etf_performance[:top_n] if etf_performance[0][1] > 0]

    if verbose and selected:
        print(f"   🏢 Sector ETFs: {selected}")

    return selected


def _get_inverse_etfs(ticker_data_grouped: Dict[str, pd.DataFrame], current_date: datetime, top_n: int = 2, verbose: bool = False) -> List[str]:
    """Get inverse ETFs for hedging."""
    from datetime import timedelta

    inverse_etfs = ['SOXS', 'SQQQ', 'SPXU', 'FAZ', 'SH', 'PSQ']

    etf_performance = []
    for etf in inverse_etfs:
        if etf in ticker_data_grouped:
            try:
                df = ticker_data_grouped[etf]
                current_ts = pd.Timestamp(current_date)
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(df.index.tz)

                df_filtered = df[df.index <= current_ts]
                if len(df_filtered) >= 20:
                    start_date = current_ts - timedelta(days=30)
                    df_period = df_filtered[df_filtered.index >= start_date]
                    if len(df_period) >= 5:
                        perf = (df_period['Close'].iloc[-1] / df_period['Close'].iloc[0] - 1) * 100
                        etf_performance.append((etf, perf))
            except:
                continue

    etf_performance.sort(key=lambda x: x[1], reverse=True)
    selected = [etf for etf, _ in etf_performance[:top_n]]

    if verbose and selected:
        print(f"   🛡️ Inverse ETFs: {selected}")

    return selected


def select_ultimate_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    verbose: bool = True,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks using Ultimate Strategy V5.

    ADAPTIVE allocation based on market regime:
    - Market UP: 70% momentum stocks, 30% sector ETFs, 0% inverse
    - Market DOWN: 30% stocks, 20% sectors, 50% inverse ETFs
    """
    from performance_filters import filter_tickers_by_performance

    # Detect market regime
    regime, spy_return = _get_market_regime(ticker_data_grouped, current_date)

    # Dynamic allocation based on regime
    if regime == 'up':
        # Bull market: focus on momentum, no hedges
        alloc_stocks = 0.70
        alloc_sectors = 0.30
        alloc_hedge = 0.00
    else:
        # Bear market: defensive, heavy on hedges
        alloc_stocks = 0.30
        alloc_sectors = 0.20
        alloc_hedge = 0.50

    n_stocks = max(1, int(top_n * alloc_stocks))
    n_sectors = max(1, int(top_n * alloc_sectors))
    n_hedge = top_n - n_stocks - n_sectors

    if verbose:
        regime_emoji = "🟢" if regime == 'up' else "🔴"
        print(f"   {regime_emoji} Ultimate V5: Market {regime.upper()} (SPY 1M: {spy_return:+.1f}%)")
        print(f"   📊 Allocation: {n_stocks} stocks + {n_sectors} sector ETFs + {n_hedge} inverse ETFs")

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    # Pre-filter tickers
    filtered = filter_tickers_by_performance(
        all_tickers, current_date, "Ultimate", price_history_cache=price_history_cache
    )
    if verbose:
        print(f"   📊 Analyzing {len(filtered)} momentum candidates")

    candidates = []

    # Debug: Check if SNDK is in filtered list
    if verbose and 'SNDK' in filtered:
        print(f"   🔍 DEBUG SNDK: IN filtered list")
    elif verbose and 'SNDK' not in filtered:
        print(f"   🔍 DEBUG SNDK: NOT in filtered list (performance filter rejected)")

    for ticker in filtered:
        try:
            df = ticker_data_grouped.get(ticker)
            if df is None or len(df) < MIN_HISTORY_DAYS:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - data={len(df) if df is not None else 0} < {MIN_HISTORY_DAYS}")
                continue

            close = df["Close"].dropna()
            if len(close) < MIN_HISTORY_DAYS:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - close={len(close)} < {MIN_HISTORY_DAYS}")
                continue

            # Basic filters
            current_price = close.iloc[-1]
            if current_price < MIN_PRICE:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - price={current_price:.2f} < {MIN_PRICE}")
                continue

            if 'Volume' in df.columns:
                avg_volume = df['Volume'].iloc[-20:].mean()
                if avg_volume < MIN_AVG_VOLUME:
                    if ticker == 'SNDK' and verbose:
                        print(f"   🔍 DEBUG SNDK: SKIP - volume={avg_volume:.0f} < {MIN_AVG_VOLUME}")
                    continue

            # Calculate multi-timeframe momentum (calendar days)
            mom_1y = _calculate_momentum_calendar(df, current_date, 365)
            mom_3m = _calculate_momentum_calendar(df, current_date, 90)
            mom_1m = _calculate_momentum_calendar(df, current_date, 30)

            # CRITICAL: Require positive 1M momentum (we want current winners)
            if REQUIRE_POSITIVE_1M and mom_1m <= 0:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - 1M momentum={mom_1m:.1f}% <= 0")
                continue

            # Count positive timeframes for consensus
            positive_count = sum([
                1 if mom_1y > 0 else 0,
                1 if mom_3m > 0 else 0,
                1 if mom_1m > 0 else 0,
            ])

            # Skip if no consensus (less than 2 positive timeframes)
            if positive_count < MIN_POSITIVE_TIMEFRAMES:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - consensus={positive_count} < {MIN_POSITIVE_TIMEFRAMES}")
                continue

            # Calculate volatility
            vol_pct = _calculate_volatility(close)
            if vol_pct <= 0:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - volatility={vol_pct:.1f}% <= 0")
                continue

            # HARD volatility cutoffs (learned from VolSweet strategies)
            # Stocks outside 10-40% annual vol are excluded
            if vol_pct < VOL_MIN or vol_pct > VOL_MAX:
                if ticker == 'SNDK' and verbose:
                    print(f"   🔍 DEBUG SNDK: SKIP - volatility={vol_pct:.1f}% outside [{VOL_MIN}-{VOL_MAX}]")
                continue

            # Calculate hourly pullback score (V3 enhancement)
            pullback_score, pullback_reason = _calculate_pullback_score(df, current_date)

            # Debug: SNDK passed all filters
            if ticker == 'SNDK' and verbose:
                print(f"   ✅ DEBUG SNDK: PASSED all filters! 1Y={mom_1y:.1f}%, 3M={mom_3m:.1f}%, 1M={mom_1m:.1f}%, vol={vol_pct:.1f}%, pullback={pullback_score:.2f} ({pullback_reason})")

            # Soft volatility adjustment within the sweet spot
            vol_deviation = abs(vol_pct - VOL_OPTIMAL)
            vol_factor = max(0.6, 1.0 - vol_deviation * VOL_PENALTY_SCALE)

            # Calculate weighted momentum score (V3: includes pullback)
            raw_score = (
                WEIGHT_LONG * mom_1y +
                WEIGHT_MEDIUM * mom_3m +
                WEIGHT_SHORT * mom_1m
            )

            # Add pullback bonus: stocks with good pullback entry get boost
            # pullback_score ranges from 0 to 1, multiply by max momentum for scaling
            max_mom = max(abs(mom_1y), abs(mom_3m), abs(mom_1m), 10.0)
            pullback_bonus = WEIGHT_PULLBACK * pullback_score * max_mom
            raw_score += pullback_bonus

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

            # Pullback timing bonus: extra boost for ideal pullback entries
            if pullback_score > 0.7:
                score *= 1.15  # 15% bonus for strong pullback signals
            elif pullback_score > 0.6:
                score *= 1.08  # 8% bonus for moderate pullback

            candidates.append({
                'ticker': ticker,
                'score': score,
                'mom_1y': mom_1y,
                'mom_3m': mom_3m,
                'mom_1m': mom_1m,
                'vol': vol_pct,
                'consensus': positive_count,
                'pullback': pullback_score,
                'pullback_reason': pullback_reason,
            })

        except Exception as e:
            continue

    # Sort by score and select top momentum stocks
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected_stocks = candidates[:n_stocks]

    # Get sector ETFs
    sector_etfs = _get_sector_etfs(ticker_data_grouped, current_date, n_sectors, verbose)

    # Get inverse ETFs ONLY in down market (V5 improvement)
    if regime == 'down' and n_hedge > 0:
        inverse_etfs = _get_inverse_etfs(ticker_data_grouped, current_date, n_hedge, verbose)
    else:
        inverse_etfs = []
        # In up market, use extra slots for more stocks
        extra_stocks = candidates[n_stocks:n_stocks + n_hedge]
        selected_stocks.extend(extra_stocks)

    # Combine all selections
    final_selection = [c['ticker'] for c in selected_stocks] + sector_etfs + inverse_etfs

    # Print selected stocks
    if verbose:
        print(f"   🎯 Ultimate V5: Selected {len(final_selection)} total positions:")
        if selected_stocks:
            print(f"      📈 Momentum stocks ({len(selected_stocks)}):")
            for i, c in enumerate(selected_stocks[:5], 1):
                print(f"         {i}. {c['ticker']}: score={c['score']:.2f}, "
                      f"1Y={c['mom_1y']:+.1f}%, 3M={c['mom_3m']:+.1f}%, 1M={c['mom_1m']:+.1f}%")
        if sector_etfs:
            print(f"      🏢 Sector ETFs ({len(sector_etfs)}): {sector_etfs}")
        if inverse_etfs:
            print(f"      🛡️ Inverse ETFs ({len(inverse_etfs)}): {inverse_etfs}")

    return final_selection


def get_strategy_description() -> str:
    """Return strategy description for logging."""
    return (
        f"Ultimate V5: Adaptive multi-asset with market regime detection. "
        f"UP market: 70% stocks + 30% sectors. DOWN market: 30% stocks + 20% sectors + 50% inverse ETFs. "
        f"Includes 1H pullback timing for better entries."
    )
