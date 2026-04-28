"""
Deep Recovery Strategy: Contrarian strategy that identifies beaten-down stocks
with early signs of reversal.

Unlike momentum strategies, this strategy searches the FULL ticker universe
after the shared prefilter and then looks for stocks in deep drawdowns that
show technical reversal signals.

Signals used:
  1. Drawdown depth from 52-week high (deeper = more bounce potential)
  2. Short-term price reversal (last 5-10 days turning positive)
  3. RSI oversold (< 30 or recovering from < 30)
  4. Volume surge relative to average (capitulation/accumulation)
  5. Price near or bouncing off Bollinger Band lower band
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_history_up_to,
    get_cached_values_between,
    resolve_cache_current_date,
)
from performance_filters import filter_tickers_by_performance


# ---------------------------------------------------------------------------
# Config constants (defaults; overridden from config.py when available)
# ---------------------------------------------------------------------------
DEEP_RECOVERY_MIN_DRAWDOWN = 25.0       # % drawdown from 52-week high to qualify
DEEP_RECOVERY_MAX_DRAWDOWN = 80.0       # % max drawdown (avoid near-zero penny stocks)
DEEP_RECOVERY_MIN_5D_RETURN = 0.0       # % minimum 5-day return (early reversal signal)
DEEP_RECOVERY_RSI_THRESHOLD = 40.0      # RSI must be below this (oversold zone)
DEEP_RECOVERY_VOLUME_SURGE_RATIO = 1.3  # Recent volume / avg volume ratio
DEEP_RECOVERY_MIN_PRICE = 2.0           # Minimum stock price to avoid penny stocks
DEEP_RECOVERY_MIN_DATA_DAYS = 253       # Minimum trading days of history required
DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS = 200  # Minimum populated rows inside trailing-year window


def _load_config_overrides() -> None:
    """Load config overrides if they exist."""
    global DEEP_RECOVERY_MIN_DRAWDOWN, DEEP_RECOVERY_MAX_DRAWDOWN
    global DEEP_RECOVERY_MIN_5D_RETURN, DEEP_RECOVERY_RSI_THRESHOLD
    global DEEP_RECOVERY_VOLUME_SURGE_RATIO, DEEP_RECOVERY_MIN_PRICE
    global DEEP_RECOVERY_MIN_DATA_DAYS, DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS
    try:
        import config
        DEEP_RECOVERY_MIN_DRAWDOWN = getattr(config, 'DEEP_RECOVERY_MIN_DRAWDOWN', DEEP_RECOVERY_MIN_DRAWDOWN)
        DEEP_RECOVERY_MAX_DRAWDOWN = getattr(config, 'DEEP_RECOVERY_MAX_DRAWDOWN', DEEP_RECOVERY_MAX_DRAWDOWN)
        DEEP_RECOVERY_MIN_5D_RETURN = getattr(config, 'DEEP_RECOVERY_MIN_5D_RETURN', DEEP_RECOVERY_MIN_5D_RETURN)
        DEEP_RECOVERY_RSI_THRESHOLD = getattr(config, 'DEEP_RECOVERY_RSI_THRESHOLD', DEEP_RECOVERY_RSI_THRESHOLD)
        DEEP_RECOVERY_VOLUME_SURGE_RATIO = getattr(config, 'DEEP_RECOVERY_VOLUME_SURGE_RATIO', DEEP_RECOVERY_VOLUME_SURGE_RATIO)
        DEEP_RECOVERY_MIN_PRICE = getattr(config, 'DEEP_RECOVERY_MIN_PRICE', DEEP_RECOVERY_MIN_PRICE)
        DEEP_RECOVERY_MIN_DATA_DAYS = getattr(config, 'DEEP_RECOVERY_MIN_DATA_DAYS', DEEP_RECOVERY_MIN_DATA_DAYS)
        DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS = getattr(
            config,
            'DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS',
            DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS,
        )
    except ImportError:
        pass


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for the last value of a price series."""
    if len(prices) < period + 1:
        return 50.0  # neutral if insufficient data
    deltas = prices.diff().dropna()
    if len(deltas) < period:
        return 50.0
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)
    avg_gain = gains.iloc[-period:].mean()
    avg_loss = losses.iloc[-period:].mean()
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def select_deep_recovery_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks in deep drawdowns showing early reversal signals.

    This strategy uses the FULL ticker_data_grouped universe after the shared
    prefilter to find beaten-down stocks that momentum strategies miss, while
    reading the underlying history from the shared price history cache.

    Args:
        all_tickers: Pre-filtered ticker list (ignored - uses ticker_data_grouped.keys())
        ticker_data_grouped: Dict of ticker -> DataFrame (full universe)
        current_date: Current analysis date
        top_n: Number of stocks to select
        price_history_cache: Optional cache used for fast daily-history access

    Returns:
        List of selected ticker symbols, sorted by recovery score descending
    """
    _load_config_overrides()

    from config import INVERSE_ETFS

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    # Use the FULL universe from ticker_data_grouped, not all_tickers.
    full_universe = sorted(ticker_data_grouped.keys())
    # Exclude inverse ETFs
    full_universe = [t for t in full_universe if t not in INVERSE_ETFS]
    full_universe = filter_tickers_by_performance(
        full_universe,
        current_date,
        "Deep Recovery",
        price_history_cache=price_history_cache,
    )

    current_date = resolve_cache_current_date(price_history_cache, current_date, full_universe)
    if current_date is None:
        return []

    print(f"   🔄 Deep Recovery: Scanning {len(full_universe)} tickers (full universe)")

    candidates = []
    skipped_no_data = 0
    skipped_insufficient = 0
    skipped_drawdown_low = 0
    skipped_drawdown_high = 0
    skipped_price_low = 0
    skipped_no_reversal = 0
    debug_count = 0

    for ticker in full_universe:
        try:
            close_history = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="close",
                min_rows=DEEP_RECOVERY_MIN_DATA_DAYS,
            )
            if close_history is None:
                skipped_insufficient += 1
                continue

            closes = np.asarray(close_history, dtype=float)
            current_price = float(closes[-1])
            if np.isnan(current_price) or current_price <= 0:
                skipped_no_data += 1
                continue

            # Minimum price filter
            if current_price < DEEP_RECOVERY_MIN_PRICE:
                skipped_price_low += 1
                continue

            # --- Signal 1: Drawdown from 52-week high ---
            year_data = get_cached_values_between(
                price_history_cache,
                ticker,
                current_date - timedelta(days=365),
                current_date,
                field_name="close",
                min_rows=DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS,
            )
            if year_data is None:
                skipped_insufficient += 1
                continue

            high_52w = float(np.max(year_data))
            if high_52w <= 0 or np.isnan(high_52w):
                skipped_no_data += 1
                continue

            drawdown_pct = ((high_52w - current_price) / high_52w) * 100.0

            if drawdown_pct < DEEP_RECOVERY_MIN_DRAWDOWN:
                skipped_drawdown_low += 1
                continue
            if drawdown_pct > DEEP_RECOVERY_MAX_DRAWDOWN:
                skipped_drawdown_high += 1
                continue

            # --- Signal 2: Short-term reversal (5-day return) ---
            if len(closes) >= 6:
                price_5d_ago = float(closes[-6])
                if price_5d_ago > 0 and not np.isnan(price_5d_ago):
                    return_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100.0
                else:
                    return_5d = 0.0
            else:
                return_5d = 0.0

            # --- Signal 3: 10-day return (slightly longer reversal window) ---
            if len(closes) >= 11:
                price_10d_ago = float(closes[-11])
                if price_10d_ago > 0 and not np.isnan(price_10d_ago):
                    return_10d = ((current_price - price_10d_ago) / price_10d_ago) * 100.0
                else:
                    return_10d = 0.0
            else:
                return_10d = 0.0

            # --- Signal 4: RSI ---
            rsi = _compute_rsi(pd.Series(closes), period=14)

            # --- Signal 5: Volume surge ---
            volume_surge = 1.0
            volume_history = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="volume",
                min_rows=1,
            )
            if volume_history is not None and len(volume_history) >= 20:
                volumes = np.asarray(volume_history, dtype=float)
                recent_vol = float(np.mean(volumes[-5:]))
                avg_vol = float(np.mean(volumes[-60:])) if len(volumes) >= 60 else float(np.mean(volumes))
                if avg_vol > 0:
                    volume_surge = recent_vol / avg_vol

            # --- Signal 6: Bollinger Band position ---
            bb_score = 0.0
            if len(closes) >= 20:
                sma_20 = float(np.mean(closes[-20:]))
                std_20 = float(np.std(closes[-20:], ddof=1))
                if std_20 > 0:
                    lower_band = sma_20 - 2 * std_20
                    # Score: how close to lower band (0 = at SMA, 1 = at lower band, >1 = below)
                    if sma_20 > lower_band:
                        bb_score = max(0.0, (sma_20 - current_price) / (sma_20 - lower_band))

            # --- Filtering: require at least some reversal signal ---
            has_reversal = (
                return_5d > DEEP_RECOVERY_MIN_5D_RETURN
                or rsi < DEEP_RECOVERY_RSI_THRESHOLD
                or volume_surge >= DEEP_RECOVERY_VOLUME_SURGE_RATIO
            )

            if not has_reversal:
                skipped_no_reversal += 1
                continue

            # --- Composite recovery score ---
            # Higher drawdown = more potential upside (capped contribution)
            drawdown_score = min(drawdown_pct / 100.0, 0.7)  # 0.0-0.7

            # Short-term reversal bonus
            reversal_score = 0.0
            if return_5d > 0:
                reversal_score += min(return_5d / 20.0, 0.3)  # 0.0-0.3
            if return_10d > 0:
                reversal_score += min(return_10d / 30.0, 0.2)  # 0.0-0.2

            # RSI: lower = more oversold = higher score
            rsi_score = max(0.0, (DEEP_RECOVERY_RSI_THRESHOLD - rsi) / DEEP_RECOVERY_RSI_THRESHOLD) * 0.2  # 0.0-0.2

            # Volume surge bonus
            vol_score = min((volume_surge - 1.0) / 2.0, 0.15) if volume_surge > 1.0 else 0.0  # 0.0-0.15

            # Bollinger band proximity bonus
            bb_bonus = min(bb_score * 0.15, 0.15)  # 0.0-0.15

            total_score = drawdown_score + reversal_score + rsi_score + vol_score + bb_bonus

            candidates.append((
                ticker,
                total_score,
                drawdown_pct,
                return_5d,
                return_10d,
                rsi,
                volume_surge,
                bb_score,
            ))

            if debug_count < 5:
                print(
                    f"   🔍 DEBUG {ticker}: drawdown={drawdown_pct:.1f}%, "
                    f"5d={return_5d:+.1f}%, 10d={return_10d:+.1f}%, "
                    f"RSI={rsi:.1f}, vol_surge={volume_surge:.2f}x, "
                    f"BB={bb_score:.2f}, score={total_score:.3f}"
                )
                debug_count += 1

        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(
        f"   📊 Deep Recovery: {len(candidates)} candidates from {len(full_universe)} universe | "
        f"Skipped: no_data={skipped_no_data}, insufficient={skipped_insufficient}, "
        f"drawdown_low={skipped_drawdown_low}, drawdown_high={skipped_drawdown_high}, "
        f"price_low={skipped_price_low}, no_reversal={skipped_no_reversal}"
    )

    if candidates:
        selected = candidates[:top_n]
        print(f"   🎯 Deep Recovery selected {len(selected)} stocks:")
        for ticker, score, dd, r5, r10, rsi_val, vs, bb in selected:
            print(
                f"      {ticker}: score={score:.3f}, drawdown={dd:.1f}%, "
                f"5d={r5:+.1f}%, 10d={r10:+.1f}%, RSI={rsi_val:.1f}, "
                f"vol={vs:.2f}x, BB={bb:.2f}"
            )
        return [t for t, *_ in selected]

    print(f"   ❌ No Deep Recovery candidates found")
    return []
