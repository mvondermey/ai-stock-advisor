from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_current_date(current_date: Optional[datetime]) -> Optional[pd.Timestamp]:
    if current_date is None:
        return None
    current_ts = pd.Timestamp(current_date)
    if current_ts.tzinfo is None:
        return current_ts.tz_localize("UTC")
    return current_ts.tz_convert("UTC")


def _get_market_risk_state(
    price_history_cache,
    current_date: pd.Timestamp,
) -> Tuple[bool, str]:
    from config import (
        DEFENSIVE_MOMENTUM_MIN_MARKET_20D_RETURN,
        DEFENSIVE_MOMENTUM_SMA_DAYS,
    )
    from strategy_cache_adapter import get_cached_history_up_to

    market_lines: List[str] = []
    for ticker in ("SPY", "QQQ"):
        closes = get_cached_history_up_to(
            price_history_cache,
            ticker,
            current_date,
            field_name="close",
            min_rows=DEFENSIVE_MOMENTUM_SMA_DAYS,
        )
        if closes is None or len(closes) < DEFENSIVE_MOMENTUM_SMA_DAYS:
            return False, f"{ticker}: insufficient market history"

        latest = float(closes[-1])
        sma_200 = float(np.mean(closes[-DEFENSIVE_MOMENTUM_SMA_DAYS:]))
        if len(closes) >= 20:
            return_20d = ((float(closes[-1]) - float(closes[-20])) / float(closes[-20])) * 100
        else:
            return_20d = 0.0

        market_lines.append(
            f"{ticker}={latest:.2f} vs SMA{DEFENSIVE_MOMENTUM_SMA_DAYS}={sma_200:.2f}, 20D={return_20d:+.1f}%"
        )
        if latest <= sma_200 or return_20d <= DEFENSIVE_MOMENTUM_MIN_MARKET_20D_RETURN:
            return False, " | ".join(market_lines)

    return True, " | ".join(market_lines)


def select_defensive_momentum_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """
    Defensive Momentum Strategy:
    - Stay in cash unless broad market trend is positive.
    - Only buy stocks above their 200-day SMA.
    - Prefer strong 1Y and 3M momentum with capped volatility.
    """
    from config import (
        DEFENSIVE_MOMENTUM_MAX_VOLATILITY,
        DEFENSIVE_MOMENTUM_MIN_1Y_RETURN,
        DEFENSIVE_MOMENTUM_MIN_3M_RETURN,
        DEFENSIVE_MOMENTUM_SMA_DAYS,
        INVERSE_ETFS,
        MIN_DATA_DAYS_1Y,
        MIN_DATA_DAYS_3M,
        PORTFOLIO_SIZE,
    )
    from performance_filters import filter_tickers_by_performance
    from strategy_cache_adapter import (
        ensure_price_history_cache,
        get_cached_history_up_to,
        get_cached_window,
        resolve_cache_current_date,
    )

    if top_n is None:
        top_n = PORTFOLIO_SIZE

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    current_date = resolve_cache_current_date(price_history_cache, current_date, all_tickers)
    if current_date is None:
        return []

    risk_on, market_summary = _get_market_risk_state(price_history_cache, current_date)
    if not risk_on:
        print(f"   🛡️ Defensive Momentum: RISK-OFF, holding cash ({market_summary})")
        return []

    tickers_to_use = [ticker for ticker in all_tickers if ticker not in INVERSE_ETFS and ticker not in {"SPY", "QQQ"}]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "Defensive Momentum",
        price_history_cache=price_history_cache,
    )

    candidates = []
    data_insufficient = 0
    volatility_filtered = 0
    sma_filtered = 0
    momentum_filtered = 0

    for ticker in filtered_tickers:
        try:
            close_history = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="close",
                min_rows=DEFENSIVE_MOMENTUM_SMA_DAYS,
            )
            if close_history is None or len(close_history) < DEFENSIVE_MOMENTUM_SMA_DAYS:
                data_insufficient += 1
                continue

            close_1y = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                365,
                field_name="close",
                min_rows=MIN_DATA_DAYS_1Y,
            )
            if close_1y is None or len(close_1y) < max(MIN_DATA_DAYS_1Y, 60):
                data_insufficient += 1
                continue

            close_3m = get_cached_window(
                price_history_cache,
                ticker,
                current_date,
                90,
                field_name="close",
                min_rows=MIN_DATA_DAYS_3M,
            )
            if close_3m is None or len(close_3m) < max(MIN_DATA_DAYS_3M, 20):
                data_insufficient += 1
                continue

            latest = float(close_history[-1])
            sma_200 = float(np.mean(close_history[-DEFENSIVE_MOMENTUM_SMA_DAYS:]))
            if latest <= sma_200:
                sma_filtered += 1
                continue

            perf_1y = ((float(close_1y[-1]) - float(close_1y[0])) / float(close_1y[0])) * 100
            perf_3m = ((float(close_3m[-1]) - float(close_3m[0])) / float(close_3m[0])) * 100

            if perf_1y < DEFENSIVE_MOMENTUM_MIN_1Y_RETURN or perf_3m < DEFENSIVE_MOMENTUM_MIN_3M_RETURN:
                momentum_filtered += 1
                continue

            returns = np.diff(close_1y) / close_1y[:-1]
            if returns.size < 20:
                data_insufficient += 1
                continue

            volatility = float(np.std(returns[-63:], ddof=1) * np.sqrt(252))
            if volatility > DEFENSIVE_MOMENTUM_MAX_VOLATILITY:
                volatility_filtered += 1
                continue

            sma_gap_pct = ((latest - sma_200) / sma_200) * 100
            score = (0.45 * perf_1y) + (0.35 * perf_3m) + (0.20 * sma_gap_pct) - (15.0 * volatility)

            if len(candidates) < 5:
                print(
                    f"   🔍 Defensive Momentum {ticker}: 1Y={perf_1y:+.1f}%, 3M={perf_3m:+.1f}%, "
                    f"SMA-gap={sma_gap_pct:+.1f}%, vol={volatility*100:.1f}%"
                )

            candidates.append((ticker, score, perf_1y, perf_3m, sma_gap_pct, volatility))
        except Exception:
            data_insufficient += 1
            continue

    if not candidates:
        print(
            f"   ❌ Defensive Momentum: No candidates found "
            f"(risk_on={risk_on}, insufficient={data_insufficient}, sma={sma_filtered}, "
            f"momentum={momentum_filtered}, vol={volatility_filtered})"
        )
        return []

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = [ticker for ticker, *_ in candidates[:top_n]]

    print(f"   🛡️ Defensive Momentum: RISK-ON, selected {len(selected)} stocks")
    print(f"      Market filter: {market_summary}")
    print(
        f"      Filter breakdown: insufficient={data_insufficient}, sma={sma_filtered}, "
        f"momentum={momentum_filtered}, vol={volatility_filtered}"
    )
    for ticker, score, perf_1y, perf_3m, sma_gap_pct, volatility in candidates[:min(5, len(candidates))]:
        print(
            f"      {ticker}: score={score:.1f}, 1Y={perf_1y:+.1f}%, 3M={perf_3m:+.1f}%, "
            f"SMA-gap={sma_gap_pct:+.1f}%, vol={volatility*100:.1f}%"
        )

    return selected
