from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def select_price_curvature_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 20,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks whose recent log-price path has the strongest upward curvature.

    The strategy fits a quadratic to the last N trading days of log-prices:
        log(price) ~= a*x^2 + b*x + c

    Positive `a` means the trend is bending upward. We then require:
    - positive curvature
    - positive end-of-window slope
    - positive recent return
    - a minimum fit quality
    """
    from config import (
        INVERSE_ETFS,
        MIN_DATA_DAYS_PERIOD_DATA,
        PRICE_CURVATURE_LOOKBACK_DAYS,
        PRICE_CURVATURE_MIN_FIT_R2,
        PRICE_CURVATURE_MIN_RECENT_RETURN,
        PRICE_CURVATURE_MIN_SLOPE,
    )
    from performance_filters import filter_tickers_by_performance
    from strategy_cache_adapter import ensure_price_history_cache

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    tickers_to_use = [ticker for ticker in all_tickers if ticker not in INVERSE_ETFS]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "Price Curvature",
        price_history_cache=price_history_cache,
    )

    print(
        f"   🌀 Price Curvature: Analyzing {len(filtered_tickers)} tickers "
        f"(filtered from {len(all_tickers)})"
    )
    print(
        "   📐 Formula: fit quadratic to recent log-price, rank by positive upward curvature"
    )

    candidates = []
    data_insufficient = 0
    non_positive_curvature = 0
    weak_trend = 0
    poor_fit = 0

    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < MIN_DATA_DAYS_PERIOD_DATA:
                data_insufficient += 1
                continue

            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, "tz") and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                    else:
                        current_ts = current_ts.tz_convert(ticker_data.index.tz)
                ticker_data = ticker_data.loc[:current_ts]

            closes = pd.to_numeric(ticker_data["Close"], errors="coerce").dropna()
            if len(closes) < max(PRICE_CURVATURE_LOOKBACK_DAYS, 40):
                data_insufficient += 1
                continue

            lookback = closes.tail(PRICE_CURVATURE_LOOKBACK_DAYS)
            if len(lookback) < 20:
                data_insufficient += 1
                continue

            log_prices = np.log(np.clip(lookback.to_numpy(dtype=float), 1e-12, None))
            x = np.linspace(-1.0, 1.0, num=len(log_prices), dtype=float)

            a, b, c = np.polyfit(x, log_prices, deg=2)
            fitted = (a * np.square(x)) + (b * x) + c

            residual_ss = float(np.square(log_prices - fitted).sum())
            total_ss = float(np.square(log_prices - log_prices.mean()).sum())
            fit_r2 = 1.0 - (residual_ss / total_ss) if total_ss > 0 else 0.0

            curvature = 2.0 * a
            end_slope = (2.0 * a * x[-1]) + b
            recent_return = float((lookback.iloc[-1] / lookback.iloc[0]) - 1.0)

            if curvature <= 0:
                non_positive_curvature += 1
                continue

            if end_slope <= PRICE_CURVATURE_MIN_SLOPE or recent_return <= PRICE_CURVATURE_MIN_RECENT_RETURN:
                weak_trend += 1
                continue

            if fit_r2 < PRICE_CURVATURE_MIN_FIT_R2:
                poor_fit += 1
                continue

            # Favor curves that are both convex and already trending up.
            score = curvature * max(end_slope, 0.0) * (0.5 + 0.5 * fit_r2) * (1.0 + recent_return)

            if len(candidates) < 3:
                print(
                    f"   🔍 DEBUG {ticker}: curvature={curvature:.6f}, "
                    f"end_slope={end_slope:.6f}, recent_return={recent_return:.2%}, r2={fit_r2:.3f}"
                )

            candidates.append(
                {
                    "ticker": ticker,
                    "score": score,
                    "curvature": curvature,
                    "end_slope": end_slope,
                    "recent_return": recent_return,
                    "fit_r2": fit_r2,
                }
            )
        except Exception:
            data_insufficient += 1

    candidates.sort(key=lambda item: item["score"], reverse=True)

    print("   📊 Analysis Summary:")
    print(f"      Total analyzed: {len(all_tickers)}")
    print(f"      Data insufficient: {data_insufficient}")
    print(f"      Non-positive curvature: {non_positive_curvature}")
    print(f"      Weak trend: {weak_trend}")
    print(f"      Poor quadratic fit: {poor_fit}")
    print(f"      Valid candidates: {len(candidates)}")

    if not candidates:
        print("   ❌ No Price Curvature candidates found")
        return []

    print(f"   📊 Top {min(len(candidates), top_n)} Price Curvature candidates:")
    for index, candidate in enumerate(candidates[:top_n], start=1):
        print(
            f"      {index}. {candidate['ticker']}: score={candidate['score']:.6f}, "
            f"curvature={candidate['curvature']:.6f}, "
            f"slope={candidate['end_slope']:.6f}, "
            f"return={candidate['recent_return']:.2%}, "
            f"r2={candidate['fit_r2']:.3f}"
        )

    selected = [candidate["ticker"] for candidate in candidates[:top_n]]
    print(f"   ✅ Price Curvature selected {len(selected)} tickers: {selected}")
    return selected
