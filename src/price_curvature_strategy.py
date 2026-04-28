from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

_KNOWN_NON_EQUITY_TICKERS = {
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLU",
    "XLRE",
    "XLC",
    "XLB",
    "GDX",
    "USO",
    "TLT",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VXX",
    "UUP",
    "GLD",
    "TNX",
}


def _price_curvature_sg_window(window: int, available_points: int) -> int:
    effective_window = min(window, available_points)
    if effective_window % 2 == 0:
        effective_window -= 1
    return effective_window


def _price_curvature_cached_daily_closes(price_history_cache, ticker: str, current_date: Optional[datetime]) -> pd.Series:
    date_ns = price_history_cache.date_ns_by_ticker.get(ticker)
    close_values = price_history_cache.close_by_ticker.get(ticker)
    if date_ns is None or close_values is None or date_ns.size == 0 or close_values.size == 0:
        return pd.Series(dtype=float)

    end_idx = date_ns.size
    if current_date is not None:
        current_ts = pd.Timestamp(current_date)
        if current_ts.tz is not None:
            current_ts = current_ts.tz_convert("UTC").tz_localize(None)
        end_idx = int(np.searchsorted(date_ns, int(current_ts.value), side="right"))
        if end_idx <= 0:
            return pd.Series(dtype=float)

    timestamp_index = pd.to_datetime(date_ns[:end_idx], unit="ns", utc=True)
    close_series = pd.Series(close_values[:end_idx], index=timestamp_index, dtype=float).dropna()
    if close_series.empty:
        return pd.Series(dtype=float)

    daily_closes = close_series.groupby(close_series.index.normalize()).last()
    return pd.to_numeric(daily_closes, errors="coerce").dropna().astype(float)


def _select_price_curvature_by_slope(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime],
    top_n: int,
    price_history_cache,
    *,
    selection_mode: str,
    strategy_label: str,
) -> List[str]:
    from config import (
        INVERSE_ETFS,
        PRICE_CURVATURE_LOOKBACK_DAYS,
        PRICE_CURVATURE_SG_WINDOW,
    )
    from performance_filters import filter_tickers_by_performance
    from strategy_cache_adapter import ensure_price_history_cache, resolve_cache_current_date

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    resolved_current_date = resolve_cache_current_date(price_history_cache, current_date, all_tickers)
    ranking_tickers = [
        ticker
        for ticker in all_tickers
        if ticker not in INVERSE_ETFS
        and ticker not in _KNOWN_NON_EQUITY_TICKERS
        and not ticker.endswith("-USD")
    ]
    prefilter_input_count = len(ranking_tickers)
    ranking_tickers = filter_tickers_by_performance(
        ranking_tickers,
        resolved_current_date,
        strategy_label,
        price_history_cache=price_history_cache,
    )
    prefilter_removed = prefilter_input_count - len(ranking_tickers)

    print(
        f"   🌀 {strategy_label}: Analyzing {len(ranking_tickers)} stocks "
        f"(filtered from {len(all_tickers)})"
    )

    candidates = []
    data_insufficient = 0
    non_positive_one_month = 0
    one_year_pool_count = 0
    one_year_scores: Dict[str, float] = {}

    if selection_mode == "1y_prefilter":
        from shared_strategies import select_top_performers_with_scores

        top_1y_with_scores = select_top_performers_with_scores(
            ranking_tickers,
            ticker_data_grouped,
            current_date,
            lookback_days=365,
            top_n=top_n,
            apply_performance_filter=True,
            filter_label=strategy_label,
            price_history_cache=price_history_cache,
        )
        if not top_1y_with_scores:
            print(f"   ❌ {strategy_label}: No 1Y performance pool available")
            return []

        one_year_scores = {ticker: perf for ticker, perf in top_1y_with_scores}
        ranking_tickers = [ticker for ticker, _ in top_1y_with_scores]
        one_year_pool_count = len(ranking_tickers)
        print(
            f"   📐 Formula: apply the shared 1M/3M/6M/1Y performance prefilter, keep the top {one_year_pool_count} 1Y performers, "
            f"then apply a {PRICE_CURVATURE_SG_WINDOW}-day SG filter to daily log-price and rank by slope"
        )
    else:
        print(
            f"   📐 Formula: apply the shared 1M/3M/6M/1Y performance prefilter, keep stocks with positive 1M return, then apply a {PRICE_CURVATURE_SG_WINDOW}-day SG filter to daily log-price and rank by slope"
        )

    for ticker in ranking_tickers:
        try:
            daily_closes = _price_curvature_cached_daily_closes(price_history_cache, ticker, current_date)
            min_points = max(PRICE_CURVATURE_LOOKBACK_DAYS, PRICE_CURVATURE_SG_WINDOW + 2, 22)
            if len(daily_closes) < min_points:
                data_insufficient += 1
                continue

            one_month_return = float((daily_closes.iloc[-1] / daily_closes.iloc[-22]) - 1.0)
            if selection_mode == "positive_1m" and one_month_return <= 0.0:
                non_positive_one_month += 1
                continue

            lookback = daily_closes.tail(PRICE_CURVATURE_LOOKBACK_DAYS)
            sg_window = _price_curvature_sg_window(PRICE_CURVATURE_SG_WINDOW, len(lookback))
            if len(lookback) < 20 or sg_window < 5:
                data_insufficient += 1
                continue

            log_prices = np.log(np.clip(lookback.to_numpy(dtype=float), 1e-12, None))
            polyorder = 3 if sg_window >= 7 else 2
            smoothed = savgol_filter(log_prices, window_length=sg_window, polyorder=polyorder, mode="interp")
            sg_slope_series = savgol_filter(
                log_prices,
                window_length=sg_window,
                polyorder=polyorder,
                deriv=1,
                delta=1.0,
                mode="interp",
            )
            sg_accel_series = savgol_filter(
                log_prices,
                window_length=sg_window,
                polyorder=polyorder,
                deriv=2,
                delta=1.0,
                mode="interp",
            )

            acceleration = float(sg_accel_series[-1])
            end_slope = float(sg_slope_series[-1])
            residual_vol = float(np.std(log_prices - smoothed))

            score = end_slope

            if len(candidates) < 3:
                print(
                    f"   🔍 DEBUG {ticker}: slope={end_slope:.6f}, accel={acceleration:.6f}, "
                    f"1M={one_month_return:.2%}, residual_vol={residual_vol:.6f}"
                )

            candidate = {
                "ticker": ticker,
                "score": score,
                "perf_1m": one_month_return,
                "acceleration": acceleration,
                "end_slope": end_slope,
                "residual_vol": residual_vol,
            }
            if selection_mode == "1y_prefilter":
                candidate["perf_1y_pct"] = float(one_year_scores.get(ticker, 0.0))

            candidates.append(candidate)
        except Exception:
            data_insufficient += 1

    candidates.sort(key=lambda item: (item["score"], item["acceleration"]), reverse=True)

    print("   📊 Analysis Summary:")
    print(f"      Stocks analyzed: {len(ranking_tickers)}")
    print(f"      Shared prefilter removed: {prefilter_removed}")
    print(f"      Data insufficient: {data_insufficient}")
    if selection_mode == "1y_prefilter":
        print(f"      1Y prefilter pool: {one_year_pool_count}")
    else:
        print(f"      Non-positive 1M return: {non_positive_one_month}")
    print(f"      Valid candidates: {len(candidates)}")

    if not candidates:
        print(f"   ❌ No {strategy_label} candidates found")
        return []

    print(f"   📊 Top {min(len(candidates), top_n)} {strategy_label} candidates:")
    for index, candidate in enumerate(candidates[:top_n], start=1):
        detail = (
            f"      {index}. {candidate['ticker']}: score={candidate['score']:.6f}, "
            f"slope={candidate['end_slope']:.6f}, "
            f"accel={candidate['acceleration']:.6f}, "
        )
        if selection_mode == "1y_prefilter":
            detail += f"1Y={candidate['perf_1y_pct']:+.1f}%, "
        else:
            detail += f"1M={candidate['perf_1m']:.2%}, "
        detail += f"noise={candidate['residual_vol']:.6f}"
        print(detail)

    selected = [candidate["ticker"] for candidate in candidates[:top_n]]
    print(f"   ✅ {strategy_label} selected {len(selected)} tickers: {selected}")
    return selected


def select_price_curvature_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 20,
    price_history_cache=None,
) -> List[str]:
    """
    Select stocks with positive 1M return, then rank them by SG slope.
    """
    return _select_price_curvature_by_slope(
        all_tickers,
        ticker_data_grouped,
        current_date,
        top_n,
        price_history_cache,
        selection_mode="positive_1m",
        strategy_label="Price Curvature",
    )


def select_price_curvature_1y_slope_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: Optional[datetime] = None,
    top_n: int = 20,
    price_history_cache=None,
) -> List[str]:
    """
    Select the top 1Y performers, then rank that pool by SG slope.
    """
    return _select_price_curvature_by_slope(
        all_tickers,
        ticker_data_grouped,
        current_date,
        top_n,
        price_history_cache,
        selection_mode="1y_prefilter",
        strategy_label="Price Curvature 1Y Slope",
    )
