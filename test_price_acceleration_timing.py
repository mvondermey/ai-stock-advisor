#!/usr/bin/env python3
import contextlib
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config

config.ENABLE_DATA_DOWNLOAD = False
config.ENABLE_JSON_OUTPUT = False

from config import INVERSE_ETFS, MIN_DATA_DAYS_PERIOD_DATA, N_TOP_TICKERS, PORTFOLIO_BUFFER_SIZE, PORTFOLIO_SIZE
from data_utils import _get_last_trading_day, load_all_market_data
from parallel_backtest import build_price_history_cache
from performance_filters import filter_tickers_by_performance
from shared_strategies import select_price_acceleration_stocks
from ticker_selection import find_top_performers, get_all_tickers


MATCH_TEST_DAYS = 10


def _prepare_universe():
    print("Loading ticker universe...")
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        raise RuntimeError("No tickers returned by get_all_tickers()")

    last_trading_day = _get_last_trading_day()
    end_date = datetime.combine(last_trading_day, datetime.min.time(), tzinfo=timezone.utc)

    load_start = time.perf_counter()
    all_tickers_data = load_all_market_data(all_available_tickers, end_date=end_date)
    load_time = time.perf_counter() - load_start

    if all_tickers_data.empty:
        raise RuntimeError("Data loading failed")

    if all_tickers_data["date"].dtype == "object" or not hasattr(all_tickers_data["date"].iloc[0], "tzinfo"):
        all_tickers_data["date"] = pd.to_datetime(all_tickers_data["date"], utc=True)
    elif all_tickers_data["date"].dt.tz is None:
        all_tickers_data["date"] = all_tickers_data["date"].dt.tz_localize("UTC")
    else:
        all_tickers_data["date"] = all_tickers_data["date"].dt.tz_convert("UTC")

    cutoff_date = end_date - timedelta(days=30)
    recent_data = all_tickers_data[all_tickers_data["date"] >= cutoff_date]
    active_tickers = recent_data["ticker"].unique().tolist()
    all_tickers_data = all_tickers_data[all_tickers_data["ticker"].isin(active_tickers)].copy()
    all_available_tickers = sorted(set(active_tickers))

    print(f"Data load complete in {load_time:.2f}s with {len(all_available_tickers)} active tickers")
    print("Selecting top performers...")

    selection_start = time.perf_counter()
    top_performers_data = find_top_performers(
        all_available_tickers=all_available_tickers,
        all_tickers_data=all_tickers_data,
        return_tickers=True,
        n_top=N_TOP_TICKERS,
        fcf_min_threshold=None,
        ebitda_min_threshold=None,
    )
    selection_time = time.perf_counter() - selection_start

    if not top_performers_data:
        raise RuntimeError("No top performers found")

    top_tickers = [ticker for ticker, _ in top_performers_data]
    print(f"Top ticker selection complete in {selection_time:.2f}s with {len(top_tickers)} tickers")

    grouped = {}
    for ticker, group in all_tickers_data.groupby("ticker"):
        grouped[ticker] = group.sort_values("date").set_index("date").copy()

    current_date = all_tickers_data["date"].max().to_pydatetime()
    available_dates = sorted(all_tickers_data["date"].drop_duplicates().tolist())
    return grouped, top_tickers, current_date, available_dates, load_time, selection_time


def _old_select_price_acceleration(all_tickers, ticker_data_grouped, current_date, top_n):
    tickers_to_use = [ticker for ticker in all_tickers if ticker not in INVERSE_ETFS]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "Price Acceleration",
    )

    candidates = []
    for ticker in filtered_tickers:
        ticker_data = ticker_data_grouped.get(ticker)
        if ticker_data is None or len(ticker_data) < MIN_DATA_DAYS_PERIOD_DATA:
            continue

        current_ts = pd.Timestamp(current_date)
        if hasattr(ticker_data.index, "tz") and ticker_data.index.tz is not None and current_ts.tz is None:
            current_ts = current_ts.tz_localize(ticker_data.index.tz)

        ticker_data_filtered = ticker_data.loc[:current_ts]
        prices = ticker_data_filtered["Close"].dropna()
        if len(prices) < 30:
            continue

        velocity = prices.pct_change().dropna()
        if len(velocity) < 20:
            continue

        acceleration = velocity.diff().dropna()
        if len(acceleration) < 10:
            continue

        recent_velocity = velocity.tail(10).mean()
        recent_acceleration = acceleration.tail(5).mean()
        latest_acceleration = acceleration.iloc[-1]
        recent_accel_series = acceleration.tail(5)
        positive_accel_days = (recent_accel_series > 0).sum()
        consistency_score = positive_accel_days / 5

        if recent_velocity <= 0.001:
            continue
        if recent_acceleration <= 0:
            continue

        accel_score = (
            recent_acceleration * 0.4
            + latest_acceleration * 0.4
            + consistency_score * recent_acceleration * 0.2
        )
        final_score = accel_score * (1 + recent_velocity * 100)

        candidates.append(
            {
                "ticker": ticker,
                "score": final_score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return [item["ticker"] for item in candidates[:top_n]]


def _run_match_check(top_tickers, grouped, top_n, test_date, price_history_cache):
    with contextlib.redirect_stdout(io.StringIO()):
        uncached_start = time.perf_counter()
        uncached_selected = _old_select_price_acceleration(
            top_tickers,
            grouped,
            current_date=test_date,
            top_n=top_n,
        )
        uncached_time = time.perf_counter() - uncached_start

    with contextlib.redirect_stdout(io.StringIO()):
        cached_start = time.perf_counter()
        cached_selected = select_price_acceleration_stocks(
            top_tickers,
            grouped,
            current_date=test_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        )
        cached_time = time.perf_counter() - cached_start

    return {
        "date": test_date,
        "uncached_selected": uncached_selected,
        "cached_selected": cached_selected,
        "uncached_time": uncached_time,
        "cached_time": cached_time,
        "exact_match": uncached_selected == cached_selected,
        "set_match": set(uncached_selected) == set(cached_selected),
    }


def main():
    grouped, top_tickers, current_date, available_dates, load_time, selection_time = _prepare_universe()
    top_n = PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE

    print("Building price history cache...")
    cache_start = time.perf_counter()
    price_history_cache = build_price_history_cache(grouped)
    cache_time = time.perf_counter() - cache_start

    print("Timing latest-day selector comparison...")
    latest_result = _run_match_check(
        top_tickers,
        grouped,
        top_n,
        current_date,
        price_history_cache,
    )

    test_dates = [pd.Timestamp(d).to_pydatetime() for d in available_dates[-MATCH_TEST_DAYS:]]
    print(f"Checking selection matches across {len(test_dates)} recent dates...")
    multi_day_results = [
        _run_match_check(top_tickers, grouped, top_n, test_date, price_history_cache)
        for test_date in test_dates
    ]

    selector_speedup = latest_result["uncached_time"] / latest_result["cached_time"] if latest_result["cached_time"] > 0 else float("inf")
    end_to_end_cached = cache_time + latest_result["cached_time"]
    end_to_end_speedup = latest_result["uncached_time"] / end_to_end_cached if end_to_end_cached > 0 else float("inf")
    exact_match_count = sum(1 for result in multi_day_results if result["exact_match"])
    set_match_count = sum(1 for result in multi_day_results if result["set_match"])

    print()
    print("=" * 72)
    print("PRICE ACCELERATION TIMING RESULTS")
    print("=" * 72)
    print(f"Universe size:            {len(top_tickers)}")
    print(f"Selection date:          {current_date}")
    print(f"Data load time:          {load_time:.2f}s")
    print(f"Top universe time:       {selection_time:.2f}s")
    print(f"Uncached selector:       {latest_result['uncached_time']:.2f}s")
    print(f"Cache build time:        {cache_time:.2f}s")
    print(f"Cached selector:         {latest_result['cached_time']:.2f}s")
    print(f"Cached end-to-end:       {end_to_end_cached:.2f}s")
    print(f"Selector-only speedup:   {selector_speedup:.2f}x")
    print(f"End-to-end speedup:      {end_to_end_speedup:.2f}x")
    print(f"Exact order match:       {latest_result['exact_match']}")
    print(f"Set match:               {latest_result['set_match']}")
    print(f"Exact match ({len(multi_day_results)}d):    {exact_match_count}/{len(multi_day_results)}")
    print(f"Set match ({len(multi_day_results)}d):      {set_match_count}/{len(multi_day_results)}")
    print()
    print(f"Uncached selected: {latest_result['uncached_selected']}")
    print(f"Cached selected:   {latest_result['cached_selected']}")

    print()
    print("Recent-date match details:")
    for result in multi_day_results:
        print(
            f"  {pd.Timestamp(result['date']).date()}: "
            f"exact={result['exact_match']} set={result['set_match']} "
            f"uncached={result['uncached_time']:.2f}s cached={result['cached_time']:.2f}s"
        )


if __name__ == "__main__":
    main()
