#!/usr/bin/env python3
import argparse
import contextlib
import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
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

# Disable AI model retraining for faster timing benchmark
config.ENABLE_WALK_FORWARD_RETRAINING = False
config.ENABLE_AI_ELITE = False
config.ENABLE_AI_REGIME = False
config.ENABLE_UNIVERSAL_MODEL = False
config.ENABLE_AI_CHAMPION = False
config.ENABLE_ELITE_HYBRID = False
config.ENABLE_ELITE_RISK = False
config.ENABLE_AI_ELITE_FILTERED = False
config.ENABLE_AI_ELITE_MARKET_UP = False

from config import (
    CONCENTRATED_3M_MAX_VOLATILITY,
    DUAL_MOM_ABSOLUTE_THRESHOLD,
    DUAL_MOM_LOOKBACK_DAYS,
    INVERSE_ETFS,
    MIN_DATA_DAYS_PERIOD_DATA,
    MOM_ACCEL_LOOKBACK_DAYS,
    MOM_ACCEL_MIN_ACCELERATION,
    MOM_ACCEL_SHORT_LOOKBACK,
    N_TOP_TICKERS,
    PORTFOLIO_BUFFER_SIZE,
    PORTFOLIO_SIZE,
)
from data_utils import _get_last_trading_day, load_all_market_data
from bollinger_bands_strategy import select_bb_squeeze_breakout_stocks
from correlation_ensemble import select_correlation_ensemble_stocks
from new_strategies import (
    select_concentrated_3m_stocks,
    select_dual_momentum_stocks,
    select_momentum_acceleration_stocks,
    select_trend_breakout_stocks,
)
from parallel_backtest import build_price_history_cache
from performance_filters import filter_tickers_by_performance
from shared_strategies import (
    select_1m_3m_ratio_stocks,
    select_1y_3m_ratio_stocks,
    select_3m_1y_ratio_stocks,
    select_mean_reversion_stocks,
    select_momentum_volatility_hybrid_stocks,
    select_momentum_volatility_hybrid_6m_stocks,
    select_momentum_volatility_hybrid_1y3m_stocks,
    select_momentum_volatility_hybrid_1y_stocks,
    select_price_acceleration_stocks,
    select_risk_adj_mom_stocks,
    select_turnaround_stocks,
)
from dynamic_pool import select_dynamic_pool_stocks
from enhanced_static_bh_strategies import select_static_bh_3m_accel
from enhanced_volatility_trader import select_enhanced_volatility_stocks
from new_strategies import select_trend_following_atr_stocks
from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
from risk_adj_mom_1m_strategy import select_risk_adj_mom_1m_stocks
from risk_adj_mom_3m_market_up_strategy import select_risk_adj_mom_3m_market_up_stocks
from risk_adj_mom_3m_sentiment_strategy import select_risk_adj_mom_3m_sentiment_stocks
from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
from risk_adj_mom_3m_with_stops_strategy import select_risk_adj_mom_3m_with_stops_stocks
from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks
from risk_adj_mom_sentiment import select_risk_adj_mom_sentiment_stocks
from ai_champion_strategy import select_ai_champion_stocks
from ai_elite_strategy import select_ai_elite_stocks, select_ai_elite_ensemble_stocks, select_ai_elite_rank_ensemble_stocks
from ai_regime_strategy import select_ai_regime_stocks
from elite_risk_strategy import select_elite_risk_stocks
from elite_hybrid_strategy import select_elite_hybrid_stocks
from inverse_etf_hedge_strategy import select_inverse_etf_hedge_stocks
from analyst_recommendation_strategy import select_analyst_recommendation_stocks
from adaptive_strategy import select_adaptive_ensemble_stocks
from multi_strategy_acceleration import select_multi_strategy_acceleration_stocks
from multi_timeframe_ensemble import select_multi_timeframe_stocks
from pairs_trading import select_pairs_trading_stocks
from factor_rotation import select_factor_rotation_stocks
from enhanced_volatility_trader import select_ai_volatility_ensemble_stocks
from universal_model_strategy import select_universal_model_stocks
from ai_elite_filtered_strategy import select_ai_elite_filtered_stocks
from ai_elite_market_up_strategy import select_ai_elite_market_up_stocks
from ultimate_strategy import select_ultimate_stocks
from llm_strategy import select_llm_portfolio
from vol_sweet_mom_strategy import select_vol_sweet_mom_stocks
from shared_strategies import (
    select_bh_1y_dynamic_accel_stocks,
    select_bh_1y_volsweet_accel_stocks,
    select_dynamic_bh_stocks,
    select_quality_momentum_stocks,
    select_sector_rotation_etfs,
    select_volatility_adj_mom_stocks,
)
from volatility_ensemble import select_volatility_ensemble_stocks
from ticker_selection import find_top_performers, get_all_tickers

DEFAULT_MATCH_DAYS = 10
SUPPORTED_STRATEGIES = [
    "price_acceleration",
    "momentum_acceleration",
    "concentrated_3m",
    "dual_momentum",
    "1y_3m_ratio",
    "1m_3m_ratio",
    "3m_1y_ratio",
    "turnaround",
    "mom_vol_hybrid",
    "mom_vol_hybrid_6m",
    "mom_vol_hybrid_1y3m",
    "mom_vol_hybrid_1y",
    "risk_adj_mom",
    "risk_adj_mom_3m",
    "risk_adj_mom_6m",
    "risk_adj_mom_1m",
    "risk_adj_mom_3m_monthly",
    "risk_adj_mom_6m_monthly",
    "risk_adj_mom_1m_monthly",
    "risk_adj_mom_3m_sentiment",
    "risk_adj_mom_3m_market_up",
    "risk_adj_mom_3m_with_stops",
    "risk_adj_mom_sentiment",
    "risk_adj_mom_1m_vol_sweet",
    "mean_reversion",
    "bb_squeeze_breakout",
    "bb_breakout",
    "bb_mean_rev",
    "bb_rsi_combo",
    "trend_breakout",
    "trend_atr",
    "volatility_ensemble",
    "correlation_ensemble",
    "savgol_trend",
    "sector_rotation",
    "vol_adj_mom",
    "vol_sweet_mom",
    "quality_mom",
    "enhanced_volatility",
    "enhanced_volatility_6m",
    "enhanced_volatility_3m",
    "static_bh_1y",
    "static_bh_6m",
    "static_bh_3m",
    "static_bh_1m",
    "bh_1y_monthly",
    "bh_6m_monthly",
    "bh_3m_monthly",
    "bh_1m_monthly",
    "dynamic_bh_1y",
    "dynamic_bh_6m",
    "dynamic_bh_3m",
    "dynamic_bh_1m",
    "dynamic_bh",
    "dynamic_bh_1y_vol_filter",
    "dynamic_bh_1y_trailing_stop",
    "bh_1y_volsweet_accel",
    "bh_1y_dynamic_accel",
    "dynamic_pool",
    "static_bh_3m_accel",
    "static_bh_1y_volatility",
    "static_bh_1y_performance",
    "static_bh_1y_momentum",
    "static_bh_1y_atr",
    "static_bh_1y_hybrid",
    "ai_champion",
    "ai_elite",
    "ai_elite_ensemble",
    "ai_elite_rank_ensemble",
    "ai_elite_filtered",
    "ai_elite_market_up",
    "ai_regime",
    "ai_regime_monthly",
    "elite_risk",
    "elite_hybrid",
    "inverse_etf_hedge",
    "analyst_recommendation",
    "adaptive_ensemble",
    "multi_strategy_acceleration",
    "multi_timeframe_ensemble",
    "multi_tf_ensemble",
    "pairs_trading",
    "factor_rotation",
    "ai_volatility_ensemble",
    "voting_ensemble",
    "momentum_ai_hybrid",
    "universal_model",
    "ultimate_strategy",
    "llm_strategy",
]


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
    available_dates = [pd.Timestamp(d).to_pydatetime() for d in sorted(all_tickers_data["date"].drop_duplicates().tolist())]
    return grouped, top_tickers, current_date, available_dates, load_time, selection_time


def _timestamp_for_ticker(ticker_data, current_date):
    current_ts = pd.Timestamp(current_date)
    if hasattr(ticker_data.index, "tz") and ticker_data.index.tz is not None:
        if current_ts.tz is None:
            current_ts = current_ts.tz_localize(ticker_data.index.tz)
        else:
            current_ts = current_ts.tz_convert(ticker_data.index.tz)
    return current_ts


def _legacy_price_acceleration(all_tickers, ticker_data_grouped, current_date, top_n):
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

        current_ts = _timestamp_for_ticker(ticker_data, current_date)
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
        candidates.append((ticker, final_score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_momentum_acceleration(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Momentum Acceleration",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None:
                continue
            if len(ticker_data) < MOM_ACCEL_LOOKBACK_DAYS + MOM_ACCEL_SHORT_LOOKBACK:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)

            start_3m = current_ts - timedelta(days=MOM_ACCEL_LOOKBACK_DAYS)
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            if len(data_3m) < 5:
                continue

            valid_close = data_3m["Close"].dropna()
            if len(valid_close) < 2:
                continue

            momentum_3m = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            if momentum_3m <= 0:
                continue

            start_1m_current = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK)
            data_1m_current = ticker_data[(ticker_data.index >= start_1m_current) & (ticker_data.index <= current_ts)]
            if len(data_1m_current) < 10:
                continue

            valid_1m = data_1m_current["Close"].dropna()
            if len(valid_1m) < 2:
                continue

            momentum_1m_current = (valid_1m.iloc[-1] / valid_1m.iloc[0] - 1) * 100

            start_1m_prev = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK * 2)
            end_1m_prev = current_ts - timedelta(days=MOM_ACCEL_SHORT_LOOKBACK)
            data_1m_prev = ticker_data[(ticker_data.index >= start_1m_prev) & (ticker_data.index <= end_1m_prev)]
            if len(data_1m_prev) < 10:
                continue

            valid_1m_prev = data_1m_prev["Close"].dropna()
            if len(valid_1m_prev) < 2:
                continue

            momentum_1m_prev = (valid_1m_prev.iloc[-1] / valid_1m_prev.iloc[0] - 1) * 100
            acceleration = momentum_1m_current - momentum_1m_prev
            if acceleration < MOM_ACCEL_MIN_ACCELERATION:
                continue

            score = momentum_3m * (1 + acceleration / 100)
            candidates.append((ticker, score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_concentrated_3m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Concentrated 3M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 90:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_3m = current_ts - timedelta(days=90)
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            if len(data_3m) < 5:
                continue

            valid_close = data_3m["Close"].dropna()
            if len(valid_close) < 2:
                continue

            momentum_3m = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            if momentum_3m <= 0:
                continue

            returns = valid_close.pct_change().dropna()
            if len(returns) < 20:
                continue

            volatility = returns.std() * np.sqrt(252)
            if volatility > CONCENTRATED_3M_MAX_VOLATILITY:
                continue

            candidates.append((ticker, momentum_3m))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_dual_momentum(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Dual Momentum",
    )

    candidates = []
    total_momentum = 0.0
    valid_count = 0

    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < DUAL_MOM_LOOKBACK_DAYS:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_date = current_ts - timedelta(days=DUAL_MOM_LOOKBACK_DAYS)
            data_window = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= current_ts)]
            if len(data_window) < 5:
                continue

            valid_close = data_window["Close"].dropna()
            if len(valid_close) < 2:
                continue

            momentum = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100
            total_momentum += momentum
            valid_count += 1

            if momentum > DUAL_MOM_ABSOLUTE_THRESHOLD:
                candidates.append((ticker, momentum))
        except Exception:
            continue

    market_momentum = total_momentum / valid_count if valid_count > 0 else 0.0
    if market_momentum <= 0:
        return []

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_1y_3m_ratio(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "1Y/3M Ratio",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_3m = current_ts - timedelta(days=90)
            start_1y = current_ts - timedelta(days=365)

            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            data_1y = ticker_data[(ticker_data.index >= start_1y) & (ticker_data.index <= current_ts)]

            if len(data_3m) < 5 or len(data_1y) < 20:
                continue

            valid_3m = data_3m["Close"].dropna()
            valid_1y = data_1y["Close"].dropna()

            if len(valid_3m) < 2 or len(valid_1y) < 2:
                continue

            perf_3m = (valid_3m.iloc[-1] / valid_3m.iloc[0] - 1) * 100
            perf_1y = (valid_1y.iloc[-1] / valid_1y.iloc[0] - 1) * 100

            if perf_3m > 30 or perf_1y < 10:
                continue

            dip_score = perf_1y - perf_3m
            if dip_score > 15:
                candidates.append((ticker, dip_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_1m_3m_ratio(all_tickers, ticker_data_grouped, current_date, top_n):
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "1M/3M Ratio",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_1m = current_ts - timedelta(days=30)
            start_3m = current_ts - timedelta(days=90)

            data_1m = ticker_data[(ticker_data.index >= start_1m) & (ticker_data.index <= current_ts)]
            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]

            if len(data_1m) < 10 or len(data_3m) < 10:
                continue

            valid_1m = data_1m["Close"].dropna()
            valid_3m = data_3m["Close"].dropna()

            if len(valid_1m) < 2 or len(valid_3m) < 2:
                continue

            perf_1m = (valid_1m.iloc[-1] / valid_1m.iloc[0] - 1) * 100
            perf_3m = (valid_3m.iloc[-1] / valid_3m.iloc[0] - 1) * 100

            if perf_1m <= 0 or perf_3m < -20:
                continue

            annualized_1m = perf_1m * 3
            acceleration = annualized_1m - perf_3m

            if acceleration > 0:
                candidates.append((ticker, acceleration))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_3m_1y_ratio(all_tickers, ticker_data_grouped, current_date, top_n):
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "3M/1Y Ratio",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_3m = current_ts - timedelta(days=90)
            start_1y = current_ts - timedelta(days=365)

            data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= current_ts)]
            data_1y = ticker_data[(ticker_data.index >= start_1y) & (ticker_data.index <= current_ts)]

            if len(data_3m) < 5 or len(data_1y) < 20:
                continue

            valid_3m = data_3m["Close"].dropna()
            valid_1y = data_1y["Close"].dropna()

            if len(valid_3m) < 2 or len(valid_1y) < 2:
                continue

            perf_3m = (valid_3m.iloc[-1] / valid_3m.iloc[0] - 1) * 100
            perf_1y = (valid_1y.iloc[-1] / valid_1y.iloc[0] - 1) * 100

            if perf_3m <= 0 or perf_1y <= 0:
                continue

            annualized_3m = perf_3m * (365/90)
            acceleration = annualized_3m - perf_1y

            if perf_1y > 5 and acceleration > 5:
                candidates.append((ticker, acceleration))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_turnaround(all_tickers, ticker_data_grouped, current_date, top_n):
    tickers_to_use = [t for t in all_tickers if t not in INVERSE_ETFS]
    filtered_tickers = filter_tickers_by_performance(
        tickers_to_use,
        ticker_data_grouped,
        current_date,
        "Turnaround",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            start_1y = current_ts - timedelta(days=365)
            start_3y = current_ts - timedelta(days=1095)

            data_1y = ticker_data[(ticker_data.index >= start_1y) & (ticker_data.index <= current_ts)]
            data_3y = ticker_data[(ticker_data.index >= start_3y) & (ticker_data.index <= current_ts)]

            if len(data_1y) < 10 or len(data_3y) < 10:
                continue

            valid_1y = data_1y["Close"].dropna()
            valid_3y = data_3y["Close"].dropna()

            if len(valid_1y) < 2 or len(valid_3y) < 2:
                continue

            perf_1y = (valid_1y.iloc[-1] / valid_1y.iloc[0] - 1) * 100
            perf_3y = (valid_3y.iloc[-1] / valid_3y.iloc[0] - 1) * 100

            if perf_3y > 30 or perf_1y < 10:
                continue

            three_year_annual = perf_3y / 3
            recovery_score = perf_1y - three_year_annual

            if recovery_score > 10:
                candidates.append((ticker, recovery_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_mom_vol_hybrid(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Mom-Vol Hybrid",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_3m = min(63, len(closes) - 1)
            if lookback_3m < 20:
                continue

            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago
            annualized_3m = (1 + perf_3m) ** (252 / lookback_3m) - 1

            lookback_1y = min(252, len(closes) - 1)
            if lookback_1y < 60:
                continue

            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 30:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if annualized_3m > 0.0 and perf_1y > -0.3 and volatility < 3.0:
                momentum_score = annualized_3m * 0.6 + max(perf_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)
                candidates.append((ticker, composite_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_mom_vol_hybrid_6m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Mom-Vol Hybrid 6M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_6m = min(126, len(closes) - 1)
            if lookback_6m < 40:
                continue

            price_6m_ago = closes.iloc[-lookback_6m]
            if price_6m_ago <= 0:
                continue

            perf_6m = (latest_price - price_6m_ago) / price_6m_ago
            annualized_6m = (1 + perf_6m) ** (252 / lookback_6m) - 1

            lookback_1y = min(252, len(closes) - 1)
            if lookback_1y < 60:
                continue

            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 30:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if annualized_6m > 0.0 and perf_1y > -0.3 and volatility < 3.0:
                momentum_score = annualized_6m * 0.6 + max(perf_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)
                candidates.append((ticker, composite_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_mom_vol_hybrid_1y3m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Mom-Vol Hybrid 1Y/3M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(252, len(closes) - 1)
            if lookback_1y < 60:
                continue

            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3m = min(63, len(closes) - 1)
            if lookback_3m < 20:
                continue

            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 30:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if perf_1y > 0.05 and perf_3m < 0.10 and perf_1y > perf_3m and volatility < 2.0:
                ratio_1y_3m = (1 + perf_1y) / (1 + perf_3m) if perf_3m > -0.5 else 2.0
                volatility_penalty = min(volatility, 1.0)
                composite_score = ratio_1y_3m * (1 - volatility_penalty * 0.3)
                candidates.append((ticker, composite_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_mom_vol_hybrid_1y(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Mom-Vol Hybrid 1Y",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 10:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 10:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(252, len(closes) - 1)
            if lookback_1y < 10:
                continue

            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3y = min(756, len(closes) - 1)
            if lookback_3y >= 200:
                price_3y_ago = closes.iloc[-lookback_3y]
                if price_3y_ago > 0:
                    perf_3y = (latest_price - price_3y_ago) / price_3y_ago
                else:
                    perf_3y = 0.0
            else:
                perf_3y = 0.0

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if perf_1y > 0.0 and perf_3y > -0.5 and volatility < 2.5:
                momentum_score = perf_1y * 0.7 + max(perf_3y, 0) * 0.3
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)
                candidates.append((ticker, composite_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_risk_adj_mom(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Risk-Adj Mom",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            risk_adj_score = perf_1y / volatility if volatility > 0 else 0

            if perf_1y > 0 and volatility < 2.5:
                candidates.append((ticker, risk_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_mean_reversion(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Mean Reversion",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_60d = min(60, len(closes) - 1)
            price_60d_ago = closes.iloc[-lookback_60d]
            if price_60d_ago <= 0:
                continue

            perf_60d = (latest_price - price_60d_ago) / price_60d_ago

            lookback_20d = min(20, len(closes) - 1)
            price_20d_ago = closes.iloc[-lookback_20d]
            if price_20d_ago <= 0:
                continue

            perf_20d = (latest_price - price_20d_ago) / price_20d_ago

            if perf_60d < -0.2 and perf_20d > 0.05:
                reversion_score = perf_20d - perf_60d
                candidates.append((ticker, reversion_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bb_squeeze_breakout(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Squeeze",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_20d = min(20, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_20d:]

            if len(recent_prices) < 20:
                continue

            sma_20 = float(np.mean(recent_prices))
            std_20 = float(np.std(recent_prices, ddof=1))
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20

            breakout_score = 0
            if latest_price > upper_band:
                breakout_score = (latest_price - upper_band) / upper_band
            elif latest_price < lower_band:
                breakout_score = (lower_band - latest_price) / lower_band

            if breakout_score > 0.01:
                candidates.append((ticker, breakout_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_trend_breakout(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Trend Breakout",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_60d = min(60, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_60d:]

            if len(recent_prices) < 60:
                continue

            sma_60 = float(np.mean(recent_prices))
            latest_sma_20 = float(np.mean(recent_prices.iloc[-20:]))

            if latest_price > sma_60 and latest_price > latest_sma_20:
                trend_strength = (latest_price - sma_60) / sma_60
                candidates.append((ticker, trend_strength))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_volatility_ensemble(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Volatility Ensemble",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 30:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            lookback_60d = min(60, len(closes) - 1)
            price_60d_ago = closes.iloc[-lookback_60d]
            if price_60d_ago <= 0:
                continue

            perf_60d = (latest_price - price_60d_ago) / price_60d_ago

            if 0.15 < volatility < 2.0 and perf_60d > 0:
                vol_adj_score = perf_60d / volatility if volatility > 0 else 0
                candidates.append((ticker, vol_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_dynamic_bh(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Dynamic BH",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3m = min(63, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            lookback_6m = min(126, len(closes) - 1)
            price_6m_ago = closes.iloc[-lookback_6m]
            if price_6m_ago <= 0:
                continue

            perf_6m = (latest_price - price_6m_ago) / price_6m_ago

            momentum_score = perf_1y * 0.5 + perf_6m * 0.3 + perf_3m * 0.2

            if perf_1y > 0 and perf_3m > 0:
                candidates.append((ticker, momentum_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bh_1y_volsweet_accel(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BH 1Y VolSweet Accel",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            lookback_3m = min(63, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            if 0.1 < volatility < 1.5 and perf_1y > 0 and perf_3m > perf_1y:
                accel_score = perf_3m - perf_1y
                vol_adj_score = accel_score / volatility if volatility > 0 else 0
                candidates.append((ticker, vol_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bh_1y_dynamic_accel(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BH 1Y Dynamic Accel",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3m = min(63, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            lookback_6m = min(126, len(closes) - 1)
            price_6m_ago = closes.iloc[-lookback_6m]
            if price_6m_ago <= 0:
                continue

            perf_6m = (latest_price - price_6m_ago) / price_6m_ago

            if perf_1y > 0 and perf_3m > perf_1y and perf_6m > perf_3m:
                accel_score = perf_3m - perf_1y
                momentum_score = perf_1y * 0.4 + perf_6m * 0.3 + perf_3m * 0.3
                candidates.append((ticker, momentum_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_dynamic_pool(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Dynamic Pool",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3m = min(63, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            pool_score = perf_1y * 0.6 + perf_3m * 0.4

            if perf_1y > 0 and volatility < 2.5:
                candidates.append((ticker, pool_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_risk_adj_mom_3m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Risk-Adj Mom 3M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 90:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 90:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_3m = min(90, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 30:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            risk_adj_score = perf_3m / volatility if volatility > 0 else 0

            if perf_3m > 0 and volatility < 2.5:
                candidates.append((ticker, risk_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_risk_adj_mom_6m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Risk-Adj Mom 6M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 180:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 180:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_6m = min(180, len(closes) - 1)
            price_6m_ago = closes.iloc[-lookback_6m]
            if price_6m_ago <= 0:
                continue

            perf_6m = (latest_price - price_6m_ago) / price_6m_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            risk_adj_score = perf_6m / volatility if volatility > 0 else 0

            if perf_6m > 0 and volatility < 2.5:
                candidates.append((ticker, risk_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_risk_adj_mom_1m(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Risk-Adj Mom 1M",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 30:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 30:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1m = min(30, len(closes) - 1)
            price_1m_ago = closes.iloc[-lookback_1m]
            if price_1m_ago <= 0:
                continue

            perf_1m = (latest_price - price_1m_ago) / price_1m_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 20:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            risk_adj_score = perf_1m / volatility if volatility > 0 else 0

            if perf_1m > 0 and volatility < 3.0:
                candidates.append((ticker, risk_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bb_breakout(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Breakout",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_20d = min(20, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_20d:]

            if len(recent_prices) < 20:
                continue

            sma_20 = float(np.mean(recent_prices))
            std_20 = float(np.std(recent_prices, ddof=1))
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20

            if latest_price > upper_band:
                breakout_score = (latest_price - upper_band) / upper_band
                candidates.append((ticker, breakout_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bb_mean_rev(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB Mean Rev",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_20d = min(20, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_20d:]

            if len(recent_prices) < 20:
                continue

            sma_20 = float(np.mean(recent_prices))
            std_20 = float(np.std(recent_prices, ddof=1))
            lower_band = sma_20 - 2 * std_20

            if latest_price < lower_band:
                reversion_score = (lower_band - latest_price) / lower_band
                candidates.append((ticker, reversion_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_bb_rsi_combo(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "BB RSI Combo",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_20d = min(20, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_20d:]

            if len(recent_prices) < 20:
                continue

            sma_20 = float(np.mean(recent_prices))
            std_20 = float(np.std(recent_prices, ddof=1))
            lower_band = sma_20 - 2 * std_20

            if latest_price < lower_band:
                reversion_score = (lower_band - latest_price) / lower_band
                candidates.append((ticker, reversion_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_correlation_ensemble(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Correlation Ensemble",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if perf_1y > 0 and volatility < 2.0:
                score = perf_1y / volatility if volatility > 0 else 0
                candidates.append((ticker, score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_savgol_trend(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "SavGol Trend",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 60:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 60:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_60d = min(60, len(closes) - 1)
            recent_prices = closes.iloc[-lookback_60d:]

            if len(recent_prices) < 60:
                continue

            sma_60 = float(np.mean(recent_prices))
            trend_score = (latest_price - sma_60) / sma_60

            if trend_score > 0:
                candidates.append((ticker, trend_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_sector_rotation(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Sector Rotation",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            lookback_3m = min(90, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            if perf_1y > 0 and perf_3m > 0:
                sector_score = perf_1y * 0.6 + perf_3m * 0.4
                candidates.append((ticker, sector_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_vol_adj_mom(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Vol-Adj Mom",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            vol_adj_score = perf_1y / volatility if volatility > 0 else 0

            if perf_1y > 0 and volatility < 2.5:
                candidates.append((ticker, vol_adj_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_quality_mom(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Quality+Mom",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if perf_1y > 0 and volatility < 2.0:
                quality_score = perf_1y * 0.7 - volatility * 0.3
                candidates.append((ticker, quality_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_enhanced_volatility(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Enhanced Volatility",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 365:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 365:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_1y = min(365, len(closes) - 1)
            price_1y_ago = closes.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue

            perf_1y = (latest_price - price_1y_ago) / price_1y_ago

            daily_returns = closes.pct_change().dropna()
            if len(daily_returns) < 60:
                continue

            volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

            if 0.15 < volatility < 2.5 and perf_1y > 0:
                vol_score = perf_1y * volatility
                candidates.append((ticker, vol_score))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


def _legacy_static_bh_3m_accel(all_tickers, ticker_data_grouped, current_date, top_n):
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        ticker_data_grouped,
        current_date,
        "Static BH 3M Accel",
    )

    candidates = []
    for ticker in filtered_tickers:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is None or len(ticker_data) < 90:
                continue

            current_ts = _timestamp_for_ticker(ticker_data, current_date)
            data = ticker_data.loc[:current_ts]
            closes = data["Close"].dropna()

            if len(closes) < 90:
                continue

            latest_price = closes.iloc[-1]
            if latest_price <= 0:
                continue

            lookback_3m = min(90, len(closes) - 1)
            price_3m_ago = closes.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue

            perf_3m = (latest_price - price_3m_ago) / price_3m_ago

            if perf_3m > 0:
                candidates.append((ticker, perf_3m))
        except Exception:
            continue

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in candidates[:top_n]]


STRATEGY_REGISTRY = {
    "price_acceleration": {
        "label": "Price Acceleration",
        "legacy": _legacy_price_acceleration,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_price_acceleration_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "momentum_acceleration": {
        "label": "Momentum Acceleration",
        "legacy": _legacy_momentum_acceleration,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_acceleration_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "concentrated_3m": {
        "label": "Concentrated 3M",
        "legacy": _legacy_concentrated_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_concentrated_3m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "dual_momentum": {
        "label": "Dual Momentum",
        "legacy": _legacy_dual_momentum,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_dual_momentum_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        )[0],
    },
    "1y_3m_ratio": {
        "label": "1Y/3M Ratio",
        "legacy": _legacy_1y_3m_ratio,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_1y_3m_ratio_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "1m_3m_ratio": {
        "label": "1M/3M Ratio",
        "legacy": _legacy_1m_3m_ratio,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_1m_3m_ratio_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "3m_1y_ratio": {
        "label": "3M/1Y Ratio",
        "legacy": _legacy_3m_1y_ratio,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_3m_1y_ratio_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "turnaround": {
        "label": "Turnaround",
        "legacy": _legacy_turnaround,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_turnaround_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "mom_vol_hybrid": {
        "label": "Mom-Vol Hybrid",
        "legacy": _legacy_mom_vol_hybrid,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_volatility_hybrid_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "mom_vol_hybrid_6m": {
        "label": "Mom-Vol Hybrid 6M",
        "legacy": _legacy_mom_vol_hybrid_6m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_volatility_hybrid_6m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "mom_vol_hybrid_1y3m": {
        "label": "Mom-Vol Hybrid 1Y/3M",
        "legacy": _legacy_mom_vol_hybrid_1y3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_volatility_hybrid_1y3m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "mom_vol_hybrid_1y": {
        "label": "Mom-Vol Hybrid 1Y",
        "legacy": _legacy_mom_vol_hybrid_1y,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_volatility_hybrid_1y_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom": {
        "label": "Risk-Adj Mom",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "mean_reversion": {
        "label": "Mean Reversion",
        "legacy": _legacy_mean_reversion,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_mean_reversion_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "bb_squeeze_breakout": {
        "label": "BB Squeeze Breakout",
        "legacy": _legacy_bb_squeeze_breakout,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bb_squeeze_breakout_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "trend_breakout": {
        "label": "Trend Breakout",
        "legacy": _legacy_trend_breakout,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_trend_breakout_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "volatility_ensemble": {
        "label": "Volatility Ensemble",
        "legacy": _legacy_volatility_ensemble,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_volatility_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "dynamic_bh": {
        "label": "Dynamic BH",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_dynamic_bh_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            period='1y',
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "bh_1y_volsweet_accel": {
        "label": "BH 1Y VolSweet Accel",
        "legacy": _legacy_bh_1y_volsweet_accel,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bh_1y_volsweet_accel_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "bh_1y_dynamic_accel": {
        "label": "BH 1Y Dynamic Accel",
        "legacy": _legacy_bh_1y_dynamic_accel,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bh_1y_dynamic_accel_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "dynamic_pool": {
        "label": "Dynamic Pool",
        "legacy": _legacy_dynamic_pool,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_dynamic_pool_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_3m": {
        "label": "Risk-Adj Mom 3M",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_3m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_6m": {
        "label": "Risk-Adj Mom 6M",
        "legacy": _legacy_risk_adj_mom_6m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_6m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_1m": {
        "label": "Risk-Adj Mom 1M",
        "legacy": _legacy_risk_adj_mom_1m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_1m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "bb_breakout": {
        "label": "BB Breakout",
        "legacy": _legacy_bb_breakout,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bb_breakout_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "bb_mean_rev": {
        "label": "BB Mean Rev",
        "legacy": _legacy_bb_mean_rev,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bb_mean_reversion_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "bb_rsi_combo": {
        "label": "BB RSI Combo",
        "legacy": _legacy_bb_rsi_combo,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_bb_rsi_combo_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "correlation_ensemble": {
        "label": "Correlation Ensemble",
        "legacy": _legacy_correlation_ensemble,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_correlation_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "savgol_trend": {
        "label": "SavGol Trend",
        "legacy": _legacy_savgol_trend,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_savgol_trend_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "sector_rotation": {
        "label": "Sector Rotation",
        "legacy": _legacy_sector_rotation,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_sector_rotation_etfs(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "vol_adj_mom": {
        "label": "Vol-Adj Mom",
        "legacy": _legacy_vol_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_volatility_adj_mom_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "quality_mom": {
        "label": "Quality+Mom",
        "legacy": _legacy_quality_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_quality_momentum_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "enhanced_volatility": {
        "label": "Enhanced Volatility",
        "legacy": _legacy_enhanced_volatility,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_enhanced_volatility_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "static_bh_3m_accel": {
        "label": "Static BH 3M Accel",
        "legacy": _legacy_static_bh_3m_accel,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_static_bh_3m_accel(
            grouped,
            current_date=current_date,
            top_n=top_n,
        )[0],
    },
    "risk_adj_mom_3m_sentiment": {
        "label": "Risk-Adj Mom 3M Sentiment",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_3m_sentiment_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_3m_market_up": {
        "label": "Risk-Adj Mom 3M Market Up",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_3m_market_up_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "risk_adj_mom_3m_with_stops": {
        "label": "Risk-Adj Mom 3M With Stops",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_3m_with_stops_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_sentiment": {
        "label": "Risk-Adj Mom Sentiment",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_sentiment_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_1m_vol_sweet": {
        "label": "Risk-Adj Mom 1M Vol Sweet",
        "legacy": _legacy_risk_adj_mom_1m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_1m_vol_sweet_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "trend_atr": {
        "label": "Trend ATR",
        "legacy": _legacy_trend_breakout,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_trend_following_atr_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "static_bh_1y": {
        "label": "Static BH 1Y",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "static_bh_6m": {
        "label": "Static BH 6M",
        "legacy": _legacy_risk_adj_mom_6m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=180,
            top_n=top_n,
        ),
    },
    "static_bh_3m": {
        "label": "Static BH 3M",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=90,
            top_n=top_n,
        ),
    },
    "static_bh_1m": {
        "label": "Static BH 1M",
        "legacy": _legacy_risk_adj_mom_1m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=30,
            top_n=top_n,
        ),
    },
    "bh_1y_monthly": {
        "label": "BH 1Y Monthly",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "bh_6m_monthly": {
        "label": "BH 6M Monthly",
        "legacy": _legacy_risk_adj_mom_6m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=180,
            top_n=top_n,
        ),
    },
    "bh_3m_monthly": {
        "label": "BH 3M Monthly",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=90,
            top_n=top_n,
        ),
    },
    "bh_1m_monthly": {
        "label": "BH 1M Monthly",
        "legacy": _legacy_risk_adj_mom_1m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=30,
            top_n=top_n,
        ),
    },
    "dynamic_bh_1y": {
        "label": "Dynamic BH 1Y",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
            apply_performance_filter=True,
        ),
    },
    "dynamic_bh_6m": {
        "label": "Dynamic BH 6M",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=180,
            top_n=top_n,
            apply_performance_filter=True,
        ),
    },
    "dynamic_bh_3m": {
        "label": "Dynamic BH 3M",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=90,
            top_n=top_n,
            apply_performance_filter=True,
        ),
    },
    "dynamic_bh_1m": {
        "label": "Dynamic BH 1M",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=30,
            top_n=top_n,
            apply_performance_filter=True,
        ),
    },
    "ai_champion": {
        "label": "AI Champion",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_champion_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            predicted_strategy="risk_adj_mom_3m",  # Default strategy for benchmark
        ),
    },
    "ai_elite": {
        "label": "AI Elite",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_elite_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_elite_ensemble": {
        "label": "AI Elite Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_elite_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_elite_rank_ensemble": {
        "label": "AI Elite Rank Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_elite_rank_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_regime": {
        "label": "AI Regime",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_regime_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "elite_risk": {
        "label": "Elite Risk",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_elite_risk_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "elite_hybrid": {
        "label": "Elite Hybrid",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_elite_hybrid_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "inverse_etf_hedge": {
        "label": "Inverse ETF Hedge",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_inverse_etf_hedge_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "analyst_recommendation": {
        "label": "Analyst Recommendation",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_analyst_recommendation_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "adaptive_ensemble": {
        "label": "Adaptive Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_adaptive_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "multi_strategy_acceleration": {
        "label": "Multi Strategy Acceleration",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_multi_strategy_acceleration_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "multi_timeframe_ensemble": {
        "label": "Multi Timeframe Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_multi_timeframe_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "pairs_trading": {
        "label": "Pairs Trading",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_pairs_trading_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "factor_rotation": {
        "label": "Factor Rotation",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_factor_rotation_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_volatility_ensemble": {
        "label": "AI Volatility Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_volatility_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "voting_ensemble": {
        "label": "Voting Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_voting_ensemble_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "momentum_ai_hybrid": {
        "label": "Momentum AI Hybrid",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_momentum_ai_hybrid_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "risk_adj_mom_3m_monthly": {
        "label": "Risk-Adj Mom 3M Monthly",
        "legacy": _legacy_risk_adj_mom_3m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_3m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_6m_monthly": {
        "label": "Risk-Adj Mom 6M Monthly",
        "legacy": _legacy_risk_adj_mom_6m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_6m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "risk_adj_mom_1m_monthly": {
        "label": "Risk-Adj Mom 1M Monthly",
        "legacy": _legacy_risk_adj_mom_1m,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_risk_adj_mom_1m_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "vol_sweet_mom": {
        "label": "Vol Sweet Mom",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_vol_sweet_mom_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "enhanced_volatility_6m": {
        "label": "Enhanced Volatility 6M",
        "legacy": _legacy_enhanced_volatility,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_enhanced_volatility_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "enhanced_volatility_3m": {
        "label": "Enhanced Volatility 3M",
        "legacy": _legacy_enhanced_volatility,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_enhanced_volatility_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "dynamic_bh_1y_vol_filter": {
        "label": "Dynamic BH 1Y Vol Filter",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers_vol_filtered(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            max_volatility=0.4,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "dynamic_bh_1y_trailing_stop": {
        "label": "Dynamic BH 1Y Trailing Stop",
        "legacy": _legacy_dynamic_bh,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "static_bh_1y_volatility": {
        "label": "Static BH 1Y Volatility",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers_vol_filtered(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            max_volatility=0.4,
            top_n=top_n,
            price_history_cache=price_history_cache,
        ),
    },
    "static_bh_1y_performance": {
        "label": "Static BH 1Y Performance",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "static_bh_1y_momentum": {
        "label": "Static BH 1Y Momentum",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "static_bh_1y_atr": {
        "label": "Static BH 1Y ATR",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "static_bh_1y_hybrid": {
        "label": "Static BH 1Y Hybrid",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_top_performers(
            all_tickers,
            grouped,
            current_date=current_date,
            lookback_days=365,
            top_n=top_n,
        ),
    },
    "ai_elite_filtered": {
        "label": "AI Elite Filtered",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_elite_filtered_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_elite_market_up": {
        "label": "AI Elite Market Up",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_elite_market_up_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ai_regime_monthly": {
        "label": "AI Regime Monthly",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ai_regime_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "multi_tf_ensemble": {
        "label": "Multi TF Ensemble",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_multi_timeframe_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "universal_model": {
        "label": "Universal Model",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_universal_model_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "ultimate_strategy": {
        "label": "Ultimate Strategy",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_ultimate_stocks(
            all_tickers,
            grouped,
            current_date=current_date,
            top_n=top_n,
        ),
    },
    "llm_strategy": {
        "label": "LLM Strategy",
        "legacy": _legacy_risk_adj_mom,
        "cached": lambda all_tickers, grouped, current_date, top_n, price_history_cache: select_llm_portfolio(
            all_tickers,
            grouped.get('all_tickers_data', pd.DataFrame()),
            current_date,
        )[:top_n],
    },
}


def _run_single_check(strategy_name, top_tickers, grouped, top_n, test_date, price_history_cache):
    strategy = STRATEGY_REGISTRY[strategy_name]

    with contextlib.redirect_stdout(io.StringIO()):
        uncached_start = time.perf_counter()
        uncached_selected = strategy["legacy"](top_tickers, grouped, test_date, top_n)
        uncached_time = time.perf_counter() - uncached_start

    with contextlib.redirect_stdout(io.StringIO()):
        cached_start = time.perf_counter()
        cached_selected = strategy["cached"](top_tickers, grouped, test_date, top_n, price_history_cache)
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


def _print_strategy_summary(
    strategy_name, label, latest_result, all_results, cache_time, universe_size, current_date, load_time, selection_time
):
    end_to_end_cached = cache_time + latest_result["cached_time"]
    selector_speedup = latest_result["uncached_time"] / latest_result["cached_time"]
    end_to_end_speedup = (load_time + selection_time + latest_result["uncached_time"]) / end_to_end_cached
    exact_match_count = sum(1 for r in all_results if r["exact_match"])
    set_match_count = sum(1 for r in all_results if r["set_match"])

    print()
    print("=" * 70)
    print(f"{label.upper()} TIMING RESULTS")
    print("=" * 70)
    print(f"Strategy key:            {strategy_name}")
    print(f"Universe size:           {universe_size}")
    print(f"Selection date:          {current_date}")
    print(f"Data load time:          {load_time:.2f}s")
    print(f"Top universe time:       {selection_time:.2f}s")
    print(f"Uncached selector:       {latest_result['uncached_time']:.2f}s")
    print(f"Cache build time:        {cache_time:.2f}s")
    print(f"Cached selector:         {latest_result['cached_time']:.2f}s")
    print(f"Cached end-to-end:       {end_to_end_cached:.2f}s")
    print(f"Selector-only speedup:   {selector_speedup:.2f}x")
    print(f"End-to-end speedup:      {end_to_end_speedup:.2f}x")
    
    # Show match status with error highlighting for mismatches
    exact_match_str = f"{latest_result['exact_match']}"
    set_match_str = f"{latest_result['set_match']}"
    if not latest_result['exact_match'] or not latest_result['set_match']:
        exact_match_str = f"ERROR: {latest_result['exact_match']}"
        set_match_str = f"ERROR: {latest_result['set_match']}"
    
    print(f"Exact order match:       {exact_match_str}")
    print(f"Set match:               {set_match_str}")
    print(f"Exact match ({len(all_results)}d):    {exact_match_count}/{len(all_results)}")
    print(f"Set match ({len(all_results)}d):      {set_match_count}/{len(all_results)}")
    
    # Always show selections when there's a mismatch
    if not latest_result['exact_match'] or not latest_result['set_match']:
        print()
        print(f"Uncached selected: {latest_result['uncached_selected']}")
        print(f"Cached selected:   {latest_result['cached_selected']}")
    
    print()
    print("Recent-date match details:")
    for result in all_results:
        date_str = str(pd.Timestamp(result['date']).date())
        exact_str = f"exact={result['exact_match']}"
        set_str = f"set={result['set_match']}"
        
        # Highlight errors
        if not result['exact_match']:
            exact_str = f"ERROR: exact={result['exact_match']}"
        if not result['set_match']:
            set_str = f"ERROR: set={result['set_match']}"
        
        print(
            f"  {date_str}: {exact_str} {set_str} "
            f"uncached={result['uncached_time']:.2f}s cached={result['cached_time']:.2f}s"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generic timing benchmark for cached strategy selectors")
    parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        choices=SUPPORTED_STRATEGIES,
        help="Strategy key to benchmark. Repeat to benchmark multiple strategies. Defaults to all supported strategies.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_MATCH_DAYS,
        help="Number of recent dates to check for cached vs uncached selection matches.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
        help="Selection size to benchmark. Defaults to the backtesting selection width.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported strategy keys and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Supported strategies:")
        for strategy in SUPPORTED_STRATEGIES:
            print(f"  {strategy}")
        return

    strategies = args.strategies or SUPPORTED_STRATEGIES
    match_days = max(1, args.days)

    grouped, top_tickers, current_date, available_dates, load_time, selection_time = _prepare_universe()

    print("Building price history cache...")
    cache_start = time.perf_counter()
    price_history_cache = build_price_history_cache(grouped)
    cache_time = time.perf_counter() - cache_start

    test_dates = available_dates[-match_days:]
    print(f"Using {len(test_dates)} recent dates for match checks")

    for strategy_name in strategies:
        label = STRATEGY_REGISTRY[strategy_name]["label"]
        print()
        print(f"Running benchmark for {label}...")

        latest_result = _run_single_check(
            strategy_name,
            top_tickers,
            grouped,
            args.top_n,
            current_date,
            price_history_cache,
        )
        all_results = [
            _run_single_check(strategy_name, top_tickers, grouped, args.top_n, test_date, price_history_cache)
            for test_date in test_dates
        ]

        _print_strategy_summary(
            strategy_name,
            label,
            latest_result,
            all_results,
            cache_time,
            len(top_tickers),
            current_date,
            load_time,
            selection_time,
        )


if __name__ == "__main__":
    main()
