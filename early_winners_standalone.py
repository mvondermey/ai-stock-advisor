from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

@dataclass
class FeatureSet:
    close: pd.DataFrame
    volume: pd.DataFrame
    coverage: pd.Series
    daily_returns: pd.DataFrame
    log_2d: pd.DataFrame
    log_5d: pd.DataFrame
    log_10d: pd.DataFrame
    log_1m: pd.DataFrame
    log_3m: pd.DataFrame
    log_6m: pd.DataFrame
    log_1y: pd.DataFrame
    sma20: pd.DataFrame
    sma30: pd.DataFrame
    sma50: pd.DataFrame
    sma75: pd.DataFrame
    sma100: pd.DataFrame
    sma150: pd.DataFrame
    sma200: pd.DataFrame
    slope20: pd.DataFrame
    slope50: pd.DataFrame
    vol20: pd.DataFrame
    dollar_vol20: pd.DataFrame
    trend_r2_60: pd.DataFrame

@dataclass
class BacktestResult:
    mode: str
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    rebalance_count: int
    executed_rebalances: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    last_nonempty_selection_date: pd.Timestamp | None
    last_nonempty_selection: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance-frequency", choices=["weekly", "monthly"], default="weekly")
    parser.add_argument("--min-universe-size", type=int, default=1500)
    parser.add_argument("--min-history-days", type=int, default=220)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=2_000_000.0)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default="early_winners,top_3m,accel_1m_3m",
        help="Comma-separated strategy modes to test.",
    )
    return parser.parse_args()


def load_daily_cache_file(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, usecols=["Datetime", "Close", "Volume"])
    except ValueError as exc:
        print(f"⚠️ Could not read expected columns from {path.name}: {exc}")
        return None
    except Exception as exc:
        print(f"⚠️ Could not read {path.name}: {exc}")
        return None

    if frame.empty:
        return None

    frame["Datetime"] = pd.to_datetime(frame["Datetime"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["Datetime", "Close"]).sort_values("Datetime")
    if frame.empty:
        return None

    frame = frame.set_index("Datetime")
    daily = pd.DataFrame(
        {
            "Close": frame["Close"].resample("1D").last(),
            "Volume": frame["Volume"].resample("1D").sum(),
        }
    ).dropna(subset=["Close"])

    if daily.empty:
        return None

    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    else:
        daily.index = daily.index.tz_convert("UTC")

    return daily.sort_index()


def load_market_data(cache_dir: Path, min_history_days: int, max_tickers: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_files = sorted(cache_dir.glob("*.csv"))
    if max_tickers is not None:
        csv_files = csv_files[:max_tickers]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")

    close_by_ticker: Dict[str, pd.Series] = {}
    volume_by_ticker: Dict[str, pd.Series] = {}
    skipped_short = 0

    for path in csv_files:
        daily = load_daily_cache_file(path)
        if daily is None:
            continue
        if len(daily) < min_history_days:
            skipped_short += 1
            continue
        ticker = path.stem
        close_by_ticker[ticker] = daily["Close"]
        volume_by_ticker[ticker] = daily["Volume"]

    if not close_by_ticker:
        raise RuntimeError("No usable ticker history was loaded from cache")

    close = pd.DataFrame(close_by_ticker).sort_index()
    volume = pd.DataFrame(volume_by_ticker).reindex(close.index)
    print(
        f"Loaded {len(close.columns)} tickers from {cache_dir} "
        f"({skipped_short} skipped for short history)"
    )
    print(f"Raw daily matrix shape: {close.shape}")
    return close, volume


def compute_trend_r2(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_var = float((x_centered ** 2).sum())

    for ticker in close.columns:
        values = np.log(close[ticker].replace(0, np.nan)).to_numpy(dtype=float)
        scores = np.full(len(values), np.nan, dtype=float)
        for end_idx in range(window - 1, len(values)):
            window_values = values[end_idx - window + 1:end_idx + 1]
            if np.isnan(window_values).any():
                continue
            y_centered = window_values - window_values.mean()
            slope = float((x_centered * y_centered).sum() / x_var)
            fitted = window_values.mean() + slope * x_centered
            residual = float(((window_values - fitted) ** 2).sum())
            total = float(((window_values - window_values.mean()) ** 2).sum())
            if total <= 0:
                continue
            scores[end_idx] = max(0.0, 1.0 - residual / total)
        result[ticker] = scores

    return result


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_universe_size: int,
) -> FeatureSet:
    coverage = close.notna().sum(axis=1)
    valid_dates = coverage[coverage >= min_universe_size].index
    if len(valid_dates) == 0:
        raise RuntimeError(
            f"No dates found with at least {min_universe_size} tickers of coverage"
        )

    filtered_close = close.loc[valid_dates]
    filtered_volume = volume.loc[valid_dates]
    daily_returns = filtered_close.pct_change()

    log_2d = np.log(filtered_close / filtered_close.shift(2))
    log_5d = np.log(filtered_close / filtered_close.shift(5))
    log_10d = np.log(filtered_close / filtered_close.shift(10))
    log_1m = np.log(filtered_close / filtered_close.shift(21)) * (252.0 / 21.0)
    log_3m = np.log(filtered_close / filtered_close.shift(63)) * (252.0 / 63.0)
    log_6m = np.log(filtered_close / filtered_close.shift(126)) * (252.0 / 126.0)
    log_1y = np.log(filtered_close / filtered_close.shift(252))

    sma20 = filtered_close.rolling(20).mean()
    sma30 = filtered_close.rolling(30).mean()
    sma50 = filtered_close.rolling(50).mean()
    sma75 = filtered_close.rolling(75).mean()
    sma100 = filtered_close.rolling(100).mean()
    sma150 = filtered_close.rolling(150).mean()
    sma200 = filtered_close.rolling(200).mean()
    slope20 = sma20 / sma20.shift(10) - 1.0
    slope50 = sma50 / sma50.shift(20) - 1.0
    vol20 = daily_returns.rolling(20).std() * np.sqrt(252.0)
    dollar_vol20 = (filtered_close * filtered_volume).rolling(20).mean()
    trend_r2_60 = compute_trend_r2(filtered_close, window=60)

    print(
        f"Coverage-filtered range: {filtered_close.index[0].date()} -> {filtered_close.index[-1].date()} "
        f"({len(filtered_close)} dates)"
    )

    return FeatureSet(
        close=filtered_close,
        volume=filtered_volume,
        coverage=coverage.loc[valid_dates],
        daily_returns=daily_returns,
        log_2d=log_2d,
        log_5d=log_5d,
        log_10d=log_10d,
        log_1m=log_1m,
        log_3m=log_3m,
        log_6m=log_6m,
        log_1y=log_1y,
        sma20=sma20,
        sma30=sma30,
        sma50=sma50,
        sma75=sma75,
        sma100=sma100,
        sma150=sma150,
        sma200=sma200,
        slope20=slope20,
        slope50=slope50,
        vol20=vol20,
        dollar_vol20=dollar_vol20,
        trend_r2_60=trend_r2_60,
    )


def get_rebalance_dates(features: FeatureSet, frequency: str, min_history_days: int) -> list[pd.Timestamp]:
    eligible_index = features.close.index[min_history_days:]
    if len(eligible_index) == 0:
        return []

    if frequency == "weekly":
        grouped = pd.Series(eligible_index, index=eligible_index).groupby(
            [eligible_index.year, eligible_index.isocalendar().week]
        )
    else:
        grouped = pd.Series(eligible_index, index=eligible_index).groupby(
            [eligible_index.year, eligible_index.month]
        )

    return [group.iloc[-1] for _, group in grouped]


def percentile_rank(series: pd.Series, higher_better: bool = True) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return valid
    return valid.rank(pct=True, ascending=higher_better)


def build_snapshot(features: FeatureSet, date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": features.close.loc[date],
            "log_2d": features.log_2d.loc[date],
            "log_5d": features.log_5d.loc[date],
            "log_10d": features.log_10d.loc[date],
            "log_1m": features.log_1m.loc[date],
            "log_3m": features.log_3m.loc[date],
            "log_6m": features.log_6m.loc[date],
            "log_1y": features.log_1y.loc[date],
            "sma20": features.sma20.loc[date],
            "sma50": features.sma50.loc[date],
            "sma200": features.sma200.loc[date],
            "slope20": features.slope20.loc[date],
            "slope50": features.slope50.loc[date],
            "vol20": features.vol20.loc[date],
            "dollar_vol20": features.dollar_vol20.loc[date],
            "trend_r2_60": features.trend_r2_60.loc[date],
        }
    ).dropna()


def select_tickers(
    features: FeatureSet,
    date: pd.Timestamp,
    mode: str,
    top_n: int,
    min_price: float,
    min_dollar_volume: float,
) -> list[str]:
    frame = build_snapshot(features, date)
    if frame.empty:
        return []

    frame = frame[
        (frame["close"] > min_price)
        & (frame["dollar_vol20"] > min_dollar_volume)
    ].copy()
    if frame.empty:
        return []

    if mode == "top_3m":
        frame = frame[
            (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
        ]
        if frame.empty:
            return []
        ranked = frame.sort_values("log_3m", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "accel_1m_3m":
        frame["accel_1m_3m"] = frame["log_1m"] - frame["log_3m"]
        frame = frame[
            (frame["log_1m"] > 0)
            & (frame["close"] > frame["sma20"])
        ]
        if frame.empty:
            return []
        ranked = frame.sort_values("accel_1m_3m", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "early_winners_6m":
        frame = frame[
            (frame["log_1m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_6m", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "top_1y":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "bh_1y_rank_sell":
        # Pure 1Y momentum ranking -- exit logic handled in run_backtest.
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Rebalancing variants (use same selection as bh_1y_rank_sell)
    if mode in (
        "rebal_drawdown_exit",
        "rebal_position_sizing",
        "bh_1y_sma50_daily",
        "bh_1y_sma200_daily",
        "bh_1y_sma50_persist",
        "bh_1y_sma50_rank",
        "bh_1y_atr_stop",
        "bh_1y_sma50_persist3",
        "bh_1y_sma50_persist5",
        "bh_1y_persist_possize",
        "bh_1y_persist_rankgate",
        "bh_1y_persist_dd",
        "bh_1y_sma30_persist",
        "bh_1y_sma75_persist",
        "bh_1y_sma100_persist",
        "bh_1y_sma150_persist",
        "bh_1y_sma75_persist3",
        "bh_1y_sma75_persist_dd",
    ):
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Timeframe variants
    if mode == "sel_6m":
        frame = frame[
            (frame["log_6m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_6m", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "sel_3m":
        frame = frame[
            (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_3m", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "sel_9m":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        # Approximate 9M as average of 1Y and 6M
        frame["log_9m_approx"] = (frame["log_1y"] + frame["log_6m"]) / 2
        ranked = frame.sort_values("log_9m_approx", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Blend variants
    if mode == "blend_60_40":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_3m = percentile_rank(frame["log_3m"])
        frame["score"] = 0.6 * rank_1y + 0.4 * rank_3m
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "blend_80_20":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_3m = percentile_rank(frame["log_3m"])
        frame["score"] = 0.8 * rank_1y + 0.2 * rank_3m
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "blend_40_60":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_3m = percentile_rank(frame["log_3m"])
        frame["score"] = 0.4 * rank_1y + 0.6 * rank_3m
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Fresh momentum blends
    if mode == "blend_6m_3m":
        frame = frame[
            (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_6m = percentile_rank(frame["log_6m"])
        rank_3m = percentile_rank(frame["log_3m"])
        frame["score"] = 0.5 * rank_6m + 0.5 * rank_3m
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "blend_1y_6m":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_6m = percentile_rank(frame["log_6m"])
        frame["score"] = 0.5 * rank_1y + 0.5 * rank_6m
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "blend_1y_3m_6m":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_3m = percentile_rank(frame["log_3m"])
        rank_6m = percentile_rank(frame["log_6m"])
        frame["score"] = (rank_1y + rank_3m + rank_6m) / 3
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Acceleration-based
    if mode == "accel_only":
        frame["accel_3m_6m"] = frame["log_3m"] - frame["log_6m"]
        frame["accel_6m_1y"] = frame["log_6m"] - frame["log_1y"]
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["accel_3m_6m"] > 0)
            & (frame["accel_6m_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode == "decel_filter":
        frame["accel_3m_6m"] = frame["log_3m"] - frame["log_6m"]
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["accel_3m_6m"] >= 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Trend + Momentum
    if mode == "trend_1y":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
            & (frame["trend_r2_60"] > 0.5)
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    # ---- Short-term momentum selection (hot recent movers) ----

    # Pure 2-day return ranking
    if mode == "sel_2d":
        frame = frame[
            (frame["log_2d"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_2d", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Pure 5-day return ranking
    if mode == "sel_5d":
        frame = frame[
            (frame["log_5d"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_5d", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Pure 10-day return ranking
    if mode == "sel_10d":
        frame = frame[
            (frame["log_10d"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        ranked = frame.sort_values("log_10d", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Hybrid: 1Y filter, then rank by 5-day return (recent acceleration within winners)
    if mode == "bh_1y_top5d":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_5d"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        # Take top 30% by 1Y, then sort by 5-day return
        frame["rank_1y"] = percentile_rank(frame["log_1y"])
        frame = frame[frame["rank_1y"] >= 0.70]
        if frame.empty:
            return []
        ranked = frame.sort_values("log_5d", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Blend: 50/50 1Y rank + 5-day rank
    if mode == "blend_1y_5d":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_5d = percentile_rank(frame["log_5d"])
        frame["score"] = 0.5 * rank_1y + 0.5 * rank_5d
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # 1Y selection with 5-day tie-breaker (top 30 by 1Y, sorted by 5-day)
    if mode in ("bh_1y_5d_tiebreak", "bh_1y_5d_exit"):
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        # Take top 30 by 1Y, then sort by 5-day return
        top30 = frame.sort_values("log_1y", ascending=False).head(30)
        ranked = top30.sort_values("log_5d", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Blend: 70/30 1Y + 5-day (1Y dominant, 5d as freshness signal)
    if mode in ("blend_1y_5d_70_30", "blend_1y_5d_smart", "blend_1y_5d_persist"):
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_5d = percentile_rank(frame["log_5d"])
        frame["score"] = 0.7 * rank_1y + 0.3 * rank_5d
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # ---- New strategies inspired by Perfect-Foresight gap analysis ----

    # Emerging momentum: stocks NOT yet in top 1Y, but with strong 3M + 1M acceleration.
    # Idea: catch winners earlier, before they're at the top of the 1Y leaderboard.
    if mode == "emerging_mom":
        frame = frame[
            (frame["log_3m"] > 0)
            & (frame["log_1m"] > 0)
            & (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])  # higher = better
        # Keep mid-pack on 1Y (not the over-extended top 5%, not laggards)
        frame["rank_1y_pct"] = rank_1y
        frame = frame[
            (frame["rank_1y_pct"] >= 0.50)
            & (frame["rank_1y_pct"] <= 0.92)
        ]
        if frame.empty:
            return []
        # Score by recent acceleration: 3M heavy, 1M secondary
        frame["score"] = 0.6 * percentile_rank(frame["log_3m"]) + 0.4 * percentile_rank(frame["log_1m"])
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Turnaround: oversold names breaking out.
    # Down >25% from 200d-equivalent peak (sma200 << close 6M ago) but recent 3M strong.
    if mode == "turnaround":
        # Estimate drawdown via 1Y log return being negative or modest, with 3M acceleration positive
        frame = frame[
            (frame["log_1y"] < 0.50)  # didn't already run up
            & (frame["log_3m"] > 0.30)  # strong 3M (annualized)
            & (frame["log_1m"] > 0)
            & (frame["close"] > frame["sma50"])
        ].copy()
        if frame.empty:
            return []
        # Score by 3M momentum, prefer best recent acceleration
        frame["score"] = percentile_rank(frame["log_3m"]) + 0.5 * percentile_rank(frame["log_1m"])
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Rank drift: select names whose 3M rank has improved most vs their 1Y rank.
    # Captures relative-strength inflection (rising stars).
    if mode == "rank_drift":
        frame = frame[
            (frame["log_3m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_3m = percentile_rank(frame["log_3m"])
        # Drift = how much 3M rank exceeds 1Y rank (positive = accelerating relative to peers)
        frame["drift"] = rank_3m - rank_1y
        # Combined: prefer high 3M rank AND positive drift
        frame["score"] = 0.6 * rank_3m + 0.4 * frame["drift"]
        # Require positive drift (relative strength improving)
        frame = frame[frame["drift"] > 0]
        if frame.empty:
            return []
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Mid-cap momentum: avoid the largest dollar-volume names (proxy for mega caps).
    # Top 1Y momentum but exclude top 10% by dollar volume.
    if mode == "mid_cap_mom":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        dv_pct = percentile_rank(frame["dollar_vol20"])
        # Keep names below 90th percentile of dollar volume (skip mega caps)
        frame = frame[dv_pct < 0.90]
        if frame.empty:
            return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    if mode != "early_winners":
        raise ValueError(f"Unknown mode: {mode}")

    # Design goal: beat pure `top_1y` by combining established 1Y strength
    # with current 3M acceleration. Established winners that are still
    # accelerating outperform both pure 1Y leaders and short-term movers.
    frame = frame[
        (frame["log_1y"] > 0)
        & (frame["log_3m"] > 0)
        & (frame["close"] > frame["sma200"])
    ].copy()
    if frame.empty:
        return []

    rank_1y = percentile_rank(frame["log_1y"])
    rank_3m = percentile_rank(frame["log_3m"])
    frame["score"] = 0.5 * rank_1y + 0.5 * rank_3m
    ranked = frame.sort_values("score", ascending=False)
    return ranked.head(top_n).index.tolist()


def max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def run_backtest(
    features: FeatureSet,
    rebalance_dates: Iterable[pd.Timestamp],
    mode: str,
    top_n: int,
    transaction_cost_bps: float,
    min_price: float,
    min_dollar_volume: float,
) -> BacktestResult:
    rebalance_list = list(rebalance_dates)
    rebalance_set = set(rebalance_list)
    weights: Dict[str, float] = {}
    equity = 1.0
    equity_points: list[tuple[pd.Timestamp, float]] = []
    turnover_sum = 0.0
    executed_rebalances = 0
    last_nonempty_selection_date: pd.Timestamp | None = None
    last_nonempty_selection: list[str] = []
    entry_ranks: Dict[str, int] = {}
    entry_dates: Dict[str, pd.Timestamp] = {}  # For drawdown_exit
    entry_prices: Dict[str, float] = {}  # For drawdown_exit
    peak_prices: Dict[str, float] = {}  # For drawdown_exit
    smart_modes = {
        "bh_1y_rank_sell",
        "rebal_drawdown_exit",
        "rebal_position_sizing",
        "bh_1y_sma50_daily",
        "bh_1y_sma200_daily",
        "bh_1y_sma50_persist",
        "bh_1y_sma50_rank",
        "bh_1y_atr_stop",
        "bh_1y_sma50_persist3",
        "bh_1y_sma50_persist5",
        "bh_1y_persist_possize",
        "bh_1y_persist_rankgate",
        "bh_1y_persist_dd",
        "bh_1y_sma30_persist",
        "bh_1y_sma75_persist",
        "bh_1y_sma100_persist",
        "bh_1y_sma150_persist",
        "bh_1y_sma75_persist3",
        "bh_1y_sma75_persist_dd",
        "blend_1y_5d_smart",
        "blend_1y_5d_persist",
        "bh_1y_5d_tiebreak",
        "bh_1y_5d_exit",
    }
    sma_breach_streak: Dict[str, int] = {}  # for persist variants
    neg_5d_streak: Dict[str, int] = {}  # for bh_1y_5d_exit

    def apply_rebalance(date: pd.Timestamp, picks: list[str]) -> None:
        nonlocal weights, equity, turnover_sum, executed_rebalances
        nonlocal last_nonempty_selection_date, last_nonempty_selection
        if not picks:
            return
        
        # Position sizing by rank strength for rebal_position_sizing mode
        if mode in ("rebal_position_sizing", "bh_1y_persist_possize"):
            # Weight by rank: top gets more weight, linear decay
            rank_weights = {}
            for i, ticker in enumerate(picks):
                rank_weights[ticker] = (len(picks) - i) / sum(range(1, len(picks) + 1))
            new_weights = rank_weights
        else:
            new_weights = {ticker: 1.0 / len(picks) for ticker in picks}
        
        turnover = sum(
            abs(new_weights.get(ticker, 0.0) - weights.get(ticker, 0.0))
            for ticker in set(new_weights) | set(weights)
        )
        turnover_sum += turnover
        equity *= 1.0 - turnover * (transaction_cost_bps / 10_000.0)
        weights = new_weights
        executed_rebalances += 1
        last_nonempty_selection_date = date
        last_nonempty_selection = picks
        # Track entry metadata for rebalancing variants
        for ticker in picks:
            if ticker not in entry_dates:
                entry_dates[ticker] = date
                entry_prices[ticker] = float(features.close.loc[date, ticker])
                peak_prices[ticker] = entry_prices[ticker]

    for date in features.close.index:
        if weights:
            daily_move = features.daily_returns.loc[date, list(weights.keys())].dropna()
            if not daily_move.empty:
                equity *= 1.0 + float((daily_move * pd.Series(weights)).sum())
            # Update peak prices for drawdown_exit
            for ticker in weights.keys():
                if ticker in features.close.columns:
                    current_price = float(features.close.loc[date, ticker])
                    if ticker in peak_prices:
                        peak_prices[ticker] = max(peak_prices[ticker], current_price)

        # ---------- Daily exit checks for SMA / ATR variants ----------
        daily_exit_modes = {
            "bh_1y_sma50_daily",
            "bh_1y_sma200_daily",
            "bh_1y_sma50_persist",
            "bh_1y_sma50_rank",
            "bh_1y_atr_stop",
            "bh_1y_sma50_persist3",
            "bh_1y_sma50_persist5",
            "bh_1y_persist_possize",
            "bh_1y_persist_rankgate",
            "bh_1y_persist_dd",
            "bh_1y_sma30_persist",
            "bh_1y_sma75_persist",
            "bh_1y_sma100_persist",
            "bh_1y_sma150_persist",
            "bh_1y_sma75_persist3",
            "bh_1y_sma75_persist_dd",
            "blend_1y_5d_persist",
        }
        # All persist-style variants need 2-day or longer breach streak
        persist_threshold = {
            "bh_1y_sma50_persist": 2,
            "bh_1y_sma50_persist3": 3,
            "bh_1y_sma50_persist5": 5,
            "bh_1y_persist_possize": 2,
            "bh_1y_persist_rankgate": 2,
            "bh_1y_persist_dd": 2,
            "bh_1y_sma30_persist": 2,
            "bh_1y_sma75_persist": 2,
            "bh_1y_sma100_persist": 2,
            "bh_1y_sma150_persist": 2,
            "bh_1y_sma75_persist3": 3,
            "bh_1y_sma75_persist_dd": 2,
            "blend_1y_5d_persist": 2,
        }
        # Which SMA each persist variant uses
        persist_sma = {
            "bh_1y_sma50_persist": "sma50",
            "bh_1y_sma50_persist3": "sma50",
            "bh_1y_sma50_persist5": "sma50",
            "bh_1y_persist_possize": "sma50",
            "bh_1y_persist_rankgate": "sma50",
            "bh_1y_persist_dd": "sma50",
            "bh_1y_sma30_persist": "sma30",
            "bh_1y_sma75_persist": "sma75",
            "bh_1y_sma100_persist": "sma100",
            "bh_1y_sma150_persist": "sma150",
            "bh_1y_sma75_persist3": "sma75",
            "bh_1y_sma75_persist_dd": "sma75",
            "blend_1y_5d_persist": "sma75",
        }
        dd_modes = {"bh_1y_persist_dd", "bh_1y_sma75_persist_dd"}
        # ---------- Daily 5-day return exit (bh_1y_5d_exit) ----------
        if mode == "bh_1y_5d_exit" and weights:
            tickers_to_exit_5d: list[str] = []
            for ticker in list(weights.keys()):
                try:
                    log5d = features.log_5d.loc[date, ticker]
                    if pd.isna(log5d):
                        continue
                    if log5d < 0:
                        neg_5d_streak[ticker] = neg_5d_streak.get(ticker, 0) + 1
                    else:
                        neg_5d_streak[ticker] = 0
                    if neg_5d_streak.get(ticker, 0) >= 2:
                        tickers_to_exit_5d.append(ticker)
                except (KeyError, ValueError):
                    continue
            if tickers_to_exit_5d:
                candidates_5d = select_tickers(
                    features=features, date=date, mode=mode, top_n=top_n * 5,
                    min_price=min_price, min_dollar_volume=min_dollar_volume,
                )
                kept_5d = [t for t in weights.keys() if t not in tickers_to_exit_5d]
                replacements_5d = [t for t in candidates_5d if t not in kept_5d][: top_n - len(kept_5d)]
                final_5d = kept_5d + replacements_5d
                for t in tickers_to_exit_5d:
                    entry_ranks.pop(t, None)
                    entry_dates.pop(t, None)
                    entry_prices.pop(t, None)
                    peak_prices.pop(t, None)
                    neg_5d_streak.pop(t, None)
                for t in replacements_5d:
                    if t in candidates_5d:
                        entry_ranks[t] = candidates_5d.index(t) + 1
                apply_rebalance(date, final_5d)

        if mode in daily_exit_modes and weights:
            tickers_to_exit: list[str] = []
            # Pre-compute current rank map for rank-gated variants
            current_rank_map: Dict[str, int] = {}
            if mode in ("bh_1y_sma50_rank", "bh_1y_persist_rankgate"):
                rank_candidates = select_tickers(
                    features=features, date=date, mode=mode, top_n=top_n * 5,
                    min_price=min_price, min_dollar_volume=min_dollar_volume,
                )
                for i, t in enumerate(rank_candidates):
                    current_rank_map[t] = i + 1
            for ticker in list(weights.keys()):
                try:
                    close_price = features.close.loc[date, ticker]
                    if pd.isna(close_price):
                        continue

                    breach = False
                    if mode == "bh_1y_sma50_daily":
                        sma_val = features.sma50.loc[date, ticker]
                        breach = pd.notna(sma_val) and close_price < sma_val
                    elif mode == "bh_1y_sma200_daily":
                        sma_val = features.sma200.loc[date, ticker]
                        breach = pd.notna(sma_val) and close_price < sma_val
                    elif mode in persist_threshold:
                        sma_attr = persist_sma[mode]
                        sma_val = getattr(features, sma_attr).loc[date, ticker]
                        is_breach = pd.notna(sma_val) and close_price < sma_val
                        if is_breach:
                            sma_breach_streak[ticker] = sma_breach_streak.get(ticker, 0) + 1
                        else:
                            sma_breach_streak[ticker] = 0
                        threshold = persist_threshold[mode]
                        persist_hit = sma_breach_streak.get(ticker, 0) >= threshold
                        if mode == "bh_1y_persist_rankgate":
                            # Only trigger if also rank dropped out of top 15
                            rank_breach = current_rank_map.get(ticker, 999) > 15
                            breach = persist_hit and rank_breach
                        elif mode in dd_modes:
                            # Trigger on either persist OR 20% drawdown from peak
                            dd_hit = False
                            if ticker in peak_prices:
                                drawdown = (close_price - peak_prices[ticker]) / peak_prices[ticker]
                                dd_hit = drawdown < -0.20
                            breach = persist_hit or dd_hit
                        else:
                            breach = persist_hit
                    elif mode == "bh_1y_sma50_rank":
                        sma_val = features.sma50.loc[date, ticker]
                        sma_breach = pd.notna(sma_val) and close_price < sma_val
                        rank_breach = current_rank_map.get(ticker, 999) > 15
                        breach = sma_breach and rank_breach
                    elif mode == "bh_1y_atr_stop":
                        # Use vol20 (annualized stdev) as ATR proxy.
                        # Stop = entry_price * (1 - 2 * vol20_daily)
                        # vol20 is annualized; daily ~= vol20 / sqrt(252)
                        vol_ann = features.vol20.loc[date, ticker]
                        if pd.notna(vol_ann) and ticker in entry_prices:
                            daily_vol = float(vol_ann) / np.sqrt(252.0)
                            # 2x daily-vol drawdown over ~20 days = ~9% threshold
                            # Use ratio drawdown from peak vs 2 * 20-day vol
                            peak = peak_prices.get(ticker, entry_prices[ticker])
                            drawdown = (close_price - peak) / peak
                            stop_threshold = -2.0 * daily_vol * np.sqrt(20.0)
                            breach = drawdown < stop_threshold

                    if breach:
                        tickers_to_exit.append(ticker)
                except (KeyError, ValueError):
                    continue
            if tickers_to_exit:
                candidates_daily = select_tickers(
                    features=features, date=date, mode=mode, top_n=top_n * 5,
                    min_price=min_price, min_dollar_volume=min_dollar_volume,
                )
                kept = [t for t in weights.keys() if t not in tickers_to_exit]
                replacements = [t for t in candidates_daily if t not in kept][: top_n - len(kept)]
                final = kept + replacements
                for ticker in tickers_to_exit:
                    entry_ranks.pop(ticker, None)
                    entry_dates.pop(ticker, None)
                    entry_prices.pop(ticker, None)
                    peak_prices.pop(ticker, None)
                    sma_breach_streak.pop(ticker, None)
                for ticker in replacements:
                    if ticker in candidates_daily:
                        entry_ranks[ticker] = candidates_daily.index(ticker) + 1
                apply_rebalance(date, final)

        if date in rebalance_set:
            candidates = select_tickers(
                features=features, date=date, mode=mode, top_n=top_n * 5 if mode in smart_modes else top_n,
                min_price=min_price, min_dollar_volume=min_dollar_volume,
            )
            if mode in smart_modes and weights:
                # Smart rebalance: keep current holdings unless rank drops too far
                positions_to_sell: list[str] = []
                for ticker in list(weights.keys()):
                    try:
                        current_rank = candidates.index(ticker) + 1
                    except ValueError:
                        current_rank = 999
                    entry_rank = entry_ranks.get(ticker, current_rank)
                    rank_drop = current_rank - entry_rank
                    should_sell_rank_based = current_rank > 15 or rank_drop >= 5

                    # Variant-specific logic
                    if mode == "rebal_drawdown_exit":
                        # Exit if >20% drawdown from peak OR rank violation
                        if ticker in peak_prices and ticker in features.close.columns:
                            current_price = float(features.close.loc[date, ticker])
                            drawdown = (current_price - peak_prices[ticker]) / peak_prices[ticker]
                            if should_sell_rank_based or drawdown < -0.20:
                                positions_to_sell.append(ticker)
                    else:
                        # Standard bh_1y_rank_sell and rebal_position_sizing logic
                        if should_sell_rank_based:
                            positions_to_sell.append(ticker)

                kept = [t for t in weights.keys() if t not in positions_to_sell]
                replacements = [t for t in candidates if t not in kept][: top_n - len(kept)]
                final = kept + replacements
                for ticker in positions_to_sell:
                    entry_ranks.pop(ticker, None)
                    entry_dates.pop(ticker, None)
                    entry_prices.pop(ticker, None)
                    peak_prices.pop(ticker, None)
                for ticker in replacements:
                    if ticker in candidates:
                        entry_ranks[ticker] = candidates.index(ticker) + 1
                apply_rebalance(date, final)
            else:
                picks = candidates[:top_n] if mode in smart_modes else candidates
                if mode in smart_modes:
                    for i, ticker in enumerate(picks):
                        entry_ranks[ticker] = i + 1
                apply_rebalance(date, picks)

        equity_points.append((date, equity))

    equity_curve = pd.Series(dict(equity_points)).sort_index()
    returns = equity_curve.pct_change().dropna()
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = float(equity_curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    annual_volatility = float(returns.std() * np.sqrt(252.0)) if not returns.empty else float("nan")
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252.0)) if returns.std() > 0 else float("nan")

    return BacktestResult(
        mode=mode,
        total_return=float(equity_curve.iloc[-1] - 1.0),
        cagr=cagr,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown(equity_curve),
        avg_turnover=float(turnover_sum / max(len(rebalance_list), 1)),
        rebalance_count=len(rebalance_list),
        executed_rebalances=executed_rebalances,
        start_date=equity_curve.index[0],
        end_date=equity_curve.index[-1],
        last_nonempty_selection_date=last_nonempty_selection_date,
        last_nonempty_selection=last_nonempty_selection,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_results(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "mode": result.mode,
                "start": result.start_date.date().isoformat(),
                "end": result.end_date.date().isoformat(),
                "total_return": format_pct(result.total_return),
                "cagr": format_pct(result.cagr),
                "ann_vol": format_pct(result.annual_volatility),
                "sharpe": f"{result.sharpe:.2f}",
                "max_drawdown": format_pct(result.max_drawdown),
                "avg_turnover": f"{result.avg_turnover:.3f}",
                "rebalance_count": result.rebalance_count,
                "executed_rebalances": result.executed_rebalances,
                "last_nonempty_selection_date": (
                    result.last_nonempty_selection_date.date().isoformat()
                    if result.last_nonempty_selection_date is not None
                    else ""
                ),
                "last_nonempty_selection": ", ".join(result.last_nonempty_selection[:10]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    close, volume = load_market_data(
        cache_dir=args.data_cache_dir,
        min_history_days=args.min_history_days,
        max_tickers=args.max_tickers,
    )
    features = build_features(
        close=close,
        volume=volume,
        min_universe_size=args.min_universe_size,
    )
    rebalance_dates = get_rebalance_dates(
        features=features,
        frequency=args.rebalance_frequency,
        min_history_days=args.min_history_days,
    )
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates available after applying warmup")

    print(
        f"Rebalance frequency: {args.rebalance_frequency} "
        f"({len(rebalance_dates)} dates, last={rebalance_dates[-1].date().isoformat()})"
    )

    results = []
    for mode in modes:
        print(f"Running mode: {mode}")
        result = run_backtest(
            features=features,
            rebalance_dates=rebalance_dates,
            mode=mode,
            top_n=args.top_n,
            transaction_cost_bps=args.transaction_cost_bps,
            min_price=args.min_price,
            min_dollar_volume=args.min_dollar_volume,
        )
        results.append(result)

    summary = format_results(results)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
