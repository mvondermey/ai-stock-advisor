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
    intra_5h: pd.DataFrame

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
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=None,
        help="If set, run only the last N trading days ending at --backtest-end-date (or latest date).",
    )
    parser.add_argument(
        "--backtest-end-date",
        type=str,
        default=None,
        help="Optional backtest window end date in YYYY-MM-DD (UTC).",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--rebalance-frequency",
        choices=[
            "daily",
            "every2",
            "every3",
            "every4",
            "every5",
            "every10",
            "weekly",
            "weekly_mon",
            "weekly_tue",
            "weekly_wed",
            "weekly_thu",
            "weekly_fri",
            "biweekly",
            "monthly",
        ],
        default="weekly",
    )
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
    # Intraday: per session, return from 5 bars before last bar to last bar (≈last-5h return)
    # Group by date in UTC; require >=6 bars in session.
    by_day = frame.groupby(frame.index.normalize())["Close"]
    intra_5h = by_day.apply(lambda s: (s.iloc[-1] / s.iloc[-6] - 1.0) if len(s) >= 6 else float("nan"))
    daily = pd.DataFrame(
        {
            "Close": frame["Close"].resample("1D").last(),
            "Volume": frame["Volume"].resample("1D").sum(),
        }
    ).dropna(subset=["Close"])
    if daily.empty:
        return None
    daily["intra_5h"] = intra_5h.reindex(daily.index)

    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    else:
        daily.index = daily.index.tz_convert("UTC")

    return daily.sort_index()


def load_market_data(cache_dir: Path, min_history_days: int, max_tickers: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_files = sorted(cache_dir.glob("*.csv"))
    if max_tickers is not None:
        csv_files = csv_files[:max_tickers]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cache_dir}")

    close_by_ticker: Dict[str, pd.Series] = {}
    volume_by_ticker: Dict[str, pd.Series] = {}
    intra_by_ticker: Dict[str, pd.Series] = {}
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
        if "intra_5h" in daily.columns:
            intra_by_ticker[ticker] = daily["intra_5h"]

    if not close_by_ticker:
        raise RuntimeError("No usable ticker history was loaded from cache")

    close = pd.DataFrame(close_by_ticker).sort_index()
    volume = pd.DataFrame(volume_by_ticker).reindex(close.index)
    intra = pd.DataFrame(intra_by_ticker).reindex(close.index) if intra_by_ticker else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    print(
        f"Loaded {len(close.columns)} tickers from {cache_dir} "
        f"({skipped_short} skipped for short history)"
    )
    print(f"Raw daily matrix shape: {close.shape}")
    return close, volume, intra


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
    intra: pd.DataFrame | None = None,
) -> FeatureSet:
    coverage = close.notna().sum(axis=1)
    valid_dates = coverage[coverage >= min_universe_size].index
    if len(valid_dates) == 0:
        raise RuntimeError(
            f"No dates found with at least {min_universe_size} tickers of coverage"
        )

    filtered_close = close.loc[valid_dates]
    filtered_volume = volume.loc[valid_dates]
    if intra is None:
        filtered_intra = pd.DataFrame(index=valid_dates, columns=filtered_close.columns, dtype=float)
    else:
        filtered_intra = intra.reindex(index=valid_dates, columns=filtered_close.columns)
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
        intra_5h=filtered_intra,
    )


def get_rebalance_dates(features: FeatureSet, frequency: str, min_history_days: int) -> list[pd.Timestamp]:
    eligible_index = features.close.index[min_history_days:]
    if len(eligible_index) == 0:
        return []

    if frequency == "daily":
        return list(eligible_index)
    if frequency.startswith("every"):
        step = int(frequency.replace("every", ""))
        dates = list(eligible_index[::step])
        if eligible_index[-1] not in dates:
            dates.append(eligible_index[-1])
        return dates
    if frequency.startswith("weekly_"):
        weekday = {
            "weekly_mon": 0,
            "weekly_tue": 1,
            "weekly_wed": 2,
            "weekly_thu": 3,
            "weekly_fri": 4,
        }[frequency]
        grouped = pd.Series(eligible_index, index=eligible_index).groupby(
            [eligible_index.year, eligible_index.isocalendar().week]
        )
        dates = []
        for _, group in grouped:
            weekday_matches = group[group.dt.weekday == weekday]
            dates.append(weekday_matches.iloc[0] if not weekday_matches.empty else group.iloc[-1])
        return dates
    if frequency == "weekly":
        grouped = pd.Series(eligible_index, index=eligible_index).groupby(
            [eligible_index.year, eligible_index.isocalendar().week]
        )
    elif frequency == "biweekly":
        iso = eligible_index.isocalendar()
        grouped = pd.Series(eligible_index, index=eligible_index).groupby(
            [eligible_index.year, iso.week // 2]
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
    snapshot = pd.DataFrame(
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
            "sma75": features.sma75.loc[date],
            "sma50": features.sma50.loc[date],
            "sma200": features.sma200.loc[date],
            "slope20": features.slope20.loc[date],
            "slope50": features.slope50.loc[date],
            "vol20": features.vol20.loc[date],
            "dollar_vol20": features.dollar_vol20.loc[date],
            "trend_r2_60": features.trend_r2_60.loc[date],
            "intra_5h": features.intra_5h.loc[date],
        }
    )
    # Fill NaN intra_5h with 0 so it doesn't cause rows to be dropped by dropna
    snapshot["intra_5h"] = snapshot["intra_5h"].fillna(0.0)
    return snapshot.dropna()


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
        "bh_1y_sma75_persist_lowvol",
        "bh_1y_sma75_persist_trend",
        "bh_1y_sma75_persist_quality",
        "bh_1y_sma75_persist_regime",
        "bh_1y_sma75_persist_invvol",
        "blend_90_10_sma75_persist",
        "blend_85_15_sma75_persist",
        "blend_80_20_sma75_persist",
        "blend_60_40_sma75_persist",
        "bh_1y_sma75_persist_exitbuf",
        "bh_1y_sma75_persist_minhold",
        "bh_1y_sma75_persist_slopegate",
        "bh_1y_sma75_persist_accel",
    ):
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        if mode == "bh_1y_sma75_persist_buffer":
            frame = frame[frame["close"] > frame["sma75"] * 1.01].copy()
            if frame.empty:
                return []
        if mode == "bh_1y_sma75_persist_lowvol":
            frame = frame[frame["slope50"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_low_vol = percentile_rank(-frame["vol20"])
            frame["score"] = 0.75 * rank_1y + 0.25 * rank_low_vol
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "bh_1y_sma75_persist_trend":
            frame = frame[
                (frame["slope50"] > 0)
                & (frame["trend_r2_60"] > 0.45)
            ].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_trend = percentile_rank(frame["trend_r2_60"])
            frame["score"] = 0.7 * rank_1y + 0.3 * rank_trend
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "bh_1y_sma75_persist_quality":
            frame = frame[
                (frame["log_3m"] > 0)
                & (frame["slope50"] > 0)
                & (frame["trend_r2_60"] > 0.35)
            ].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_3m = percentile_rank(frame["log_3m"])
            rank_trend = percentile_rank(frame["trend_r2_60"])
            rank_low_vol = percentile_rank(-frame["vol20"])
            frame["score"] = 0.45 * rank_1y + 0.25 * rank_3m + 0.20 * rank_trend + 0.10 * rank_low_vol
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "bh_1y_sma75_persist_regime":
            breadth = float((features.close.loc[date] > features.sma200.loc[date]).mean())
            if breadth < 0.55:
                return []
        if mode == "bh_1y_sma75_persist_invvol":
            frame = frame[frame["slope50"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_inv_vol = percentile_rank(-frame["vol20"])
            frame["score"] = 0.5 * rank_1y + 0.5 * rank_inv_vol
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "blend_90_10_sma75_persist":
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_3m = percentile_rank(frame["log_3m"])
            frame["score"] = 0.9 * rank_1y + 0.1 * rank_3m
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "blend_85_15_sma75_persist":
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_3m = percentile_rank(frame["log_3m"])
            frame["score"] = 0.85 * rank_1y + 0.15 * rank_3m
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "blend_80_20_sma75_persist":
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_3m = percentile_rank(frame["log_3m"])
            frame["score"] = 0.8 * rank_1y + 0.2 * rank_3m
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "blend_60_40_sma75_persist":
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
            rank_1y = percentile_rank(frame["log_1y"])
            rank_3m = percentile_rank(frame["log_3m"])
            frame["score"] = 0.6 * rank_1y + 0.4 * rank_3m
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        if mode == "bh_1y_sma75_persist_slopegate":
            frame = frame[frame["slope50"] > 0].copy()
            if frame.empty:
                return []
        if mode == "bh_1y_sma75_persist_accel":
            frame = frame[frame["log_1m"] > 0].copy()
            if frame.empty:
                return []
        ranked = frame.sort_values("log_1y", ascending=False)
        return ranked.head(top_n).index.tolist()

    # 6M-based variants (different ranking universe)
    blend_1y_6m_weights = {
        "blend_1y_6m_sma75_persist": 0.5,
        "blend_1y_6m_80_20_sma75_persist": 0.8,
        "blend_1y_6m_75_25_sma75_persist": 0.75,
        "blend_1y_6m_70_30_sma75_persist": 0.7,
        "blend_1y_6m_70_30_sma75_persist1": 0.7,
        "blend_1y_6m_70_30_sma75_persist3": 0.7,
        "blend_1y_6m_70_30_sma75_persist_rank20": 0.7,
        "blend_1y_6m_70_30_sma75_persist_rankdrop7": 0.7,
        "blend_1y_6m_70_30_sma75_persist_entrybuf": 0.7,
        "blend_1y_6m_70_30_sma75_persist_pos3m": 0.7,
        "blend_1y_6m_80_20_sma75_persist_pos3m": 0.8,
        "blend_1y_6m_75_25_sma75_persist_pos3m": 0.75,
        "blend_1y_6m_65_35_sma75_persist_pos3m": 0.65,
        "blend_1y_6m_60_40_sma75_persist_pos3m": 0.6,
        "blend_1y_6m_55_45_sma75_persist_pos3m": 0.55,
        "blend_1y_6m_50_50_sma75_persist_pos3m": 0.5,
        "blend_1y_6m_47_5_52_5_sma75_persist_pos3m": 0.475,
        "blend_1y_6m_45_55_sma75_persist_pos3m": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_full": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_rank10": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_accel": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_slope": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_possize": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_volexit": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_staggered": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth": 0.45,
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage": 0.4,
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp": 0.4,
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage": 0.4,
        "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage": 0.35,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop": 0.3,
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra": 0.3,
        "blend_1y_6m_45_55_sma75_persist_pos3m_rank20": 0.45,
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7": 0.45,
        "blend_1y_6m_45_55_sma75_persist1_pos3m": 0.45,
        "blend_1y_6m_45_55_sma75_persist3_pos3m": 0.45,
        "blend_1y_6m_45_55_sma50_persist_pos3m": 0.45,
        "blend_1y_6m_45_55_sma100_persist_pos3m": 0.45,
        "blend_1y_6m_50_50_sma75_persist3_pos3m": 0.5,
        "blend_1y_6m_47_5_52_5_sma75_persist3_pos3m": 0.475,
        "blend_1y_6m_42_5_57_5_sma75_persist3_pos3m": 0.425,
        "blend_1y_6m_40_60_sma75_persist3_pos3m": 0.4,
        "blend_1y_6m_42_5_57_5_sma75_persist_pos3m": 0.425,
        "blend_1y_6m_40_60_sma75_persist_pos3m": 0.4,
        "blend_1y_6m_35_65_sma75_persist_pos3m": 0.35,
        "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m": 0.35,
        "blend_1y_6m_30_70_sma75_persist_pos3m": 0.3,
        "blend_1y_6m_25_75_sma75_persist_pos3m": 0.25,
        "blend_1y_6m_0_100_sma75_persist_pos3m": 0.0,
        "blend_1y_6m_70_30_sma75_persist_dd": 0.7,
        "blend_1y_6m_70_30_sma100_persist": 0.7,
        "blend_1y_6m_70_30_sma50_persist": 0.7,
        "blend_1y_6m_70_30_volscaled_sma75_persist": 0.7,
        "blend_1y_6m_65_35_sma75_persist": 0.65,
        "blend_1y_6m_60_40_sma75_persist": 0.6,
        "blend_1y_6m_40_60_sma75_persist": 0.4,
        "blend_1y_6m_30_70_sma75_persist": 0.3,
    }
    triple_blend_weights = {
        # (w_1y, w_6m, w_3m)
        "blend_1y_6m_3m_sma75_persist": (0.5, 0.3, 0.2),
        "blend_1y_6m_3m_60_25_15_sma75_persist": (0.6, 0.25, 0.15),
        "blend_1y_6m_3m_70_20_10_sma75_persist": (0.7, 0.2, 0.1),
        "blend_1y_6m_3m_50_25_25_sma75_persist": (0.5, 0.25, 0.25),
        "blend_1y_6m_3m_40_30_30_sma75_persist": (0.4, 0.3, 0.3),
    }
    if (
        mode == "bh_6m_sma75_persist"
        or mode in blend_1y_6m_weights
        or mode in triple_blend_weights
    ):
        frame = frame[
            (frame["log_6m"] > 0)
            & (frame["close"] > frame["sma200"])
        ].copy()
        if frame.empty:
            return []
        if mode == "bh_6m_sma75_persist":
            ranked = frame.sort_values("log_6m", ascending=False)
            return ranked.head(top_n).index.tolist()
        frame = frame[frame["log_1y"] > 0].copy()
        if frame.empty:
            return []
        if mode in (
            "blend_1y_6m_70_30_sma75_persist_entrybuf",
            "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf",
        ):
            frame = frame[frame["close"] > frame["sma75"] * 1.01].copy()
            if frame.empty:
                return []
        if "_pos3m" in mode:
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
        if mode in (
            "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m",
            "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m",
        ):
            frame = frame[frame["log_1m"] > 0].copy()
            if frame.empty:
                return []
        if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_accel":
            frame = frame[frame["log_1m"] > frame["log_3m"] / 3.0].copy()
            if frame.empty:
                return []
        if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_slope":
            frame = frame[(frame["slope20"] > 0) & (frame["slope50"] > 0)].copy()
            if frame.empty:
                return []
        if mode in triple_blend_weights:
            frame = frame[frame["log_3m"] > 0].copy()
            if frame.empty:
                return []
            w1, w6, w3 = triple_blend_weights[mode]
            score = (
                w1 * percentile_rank(frame["log_1y"])
                + w6 * percentile_rank(frame["log_6m"])
                + w3 * percentile_rank(frame["log_3m"])
            )
            frame["score"] = score
            ranked = frame.sort_values("score", ascending=False)
            return ranked.head(top_n).index.tolist()
        rank_1y = percentile_rank(frame["log_1y"])
        rank_6m = percentile_rank(frame["log_6m"])
        w_1y = blend_1y_6m_weights[mode]
        frame["score"] = w_1y * rank_1y + (1.0 - w_1y) * rank_6m
        if mode == "blend_1y_6m_70_30_volscaled_sma75_persist":
            # Penalize high-volatility names (subtract 20% rank-vol weight)
            rank_low_vol = percentile_rank(-frame["vol20"])
            frame["score"] = 0.8 * frame["score"] + 0.2 * rank_low_vol
        # _regime switch: swap selection blend based on market breadth (not cash)
        if mode.endswith("_regime"):
            breadth = float((features.close.loc[date] > features.sma200.loc[date]).mean())
            if breadth < 0.55:
                # use safer 45/55 blend when regime is weak
                frame["score"] = 0.45 * rank_1y + 0.55 * rank_6m
        # _spygate / _chand_tstop_spygate: hard gate on SPY trend (no entry below SPY-SMA200)
        if mode.endswith("_spygate") and "SPY" in features.close.columns:
            try:
                spy_close = float(features.close.loc[date, "SPY"])
                spy_sma = float(features.sma200.loc[date, "SPY"])
                if pd.notna(spy_sma) and spy_close < spy_sma:
                    return []
            except (KeyError, ValueError):
                pass
        # _spytrend: 3-state regime - cash if SPY<SMA200, safer if SPY<SMA50, full otherwise
        if mode.endswith("_spytrend") and "SPY" in features.close.columns:
            try:
                spy_close = float(features.close.loc[date, "SPY"])
                spy_sma200 = float(features.sma200.loc[date, "SPY"])
                spy_sma50 = float(features.sma50.loc[date, "SPY"])
                if pd.notna(spy_sma200) and spy_close < spy_sma200:
                    return []  # cash regime
                if pd.notna(spy_sma50) and spy_close < spy_sma50:
                    # transition regime: use safer 45/55 blend
                    frame["score"] = 0.45 * rank_1y + 0.55 * rank_6m
            except (KeyError, ValueError):
                pass
        # _spydd: portfolio circuit breaker on SPY drawdown from 60d peak
        if mode.endswith("_spydd") and "SPY" in features.close.columns:
            try:
                idx = features.close.index.get_loc(date)
                lookback = features.close.iloc[max(0, idx - 60): idx + 1]["SPY"]
                spy_peak = float(lookback.max())
                spy_close = float(features.close.loc[date, "SPY"])
                if spy_peak > 0 and (spy_close - spy_peak) / spy_peak < -0.08:
                    return []
            except (KeyError, ValueError, IndexError):
                pass
        ranked = frame.sort_values("score", ascending=False)
        # _ensemble: reorder so names that also rank in safer 45/55 top 15 come first
        if mode.endswith("_ensemble"):
            score_safer = 0.45 * rank_1y + 0.55 * rank_6m
            safer_top = set(score_safer.nlargest(7).index)
            ordered_idx = list(ranked.index)
            matched = [t for t in ordered_idx if t in safer_top]
            unmatched = [t for t in ordered_idx if t not in safer_top]
            reordered = matched + unmatched
            return reordered[:top_n]
        # _intra: use intraday momentum (intra_5h) as tiebreaker for close scores
        if mode.endswith("_intra"):
            ranked = ranked.copy()
            # Get top N + buffer candidates
            candidates = ranked.head(top_n + 10)
            # For stocks with similar scores (within 1% range), use intra_5h as tiebreaker
            if len(candidates) > 1:
                score_range = candidates["score"].max() - candidates["score"].min()
                if score_range > 0:
                    # Normalize score to 0-1, then add intra_5h as small tiebreaker (weight 0.01)
                    norm_score = (candidates["score"] - candidates["score"].min()) / score_range
                    # Use intra_5h if available, otherwise 0
                    intra_values = candidates.get("intra_5h", pd.Series(0.0, index=candidates.index))
                    intra_values = intra_values.fillna(0.0)
                    # Combined score: primary score + 0.1 * intra_5h (as tiebreaker)
                    candidates["combined_score"] = norm_score + 0.1 * intra_values
                    ranked = frame.loc[candidates.sort_values("combined_score", ascending=False).index]
                else:
                    # All scores equal, use intra_5h as primary sorter
                    intra_values = candidates.get("intra_5h", pd.Series(0.0, index=candidates.index))
                    intra_values = intra_values.fillna(0.0)
                    ranked = frame.loc[candidates.sort_values(intra_values, ascending=False).index]
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

    # Convex short-window momentum: emphasize recent acceleration but keep trend quality guardrails.
    if mode == "convex_mom_quality":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["log_1m"] > 0)
            & (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
            & (frame["slope50"] > 0)
            & (frame["trend_r2_60"] > 0.35)
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_6m = percentile_rank(frame["log_6m"])
        rank_3m = percentile_rank(frame["log_3m"])
        rank_1m = percentile_rank(frame["log_1m"])
        rank_liq = percentile_rank(frame["dollar_vol20"])
        base = 0.15 * rank_1y + 0.25 * rank_6m + 0.35 * rank_3m + 0.25 * rank_1m
        frame["score"] = (base.pow(2) * rank_liq).clip(lower=1e-6)
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Breakout momentum: prefer fresh breakouts with strong near-term trend and liquidity.
    if mode == "breakout_mom_liq":
        frame = frame[
            (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["log_1m"] > 0)
            & (frame["log_10d"] > 0)
            & (frame["close"] > frame["sma20"])
            & (frame["sma20"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
            & (frame["slope20"] > 0)
        ].copy()
        if frame.empty:
            return []
        rank_10d = percentile_rank(frame["log_10d"])
        rank_1m = percentile_rank(frame["log_1m"])
        rank_3m = percentile_rank(frame["log_3m"])
        rank_6m = percentile_rank(frame["log_6m"])
        rank_liq = percentile_rank(frame["dollar_vol20"])
        frame["score"] = (rank_10d.pow(2) * rank_1m * (0.6 * rank_3m + 0.4 * rank_6m) * rank_liq).clip(lower=1e-6)
        ranked = frame.sort_values("score", ascending=False)
        return ranked.head(top_n).index.tolist()

    # Guarded convex blend: keep medium/long trend confirmation while emphasizing near-term acceleration.
    if mode == "mom2_guarded":
        frame = frame[
            (frame["log_1y"] > 0)
            & (frame["log_6m"] > 0)
            & (frame["log_3m"] > 0)
            & (frame["log_1m"] > 0)
            & (frame["close"] > frame["sma50"])
            & (frame["close"] > frame["sma200"])
            & (frame["slope50"] > 0)
            & (frame["trend_r2_60"] > 0.30)
        ].copy()
        if frame.empty:
            return []
        rank_1y = percentile_rank(frame["log_1y"])
        rank_6m = percentile_rank(frame["log_6m"])
        rank_3m = percentile_rank(frame["log_3m"])
        rank_1m = percentile_rank(frame["log_1m"])
        rank_liq = percentile_rank(frame["dollar_vol20"])
        core = (0.45 * rank_1y + 0.55 * rank_6m).pow(2)
        accel = (0.7 * rank_3m + 0.3 * rank_1m).pow(2)
        frame["score"] = (core * accel * rank_liq).clip(lower=1e-6)
        ranked = frame.sort_values("score", ascending=False)
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
    backtest_dates: Iterable[pd.Timestamp],
    mode: str,
    top_n: int,
    transaction_cost_bps: float,
    min_price: float,
    min_dollar_volume: float,
) -> BacktestResult:
    rebalance_list = list(rebalance_dates)
    trade_dates = list(backtest_dates)
    if not trade_dates:
        raise RuntimeError("No trading dates available for backtest window")
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
        "bh_1y_sma75_persist_lowvol",
        "bh_1y_sma75_persist_trend",
        "bh_1y_sma75_persist_quality",
        "bh_1y_sma75_persist_regime",
        "bh_1y_sma75_persist_cooldown5",
        "bh_1y_sma75_persist_cooldown10",
        "bh_1y_sma75_persist_buffer",
        "bh_1y_sma75_persist_rank20",
        "bh_1y_sma75_persist_rankdrop7",
        "bh_1y_sma75_persist_invvol",
        "blend_90_10_sma75_persist",
        "blend_85_15_sma75_persist",
        "blend_80_20_sma75_persist",
        "blend_60_40_sma75_persist",
        "bh_6m_sma75_persist",
        "blend_1y_6m_sma75_persist",
        "blend_1y_6m_80_20_sma75_persist",
        "blend_1y_6m_75_25_sma75_persist",
        "blend_1y_6m_70_30_sma75_persist",
        "blend_1y_6m_65_35_sma75_persist",
        "blend_1y_6m_60_40_sma75_persist",
        "blend_1y_6m_40_60_sma75_persist",
        "blend_1y_6m_30_70_sma75_persist",
        "blend_1y_6m_70_30_sma75_persist1",
        "blend_1y_6m_70_30_sma75_persist3",
        "blend_1y_6m_70_30_sma75_persist_rank20",
        "blend_1y_6m_70_30_sma75_persist_rankdrop7",
        "blend_1y_6m_70_30_sma75_persist_entrybuf",
        "blend_1y_6m_70_30_sma75_persist_pos3m",
        "blend_1y_6m_80_20_sma75_persist_pos3m",
        "blend_1y_6m_75_25_sma75_persist_pos3m",
        "blend_1y_6m_65_35_sma75_persist_pos3m",
        "blend_1y_6m_60_40_sma75_persist_pos3m",
        "blend_1y_6m_55_45_sma75_persist_pos3m",
        "blend_1y_6m_50_50_sma75_persist_pos3m",
        "blend_1y_6m_47_5_52_5_sma75_persist_pos3m",
        "blend_1y_6m_45_55_sma75_persist_pos3m",
        "blend_1y_6m_45_55_sma75_persist_pos3m_rank10",
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3",
        "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf",
        "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m",
        "blend_1y_6m_45_55_sma75_persist_pos3m_accel",
        "blend_1y_6m_45_55_sma75_persist_pos3m_slope",
        "blend_1y_6m_45_55_sma75_persist_pos3m_possize",
        "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget",
        "blend_1y_6m_45_55_sma75_persist_pos3m_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven",
        "blend_1y_6m_45_55_sma75_persist_pos3m_volexit",
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap",
        "blend_1y_6m_45_55_sma75_persist_pos3m_staggered",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
        "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
        "blend_1y_6m_45_55_sma75_persist_pos3m_rank20",
        "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7",
        "blend_1y_6m_45_55_sma75_persist1_pos3m",
        "blend_1y_6m_45_55_sma75_persist3_pos3m",
        "blend_1y_6m_45_55_sma50_persist_pos3m",
        "blend_1y_6m_45_55_sma100_persist_pos3m",
        "blend_1y_6m_50_50_sma75_persist3_pos3m",
        "blend_1y_6m_47_5_52_5_sma75_persist3_pos3m",
        "blend_1y_6m_42_5_57_5_sma75_persist3_pos3m",
        "blend_1y_6m_40_60_sma75_persist3_pos3m",
        "blend_1y_6m_42_5_57_5_sma75_persist_pos3m",
        "blend_1y_6m_40_60_sma75_persist_pos3m",
        "blend_1y_6m_35_65_sma75_persist_pos3m",
        "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m",
        "blend_1y_6m_30_70_sma75_persist_pos3m",
        "blend_1y_6m_25_75_sma75_persist_pos3m",
        "blend_1y_6m_0_100_sma75_persist_pos3m",
        "blend_1y_6m_70_30_sma75_persist_dd",
        "blend_1y_6m_70_30_sma100_persist",
        "blend_1y_6m_70_30_sma50_persist",
        "blend_1y_6m_70_30_volscaled_sma75_persist",
        "blend_1y_6m_3m_sma75_persist",
        "blend_1y_6m_3m_60_25_15_sma75_persist",
        "blend_1y_6m_3m_70_20_10_sma75_persist",
        "blend_1y_6m_3m_50_25_25_sma75_persist",
        "blend_1y_6m_3m_40_30_30_sma75_persist",
        "bh_1y_sma75_persist_exitbuf",
        "bh_1y_sma75_persist_minhold",
        "bh_1y_sma75_persist_slopegate",
        "bh_1y_sma75_persist_accel",
        "blend_1y_5d_smart",
        "blend_1y_5d_persist",
        "bh_1y_5d_tiebreak",
        "bh_1y_5d_exit",
    }
    sma_breach_streak: Dict[str, int] = {}  # for persist variants
    neg_5d_streak: Dict[str, int] = {}  # for bh_1y_5d_exit
    cooldown_until: Dict[str, pd.Timestamp] = {}
    twostage_reduced: Dict[str, bool] = {}
    time_stop_counter: Dict[str, int] = {}  # for _timestop variants
    pt_done: Dict[str, bool] = {}  # for _pt / _late_pt profit-take variants
    staggered_bucket: Dict[str, int] = {}

    def dynamic_top_n(current_date: pd.Timestamp) -> int:
        if mode != "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven":
            return top_n
        breadth = float((features.close.loc[current_date] > features.sma200.loc[current_date]).mean())
        if breadth >= 0.70:
            return 5
        if breadth <= 0.55:
            return 3
        return 4

    def apply_weight_change(date: pd.Timestamp, new_weights: Dict[str, float]) -> None:
        nonlocal weights, equity, turnover_sum, executed_rebalances
        turnover = sum(
            abs(new_weights.get(ticker, 0.0) - weights.get(ticker, 0.0))
            for ticker in set(new_weights) | set(weights)
        )
        turnover_sum += turnover
        equity *= 1.0 - turnover * (transaction_cost_bps / 10_000.0)
        weights = new_weights
        executed_rebalances += 1

    def apply_rebalance(date: pd.Timestamp, picks: list[str]) -> None:
        nonlocal last_nonempty_selection_date, last_nonempty_selection
        if not picks:
            new_weights: Dict[str, float] = {}
        else:
            # Position sizing by rank strength for rebal_position_sizing mode
            if mode in (
                "rebal_position_sizing",
                "bh_1y_persist_possize",
                "blend_1y_6m_45_55_sma75_persist_pos3m_possize",
            ):
                # Weight by rank: top gets more weight, linear decay
                rank_weights = {}
                for i, ticker in enumerate(picks):
                    rank_weights[ticker] = (len(picks) - i) / sum(range(1, len(picks) + 1))
                new_weights = rank_weights
            elif mode in (
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
            ):
                pick_frame = build_snapshot(features, date).reindex(picks).dropna(subset=["log_1y", "log_6m", "log_3m", "dollar_vol20"])
                if pick_frame.empty:
                    new_weights = {ticker: 1.0 / len(picks) for ticker in picks}
                else:
                    rank_1y = percentile_rank(pick_frame["log_1y"])
                    rank_6m = percentile_rank(pick_frame["log_6m"])
                    rank_3m = percentile_rank(pick_frame["log_3m"])
                    mom_score = 0.45 * rank_1y + 0.55 * rank_6m
                    liq_score = percentile_rank(pick_frame["dollar_vol20"])
                    if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage":
                        liq_score = liq_score.pow(2)
                    if mode in (
                        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
                        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                        "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
                        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
                        "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                        "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                        "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
                    ):
                        mom_base = 0.25 * rank_1y + 0.50 * rank_6m + 0.25 * rank_3m
                        exponent = 2.8 if len(picks) >= 5 else 2.0
                        if mode == "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp":
                            exponent = 2.4 if len(picks) >= 5 else 1.8
                        if mode in (
                            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                            "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                        ):
                            mom_base = 0.15 * rank_1y + 0.55 * rank_6m + 0.30 * rank_3m
                            exponent = 3.4 if len(picks) >= 5 else 2.6
                        if mode in (
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
                        ):
                            mom_base = 0.10 * rank_1y + 0.55 * rank_6m + 0.35 * rank_3m
                            exponent = 4.0 if len(picks) >= 5 else 3.0
                        mom_score = mom_base.pow(exponent)
                        if mode == "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp":
                            rank_low_vol = percentile_rank(-pick_frame["vol20"].fillna(pick_frame["vol20"].median()))
                            mom_score = mom_score * (0.70 + 0.30 * rank_low_vol)
                    raw = (mom_score * liq_score).clip(lower=1e-6)
                    raw_sum = float(raw.sum())
                    new_weights = {
                        ticker: float(raw.loc[ticker] / raw_sum)
                        for ticker in pick_frame.index
                    }
                    if mode == "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp":
                        max_w = 0.34
                        residual = 0.0
                        capped_weights: Dict[str, float] = {}
                        for ticker, w in new_weights.items():
                            cap_w = min(w, max_w)
                            capped_weights[ticker] = cap_w
                            residual += w - cap_w
                        if residual > 1e-8:
                            uncapped = [t for t, w in capped_weights.items() if w < max_w - 1e-8]
                            uncapped_sum = sum(capped_weights[t] for t in uncapped)
                            if uncapped_sum > 1e-8:
                                for ticker in uncapped:
                                    add_w = residual * (capped_weights[ticker] / uncapped_sum)
                                    capped_weights[ticker] = min(max_w, capped_weights[ticker] + add_w)
                        total_weight = float(sum(capped_weights.values()))
                        if total_weight > 1e-8:
                            new_weights = {ticker: weight / total_weight for ticker, weight in capped_weights.items()}
                    for ticker in picks:
                        new_weights.setdefault(ticker, 0.0)
            elif mode == "bh_1y_sma75_persist_invvol":
                inv_vol_weights = {}
                for ticker in picks:
                    vol_value = features.vol20.loc[date, ticker]
                    inv_vol_weights[ticker] = 1.0 / max(float(vol_value), 1e-6)
                weight_sum = sum(inv_vol_weights.values())
                new_weights = {
                    ticker: weight / weight_sum
                    for ticker, weight in inv_vol_weights.items()
                }
            else:
                new_weights = {ticker: 1.0 / len(picks) for ticker in picks}

            if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget":
                target_vol = 0.30
                gross_cap = 1.20
                port_vol = 0.0
                for ticker, weight in new_weights.items():
                    vol_val = float(features.vol20.loc[date, ticker]) if pd.notna(features.vol20.loc[date, ticker]) else 0.0
                    port_vol += (weight * vol_val) ** 2
                port_vol = np.sqrt(port_vol)
                if port_vol > 1e-6:
                    scale = min(gross_cap, max(0.0, target_vol / port_vol))
                    new_weights = {ticker: weight * scale for ticker, weight in new_weights.items()}

            if mode in (
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
            ) and weights and new_weights:
                alpha = 0.75
                if mode == "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp":
                    alpha = 0.62
                smoothed_weights: Dict[str, float] = {}
                for ticker, target_weight in new_weights.items():
                    if ticker in weights:
                        smoothed_weights[ticker] = alpha * target_weight + (1.0 - alpha) * weights[ticker]
                    else:
                        smoothed_weights[ticker] = target_weight
                total_weight = float(sum(smoothed_weights.values()))
                if total_weight > 1e-8:
                    new_weights = {
                        ticker: weight / total_weight
                        for ticker, weight in smoothed_weights.items()
                        if weight > 1e-8
                    }

        apply_weight_change(date, new_weights)
        if picks:
            last_nonempty_selection_date = date
            last_nonempty_selection = picks
        # Track entry metadata for rebalancing variants
        for ticker in picks:
            if ticker not in entry_dates:
                entry_dates[ticker] = date
                entry_prices[ticker] = float(features.close.loc[date, ticker])
                peak_prices[ticker] = entry_prices[ticker]

    for date in trade_dates:
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

        # _regime: no daily cash force (too costly). Entry gate in select_tickers handles new entries when breadth weak.
        # Time-stop counter update: reset on new peak, else increment
        if (mode.endswith("_timestop") or "_chand_tstop" in mode) and weights:
            for _t in list(weights.keys()):
                if _t in peak_prices and _t in features.close.columns:
                    _cp = features.close.loc[date, _t]
                    if pd.notna(_cp):
                        if float(_cp) >= peak_prices[_t] * 0.995:
                            time_stop_counter[_t] = 0
                        else:
                            time_stop_counter[_t] = time_stop_counter.get(_t, 0) + 1

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
            "bh_1y_sma75_persist_lowvol",
            "bh_1y_sma75_persist_trend",
            "bh_1y_sma75_persist_quality",
            "bh_1y_sma75_persist_regime",
            "bh_1y_sma75_persist_cooldown5",
            "bh_1y_sma75_persist_cooldown10",
            "bh_1y_sma75_persist_buffer",
            "bh_1y_sma75_persist_rank20",
            "bh_1y_sma75_persist_rankdrop7",
            "bh_1y_sma75_persist_invvol",
            "blend_90_10_sma75_persist",
            "blend_85_15_sma75_persist",
            "blend_80_20_sma75_persist",
            "blend_60_40_sma75_persist",
            "bh_6m_sma75_persist",
            "blend_1y_6m_sma75_persist",
            "blend_1y_6m_80_20_sma75_persist",
            "blend_1y_6m_75_25_sma75_persist",
            "blend_1y_6m_70_30_sma75_persist",
            "blend_1y_6m_65_35_sma75_persist",
            "blend_1y_6m_60_40_sma75_persist",
            "blend_1y_6m_40_60_sma75_persist",
            "blend_1y_6m_30_70_sma75_persist",
            "blend_1y_6m_70_30_sma75_persist1",
            "blend_1y_6m_70_30_sma75_persist3",
            "blend_1y_6m_70_30_sma75_persist_rank20",
            "blend_1y_6m_70_30_sma75_persist_rankdrop7",
            "blend_1y_6m_70_30_sma75_persist_entrybuf",
            "blend_1y_6m_70_30_sma75_persist_pos3m",
            "blend_1y_6m_80_20_sma75_persist_pos3m",
            "blend_1y_6m_75_25_sma75_persist_pos3m",
            "blend_1y_6m_65_35_sma75_persist_pos3m",
            "blend_1y_6m_60_40_sma75_persist_pos3m",
            "blend_1y_6m_55_45_sma75_persist_pos3m",
            "blend_1y_6m_50_50_sma75_persist_pos3m",
            "blend_1y_6m_47_5_52_5_sma75_persist_pos3m",
            "blend_1y_6m_45_55_sma75_persist_pos3m",
            "blend_1y_6m_45_55_sma75_persist_pos3m_full",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank10",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3",
            "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf",
            "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m",
            "blend_1y_6m_45_55_sma75_persist_pos3m_accel",
            "blend_1y_6m_45_55_sma75_persist_pos3m_slope",
            "blend_1y_6m_45_55_sma75_persist_pos3m_possize",
            "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget",
            "blend_1y_6m_45_55_sma75_persist_pos3m_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven",
            "blend_1y_6m_45_55_sma75_persist_pos3m_volexit",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap",
            "blend_1y_6m_45_55_sma75_persist_pos3m_staggered",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
            "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank20",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7",
            "blend_1y_6m_45_55_sma75_persist1_pos3m",
            "blend_1y_6m_45_55_sma75_persist3_pos3m",
            "blend_1y_6m_45_55_sma50_persist_pos3m",
            "blend_1y_6m_45_55_sma100_persist_pos3m",
            "blend_1y_6m_50_50_sma75_persist3_pos3m",
            "blend_1y_6m_47_5_52_5_sma75_persist3_pos3m",
            "blend_1y_6m_42_5_57_5_sma75_persist3_pos3m",
            "blend_1y_6m_40_60_sma75_persist3_pos3m",
            "blend_1y_6m_42_5_57_5_sma75_persist_pos3m",
            "blend_1y_6m_40_60_sma75_persist_pos3m",
            "blend_1y_6m_35_65_sma75_persist_pos3m",
            "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m",
            "blend_1y_6m_30_70_sma75_persist_pos3m",
            "blend_1y_6m_25_75_sma75_persist_pos3m",
            "blend_1y_6m_0_100_sma75_persist_pos3m",
            "blend_1y_6m_70_30_sma75_persist_dd",
            "blend_1y_6m_70_30_sma100_persist",
            "blend_1y_6m_70_30_sma50_persist",
            "blend_1y_6m_70_30_volscaled_sma75_persist",
            "blend_1y_6m_3m_sma75_persist",
            "blend_1y_6m_3m_60_25_15_sma75_persist",
            "blend_1y_6m_3m_70_20_10_sma75_persist",
            "blend_1y_6m_3m_50_25_25_sma75_persist",
            "blend_1y_6m_3m_40_30_30_sma75_persist",
            "bh_1y_sma75_persist_exitbuf",
            "bh_1y_sma75_persist_minhold",
            "bh_1y_sma75_persist_slopegate",
            "bh_1y_sma75_persist_accel",
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
            "bh_1y_sma75_persist_lowvol": 2,
            "bh_1y_sma75_persist_trend": 2,
            "bh_1y_sma75_persist_quality": 2,
            "bh_1y_sma75_persist_regime": 2,
            "bh_1y_sma75_persist_cooldown5": 2,
            "bh_1y_sma75_persist_cooldown10": 2,
            "bh_1y_sma75_persist_buffer": 2,
            "bh_1y_sma75_persist_rank20": 2,
            "bh_1y_sma75_persist_rankdrop7": 2,
            "bh_1y_sma75_persist_invvol": 2,
            "blend_90_10_sma75_persist": 2,
            "blend_85_15_sma75_persist": 2,
            "blend_80_20_sma75_persist": 2,
            "blend_60_40_sma75_persist": 2,
            "bh_6m_sma75_persist": 2,
            "blend_1y_6m_sma75_persist": 2,
            "blend_1y_6m_80_20_sma75_persist": 2,
            "blend_1y_6m_75_25_sma75_persist": 2,
            "blend_1y_6m_70_30_sma75_persist": 2,
            "blend_1y_6m_65_35_sma75_persist": 2,
            "blend_1y_6m_60_40_sma75_persist": 2,
            "blend_1y_6m_40_60_sma75_persist": 2,
            "blend_1y_6m_30_70_sma75_persist": 2,
            "blend_1y_6m_70_30_sma75_persist1": 1,
            "blend_1y_6m_70_30_sma75_persist3": 3,
            "blend_1y_6m_70_30_sma75_persist_rank20": 2,
            "blend_1y_6m_70_30_sma75_persist_rankdrop7": 2,
            "blend_1y_6m_70_30_sma75_persist_entrybuf": 2,
            "blend_1y_6m_70_30_sma75_persist_pos3m": 2,
            "blend_1y_6m_80_20_sma75_persist_pos3m": 2,
            "blend_1y_6m_75_25_sma75_persist_pos3m": 2,
            "blend_1y_6m_65_35_sma75_persist_pos3m": 2,
            "blend_1y_6m_60_40_sma75_persist_pos3m": 2,
            "blend_1y_6m_55_45_sma75_persist_pos3m": 2,
            "blend_1y_6m_50_50_sma75_persist_pos3m": 2,
            "blend_1y_6m_47_5_52_5_sma75_persist_pos3m": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_full": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank10": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_accel": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_slope": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_possize": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_volexit": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_staggered": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth": 2,
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage": 2,
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp": 2,
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage": 2,
            "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank20": 2,
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7": 2,
            "blend_1y_6m_45_55_sma75_persist1_pos3m": 1,
            "blend_1y_6m_45_55_sma75_persist3_pos3m": 3,
            "blend_1y_6m_45_55_sma50_persist_pos3m": 2,
            "blend_1y_6m_45_55_sma100_persist_pos3m": 2,
            "blend_1y_6m_50_50_sma75_persist3_pos3m": 3,
            "blend_1y_6m_47_5_52_5_sma75_persist3_pos3m": 3,
            "blend_1y_6m_42_5_57_5_sma75_persist3_pos3m": 3,
            "blend_1y_6m_40_60_sma75_persist3_pos3m": 3,
            "blend_1y_6m_42_5_57_5_sma75_persist_pos3m": 2,
            "blend_1y_6m_40_60_sma75_persist_pos3m": 2,
            "blend_1y_6m_35_65_sma75_persist_pos3m": 2,
            "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m": 2,
            "blend_1y_6m_30_70_sma75_persist_pos3m": 2,
            "blend_1y_6m_25_75_sma75_persist_pos3m": 2,
            "blend_1y_6m_0_100_sma75_persist_pos3m": 2,
            "blend_1y_6m_70_30_sma75_persist_dd": 2,
            "blend_1y_6m_70_30_sma100_persist": 2,
            "blend_1y_6m_70_30_sma50_persist": 2,
            "blend_1y_6m_70_30_volscaled_sma75_persist": 2,
            "blend_1y_6m_3m_sma75_persist": 2,
            "blend_1y_6m_3m_60_25_15_sma75_persist": 2,
            "blend_1y_6m_3m_70_20_10_sma75_persist": 2,
            "blend_1y_6m_3m_50_25_25_sma75_persist": 2,
            "blend_1y_6m_3m_40_30_30_sma75_persist": 2,
            "bh_1y_sma75_persist_exitbuf": 2,
            "bh_1y_sma75_persist_minhold": 2,
            "bh_1y_sma75_persist_slopegate": 2,
            "bh_1y_sma75_persist_accel": 2,
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
            "bh_1y_sma75_persist_lowvol": "sma75",
            "bh_1y_sma75_persist_trend": "sma75",
            "bh_1y_sma75_persist_quality": "sma75",
            "bh_1y_sma75_persist_regime": "sma75",
            "bh_1y_sma75_persist_cooldown5": "sma75",
            "bh_1y_sma75_persist_cooldown10": "sma75",
            "bh_1y_sma75_persist_buffer": "sma75",
            "bh_1y_sma75_persist_rank20": "sma75",
            "bh_1y_sma75_persist_rankdrop7": "sma75",
            "bh_1y_sma75_persist_invvol": "sma75",
            "blend_90_10_sma75_persist": "sma75",
            "blend_85_15_sma75_persist": "sma75",
            "blend_80_20_sma75_persist": "sma75",
            "blend_60_40_sma75_persist": "sma75",
            "bh_6m_sma75_persist": "sma75",
            "blend_1y_6m_sma75_persist": "sma75",
            "blend_1y_6m_80_20_sma75_persist": "sma75",
            "blend_1y_6m_75_25_sma75_persist": "sma75",
            "blend_1y_6m_70_30_sma75_persist": "sma75",
            "blend_1y_6m_65_35_sma75_persist": "sma75",
            "blend_1y_6m_60_40_sma75_persist": "sma75",
            "blend_1y_6m_40_60_sma75_persist": "sma75",
            "blend_1y_6m_30_70_sma75_persist": "sma75",
            "blend_1y_6m_70_30_sma75_persist1": "sma75",
            "blend_1y_6m_70_30_sma75_persist3": "sma75",
            "blend_1y_6m_70_30_sma75_persist_rank20": "sma75",
            "blend_1y_6m_70_30_sma75_persist_rankdrop7": "sma75",
            "blend_1y_6m_70_30_sma75_persist_entrybuf": "sma75",
            "blend_1y_6m_70_30_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_80_20_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_75_25_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_65_35_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_60_40_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_55_45_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_50_50_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_47_5_52_5_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_full": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank10": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_entrybuf": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_pos1m": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_accel": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_slope": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_possize": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_voltarget": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_adaptiven": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_volexit": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_staggered": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth": "sma75",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage": "sma75",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp": "sma75",
            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage": "sma75",
            "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rank20": "sma75",
            "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7": "sma75",
            "blend_1y_6m_45_55_sma75_persist1_pos3m": "sma75",
            "blend_1y_6m_45_55_sma75_persist3_pos3m": "sma75",
            "blend_1y_6m_45_55_sma50_persist_pos3m": "sma50",
            "blend_1y_6m_45_55_sma100_persist_pos3m": "sma100",
            "blend_1y_6m_50_50_sma75_persist3_pos3m": "sma75",
            "blend_1y_6m_47_5_52_5_sma75_persist3_pos3m": "sma75",
            "blend_1y_6m_42_5_57_5_sma75_persist3_pos3m": "sma75",
            "blend_1y_6m_40_60_sma75_persist3_pos3m": "sma75",
            "blend_1y_6m_42_5_57_5_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_40_60_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_35_65_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_35_65_sma75_persist_pos3m_pos1m": "sma75",
            "blend_1y_6m_30_70_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_25_75_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_0_100_sma75_persist_pos3m": "sma75",
            "blend_1y_6m_70_30_sma75_persist_dd": "sma75",
            "blend_1y_6m_70_30_sma100_persist": "sma100",
            "blend_1y_6m_70_30_sma50_persist": "sma50",
            "blend_1y_6m_70_30_volscaled_sma75_persist": "sma75",
            "blend_1y_6m_3m_sma75_persist": "sma75",
            "blend_1y_6m_3m_60_25_15_sma75_persist": "sma75",
            "blend_1y_6m_3m_70_20_10_sma75_persist": "sma75",
            "blend_1y_6m_3m_50_25_25_sma75_persist": "sma75",
            "blend_1y_6m_3m_40_30_30_sma75_persist": "sma75",
            "bh_1y_sma75_persist_exitbuf": "sma75",
            "bh_1y_sma75_persist_minhold": "sma75",
            "bh_1y_sma75_persist_slopegate": "sma75",
            "bh_1y_sma75_persist_accel": "sma75",
            "blend_1y_5d_persist": "sma75",
        }
        dd_modes = {
            "bh_1y_persist_dd",
            "bh_1y_sma75_persist_dd",
            "blend_1y_6m_70_30_sma75_persist_dd",
        }
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
                        # exitbuf: only count breach if 3% below SMA
                        if mode == "bh_1y_sma75_persist_exitbuf":
                            is_breach = pd.notna(sma_val) and close_price < sma_val * 0.97
                        else:
                            if mode in (
                                "blend_1y_6m_45_55_sma75_persist_pos3m_volexit",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
                                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
                                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                                "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
                            ):
                                vol_value = features.vol20.loc[date, ticker]
                                vol_switch = 0.55
                                if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage":
                                    vol_switch = 0.45
                                elif mode == "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage":
                                    vol_switch = 0.65
                                dynamic_sma_attr = "sma100" if pd.notna(vol_value) and float(vol_value) > vol_switch else "sma75"
                                sma_val = getattr(features, dynamic_sma_attr).loc[date, ticker]
                            is_breach = pd.notna(sma_val) and close_price < sma_val
                        if is_breach:
                            sma_breach_streak[ticker] = sma_breach_streak.get(ticker, 0) + 1
                        else:
                            sma_breach_streak[ticker] = 0
                        threshold = persist_threshold[mode]
                        persist_hit = sma_breach_streak.get(ticker, 0) >= threshold
                        # minhold: suppress exit during 5 trading-day grace period
                        if mode == "bh_1y_sma75_persist_minhold" and ticker in entry_dates:
                            days_held = (date - entry_dates[ticker]).days
                            if days_held < 7:  # ~5 trading days
                                persist_hit = False
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
                        if mode in (
                            "blend_1y_6m_45_55_sma75_persist_pos3m_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
                            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
                            "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                            "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                            "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
                            "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
                        ) and breach:
                            if not twostage_reduced.get(ticker, False):
                                twostage_reduced[ticker] = True
                                breach = False
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

                    # _chandelier: volatility-scaled trailing stop (bypasses persist delay)
                    if (mode.endswith("_chandelier") or "_chand_tstop" in mode) and ticker in peak_prices:
                        vol_ann = features.vol20.loc[date, ticker]
                        if pd.notna(vol_ann):
                            daily_vol = float(vol_ann) / np.sqrt(252.0)
                            peak = peak_prices[ticker]
                            drawdown = (close_price - peak) / peak if peak > 0 else 0.0
                            # parametric threshold per variant
                            if mode.endswith("_v2"):
                                chand_mult = 0.8
                            elif mode.endswith("_v3"):
                                chand_mult = 1.3
                            elif mode.endswith("_spyadapt") and "SPY" in features.close.columns:
                                # Soft adaptation: tighten/loosen chand based on SPY trend
                                try:
                                    spy_c = float(features.close.loc[date, "SPY"])
                                    spy_50 = float(features.sma50.loc[date, "SPY"])
                                    spy_200 = float(features.sma200.loc[date, "SPY"])
                                    if pd.notna(spy_50) and pd.notna(spy_200):
                                        if spy_c > spy_50:
                                            chand_mult = 1.3   # loose: market strong
                                        elif spy_c > spy_200:
                                            chand_mult = 1.0   # neutral
                                        else:
                                            chand_mult = 0.6   # tight: market weak
                                    else:
                                        chand_mult = 1.0
                                except (KeyError, ValueError):
                                    chand_mult = 1.0
                            else:
                                chand_mult = 1.0
                            chand_threshold = -chand_mult * daily_vol * np.sqrt(20.0)
                            # _late / _late_pt: only arm chandelier after peak >= entry * 1.10
                            armed = True
                            if ("_late" in mode) and ticker in entry_prices:
                                armed = peak >= entry_prices[ticker] * 1.10
                            if armed and drawdown < chand_threshold:
                                breach = True
                    # _timestop: force exit if price has not revisited peak in N days (parametric)
                    if mode.endswith("_timestop") or "_chand_tstop" in mode:
                        if mode.endswith("_v2"):
                            tstop_days = 7
                        elif mode.endswith("_v3"):
                            tstop_days = 14
                        else:
                            tstop_days = 10
                        # _late / _late_pt: only arm tstop after peak >= entry * 1.10
                        armed_t = True
                        if ("_late" in mode) and ticker in entry_prices and ticker in peak_prices:
                            armed_t = peak_prices[ticker] >= entry_prices[ticker] * 1.10
                        if armed_t and time_stop_counter.get(ticker, 0) >= tstop_days:
                            breach = True
                    # _hard: hard stop loss at -15% from entry (no peak required)
                    if mode.endswith("_hard") and ticker in entry_prices:
                        if close_price < entry_prices[ticker] * 0.85:
                            breach = True
                    # _estop: vol-scaled stop from entry (ATR-like, anchored to entry not peak)
                    if mode.endswith("_estop") and ticker in entry_prices:
                        vol_ann_e = features.vol20.loc[date, ticker]
                        if pd.notna(vol_ann_e):
                            daily_vol_e = float(vol_ann_e) / np.sqrt(252.0)
                            entry_dd = (close_price - entry_prices[ticker]) / entry_prices[ticker]
                            estop_threshold = -0.5 * daily_vol_e * np.sqrt(20.0)
                            if entry_dd < estop_threshold:
                                breach = True

                    if breach:
                        tickers_to_exit.append(ticker)
                except (KeyError, ValueError):
                    continue
            if mode in (
                "blend_1y_6m_45_55_sma75_persist_pos3m_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit45_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit65_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_smooth",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp",
                "blend_1y_6m_40_60_sma75_persist_pos3m_momweight3_volexit_twostage",
                "blend_1y_6m_35_65_sma75_persist_pos3m_momweight3_volexit_twostage",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_regime",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chandelier",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_timestop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_ensemble",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v2",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_v3",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_cd5",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_pt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_hard",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_late_pt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spygate",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spytrend",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_spydd",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spygate",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_spyadapt",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_estop",
                "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop_intra",
                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
            ):
                to_reduce = [t for t in list(weights.keys()) if twostage_reduced.get(t, False) and t not in tickers_to_exit]
                if to_reduce:
                    reduced_weights = dict(weights)
                    for ticker in to_reduce:
                        reduction_mult = 0.5
                        if mode == "blend_1y_6m_40_60_sma75_persist_pos3m_momweight2_volexit_twostage_voldamp":
                            reduction_mult = 0.4
                        reduced_weights[ticker] = reduced_weights.get(ticker, 0.0) * reduction_mult
                    apply_weight_change(date, reduced_weights)
                    for ticker in to_reduce:
                        twostage_reduced.pop(ticker, None)
            if tickers_to_exit:
                current_top_n = dynamic_top_n(date)
                candidates_daily = select_tickers(
                    features=features, date=date, mode=mode, top_n=current_top_n * 5,
                    min_price=min_price, min_dollar_volume=min_dollar_volume,
                )
                if mode == "bh_1y_sma75_persist_cooldown5" or mode.endswith("_chand_tstop_cd5"):
                    for ticker in tickers_to_exit:
                        cooldown_until[ticker] = date + pd.Timedelta(days=5)
                elif mode == "bh_1y_sma75_persist_cooldown10":
                    for ticker in tickers_to_exit:
                        cooldown_until[ticker] = date + pd.Timedelta(days=10)
                if mode in ("bh_1y_sma75_persist_cooldown5", "bh_1y_sma75_persist_cooldown10") or mode.endswith("_chand_tstop_cd5"):
                    candidates_daily = [
                        ticker for ticker in candidates_daily
                        if cooldown_until.get(ticker) is None or cooldown_until[ticker] < date
                    ]
                kept = [t for t in weights.keys() if t not in tickers_to_exit]
                replacements = [t for t in candidates_daily if t not in kept][: current_top_n - len(kept)]
                final = kept + replacements
                for ticker in tickers_to_exit:
                    entry_ranks.pop(ticker, None)
                    entry_dates.pop(ticker, None)
                    entry_prices.pop(ticker, None)
                    peak_prices.pop(ticker, None)
                    sma_breach_streak.pop(ticker, None)
                    twostage_reduced.pop(ticker, None)
                    staggered_bucket.pop(ticker, None)
                for ticker in replacements:
                    if ticker in candidates_daily:
                        entry_ranks[ticker] = candidates_daily.index(ticker) + 1
                        if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_staggered":
                            staggered_bucket[ticker] = int(date.isocalendar().week) % 2
                apply_rebalance(date, final)

        if date in rebalance_set:
            current_top_n = dynamic_top_n(date)
            candidates = select_tickers(
                features=features,
                date=date,
                mode=mode,
                top_n=current_top_n * 5 if mode in smart_modes else current_top_n,
                min_price=min_price, min_dollar_volume=min_dollar_volume,
            )
            if mode in ("bh_1y_sma75_persist_cooldown5", "bh_1y_sma75_persist_cooldown10") or mode.endswith("_chand_tstop_cd5"):
                candidates = [
                    ticker for ticker in candidates
                    if cooldown_until.get(ticker) is None or cooldown_until[ticker] < date
                ]
            if mode in smart_modes and weights:
                # Smart rebalance: keep current holdings unless rank drops too far
                positions_to_sell: list[str] = []
                active_bucket = int(date.isocalendar().week) % 2
                for ticker in list(weights.keys()):
                    if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_staggered":
                        if staggered_bucket.get(ticker, active_bucket) != active_bucket:
                            continue
                    try:
                        current_rank = candidates.index(ticker) + 1
                    except ValueError:
                        current_rank = 999
                    entry_rank = entry_ranks.get(ticker, current_rank)
                    rank_drop = current_rank - entry_rank
                    max_rank = 20 if mode in (
                        "bh_1y_sma75_persist_rank20",
                        "blend_1y_6m_70_30_sma75_persist_rank20",
                        "blend_1y_6m_45_55_sma75_persist_pos3m_rank20",
                    ) else 10 if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_rank10" else 15
                    rank_drop_threshold = 7 if mode in (
                        "bh_1y_sma75_persist_rankdrop7",
                        "blend_1y_6m_70_30_sma75_persist_rankdrop7",
                        "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop7",
                    ) else 3 if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_rankdrop3" else 5
                    should_sell_rank_based = current_rank > max_rank or rank_drop >= rank_drop_threshold

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
                            if mode in (
                                "blend_1y_6m_45_55_sma75_persist_pos3m_rankgap",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_rankgap",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_momweight2_volexit_twostage_rankgap",
                                "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight_volexit_twostage_rankgap",
                            ):
                                replacement_rank = 999
                                for i, candidate in enumerate(candidates):
                                    if candidate not in weights:
                                        replacement_rank = i + 1
                                        break
                                if replacement_rank + 3 <= current_rank:
                                    positions_to_sell.append(ticker)
                            else:
                                positions_to_sell.append(ticker)

                kept = [t for t in weights.keys() if t not in positions_to_sell]

                if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_staggered":
                    target_active = max(1, current_top_n // 2)
                    active_kept = [t for t in kept if staggered_bucket.get(t, active_bucket) == active_bucket]
                    replacements_needed = max(0, target_active - len(active_kept))
                else:
                    replacements_needed = current_top_n - len(kept)

                replacements = [t for t in candidates if t not in kept][: replacements_needed]
                final = kept + replacements
                for ticker in positions_to_sell:
                    entry_ranks.pop(ticker, None)
                    entry_dates.pop(ticker, None)
                    entry_prices.pop(ticker, None)
                    peak_prices.pop(ticker, None)
                    twostage_reduced.pop(ticker, None)
                    staggered_bucket.pop(ticker, None)
                for ticker in replacements:
                    if ticker in candidates:
                        entry_ranks[ticker] = candidates.index(ticker) + 1
                        if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_staggered":
                            staggered_bucket[ticker] = active_bucket
                apply_rebalance(date, final)
            else:
                picks = candidates[:current_top_n] if mode in smart_modes else candidates
                if mode in smart_modes:
                    for i, ticker in enumerate(picks):
                        entry_ranks[ticker] = i + 1
                        if mode == "blend_1y_6m_45_55_sma75_persist_pos3m_staggered":
                            staggered_bucket[ticker] = i % 2
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
    if args.backtest_days is not None and args.backtest_days <= 0:
        raise ValueError("--backtest-days must be a positive integer")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    close, volume, intra = load_market_data(
        cache_dir=args.data_cache_dir,
        min_history_days=args.min_history_days,
        max_tickers=args.max_tickers,
    )
    features = build_features(
        close=close,
        volume=volume,
        min_universe_size=args.min_universe_size,
        intra=intra,
    )
    rebalance_dates = get_rebalance_dates(
        features=features,
        frequency=args.rebalance_frequency,
        min_history_days=args.min_history_days,
    )
    if not rebalance_dates:
        raise RuntimeError("No rebalance dates available after applying warmup")

    if args.backtest_end_date:
        bt_end = pd.Timestamp(args.backtest_end_date, tz="UTC")
    else:
        bt_end = features.close.index[-1]

    index_upto_end = features.close.index[features.close.index <= bt_end]
    if len(index_upto_end) == 0:
        raise RuntimeError(f"No feature dates found on or before backtest end date {bt_end.date()}")

    if args.backtest_days is not None:
        backtest_dates = list(index_upto_end[-args.backtest_days:])
    else:
        backtest_dates = list(index_upto_end)

    if not backtest_dates:
        raise RuntimeError("Backtest date window is empty")

    first_bt_date = backtest_dates[0]
    rebalance_dates = [d for d in rebalance_dates if first_bt_date <= d <= backtest_dates[-1]]
    if not rebalance_dates or rebalance_dates[0] != first_bt_date:
        rebalance_dates = [first_bt_date] + rebalance_dates

    print(
        f"Rebalance frequency: {args.rebalance_frequency} "
        f"({len(rebalance_dates)} dates, last={rebalance_dates[-1].date().isoformat()})"
    )
    print(
        f"Backtest window: {backtest_dates[0].date().isoformat()} -> {backtest_dates[-1].date().isoformat()} "
        f"({len(backtest_dates)} trading days)"
    )

    results = []
    for mode in modes:
        print(f"Running mode: {mode}")
        result = run_backtest(
            features=features,
            rebalance_dates=rebalance_dates,
            backtest_dates=backtest_dates,
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
