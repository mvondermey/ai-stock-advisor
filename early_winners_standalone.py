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
    log_1m: pd.DataFrame
    log_3m: pd.DataFrame
    log_6m: pd.DataFrame
    log_1y: pd.DataFrame
    sma20: pd.DataFrame
    sma50: pd.DataFrame
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

    log_1m = np.log(filtered_close / filtered_close.shift(21)) * (252.0 / 21.0)
    log_3m = np.log(filtered_close / filtered_close.shift(63)) * (252.0 / 63.0)
    log_6m = np.log(filtered_close / filtered_close.shift(126)) * (252.0 / 126.0)
    log_1y = np.log(filtered_close / filtered_close.shift(252))

    sma20 = filtered_close.rolling(20).mean()
    sma50 = filtered_close.rolling(50).mean()
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
        log_1m=log_1m,
        log_3m=log_3m,
        log_6m=log_6m,
        log_1y=log_1y,
        sma20=sma20,
        sma50=sma50,
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

    for date in features.close.index:
        if weights:
            daily_move = features.daily_returns.loc[date, list(weights.keys())].dropna()
            if not daily_move.empty:
                equity *= 1.0 + float((daily_move * pd.Series(weights)).sum())

        if date in rebalance_set:
            picks = select_tickers(
                features=features,
                date=date,
                mode=mode,
                top_n=top_n,
                min_price=min_price,
                min_dollar_volume=min_dollar_volume,
            )
            if picks:
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
