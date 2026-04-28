#!/usr/bin/env python3
"""
Parity and timing validation for cache-backed selectors.

This script compares explicit shared-cache execution against the implicit
path across the full shared strategy registry and dumps a summary table.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_utils import load_all_market_data
from parallel_backtest import build_price_history_cache


def _dataframe_to_markdown_without_tabulate(df: pd.DataFrame) -> str:
    """Render a simple GitHub-flavored markdown table without optional deps."""
    columns = [str(column) for column in df.columns]
    rows = []
    for _, series in df.iterrows():
        row = []
        for value in series.tolist():
            if pd.isna(value):
                text = ""
            else:
                text = str(value)
            row.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append(row)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _dataframe_to_terminal_table(df: pd.DataFrame) -> str:
    """Render a compact fixed-width table for terminal output."""
    display_df = df.copy()

    for column in ("implicit_seconds", "explicit_seconds", "delta_seconds"):
        if column in display_df.columns:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}s")

    if "speedup_x" in display_df.columns:
        def _format_speedup(value) -> str:
            if pd.isna(value):
                return ""
            if value == float("inf"):
                return "inf"
            return f"{float(value):.2f}x"

        display_df["speedup_x"] = display_df["speedup_x"].map(_format_speedup)

    if "match" in display_df.columns:
        display_df["match"] = display_df["match"].map(lambda value: "yes" if bool(value) else "no")

    columns = list(display_df.columns)
    string_rows = []
    for _, row in display_df.iterrows():
        string_rows.append([
            "" if pd.isna(value) else str(value).replace("\n", " ")
            for value in row.tolist()
        ])

    widths = []
    for idx, column in enumerate(columns):
        max_row_width = max((len(row[idx]) for row in string_rows), default=0)
        widths.append(max(len(str(column)), max_row_width))

    right_align = {
        "implicit_seconds",
        "explicit_seconds",
        "delta_seconds",
        "speedup_x",
        "implicit_count",
        "explicit_count",
    }

    def _format_row(values: list[str]) -> str:
        cells = []
        for idx, value in enumerate(values):
            column = columns[idx]
            width = widths[idx]
            if column in right_align:
                cells.append(value.rjust(width))
            else:
                cells.append(value.ljust(width))
        return "  ".join(cells)

    header = _format_row([str(column) for column in columns])
    separator = "  ".join("-" * width for width in widths)
    body = [_format_row(row) for row in string_rows]
    return "\n".join([header, separator, *body])


def _disable_ai_retraining_for_validation() -> None:
    """Freeze most AI retraining during validation, but allow AI Champion rebuilds."""
    config.ENABLE_WALK_FORWARD_RETRAINING = False
    config.AI_ELITE_RETRAIN_DAYS = 10**9
    config.AI_REBALANCE_RETRAIN_DAYS = 10**9
    config.AI_REGIME_RETRAIN_DAYS = 10**9
    config.UNIVERSAL_MODEL_RETRAIN_DAYS = 10**9
    os.environ["CACHE_VALIDATION_DISABLE_AI_TRAINING"] = "1"

    try:
        import shared_strategies
        from ai_elite_strategy import select_ai_elite_stocks
        from model_training_safety import restore_native_model_artifacts

        original_select_ai_elite_with_training = shared_strategies.select_ai_elite_with_training

        def _load_saved_ai_elite_base_model(base_model_path: Path):
            if not base_model_path.exists():
                return None

            try:
                import joblib

                try:
                    model_data = joblib.load(base_model_path)
                except Exception:
                    with open(base_model_path, "rb") as handle:
                        model_data = pickle.load(handle)

                if isinstance(model_data, dict) and "all_models" in model_data:
                    loaded_model = model_data
                elif isinstance(model_data, dict) and "model" in model_data:
                    loaded_model = model_data["model"]
                else:
                    loaded_model = model_data

                return restore_native_model_artifacts(loaded_model, str(base_model_path))
            except Exception as exc:
                print(f"   ⚠️ AI validation override: failed to load saved AI Elite model: {exc}")
                return None

        def _select_ai_elite_without_retraining(
            all_tickers: list,
            ticker_data_grouped: dict,
            current_date=None,
            top_n: int = 10,
            ai_elite_models: dict = None,
            force_train: bool = False,
            cache_start_date=None,
            cache_end_date=None,
        ):
            del force_train, cache_start_date, cache_end_date

            models = ai_elite_models if ai_elite_models is not None else {}
            if models.get("_shared_base") is None:
                base_model_path = Path("logs/models/_shared_base_ai_elite.joblib")
                loaded_model = _load_saved_ai_elite_base_model(base_model_path)
                if loaded_model is not None:
                    models["_shared_base"] = loaded_model
                    for ticker in all_tickers:
                        models[ticker] = loaded_model

            selected = select_ai_elite_stocks(
                all_tickers=all_tickers,
                ticker_data_grouped=ticker_data_grouped,
                current_date=current_date,
                top_n=top_n,
                per_ticker_models=models,
            )
            return selected, models

        if getattr(shared_strategies.select_ai_elite_with_training, "__name__", "") != "_select_ai_elite_without_retraining":
            shared_strategies._original_select_ai_elite_with_training_for_validation = original_select_ai_elite_with_training
            shared_strategies.select_ai_elite_with_training = _select_ai_elite_without_retraining
    except Exception as exc:
        print(f"   ⚠️ AI validation override: could not patch AI Elite training wrapper: {exc}")

def _wire_analyst_data_for_validation() -> None:
    """Use the real analyst-data fetch/select path during validation."""
    try:
        import shared_strategies
        from analyst_recommendation_strategy import (
            fetch_all_analyst_data,
            select_analyst_recommendation_stocks,
        )

        analyst_data_cache: Dict[str, pd.DataFrame] = {}

        def _select_analyst_rec_with_real_data(all_tickers, ticker_data_grouped, current_date, top_n):
            nonlocal analyst_data_cache

            if not analyst_data_cache:
                print(f"   📊 Analyst Recommendation: Fetching analyst data for {len(all_tickers)} tickers...")
                analyst_data_cache.update(
                    fetch_all_analyst_data(all_tickers, max_workers=10, show_progress=False)
                )
                print(f"   📊 Analyst data available for {len(analyst_data_cache)} tickers")

            return select_analyst_recommendation_stocks(
                tickers=all_tickers,
                ticker_data_grouped=ticker_data_grouped,
                analyst_data=analyst_data_cache,
                current_date=current_date,
                top_n=top_n,
                lookback_days=config.ANALYST_LOOKBACK_DAYS,
                min_actions=config.ANALYST_MIN_ACTIONS,
            )

        shared_strategies._select_analyst_rec_stocks = _select_analyst_rec_with_real_data
    except Exception as exc:
        print(f"   ⚠️ Analyst validation override: could not wire real analyst data: {exc}")


def _load_grouped_data() -> Dict[str, pd.DataFrame]:
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = repo_root / "data_cache"
    all_tickers = sorted(path.stem for path in cache_dir.glob("*.csv"))
    if not all_tickers:
        raise RuntimeError(f"No cached ticker CSVs found in {cache_dir}")

    # Validation should never trigger network downloads; use only on-disk cache.
    config.ENABLE_DATA_DOWNLOAD = False
    all_tickers_data = load_all_market_data(all_tickers)
    if all_tickers_data.empty:
        raise RuntimeError("No market data loaded")

    ticker_data_grouped: Dict[str, pd.DataFrame] = {}
    grouped = all_tickers_data.groupby("ticker")
    for ticker in all_tickers_data["ticker"].unique():
        try:
            ticker_df = grouped.get_group(ticker).copy()
            if "date" in ticker_df.columns:
                ticker_df = ticker_df.set_index("date")
            ticker_df = ticker_df.drop("ticker", axis=1, errors="ignore")
            ticker_data_grouped[ticker] = ticker_df
        except KeyError:
            continue
    return ticker_data_grouped


def _latest_current_date(ticker_data_grouped: Dict[str, pd.DataFrame]) -> datetime:
    latest = max(df.index.max() for df in ticker_data_grouped.values() if len(df) > 0)
    ts = pd.Timestamp(latest)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _get_last_n_business_days(ticker_data_grouped: Dict[str, pd.DataFrame], n: int) -> List[datetime]:
    """Get the last N business days from the data, sorted oldest to newest."""
    # Get all unique dates across all tickers
    all_dates = set()
    for df in ticker_data_grouped.values():
        if len(df) > 0:
            all_dates.update(df.index.tolist())
    
    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < n:
        n = len(sorted_dates)
    
    last_n = sorted_dates[-n:]
    result = []
    for d in last_n:
        ts = pd.Timestamp(d)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        result.append(ts.to_pydatetime())
    return result


def _run_with_timing(fn: Callable[[], List[str]]) -> tuple[List[str], float]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return list(result or []), elapsed


def _safe_run_with_timing(fn: Callable[[], List[str]]) -> tuple[List[str], float, Optional[str]]:
    start = time.perf_counter()
    try:
        result = list(fn() or [])
        return result, time.perf_counter() - start, None
    except Exception as exc:  # pragma: no cover - validation helper
        return [], time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def _normalize_selection_list(result) -> List[str]:
    """Flatten nested selection containers into a simple list of ticker strings."""
    normalized: List[str] = []

    def _append_items(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            normalized.append(value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _append_items(item)
            return
        normalized.append(str(value))

    _append_items(result)
    return normalized


def _build_reset_map() -> Dict[str, Callable[[], None]]:
    reset_map: Dict[str, Callable[[], None]] = {}

    try:
        from adaptive_ensemble import reset_ensemble_state
        reset_map["adaptive_ensemble"] = reset_ensemble_state
    except Exception:
        pass

    try:
        from volatility_ensemble import reset_vol_ensemble_state
        reset_map["volatility_ensemble"] = reset_vol_ensemble_state
    except Exception:
        pass

    try:
        from new_strategies import reset_trend_atr_state
        reset_map["trend_atr"] = reset_trend_atr_state
    except Exception:
        pass

    try:
        from bollinger_bands_strategy import reset_bb_strategy_states
        for key in ["bb_mean_reversion", "bb_breakout", "bb_squeeze_breakout", "bb_rsi_combo"]:
            reset_map[key] = reset_bb_strategy_states
    except Exception:
        pass

    try:
        from factor_rotation import reset_factor_rotation_state
        reset_map["factor_rotation"] = reset_factor_rotation_state
    except Exception:
        pass

    try:
        from sentiment_ensemble import reset_sentiment_ensemble_state
        reset_map["sentiment_ensemble"] = reset_sentiment_ensemble_state
    except Exception:
        pass

    try:
        from earnings_momentum import reset_earnings_state
        reset_map["earnings_momentum"] = reset_earnings_state
    except Exception:
        pass

    try:
        from insider_trading import reset_insider_state
        reset_map["insider_trading"] = reset_insider_state
    except Exception:
        pass

    try:
        from options_sentiment import reset_options_state
        reset_map["options_sentiment"] = reset_options_state
    except Exception:
        pass

    try:
        from pairs_trading import reset_pairs_state
        reset_map["pairs_trading"] = reset_pairs_state
    except Exception:
        pass

    try:
        from momentum_breakout import reset_breakout_state
        reset_map["momentum_breakout"] = reset_breakout_state
    except Exception:
        pass

    return reset_map


def _make_registry_runner(
    strategy_name: str,
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    reset_map: Dict[str, Callable[[], None]],
) -> Callable[..., List[str]]:
    from shared_strategies import get_strategy_tickers

    def runner(*, price_history_cache=None) -> List[str]:
        reset_fn = reset_map.get(strategy_name)
        if reset_fn is not None:
            reset_fn()
        return get_strategy_tickers(
            strategy_name,
            tickers,
            ticker_data_grouped,
            current_date,
            top_n,
            price_history_cache=price_history_cache,
            raise_on_error=True,
        )

    return runner


def _make_multi_day_runner(
    strategy_name: str,
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    dates: List[datetime],
    top_n: int,
    reset_map: Dict[str, Callable[[], None]],
) -> Callable[..., tuple[List[List[str]], float]]:
    """
    Create a runner that executes a strategy over multiple consecutive days.
    This properly tests stateful strategies by simulating a mini-backtest loop.
    Returns all daily selections and accumulates total time.
    """
    from shared_strategies import get_strategy_tickers

    def runner(*, price_history_cache=None, run_label: str = "") -> tuple[List[List[str]], float]:
        # Reset state before multi-day run
        reset_fn = reset_map.get(strategy_name)
        if reset_fn is not None:
            reset_fn()

        total_time = 0.0
        daily_selections: List[List[str]] = []
        progress_desc = f"{strategy_name} [{run_label or 'run'}]"

        for current_date in tqdm(
            dates,
            desc=progress_desc,
            unit="day",
            leave=False,
        ):
            start = time.perf_counter()
            try:
                result = get_strategy_tickers(
                    strategy_name,
                    tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n,
                    price_history_cache=price_history_cache,
                    raise_on_error=True,
                )
            except Exception as exc:
                date_text = pd.Timestamp(current_date).strftime("%Y-%m-%d")
                raise RuntimeError(f"{strategy_name} failed on {date_text}: {exc}") from exc
            elapsed = time.perf_counter() - start
            total_time += elapsed
            daily_selections.append(_normalize_selection_list(result))

        return daily_selections, total_time

    return runner


def _safe_run_multi_day(
    runner: Callable, price_history_cache, run_label: str
) -> tuple[List[List[str]], float, Optional[str]]:
    """Safely run a multi-day runner and return (daily_selections, total_time, error)."""
    try:
        daily_selections, total_time = runner(
            price_history_cache=price_history_cache,
            run_label=run_label,
        )
        return daily_selections, total_time, None
    except Exception as exc:
        return [], 0.0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    from strategy_metadata_registry import get_canonical_strategy_names

    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else int(config.PORTFOLIO_SIZE)
    num_days = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    _disable_ai_retraining_for_validation()
    print("Most AI retraining disabled for validation; AI Champion rebuilds remain enabled.")

    print("Loading grouped market data...")
    ticker_data_grouped = _load_grouped_data()
    tickers = list(ticker_data_grouped.keys())
    dates = _get_last_n_business_days(ticker_data_grouped, num_days)
    print(f"Loaded {len(tickers)} tickers")
    print(f"Testing over {len(dates)} days: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")
    _wire_analyst_data_for_validation()

    print("Building shared price history cache...")
    price_history_cache = build_price_history_cache(ticker_data_grouped)

    strategy_names = get_canonical_strategy_names()
    reset_map = _build_reset_map()

    print(f"\nStrategy validation ({num_days}-day loop for stateful strategies)")
    print(f"Registry strategies: {len(strategy_names)}")
    print(f"Top N: {top_n} (from config.PORTFOLIO_SIZE by default)")
    print("-" * 160)
    print(
        f"{'Strategy':<34} {'Implicit':>10} {'Explicit':>10} {'Match':>8} "
        f"{'ImpCnt':>6} {'ExpCnt':>6} {'Status':<12} {'MismatchDay':<12} Selections"
    )
    print("-" * 160)

    rows = []
    mismatches = []
    failures = []

    for name in strategy_names:
        runner = _make_multi_day_runner(name, tickers, ticker_data_grouped, dates, top_n, reset_map)

        implicit_daily, implicit_time, implicit_error = _safe_run_multi_day(runner, None, "implicit")
        explicit_daily, explicit_time, explicit_error = _safe_run_multi_day(
            runner,
            price_history_cache,
            "explicit",
        )

        first_mismatch_day = ""
        match = implicit_error is None and explicit_error is None
        if match:
            if len(implicit_daily) != len(explicit_daily):
                match = False
                first_mismatch_day = "day-count"
            else:
                for day_idx, (implicit_selection, explicit_selection) in enumerate(
                    zip(implicit_daily, explicit_daily),
                    start=1,
                ):
                    if implicit_selection != explicit_selection:
                        match = False
                        first_mismatch_day = dates[day_idx - 1].strftime("%Y-%m-%d")
                        break

        final_implicit_selection = implicit_daily[-1] if implicit_daily else []
        final_explicit_selection = explicit_daily[-1] if explicit_daily else []
        status = "ok"
        if implicit_error or explicit_error:
            status = "error"
            failures.append(name)
        elif not match:
            status = "mismatch"
            mismatches.append(name)

        preview = ",".join(final_explicit_selection[: min(top_n, len(final_explicit_selection))])
        row = {
            "strategy": name,
            "implicit_seconds": round(implicit_time, 6),
            "explicit_seconds": round(explicit_time, 6),
            "match": match,
            "implicit_count": len(final_implicit_selection),
            "explicit_count": len(final_explicit_selection),
            "status": status,
            "first_mismatch_day": first_mismatch_day,
            "implicit_error": implicit_error or "",
            "explicit_error": explicit_error or "",
            "selections": preview,
        }
        rows.append(row)

        print(
            f"{name:<34} {implicit_time:>9.3f}s {explicit_time:>9.3f}s {str(match):>8} "
            f"{len(final_implicit_selection):>6} {len(final_explicit_selection):>6} "
            f"{status:<12} {first_mismatch_day or '-':<12} {preview}"
        )

    print("-" * 160)

    summary_df = pd.DataFrame(rows)
    summary_df["delta_seconds"] = (
        summary_df["implicit_seconds"] - summary_df["explicit_seconds"]
    ).round(6)
    summary_df["speedup_x"] = summary_df.apply(
        lambda row: round(
            row["implicit_seconds"] / row["explicit_seconds"],
            3,
        )
        if row["explicit_seconds"] not in (0, 0.0)
        else float("inf"),
        axis=1,
    )
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "cache_validation_summary.csv"
    md_path = repo_root / "cache_validation_summary.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(_dataframe_to_markdown_without_tabulate(summary_df) + "\n", encoding="utf-8")

    print("\nSummary table")
    print("-" * 160)
    print(
        _dataframe_to_terminal_table(
            summary_df[
                [
                    "strategy",
                    "implicit_seconds",
                    "explicit_seconds",
                    "delta_seconds",
                    "speedup_x",
                    "match",
                    "implicit_count",
                    "explicit_count",
                    "status",
                    "first_mismatch_day",
                ]
            ]
        )
    )
    print("-" * 160)
    print(f"Summary CSV: {csv_path}")
    print(f"Summary Markdown: {md_path}")

    if failures:
        print(f"Failures ({len(failures)}): {', '.join(failures)}")
    if mismatches:
        print(f"Mismatches ({len(mismatches)}): {', '.join(mismatches)}")

    if not failures and not mismatches:
        print("All explicit-cache runs matched the implicit path.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
