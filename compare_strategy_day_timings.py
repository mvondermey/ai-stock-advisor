#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


STRATEGY_RE = re.compile(r"^\s*(?:📊\s+)?(.*?)\s+Day\s+(\d+):")
PARALLEL_RE = re.compile(r"^\s*⏱️\s+Parallel processing:\s+\d+\s+tickers in\s+([0-9.]+)s")
CACHED_PERF_RE = re.compile(
    r"^\s*⏱️\s+Cached performance:\s+\d+\s+tickers in\s+([0-9.]+)s\s+\(window=(\d+)d\)"
)
CACHED_RA_RE = re.compile(
    r"^\s*⏱️\s+Cached risk-adjusted:\s+\d+\s+tickers in\s+([0-9.]+)s\s+\(window=(\d+)d\)"
)
PARALLEL_RA_RE = re.compile(
    r"^\s*⏱️\s+Parallel risk-adjusted:\s+\d+\s+tickers in\s+([0-9.]+)s"
)
COLLECT_RE = re.compile(
    r"^\s*📊\s+(.*?)\:\s+Collected\s+\d+\s+samples from\s+\d+\s+tickers\s+\(([0-9.]+)s\)"
)
PREDICT_RE = re.compile(
    r"^\s*📊\s+(.*?)\:\s+Predicted\s+\d+\s+tickers\s+\(([0-9.]+)s\)"
)


def parse_timing(line: str):
    match = PARALLEL_RE.match(line)
    if match:
        return float(match.group(1)), "parallel"

    match = CACHED_PERF_RE.match(line)
    if match:
        return float(match.group(1)), f"cached({match.group(2)}d)"

    match = CACHED_RA_RE.match(line)
    if match:
        return float(match.group(1)), f"cached-ra({match.group(2)}d)"

    match = PARALLEL_RA_RE.match(line)
    if match:
        return float(match.group(1)), "parallel-ra"

    match = COLLECT_RE.match(line)
    if match:
        return float(match.group(2)), "collect"

    match = PREDICT_RE.match(line)
    if match:
        return float(match.group(2)), "predict"

    return None


def parse_log(log_path: Path):
    rows: dict[str, dict[int, tuple[float, str] | None]] = {}
    pending_timing = None

    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timing = parse_timing(line)
            if timing is not None:
                pending_timing = timing
                continue

            match = STRATEGY_RE.match(line)
            if not match:
                continue

            strategy_name = match.group(1).strip()
            day = int(match.group(2))
            rows.setdefault(strategy_name, {})[day] = pending_timing
            pending_timing = None

    return rows


def fmt_timing(value):
    if not value:
        return ""
    seconds, label = value
    return f"{seconds:.2f}s {label}"


def fmt_delta(day_a, day_b):
    if not day_a or not day_b:
        return ""

    delta = day_b[0] - day_a[0]
    pct = (delta / day_a[0] * 100.0) if day_a[0] else 0.0
    return f"{delta:+.2f}s ({pct:+.0f}%)"


def build_table(rows, day_a: int, day_b: int, only_changed: bool):
    header = ["Strategy", f"Day {day_a}", f"Day {day_b}", "Delta"]
    output = [" | ".join(header), " | ".join(["---"] * len(header))]

    for strategy_name in sorted(rows):
        first = rows[strategy_name].get(day_a)
        second = rows[strategy_name].get(day_b)
        if not first and not second:
            continue
        if only_changed and first and second and abs(first[0] - second[0]) < 1e-9:
            continue
        output.append(
            " | ".join(
                [
                    strategy_name,
                    fmt_timing(first),
                    fmt_timing(second),
                    fmt_delta(first, second),
                ]
            )
        )

    return "\n".join(output)


def build_cross_log_table(
    old_rows: dict,
    new_rows: dict,
    day: int,
    old_label: str,
    new_label: str,
    only_changed: bool,
):
    header = ["Strategy", f"{old_label} Day {day}", f"{new_label} Day {day}", "Delta"]
    output = [" | ".join(header), " | ".join(["---"] * len(header))]

    strategies = sorted(set(old_rows) | set(new_rows))
    for strategy_name in strategies:
        old_val = old_rows.get(strategy_name, {}).get(day)
        new_val = new_rows.get(strategy_name, {}).get(day)
        if not old_val and not new_val:
            continue
        if only_changed and old_val and new_val and abs(old_val[0] - new_val[0]) < 1e-9:
            continue
        output.append(
            " | ".join(
                [
                    strategy_name,
                    fmt_timing(old_val),
                    fmt_timing(new_val),
                    fmt_delta(old_val, new_val),
                ]
            )
        )

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Compare timing lines attached to the next strategy Day N entry in backtest logs."
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        default="output.log",
        help="Path to the primary log file. Defaults to output.log.",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        default=None,
        help="Path to a second log file for cross-log comparison.",
    )
    parser.add_argument("--day-a", type=int, default=1, help="First day to compare. Default: 1")
    parser.add_argument("--day-b", type=int, default=2, help="Second day to compare (single-log mode). Default: 2")
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Show only strategies where both entries exist and the timing changed.",
    )
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    if args.compare_to:
        compare_path = Path(args.compare_to)
        if not compare_path.exists():
            raise SystemExit(f"Comparison log file not found: {compare_path}")
        old_rows = parse_log(compare_path)
        new_rows = parse_log(log_path)
        print(build_cross_log_table(
            old_rows,
            new_rows,
            day=args.day_a,
            old_label=compare_path.stem,
            new_label=log_path.stem,
            only_changed=args.only_changed,
        ))
    else:
        rows = parse_log(log_path)
        print(build_table(rows, args.day_a, args.day_b, args.only_changed))


if __name__ == "__main__":
    main()
