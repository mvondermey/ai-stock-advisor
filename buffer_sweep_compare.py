#!/usr/bin/env python3
"""
Run a portfolio buffer sweep for BH 1Y / 1M Rank vs BH 1Y Perf Trig.

This script does not modify main.py. Each buffer value is run in a fresh
subprocess with in-memory config overrides only.

Sweep runs are console-only and do not overwrite the shared selection JSON
exports used by live trading.

Examples:
  .venv/bin/python buffer_sweep_compare.py
  .venv/bin/python buffer_sweep_compare.py --buffers 0 2 3 5 7 10 15
  .venv/bin/python buffer_sweep_compare.py --buffers 3 5 7 --backtest-days 37
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
LOGS_DIR = REPO_ROOT / "logs"

DEFAULT_BUFFERS = [0, 2, 3, 5, 7, 10, 15]
DEFAULT_STRATEGIES = ["bh_1y_1m_rank", "static_bh_1y_perf"]
DISPLAY_NAMES = {
    "bh_1y_1m_rank": "BH 1Y / 1M Rank",
    "static_bh_1y_perf": "BH 1Y Perf Trig",
}


CHILD_CODE = r"""
import io
import json
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

cfg = json.loads(os.environ["BUFFER_SWEEP_CONFIG"])
repo_root = Path(cfg["repo_root"])
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

argv = ["main.py", "--strategy", ",".join(cfg["strategies"])]
if cfg.get("no_download", True):
    argv.append("--no-download")
if cfg.get("backtest_days") is not None:
    argv.extend(["--backtest-days", str(cfg["backtest_days"])])
sys.argv = argv

import config  # noqa: E402

config.PORTFOLIO_BUFFER_SIZE = int(cfg["buffer"])
if cfg.get("no_download", True):
    config.ENABLE_DATA_DOWNLOAD = False
config.ENABLE_JSON_OUTPUT = False

import main  # noqa: E402

main._CLI_BACKTEST_STRATEGIES = main._apply_backtest_cli_overrides()

captured = io.StringIO()
result = {
    "buffer": cfg["buffer"],
    "strategies": {},
}

try:
    with redirect_stdout(captured), redirect_stderr(captured):
        main.main()
    text = captured.getvalue()
    for strategy_key in cfg["strategies"]:
        display_name = cfg["display_names"][strategy_key]
        pattern = re.compile(
            rf"(?m)^\s*\d+\s+{re.escape(display_name)}\s+\$\s*([\d,]+)\s+([+-]?\d+(?:\.\d+)?)%",
        )
        matches = pattern.findall(text)
        if matches:
            final_value, return_pct = matches[-1]
            result["strategies"][strategy_key] = {
                "display_name": display_name,
                "final_value": int(final_value.replace(",", "")),
                "return_pct": float(return_pct),
            }
        else:
            result["strategies"][strategy_key] = {
                "display_name": display_name,
                "final_value": None,
                "return_pct": None,
            }
    print(json.dumps(result))
except Exception as exc:
    debug_tail = captured.getvalue()[-6000:]
    error_result = {
        "buffer": cfg["buffer"],
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "captured_tail": debug_tail,
    }
    print(json.dumps(error_result))
    sys.exit(1)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BH 1Y / 1M Rank and BH 1Y Perf Trig across buffer sizes."
    )
    parser.add_argument(
        "--buffers",
        nargs="+",
        type=int,
        default=DEFAULT_BUFFERS,
        help="Buffer sizes to test. Default: %(default)s",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=None,
        help="Optional BACKTEST_DAYS override for quicker experiments.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow downloads during the run. Default uses cache only.",
    )
    return parser.parse_args()


def run_single_buffer(
    python_executable: str,
    buffer_size: int,
    strategies: List[str],
    backtest_days: int | None,
    no_download: bool,
) -> Dict[str, object]:
    env = os.environ.copy()
    env["BUFFER_SWEEP_CONFIG"] = json.dumps(
        {
            "repo_root": str(REPO_ROOT),
            "buffer": buffer_size,
            "strategies": strategies,
            "display_names": DISPLAY_NAMES,
            "backtest_days": backtest_days,
            "no_download": no_download,
        }
    )

    completed = subprocess.run(
        [python_executable, "-c", CHILD_CODE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        details = stdout or stderr or f"Child process failed with code {completed.returncode}"
        raise RuntimeError(f"Buffer {buffer_size} failed:\n{details}")

    json_candidate = stdout
    if stdout:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                json_candidate = stripped
                break

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Buffer {buffer_size} returned non-JSON output:\n{stdout[-4000:]}"
        ) from exc


def write_outputs(results: List[Dict[str, object]], prefix: str) -> tuple[Path, Path]:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = LOGS_DIR / f"{prefix}_{timestamp}.json"
    csv_path = LOGS_DIR / f"{prefix}_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "buffer",
        "bh_1y_1m_rank_value",
        "bh_1y_1m_rank_return_pct",
        "static_bh_1y_perf_value",
        "static_bh_1y_perf_return_pct",
        "value_gap",
        "return_gap_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            strat = row["strategies"]
            one_m = strat["bh_1y_1m_rank"]
            perf = strat["static_bh_1y_perf"]
            writer.writerow(
                {
                    "buffer": row["buffer"],
                    "bh_1y_1m_rank_value": one_m["final_value"],
                    "bh_1y_1m_rank_return_pct": one_m["return_pct"],
                    "static_bh_1y_perf_value": perf["final_value"],
                    "static_bh_1y_perf_return_pct": perf["return_pct"],
                    "value_gap": (
                        None
                        if one_m["final_value"] is None or perf["final_value"] is None
                        else one_m["final_value"] - perf["final_value"]
                    ),
                    "return_gap_pct": (
                        None
                        if one_m["return_pct"] is None or perf["return_pct"] is None
                        else round(one_m["return_pct"] - perf["return_pct"], 4)
                    ),
                }
            )

    return json_path, csv_path


def print_summary(results: List[Dict[str, object]]) -> None:
    print()
    print("buffer | BH 1Y / 1M Rank | BH 1Y Perf Trig | delta")
    print("-------+------------------+-----------------+-------")
    for row in results:
        one_m = row["strategies"]["bh_1y_1m_rank"]
        perf = row["strategies"]["static_bh_1y_perf"]
        if one_m["final_value"] is None or perf["final_value"] is None:
            delta = "n/a"
        else:
            delta = f"{one_m['final_value'] - perf['final_value']:+,}"
        one_m_txt = (
            "n/a"
            if one_m["final_value"] is None
            else f"${one_m['final_value']:,} ({one_m['return_pct']:+.2f}%)"
        )
        perf_txt = (
            "n/a"
            if perf["final_value"] is None
            else f"${perf['final_value']:,} ({perf['return_pct']:+.2f}%)"
        )
        print(f"{row['buffer']:>6} | {one_m_txt:<16} | {perf_txt:<15} | {delta}")


def main() -> int:
    args = parse_args()
    results: List[Dict[str, object]] = []

    print(f"Using Python: {sys.executable}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Buffers: {args.buffers}")
    if args.backtest_days is not None:
        print(f"Backtest days override: {args.backtest_days}")
    print(f"Downloads enabled: {args.download}")

    for buffer_size in args.buffers:
        print(f"\n=== Running buffer {buffer_size} ===", flush=True)
        result = run_single_buffer(
            python_executable=sys.executable,
            buffer_size=buffer_size,
            strategies=DEFAULT_STRATEGIES,
            backtest_days=args.backtest_days,
            no_download=not args.download,
        )
        results.append(result)

        one_m = result["strategies"]["bh_1y_1m_rank"]
        perf = result["strategies"]["static_bh_1y_perf"]
        print(
            f"buffer={buffer_size} | "
            f"BH 1Y / 1M Rank={one_m['final_value']} ({one_m['return_pct']:+.2f}%) | "
            f"BH 1Y Perf Trig={perf['final_value']} ({perf['return_pct']:+.2f}%)",
            flush=True,
        )

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
