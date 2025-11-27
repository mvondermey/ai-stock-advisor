from __future__ import annotations
"""
Alpha-aware integration for RuleTradingEnv: cached thresholds + logging + exit summary.
Usage in main.py AFTER class RuleTradingEnv is defined:
    from alpha_integration import apply_alpha_thresholds, USE_ALPHA_THRESHOLD_BUY, USE_ALPHA_THRESHOLD_SELL
    USE_ALPHA_THRESHOLD_BUY = True
    USE_ALPHA_THRESHOLD_SELL = True
    apply_alpha_thresholds(RuleTradingEnv)
"""
from typing import Optional
from datetime import datetime
import atexit
import numpy as np
import pandas as pd
from alpha_training import AlphaThresholdConfig, select_threshold_by_alpha

# Feature flags
USE_ALPHA_THRESHOLD_BUY: bool = True
USE_ALPHA_THRESHOLD_SELL: bool = True
ALPHA_HORIZON_DAYS: int = 5
COSTS_BPS: float = 5.0
SLIPPAGE_BPS: float = 2.0
FREQ: str = "D"
METRIC: str = "alpha"  # or 'active_ir'

# Logs & cache
ALPHA_THRESH_LOG = []  # list[dict]
CLAMP_MIN_THR = 0.45
_THRESH_CACHE: dict[tuple, float] = {}

def _thr_key(df, model, horizon_days: int, ticker: str) -> tuple:
    n = len(df) if hasattr(df, '__len__') else 0
    mname = type(model).__name__ if model is not None else 'None'
    return (ticker or 'UNK', int(n), int(horizon_days), mname)

def _compute_model_proba_series(df: pd.DataFrame, model, scaler, RuleTradingEnv) -> pd.Series:
    if df is None or df.empty or "Close" not in df.columns or model is None or scaler is None:
        return pd.Series(dtype=float)
    env_tmp = RuleTradingEnv(
        df=df.copy(),
        ticker="TICK",
        initial_balance=10_000.0,
        transaction_cost=0.0,
        model_buy=model,
        model_sell=None,
        scaler=scaler,
        use_gate=True,
        feature_set=None,
        per_ticker_min_proba_buy=0.5,
        per_ticker_min_proba_sell=0.5,
        use_simple_rule_strategy=False,
    )
    idx, vals = [], []
    for i in range(len(env_tmp.df)):
        try:
            p = env_tmp._get_model_prediction(i, model)  # noqa: SLF001
        except Exception:
            p = 0.0
        vals.append(float(p or 0.0))
        idx.append(pd.to_datetime(env_tmp.df.loc[i, "Date"]) if "Date" in env_tmp.df.columns else i)
    return pd.Series(vals, index=idx).sort_index()

def _future_returns(close: pd.Series, horizon_days: int) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    return (close.shift(-horizon_days) / close - 1.0)

def _select_alpha_threshold(proba: pd.Series, fut: pd.Series) -> float:
    if proba.empty or fut.empty:
        return 0.5
    cfg = AlphaThresholdConfig(
        thresholds=tuple(np.round(np.linspace(0.10, 0.90, 33), 4)),
        rebalance_freq=FREQ,
        costs_bps=COSTS_BPS,
        slippage_bps=SLIPPAGE_BPS,
        metric=METRIC,
    )
    bench = fut
    try:
        t, _, _, _ = select_threshold_by_alpha(proba, fut, bench, cfg)
        return float(t)
    except Exception:
        return 0.5

def _alpha_opt_threshold(df: pd.DataFrame, model, scaler, horizon_days: int, RuleTradingEnv) -> float:
    ticker = df.get('Ticker', ['UNK'])[0] if hasattr(df, 'get') and 'Ticker' in df.columns else (getattr(df, 'name', 'UNK'))
    key = _thr_key(df, model, horizon_days, str(ticker))
    if key in _THRESH_CACHE:
        return _THRESH_CACHE[key]
    proba = _compute_model_proba_series(df, model, scaler, RuleTradingEnv)
    fut = _future_returns(df.set_index('Date')['Close'] if 'Date' in df.columns else df['Close'], horizon_days)
    fut = fut.reindex(proba.index).fillna(0.0)
    t = _select_alpha_threshold(proba, fut)
    _THRESH_CACHE[key] = t
    return t if t >= CLAMP_MIN_THR else CLAMP_MIN_THR

def _append_alpha_log(ticker: str, side: str, thr: float, n: int | None = None) -> None:
    try:
        ALPHA_THRESH_LOG.append({
            "ticker": ticker or "UNK",
            "side": side,
            "thr": float(thr),
            "n": int(n) if n is not None else None,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })
    except Exception:
        pass

def _print_alpha_summary() -> None:
    if not ALPHA_THRESH_LOG:
        return
    last = {}
    for row in ALPHA_THRESH_LOG:
        key = (row.get("ticker","UNK"), row.get("side"))
        last[key] = row
    print("\n=== Alpha Thresholds Summary ===")
    for (tic, side), row in sorted(last.items()):
        print(f"  {tic:>8s} | {side.upper():4s} | thr={row['thr']:.4f} | obs={row.get('n')} | ts={row['ts']}")
    print("=== End Alpha Thresholds Summary ===\n")

try:
    atexit.unregister(_print_alpha_summary)
except Exception:
    pass
atexit.register(_print_alpha_summary)

def apply_alpha_thresholds(RuleTradingEnv) -> None:
    import logging
    _alog = logging.getLogger("alpha_thresholds")
    orig_init = RuleTradingEnv.__init__

    def patched_init(self, *args, **kwargs):
        df = kwargs.get("df", None)
        scaler = kwargs.get("scaler", None)

        if USE_ALPHA_THRESHOLD_BUY:
            buy_thr = kwargs.get("per_ticker_min_proba_buy", None)
            model_buy = kwargs.get("model_buy", None)
            if (buy_thr is None or float(buy_thr) <= 0.0) and df is not None and model_buy is not None and scaler is not None:
                kwargs["per_ticker_min_proba_buy"] = _alpha_opt_threshold(df, model_buy, scaler, ALPHA_HORIZON_DAYS, RuleTradingEnv)
                try:
                    _alog.info("BUY alpha-threshold set | ticker=%s thr=%.4f", kwargs.get("ticker", "UNK"), kwargs["per_ticker_min_proba_buy"])
                    _append_alpha_log(kwargs.get("ticker","UNK"), "buy", kwargs["per_ticker_min_proba_buy"], len(df) if hasattr(df, "__len__") else None)
                except Exception:
                    pass

        if USE_ALPHA_THRESHOLD_SELL:
            sell_thr = kwargs.get("per_ticker_min_proba_sell", None)
            model_sell = kwargs.get("model_sell", None)
            if (sell_thr is None or float(sell_thr) <= 0.0) and df is not None and model_sell is not None and scaler is not None:
                kwargs["per_ticker_min_proba_sell"] = _alpha_opt_threshold(df, model_sell, scaler, ALPHA_HORIZON_DAYS, RuleTradingEnv)
                try:
                    _alog.info("SELL alpha-threshold set | ticker=%s thr=%.4f", kwargs.get("ticker", "UNK"), kwargs["per_ticker_min_proba_sell"])
                    _append_alpha_log(kwargs.get("ticker","UNK"), "sell", kwargs["per_ticker_min_proba_sell"], len(df) if hasattr(df, "__len__") else None)
                except Exception:
                    pass

        return orig_init(self, *args, **kwargs)

    RuleTradingEnv.__init__ = patched_init  # type: ignore[method-assign]
