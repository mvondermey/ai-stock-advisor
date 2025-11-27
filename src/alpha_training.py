# src/alpha_training.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import math
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class AlphaThresholdConfig:
    thresholds: Iterable[float] = tuple(np.linspace(0.1, 0.9, 33))
    rebalance_freq: str = "D"     # 'D','W','M'
    costs_bps: float = 5.0        # one-way fee
    slippage_bps: float = 2.0     # per trade
    metric: str = "alpha"         # 'alpha' | 'active_ir'
    risk_free_rate_annual: float = 0.0

def _infer_ppy(freq: str) -> int:
    return {"D": 252, "W": 52, "M": 12}.get(freq.upper(), 252)

def _bps_to_frac(bps: float) -> float:
    return bps / 10_000.0

def _ann_factor(freq: str) -> float:
    ppy = _infer_ppy(freq)
    return float(ppy)

def _ols_alpha_beta(
    r_s: pd.Series,
    r_b: pd.Series,
    rf: float,
    freq: str,
) -> tuple[float, float]:
    r_s = r_s.align(r_b, join="inner")[0]
    r_b = r_b.align(r_s, join="inner")[0]
    x = r_b - rf / _infer_ppy(freq)
    y = r_s - rf / _infer_ppy(freq)
    x = x.fillna(0.0).values
    y = y.fillna(0.0).values
    if len(y) < 2 or np.allclose(x.std(), 0.0):
        return 0.0, 0.0
    x_ = np.column_stack([np.ones_like(x), x])
    beta_hat = np.linalg.lstsq(x_, y, rcond=None)[0]
    alpha_per_period = float(beta_hat[0])
    beta = float(beta_hat[1])
    alpha_annual = alpha_per_period * _ann_factor(freq)
    return alpha_annual, beta

def _active_information_ratio(active_ret: pd.Series, freq: str) -> float:
    ppy = _infer_ppy(freq)
    mu = active_ret.mean() * ppy
    vol = active_ret.std(ddof=0) * math.sqrt(ppy)
    return float(mu / vol) if vol > 0 else 0.0

def _strategy_returns_from_entries(
    future_returns: pd.Series,
    entries: pd.Series,
    freq: str,
    costs_bps: float,
    slippage_bps: float,
) -> pd.Series:
    entries = entries.reindex_like(future_returns).fillna(0).astype(int)
    turns = entries.diff().abs().fillna(0)
    costs = (turns * (_bps_to_frac(costs_bps) + _bps_to_frac(slippage_bps))).astype(float)
    r = (future_returns * entries).fillna(0.0) - costs
    r.name = "r_strategy"
    return r

def select_threshold_by_alpha(
    proba: pd.Series,
    future_returns: pd.Series,
    bench_future_returns: pd.Series,
    cfg: AlphaThresholdConfig,
) -> tuple[float, pd.Series, float, dict]:
    assert proba.index.equals(future_returns.index)
    assert future_returns.index.equals(bench_future_returns.index)

    best_t, best_score = 0.5, -np.inf
    best_entries: Optional[pd.Series] = None

    for t in cfg.thresholds:
        entries = (proba >= t).astype(int)
        r_s = _strategy_returns_from_entries(
            future_returns=future_returns,
            entries=entries,
            freq=cfg.rebalance_freq,
            costs_bps=cfg.costs_bps,
            slippage_bps=cfg.slippage_bps,
        )
        r_b = bench_future_returns
        if cfg.metric == "alpha":
            alpha, beta = _ols_alpha_beta(r_s, r_b, rf=cfg.risk_free_rate_annual, freq=cfg.rebalance_freq)
            score = alpha
        else:
            active = (r_s - r_b).fillna(0.0)
            score = _active_information_ratio(active, cfg.rebalance_freq)

        if score > best_score:
            best_score = score
            best_t = t
            best_entries = entries

    if best_entries is None:
        best_entries = (proba >= best_t).astype(int)

    diag = {
        "metric": cfg.metric,
        "score": float(best_score),
        "thresholds_tested": len(tuple(cfg.thresholds)),
    }
    return best_t, best_entries, float(best_score), diag

def alpha_sample_weights(
    future_returns: pd.Series,
    bench_future_returns: pd.Series,
    clip: float = 0.05,
) -> pd.Series:
    active = (future_returns - bench_future_returns).fillna(0.0)
    w = active.clip(lower=-clip, upper=clip) / clip
    w_pos = (w - w.min()) / (w.max() - w.min() + 1e-12) + 1e-6
    w_pos.name = "alpha_weight"
    return w_pos

def alpha_aware_thresholding(
    proba: pd.Series,
    future_returns: pd.Series,
    bench_future_returns: pd.Series,
    freq: str = "D",
    metric: str = "alpha",
) -> tuple[float, pd.Series, float]:
    cfg = AlphaThresholdConfig(rebalance_freq=freq, metric=metric)
    t, entries, score, _ = select_threshold_by_alpha(
        proba=proba,
        future_returns=future_returns,
        bench_future_returns=bench_future_returns,
        cfg=cfg,
    )
    return t, entries, score
