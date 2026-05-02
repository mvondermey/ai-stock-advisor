from __future__ import annotations

import re
from typing import Iterable


_ALIASES = {
    "aielite": "ai_elite",
    "ai_elite_strategy": "ai_elite",
    "ai_elite_monthly": "ai_elite_monthly_shared",
    "1m_volsweet": "risk_adj_mom_1m_vol_sweet",
    "volsweet_mom": "vol_sweet_mom",
    "vol_sweet": "vol_sweet_mom",
    "rebal_1y_voladj": "bh_1y_vol_adj_rebal",
    "bh_1y_voladj": "bh_1y_vol_adj_rebal",
    "mom_vol_hybrid": "momentum_volatility_hybrid",
    "mom_vol_6m": "momentum_volatility_hybrid_6m",
    "mom_vol_1y": "momentum_volatility_hybrid_1y",
    "risk_adj_3m": "risk_adj_mom_3m",
    "risk_adj_6m": "risk_adj_mom_6m",
    "risk_adj_1m": "risk_adj_mom_1m",
    "risk_adj_1y": "risk_adj_mom",
    "elite": "elite_hybrid",
    "dynamic_1y": "dynamic_bh_1y",
    "dynamic_6m": "dynamic_bh_6m",
    "dynamic_3m": "dynamic_bh_3m",
    "static_1y": "static_bh_1y",
    "static_6m": "static_bh_6m",
    "static_3m": "static_bh_3m",
    "bh_6m_perf": "static_bh_6m_perf",
    "bh_9m_perf": "static_bh_9m_perf",
    "bh_1y_1m_rank": "bh_1y_1m_rank",
    "bh_1y_6m_rank": "bh_1y_6m_rank",
    "bh_1y_6m_blend": "bh_1y_6m_blend",
    "blend_1y_6m_45_55_sma75_persist3_pos3m": "blend_1y_6m_45_55_sma75_persist3_pos3m",
    "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage": "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
    "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop": "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
    "early_leader_accel": "early_leader_accel",
    "bh_1y_weekly_start": "bh_1y_weekly_start",
    "bh_1y_weekly_end": "bh_1y_weekly_end",
    "1y_6m": "bh_1y_6m_rank",
    "1y_6m_rank": "bh_1y_6m_rank",
    "1y_6m_blend": "bh_1y_6m_blend",
    "blend_45_55_pos3m": "blend_1y_6m_45_55_sma75_persist3_pos3m",
    "early_winners_blend": "blend_1y_6m_45_55_sma75_persist3_pos3m",
    "blend_45_55_liqweight2_volexit_twostage": "blend_1y_6m_45_55_sma75_persist_pos3m_liqweight2_volexit_twostage",
    "blend_30_70_momweight4_chand_tstop": "blend_1y_6m_30_70_sma75_persist_pos3m_momweight4_volexit_twostage_chand_tstop",
    "early_winners": "early_leader_accel",
    "early_leaders": "early_leader_accel",
    "bh_1y_sma200": "bh_1y_sma200",
    "bh_1y_fcf_rank": "bh_1y_fcf_rank",
    "1y_fcf": "bh_1y_fcf_rank",
    "1y_fcf_rank": "bh_1y_fcf_rank",
    "1y_sma200": "bh_1y_sma200",
    "1y_sma": "bh_1y_sma200",
    "1y_1m": "bh_1y_1m_rank",
    "1y_1m_rank": "bh_1y_1m_rank",
    "1y_weekly_start": "bh_1y_weekly_start",
    "1y_week_start": "bh_1y_weekly_start",
    "weekly_start_1y": "bh_1y_weekly_start",
    "1y_weekly_end": "bh_1y_weekly_end",
    "1y_week_end": "bh_1y_weekly_end",
    "weekly_end_1y": "bh_1y_weekly_end",
    "foresight": "foresight_mimic",
    "foresight_mimic": "foresight_mimic",
    "sma50_6m_delta_ratio": "sma50_1y_delta_ratio",
    "sma50_6m_ratio": "sma50_1y_delta_ratio",
    "recovery": "deep_recovery",
    "contrarian": "deep_recovery",
    "pmc": "persistent_momentum_core",
    "persistent_momentum": "persistent_momentum_core",
}


def _canonicalize(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def get_strategy_registry() -> dict:
    """Return the canonical backtest strategy registry."""
    from backtesting import STRATEGY_REGISTRY

    return STRATEGY_REGISTRY


def get_strategy_names() -> list[str]:
    """Return canonical strategy names from the shared registry."""
    return sorted(get_strategy_registry().keys())


def normalize_strategy_name(strategy_name: str, available_strategies: Iterable[str] | None = None) -> str:
    """Normalize user input to a canonical registry strategy name when possible."""
    available = list(available_strategies) if available_strategies is not None else get_strategy_names()
    if strategy_name in available:
        return strategy_name

    normalized = _canonicalize(strategy_name)
    if normalized in available:
        return normalized

    lookup = {_canonicalize(name): name for name in available}
    if normalized in lookup:
        return lookup[normalized]

    alias_target = _ALIASES.get(normalized)
    if alias_target and alias_target in available:
        return alias_target

    matches = [name for name in available if normalized in _canonicalize(name)]
    if len(matches) == 1:
        return matches[0]

    return strategy_name


def parse_strategy_list(strategy_arg: str, available_strategies: Iterable[str] | None = None) -> list[str]:
    """Split and normalize a comma-separated strategy list."""
    return [
        normalize_strategy_name(name, available_strategies)
        for name in strategy_arg.split(",")
        if name.strip()
    ]
