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
