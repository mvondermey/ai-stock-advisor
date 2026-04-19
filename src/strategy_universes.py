from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import config


StrategySourceEntry = Dict[str, str]


def _source(enable_flag_name: str, value_var: str) -> StrategySourceEntry:
    return {
        "enable_flag_name": enable_flag_name,
        "value_var": value_var,
    }


def get_enabled_strategy_aliases(strategy_sources: Dict[str, StrategySourceEntry]) -> List[str]:
    """Return aliases whose backing config flag is enabled."""
    enabled: List[str] = []
    for alias, source in strategy_sources.items():
        if getattr(config, source["enable_flag_name"], False):
            enabled.append(alias)
    return enabled


def build_strategy_values_from_locals(
    locals_dict: Dict[str, object],
    strategy_sources: Dict[str, StrategySourceEntry],
) -> Dict[str, object]:
    """Build {alias: value} from the current function locals."""
    values: Dict[str, object] = {}
    for alias, source in strategy_sources.items():
        if getattr(config, source["enable_flag_name"], False):
            values[alias] = locals_dict.get(source["value_var"])
        else:
            values[alias] = None
    return values


def build_pairwise_feature_names(strategy_names: Iterable[str]) -> List[Tuple[str, str]]:
    """Build all ordered pairwise feature names from the provided strategy names."""
    return list(combinations(list(strategy_names), 2))


AI_CHAMPION_STRATEGY_SOURCES: Dict[str, StrategySourceEntry] = {
    "ai_elite": _source("ENABLE_AI_ELITE", "ai_elite_portfolio_value"),
    "ai_elite_market_up": _source("ENABLE_AI_ELITE_MARKET_UP", "ai_elite_market_up_portfolio_value"),
    "ai_elite_filtered": _source("ENABLE_AI_ELITE_FILTERED", "ai_elite_filtered_portfolio_value"),
    "multi_tf_ensemble": _source("ENABLE_MULTI_TIMEFRAME_ENSEMBLE", "multi_tf_ensemble_portfolio_value"),
}


AI_REGIME_STRATEGY_SOURCES: Dict[str, StrategySourceEntry] = {
    "risk_adj_mom_3m": _source("ENABLE_RISK_ADJ_MOM_3M", "risk_adj_mom_3m_portfolio_value"),
    "risk_adj_mom_3m_monthly": _source("ENABLE_RISK_ADJ_MOM_3M", "risk_adj_mom_3m_monthly_portfolio_value"),
    "risk_adj_mom_6m": _source("ENABLE_RISK_ADJ_MOM_6M", "risk_adj_mom_6m_portfolio_value"),
    "risk_adj_mom_6m_monthly": _source("ENABLE_RISK_ADJ_MOM_6M", "risk_adj_mom_6m_monthly_portfolio_value"),
    "risk_adj_mom": _source("ENABLE_RISK_ADJ_MOM", "risk_adj_mom_portfolio_value"),
    "risk_adj_mom_1m": _source("ENABLE_RISK_ADJ_MOM_1M", "risk_adj_mom_1m_portfolio_value"),
    "risk_adj_mom_1m_monthly": _source("ENABLE_RISK_ADJ_MOM_1M", "risk_adj_mom_1m_monthly_portfolio_value"),
    "elite_hybrid": _source("ENABLE_ELITE_HYBRID", "elite_hybrid_portfolio_value"),
    "elite_risk": _source("ENABLE_ELITE_RISK", "elite_risk_portfolio_value"),
    "ai_elite": _source("ENABLE_AI_ELITE", "ai_elite_portfolio_value"),
    "momentum_volatility_hybrid_6m": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M", "momentum_volatility_hybrid_6m_portfolio_value"),
    "momentum_volatility_hybrid": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID", "momentum_volatility_hybrid_portfolio_value"),
    "concentrated_3m": _source("ENABLE_CONCENTRATED_3M", "concentrated_3m_portfolio_value"),
    "trend_atr": _source("ENABLE_TREND_FOLLOWING_ATR", "trend_atr_portfolio_value"),
    "dual_momentum": _source("ENABLE_DUAL_MOMENTUM", "dual_mom_portfolio_value"),
    "static_bh_1y": _source("ENABLE_STATIC_BH", "static_bh_1y_portfolio_value"),
    "static_bh_3m": _source("ENABLE_STATIC_BH", "static_bh_3m_portfolio_value"),
    "adaptive_ensemble": _source("ENABLE_ADAPTIVE_STRATEGY", "adaptive_ensemble_portfolio_value"),
}


META_STRATEGY_SOURCES: Dict[str, StrategySourceEntry] = {
    "static_bh_1y": _source("ENABLE_STATIC_BH", "static_bh_1y_portfolio_value"),
    "static_bh_3m": _source("ENABLE_STATIC_BH", "static_bh_3m_portfolio_value"),
    "static_bh_6m": _source("ENABLE_STATIC_BH_6M", "static_bh_6m_portfolio_value"),
    "static_bh_1m": _source("ENABLE_STATIC_BH", "static_bh_1m_portfolio_value"),
    "bh_1y_monthly": _source("ENABLE_STATIC_BH_1Y_MONTHLY", "static_bh_1y_monthly_portfolio_value"),
    "bh_6m_monthly": _source("ENABLE_STATIC_BH_6M_MONTHLY", "static_bh_6m_monthly_portfolio_value"),
    "bh_3m_monthly": _source("ENABLE_STATIC_BH_3M_MONTHLY", "static_bh_3m_monthly_portfolio_value"),
    "bh_1m_monthly": _source("ENABLE_STATIC_BH_1M_MONTHLY", "static_bh_1m_monthly_portfolio_value"),
    "dynamic_bh_1y": _source("ENABLE_DYNAMIC_BH_1Y", "dynamic_bh_portfolio_value"),
    "dynamic_bh_6m": _source("ENABLE_DYNAMIC_BH_6M", "dynamic_bh_6m_portfolio_value"),
    "dynamic_bh_3m": _source("ENABLE_DYNAMIC_BH_3M", "dynamic_bh_3m_portfolio_value"),
    "dynamic_bh_1m": _source("ENABLE_DYNAMIC_BH_1M", "dynamic_bh_1m_portfolio_value"),
    "risk_adj_mom": _source("ENABLE_RISK_ADJ_MOM", "risk_adj_mom_portfolio_value"),
    "risk_adj_mom_3m": _source("ENABLE_RISK_ADJ_MOM_3M", "risk_adj_mom_3m_portfolio_value"),
    "risk_adj_mom_6m": _source("ENABLE_RISK_ADJ_MOM_6M", "risk_adj_mom_6m_portfolio_value"),
    "risk_adj_mom_1m": _source("ENABLE_RISK_ADJ_MOM_1M", "risk_adj_mom_1m_portfolio_value"),
    "mean_reversion": _source("ENABLE_MEAN_REVERSION", "mean_reversion_portfolio_value"),
    "quality_momentum": _source("ENABLE_QUALITY_MOM", "quality_momentum_portfolio_value"),
    "momentum_ai_hybrid": _source("ENABLE_MOMENTUM_AI_HYBRID", "momentum_ai_hybrid_portfolio_value"),
    "volatility_adj_mom": _source("ENABLE_VOLATILITY_ADJ_MOM", "volatility_adj_mom_portfolio_value"),
    "enhanced_volatility": _source("ENABLE_ENHANCED_VOLATILITY", "enhanced_volatility_portfolio_value"),
    "trend_atr": _source("ENABLE_TREND_FOLLOWING_ATR", "trend_atr_portfolio_value"),
    "dual_momentum": _source("ENABLE_DUAL_MOMENTUM", "dual_mom_portfolio_value"),
    "defensive_momentum": _source("ENABLE_DEFENSIVE_MOMENTUM", "defensive_momentum_portfolio_value"),
    "momentum_acceleration": _source("ENABLE_MOMENTUM_ACCELERATION", "mom_accel_portfolio_value"),
    "concentrated_3m": _source("ENABLE_CONCENTRATED_3M", "concentrated_3m_portfolio_value"),
    "price_acceleration": _source("ENABLE_PRICE_ACCELERATION", "price_acceleration_portfolio_value"),
    "price_curvature": _source("ENABLE_PRICE_CURVATURE", "price_curvature_portfolio_value"),
    "foresight_mimic": _source("ENABLE_FORESIGHT_MIMIC", "foresight_mimic_portfolio_value"),
    "bh_1y_1m_rank": _source("ENABLE_BH_1Y_1M_RANK", "bh_1y_1m_rank_portfolio_value"),
    "bh_1y_6m_rank": _source("ENABLE_BH_1Y_6M_RANK", "bh_1y_6m_rank_portfolio_value"),
    "bh_1y_sma200": _source("ENABLE_BH_1Y_SMA200", "bh_1y_sma200_portfolio_value"),
    "bh_1y_fcf_rank": _source("ENABLE_BH_1Y_FCF_RANK", "bh_1y_fcf_rank_portfolio_value"),
    "turnaround": _source("ENABLE_TURNAROUND", "turnaround_portfolio_value"),
    "elite_hybrid": _source("ENABLE_ELITE_HYBRID", "elite_hybrid_portfolio_value"),
    "elite_risk": _source("ENABLE_ELITE_RISK", "elite_risk_portfolio_value"),
    "momentum_volatility_hybrid": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID", "momentum_volatility_hybrid_portfolio_value"),
    "momentum_volatility_hybrid_6m": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M", "momentum_volatility_hybrid_6m_portfolio_value"),
    "momentum_volatility_hybrid_1y": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y", "momentum_volatility_hybrid_1y_portfolio_value"),
    "momentum_volatility_hybrid_1y3m": _source("ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M", "momentum_volatility_hybrid_1y3m_portfolio_value"),
    "ratio_3m_1y": _source("ENABLE_3M_1Y_RATIO", "ratio_3m_1y_portfolio_value"),
    "adaptive_ensemble": _source("ENABLE_ADAPTIVE_STRATEGY", "adaptive_ensemble_portfolio_value"),
    "ai_volatility_ensemble": _source("ENABLE_AI_VOLATILITY_ENSEMBLE", "ai_volatility_ensemble_portfolio_value"),
    "correlation_ensemble": _source("ENABLE_CORRELATION_ENSEMBLE", "correlation_ensemble_portfolio_value"),
    "voting_ensemble": _source("ENABLE_VOTING_ENSEMBLE", "voting_ensemble_portfolio_value"),
    "ai_elite": _source("ENABLE_AI_ELITE", "ai_elite_portfolio_value"),
    "ai_elite_filtered": _source("ENABLE_AI_ELITE_FILTERED", "ai_elite_filtered_portfolio_value"),
    "ai_elite_market_up": _source("ENABLE_AI_ELITE_MARKET_UP", "ai_elite_market_up_portfolio_value"),
    "inverse_etf_hedge": _source("ENABLE_INVERSE_ETF_HEDGE", "inverse_etf_hedge_portfolio_value"),
    "analyst_recommendation": _source("ENABLE_ANALYST_RECOMMENDATION", "analyst_rec_portfolio_value"),
    "static_bh_1y_volatility": _source("ENABLE_STATIC_BH_1Y_VOLATILITY", "static_bh_1y_vol_portfolio_value"),
    "static_bh_1y_performance": _source("ENABLE_STATIC_BH_1Y_PERFORMANCE", "static_bh_1y_perf_portfolio_value"),
    "static_bh_6m_performance": _source("ENABLE_STATIC_BH_6M_PERFORMANCE", "static_bh_6m_perf_portfolio_value"),
    "static_bh_9m_performance": _source("ENABLE_STATIC_BH_9M_PERFORMANCE", "static_bh_9m_perf_portfolio_value"),
    "static_bh_1y_momentum": _source("ENABLE_STATIC_BH_1Y_MOMENTUM", "static_bh_1y_mom_portfolio_value"),
    "static_bh_1y_atr": _source("ENABLE_STATIC_BH_1Y_ATR", "static_bh_1y_atr_portfolio_value"),
    "static_bh_1y_hybrid": _source("ENABLE_STATIC_BH_1Y_HYBRID", "static_bh_1y_hybrid_portfolio_value"),
    "bb_mean_reversion": _source("ENABLE_BB_MEAN_REVERSION", "bb_mean_reversion_value"),
    "bb_breakout": _source("ENABLE_BB_BREAKOUT", "bb_breakout_value"),
    "bb_squeeze_breakout": _source("ENABLE_BB_SQUEEZE_BREAKOUT", "bb_squeeze_breakout_value"),
    "bb_rsi_combo": _source("ENABLE_BB_RSI_COMBO", "bb_rsi_combo_value"),
    "trend_breakout": _source("ENABLE_TREND_BREAKOUT", "trend_breakout_value"),
    "dynamic_bh_1y_vol_filter": _source("ENABLE_DYNAMIC_BH_1Y_VOL_FILTER", "dynamic_bh_1y_vol_filter_portfolio_value"),
    "dynamic_bh_1y_trailing_stop": _source("ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP", "dynamic_bh_1y_trailing_stop_portfolio_value"),
}


META_STATIC_BH_STRATEGIES = {"static_bh_1y", "static_bh_3m"}
META_MONTHLY_BH_STRATEGIES = {"bh_1y_monthly", "bh_6m_monthly", "bh_3m_monthly", "bh_1m_monthly"}
META_DYNAMIC_BH_STRATEGIES = {"dynamic_bh_1y", "dynamic_bh_3m"}
META_RISK_ADJ_MOM_STRATEGIES = {"risk_adj_mom", "risk_adj_mom_3m", "risk_adj_mom_6m", "risk_adj_mom_1m"}
META_RISK_ADJ_MOM_MONTHLY_STRATEGIES = {"risk_adj_mom_3m_monthly", "risk_adj_mom_6m_monthly", "risk_adj_mom_1m_monthly"}
META_RISK_ADJ_MOM_VARIANT_STRATEGIES = {"risk_adj_mom_3m_sentiment", "risk_adj_mom_3m_market_up", "risk_adj_mom_3m_with_stops"}
META_ENHANCED_VOLATILITY_STRATEGIES = {"enhanced_volatility", "enhanced_volatility_6m", "enhanced_volatility_3m"}
META_MOMENTUM_VOLATILITY_HYBRID_STRATEGIES = {
    "momentum_volatility_hybrid",
    "momentum_volatility_hybrid_6m",
    "momentum_volatility_hybrid_1y",
    "momentum_volatility_hybrid_1y3m",
}
META_RATIO_STRATEGIES = {"ratio_3m_1y", "ratio_1y_3m"}
META_ENSEMBLE_STRATEGIES = {
    "adaptive_ensemble",
    "ai_volatility_ensemble",
    "correlation_ensemble",
    "voting_ensemble",
}
META_AI_STRATEGIES = {"ai_elite", "ai_elite_filtered", "ai_elite_market_up"}
META_ADAPTIVE_REBALANCING_STRATEGIES = {
    "static_bh_1y_volatility",
    "static_bh_1y_performance",
    "static_bh_6m_performance",
    "static_bh_9m_performance",
    "static_bh_1y_momentum",
    "static_bh_1y_atr",
    "static_bh_1y_hybrid",
}
META_BB_STRATEGIES = {"bb_mean_reversion", "bb_breakout", "bb_squeeze_breakout", "bb_rsi_combo"}


META_REGIME_HIGH_VOL_CANDIDATES = ("mean_reversion", "risk_adj_mom", "risk_adj_mom_3m", "elite_risk")
META_REGIME_LOW_VOL_CANDIDATES = ("static_bh_1y", "dynamic_bh_1y", "momentum_volatility_hybrid", "defensive_momentum")
META_REGIME_MEDIUM_VOL_CANDIDATES = ("trend_atr", "dual_momentum", "elite_hybrid")
META_ADAPTIVE_CONVEX_DEFENSIVE = ("mean_reversion", "risk_adj_mom", "risk_adj_mom_3m", "elite_risk", "defensive_momentum")
META_ADAPTIVE_CONVEX_MOMENTUM = ("static_bh_1y", "dynamic_bh_1y", "momentum_volatility_hybrid", "momentum_volatility_hybrid_6m", "defensive_momentum")
META_ADAPTIVE_CONVEX_BALANCED = ("trend_atr", "dual_momentum", "elite_hybrid", "enhanced_volatility")
