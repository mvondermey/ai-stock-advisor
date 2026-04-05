"""
backtesting.py
Final version – includes 1D sequential optimisation, compatible with main.py and accepts extra kwargs (e.g., top_tickers).
"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
from backtesting_env import RuleTradingEnv
from ml_models import initialize_ml_libraries, train_and_evaluate_models
from data_utils import load_prices, fetch_training_data, _ensure_dir, _calculate_technical_indicators
import logging
from pathlib import Path
from scipy import stats
from config import (
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    TRANSACTION_COST, SEED, INVESTMENT_PER_STOCK, PORTFOLIO_SIZE,
    BACKTEST_DAYS,
    N_TOP_TICKERS, USE_PERFORMANCE_BENCHMARK, PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, NUM_PROCESSES, SEQUENCE_LENGTH,
    RETRAIN_FREQUENCY_DAYS, PREDICTION_LOOKBACK_DAYS, PREDICTION_TIMEOUT,
    VOLATILITY_ADJ_MOM_LOOKBACK, VOLATILITY_ADJ_MOM_VOL_WINDOW, VOLATILITY_ADJ_MOM_MIN_SCORE,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M, MIN_DATA_DAYS_1M, MIN_DATA_DAYS_GENERAL,
    CALENDAR_DAYS_PER_YEAR,
    CONCENTRATED_3M_REBALANCE_DAYS,
    AI_ELITE_RETRAIN_DAYS, AI_ELITE_TRAINING_LOOKBACK, AI_ELITE_FORWARD_DAYS, AI_ELITE_INTRADAY_LOOKBACK
)
import signal
from contextlib import contextmanager
from strategy_universes import (
    AI_CHAMPION_STRATEGY_SOURCES,
    AI_REGIME_STRATEGY_SOURCES,
    build_strategy_values_from_locals,
)

class PredictionTimeoutError(Exception):
    """Raised when a prediction takes too long."""
    pass

def calculate_daily_returns(history: List[float]) -> List[float]:
    """Calculate daily returns from portfolio history."""
    returns = []
    for i in range(1, len(history)):
        if history[i-1] > 0 and not pd.isna(history[i-1]) and history[i] > 0 and not pd.isna(history[i]):
            daily_ret = (history[i] - history[i-1]) / history[i-1] * 100
            returns.append(daily_ret)
    return returns

def calculate_std_dev(history: List[float]) -> float:
    """Calculate standard deviation of daily returns (annualized)."""
    returns = calculate_daily_returns(history)
    if len(returns) < 2:
        return 0.0
    return np.std(returns) * np.sqrt(252)  # Annualized

def paired_t_test(history1: List[float], history2: List[float]) -> Tuple[float, float]:
    """Perform paired t-test on two strategy returns.
    Returns: (t_statistic, p_value)
    """
    returns1 = calculate_daily_returns(history1)
    returns2 = calculate_daily_returns(history2)

    if len(returns1) < 2 or len(returns2) < 2:
        return 0.0, 1.0

    # Ensure same length
    min_len = min(len(returns1), len(returns2))
    returns1 = returns1[:min_len]
    returns2 = returns2[:min_len]

    if min_len < 2:
        return 0.0, 1.0

    t_stat, p_value = stats.ttest_rel(returns1, returns2)
    return t_stat, p_value

@contextmanager
def prediction_timeout(seconds: int, ticker: str):
    """Context manager for prediction timeout using SIGALRM (Unix only)."""
    def timeout_handler(signum, frame):
        raise PredictionTimeoutError(f"Prediction for {ticker} timed out after {seconds}s")

    if seconds is None or seconds <= 0:
        yield
        return

    # Only use signal-based timeout on Unix systems
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # SIGALRM not available (Windows), just run without timeout
        yield
import config  # Import config module for dynamic ENABLE_ flag access
from config import (
    ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, PERIOD_HORIZONS,
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, USE_LSTM, USE_GRU,
    PORTFOLIO_SIZE, MOMENTUM_AI_HYBRID_BUY_THRESHOLD,
    MOMENTUM_AI_HYBRID_SELL_THRESHOLD, MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK,
    MOMENTUM_AI_HYBRID_STOP_LOSS, MOMENTUM_AI_HYBRID_TRAILING_STOP,
    STATIC_BH_1Y_REBALANCE_DAYS, STATIC_BH_6M_REBALANCE_DAYS, STATIC_BH_3M_REBALANCE_DAYS, STATIC_BH_1M_REBALANCE_DAYS,
    DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY,
    DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT,
    AI_REBALANCE_FREQUENCY_DAYS, STOP_LOSS_PCT, STRATEGY_STOP_LOSS_PCT, PORTFOLIO_BUFFER_SIZE,
    RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_CONFIRM_SHORT, RISK_ADJ_MOM_CONFIRM_MEDIUM, RISK_ADJ_MOM_CONFIRM_LONG, RISK_ADJ_MOM_MIN_CONFIRMATIONS,
    RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION, RISK_ADJ_MOM_VOLUME_WINDOW, RISK_ADJ_MOM_VOLUME_MULTIPLIER,
    OPTIMIZE_REBALANCE_HORIZON,
    AI_ELITE_FORCE_FRESH_TRAIN,
    ANALYST_LOOKBACK_DAYS, ANALYST_MIN_ACTIONS, ANALYST_REBALANCE_DAYS,
    STATIC_BH_1Y_VOLUME_MIN_DAILY, STATIC_BH_1Y_SECTOR_MAX_PER_SECTOR,
    STATIC_BH_1Y_PERFORMANCE_MIN_IMPROVEMENT, STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_HIGH_VOL_DAYS, STATIC_BH_1Y_MARKET_REGIME_LOW_VOL_DAYS,
    STATIC_BH_1Y_MARKET_REGIME_VOL_THRESHOLD,
    BH_1Y_MOM_SELL_LOOKBACK_SHORT, BH_1Y_MOM_SELL_LOOKBACK_LONG,
    BH_1Y_RANK_SELL_DROP_THRESHOLD, BH_1Y_RANK_SELL_MAX_RANK,
    BH_1Y_TRAILING_MOM_SOFT_STOP, BH_1Y_TRAILING_MOM_HARD_STOP,
    BH_1Y_VOLUME_CONFIRM_MULTIPLIER,
    BH_1Y_SECTOR_AWARE_MIN_PER_SECTOR, BH_1Y_SECTOR_AWARE_MAX_RANK,
    BH_1Y_ACCEL_BUY_WEIGHT,
    STATIC_BH_3M_ACCEL_LOOKBACK_SHORT, STATIC_BH_3M_ACCEL_LOOKBACK_LONG, STATIC_BH_3M_ACCEL_WEIGHT,
    STATIC_BH_1Y_MOMENTUM_PERSIST_DAYS, STATIC_BH_1Y_MOMENTUM_PERSIST_MIN_OVERLAP,
    STATIC_BH_1Y_OVERLAP_THRESHOLD,
    STATIC_BH_1Y_RANK_DRIFT_THRESHOLD, STATIC_BH_1Y_DRAWDOWN_THRESHOLD,
    STATIC_BH_1Y_SMART_MONTHLY_DRAWDOWN,
    BH_1Y_VOL_ADJ_REBAL_LOW_VOL_THRESH, BH_1Y_VOL_ADJ_REBAL_HIGH_VOL_THRESH, BH_1Y_VOL_ADJ_REBAL_MIN_DAYS, BH_1Y_VOL_ADJ_REBAL_MAX_DAYS,
    BH_1Y_CORR_FILTER_THRESH, BH_1Y_CORR_FILTER_LOOKBACK,
    BH_1Y_REGIME_BULL_MA, BH_1Y_REGIME_BEAR_MA, BH_1Y_REGIME_REBAL_BULL, BH_1Y_REGIME_REBAL_BEAR, BH_1Y_REGIME_REBAL_SIDEWAYS,
    BH_1Y_RISK_PARITY_VOL_LOOKBACK, BH_1Y_RISK_PARITY_MIN_WEIGHT, BH_1Y_RISK_PARITY_MAX_WEIGHT,
    BH_1Y_DRIFT_THRESH_TARGET, BH_1Y_DRIFT_THRESH_TRIGGER,
    BH_1Y_MOM_QUALITY_MOM_WEIGHT, BH_1Y_MOM_QUALITY_CONS_WEIGHT, BH_1Y_MOM_QUALITY_MIN_QUALITY,
    BH_1Y_LIQUIDITY_AVG_VOL_DAYS, BH_1Y_LIQUIDITY_MIN_DOLLAR_VOL, BH_1Y_LIQUIDITY_MAX_WEIGHT,
    BH_1Y_EARNINGS_AVOID_DAYS_BEFORE, BH_1Y_EARNINGS_AVOID_DAYS_AFTER,
    BH_1Y_MULTI_FACTOR_MOM_WEIGHT, BH_1Y_MULTI_FACTOR_VAL_WEIGHT, BH_1Y_MULTI_FACTOR_QUAL_WEIGHT,
    BH_1Y_TIME_DECAY_EXIT_DAYS, BH_1Y_TIME_DECAY_EXIT_PCT_DAILY,
    TREND_BREAKOUT_USE_ATR_STOP, TREND_BREAKOUT_ATR_MULTIPLIER,
)
# Helper function to get ENABLE_ flags dynamically from config module
def _get_enable_flag(flag_name: str, default: bool = False) -> bool:
    """Get ENABLE_ flag from config module dynamically (not cached at import time)."""
    return getattr(config, flag_name, default)
from scipy.stats import uniform, beta

# =============================================================================
# DAILY STRATEGY REGISTRY
# Single source of truth for day-level strategy aggregation metadata used by:
# - `_build_daily_strategy_data()`
# - daily summary ranking
# - `top5_consistency_blend`
# - `voting_ensemble`
# =============================================================================
def _strategy_registry_entry(
    display_name: str,
    enable_flag_name: str,
    value_var: str,
    history_var: str,
    cash_var: str,
    positions_var: str,
) -> Dict[str, str]:
    return {
        'display_name': display_name,
        'enable_flag_name': enable_flag_name,
        'value_var': value_var,
        'history_var': history_var,
        'cash_var': cash_var,
        'positions_var': positions_var,
    }


STRATEGY_REGISTRY = {
    'static_bh_1y': _strategy_registry_entry('Static BH 1Y', 'ENABLE_STATIC_BH', 'static_bh_1y_portfolio_value', 'static_bh_1y_portfolio_history', 'static_bh_1y_cash', 'current_static_bh_1y_stocks'),
    'static_bh_6m': _strategy_registry_entry('Static BH 6M', 'ENABLE_STATIC_BH', 'static_bh_6m_portfolio_value', 'static_bh_6m_portfolio_history', 'static_bh_6m_cash', 'current_static_bh_6m_stocks'),
    'static_bh_3m': _strategy_registry_entry('Static BH 3M', 'ENABLE_STATIC_BH', 'static_bh_3m_portfolio_value', 'static_bh_3m_portfolio_history', 'static_bh_3m_cash', 'current_static_bh_3m_stocks'),
    'static_bh_1m': _strategy_registry_entry('Static BH 1M', 'ENABLE_STATIC_BH', 'static_bh_1m_portfolio_value', 'static_bh_1m_portfolio_history', 'static_bh_1m_cash', 'current_static_bh_1m_stocks'),
    'dynamic_bh_1y': _strategy_registry_entry('Dynamic BH 1Y', 'ENABLE_DYNAMIC_BH_1Y', 'dynamic_bh_portfolio_value', 'dynamic_bh_portfolio_history', 'dynamic_bh_cash', 'current_dynamic_bh_stocks'),
    'dynamic_bh_6m': _strategy_registry_entry('Dynamic BH 6M', 'ENABLE_DYNAMIC_BH_6M', 'dynamic_bh_6m_portfolio_value', 'dynamic_bh_6m_portfolio_history', 'dynamic_bh_6m_cash', 'current_dynamic_bh_6m_stocks'),
    'dynamic_bh_3m': _strategy_registry_entry('Dynamic BH 3M', 'ENABLE_DYNAMIC_BH_3M', 'dynamic_bh_3m_portfolio_value', 'dynamic_bh_3m_portfolio_history', 'dynamic_bh_3m_cash', 'current_dynamic_bh_3m_stocks'),
    'dynamic_bh_1m': _strategy_registry_entry('Dynamic BH 1M', 'ENABLE_DYNAMIC_BH_1M', 'dynamic_bh_1m_portfolio_value', 'dynamic_bh_1m_portfolio_history', 'dynamic_bh_1m_cash', 'current_dynamic_bh_1m_stocks'),
    'dynamic_bh_1y_vol_filter': _strategy_registry_entry('Dynamic BH 1Y+Vol', 'ENABLE_DYNAMIC_BH_1Y_VOL_FILTER', 'dynamic_bh_1y_vol_filter_portfolio_value', 'dynamic_bh_1y_vol_filter_portfolio_history', 'dynamic_bh_1y_vol_filter_cash', 'current_dynamic_bh_1y_vol_filter_stocks'),
    'dynamic_bh_1y_trailing_stop': _strategy_registry_entry('Dynamic BH 1Y+TS', 'ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP', 'dynamic_bh_1y_trailing_stop_portfolio_value', 'dynamic_bh_1y_trailing_stop_portfolio_history', 'dynamic_bh_1y_trailing_stop_cash', 'current_dynamic_bh_1y_trailing_stop_stocks'),
    'risk_adj_mom': _strategy_registry_entry('Risk-Adj Mom', 'ENABLE_RISK_ADJ_MOM', 'risk_adj_mom_portfolio_value', 'risk_adj_mom_portfolio_history', 'risk_adj_mom_cash', 'current_risk_adj_mom_stocks'),
    'mean_reversion': _strategy_registry_entry('Mean Reversion', 'ENABLE_MEAN_REVERSION', 'mean_reversion_portfolio_value', 'mean_reversion_portfolio_history', 'mean_reversion_cash', 'current_mean_reversion_stocks'),
    'quality_momentum': _strategy_registry_entry('Quality+Mom', 'ENABLE_QUALITY_MOM', 'quality_momentum_portfolio_value', 'quality_momentum_portfolio_history', 'quality_momentum_cash', 'current_quality_momentum_stocks'),
    'momentum_ai_hybrid': _strategy_registry_entry('Momentum+AI', 'ENABLE_MOMENTUM_AI_HYBRID', 'momentum_ai_hybrid_portfolio_value', 'momentum_ai_hybrid_portfolio_history', 'momentum_ai_hybrid_cash', 'momentum_ai_hybrid_positions'),
    'volatility_adj_mom': _strategy_registry_entry('Vol-Adj Mom', 'ENABLE_VOLATILITY_ADJ_MOM', 'volatility_adj_mom_portfolio_value', 'volatility_adj_mom_portfolio_history', 'volatility_adj_mom_cash', 'current_volatility_adj_mom_stocks'),
    'sector_rotation': _strategy_registry_entry('Sector Rotation', 'ENABLE_SECTOR_ROTATION', 'sector_rotation_portfolio_value', 'sector_rotation_portfolio_history', 'sector_rotation_cash', 'current_sector_rotation_etfs'),
    'ratio_3m_1y': _strategy_registry_entry('3M/1Y Ratio', 'ENABLE_3M_1Y_RATIO', 'ratio_3m_1y_portfolio_value', 'ratio_3m_1y_portfolio_history', 'ratio_3m_1y_cash', 'ratio_3m_1y_positions'),
    'ratio_1y_3m': _strategy_registry_entry('1Y/3M Ratio', 'ENABLE_3M_1Y_RATIO', 'ratio_1y_3m_portfolio_value', 'ratio_1y_3m_portfolio_history', 'ratio_1y_3m_cash', 'ratio_1y_3m_positions'),
    'ratio_1m_3m': _strategy_registry_entry('1M/3M Ratio', 'ENABLE_1M_3M_RATIO', 'ratio_1m_3m_portfolio_value', 'ratio_1m_3m_portfolio_history', 'ratio_1m_3m_cash', 'ratio_1m_3m_positions'),
    'momentum_volatility_hybrid': _strategy_registry_entry('Mom-Vol Hybrid', 'ENABLE_MOMENTUM_VOLATILITY_HYBRID', 'momentum_volatility_hybrid_portfolio_value', 'momentum_volatility_hybrid_portfolio_history', 'momentum_volatility_hybrid_cash', 'momentum_volatility_hybrid_positions'),
    'momentum_volatility_hybrid_6m': _strategy_registry_entry('Mom-Vol Hybrid 6M', 'ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M', 'momentum_volatility_hybrid_6m_portfolio_value', 'momentum_volatility_hybrid_6m_portfolio_history', 'momentum_volatility_hybrid_6m_cash', 'momentum_volatility_hybrid_6m_positions'),
    'momentum_volatility_hybrid_1y': _strategy_registry_entry('Mom-Vol Hybrid 1Y', 'ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y', 'momentum_volatility_hybrid_1y_portfolio_value', 'momentum_volatility_hybrid_1y_portfolio_history', 'momentum_volatility_hybrid_1y_cash', 'momentum_volatility_hybrid_1y_positions'),
    'momentum_volatility_hybrid_1y3m': _strategy_registry_entry('Mom-Vol Hybrid 1Y/3M', 'ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M', 'momentum_volatility_hybrid_1y3m_portfolio_value', 'momentum_volatility_hybrid_1y3m_portfolio_history', 'momentum_volatility_hybrid_1y3m_cash', 'momentum_volatility_hybrid_1y3m_positions'),
    'price_acceleration': _strategy_registry_entry('Price Acceleration', 'ENABLE_PRICE_ACCELERATION', 'price_acceleration_portfolio_value', 'price_acceleration_portfolio_history', 'price_acceleration_cash', 'current_price_acceleration_stocks'),
    'turnaround': _strategy_registry_entry('Turnaround', 'ENABLE_TURNAROUND', 'turnaround_portfolio_value', 'turnaround_portfolio_history', 'turnaround_cash', 'current_turnaround_stocks'),
    'adaptive_ensemble': _strategy_registry_entry('Adaptive Ensemble', 'ENABLE_ADAPTIVE_STRATEGY', 'adaptive_ensemble_portfolio_value', 'adaptive_ensemble_portfolio_history', 'adaptive_ensemble_cash', 'adaptive_ensemble_positions'),
    'volatility_ensemble': _strategy_registry_entry('Volatility Ensemble', 'ENABLE_VOLATILITY_ENSEMBLE', 'volatility_ensemble_portfolio_value', 'volatility_ensemble_portfolio_history', 'volatility_ensemble_cash', 'volatility_ensemble_positions'),
    'enhanced_volatility': _strategy_registry_entry('Enhanced Volatility', 'ENABLE_ENHANCED_VOLATILITY', 'enhanced_volatility_portfolio_value', 'enhanced_volatility_portfolio_history', 'enhanced_volatility_cash', 'enhanced_volatility_positions'),
    'ai_volatility_ensemble': _strategy_registry_entry('AI Volatility Ensemble', 'ENABLE_AI_VOLATILITY_ENSEMBLE', 'ai_volatility_ensemble_portfolio_value', 'ai_volatility_ensemble_portfolio_history', 'ai_volatility_ensemble_cash', 'ai_volatility_ensemble_positions'),
    'multi_tf_ensemble': _strategy_registry_entry('Multi-Horizon Ensemble', 'ENABLE_MULTI_TIMEFRAME_ENSEMBLE', 'multi_tf_ensemble_portfolio_value', 'multi_tf_ensemble_portfolio_history', 'multi_tf_ensemble_cash', 'multi_tf_ensemble_positions'),
    'multi_tf_intraday_ensemble': _strategy_registry_entry('Multi-Horizon Intraday', 'ENABLE_MULTI_TIMEFRAME_INTRADAY_ENSEMBLE', 'multi_tf_intraday_ensemble_portfolio_value', 'multi_tf_intraday_ensemble_portfolio_history', 'multi_tf_intraday_ensemble_cash', 'multi_tf_intraday_ensemble_positions'),
    'correlation_ensemble': _strategy_registry_entry('Correlation Ensemble', 'ENABLE_CORRELATION_ENSEMBLE', 'correlation_ensemble_portfolio_value', 'correlation_ensemble_portfolio_history', 'correlation_ensemble_cash', 'correlation_ensemble_positions'),
    'dynamic_pool': _strategy_registry_entry('Dynamic Pool', 'ENABLE_DYNAMIC_POOL', 'dynamic_pool_portfolio_value', 'dynamic_pool_portfolio_history', 'dynamic_pool_cash', 'dynamic_pool_positions'),
    'risk_adj_mom_sentiment': _strategy_registry_entry('RiskAdj Sent', 'ENABLE_RISK_ADJ_MOM_SENTIMENT', 'risk_adj_mom_sentiment_portfolio_value', 'risk_adj_mom_sentiment_portfolio_history', 'risk_adj_mom_sentiment_cash', 'risk_adj_mom_sentiment_positions'),
    'voting_ensemble': _strategy_registry_entry('Voting Ensemble', 'ENABLE_VOTING_ENSEMBLE', 'voting_ensemble_portfolio_value', 'voting_ensemble_portfolio_history', 'voting_ensemble_cash', 'voting_ensemble_positions'),
    'mom_accel': _strategy_registry_entry('Mom Acceleration', 'ENABLE_MOMENTUM_ACCELERATION', 'mom_accel_portfolio_value', 'mom_accel_portfolio_history', 'mom_accel_cash', 'mom_accel_positions'),
    'concentrated_3m': _strategy_registry_entry('Concentrated 3M', 'ENABLE_CONCENTRATED_3M', 'concentrated_3m_portfolio_value', 'concentrated_3m_portfolio_history', 'concentrated_3m_cash', 'concentrated_3m_positions'),
    'dual_momentum': _strategy_registry_entry('Dual Momentum', 'ENABLE_DUAL_MOMENTUM', 'dual_mom_portfolio_value', 'dual_mom_portfolio_history', 'dual_mom_cash', 'dual_mom_positions'),
    'trend_atr': _strategy_registry_entry('Trend ATR', 'ENABLE_TREND_FOLLOWING_ATR', 'trend_atr_portfolio_value', 'trend_atr_portfolio_history', 'trend_atr_cash', 'trend_atr_positions'),
    'elite_hybrid': _strategy_registry_entry('Elite Hybrid', 'ENABLE_ELITE_HYBRID', 'elite_hybrid_portfolio_value', 'elite_hybrid_portfolio_history', 'elite_hybrid_cash', 'elite_hybrid_positions'),
    'elite_risk': _strategy_registry_entry('Elite Risk', 'ENABLE_ELITE_RISK', 'elite_risk_portfolio_value', 'elite_risk_portfolio_history', 'elite_risk_cash', 'elite_risk_positions'),
    'risk_adj_mom_6m': _strategy_registry_entry('Risk-Adj Mom 6M', 'ENABLE_RISK_ADJ_MOM_6M', 'risk_adj_mom_6m_portfolio_value', 'risk_adj_mom_6m_portfolio_history', 'risk_adj_mom_6m_cash', 'current_risk_adj_mom_6m_stocks'),
    'risk_adj_mom_6m_monthly': _strategy_registry_entry('RiskAdj 6M Mth', 'ENABLE_RISK_ADJ_MOM_6M_MONTHLY', 'risk_adj_mom_6m_monthly_portfolio_value', 'risk_adj_mom_6m_monthly_portfolio_history', 'risk_adj_mom_6m_monthly_cash', 'risk_adj_mom_6m_monthly_positions'),
    'risk_adj_mom_3m': _strategy_registry_entry('Risk-Adj Mom 3M', 'ENABLE_RISK_ADJ_MOM_3M', 'risk_adj_mom_3m_portfolio_value', 'risk_adj_mom_3m_portfolio_history', 'risk_adj_mom_3m_cash', 'current_risk_adj_mom_3m_stocks'),
    'risk_adj_mom_3m_monthly': _strategy_registry_entry('RiskAdj 3M Mth', 'ENABLE_RISK_ADJ_MOM_3M_MONTHLY', 'risk_adj_mom_3m_monthly_portfolio_value', 'risk_adj_mom_3m_monthly_portfolio_history', 'risk_adj_mom_3m_monthly_cash', 'risk_adj_mom_3m_monthly_positions'),
    'risk_adj_mom_3m_sentiment': _strategy_registry_entry('RiskAdj 3M Sent', 'ENABLE_RISK_ADJ_MOM_3M_SENTIMENT', 'risk_adj_mom_3m_sentiment_portfolio_value', 'risk_adj_mom_3m_sentiment_portfolio_history', 'risk_adj_mom_3m_sentiment_cash', 'risk_adj_mom_3m_sentiment_positions'),
    'risk_adj_mom_3m_market_up': _strategy_registry_entry('RiskAdj 3M Up', 'ENABLE_RISK_ADJ_MOM_3M_MARKET_UP', 'risk_adj_mom_3m_market_up_portfolio_value', 'risk_adj_mom_3m_market_up_portfolio_history', 'risk_adj_mom_3m_market_up_cash', 'risk_adj_mom_3m_market_up_positions'),
    'risk_adj_mom_3m_with_stops': _strategy_registry_entry('RiskAdj 3M Stop', 'ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS', 'risk_adj_mom_3m_with_stops_portfolio_value', 'risk_adj_mom_3m_with_stops_portfolio_history', 'risk_adj_mom_3m_with_stops_cash', 'risk_adj_mom_3m_with_stops_positions'),
    'vol_sweet_mom': _strategy_registry_entry('VolSweet Mom', 'ENABLE_VOL_SWEET_MOM', 'vol_sweet_mom_portfolio_value', 'vol_sweet_mom_portfolio_history', 'vol_sweet_mom_cash', 'vol_sweet_mom_positions'),
    'risk_adj_mom_1m_vol_sweet': _strategy_registry_entry('1M VolSweet', 'ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET', 'risk_adj_mom_1m_vol_sweet_portfolio_value', 'risk_adj_mom_1m_vol_sweet_portfolio_history', 'risk_adj_mom_1m_vol_sweet_cash', 'current_risk_adj_mom_1m_vol_sweet_stocks'),
    'bh_1y_volsweet_accel': _strategy_registry_entry('BH 1Y VolSweet Accel', 'ENABLE_BH_1Y_VOL_SWEET_ACCEL', 'bh_1y_volsweet_accel_portfolio_value', 'bh_1y_volsweet_accel_portfolio_history', 'bh_1y_volsweet_accel_cash', 'current_bh_1y_volsweet_accel_stocks'),
    'bh_1y_dynamic_accel': _strategy_registry_entry('BH 1Y Dynamic Accel', 'ENABLE_BH_1Y_DYNAMIC_ACCEL', 'bh_1y_dynamic_accel_portfolio_value', 'bh_1y_dynamic_accel_portfolio_history', 'bh_1y_dynamic_accel_cash', 'bh_1y_dynamic_accel_positions'),
    'risk_adj_mom_1m': _strategy_registry_entry('Risk-Adj Mom 1M', 'ENABLE_RISK_ADJ_MOM_1M', 'risk_adj_mom_1m_portfolio_value', 'risk_adj_mom_1m_portfolio_history', 'risk_adj_mom_1m_cash', 'current_risk_adj_mom_1m_stocks'),
    'risk_adj_mom_1m_monthly': _strategy_registry_entry('RiskAdj 1M Mth', 'ENABLE_RISK_ADJ_MOM_1M_MONTHLY', 'risk_adj_mom_1m_monthly_portfolio_value', 'risk_adj_mom_1m_monthly_portfolio_history', 'risk_adj_mom_1m_monthly_cash', 'risk_adj_mom_1m_monthly_positions'),
    'ai_elite': _strategy_registry_entry('AI Elite', 'ENABLE_AI_ELITE', 'ai_elite_portfolio_value', 'ai_elite_portfolio_history', 'ai_elite_cash', 'ai_elite_positions'),
    'ai_elite_monthly': _strategy_registry_entry('AI Elite Mth', 'ENABLE_AI_ELITE_MONTHLY', 'ai_elite_monthly_portfolio_value', 'ai_elite_monthly_portfolio_history', 'ai_elite_monthly_cash', 'ai_elite_monthly_positions'),
    'ai_elite_filtered': _strategy_registry_entry('AI Elite Flt', 'ENABLE_AI_ELITE_FILTERED', 'ai_elite_filtered_portfolio_value', 'ai_elite_filtered_portfolio_history', 'ai_elite_filtered_cash', 'ai_elite_filtered_positions'),
    'ai_elite_market_up': _strategy_registry_entry('AI Elite Mkt-Up', 'ENABLE_AI_ELITE_MARKET_UP', 'ai_elite_market_up_portfolio_value', 'ai_elite_market_up_portfolio_history', 'ai_elite_market_up_cash', 'ai_elite_market_up_positions'),
    'ai_champion': _strategy_registry_entry('AI Champion', 'ENABLE_AI_CHAMPION', 'ai_champion_portfolio_value', 'ai_champion_portfolio_history', 'ai_champion_cash', 'ai_champion_positions'),
    'ai_regime': _strategy_registry_entry('AI Regime', 'ENABLE_AI_REGIME', 'ai_regime_portfolio_value', 'ai_regime_portfolio_history', 'ai_regime_cash', 'ai_regime_positions'),
    'ai_regime_monthly': _strategy_registry_entry('AI Regime Mth', 'ENABLE_AI_REGIME_MONTHLY', 'ai_regime_monthly_portfolio_value', 'ai_regime_monthly_portfolio_history', 'ai_regime_monthly_cash', 'ai_regime_monthly_positions'),
    'universal_model': _strategy_registry_entry('Universal Model', 'ENABLE_UNIVERSAL_MODEL', 'universal_model_portfolio_value', 'universal_model_portfolio_history', 'universal_model_cash', 'universal_model_positions'),
    'savgol_trend': _strategy_registry_entry('SavGol Trend', 'ENABLE_SAVGOL_TREND', 'savgol_trend_portfolio_value', 'savgol_trend_portfolio_history', 'savgol_trend_cash', 'savgol_trend_positions'),
    'top5_consistency_blend': _strategy_registry_entry('Top5 Cons Blend', 'ENABLE_TOP5_CONSISTENCY_BLEND', 'top5_consistency_blend_portfolio_value', 'top5_consistency_blend_portfolio_history', 'top5_consistency_blend_cash', 'top5_consistency_blend_positions'),
    'inverse_etf_hedge': _strategy_registry_entry('🛡️ Inv ETF Hedge', 'ENABLE_INVERSE_ETF_HEDGE', 'inverse_etf_hedge_portfolio_value', 'inverse_etf_hedge_portfolio_history', 'inverse_etf_hedge_cash', 'inverse_etf_hedge_positions'),
    'analyst_rec': _strategy_registry_entry('Analyst Rec', 'ENABLE_ANALYST_RECOMMENDATION', 'analyst_rec_portfolio_value', 'analyst_rec_portfolio_history', 'analyst_rec_cash', 'analyst_rec_positions'),
    'bh_1y_monthly': _strategy_registry_entry('BH 1Y Monthly', 'ENABLE_STATIC_BH_1Y_MONTHLY', 'static_bh_1y_monthly_portfolio_value', 'static_bh_1y_monthly_portfolio_history', 'static_bh_1y_monthly_cash', 'static_bh_1y_monthly_positions'),
    'bh_6m_monthly': _strategy_registry_entry('BH 6M Monthly', 'ENABLE_STATIC_BH_6M_MONTHLY', 'static_bh_6m_monthly_portfolio_value', 'static_bh_6m_monthly_portfolio_history', 'static_bh_6m_monthly_cash', 'static_bh_6m_monthly_positions'),
    'bh_3m_monthly': _strategy_registry_entry('BH 3M Monthly', 'ENABLE_STATIC_BH_3M_MONTHLY', 'static_bh_3m_monthly_portfolio_value', 'static_bh_3m_monthly_portfolio_history', 'static_bh_3m_monthly_cash', 'static_bh_3m_monthly_positions'),
    'bh_1m_monthly': _strategy_registry_entry('BH 1M Monthly', 'ENABLE_STATIC_BH_1M_MONTHLY', 'static_bh_1m_monthly_portfolio_value', 'static_bh_1m_monthly_portfolio_history', 'static_bh_1m_monthly_cash', 'static_bh_1m_monthly_positions'),
    'meta_weighted_composite': _strategy_registry_entry('Meta Weighted', 'ENABLE_META_WEIGHTED_COMPOSITE', 'meta_weighted_composite_value', 'meta_weighted_composite_history', 'meta_weighted_composite_cash', 'meta_weighted_composite_positions'),
    'meta_tiered_selection': _strategy_registry_entry('Meta Tiered', 'ENABLE_META_TIERED_SELECTION', 'meta_tiered_selection_value', 'meta_tiered_selection_history', 'meta_tiered_selection_cash', 'meta_tiered_selection_positions'),
    'meta_ensemble_alloc': _strategy_registry_entry('Meta Ensemble', 'ENABLE_META_ENSEMBLE_ALLOC', 'meta_ensemble_alloc_value', 'meta_ensemble_alloc_history', 'meta_ensemble_alloc_cash', 'meta_ensemble_alloc_positions'),
    'meta_regime_based': _strategy_registry_entry('Meta Regime', 'ENABLE_META_REGIME_BASED', 'meta_regime_based_value', 'meta_regime_based_history', 'meta_regime_based_cash', 'meta_regime_based_positions'),
    'meta_recency_weighted': _strategy_registry_entry('Meta Recency', 'ENABLE_META_RECENCY_WEIGHTED', 'meta_recency_weighted_value', 'meta_recency_weighted_history', 'meta_recency_weighted_cash', 'meta_recency_weighted_positions'),
    'meta_efficiency_ratio': _strategy_registry_entry('Meta Efficiency', 'ENABLE_META_EFFICIENCY_RATIO', 'meta_efficiency_ratio_value', 'meta_efficiency_ratio_history', 'meta_efficiency_ratio_cash', 'meta_efficiency_ratio_positions'),
    'meta_min_variance': _strategy_registry_entry('Meta MinVar', 'ENABLE_META_MIN_VARIANCE', 'meta_min_variance_value', 'meta_min_variance_history', 'meta_min_variance_cash', 'meta_min_variance_positions'),
    'meta_bayesian': _strategy_registry_entry('Meta Bayesian', 'ENABLE_META_BAYESIAN', 'meta_bayesian_value', 'meta_bayesian_history', 'meta_bayesian_cash', 'meta_bayesian_positions'),
    'meta_adaptive_convex': _strategy_registry_entry('Meta Adaptive', 'ENABLE_META_ADAPTIVE_CONVEX', 'meta_adaptive_convex_value', 'meta_adaptive_convex_history', 'meta_adaptive_convex_cash', 'meta_adaptive_convex_positions'),
    'meta_consensus': _strategy_registry_entry('Meta Consensus', 'ENABLE_META_CONSENSUS', 'meta_consensus_value', 'meta_consensus_history', 'meta_consensus_cash', 'meta_consensus_positions'),
    'bb_mean_reversion': _strategy_registry_entry('BB Mean Rev', 'ENABLE_BB_MEAN_REVERSION', 'bb_mean_reversion_value', 'bb_mean_reversion_history', 'bb_mean_reversion_cash', 'bb_mean_reversion_positions'),
    'bb_breakout': _strategy_registry_entry('BB Breakout', 'ENABLE_BB_BREAKOUT', 'bb_breakout_value', 'bb_breakout_history', 'bb_breakout_cash', 'bb_breakout_positions'),
    'bb_squeeze_breakout': _strategy_registry_entry('BB Squeeze', 'ENABLE_BB_SQUEEZE_BREAKOUT', 'bb_squeeze_breakout_value', 'bb_squeeze_breakout_history', 'bb_squeeze_breakout_cash', 'bb_squeeze_breakout_positions'),
    'bb_rsi_combo': _strategy_registry_entry('BB RSI Combo', 'ENABLE_BB_RSI_COMBO', 'bb_rsi_combo_value', 'bb_rsi_combo_history', 'bb_rsi_combo_cash', 'bb_rsi_combo_positions'),
    'trend_breakout': _strategy_registry_entry('Trend Breakout', 'ENABLE_TREND_BREAKOUT', 'trend_breakout_value', 'trend_breakout_history', 'trend_breakout_cash', 'trend_breakout_positions'),
    'static_bh_1y_vol': _strategy_registry_entry('BH 1Y Vol Trig', 'ENABLE_STATIC_BH_1Y_VOLATILITY', 'static_bh_1y_vol_portfolio_value', 'static_bh_1y_vol_portfolio_history', 'static_bh_1y_vol_cash', 'static_bh_1y_vol_positions'),
    'static_bh_1y_perf': _strategy_registry_entry('BH 1Y Perf Trig', 'ENABLE_STATIC_BH_1Y_PERFORMANCE', 'static_bh_1y_perf_portfolio_value', 'static_bh_1y_perf_portfolio_history', 'static_bh_1y_perf_cash', 'static_bh_1y_perf_positions'),
    'static_bh_1y_mom': _strategy_registry_entry('BH 1Y Mom Trig', 'ENABLE_STATIC_BH_1Y_MOMENTUM', 'static_bh_1y_mom_portfolio_value', 'static_bh_1y_mom_portfolio_history', 'static_bh_1y_mom_cash', 'static_bh_1y_mom_positions'),
    'static_bh_1y_atr': _strategy_registry_entry('BH 1Y ATR Trig', 'ENABLE_STATIC_BH_1Y_ATR', 'static_bh_1y_atr_portfolio_value', 'static_bh_1y_atr_portfolio_history', 'static_bh_1y_atr_cash', 'static_bh_1y_atr_positions'),
    'static_bh_1y_hybrid': _strategy_registry_entry('BH 1Y Hybrid Trig', 'ENABLE_STATIC_BH_1Y_HYBRID', 'static_bh_1y_hybrid_portfolio_value', 'static_bh_1y_hybrid_portfolio_history', 'static_bh_1y_hybrid_cash', 'static_bh_1y_hybrid_positions'),
    'static_bh_1y_volume': _strategy_registry_entry('BH 1Y Volume', 'ENABLE_STATIC_BH_1Y_VOLUME_FILTER', 'static_bh_1y_volume_portfolio_value', 'static_bh_1y_volume_portfolio_history', 'static_bh_1y_volume_cash', 'static_bh_1y_volume_positions'),
    'static_bh_1y_sector': _strategy_registry_entry('BH 1Y Sector', 'ENABLE_STATIC_BH_1Y_SECTOR_ROTATION', 'static_bh_1y_sector_portfolio_value', 'static_bh_1y_sector_portfolio_history', 'static_bh_1y_sector_cash', 'static_bh_1y_sector_positions'),
    'static_bh_1y_perf_threshold': _strategy_registry_entry('BH 1Y Perf Thresh', 'ENABLE_STATIC_BH_1Y_PERFORMANCE_THRESHOLD', 'static_bh_1y_perf_threshold_portfolio_value', 'static_bh_1y_perf_threshold_portfolio_history', 'static_bh_1y_perf_threshold_cash', 'static_bh_1y_perf_threshold_positions'),
    'static_bh_1y_market_regime': _strategy_registry_entry('BH 1Y Market Regime', 'ENABLE_STATIC_BH_1Y_MARKET_REGIME', 'static_bh_1y_market_regime_portfolio_value', 'static_bh_1y_market_regime_portfolio_history', 'static_bh_1y_market_regime_cash', 'static_bh_1y_market_regime_positions'),
    'static_bh_1y_mom_persist': _strategy_registry_entry('BH 1Y Mom Persist', 'ENABLE_STATIC_BH_1Y_MOMENTUM_PERSIST', 'static_bh_1y_mom_persist_portfolio_value', 'static_bh_1y_mom_persist_portfolio_history', 'static_bh_1y_mom_persist_cash', 'static_bh_1y_mom_persist_positions'),
    'static_bh_1y_overlap': _strategy_registry_entry('BH 1Y Overlap', 'ENABLE_STATIC_BH_1Y_OVERLAP', 'static_bh_1y_overlap_portfolio_value', 'static_bh_1y_overlap_portfolio_history', 'static_bh_1y_overlap_cash', 'static_bh_1y_overlap_positions'),
    'static_bh_1y_rank_drift': _strategy_registry_entry('BH 1Y Rank Drift', 'ENABLE_STATIC_BH_1Y_RANK_DRIFT', 'static_bh_1y_rank_drift_portfolio_value', 'static_bh_1y_rank_drift_portfolio_history', 'static_bh_1y_rank_drift_cash', 'static_bh_1y_rank_drift_positions'),
    'static_bh_1y_drawdown': _strategy_registry_entry('BH 1Y Drawdown', 'ENABLE_STATIC_BH_1Y_DRAWDOWN', 'static_bh_1y_drawdown_portfolio_value', 'static_bh_1y_drawdown_portfolio_history', 'static_bh_1y_drawdown_cash', 'static_bh_1y_drawdown_positions'),
    'static_bh_1y_smart_monthly': _strategy_registry_entry('BH 1Y Smart Mth', 'ENABLE_STATIC_BH_1Y_SMART_MONTHLY', 'static_bh_1y_smart_monthly_portfolio_value', 'static_bh_1y_smart_monthly_portfolio_history', 'static_bh_1y_smart_monthly_cash', 'static_bh_1y_smart_monthly_positions'),
    'bh_1y_mom_sell': _strategy_registry_entry('BH 1Y Mom Sell', 'ENABLE_BH_1Y_MOM_SELL', 'bh_1y_mom_sell_portfolio_value', 'bh_1y_mom_sell_portfolio_history', 'bh_1y_mom_sell_cash', 'bh_1y_mom_sell_positions'),
    'bh_1y_rank_sell': _strategy_registry_entry('BH 1Y Rank Sell', 'ENABLE_BH_1Y_RANK_SELL', 'bh_1y_rank_sell_portfolio_value', 'bh_1y_rank_sell_portfolio_history', 'bh_1y_rank_sell_cash', 'bh_1y_rank_sell_positions'),
    'bh_1y_trailing_mom': _strategy_registry_entry('BH 1Y Trail Mom', 'ENABLE_BH_1Y_TRAILING_MOM', 'bh_1y_trailing_mom_portfolio_value', 'bh_1y_trailing_mom_portfolio_history', 'bh_1y_trailing_mom_cash', 'bh_1y_trailing_mom_positions'),
    'bh_1y_volume_confirm': _strategy_registry_entry('BH 1Y Vol Conf', 'ENABLE_BH_1Y_VOLUME_CONFIRM', 'bh_1y_volume_confirm_portfolio_value', 'bh_1y_volume_confirm_portfolio_history', 'bh_1y_volume_confirm_cash', 'bh_1y_volume_confirm_positions'),
    'bh_1y_sector_aware': _strategy_registry_entry('BH 1Y Sect Aware', 'ENABLE_BH_1Y_SECTOR_AWARE', 'bh_1y_sector_aware_portfolio_value', 'bh_1y_sector_aware_portfolio_history', 'bh_1y_sector_aware_cash', 'bh_1y_sector_aware_positions'),
    'bh_1y_accel_buy': _strategy_registry_entry('BH 1Y Accel', 'ENABLE_BH_1Y_ACCEL_BUY', 'bh_1y_accel_buy_portfolio_value', 'bh_1y_accel_buy_portfolio_history', 'bh_1y_accel_buy_cash', 'bh_1y_accel_buy_positions'),
    'static_bh_3m_accel': _strategy_registry_entry('Static BH 3M Accel', 'ENABLE_STATIC_BH_3M_ACCEL', 'static_bh_3m_accel_portfolio_value', 'static_bh_3m_accel_portfolio_history', 'static_bh_3m_accel_cash', 'static_bh_3m_accel_positions'),
    'bh_1y_vol_adj_rebal': _strategy_registry_entry('Rebal 1Y VolAdj', 'ENABLE_BH_1Y_VOL_ADJ_REBAL', 'bh_1y_vol_adj_rebal_portfolio_value', 'bh_1y_vol_adj_rebal_portfolio_history', 'bh_1y_vol_adj_rebal_cash', 'bh_1y_vol_adj_rebal_positions'),
    'ratio_1m3m_vol_adj_rebal': _strategy_registry_entry('Rebal 1M3M VolAdj', 'ENABLE_RATIO_1M3M_VOL_ADJ_REBAL', 'ratio_1m3m_vol_adj_rebal_portfolio_value', 'ratio_1m3m_vol_adj_rebal_portfolio_history', 'ratio_1m3m_vol_adj_rebal_cash', 'ratio_1m3m_vol_adj_rebal_positions'),
    'bh_1y_corr_filter': _strategy_registry_entry('Rebal 1Y CorrFilt', 'ENABLE_BH_1Y_CORR_FILTER', 'bh_1y_corr_filter_portfolio_value', 'bh_1y_corr_filter_portfolio_history', 'bh_1y_corr_filter_cash', 'bh_1y_corr_filter_positions'),
    'bh_1y_regime_aware': _strategy_registry_entry('Rebal 1Y Regime', 'ENABLE_BH_1Y_REGIME_AWARE', 'bh_1y_regime_aware_portfolio_value', 'bh_1y_regime_aware_portfolio_history', 'bh_1y_regime_aware_cash', 'bh_1y_regime_aware_positions'),
    'bh_1y_risk_parity': _strategy_registry_entry('Rebal 1Y RiskPar', 'ENABLE_BH_1Y_RISK_PARITY', 'bh_1y_risk_parity_portfolio_value', 'bh_1y_risk_parity_portfolio_history', 'bh_1y_risk_parity_cash', 'bh_1y_risk_parity_positions'),
    'bh_1y_drift_thresh': _strategy_registry_entry('Rebal 1Y Drift', 'ENABLE_BH_1Y_DRIFT_THRESH', 'bh_1y_drift_thresh_portfolio_value', 'bh_1y_drift_thresh_portfolio_history', 'bh_1y_drift_thresh_cash', 'bh_1y_drift_thresh_positions'),
    'bh_1y_mom_quality': _strategy_registry_entry('Rebal 1Y MomQual', 'ENABLE_BH_1Y_MOM_QUALITY', 'bh_1y_mom_quality_portfolio_value', 'bh_1y_mom_quality_portfolio_history', 'bh_1y_mom_quality_cash', 'bh_1y_mom_quality_positions'),
    'bh_1y_liquidity': _strategy_registry_entry('Rebal 1Y Liquid', 'ENABLE_BH_1Y_LIQUIDITY', 'bh_1y_liquidity_portfolio_value', 'bh_1y_liquidity_portfolio_history', 'bh_1y_liquidity_cash', 'bh_1y_liquidity_positions'),
    'bh_1y_earnings_avoid': _strategy_registry_entry('Rebal 1Y EarnAvd', 'ENABLE_BH_1Y_EARNINGS_AVOID', 'bh_1y_earnings_avoid_portfolio_value', 'bh_1y_earnings_avoid_portfolio_history', 'bh_1y_earnings_avoid_cash', 'bh_1y_earnings_avoid_positions'),
    'bh_1y_multi_factor': _strategy_registry_entry('Rebal 1Y MultiFact', 'ENABLE_BH_1Y_MULTI_FACTOR', 'bh_1y_multi_factor_portfolio_value', 'bh_1y_multi_factor_portfolio_history', 'bh_1y_multi_factor_cash', 'bh_1y_multi_factor_positions'),
    'bh_1y_time_decay': _strategy_registry_entry('Rebal 1Y TimeDec', 'ENABLE_BH_1Y_TIME_DECAY', 'bh_1y_time_decay_portfolio_value', 'bh_1y_time_decay_portfolio_history', 'bh_1y_time_decay_cash', 'bh_1y_time_decay_positions'),
}

STRATEGY_DISPLAY_NAMES = {
    key: meta['display_name']
    for key, meta in STRATEGY_REGISTRY.items()
}


def _get_strategy_display_name(strategy_key: str) -> str:
    meta = STRATEGY_REGISTRY.get(strategy_key)
    return meta['display_name'] if meta else strategy_key


def _is_strategy_enabled(strategy_key: str, config_module=None) -> Optional[bool]:
    if config_module is None:
        import config as config_module

    meta = STRATEGY_REGISTRY.get(strategy_key)
    if meta is None:
        return None
    return bool(getattr(config_module, meta['enable_flag_name'], False))

# Global transaction cost tracking variables (initialized in main function)
static_bh_transaction_costs = 0
static_bh_3m_transaction_costs = 0
static_bh_6m_transaction_costs = 0
static_bh_1m_transaction_costs = 0
dynamic_bh_transaction_costs = 0
dynamic_bh_1y_transaction_costs = 0
dynamic_bh_6m_transaction_costs = 0
dynamic_bh_3m_transaction_costs = 0
dynamic_bh_1m_transaction_costs = 0
risk_adj_mom_transaction_costs = 0
mean_reversion_transaction_costs = 0
quality_momentum_transaction_costs = 0
momentum_ai_hybrid_transaction_costs = 0
volatility_adj_mom_transaction_costs = 0
sector_rotation_transaction_costs = 0
ratio_3m_1y_transaction_costs = 0
ratio_1y_3m_transaction_costs = 0
turnaround_transaction_costs = 0
adaptive_strategy_transaction_costs = 0
adaptive_strategy_portfolio_value = 0
volatility_ensemble_transaction_costs = 0
enhanced_volatility_transaction_costs = 0
correlation_ensemble_transaction_costs = 0
dynamic_pool_transaction_costs = 0
sentiment_ensemble_transaction_costs = 0
voting_ensemble_transaction_costs = 0
momentum_volatility_hybrid_transaction_costs = 0
momentum_volatility_hybrid_6m_transaction_costs = 0
price_acceleration_transaction_costs = 0
dynamic_bh_1y_vol_filter_transaction_costs = 0
dynamic_bh_1y_trailing_stop_transaction_costs = 0
adaptive_ensemble_transaction_costs = 0
ai_volatility_ensemble_transaction_costs = 0
multi_tf_ensemble_transaction_costs = 0
multi_tf_intraday_ensemble_transaction_costs = 0
top5_consistency_blend_transaction_costs = 0


def _build_daily_strategy_data(locals_dict):
    """
    Build strategy data for daily summary from local variables.
    Returns dict mapping strategy_key -> {'value': float, 'history': list, 'cash': float, 'positions': int}
    Uses `STRATEGY_REGISTRY` as the single source of truth for strategy metadata.
    """
    import config

    def _get(name, default=None):
        return locals_dict.get(name, default)

    data = {}
    for key, meta in STRATEGY_REGISTRY.items():
        if not getattr(config, meta['enable_flag_name'], False):
            continue

        value = _get(meta['value_var'])
        if value is not None and value > 0:
            history = _get(meta['history_var'], [])
            cash = _get(meta['cash_var'], 0)
            positions = _get(meta['positions_var'])

            if isinstance(positions, dict):
                num_positions = len(positions)
                tickers = list(positions.keys())
            elif isinstance(positions, (list, tuple)):
                num_positions = len(positions)
                tickers = list(positions)
            elif positions is None:
                num_positions = 0
                tickers = []
            else:
                num_positions = 0
                tickers = []

            data[key] = {
                'value': value,
                'history': history,
                'cash': cash,
                'positions': num_positions,
                'tickers': tickers,
                'display_name': meta['display_name'],
            }

    return data


def _collect_data_worker(args):
    """Top-level worker for ProcessPoolExecutor - collects training data for one ticker."""
    ticker, ticker_data, train_start, train_end, forward_days, market_returns = args
    from ai_elite_strategy_per_ticker import collect_ticker_training_data
    samples = collect_ticker_training_data(
        ticker=ticker, ticker_data=ticker_data,
        train_start_date=train_start, train_end_date=train_end,
        forward_days=forward_days, market_returns=market_returns
    )
    return ticker, samples


def _fine_tune_worker(args):
    """Top-level worker for ProcessPoolExecutor - fine-tunes base model for one ticker."""
    ticker, ticker_samples, base_model, save_path = args
    from ai_elite_strategy_per_ticker import fine_tune_per_ticker
    model = fine_tune_per_ticker(
        ticker=ticker, ticker_samples=ticker_samples,
        base_model=base_model, save_path=save_path
    )
    return ticker, model


def _select_top5_consistency_blend_stocks(
    source_strategy_data: Dict[str, Dict],
    top5_consistency_counts: Dict[str, int],
    total_days: int,
    top_n: int = 10,
    max_source_strategies: int = 5,
) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Blend current source strategy holdings using lagged top-5 consistency weights.

    The weights are based on completed backtest days only, so Day N+1 uses the
    consistency results accumulated through Day N.
    """
    if total_days <= 0 or not top5_consistency_counts:
        return [], {}, []

    eligible_sources = []
    for strategy_key, strategy_data in source_strategy_data.items():
        if strategy_key == 'top5_consistency_blend':
            continue

        display_name = strategy_data.get('display_name', strategy_key)
        tickers = list(strategy_data.get('tickers') or [])
        consistency_count = top5_consistency_counts.get(display_name, 0)
        if consistency_count > 0 and tickers:
            eligible_sources.append((display_name, consistency_count, tickers[:top_n]))

    if not eligible_sources:
        return [], {}, []

    eligible_sources.sort(key=lambda item: (-item[1], item[0]))
    selected_sources = eligible_sources[:max_source_strategies]

    raw_weights = {
        display_name: consistency_count / total_days
        for display_name, consistency_count, _ in selected_sources
    }
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        return [], {}, []

    normalized_weights = {
        display_name: weight / total_weight
        for display_name, weight in raw_weights.items()
    }

    ticker_scores: Dict[str, float] = {}
    for display_name, _, tickers in selected_sources:
        num_tickers = max(1, len(tickers))
        strategy_weight = normalized_weights[display_name]
        for rank, ticker in enumerate(tickers):
            rank_multiplier = (num_tickers - rank) / num_tickers
            ticker_scores[ticker] = ticker_scores.get(ticker, 0.0) + strategy_weight * rank_multiplier

    ranked_tickers = sorted(ticker_scores.items(), key=lambda item: (-item[1], item[0]))
    selected_tickers = [ticker for ticker, _ in ranked_tickers[:top_n]]
    source_names = [display_name for display_name, _, _ in selected_sources]
    return selected_tickers, normalized_weights, source_names


def _select_voting_ensemble_stocks_from_strategy_data(
    source_strategy_data: Dict[str, Dict],
    top_n: int = 10,
    max_source_strategies: int = 3,
    excluded_strategy_keys: Optional[List[str]] = None,
) -> Tuple[List[str], List[Tuple[str, str, float, List[str]]], List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Build voting picks from the best-performing in-memory strategies for the day.

    This uses each strategy's current day-end portfolio value as the ranking metric
    and votes on the tickers that strategy is actually holding/selected in memory.
    """
    excluded = set(excluded_strategy_keys or [])
    eligible_sources: List[Tuple[str, str, float, List[str]]] = []

    for strategy_key, strategy_data in source_strategy_data.items():
        if strategy_key in excluded:
            continue

        display_name = strategy_data.get('display_name', strategy_key)
        value = float(strategy_data.get('value') or 0.0)
        raw_tickers = list(strategy_data.get('tickers') or [])
        deduped_tickers: List[str] = []
        seen = set()
        for ticker in raw_tickers:
            if ticker and ticker not in seen:
                deduped_tickers.append(ticker)
                seen.add(ticker)

        if deduped_tickers:
            eligible_sources.append((strategy_key, display_name, value, deduped_tickers[:top_n]))

    if not eligible_sources:
        return [], [], [], {}

    eligible_sources.sort(key=lambda item: (-item[2], item[1]))
    selected_sources = eligible_sources[:max_source_strategies]

    vote_counts: Dict[str, int] = {}
    vote_details: Dict[str, List[str]] = {}
    tie_break_scores: Dict[str, float] = {}

    for source_rank, (_, display_name, _, tickers) in enumerate(selected_sources, 1):
        source_bonus = max_source_strategies - source_rank + 1
        num_tickers = max(1, len(tickers))
        for ticker_rank, ticker in enumerate(tickers):
            vote_counts[ticker] = vote_counts.get(ticker, 0) + 1
            vote_details.setdefault(ticker, []).append(display_name)
            rank_multiplier = (num_tickers - ticker_rank) / num_tickers
            tie_break_scores[ticker] = tie_break_scores.get(ticker, 0.0) + (source_bonus * rank_multiplier)

    ranked_tickers = sorted(
        vote_counts.items(),
        key=lambda item: (-item[1], -tie_break_scores.get(item[0], 0.0), item[0])
    )
    selected_tickers = [ticker for ticker, _ in ranked_tickers[:top_n]]
    return selected_tickers, selected_sources, ranked_tickers, vote_details


def _last_valid_close_up_to(ticker_df: pd.DataFrame, current_date: datetime) -> Optional[float]:
    try:
        s = ticker_df.loc[:current_date, "Close"].dropna()
        if len(s) == 0:
            return None
        v = float(s.iloc[-1])
        return None if pd.isna(v) else v
    except Exception:
        return None


def _return_over_lookback(
    ticker_df: pd.DataFrame,
    current_date: datetime,
    lookback_days: int
) -> Optional[float]:
    """
    Return over lookback as a fraction (e.g. 0.10 = +10%).
    Uses first valid close at/after lookback start, and last valid close up to current_date.
    """
    try:
        start_date = current_date - timedelta(days=int(lookback_days))
        window = ticker_df.loc[start_date:current_date, "Close"].dropna()
        if len(window) < 2:
            return None
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
            return None
        return (end_price / start_price) - 1.0
    except Exception:
        return None


def _mark_to_market_value(
    positions: Dict[str, Dict],
    cash: float,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime
) -> float:
    total = float(cash or 0.0)
    for t, pos in (positions or {}).items():
        try:
            if t not in ticker_data_grouped:
                continue
            px = _last_valid_close_up_to(ticker_data_grouped[t], current_date)
            if px is None:
                continue
            shares = float(pos.get("shares", 0.0) or 0.0)
            total += shares * px
        except Exception:
            continue
    return float(total)


def _should_rebalance_by_profit_since_last_rebalance(
    current_stocks: List[str],
    new_stocks: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    positions: Dict[str, Dict],
    cash: float,
    transaction_cost: float,
    last_rebalance_value: float
) -> Tuple[bool, str]:
    """
    Transaction cost guard: Only rebalance if the portfolio has grown enough
    since the last rebalance to cover the transaction costs.

    Logic:
    1. Save portfolio value when last rebalanced (last_rebalance_value)
    2. Calculate current portfolio value (mark-to-market)
    3. Calculate transaction costs for the stocks being changed
    4. Rebalance if: (current_value - transaction_costs) > last_rebalance_value

    This ensures we only trade if we are "up enough" since the last rebalance
    to pay for the next rebalance.
    """
    if not new_stocks:
        return False, "no new selection"
    if not current_stocks:
        # For initial allocation, always allow rebalancing regardless of transaction costs
        return True, "initial allocation"

    current_set = set(current_stocks)
    new_set = set(new_stocks)

    stocks_to_sell = current_set - new_set  # Stocks to exit
    stocks_to_buy = new_set - current_set   # Stocks to enter

    if len(stocks_to_sell) == 0 and len(stocks_to_buy) == 0:
        return False, "no change"

    # Calculate current portfolio value
    portfolio_value_now = _mark_to_market_value(positions, cash, ticker_data_grouped, current_date)

    # Calculate actual transaction costs for the specific trades
    total_sell_value = 0.0
    for ticker in stocks_to_sell:
        if ticker in positions:
            pos = positions[ticker]
            shares = pos.get('shares', 0)
            if ticker in ticker_data_grouped:
                price_data = ticker_data_grouped[ticker].loc[:current_date]
                if not price_data.empty:
                    current_price = price_data['Close'].dropna().iloc[-1]
                    total_sell_value += shares * current_price

    total_buy_value = 0.0
    # Estimate buy value based on equal allocation of freed capital
    if stocks_to_buy:
        # Capital available = cash + sell proceeds (after sell costs)
        sell_proceeds_after_cost = total_sell_value * (1 - transaction_cost)
        available_capital = cash + sell_proceeds_after_cost
        # Each new stock gets equal share
        per_stock_allocation = available_capital / len(new_stocks) if new_stocks else 0
        total_buy_value = per_stock_allocation * len(stocks_to_buy)

    # Transaction costs: sell cost + buy cost
    sell_cost = total_sell_value * transaction_cost
    buy_cost = total_buy_value * transaction_cost
    total_transaction_cost = sell_cost + buy_cost

    # Check if portfolio value after costs exceeds last rebalance value
    value_after_costs = portfolio_value_now - total_transaction_cost
    delta_vs_last = value_after_costs - float(last_rebalance_value or 0.0)

    if delta_vs_last > 0:
        return True, (
            f"value_after_cost>last (now ${portfolio_value_now:,.0f} - cost ${total_transaction_cost:,.0f} = "
            f"${value_after_costs:,.0f} > last ${last_rebalance_value:,.0f}; Δ=${delta_vs_last:,.0f})"
        )
    return False, (
        f"value_after_cost<=last (now ${portfolio_value_now:,.0f} - cost ${total_transaction_cost:,.0f} = "
        f"${value_after_costs:,.0f} <= last ${last_rebalance_value:,.0f}; Δ=${delta_vs_last:,.0f})"
    )


def _smart_rebalance_portfolio(
    strategy_name: str,
    current_stocks: List[str],
    new_stocks: List[str],
    positions: Dict[str, Dict],
    cash: float,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    transaction_cost: float,
    portfolio_size: int = 10,
    force_rebalance: bool = False,
    buffer_size: int = None,
    strategy_stop_loss: float = None,
    use_atr_trailing_stop: bool = False,
    atr_multiplier: float = 2.0,
    custom_weights: Dict[str, float] = None
) -> Tuple[Dict[str, Dict], float, List[str], float, bool]:
    """
    Universal smart rebalancing function for all strategies.

    Implements selective rebalancing with individual stock profit guards.

    Returns:
        Tuple of (updated_positions, updated_cash, final_stocks, total_transaction_costs, rebalanced)
        where rebalanced is True if positions actually changed
    """
    if not new_stocks:
        return positions, cash, current_stocks, 0.0, False

    # Use buffer size if provided, otherwise use portfolio_size + buffer (keep stocks in top 12 for 10-stock portfolio with buffer=2)
    effective_buffer_size = buffer_size if buffer_size is not None else (portfolio_size + PORTFOLIO_BUFFER_SIZE)
    debug_rebalance_logs = getattr(config, 'DEBUG_REBALANCE_LOGS', False)

    current_positions_list = []
    seen_positions = set()
    for ticker in current_stocks:
        if ticker in positions and ticker not in seen_positions:
            current_positions_list.append(ticker)
            seen_positions.add(ticker)
    for ticker in positions:
        if ticker not in seen_positions:
            current_positions_list.append(ticker)
            seen_positions.add(ticker)

    current_positions_set = set(current_positions_list)
    buffer_list = new_stocks[:effective_buffer_size]
    buffer_set = set(buffer_list)
    target_list = new_stocks[:portfolio_size]

    if debug_rebalance_logs:
        date_label = current_date.strftime('%Y-%m-%d') if hasattr(current_date, 'strftime') else str(current_date)
        print(f"   🔎 {strategy_name} {date_label} current portfolio ({len(current_positions_list)}): {current_positions_list}")
        print(f"   🎯 {strategy_name} {date_label} top {len(target_list)} target candidates: {target_list}")
        print(f"   🧺 {strategy_name} {date_label} top {len(buffer_list)} buffer candidates: {buffer_list}")

    # Keep existing holdings only while they stay inside the top-N buffer.
    positions_to_keep_list = [ticker for ticker in buffer_list if ticker in current_positions_set][:portfolio_size]
    positions_to_keep_set = set(positions_to_keep_list)

    # ATR exits apply to kept positions before replacement buys are chosen.
    atr_positions_to_sell = set()
    if use_atr_trailing_stop and positions_to_keep_list:
        for ticker in positions_to_keep_list:
            if ticker in ticker_data_grouped and ticker in positions:
                try:
                    price_data = ticker_data_grouped[ticker].loc[:current_date]
                    if len(price_data) < 14:
                        continue

                    high_low = price_data['High'] - price_data['Low']
                    high_close = abs(price_data['High'] - price_data['Close'].shift())
                    low_close = abs(price_data['Low'] - price_data['Close'].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(14).mean().iloc[-1]

                    if pd.isna(atr) or atr <= 0:
                        continue

                    price_history = price_data['Close']
                    recent_prices = price_history.tail(60)
                    highest_since_entry = recent_prices.max()
                    current_price = price_history.iloc[-1]
                    trailing_stop = highest_since_entry - (atr * atr_multiplier)

                    if current_price < trailing_stop:
                        atr_positions_to_sell.add(ticker)
                        print(f"   🛑 {strategy_name} ATR trailing stop triggered for {ticker}: ${current_price:.2f} < ${trailing_stop:.2f} (ATR: ${atr:.2f})")

                except Exception as e:
                    print(f"   ⚠️ Error checking ATR stop for {ticker}: {e}")
                    continue

    if atr_positions_to_sell:
        positions_to_keep_list = [ticker for ticker in positions_to_keep_list if ticker not in atr_positions_to_sell]
        positions_to_keep_set = set(positions_to_keep_list)

    positions_to_sell_list = [ticker for ticker in current_positions_list if ticker not in positions_to_keep_set]
    planned_buy_count = max(0, portfolio_size - len(positions_to_keep_list))
    print(
        f"   📊 {strategy_name} Rebalance summary (buffer={effective_buffer_size}): "
        f"{len(positions_to_keep_list)} keep, {len(positions_to_sell_list)} sell, {planned_buy_count} replacement slot(s)"
    )

    sell_reasons = {}
    for ticker in positions_to_sell_list:
        if ticker in atr_positions_to_sell:
            sell_reasons[ticker] = "ATR trailing stop"
        elif ticker in buffer_set:
            sell_reasons[ticker] = f"Exceeds max {portfolio_size} holdings"
        else:
            sell_reasons[ticker] = f"Outside top {effective_buffer_size} buffer"

    if debug_rebalance_logs:
        print(f"   ✅ {strategy_name} Holdings kept in buffer ({len(positions_to_keep_list)}): {positions_to_keep_list}")
        if positions_to_sell_list:
            print(f"   🧾 {strategy_name} Planned sells ({len(positions_to_sell_list)}): {[f'{ticker} [{sell_reasons[ticker]}]' for ticker in positions_to_sell_list]}")
        else:
            print(f"   🧾 {strategy_name} Planned sells (0): []")

    total_transaction_costs = 0.0
    updated_positions = positions.copy()
    updated_cash = cash
    positions_changed = False
    failed_sells = []

    for ticker in positions_to_sell_list:
        if ticker not in updated_positions:
            continue

        if ticker not in ticker_data_grouped:
            print(f"   ⚠️ {strategy_name} Cannot sell {ticker}: no price data, keeping position")
            failed_sells.append(ticker)
            continue

        price_data = ticker_data_grouped[ticker].loc[:current_date]
        if price_data.empty or price_data['Close'].dropna().empty:
            print(f"   ⚠️ {strategy_name} Cannot sell {ticker}: no valid close price, keeping position")
            failed_sells.append(ticker)
            continue

        current_price = price_data['Close'].dropna().iloc[-1]
        shares = updated_positions[ticker]['shares']
        gross_sale = shares * current_price
        sell_cost = gross_sale * transaction_cost

        print(f"   💰 {strategy_name} Selling {ticker}: {sell_reasons[ticker]}")
        total_transaction_costs += sell_cost
        updated_cash += gross_sale - sell_cost
        del updated_positions[ticker]
        positions_changed = True

    held_after_sells = [ticker for ticker in positions_to_keep_list if ticker in updated_positions]
    held_after_sells.extend(
        ticker for ticker in failed_sells
        if ticker in updated_positions and ticker not in held_after_sells
    )

    current_position_count = len(updated_positions)
    available_slots = max(0, portfolio_size - current_position_count)
    positions_to_actually_buy = [ticker for ticker in new_stocks if ticker not in updated_positions][:available_slots]

    if debug_rebalance_logs:
        print(f"   🔄 {strategy_name} Replacement candidates ({len(positions_to_actually_buy)}): {positions_to_actually_buy}")

    if current_position_count >= portfolio_size:
        print(f"   📊 {strategy_name} Already at max {portfolio_size} positions, skipping new buys")

    if positions_to_actually_buy:
        held_positions_value = sum(
            updated_positions[ticker]['shares'] * ticker_data_grouped[ticker].loc[:current_date]['Close'].dropna().iloc[-1]
            for ticker in updated_positions
            if ticker in ticker_data_grouped
            and not ticker_data_grouped[ticker].loc[:current_date].empty
            and not ticker_data_grouped[ticker].loc[:current_date]['Close'].dropna().empty
        )
        total_portfolio_value = updated_cash + held_positions_value
        capital_per_stock = total_portfolio_value / portfolio_size if portfolio_size > 0 else 0
    else:
        total_portfolio_value = updated_cash
        capital_per_stock = 0

    final_stocks = held_after_sells.copy()
    newly_bought = []
    for ticker in positions_to_actually_buy:
        if len(updated_positions) >= portfolio_size:
            break
        if ticker in ticker_data_grouped:
            price_data = ticker_data_grouped[ticker].loc[:current_date]
            if not price_data.empty and not price_data['Close'].dropna().empty:
                current_price = price_data['Close'].dropna().iloc[-1]
                if current_price > 0:
                    if custom_weights and ticker in custom_weights:
                        ticker_capital = total_portfolio_value * custom_weights[ticker]
                        available_for_stock = min(ticker_capital, updated_cash)
                    else:
                        available_for_stock = min(capital_per_stock, updated_cash)
                    shares = int(available_for_stock / (current_price * (1 + transaction_cost)))
                    if shares > 0:
                        buy_value = shares * current_price
                        buy_cost = buy_value * transaction_cost
                        total_cost = buy_value + buy_cost

                        if total_cost <= updated_cash:
                            print(f"   🛒 {strategy_name} Buying {ticker}: {shares} shares @ ${current_price:.2f}")
                            total_transaction_costs += buy_cost
                            updated_cash -= total_cost
                            updated_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}
                            final_stocks.append(ticker)
                            newly_bought.append(ticker)
                            positions_changed = True
                        else:
                            print(f"   ❌ {strategy_name} Insufficient cash for {ticker}: need ${total_cost:,.0f}, have ${updated_cash:,.0f}")

    skipped_tickers = [ticker for ticker in positions_to_actually_buy if ticker not in newly_bought]
    for ticker in skipped_tickers:
        if updated_cash <= 0 or len(updated_positions) >= portfolio_size:
            break
        if ticker in ticker_data_grouped:
            price_data = ticker_data_grouped[ticker].loc[:current_date]
            if not price_data.empty and not price_data['Close'].dropna().empty:
                current_price = price_data['Close'].dropna().iloc[-1]
                if current_price > 0:
                    shares = int(updated_cash / (current_price * (1 + transaction_cost)))
                    if shares > 0:
                        buy_value = shares * current_price
                        buy_cost = buy_value * transaction_cost
                        total_cost = buy_value + buy_cost
                        if total_cost <= updated_cash:
                            print(f"   {strategy_name} Buying {ticker} (2nd pass): {shares} shares @ ${current_price:.2f}")
                            total_transaction_costs += buy_cost
                            updated_cash -= total_cost
                            updated_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}
                            final_stocks.append(ticker)
                            newly_bought.append(ticker)
                            positions_changed = True

    held_set = set(updated_positions)
    ordered_existing = [ticker for ticker in held_after_sells if ticker in held_set]
    ordered_new = [ticker for ticker in final_stocks if ticker in held_set and ticker not in ordered_existing]
    ordered_remaining = [ticker for ticker in updated_positions if ticker not in ordered_existing and ticker not in ordered_new]
    final_stocks = ordered_existing + ordered_new + ordered_remaining

    if len(final_stocks) > portfolio_size:
        print(f"   ⚠️ {strategy_name} Still holding {len(final_stocks)} positions after rebalance; check missing sell prices")

    if debug_rebalance_logs:
        print(f"   📦 {strategy_name} Final portfolio ({len(final_stocks)}): {final_stocks}")
        print(f"   💵 {strategy_name} End cash: ${updated_cash:,.2f}, transaction costs: ${total_transaction_costs:,.2f}")

    return updated_positions, updated_cash, final_stocks, total_transaction_costs, positions_changed


def _run_ai_elite_step(
    initial_top_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    day_count: int,
    ai_elite_models: Dict,
    ai_elite_last_train_days: Dict[str, int],
    ai_elite_positions: Dict[str, Dict],
    ai_elite_cash: float,
    current_ai_elite_stocks: List[str],
    ai_elite_transaction_costs: float,
    ai_elite_portfolio_value: float,
    ai_elite_last_rebalance_value: float,
) -> Tuple[Dict, Dict[str, int], Dict[str, Dict], float, List[str], float, float, bool]:
    """Run one AI Elite backtest step and return updated state."""
    from shared_strategies import select_ai_elite_with_training

    should_train_ai_elite = False
    if ai_elite_models.get('_shared_base') is not None:
        last_train_day = max(ai_elite_last_train_days.values()) if ai_elite_last_train_days else 0
        days_since_train = day_count - last_train_day
        if days_since_train >= AI_ELITE_RETRAIN_DAYS:
            should_train_ai_elite = True
            print(
                f"   📊 AI Elite: Retraining triggered (day {day_count}, "
                f"last train day {last_train_day}, interval {AI_ELITE_RETRAIN_DAYS})"
            )

    if not should_train_ai_elite and ai_elite_models.get('_shared_base') is not None:
        last_train_day = max(ai_elite_last_train_days.values()) if ai_elite_last_train_days else 0
        days_ago = day_count - last_train_day if last_train_day > 0 else 0
        print(f"   📊 AI Elite: Using existing model (trained {days_ago} days ago)")

    new_ai_elite_stocks, ai_elite_models = select_ai_elite_with_training(
        all_tickers=initial_top_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
        ai_elite_models=ai_elite_models,
        force_train=should_train_ai_elite,
    )

    if should_train_ai_elite and ai_elite_models.get('_shared_base') is not None:
        for ticker in initial_top_tickers:
            ai_elite_last_train_days[ticker] = day_count

    rebalanced_flag = False
    if new_ai_elite_stocks:
        print(f"   📊 AI Elite Day {day_count}: {new_ai_elite_stocks}")
        (
            ai_elite_positions,
            ai_elite_cash,
            current_ai_elite_stocks,
            rebalance_costs,
            rebalanced_flag,
        ) = _smart_rebalance_portfolio(
            strategy_name="AI Elite",
            current_stocks=current_ai_elite_stocks,
            new_stocks=new_ai_elite_stocks,
            positions=ai_elite_positions,
            cash=ai_elite_cash,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            transaction_cost=TRANSACTION_COST,
            portfolio_size=PORTFOLIO_SIZE,
            buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
            force_rebalance=not current_ai_elite_stocks,
        )
        ai_elite_transaction_costs += rebalance_costs
        ai_elite_last_rebalance_value = ai_elite_portfolio_value

    return (
        ai_elite_models,
        ai_elite_last_train_days,
        ai_elite_positions,
        ai_elite_cash,
        current_ai_elite_stocks,
        ai_elite_transaction_costs,
        ai_elite_last_rebalance_value,
        rebalanced_flag,
    )


from data_validation import validate_prediction_data, validate_features_after_engineering, InsufficientDataError
import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path("logs").mkdir(exist_ok=True)


def _prepare_model_for_multiprocessing(model):
    """Prepare PyTorch models for multiprocessing by converting to numpy arrays.

    Returns a dict with model info that can be pickled (numpy arrays only),
    and the model will be reconstructed on GPU in the worker process.
    """
    if model is None:
        return None
    if PYTORCH_AVAILABLE:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor
            if isinstance(model, (LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor)):
                # Force model to CPU first to ensure all tensors are on CPU
                model_cpu = model.cpu()

                # Extract state dict and convert tensors to numpy arrays for safe pickling
                state_dict = model_cpu.state_dict()
                numpy_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Convert to numpy array (detach to break computation graph, cpu to ensure no CUDA context)
                        numpy_state_dict[key] = value.detach().cpu().numpy().copy()
                    else:
                        numpy_state_dict[key] = value

                # Extract architecture info from model and state dict
                model_type_name = type(model_cpu).__name__

                # Handle TCNRegressor differently from RNN models
                if model_type_name == 'TCNRegressor':
                    # TCN uses Conv1d layers, get input_size from first conv layer
                    input_size = None
                    num_filters = 32
                    kernel_size = 3
                    num_levels = 2
                    dropout = 0.1

                    # Get input_size from first conv layer weight
                    for key in numpy_state_dict.keys():
                        if 'net.0.weight' in key:  # First Conv1d layer
                            input_size = numpy_state_dict[key].shape[1]  # in_channels
                            num_filters = numpy_state_dict[key].shape[0]  # out_channels
                            break

                    if input_size is None:
                        # Fallback: try to get from model's first conv layer
                        if hasattr(model_cpu, 'net') and len(model_cpu.net) > 0:
                            first_conv = model_cpu.net[0]
                            if hasattr(first_conv, 'in_channels'):
                                input_size = first_conv.in_channels

                    if input_size is None:
                        sys.stderr.write("  ⚠️ ERROR in _prepare_model_for_multiprocessing: missing TCN input_size\n")
                        sys.stderr.flush()
                        return None

                    model_info = {
                        'type': model_type_name,
                        'state_dict': numpy_state_dict,
                        'input_size': input_size,
                        'num_filters': num_filters,
                        'kernel_size': kernel_size,
                        'num_levels': num_levels,
                        'dropout': dropout,
                    }
                    sys.stderr.write(f"  [PREP] {model_info['type']}: in={input_size}, filters={num_filters}, levels={num_levels}\n")
                    sys.stderr.flush()
                    return model_info

                # For RNN models (GRU, LSTM)
                hidden_size = model_cpu.hidden_size if hasattr(model_cpu, 'hidden_size') else None
                num_layers = model_cpu.num_layers if hasattr(model_cpu, 'num_layers') else None

                # Infer input_size and output_size from state dict
                input_size = None
                output_size = None
                dropout = 0.0

                for key in numpy_state_dict.keys():
                    if 'weight_ih_l0' in key:  # First layer input weights
                        # Shape is [hidden_size*3 (GRU) or hidden_size*4 (LSTM), input_size]
                        input_size = numpy_state_dict[key].shape[1]
                    elif 'fc.weight' in key:
                        # For simple fc layer: 'fc.weight'
                        output_size = numpy_state_dict[key].shape[0]
                    elif 'fc.2.weight' in key:
                        # For nn.Sequential fc layer (GRURegressor): 'fc.2.weight' is the final layer
                        output_size = numpy_state_dict[key].shape[0]

                # Fallback: try to get output_size from model attributes or final Linear in fc
                if output_size is None and hasattr(model_cpu, 'output_size'):
                    output_size = model_cpu.output_size
                if output_size is None and hasattr(model_cpu, 'fc'):
                    try:
                        import torch.nn as nn
                        for module in reversed(list(model_cpu.fc.modules())):
                            if isinstance(module, nn.Linear):
                                output_size = module.out_features
                                break
                    except Exception:
                        pass

                # Get dropout from model's RNN layer
                if hasattr(model_cpu, 'gru'):
                    dropout = model_cpu.gru.dropout if hasattr(model_cpu.gru, 'dropout') else 0.0
                elif hasattr(model_cpu, 'lstm'):
                    dropout = model_cpu.lstm.dropout if hasattr(model_cpu.lstm, 'dropout') else 0.0

                # Defensive guard: ensure we have all dims
                if any(v is None for v in [input_size, hidden_size, num_layers, output_size]):
                    sys.stderr.write("  ⚠️ ERROR in _prepare_model_for_multiprocessing: missing model dimensions\n")
                    sys.stderr.write(f"     input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, output_size={output_size}\n")
                    sys.stderr.flush()
                    return None

                # Clear CUDA cache to ensure all references are gone
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model_info = {
                    'type': type(model_cpu).__name__,
                    'state_dict': numpy_state_dict,  # Now contains numpy arrays, not tensors
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout': dropout,
                }
                # DEBUG: Log model architecture for verification
                sys.stderr.write(f"  [PREP] {model_info['type']}: in={input_size}, hid={hidden_size}, layers={num_layers}, out={output_size}, dropout={dropout}\n")
                sys.stderr.flush()
                return model_info
            else:
                # For non-PyTorch models (LightGBM, XGBoost, etc.), return as-is
                return model
        except (ImportError, AttributeError, Exception) as e:
            sys.stderr.write(f"  ⚠️ ERROR in _prepare_model_for_multiprocessing: {e}\n")
            sys.stderr.write(f"     Model type: {type(model).__name__ if model else 'None'}\n")
            sys.stderr.flush()
            return None
    return model  # For non-PyTorch models, return as-is


def _reconstruct_model_from_info(model_info, device='cpu'):
    """Reconstruct a PyTorch model from model_info (with numpy arrays) on the specified device."""
    if model_info is None:
        return None
    if isinstance(model_info, dict) and 'type' in model_info:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor

            # Resolve device
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            torch_device = torch.device(device)

            model_type = model_info['type']
            numpy_state_dict = model_info['state_dict']

            # Convert numpy arrays back to tensors on CPU first
            state_dict = {}
            for key, value in numpy_state_dict.items():
                if isinstance(value, np.ndarray):
                    # Convert to numpy array (detach to break computation graph, cpu to ensure no CUDA context)
                    state_dict[key] = torch.from_numpy(value)  # keep on CPU for load_state_dict
                else:
                    state_dict[key] = value

            # Reconstruct model based on type (on CPU)
            if model_type == 'TCNRegressor':
                model = TCNRegressor(
                    model_info['input_size'],
                    num_filters=model_info.get('num_filters', 32),
                    kernel_size=model_info.get('kernel_size', 3),
                    num_levels=model_info.get('num_levels', 2),
                    dropout=model_info.get('dropout', 0.1)
                )
            elif model_type == 'GRUClassifier':
                model = GRUClassifier(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            elif model_type == 'GRURegressor':
                model = GRURegressor(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            elif model_type == 'LSTMClassifier':
                model = LSTMClassifier(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            else:
                return None

            # Load state dict on CPU, then move to target device
            model.load_state_dict(state_dict)
            model = model.to(torch_device)
            model.eval()
            return model
        except (ImportError, AttributeError, Exception) as e:
            import sys
            sys.stderr.write(f"  ⚠️ ERROR in _reconstruct_model_from_info: {e}\n")
            sys.stderr.write(f"     Model type: {model_info.get('type', 'unknown')}\n")
            sys.stderr.flush()
            return None
    return model_info  # For non-PyTorch models, return as-is


# -----------------------------------------------------------------------------
# Single-ticker optimisation worker
# -----------------------------------------------------------------------------
def optimize_single_ticker_worker(params):
    (
        ticker, train_data, capital_per_stock, class_horizon,
        force_thresholds_optimization, force_percentage_optimization,
        use_alpha_threshold_buy, use_alpha_threshold_sell,
        alpha_config, current_min_proba_buy, current_min_proba_sell,
        initial_class_horizon,
        target_percentage_options, class_horizon_options,
        seed, feature_set, model_buy, model_sell, scaler
    ) = params

    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Starting optimization...\n")

    df_backtest_opt = train_data.copy()
    if df_backtest_opt.empty:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: No backtest data. Skipping optimization.\n")
        return {
            'ticker': ticker,
            'min_proba_buy': current_min_proba_buy,
            'min_proba_sell': current_min_proba_sell,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no data)"
        }

    # Reconstruct PyTorch models on GPU if they were passed as model_info dicts
    if PYTORCH_AVAILABLE:
        import torch
        from config import FORCE_CPU
        device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)

    if model_buy is None or model_sell is None or scaler is None:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Models or scaler not provided. Skipping optimization.\n")
        return {
            'ticker': ticker,
            'min_proba_buy': current_min_proba_buy,
            'min_proba_sell': current_min_proba_sell,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no models)"
        }

    best_alpha = -np.inf
    best_revenue = -np.inf
    best_min_proba_buy = current_min_proba_buy
    best_min_proba_sell = current_min_proba_sell
    best_class_horizon = initial_class_horizon

    # Store all tested combinations for backtesting
    tested_combinations = []  # List of dicts with params and revenue

    # ITERATIVE HILL-CLIMBING OPTIMIZATION
    # Start from current saved thresholds, try one step up/down, move if better
    # Probability threshold optimization removed - using simplified trading logic
    buy_options = [0.0]  # Single disabled threshold
    sell_options = [1.0]  # Single disabled threshold

    # Find closest indices to current thresholds
    current_buy_idx = min(range(len(buy_options)), key=lambda i: abs(buy_options[i] - current_min_proba_buy))
    current_sell_idx = min(range(len(sell_options)), key=lambda i: abs(sell_options[i] - current_min_proba_sell))

    p_buy = buy_options[current_buy_idx]
    p_sell = sell_options[current_sell_idx]

    print(f"  🔍 Iterative optimization for {ticker} starting from Buy={p_buy:.2f}, Sell={p_sell:.2f}...")

    # Helper function to test a single combination
    def _test_combination(p_buy, p_sell):
        env = RuleTradingEnv(
            df=df_backtest_opt.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model=model_buy,  # Use single model
            scaler=scaler,
            y_scaler=y_scaler,
            use_gate=False,  # Simplified buy-and-hold logic
            feature_set=feature_set
        )
        final_val, _, _, _, _, _ = env.run()
        revenue = final_val

        # Calculate buy & hold for this combination
        start_price_bh = float(df_backtest_opt["Close"].iloc[0])
        end_price_bh = float(df_backtest_opt["Close"].iloc[-1])
        shares_bh = int(capital_per_stock / start_price_bh) if start_price_bh > 0 else 0
        cash_bh = capital_per_stock - shares_bh * start_price_bh
        buy_hold_final_val = cash_bh + shares_bh * end_price_bh
        buy_hold_revenue = buy_hold_final_val - capital_per_stock

        # Calculate alpha from portfolio history vs buy & hold
        portfolio_history = env.portfolio_history
        if len(portfolio_history) > 1 and len(df_backtest_opt) > 1:
            # Calculate daily returns for strategy (portfolio value changes)
            strategy_values = pd.Series(portfolio_history)
            strategy_returns = strategy_values.pct_change(fill_method=None).dropna()

            # Calculate daily returns for buy & hold (price changes)
            close_prices = pd.to_numeric(df_backtest_opt["Close"], errors='coerce').dropna()
            bh_returns = close_prices.pct_change(fill_method=None).dropna()

            # Align returns - portfolio_history has same length as df_backtest_opt rows
            # Both should start from index 1 (after pct_change().dropna())
            min_len = min(len(strategy_returns), len(bh_returns))
            if min_len > 1:  # Need at least 2 data points for regression
                strategy_returns_aligned = strategy_returns.iloc[:min_len].values
                bh_returns_aligned = bh_returns.iloc[:min_len].values

                # Calculate alpha using OLS regression: strategy_ret = alpha + beta * bh_ret
                # Alpha is the intercept, annualized
                try:
                    X = np.column_stack([np.ones_like(bh_returns_aligned), bh_returns_aligned])
                    beta_coeffs = np.linalg.lstsq(X, strategy_returns_aligned, rcond=None)[0]
                    alpha_per_day = float(beta_coeffs[0])
                    alpha_annualized = alpha_per_day * CALENDAR_DAYS_PER_YEAR  # Annualize using calendar days per year
                except Exception as e:
                    alpha_annualized = 0.0
            else:
                alpha_annualized = 0.0

        # Calculate percentages
        revenue_pct = ((revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
        bh_revenue_pct = (buy_hold_revenue / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
        diff = revenue - buy_hold_final_val
        diff_pct = revenue_pct - bh_revenue_pct

        # Print immediately after each combination is tested
        # Use sys.stdout.write with flush to ensure ticker-specific output doesn't get mixed
        sys.stdout.write(f"  [{ticker}] Buy={p_buy:.2f}, Sell={p_sell:.2f} → "
              f"AI: ${revenue:,.2f} ({revenue_pct:+.2f}%), B&H: ${buy_hold_final_val:,.2f} ({bh_revenue_pct:+.2f}%), "
              f"Alpha: {alpha_annualized:.4f}\n")
        sys.stdout.flush()

        return revenue, alpha_annualized, buy_hold_final_val, buy_hold_revenue

    # Test initial position
    revenue_current, alpha_current, bh_val_current, bh_rev_current = _test_combination(p_buy, p_sell)
    best_revenue = revenue_current
    best_alpha = alpha_current
    best_min_proba_buy = p_buy
    best_min_proba_sell = p_sell

    tested_combinations.append({
        'min_proba_buy': p_buy,
        'min_proba_sell': p_sell,
        'class_horizon': initial_class_horizon,
        'revenue': revenue_current,
        'buy_hold_revenue': bh_rev_current,
        'buy_hold_final_val': bh_val_current,
        'alpha_annualized': alpha_current,
        'model_buy': model_buy,
        'model_sell': model_sell,
        'scaler': scaler
    })

    # Iterative hill-climbing: Test one step in each direction until no improvement
    max_iterations = 20  # Safety limit
    iteration = 0
    improvement_found = True

    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1

        # Get current indices
        current_buy_idx = buy_options.index(best_min_proba_buy)
        current_sell_idx = sell_options.index(best_min_proba_sell)

        # Test 4 neighbors: buy-1, buy+1, sell-1, sell+1
        neighbors = []

        if current_buy_idx > 0:
            neighbors.append((buy_options[current_buy_idx - 1], best_min_proba_sell, 'buy_down'))
        if current_buy_idx < len(buy_options) - 1:
            neighbors.append((buy_options[current_buy_idx + 1], best_min_proba_sell, 'buy_up'))
        if current_sell_idx > 0:
            neighbors.append((best_min_proba_buy, sell_options[current_sell_idx - 1], 'sell_down'))
        if current_sell_idx < len(sell_options) - 1:
            neighbors.append((best_min_proba_buy, sell_options[current_sell_idx + 1], 'sell_up'))

        # Test each neighbor
        for test_buy, test_sell, direction in neighbors:
            revenue, alpha, bh_val, bh_rev = _test_combination(test_buy, test_sell)

            tested_combinations.append({
                'min_proba_buy': test_buy,
                'min_proba_sell': test_sell,
                'class_horizon': initial_class_horizon,
                'revenue': revenue,
                'buy_hold_revenue': bh_rev,
                'buy_hold_final_val': bh_val,
                'alpha_annualized': alpha,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler
            })

            # If this neighbor is better, move there
            if alpha > best_alpha:
                best_alpha = alpha
                best_revenue = revenue
                best_min_proba_buy = test_buy
                best_min_proba_sell = test_sell
                improvement_found = True
                sys.stdout.write(f"  [{ticker}] ✨ Improvement found ({direction}): Buy={test_buy:.2f}, Sell={test_sell:.2f}, Alpha={alpha:.4f}\n")
                sys.stdout.flush()
                break  # Move to this position and start again

    if iteration > 1:
        sys.stdout.write(f"  [{ticker}] 🎯 Converged after {iteration} iterations with {len(tested_combinations)} tests\n")
        sys.stdout.flush()

    best_class_horizon = initial_class_horizon
    optimization_status = "Optimized" if iteration > 1 else "No Change"
    if not np.isclose(best_min_proba_buy, current_min_proba_buy) or \
       not np.isclose(best_min_proba_sell, current_min_proba_sell) or \
       not np.isclose(best_class_horizon, initial_class_horizon):
        optimization_status = "Optimized"

    # Calculate buy & hold for the best combination
    start_price_bh_best = float(df_backtest_opt["Close"].iloc[0])
    end_price_bh_best = float(df_backtest_opt["Close"].iloc[-1])
    shares_bh_best = int(capital_per_stock / start_price_bh_best) if start_price_bh_best > 0 else 0
    cash_bh_best = capital_per_stock - shares_bh_best * start_price_bh_best
    buy_hold_final_val_best = cash_bh_best + shares_bh_best * end_price_bh_best
    buy_hold_revenue_best = buy_hold_final_val_best - capital_per_stock
    revenue_pct_best = ((best_revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
    bh_revenue_pct_best = (buy_hold_revenue_best / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
    diff_best = best_revenue - buy_hold_final_val_best
    diff_pct_best = revenue_pct_best - bh_revenue_pct_best

    # Recalculate alpha for the best combination
    best_alpha_final = 0.0
    if best_revenue > -np.inf:
        # Find the best combination in tested_combinations
        for combo in tested_combinations:
            if (np.isclose(combo['min_proba_buy'], best_min_proba_buy) and
                np.isclose(combo['min_proba_sell'], best_min_proba_sell) and
                np.isclose(combo['class_horizon'], best_class_horizon)):
                best_alpha_final = combo.get('alpha_annualized', 0.0)
                break

    # Validation: Check if selected parameters beat B&H in revenue
    # If not, find the best combination that beats B&H (by revenue)
    revenue_beats_bh = best_revenue > buy_hold_final_val_best
    if not revenue_beats_bh and tested_combinations:
        # Find combinations that beat B&H
        combinations_beating_bh = [c for c in tested_combinations if c['revenue'] > c['buy_hold_final_val']]

        if combinations_beating_bh:
            # Among those that beat B&H, find the one with highest alpha
            best_beating_bh = max(combinations_beating_bh, key=lambda x: x.get('alpha_annualized', -np.inf))

            # If the best alpha combination doesn't beat B&H, but we found one that does, use it
            if best_beating_bh.get('alpha_annualized', -np.inf) > -np.inf:
                print(f"  ⚠️ [{ticker}] WARNING: Best alpha combination (Alpha={best_alpha_final:.4f}) does NOT beat B&H in revenue.")
                print(f"     Selecting alternative: Alpha={best_beating_bh.get('alpha_annualized', 0.0):.4f} that beats B&H")
                best_alpha_final = best_beating_bh.get('alpha_annualized', 0.0)
                best_revenue = best_beating_bh['revenue']
                best_min_proba_buy = best_beating_bh['min_proba_buy']
                best_min_proba_sell = best_beating_bh['min_proba_sell']
                best_class_horizon = best_beating_bh['class_horizon']
                buy_hold_final_val_best = best_beating_bh['buy_hold_final_val']
                buy_hold_revenue_best = best_beating_bh['buy_hold_revenue']
                revenue_pct_best = ((best_revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                bh_revenue_pct_best = (buy_hold_revenue_best / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                diff_best = best_revenue - buy_hold_final_val_best
                diff_pct_best = revenue_pct_best - bh_revenue_pct_best
                revenue_beats_bh = True
        else:
            # No combination beats B&H - warn but keep the best alpha
            print(f"  ⚠️ [{ticker}] WARNING: No tested combination beats Buy & Hold in revenue!")
            print(f"     Best alpha combination selected, but revenue is ${best_revenue:,.2f} vs B&H ${buy_hold_final_val_best:,.2f}")

    # Final validation: Ensure selected values beat B&H
    # Recalculate to be absolutely sure
    final_revenue_beats_bh = best_revenue > buy_hold_final_val_best

    # Print summary of selected values
    revenue_status = "✅ Beats B&H" if final_revenue_beats_bh else "❌ Below B&H"
    print(f"\n  ✅ [{ticker}] Optimization complete - Selected values (optimized for highest alpha):")
    print(f"     Horizon={best_class_horizon}, Buy={best_min_proba_buy:.2f}, Sell={best_min_proba_sell:.2f}")
    print(f"     Best Alpha (annualized): {best_alpha_final:.4f}")
    print(f"     Best AI Revenue: ${best_revenue:,.2f} ({revenue_pct_best:+.2f}%) {revenue_status}")
    print(f"     Buy & Hold Revenue: ${buy_hold_final_val_best:,.2f} ({bh_revenue_pct_best:+.2f}%)")
    print(f"     Difference: ${diff_best:,.2f} ({diff_pct_best:+.2f}%)")

    # Add explicit check result
    if not final_revenue_beats_bh:
        print(f"     ⚠️  WARNING: Selected parameters do NOT beat Buy & Hold in revenue!")
        print(f"        AI Strategy: ${best_revenue:,.2f} vs Buy & Hold: ${buy_hold_final_val_best:,.2f}")
        print(f"        Shortfall: ${buy_hold_final_val_best - best_revenue:,.2f}")
    else:
        print(f"     ✅ SUCCESS: Selected parameters beat Buy & Hold by ${diff_best:,.2f} ({diff_pct_best:+.2f}%)")

    print(f"     Status: {optimization_status}\n")

    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Optimization complete. Best Alpha={best_alpha_final:.4f}, Best Revenue=${best_revenue:,.2f}, Beats B&H={final_revenue_beats_bh}, Status: {optimization_status}\n")
    return {
        'ticker': ticker,
        'min_proba_buy': best_min_proba_buy,
        'min_proba_sell': best_min_proba_sell,
        'class_horizon': best_class_horizon,
        'best_revenue': best_revenue,
        'buy_hold_revenue': buy_hold_final_val_best,
        'revenue_beats_bh': final_revenue_beats_bh,  # Add explicit flag
        'optimization_status': optimization_status,
        'tested_combinations': tested_combinations  # Return all tested combinations
    }

def optimize_thresholds_for_portfolio_parallel(optimization_params, num_processes=None):
    import gc
    num_processes = num_processes or max(1, cpu_count() - 5)
    print(f"\nOptimizing thresholds for {len(optimization_params)} tickers using {num_processes} processes...")

    with Pool(processes=num_processes, maxtasksperchild=20) as pool:
        results = list(tqdm(
            pool.imap_unordered(optimize_single_ticker_worker, optimization_params),
            total=len(optimization_params),
            desc="Optimizing Thresholds"
        ))
    gc.collect()  # Release semaphores after Pool closes (WSL fix)

    optimized_params = {}
    all_tested_combinations = {}  # Store all tested combinations per ticker

    for res in results:
        if res and res.get('ticker'):
            optimized_params[res['ticker']] = {
                k: res[k] for k in ['min_proba_buy', 'min_proba_sell', 'class_horizon', 'optimization_status', 'revenue_beats_bh']
            }
            if 'tested_combinations' in res and res['tested_combinations']:
                all_tested_combinations[res['ticker']] = res['tested_combinations']
            beats_bh_status = "✅ Beats B&H" if res.get('revenue_beats_bh', False) else "❌ Below B&H"
            print(f"Optimized {res['ticker']}: Buy>{res['min_proba_buy']:.2f}, Sell>{res['min_proba_sell']:.2f}, "
                  f"Horizon={res['class_horizon']}d → {res['optimization_status']} | {beats_bh_status}")

    return optimized_params, all_tested_combinations


# -----------------------------------------------------------------------------
# Backtest worker function
# -----------------------------------------------------------------------------
def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, y_scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy, horizon_days = params

    # Initial log to confirm the worker has started for a ticker
    with open("logs/worker_debug.log", "a") as f:
        f.write(f"Worker started for ticker: {ticker}\n")

    # Reconstruct PyTorch models from prepared dict format
    if PYTORCH_AVAILABLE:
        import torch
        from config import FORCE_CPU
        device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)

    if df_backtest.empty:
        print(f"  ⚠️ Skipping backtest for {ticker}: DataFrame is empty.")
        return None

    # DEBUG: Log simplified approach
    import sys
    sys.stderr.write(f"\n[DEBUG {ticker}] Simplified backtest: Buy at start, hold until end\n")
    sys.stderr.flush()

    try:
        env = RuleTradingEnv(
            df=df_backtest.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model=model_buy,  # Use single model
            scaler=scaler,
            y_scaler=y_scaler,  # ✅ Pass y_scaler
            use_gate=False,  # Simplified buy-and-hold logic
            feature_set=feature_set,
            horizon_days=horizon_days
        )
        final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation = env.run()

        # Calculate individual Buy & Hold for the same period
        start_price_bh = float(df_backtest["Close"].iloc[0])
        end_price_bh = float(df_backtest["Close"].iloc[-1])
        individual_bh_return = ((end_price_bh - start_price_bh) / start_price_bh) * 100 if start_price_bh > 0 else 0.0

        # Analyze performance for this ticker
        perf_data = analyze_performance(trade_log, env.portfolio_history, df_backtest["Close"].tolist(), ticker)

        # Calculate Buy & Hold history for this ticker
        bh_history_for_ticker = []
        if not df_backtest.empty:
            start_price = float(df_backtest["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            for price_day in df_backtest["Close"].tolist():
                bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
        else:
            bh_history_for_ticker.append(capital_per_stock)

        # Print prediction summary and store stats for final summary
        if hasattr(env, 'all_predictions_buy') and len(env.all_predictions_buy) > 0:
            import numpy as np
            preds_buy = np.array(env.all_predictions_buy)
            print(f"\n📊 [{ticker}] BUY Prediction Summary:")
            print(f"   Min: {np.min(preds_buy)*100:.4f}%, Max: {np.max(preds_buy)*100:.4f}%, Mean: {np.mean(preds_buy)*100:.4f}%")
            print(f"   Positive predictions: {np.sum(preds_buy > 0)} out of {len(preds_buy)} days ({np.sum(preds_buy > 0)/len(preds_buy)*100:.1f}%)")
            print(f"   Threshold: {min_proba_buy*100:.2f}%")
            pred_min_pct = float(np.min(preds_buy) * 100)
            pred_max_pct = float(np.max(preds_buy) * 100)
            pred_mean_pct = float(np.mean(preds_buy) * 100)
        else:
            pred_min_pct = pred_max_pct = pred_mean_pct = None

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob,
            'shares_before_liquidation': shares_before_liquidation,
            'pred_min_pct': pred_min_pct,
            'pred_max_pct': pred_max_pct,
            'pred_mean_pct': pred_mean_pct,
            'buy_hold_history': bh_history_for_ticker
        }
    finally:
        # This block will execute whether an exception occurred or not.
        with open("logs/worker_debug.log", "a") as f:
            final_val_to_log = 'Error' if 'final_val' not in locals() else final_val
            f.write(f"Worker finished for ticker: {ticker}. Final Value: {final_val_to_log}\n")


def analyze_performance(
    trade_log: List[tuple],
    strategy_history: List[float],
    buy_hold_history: List[float],
    ticker: str
) -> Dict[str, float]:
    """Analyzes trades and calculates key performance metrics."""
    # --- Trade Analysis ---
    buys = [t for t in trade_log if t[1] == "BUY"]
    sells = [t for t in trade_log if t[1] == "SELL"]
    profits = []
    n = min(len(buys), len(sells))
    for i in range(n):
        pb, sb = float(buys[i][2]), float(sells[i][2])
        qb, qs = float(buys[i][3]), float(sells[i][3])
        qty = min(qb, qs)
        fee_b = float(buys[i][6]) if len(buys[i]) > 6 else 0.0
        fee_s = float(sells[i][6]) if len(sells[i]) > 6 else 0.0
        profits.append((sb - pb) * qty - (fee_b + fee_s))

    total_pnl = float(sum(profits))
    win_rate = (sum(1 for p in profits if p > 0) / len(profits)) if profits else 0.0
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  - Trades: {n}, Win Rate: {win_rate:.2%}")
    print(f"  - Total PnL: ${total_pnl:,.2f}")

    # --- Performance Metrics ---
    strat_returns = pd.Series(strategy_history).pct_change(fill_method=None).dropna()
    bh_returns = pd.Series(buy_hold_history).pct_change(fill_method=None).dropna()

    # Sharpe Ratio (annualized, using calendar days per year)
    sharpe_strat = (strat_returns.mean() / strat_returns.std()) * np.sqrt(CALENDAR_DAYS_PER_YEAR) if strat_returns.std() > 0 else 0
    sharpe_bh = (bh_returns.mean() / bh_returns.std()) * np.sqrt(CALENDAR_DAYS_PER_YEAR) if bh_returns.std() > 0 else 0

    # Max Drawdown
    strat_series = pd.Series(strategy_history)
    strat_cummax = strat_series.cummax()
    strat_drawdown = ((strat_series - strat_cummax) / strat_cummax).min()

    bh_series = pd.Series(buy_hold_history)
    bh_cummax = bh_series.cummax()
    bh_drawdown = ((bh_series - bh_cummax) / bh_cummax).min()

    print(f"\n📈 {ticker} Performance Metrics:")
    print(f"  | Metric         | Strategy      | Buy & Hold    |")
    print(f"  |----------------|---------------|---------------|")
    print(f"  | Sharpe Ratio   | {sharpe_strat:13.2f} | {sharpe_bh:13.2f} |")
    print(f"  | Max Drawdown   | {strat_drawdown:12.2%} | {bh_drawdown:12.2%} |")

    return {
        "trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe_ratio": sharpe_strat, "max_drawdown": strat_drawdown
    }


# -----------------------------------------------------------------------------
# Portfolio-level backtesting
# -----------------------------------------------------------------------------

def _run_portfolio_backtest_walk_forward(
    all_tickers_data: pd.DataFrame,
    backtest_start_date: datetime,
    backtest_end_date: datetime,
    initial_top_tickers: List[str],
    capital_per_stock: float,
    period_name: str,
    top_performers_data: List[Tuple],
    horizon_days: int = 20,
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]], float, float, List[float], float, List[float], float, List[float]]:
    """
    Walk-forward backtest: Daily selection from top 40 stocks with 10-day retraining.

    Your desired approach (NOW IMPLEMENTED):
    - Initial selection: Top 40 stocks by momentum (N_TOP_TICKERS = 40)
    - Model retraining: Every 10 days for all 40 stocks
    - Daily selection: Use current models to pick best 3 from 40 stocks EVERY DAY
    - Portfolio: Rebalance only when selection changes (cost-effective)
    """

    print(f"🔄 Walk-forward backtest for {period_name}")
    print(f"   📊 Universe: Top {len(initial_top_tickers)} stocks by momentum")
    print(f"   💰 Running all enabled strategies")

    # Clear inverse ETF hedge log at start of backtest
    from shared_strategies import clear_inverse_etf_hedge_log, get_strategy_tickers, select_top_performers, select_top_performers_vol_filtered
    clear_inverse_etf_hedge_log()

    # Centralized rebalancing tracking for all strategies (strategy_name -> bool)
    strategies_rebalanced_today = {}

    # Initialize - AI Elite models are trained during walk-forward, not loaded upfront

    # Track current portfolio (starts empty)
    current_portfolio_stocks = []
    total_portfolio_value = 0.0  # Start with no capital invested
    portfolio_values_history = [total_portfolio_value]

    # Track actual positions for proper portfolio management
    positions = {}  # ticker -> {'shares': float, 'avg_price': float, 'value': float}
    cash_balance = 0.0  # Available cash

    # Calculate initial capital that should be allocated (PORTFOLIO_SIZE stocks * capital_per_stock)
    from config import PORTFOLIO_SIZE
    initial_capital_needed = PORTFOLIO_SIZE * capital_per_stock
    cash_balance = initial_capital_needed  # Start with cash available for initial purchases

    # Fixed benchmark: top PORTFOLIO_SIZE performers selected once at backtest start.
    benchmark_top_start_name = f"Start Top {PORTFOLIO_SIZE}"
    benchmark_top_start_portfolio_value = initial_capital_needed
    benchmark_top_start_portfolio_history = []
    benchmark_top_start_positions = {}
    benchmark_top_start_cash = initial_capital_needed
    benchmark_top_start_stocks = []
    benchmark_top_start_transaction_costs = 0.0


    # Track STATIC BH 1Y PORTFOLIO (with optional periodic rebalancing)
    static_bh_1y_portfolio_value = initial_capital_needed
    static_bh_1y_portfolio_history = [static_bh_1y_portfolio_value]
    static_bh_1y_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_1y_cash = initial_capital_needed
    current_static_bh_1y_stocks = []
    static_bh_1y_days_since_rebalance = 0
    static_bh_1y_initialized = False

    # Track ADAPTIVE REBALANCING STRATEGIES (based on Static BH 1Y)
    # 1. Volatility-triggered
    static_bh_1y_vol_portfolio_value = initial_capital_needed
    static_bh_1y_vol_portfolio_history = [static_bh_1y_vol_portfolio_value]
    static_bh_1y_vol_positions = {}
    static_bh_1y_vol_cash = initial_capital_needed
    current_static_bh_1y_vol_stocks = []
    static_bh_1y_vol_initialized = False
    static_bh_1y_vol_transaction_costs = 0.0

    # 2. Performance-triggered
    static_bh_1y_perf_portfolio_value = initial_capital_needed
    static_bh_1y_perf_portfolio_history = [static_bh_1y_perf_portfolio_value]
    static_bh_1y_perf_positions = {}
    static_bh_1y_perf_cash = initial_capital_needed
    current_static_bh_1y_perf_stocks = []
    static_bh_1y_perf_initialized = False
    static_bh_1y_perf_transaction_costs = 0.0

    # 3. Momentum-triggered
    static_bh_1y_mom_portfolio_value = initial_capital_needed
    static_bh_1y_mom_portfolio_history = [static_bh_1y_mom_portfolio_value]
    static_bh_1y_mom_positions = {}
    static_bh_1y_mom_cash = initial_capital_needed
    current_static_bh_1y_mom_stocks = []
    static_bh_1y_mom_initialized = False
    static_bh_1y_mom_transaction_costs = 0.0

    # 4. ATR-triggered
    static_bh_1y_atr_portfolio_value = initial_capital_needed
    static_bh_1y_atr_portfolio_history = [static_bh_1y_atr_portfolio_value]
    static_bh_1y_atr_positions = {}
    static_bh_1y_atr_cash = initial_capital_needed
    current_static_bh_1y_atr_stocks = []
    static_bh_1y_atr_initialized = False
    static_bh_1y_atr_transaction_costs = 0.0

    # 5. Hybrid
    static_bh_1y_hybrid_portfolio_value = initial_capital_needed
    static_bh_1y_hybrid_portfolio_history = [static_bh_1y_hybrid_portfolio_value]
    static_bh_1y_hybrid_positions = {}
    static_bh_1y_hybrid_cash = initial_capital_needed
    current_static_bh_1y_hybrid_stocks = []
    static_bh_1y_hybrid_initialized = False
    static_bh_1y_hybrid_transaction_costs = 0.0
    static_bh_1y_hybrid_days_since_rebalance = 0

    # 6. Volume Filter
    static_bh_1y_volume_portfolio_value = initial_capital_needed
    static_bh_1y_volume_portfolio_history = [static_bh_1y_volume_portfolio_value]
    static_bh_1y_volume_positions = {}
    static_bh_1y_volume_cash = initial_capital_needed
    current_static_bh_1y_volume_stocks = []
    static_bh_1y_volume_initialized = False
    static_bh_1y_volume_transaction_costs = 0.0

    # 7. Sector Rotation
    static_bh_1y_sector_portfolio_value = initial_capital_needed
    static_bh_1y_sector_portfolio_history = [static_bh_1y_sector_portfolio_value]
    static_bh_1y_sector_positions = {}
    static_bh_1y_sector_cash = initial_capital_needed
    current_static_bh_1y_sector_stocks = []
    static_bh_1y_sector_initialized = False
    static_bh_1y_sector_transaction_costs = 0.0

    # 8. Performance Threshold
    static_bh_1y_perf_threshold_portfolio_value = initial_capital_needed
    static_bh_1y_perf_threshold_portfolio_history = [static_bh_1y_perf_threshold_portfolio_value]
    static_bh_1y_perf_threshold_positions = {}
    static_bh_1y_perf_threshold_cash = initial_capital_needed
    current_static_bh_1y_perf_threshold_stocks = []
    static_bh_1y_perf_threshold_initialized = False
    static_bh_1y_perf_threshold_transaction_costs = 0.0

    # 9. Market Regime
    static_bh_1y_market_regime_portfolio_value = initial_capital_needed
    static_bh_1y_market_regime_portfolio_history = [static_bh_1y_market_regime_portfolio_value]
    static_bh_1y_market_regime_positions = {}
    static_bh_1y_market_regime_cash = initial_capital_needed
    current_static_bh_1y_market_regime_stocks = []
    static_bh_1y_market_regime_initialized = False
    static_bh_1y_market_regime_transaction_costs = 0.0
    static_bh_1y_market_regime_days_since_rebalance = 0
    static_bh_1y_market_regime_next_rebalance_days = STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS

    # 10. Momentum Persistence
    static_bh_1y_mom_persist_portfolio_value = initial_capital_needed
    static_bh_1y_mom_persist_portfolio_history = [static_bh_1y_mom_persist_portfolio_value]
    static_bh_1y_mom_persist_positions = {}
    static_bh_1y_mom_persist_cash = initial_capital_needed
    current_static_bh_1y_mom_persist_stocks = []
    static_bh_1y_mom_persist_initialized = False
    static_bh_1y_mom_persist_transaction_costs = 0.0
    static_bh_1y_mom_persist_top_stocks_history = []  # Track history of top stocks for persistence check

    # 11. Overlap-Based
    static_bh_1y_overlap_portfolio_value = initial_capital_needed
    static_bh_1y_overlap_portfolio_history = [static_bh_1y_overlap_portfolio_value]
    static_bh_1y_overlap_positions = {}
    static_bh_1y_overlap_cash = initial_capital_needed
    current_static_bh_1y_overlap_stocks = []
    static_bh_1y_overlap_initialized = False
    static_bh_1y_overlap_transaction_costs = 0.0

    # 12. Rank Drift
    static_bh_1y_rank_drift_portfolio_value = initial_capital_needed
    static_bh_1y_rank_drift_portfolio_history = [static_bh_1y_rank_drift_portfolio_value]
    static_bh_1y_rank_drift_positions = {}
    static_bh_1y_rank_drift_cash = initial_capital_needed
    current_static_bh_1y_rank_drift_stocks = []
    static_bh_1y_rank_drift_initialized = False
    static_bh_1y_rank_drift_transaction_costs = 0.0
    static_bh_1y_rank_drift_previous_rankings = {}  # Track previous rankings for drift calculation

    # 13. Drawdown Trigger
    static_bh_1y_drawdown_portfolio_value = initial_capital_needed
    static_bh_1y_drawdown_portfolio_history = [static_bh_1y_drawdown_portfolio_value]
    static_bh_1y_drawdown_positions = {}
    static_bh_1y_drawdown_cash = initial_capital_needed
    current_static_bh_1y_drawdown_stocks = []
    static_bh_1y_drawdown_initialized = False
    static_bh_1y_drawdown_transaction_costs = 0.0
    static_bh_1y_drawdown_peak = initial_capital_needed  # Track portfolio peak for drawdown calculation

    # 14. Smart Monthly
    static_bh_1y_smart_monthly_portfolio_value = initial_capital_needed
    static_bh_1y_smart_monthly_portfolio_history = [static_bh_1y_smart_monthly_portfolio_value]
    static_bh_1y_smart_monthly_positions = {}
    static_bh_1y_smart_monthly_cash = initial_capital_needed
    current_static_bh_1y_smart_monthly_stocks = []
    static_bh_1y_smart_monthly_initialized = False
    static_bh_1y_smart_monthly_transaction_costs = 0.0
    static_bh_1y_smart_monthly_peak = initial_capital_needed
    static_bh_1y_smart_monthly_last_rebalance_month = 0

    # =========================================================================
    # SMART REBALANCING STRATEGIES (6 new strategies based on Static BH 1Y)
    # =========================================================================

    # 1. BH 1Y Mom Sell - Momentum-Based Sell
    bh_1y_mom_sell_portfolio_value = initial_capital_needed
    bh_1y_mom_sell_portfolio_history = [bh_1y_mom_sell_portfolio_value]
    bh_1y_mom_sell_positions = {}
    bh_1y_mom_sell_cash = initial_capital_needed
    current_bh_1y_mom_sell_stocks = []
    bh_1y_mom_sell_initialized = False
    bh_1y_mom_sell_transaction_costs = 0.0
    bh_1y_mom_sell_days_since_rebalance = 0

    # 2. BH 1Y Rank Sell - Relative Strength Ranking
    bh_1y_rank_sell_portfolio_value = initial_capital_needed
    bh_1y_rank_sell_portfolio_history = [bh_1y_rank_sell_portfolio_value]
    bh_1y_rank_sell_positions = {}
    bh_1y_rank_sell_cash = initial_capital_needed
    current_bh_1y_rank_sell_stocks = []
    bh_1y_rank_sell_initialized = False
    bh_1y_rank_sell_transaction_costs = 0.0
    bh_1y_rank_sell_days_since_rebalance = 0
    bh_1y_rank_sell_entry_ranks = {}  # ticker -> entry rank

    # 3. BH 1Y Trailing Mom - Trailing Stop + Momentum
    bh_1y_trailing_mom_portfolio_value = initial_capital_needed
    bh_1y_trailing_mom_portfolio_history = [bh_1y_trailing_mom_portfolio_value]
    bh_1y_trailing_mom_positions = {}
    bh_1y_trailing_mom_cash = initial_capital_needed
    current_bh_1y_trailing_mom_stocks = []
    bh_1y_trailing_mom_initialized = False
    bh_1y_trailing_mom_transaction_costs = 0.0
    bh_1y_trailing_mom_days_since_rebalance = 0
    bh_1y_trailing_mom_peaks = {}  # ticker -> peak price

    # 4. BH 1Y Volume Confirm - Volume Confirmation
    bh_1y_volume_confirm_portfolio_value = initial_capital_needed
    bh_1y_volume_confirm_portfolio_history = [bh_1y_volume_confirm_portfolio_value]
    bh_1y_volume_confirm_positions = {}
    bh_1y_volume_confirm_cash = initial_capital_needed
    current_bh_1y_volume_confirm_stocks = []
    bh_1y_volume_confirm_initialized = False
    bh_1y_volume_confirm_transaction_costs = 0.0
    bh_1y_volume_confirm_days_since_rebalance = 0

    # 5. BH 1Y Sector Aware - Sector Rotation Awareness
    bh_1y_sector_aware_portfolio_value = initial_capital_needed
    bh_1y_sector_aware_portfolio_history = [bh_1y_sector_aware_portfolio_value]
    bh_1y_sector_aware_positions = {}
    bh_1y_sector_aware_cash = initial_capital_needed
    current_bh_1y_sector_aware_stocks = []
    bh_1y_sector_aware_initialized = False
    bh_1y_sector_aware_transaction_costs = 0.0
    bh_1y_sector_aware_days_since_rebalance = 0

    # 6. BH 1Y Accel Buy - Accelerating Momentum Buy
    bh_1y_accel_buy_portfolio_value = initial_capital_needed
    bh_1y_accel_buy_portfolio_history = [bh_1y_accel_buy_portfolio_value]
    bh_1y_accel_buy_positions = {}
    bh_1y_accel_buy_cash = initial_capital_needed
    current_bh_1y_accel_buy_stocks = []
    bh_1y_accel_buy_initialized = False
    bh_1y_accel_buy_transaction_costs = 0.0
    bh_1y_accel_buy_days_since_rebalance = 0

    # Static BH 3M with Acceleration
    static_bh_3m_accel_portfolio_value = initial_capital_needed
    static_bh_3m_accel_portfolio_history = [static_bh_3m_accel_portfolio_value]
    static_bh_3m_accel_positions = {}
    static_bh_3m_accel_cash = initial_capital_needed
    current_static_bh_3m_accel_stocks = []
    static_bh_3m_accel_initialized = False
    static_bh_3m_accel_transaction_costs = 0.0
    static_bh_3m_accel_days_since_rebalance = 0

    # =====================================================================
    # 10 NEW REBALANCING STRATEGIES (Based on Static BH 1Y)
    # =====================================================================

    # 1. Volatility-Adjusted Rebalancing
    bh_1y_vol_adj_rebal_portfolio_value = initial_capital_needed
    bh_1y_vol_adj_rebal_portfolio_history = [bh_1y_vol_adj_rebal_portfolio_value]
    bh_1y_vol_adj_rebal_positions = {}
    bh_1y_vol_adj_rebal_cash = initial_capital_needed
    current_bh_1y_vol_adj_rebal_stocks = []
    bh_1y_vol_adj_rebal_initialized = False
    bh_1y_vol_adj_rebal_transaction_costs = 0.0
    bh_1y_vol_adj_rebal_days_since_rebalance = 0

    # 1b. Volatility-Adjusted Rebalancing with 1M/3M Ratio ranking
    ratio_1m3m_vol_adj_rebal_portfolio_value = initial_capital_needed
    ratio_1m3m_vol_adj_rebal_portfolio_history = [ratio_1m3m_vol_adj_rebal_portfolio_value]
    ratio_1m3m_vol_adj_rebal_positions = {}
    ratio_1m3m_vol_adj_rebal_cash = initial_capital_needed
    current_ratio_1m3m_vol_adj_rebal_stocks = []
    ratio_1m3m_vol_adj_rebal_initialized = False
    ratio_1m3m_vol_adj_rebal_transaction_costs = 0.0
    ratio_1m3m_vol_adj_rebal_days_since_rebalance = 0

    # 2. Correlation-Based Filtering
    bh_1y_corr_filter_portfolio_value = initial_capital_needed
    bh_1y_corr_filter_portfolio_history = [bh_1y_corr_filter_portfolio_value]
    bh_1y_corr_filter_positions = {}
    bh_1y_corr_filter_cash = initial_capital_needed
    current_bh_1y_corr_filter_stocks = []
    bh_1y_corr_filter_initialized = False
    bh_1y_corr_filter_transaction_costs = 0.0
    bh_1y_corr_filter_days_since_rebalance = 0

    # 3. Market Regime Detection
    bh_1y_regime_aware_portfolio_value = initial_capital_needed
    bh_1y_regime_aware_portfolio_history = [bh_1y_regime_aware_portfolio_value]
    bh_1y_regime_aware_positions = {}
    bh_1y_regime_aware_cash = initial_capital_needed
    current_bh_1y_regime_aware_stocks = []
    bh_1y_regime_aware_initialized = False
    bh_1y_regime_aware_transaction_costs = 0.0
    bh_1y_regime_aware_days_since_rebalance = 0
    bh_1y_regime_aware_current_regime = 'sideways'

    # 4. Risk Parity Allocation
    bh_1y_risk_parity_portfolio_value = initial_capital_needed
    bh_1y_risk_parity_portfolio_history = [bh_1y_risk_parity_portfolio_value]
    bh_1y_risk_parity_positions = {}
    bh_1y_risk_parity_cash = initial_capital_needed
    current_bh_1y_risk_parity_stocks = []
    bh_1y_risk_parity_initialized = False
    bh_1y_risk_parity_transaction_costs = 0.0
    bh_1y_risk_parity_days_since_rebalance = 0

    # 5. Adaptive Drift Threshold
    bh_1y_drift_thresh_portfolio_value = initial_capital_needed
    bh_1y_drift_thresh_portfolio_history = [bh_1y_drift_thresh_portfolio_value]
    bh_1y_drift_thresh_positions = {}
    bh_1y_drift_thresh_cash = initial_capital_needed
    current_bh_1y_drift_thresh_stocks = []
    bh_1y_drift_thresh_initialized = False
    bh_1y_drift_thresh_transaction_costs = 0.0
    bh_1y_drift_thresh_days_since_rebalance = 0

    # 6. Momentum Quality Score
    bh_1y_mom_quality_portfolio_value = initial_capital_needed
    bh_1y_mom_quality_portfolio_history = [bh_1y_mom_quality_portfolio_value]
    bh_1y_mom_quality_positions = {}
    bh_1y_mom_quality_cash = initial_capital_needed
    current_bh_1y_mom_quality_stocks = []
    bh_1y_mom_quality_initialized = False
    bh_1y_mom_quality_transaction_costs = 0.0
    bh_1y_mom_quality_days_since_rebalance = 0

    # 7. Liquidity-Based Sizing
    bh_1y_liquidity_portfolio_value = initial_capital_needed
    bh_1y_liquidity_portfolio_history = [bh_1y_liquidity_portfolio_value]
    bh_1y_liquidity_positions = {}
    bh_1y_liquidity_cash = initial_capital_needed
    current_bh_1y_liquidity_stocks = []
    bh_1y_liquidity_initialized = False
    bh_1y_liquidity_transaction_costs = 0.0
    bh_1y_liquidity_days_since_rebalance = 0

    # 8. Earnings Avoidance
    bh_1y_earnings_avoid_portfolio_value = initial_capital_needed
    bh_1y_earnings_avoid_portfolio_history = [bh_1y_earnings_avoid_portfolio_value]
    bh_1y_earnings_avoid_positions = {}
    bh_1y_earnings_avoid_cash = initial_capital_needed
    current_bh_1y_earnings_avoid_stocks = []
    bh_1y_earnings_avoid_initialized = False
    bh_1y_earnings_avoid_transaction_costs = 0.0
    bh_1y_earnings_avoid_days_since_rebalance = 0

    # 9. Multi-Factor Composite
    bh_1y_multi_factor_portfolio_value = initial_capital_needed
    bh_1y_multi_factor_portfolio_history = [bh_1y_multi_factor_portfolio_value]
    bh_1y_multi_factor_positions = {}
    bh_1y_multi_factor_cash = initial_capital_needed
    current_bh_1y_multi_factor_stocks = []
    bh_1y_multi_factor_initialized = False
    bh_1y_multi_factor_transaction_costs = 0.0
    bh_1y_multi_factor_days_since_rebalance = 0

    # 10. Time-Decay Holdings
    bh_1y_time_decay_portfolio_value = initial_capital_needed
    bh_1y_time_decay_portfolio_history = [bh_1y_time_decay_portfolio_value]
    bh_1y_time_decay_positions = {}
    bh_1y_time_decay_cash = initial_capital_needed
    current_bh_1y_time_decay_stocks = []
    bh_1y_time_decay_initialized = False
    bh_1y_time_decay_transaction_costs = 0.0
    bh_1y_time_decay_days_since_rebalance = 0
    bh_1y_time_decay_entry_dates = {}  # ticker -> entry date

    # Track STATIC BH 3M PORTFOLIO (with optional periodic rebalancing)
    static_bh_3m_portfolio_value = initial_capital_needed
    static_bh_3m_portfolio_history = [static_bh_3m_portfolio_value]
    static_bh_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_3m_cash = initial_capital_needed
    current_static_bh_3m_stocks = []

    # Track STATIC BH 6M PORTFOLIO (with optional periodic rebalancing)
    static_bh_6m_portfolio_value = initial_capital_needed
    static_bh_6m_portfolio_history = [static_bh_6m_portfolio_value]
    static_bh_6m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_6m_cash = initial_capital_needed
    current_static_bh_6m_stocks = []
    static_bh_6m_days_since_rebalance = 0
    static_bh_6m_initialized = False
    static_bh_3m_days_since_rebalance = 0
    static_bh_3m_initialized = False

    # Track STATIC BH 1M PORTFOLIO (with optional periodic rebalancing)
    static_bh_1m_portfolio_value = initial_capital_needed
    static_bh_1m_portfolio_history = [static_bh_1m_portfolio_value]
    static_bh_1m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_1m_cash = initial_capital_needed
    current_static_bh_1m_stocks = []
    static_bh_1m_days_since_rebalance = 0
    static_bh_1m_initialized = False

    # Track STATIC BH MONTHLY REBALANCE variants (rebalance on first trading day of each month)
    # Static BH 1Y Monthly
    static_bh_1y_monthly_portfolio_value = initial_capital_needed
    static_bh_1y_monthly_portfolio_history = [static_bh_1y_monthly_portfolio_value]
    static_bh_1y_monthly_positions = {}
    static_bh_1y_monthly_cash = initial_capital_needed
    current_static_bh_1y_monthly_stocks = []
    static_bh_1y_monthly_initialized = False
    static_bh_1y_monthly_transaction_costs = 0.0
    static_bh_1y_monthly_last_month = None  # Track last rebalanced month

    # Static BH 6M Monthly
    static_bh_6m_monthly_portfolio_value = initial_capital_needed
    static_bh_6m_monthly_portfolio_history = [static_bh_6m_monthly_portfolio_value]
    static_bh_6m_monthly_positions = {}
    static_bh_6m_monthly_cash = initial_capital_needed
    current_static_bh_6m_monthly_stocks = []
    static_bh_6m_monthly_initialized = False
    static_bh_6m_monthly_transaction_costs = 0.0
    static_bh_6m_monthly_last_month = None

    # Static BH 3M Monthly
    static_bh_3m_monthly_portfolio_value = initial_capital_needed
    static_bh_3m_monthly_portfolio_history = [static_bh_3m_monthly_portfolio_value]
    static_bh_3m_monthly_positions = {}
    static_bh_3m_monthly_cash = initial_capital_needed
    current_static_bh_3m_monthly_stocks = []
    static_bh_3m_monthly_initialized = False
    static_bh_3m_monthly_transaction_costs = 0.0
    static_bh_3m_monthly_last_month = None

    # Static BH 1M Monthly
    static_bh_1m_monthly_portfolio_value = initial_capital_needed
    static_bh_1m_monthly_portfolio_history = [static_bh_1m_monthly_portfolio_value]
    static_bh_1m_monthly_positions = {}
    static_bh_1m_monthly_cash = initial_capital_needed
    current_static_bh_1m_monthly_stocks = []
    static_bh_1m_monthly_initialized = False
    static_bh_1m_monthly_transaction_costs = 0.0
    static_bh_1m_monthly_last_month = None

    # Track DYNAMIC BH PORTFOLIO (rebalances to top N performers periodically)
    dynamic_bh_portfolio_value = initial_capital_needed
    dynamic_bh_portfolio_history = [dynamic_bh_portfolio_value]
    dynamic_bh_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_stocks = []  # Current top N stocks held by dynamic BH
    dynamic_bh_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track DYNAMIC BH 1Y + VOLATILITY FILTER PORTFOLIO (same as Dynamic BH 1Y but with volatility filter)
    dynamic_bh_1y_vol_filter_portfolio_value = initial_capital_needed
    dynamic_bh_1y_vol_filter_portfolio_history = [dynamic_bh_1y_vol_filter_portfolio_value]
    dynamic_bh_1y_vol_filter_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_1y_vol_filter_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1y_vol_filter_stocks = []  # Current top N stocks held by dynamic BH 1Y vol filter
    dynamic_bh_1y_vol_filter_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track DYNAMIC BH 1Y + TRAILING STOP PORTFOLIO (same as Dynamic BH 1Y but with 20% trailing stop)
    dynamic_bh_1y_trailing_stop_portfolio_value = initial_capital_needed
    dynamic_bh_1y_trailing_stop_portfolio_history = [dynamic_bh_1y_trailing_stop_portfolio_value]
    dynamic_bh_1y_trailing_stop_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'peak_price': float}
    dynamic_bh_1y_trailing_stop_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1y_trailing_stop_stocks = []  # Current top stocks held by dynamic BH 1Y trailing stop
    dynamic_bh_1y_trailing_stop_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track SECTOR ROTATION PORTFOLIO
    sector_rotation_portfolio_value = initial_capital_needed
    sector_rotation_portfolio_history = [sector_rotation_portfolio_value]
    sector_rotation_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    sector_rotation_cash = initial_capital_needed  # Start with same capital as AI
    current_sector_rotation_etfs = []  # Current sector ETFs held
    sector_rotation_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance
    sector_rotation_last_rebalance_date = backtest_start_date  # Initialize to backtest start date

    # Track DYNAMIC BH 3-MONTH PORTFOLIO (rebalances to top N based on 3-month performance)
    dynamic_bh_3m_portfolio_value = initial_capital_needed
    dynamic_bh_3m_portfolio_history = [dynamic_bh_3m_portfolio_value]
    dynamic_bh_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_3m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_3m_stocks = []  # Current top N stocks held by 3-month dynamic BH
    dynamic_bh_3m_last_rebalance_value = initial_capital_needed

    # Track DYNAMIC BH 6-MONTH PORTFOLIO (rebalances to top N based on 6-month performance)
    dynamic_bh_6m_portfolio_value = initial_capital_needed
    dynamic_bh_6m_portfolio_history = [dynamic_bh_6m_portfolio_value]
    dynamic_bh_6m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_6m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_6m_stocks = []  # Current top N stocks held by 6-month dynamic BH
    dynamic_bh_6m_last_rebalance_value = initial_capital_needed

    # Track DYNAMIC BH 1-MONTH PORTFOLIO (rebalances to top N based on 1-month performance)
    dynamic_bh_1m_portfolio_value = initial_capital_needed
    dynamic_bh_1m_portfolio_history = [dynamic_bh_1m_portfolio_value]
    dynamic_bh_1m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_1m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1m_stocks = []  # Current top N stocks held by 1-month dynamic BH
    dynamic_bh_1m_last_rebalance_value = initial_capital_needed

    # RISK-ADJUSTED MOMENTUM: Initialize portfolio tracking
    risk_adj_mom_portfolio_value = initial_capital_needed
    risk_adj_mom_portfolio_history = [risk_adj_mom_portfolio_value]
    risk_adj_mom_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    risk_adj_mom_cash = initial_capital_needed  # Start with same capital as AI
    current_risk_adj_mom_stocks = []  # Current top N stocks held by risk-adjusted momentum
    risk_adj_mom_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # 3M/1Y RATIO: Initialize portfolio tracking
    ratio_3m_1y_portfolio_value = initial_capital_needed
    ratio_3m_1y_portfolio_history = [ratio_3m_1y_portfolio_value]
    ratio_3m_1y_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ratio_3m_1y_cash = initial_capital_needed  # Start with same capital as AI
    current_ratio_3m_1y_stocks = []  # Current top stocks held by 3M/1Y ratio strategy
    ratio_3m_1y_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # 1Y/3M RATIO: Initialize portfolio tracking
    ratio_1y_3m_portfolio_value = initial_capital_needed
    ratio_1y_3m_portfolio_history = [ratio_1y_3m_portfolio_value]
    ratio_1y_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ratio_1y_3m_cash = initial_capital_needed  # Start with same capital as AI
    current_ratio_1y_3m_stocks = []  # Current top stocks held by 1Y/3M ratio strategy
    ratio_1y_3m_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # 1M/3M RATIO: Initialize portfolio tracking (short-term momentum acceleration)
    ratio_1m_3m_portfolio_value = initial_capital_needed
    ratio_1m_3m_portfolio_history = [ratio_1m_3m_portfolio_value]
    ratio_1m_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ratio_1m_3m_cash = initial_capital_needed
    current_ratio_1m_3m_stocks = []
    ratio_1m_3m_last_rebalance_value = initial_capital_needed

    # MOMENTUM-VOLATILITY HYBRID: Initialize portfolio tracking
    momentum_volatility_hybrid_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_portfolio_history = [momentum_volatility_hybrid_portfolio_value]
    momentum_volatility_hybrid_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    momentum_volatility_hybrid_cash = initial_capital_needed  # Start with same capital as AI
    current_momentum_volatility_hybrid_stocks = []  # Current top stocks held by momentum-volatility hybrid strategy
    momentum_volatility_hybrid_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MOMENTUM-VOLATILITY HYBRID 6M: Initialize portfolio tracking
    momentum_volatility_hybrid_6m_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_6m_portfolio_history = [momentum_volatility_hybrid_6m_portfolio_value]
    momentum_volatility_hybrid_6m_positions = {}
    momentum_volatility_hybrid_6m_cash = initial_capital_needed
    current_momentum_volatility_hybrid_6m_stocks = []
    momentum_volatility_hybrid_6m_last_rebalance_value = initial_capital_needed

    # MOMENTUM-VOLATILITY HYBRID 1Y: Initialize portfolio tracking
    momentum_volatility_hybrid_1y_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_1y_portfolio_history = [momentum_volatility_hybrid_1y_portfolio_value]
    momentum_volatility_hybrid_1y_positions = {}
    momentum_volatility_hybrid_1y_cash = initial_capital_needed
    current_momentum_volatility_hybrid_1y_stocks = []
    momentum_volatility_hybrid_1y_last_rebalance_value = initial_capital_needed

    # MOMENTUM-VOLATILITY HYBRID 1Y/3M: Initialize portfolio tracking
    momentum_volatility_hybrid_1y3m_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_1y3m_portfolio_history = [momentum_volatility_hybrid_1y3m_portfolio_value]
    momentum_volatility_hybrid_1y3m_positions = {}
    momentum_volatility_hybrid_1y3m_cash = initial_capital_needed
    current_momentum_volatility_hybrid_1y3m_stocks = []
    momentum_volatility_hybrid_1y3m_last_rebalance_value = initial_capital_needed

    # PRICE ACCELERATION: Initialize portfolio tracking
    price_acceleration_portfolio_value = initial_capital_needed
    price_acceleration_portfolio_history = [price_acceleration_portfolio_value]
    price_acceleration_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    price_acceleration_cash = initial_capital_needed  # Start with same capital as AI
    current_price_acceleration_stocks = []  # Current top stocks held by price acceleration strategy
    price_acceleration_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # TURNAROUND: Initialize portfolio tracking
    turnaround_portfolio_value = initial_capital_needed
    turnaround_portfolio_history = [turnaround_portfolio_value]
    turnaround_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    turnaround_cash = initial_capital_needed  # Start with same capital as AI
    current_turnaround_stocks = []  # Current top stocks held by turnaround strategy
    turnaround_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # ADAPTIVE ENSEMBLE: Initialize portfolio tracking
    adaptive_ensemble_portfolio_value = initial_capital_needed
    adaptive_ensemble_portfolio_history = [adaptive_ensemble_portfolio_value]
    adaptive_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    adaptive_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_adaptive_ensemble_stocks = []  # Current top stocks held by adaptive ensemble
    adaptive_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # VOLATILITY ENSEMBLE: Initialize portfolio tracking
    volatility_ensemble_portfolio_value = initial_capital_needed
    volatility_ensemble_portfolio_history = [volatility_ensemble_portfolio_value]
    volatility_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    volatility_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_volatility_ensemble_stocks = []  # Current top stocks held by volatility ensemble
    volatility_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # ENHANCED VOLATILITY TRADER: Initialize portfolio tracking
    enhanced_volatility_portfolio_value = initial_capital_needed  # Start with full capital
    enhanced_volatility_portfolio_history = [enhanced_volatility_portfolio_value]
    enhanced_volatility_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'stop_loss': float, 'take_profit': float}
    enhanced_volatility_cash = initial_capital_needed  # Start with same capital as AI
    current_enhanced_volatility_stocks = []  # Current top stocks held by enhanced volatility trader
    enhanced_volatility_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # AI VOLATILITY ENSEMBLE: Initialize portfolio tracking
    ai_volatility_ensemble_portfolio_value = initial_capital_needed
    ai_volatility_ensemble_portfolio_history = [ai_volatility_ensemble_portfolio_value]
    ai_volatility_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ai_volatility_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_ai_volatility_ensemble_stocks = []  # Current stocks held by AI volatility ensemble
    ai_volatility_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MULTI-HORIZON ENSEMBLE: Initialize portfolio tracking
    multi_tf_ensemble_portfolio_value = initial_capital_needed
    multi_tf_ensemble_portfolio_history = [multi_tf_ensemble_portfolio_value]
    multi_tf_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    multi_tf_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_multi_tf_ensemble_stocks = []  # Current multi-timeframe ensemble stocks held
    multi_tf_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MULTI-HORIZON INTRADAY: Initialize portfolio tracking
    multi_tf_intraday_ensemble_portfolio_value = initial_capital_needed
    multi_tf_intraday_ensemble_portfolio_history = [multi_tf_intraday_ensemble_portfolio_value]
    multi_tf_intraday_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    multi_tf_intraday_ensemble_cash = initial_capital_needed
    current_multi_tf_intraday_ensemble_stocks = []
    multi_tf_intraday_ensemble_last_rebalance_value = initial_capital_needed

    # CORRELATION ENSEMBLE: Initialize portfolio tracking
    correlation_ensemble_portfolio_value = initial_capital_needed
    correlation_ensemble_portfolio_history = [correlation_ensemble_portfolio_value]
    correlation_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    correlation_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_correlation_ensemble_stocks = []  # Current top stocks held by correlation ensemble
    correlation_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # DYNAMIC POOL: Initialize portfolio tracking
    dynamic_pool_portfolio_value = initial_capital_needed
    dynamic_pool_portfolio_history = [dynamic_pool_portfolio_value]
    dynamic_pool_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_pool_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_pool_stocks = []  # Current top stocks held by dynamic pool
    dynamic_pool_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # SENTIMENT ENSEMBLE: Initialize portfolio tracking
    sentiment_ensemble_portfolio_value = initial_capital_needed
    sentiment_ensemble_portfolio_history = [sentiment_ensemble_portfolio_value]
    sentiment_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    sentiment_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_sentiment_ensemble_stocks = []  # Current top stocks held by sentiment ensemble
    sentiment_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # RISK-ADJ MOMENTUM SENTIMENT: Initialize portfolio tracking
    risk_adj_mom_sentiment_portfolio_value = initial_capital_needed
    risk_adj_mom_sentiment_portfolio_history = [risk_adj_mom_sentiment_portfolio_value]
    risk_adj_mom_sentiment_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    risk_adj_mom_sentiment_cash = initial_capital_needed  # Start with same capital as AI
    current_risk_adj_mom_sentiment_stocks = []  # Current top stocks held by risk adj mom sentiment
    risk_adj_mom_sentiment_last_rebalance_value = initial_capital_needed  # Transaction cost guard
    risk_adj_mom_sentiment_transaction_costs = 0.0

    # VOTING ENSEMBLE: Initialize portfolio tracking
    voting_ensemble_portfolio_value = initial_capital_needed
    voting_ensemble_portfolio_history = [voting_ensemble_portfolio_value]
    voting_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    voting_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_voting_ensemble_stocks = []  # Current top stocks held by voting ensemble
    voting_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # TOP 5 CONSISTENCY TRACKER: Count how many days each strategy is in top 5
    top5_consistency_counts = {}  # strategy_name -> count of days in top 5
    recent_top5_history = []  # rolling list of daily top-5 display names

    # AI CLASSIFICATION: Removed - stub variables for compatibility
    ai_classification_portfolio_value = 0
    ai_classification_portfolio_history = []
    ai_classification_positions = {}
    ai_classification_cash = 0
    ai_classification_transaction_costs = 0

    # MOMENTUM ACCELERATION: Initialize portfolio tracking
    mom_accel_portfolio_value = initial_capital_needed
    mom_accel_portfolio_history = [mom_accel_portfolio_value]
    mom_accel_positions = {}
    mom_accel_cash = initial_capital_needed
    current_mom_accel_stocks = []
    mom_accel_last_rebalance_value = initial_capital_needed

    # CONCENTRATED 3M: Initialize portfolio tracking
    concentrated_3m_portfolio_value = initial_capital_needed
    concentrated_3m_portfolio_history = [concentrated_3m_portfolio_value]
    concentrated_3m_positions = {}
    concentrated_3m_cash = initial_capital_needed
    current_concentrated_3m_stocks = []
    concentrated_3m_last_rebalance_value = initial_capital_needed
    concentrated_3m_days_since_rebalance = 0

    # DUAL MOMENTUM: Initialize portfolio tracking
    dual_mom_portfolio_value = initial_capital_needed
    dual_mom_portfolio_history = [dual_mom_portfolio_value]
    dual_mom_positions = {}
    dual_mom_cash = initial_capital_needed
    current_dual_mom_stocks = []
    dual_mom_is_risk_on = True

    # TREND FOLLOWING ATR: Initialize portfolio tracking
    trend_atr_portfolio_value = initial_capital_needed
    trend_atr_portfolio_history = [trend_atr_portfolio_value]
    trend_atr_positions = {}
    trend_atr_cash = initial_capital_needed
    current_trend_atr_stocks = []

    # ELITE HYBRID: Initialize portfolio tracking
    elite_hybrid_portfolio_value = initial_capital_needed
    elite_hybrid_portfolio_history = [elite_hybrid_portfolio_value]
    elite_hybrid_positions = {}
    elite_hybrid_cash = initial_capital_needed
    current_elite_hybrid_stocks = []
    elite_hybrid_last_rebalance_value = initial_capital_needed

    # ELITE RISK: Initialize portfolio tracking
    elite_risk_portfolio_value = initial_capital_needed
    elite_risk_portfolio_history = [elite_risk_portfolio_value]
    elite_risk_positions = {}
    elite_risk_cash = initial_capital_needed
    current_elite_risk_stocks = []
    elite_risk_last_rebalance_value = initial_capital_needed

    # RISK-ADJ MOM 6M: Initialize portfolio tracking
    risk_adj_mom_6m_portfolio_value = initial_capital_needed
    risk_adj_mom_6m_portfolio_history = [risk_adj_mom_6m_portfolio_value]
    risk_adj_mom_6m_positions = {}
    risk_adj_mom_6m_cash = initial_capital_needed
    current_risk_adj_mom_6m_stocks = []

    # RISK-ADJ MOM 6M MONTHLY: Initialize portfolio tracking (rebalance start of month only)
    risk_adj_mom_6m_monthly_portfolio_value = initial_capital_needed
    risk_adj_mom_6m_monthly_portfolio_history = [risk_adj_mom_6m_monthly_portfolio_value]
    risk_adj_mom_6m_monthly_positions = {}
    risk_adj_mom_6m_monthly_cash = initial_capital_needed
    current_risk_adj_mom_6m_monthly_stocks = []
    risk_adj_mom_6m_monthly_initialized = False
    risk_adj_mom_6m_monthly_last_month = None

    # RISK-ADJ MOM 3M: Initialize portfolio tracking
    risk_adj_mom_3m_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_portfolio_history = [risk_adj_mom_3m_portfolio_value]
    risk_adj_mom_3m_positions = {}
    risk_adj_mom_3m_cash = initial_capital_needed
    current_risk_adj_mom_3m_stocks = []

    # RISK-ADJ MOM 3M MONTHLY: Initialize portfolio tracking (rebalance start of month only)
    risk_adj_mom_3m_monthly_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_monthly_portfolio_history = [risk_adj_mom_3m_monthly_portfolio_value]
    risk_adj_mom_3m_monthly_positions = {}
    risk_adj_mom_3m_monthly_cash = initial_capital_needed
    current_risk_adj_mom_3m_monthly_stocks = []
    risk_adj_mom_3m_monthly_initialized = False
    risk_adj_mom_3m_monthly_last_month = None

    # RISK-ADJ MOM 3M SENTIMENT: Initialize portfolio tracking
    risk_adj_mom_3m_sentiment_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_sentiment_portfolio_history = [risk_adj_mom_3m_sentiment_portfolio_value]
    risk_adj_mom_3m_sentiment_positions = {}
    risk_adj_mom_3m_sentiment_cash = initial_capital_needed
    current_risk_adj_mom_3m_sentiment_stocks = []
    risk_adj_mom_3m_sentiment_transaction_costs = 0.0

    # RISK-ADJ MOM 3M MARKET-UP ONLY: Initialize portfolio tracking
    risk_adj_mom_3m_market_up_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_market_up_portfolio_history = [risk_adj_mom_3m_market_up_portfolio_value]
    risk_adj_mom_3m_market_up_positions = {}
    risk_adj_mom_3m_market_up_cash = initial_capital_needed
    current_risk_adj_mom_3m_market_up_stocks = []
    prev_risk_adj_mom_3m_market_up_stocks = []  # Track for rebalancing indicator
    risk_adj_mom_3m_market_up_rebalanced_today = False
    risk_adj_mom_3m_market_up_transaction_costs = 0.0

    # RISK-ADJ MOM 3M WITH STOPS: Initialize portfolio tracking
    risk_adj_mom_3m_with_stops_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_with_stops_portfolio_history = [risk_adj_mom_3m_with_stops_portfolio_value]
    risk_adj_mom_3m_with_stops_positions = {}
    risk_adj_mom_3m_with_stops_cash = initial_capital_needed
    current_risk_adj_mom_3m_with_stops_stocks = []
    prev_risk_adj_mom_3m_with_stops_stocks = []  # Track for rebalancing indicator
    risk_adj_mom_3m_with_stops_rebalanced_today = False
    risk_adj_mom_3m_with_stops_transaction_costs = 0.0

    # VOL-SWEET MOMENTUM: Initialize portfolio tracking
    vol_sweet_mom_portfolio_value = initial_capital_needed
    vol_sweet_mom_portfolio_history = [vol_sweet_mom_portfolio_value]
    vol_sweet_mom_positions = {}
    vol_sweet_mom_cash = initial_capital_needed
    current_vol_sweet_mom_stocks = []
    vol_sweet_mom_transaction_costs = 0.0

    # RISK-ADJ MOM 1M + VOL-SWEET: Initialize portfolio tracking
    risk_adj_mom_1m_vol_sweet_portfolio_value = initial_capital_needed
    risk_adj_mom_1m_vol_sweet_portfolio_history = [risk_adj_mom_1m_vol_sweet_portfolio_value]
    risk_adj_mom_1m_vol_sweet_positions = {}
    risk_adj_mom_1m_vol_sweet_cash = initial_capital_needed
    current_risk_adj_mom_1m_vol_sweet_stocks = []
    risk_adj_mom_1m_vol_sweet_transaction_costs = 0.0

    # BH 1Y VOL-SWEET ACCEL: Initialize portfolio tracking
    bh_1y_volsweet_accel_portfolio_value = initial_capital_needed
    bh_1y_volsweet_accel_portfolio_history = [bh_1y_volsweet_accel_portfolio_value]
    bh_1y_volsweet_accel_positions = {}
    bh_1y_volsweet_accel_cash = initial_capital_needed
    current_bh_1y_volsweet_accel_stocks = []
    bh_1y_volsweet_accel_transaction_costs = 0.0
    bh_1y_volsweet_accel_days_since_rebalance = 0

    # BH 1Y DYNAMIC ACCEL: Initialize portfolio tracking (dynamic rebalancing)
    bh_1y_dynamic_accel_portfolio_value = initial_capital_needed
    bh_1y_dynamic_accel_portfolio_history = [bh_1y_dynamic_accel_portfolio_value]
    bh_1y_dynamic_accel_positions = {}
    bh_1y_dynamic_accel_cash = initial_capital_needed
    current_bh_1y_dynamic_accel_stocks = []
    bh_1y_dynamic_accel_transaction_costs = 0.0
    bh_1y_dynamic_accel_days_since_rebalance = 0
    bh_1y_dynamic_accel_initialized = False

    # RISK-ADJ MOM 1M: Initialize portfolio tracking
    risk_adj_mom_1m_portfolio_value = initial_capital_needed
    risk_adj_mom_1m_portfolio_history = [risk_adj_mom_1m_portfolio_value]
    risk_adj_mom_1m_positions = {}
    risk_adj_mom_1m_cash = initial_capital_needed
    current_risk_adj_mom_1m_stocks = []

    # RISK-ADJ MOM 1M MONTHLY: Initialize portfolio tracking (rebalance start of month only)
    risk_adj_mom_1m_monthly_portfolio_value = initial_capital_needed
    risk_adj_mom_1m_monthly_portfolio_history = [risk_adj_mom_1m_monthly_portfolio_value]
    risk_adj_mom_1m_monthly_positions = {}
    risk_adj_mom_1m_monthly_cash = initial_capital_needed
    current_risk_adj_mom_1m_monthly_stocks = []
    risk_adj_mom_1m_monthly_initialized = False
    risk_adj_mom_1m_monthly_last_month = None

    # AI ELITE: Initialize portfolio tracking
    ai_elite_portfolio_value = initial_capital_needed
    ai_elite_portfolio_history = [ai_elite_portfolio_value] if config.ENABLE_AI_ELITE else []
    ai_elite_positions = {}
    ai_elite_cash = initial_capital_needed
    current_ai_elite_stocks = []
    ai_elite_last_rebalance_value = initial_capital_needed
    ai_elite_models = {}  # Per-ticker models: ticker -> model
    ai_elite_last_train_days = {}  # Per-ticker training tracking: ticker -> last_train_day

    # AI ELITE MONTHLY: Initialize portfolio tracking (same ML, monthly retrain + rebalance)
    ai_elite_monthly_portfolio_value = initial_capital_needed
    ai_elite_monthly_portfolio_history = [ai_elite_monthly_portfolio_value] if config.ENABLE_AI_ELITE_MONTHLY else []
    ai_elite_monthly_positions = {}
    ai_elite_monthly_cash = initial_capital_needed
    current_ai_elite_monthly_stocks = []
    ai_elite_monthly_transaction_costs = 0.0
    ai_elite_monthly_initialized = False
    ai_elite_monthly_last_month = None
    ai_elite_monthly_models = {}  # Separate model dict for monthly variant
    ai_elite_monthly_last_train_days = {}

    # AI ELITE FILTERED: Initialize portfolio tracking (Risk-Adj Mom 3M pre-filter + AI Elite re-rank)
    ai_elite_filtered_portfolio_value = initial_capital_needed
    ai_elite_filtered_portfolio_history = [ai_elite_filtered_portfolio_value] if config.ENABLE_AI_ELITE_FILTERED else []
    ai_elite_filtered_positions = {}
    ai_elite_filtered_cash = initial_capital_needed
    current_ai_elite_filtered_stocks = []
    ai_elite_filtered_transaction_costs = 0.0
    ai_elite_filtered_initialized = False

    # AI ELITE MARKET-UP: Initialize portfolio tracking (AI Elite only when market is up)
    ai_elite_market_up_portfolio_value = initial_capital_needed
    ai_elite_market_up_portfolio_history = [ai_elite_market_up_portfolio_value] if config.ENABLE_AI_ELITE_MARKET_UP else []
    ai_elite_market_up_positions = {}
    ai_elite_market_up_cash = initial_capital_needed
    current_ai_elite_market_up_stocks = []
    prev_ai_elite_market_up_stocks = []  # Track for rebalancing indicator
    ai_elite_market_up_rebalanced_today = False
    ai_elite_market_up_transaction_costs = 0.0
    ai_elite_market_up_initialized = False

    # AI CHAMPION: Initialize portfolio tracking (ML selector across top candidates)
    ai_champion_portfolio_value = initial_capital_needed
    ai_champion_portfolio_history = [ai_champion_portfolio_value] if config.ENABLE_AI_CHAMPION else []
    ai_champion_positions = {}
    ai_champion_cash = initial_capital_needed
    current_ai_champion_stocks = []
    ai_champion_transaction_costs = 0.0
    ai_champion_initialized = False
    ai_champion_allocator = None

    # AI REGIME: Initialize portfolio tracking (ML predicts which strategy to use)
    ai_regime_portfolio_value = initial_capital_needed
    ai_regime_portfolio_history = [ai_regime_portfolio_value]
    ai_regime_positions = {}
    ai_regime_cash = initial_capital_needed
    current_ai_regime_stocks = []
    ai_regime_transaction_costs = 0.0
    ai_regime_initialized = False
    ai_regime_allocator = None

    # AI REGIME MONTHLY: Initialize portfolio tracking (same as AI Regime but monthly rebalance)
    ai_regime_monthly_portfolio_value = initial_capital_needed
    ai_regime_monthly_portfolio_history = [ai_regime_monthly_portfolio_value]
    ai_regime_monthly_positions = {}
    ai_regime_monthly_cash = initial_capital_needed
    current_ai_regime_monthly_stocks = []
    ai_regime_monthly_transaction_costs = 0.0
    ai_regime_monthly_initialized = False
    ai_regime_monthly_allocator = None

    # UNIVERSAL MODEL: Initialize portfolio tracking (single ML model for all tickers)
    universal_model_portfolio_value = initial_capital_needed
    universal_model_portfolio_history = [universal_model_portfolio_value]
    universal_model_positions = {}
    universal_model_cash = initial_capital_needed
    current_universal_model_stocks = []
    universal_model_transaction_costs = 0.0
    universal_model_initialized = False
    universal_model_strategy = None  # Will be initialized on first use

    # SAVGOL TREND: Initialize portfolio tracking (pooled ML on local polynomial trend features)
    savgol_trend_portfolio_value = initial_capital_needed
    savgol_trend_portfolio_history = [savgol_trend_portfolio_value] if config.ENABLE_SAVGOL_TREND else []
    savgol_trend_positions = {}
    savgol_trend_cash = initial_capital_needed
    current_savgol_trend_stocks = []
    savgol_trend_transaction_costs = 0.0
    savgol_trend_initialized = False
    savgol_trend_strategy = None

    # TOP5 CONSISTENCY BLEND: Weight current holdings by prior-day top-5 consistency.
    top5_consistency_blend_portfolio_value = initial_capital_needed
    top5_consistency_blend_portfolio_history = [top5_consistency_blend_portfolio_value] if config.ENABLE_TOP5_CONSISTENCY_BLEND else []
    top5_consistency_blend_positions = {}
    top5_consistency_blend_cash = initial_capital_needed
    current_top5_consistency_blend_stocks = []
    top5_consistency_blend_transaction_costs = 0.0
    top5_consistency_blend_initialized = False
    top5_consistency_blend_last_weights = {}

    # MEAN REVERSION: Initialize portfolio tracking
    mean_reversion_portfolio_value = initial_capital_needed
    mean_reversion_portfolio_history = [mean_reversion_portfolio_value]
    mean_reversion_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    mean_reversion_cash = initial_capital_needed  # Start with same capital as AI
    current_mean_reversion_stocks = []  # Current bottom N stocks held by mean reversion
    mean_reversion_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # QUALITY + MOMENTUM: Initialize portfolio tracking
    quality_momentum_portfolio_value = initial_capital_needed
    quality_momentum_portfolio_history = [quality_momentum_portfolio_value]
    quality_momentum_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    quality_momentum_cash = initial_capital_needed  # Start with same capital as AI
    current_quality_momentum_stocks = []  # Current top N stocks held by quality + momentum
    quality_momentum_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MOMENTUM + AI HYBRID: Initialize portfolio tracking
    momentum_ai_hybrid_portfolio_value = initial_capital_needed
    momentum_ai_hybrid_portfolio_history = [momentum_ai_hybrid_portfolio_value]
    momentum_ai_hybrid_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'entry_date': str, 'peak_price': float}
    momentum_ai_hybrid_cash = initial_capital_needed  # Start with same capital as AI
    current_momentum_ai_hybrid_stocks = []  # Current stocks held by momentum + AI hybrid
    last_momentum_ai_hybrid_rebalance_day = 0  # Track days since last rebalance

    # VOLATILITY-ADJUSTED MOMENTUM: Initialize portfolio tracking
    volatility_adj_mom_portfolio_value = initial_capital_needed
    volatility_adj_mom_portfolio_history = [volatility_adj_mom_portfolio_value]
    volatility_adj_mom_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    volatility_adj_mom_cash = initial_capital_needed  # Start with same capital as AI
    current_volatility_adj_mom_stocks = []  # Current top N stocks held by volatility-adjusted momentum
    volatility_adj_mom_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # INVERSE ETF HEDGE: Standalone strategy that only holds inverse ETFs when market is down
    inverse_etf_hedge_portfolio_value = initial_capital_needed
    inverse_etf_hedge_portfolio_history = [inverse_etf_hedge_portfolio_value]
    inverse_etf_hedge_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    inverse_etf_hedge_cash = initial_capital_needed

    # ANALYST RECOMMENDATION: Strategy based on analyst upgrades/downgrades
    analyst_rec_portfolio_value = initial_capital_needed
    analyst_rec_portfolio_history = [analyst_rec_portfolio_value]
    analyst_rec_positions = {}
    analyst_rec_cash = initial_capital_needed
    current_analyst_rec_stocks = []
    analyst_rec_transaction_costs = 0.0
    analyst_rec_initialized = False
    analyst_rec_last_rebalance_day = 0
    analyst_data_cache = {}  # Cache analyst data to avoid repeated API calls
    current_inverse_etf_hedge_stocks = []
    inverse_etf_hedge_transaction_costs = 0.0
    inverse_etf_hedge_initialized = False
    inverse_etf_hedge_days_in_hedge = 0  # Track days in hedge for minimum hold period

    # META-STRATEGY SELECTORS: Initialize with actual portfolio tracking
    # Meta-strategy flags now read dynamically from config
    from meta_strategy_selector import META_SUB_STRATEGIES, MetaStrategyManager, calculate_meta_portfolio_value, select_meta_strategy_stocks

    meta_strategy_manager = MetaStrategyManager(initial_capital_needed)

    # Meta Weighted Composite - actual portfolio
    meta_weighted_composite_value = initial_capital_needed
    meta_weighted_composite_history = [meta_weighted_composite_value]
    meta_weighted_composite_positions = {}
    meta_weighted_composite_cash = initial_capital_needed
    current_meta_weighted_composite_stocks = []
    meta_weighted_composite_transaction_costs = 0.0
    meta_weighted_composite_initialized = False
    meta_weighted_composite_selected_strategy = ''

    # Meta Tiered Selection - actual portfolio
    meta_tiered_selection_value = initial_capital_needed
    meta_tiered_selection_history = [meta_tiered_selection_value]
    meta_tiered_selection_positions = {}
    meta_tiered_selection_cash = initial_capital_needed
    current_meta_tiered_selection_stocks = []
    meta_tiered_selection_transaction_costs = 0.0
    meta_tiered_selection_initialized = False
    meta_tiered_selection_selected_strategy = ''

    # Meta Ensemble Allocation - actual portfolio
    meta_ensemble_alloc_value = initial_capital_needed
    meta_ensemble_alloc_history = [meta_ensemble_alloc_value]
    meta_ensemble_alloc_positions = {}
    meta_ensemble_alloc_cash = initial_capital_needed
    current_meta_ensemble_alloc_stocks = []
    meta_ensemble_alloc_transaction_costs = 0.0
    meta_ensemble_alloc_initialized = False
    meta_ensemble_alloc_selected_strategy = ''

    # Meta Regime Based - actual portfolio
    meta_regime_based_value = initial_capital_needed
    meta_regime_based_history = [meta_regime_based_value]
    meta_regime_based_positions = {}
    meta_regime_based_cash = initial_capital_needed
    current_meta_regime_based_stocks = []
    meta_regime_based_transaction_costs = 0.0
    meta_regime_based_initialized = False
    meta_regime_based_selected_strategy = ''

    # Meta Recency Weighted - actual portfolio
    meta_recency_weighted_value = initial_capital_needed
    meta_recency_weighted_history = [meta_recency_weighted_value]
    meta_recency_weighted_positions = {}
    meta_recency_weighted_cash = initial_capital_needed
    current_meta_recency_weighted_stocks = []
    meta_recency_weighted_transaction_costs = 0.0
    meta_recency_weighted_initialized = False
    meta_recency_weighted_selected_strategy = ''

    # Meta Efficiency Ratio - actual portfolio
    meta_efficiency_ratio_value = initial_capital_needed
    meta_efficiency_ratio_history = [meta_efficiency_ratio_value]
    meta_efficiency_ratio_positions = {}
    meta_efficiency_ratio_cash = initial_capital_needed
    current_meta_efficiency_ratio_stocks = []
    meta_efficiency_ratio_transaction_costs = 0.0
    meta_efficiency_ratio_initialized = False
    meta_efficiency_ratio_selected_strategy = ''

    # Meta Min Variance - actual portfolio
    meta_min_variance_value = initial_capital_needed
    meta_min_variance_history = [meta_min_variance_value]
    meta_min_variance_positions = {}
    meta_min_variance_cash = initial_capital_needed
    current_meta_min_variance_stocks = []
    meta_min_variance_transaction_costs = 0.0
    meta_min_variance_initialized = False
    meta_min_variance_selected_strategy = ''

    # Meta Bayesian - actual portfolio
    meta_bayesian_value = initial_capital_needed
    meta_bayesian_history = [meta_bayesian_value]
    meta_bayesian_positions = {}
    meta_bayesian_cash = initial_capital_needed
    current_meta_bayesian_stocks = []
    meta_bayesian_transaction_costs = 0.0
    meta_bayesian_initialized = False
    meta_bayesian_selected_strategy = ''

    # Meta Adaptive Convex - actual portfolio
    meta_adaptive_convex_value = initial_capital_needed
    meta_adaptive_convex_history = [meta_adaptive_convex_value]
    meta_adaptive_convex_positions = {}
    meta_adaptive_convex_cash = initial_capital_needed
    current_meta_adaptive_convex_stocks = []
    meta_adaptive_convex_transaction_costs = 0.0
    meta_adaptive_convex_initialized = False
    meta_adaptive_convex_selected_strategy = ''

    # Meta Consensus - actual portfolio
    meta_consensus_value = initial_capital_needed
    meta_consensus_history = [meta_consensus_value]
    meta_consensus_positions = {}
    meta_consensus_cash = initial_capital_needed
    current_meta_consensus_stocks = []
    meta_consensus_transaction_costs = 0.0
    meta_consensus_initialized = False
    meta_consensus_selected_strategy = ''

    # BOLLINGER BANDS STRATEGIES: Initialize with portfolio tracking
    # BB strategy flags now read dynamically from config

    # BB Mean Reversion
    bb_mean_reversion_value = initial_capital_needed
    bb_mean_reversion_history = [bb_mean_reversion_value]
    bb_mean_reversion_positions = {}
    bb_mean_reversion_cash = initial_capital_needed
    current_bb_mean_reversion_stocks = []
    bb_mean_reversion_transaction_costs = 0.0
    bb_mean_reversion_initialized = False

    # BB Breakout
    bb_breakout_value = initial_capital_needed
    bb_breakout_history = [bb_breakout_value]
    bb_breakout_positions = {}
    bb_breakout_cash = initial_capital_needed
    current_bb_breakout_stocks = []
    bb_breakout_transaction_costs = 0.0
    bb_breakout_initialized = False

    # BB Squeeze Breakout
    bb_squeeze_breakout_value = initial_capital_needed
    bb_squeeze_breakout_history = [bb_squeeze_breakout_value]
    bb_squeeze_breakout_positions = {}
    bb_squeeze_breakout_cash = initial_capital_needed
    current_bb_squeeze_breakout_stocks = []
    bb_squeeze_breakout_transaction_costs = 0.0
    bb_squeeze_breakout_initialized = False

    # BB RSI Combo
    bb_rsi_combo_value = initial_capital_needed
    bb_rsi_combo_history = [bb_rsi_combo_value]
    bb_rsi_combo_positions = {}
    bb_rsi_combo_cash = initial_capital_needed
    current_bb_rsi_combo_stocks = []
    bb_rsi_combo_transaction_costs = 0.0
    bb_rsi_combo_initialized = False

    # TREND BREAKOUT: Initialize with portfolio tracking
    # Trend breakout flag now read dynamically from config
    trend_breakout_value = initial_capital_needed
    trend_breakout_history = [trend_breakout_value]
    trend_breakout_positions = {}
    trend_breakout_cash = initial_capital_needed
    current_trend_breakout_stocks = []
    trend_breakout_transaction_costs = 0.0
    trend_breakout_initialized = False

    # Reset global transaction cost tracking variables for this backtest
    global static_bh_transaction_costs, static_bh_3m_transaction_costs, static_bh_6m_transaction_costs, static_bh_1m_transaction_costs, dynamic_bh_1y_transaction_costs, dynamic_bh_transaction_costs
    global dynamic_bh_3m_transaction_costs, dynamic_bh_6m_transaction_costs, dynamic_bh_1m_transaction_costs, risk_adj_mom_transaction_costs, mean_reversion_transaction_costs, quality_momentum_transaction_costs, momentum_ai_hybrid_transaction_costs, volatility_adj_mom_transaction_costs, dynamic_bh_1y_vol_filter_transaction_costs, dynamic_bh_1y_trailing_stop_transaction_costs, ratio_3m_1y_transaction_costs, ratio_1y_3m_transaction_costs, turnaround_transaction_costs, adaptive_ensemble_transaction_costs, volatility_ensemble_transaction_costs, correlation_ensemble_transaction_costs, dynamic_pool_transaction_costs, sentiment_ensemble_transaction_costs
    global sector_rotation_transaction_costs, momentum_volatility_hybrid_transaction_costs, momentum_volatility_hybrid_6m_transaction_costs, momentum_volatility_hybrid_1y_transaction_costs, momentum_volatility_hybrid_1y3m_transaction_costs, enhanced_volatility_transaction_costs, ai_volatility_ensemble_transaction_costs, multi_tf_ensemble_transaction_costs, multi_tf_intraday_ensemble_transaction_costs
    static_bh_transaction_costs = 0.0  # Static BH has no transaction costs (buy once, hold)
    static_bh_3m_transaction_costs = 0.0
    static_bh_6m_transaction_costs = 0.0
    static_bh_1m_transaction_costs = 0.0
    dynamic_bh_1y_transaction_costs = 0.0
    dynamic_bh_transaction_costs = 0.0
    dynamic_bh_6m_transaction_costs = 0.0
    dynamic_bh_3m_transaction_costs = 0.0
    dynamic_bh_1m_transaction_costs = 0.0
    risk_adj_mom_transaction_costs = 0.0
    mean_reversion_transaction_costs = 0.0
    quality_momentum_transaction_costs = 0.0
    momentum_ai_hybrid_transaction_costs = 0.0
    volatility_adj_mom_transaction_costs = 0.0
    dynamic_bh_1y_vol_filter_transaction_costs = 0.0
    dynamic_bh_1y_trailing_stop_transaction_costs = 0.0
    ratio_3m_1y_transaction_costs = 0.0
    ratio_1y_3m_transaction_costs = 0.0
    ratio_1m_3m_transaction_costs = 0.0
    momentum_volatility_hybrid_transaction_costs = 0.0
    momentum_volatility_hybrid_6m_transaction_costs = 0.0
    momentum_volatility_hybrid_1y_transaction_costs = 0.0
    momentum_volatility_hybrid_1y3m_transaction_costs = 0.0
    turnaround_transaction_costs = 0.0
    sector_rotation_transaction_costs = 0.0
    adaptive_ensemble_transaction_costs = 0.0
    volatility_ensemble_transaction_costs = 0.0
    enhanced_volatility_transaction_costs = 0.0
    ai_volatility_ensemble_transaction_costs = 0.0
    multi_tf_ensemble_transaction_costs = 0.0
    multi_tf_intraday_ensemble_transaction_costs = 0.0
    correlation_ensemble_transaction_costs = 0.0
    dynamic_pool_transaction_costs = 0.0
    sentiment_ensemble_transaction_costs = 0.0
    voting_ensemble_transaction_costs = 0.0
    mom_accel_transaction_costs = 0.0
    concentrated_3m_transaction_costs = 0.0
    dual_mom_transaction_costs = 0.0
    trend_atr_transaction_costs = 0.0
    elite_hybrid_transaction_costs = 0.0
    elite_risk_transaction_costs = 0.0
    risk_adj_mom_6m_transaction_costs = 0.0
    risk_adj_mom_6m_monthly_transaction_costs = 0.0
    risk_adj_mom_3m_transaction_costs = 0.0
    risk_adj_mom_3m_monthly_transaction_costs = 0.0
    risk_adj_mom_1m_transaction_costs = 0.0
    risk_adj_mom_1m_monthly_transaction_costs = 0.0
    ai_elite_transaction_costs = 0.0

    # Transaction cost tracking for strategies that don't initialize it elsewhere
    ratio_3m_1y_transaction_costs = 0.0
    ratio_1y_3m_transaction_costs = 0.0
    momentum_volatility_hybrid_transaction_costs = 0.0
    momentum_volatility_hybrid_6m_transaction_costs = 0.0
    momentum_volatility_hybrid_1y_transaction_costs = 0.0
    momentum_volatility_hybrid_1y3m_transaction_costs = 0.0
    price_acceleration_transaction_costs = 0.0
    turnaround_transaction_costs = 0.0
    ai_classification_transaction_costs = 0.0

    all_processed_tickers = []
    all_performance_metrics = []
    all_buy_hold_histories = {}

    # NEW: Track per-stock contributions
    # ✅ NEW: Track per-stock contributions
    stock_performance_tracking = {}  # ticker -> {'days_held': int, 'contribution': float, 'max_shares': float, 'entry_value': float, 'exit_value': float}

    # Get all trading days in the backtest period using market calendars
    try:
        import pandas_market_calendars as mcal

        # Get trading calendars for different markets
        nyse = mcal.get_calendar('NYSE')      # US stocks (NASDAQ, NYSE, etc.)
        xetra = mcal.get_calendar('XETR')     # German stocks (DAX)
        lse = mcal.get_calendar('LSE')        # UK stocks
        tsx = mcal.get_calendar('TSX')        # Canadian stocks

        # Get valid trading days for each market
        us_trading_days = nyse.valid_days(start_date=backtest_start_date, end_date=backtest_end_date)
        german_trading_days = xetra.valid_days(start_date=backtest_start_date, end_date=backtest_end_date)
        uk_trading_days = lse.valid_days(start_date=backtest_start_date, end_date=backtest_end_date)
        canadian_trading_days = tsx.valid_days(start_date=backtest_start_date, end_date=backtest_end_date)

        # Combine all trading days (union of all markets)
        all_trading_days = sorted(set(us_trading_days) | set(german_trading_days) |
                                 set(uk_trading_days) | set(canadian_trading_days))

        # Convert to list like before
        business_days = list(all_trading_days)

        print(f"   📅 Trading days (US: {len(us_trading_days)}, DE: {len(german_trading_days)}, UK: {len(uk_trading_days)}, CA: {len(canadian_trading_days)})")
        print(f"   📅 Combined trading days to process: {len(business_days)}")

    except ImportError:
        # Fallback to simple weekday filter if pandas_market_calendars not available
        print("   ⚠️ pandas_market_calendars not available, using simple weekday filter")
        date_range = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq='D')
        business_days = [d for d in date_range if d.weekday() < 5]  # Filter to weekdays
        print(f"   📅 Total trading days to process: {len(business_days)}")

    # ✅ OPTIMIZATION: Pre-group data by ticker ONCE (instead of filtering 5644 times per day!)
    print(f"   🔧 Pre-grouping data by ticker for fast lookups...", flush=True)
    ticker_data_grouped = {}
    grouped = all_tickers_data.groupby('ticker')
    available_tickers_in_data = set(all_tickers_data['ticker'].unique())
    missing_tickers = []

    for ticker in initial_top_tickers:
        try:
            ticker_df = grouped.get_group(ticker).copy()
            if 'date' in ticker_df.columns:
                ticker_df = ticker_df.set_index('date')
            ticker_df = ticker_df.drop('ticker', axis=1, errors='ignore')
            ticker_data_grouped[ticker] = ticker_df
        except KeyError:
            missing_tickers.append(ticker)

    # ✅ ADD SECTOR ETFs AND INVERSE ETFs (required for Sector Rotation and Inverse ETF Hedge strategies)
    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB', 'GDX', 'USO', 'TLT']
    inverse_etfs = config.INVERSE_ETFS if hasattr(config, 'INVERSE_ETFS') else []
    special_etfs = set(sector_etfs + inverse_etfs)
    added_etfs = 0
    missing_special_etfs = []
    for etf in special_etfs:
        if etf not in ticker_data_grouped:
            if etf in available_tickers_in_data:
                try:
                    ticker_df = grouped.get_group(etf).copy()
                    if 'date' in ticker_df.columns:
                        ticker_df = ticker_df.set_index('date')
                    ticker_df = ticker_df.drop('ticker', axis=1, errors='ignore')
                    ticker_data_grouped[etf] = ticker_df
                    added_etfs += 1
                except KeyError:
                    missing_special_etfs.append(etf)
            else:
                missing_special_etfs.append(etf)
    if added_etfs > 0:
        print(f"   ✅ Added {added_etfs} sector/inverse ETFs to data pool")
    if missing_special_etfs:
        print(f"   ⚠️ Missing ETFs not in downloaded data: {missing_special_etfs[:15]}{'...' if len(missing_special_etfs) > 15 else ''}")

    print(f"   ✅ Pre-grouped {len(ticker_data_grouped)} tickers", flush=True)
    if missing_tickers:
        print(f"   ⚠️ {len(missing_tickers)} tickers NOT found in data: {missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}")
        print(f"   🔍 DEBUG: initial_top_tickers sample: {initial_top_tickers[:5]}")
        print(f"   🔍 DEBUG: available_tickers_in_data sample: {list(available_tickers_in_data)[:5]}")

    # Build a fixed benchmark portfolio from the highest performers at the backtest start.
    benchmark_start_date = business_days[0] if business_days else backtest_start_date
    benchmark_top_start_stocks = select_top_performers(
        initial_top_tickers,
        ticker_data_grouped,
        benchmark_start_date,
        lookback_days=365,
        top_n=PORTFOLIO_SIZE,
    )
    if benchmark_top_start_stocks:
        (
            benchmark_top_start_positions,
            benchmark_top_start_cash,
            benchmark_top_start_stocks,
            benchmark_top_start_transaction_costs,
            _,
        ) = _smart_rebalance_portfolio(
            strategy_name=benchmark_top_start_name,
            current_stocks=[],
            new_stocks=benchmark_top_start_stocks,
            positions=benchmark_top_start_positions,
            cash=benchmark_top_start_cash,
            ticker_data_grouped=ticker_data_grouped,
            current_date=benchmark_start_date,
            transaction_cost=TRANSACTION_COST,
            portfolio_size=PORTFOLIO_SIZE,
            force_rebalance=True,
        )
        benchmark_top_start_portfolio_value = _mark_to_market_value(
            benchmark_top_start_positions,
            benchmark_top_start_cash,
            ticker_data_grouped,
            benchmark_start_date,
        )
        print(
            f"   📌 {benchmark_top_start_name}: Fixed benchmark holdings at start = "
            f"{benchmark_top_start_stocks}"
        )
    else:
        print(f"   ⚠️ {benchmark_top_start_name}: Could not build start-of-backtest benchmark holdings")

    day_count = 0
    retrain_count = 0
    rebalance_count = 0

    # Meta-Strategy removed - no longer used
    meta_allocator = None

    # ✅ NEW: Track consecutive failures for fail-fast
    consecutive_no_predictions = 0
    consecutive_training_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # Abort if 5 days in a row fail

    # ✅ NEW: Track daily predictions vs actuals
    daily_prediction_log = []

    for current_date in business_days:
        day_count += 1

        # AI ELITE STRATEGY (ML-powered scoring) - PER-TICKER MODELS
        if config.ENABLE_AI_ELITE:
            try:
                (
                    ai_elite_models,
                    ai_elite_last_train_days,
                    ai_elite_positions,
                    ai_elite_cash,
                    current_ai_elite_stocks,
                    ai_elite_transaction_costs,
                    ai_elite_last_rebalance_value,
                    rebalanced_flag,
                ) = _run_ai_elite_step(
                    initial_top_tickers=initial_top_tickers,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    day_count=day_count,
                    ai_elite_models=ai_elite_models,
                    ai_elite_last_train_days=ai_elite_last_train_days,
                    ai_elite_positions=ai_elite_positions,
                    ai_elite_cash=ai_elite_cash,
                    current_ai_elite_stocks=current_ai_elite_stocks,
                    ai_elite_transaction_costs=ai_elite_transaction_costs,
                    ai_elite_portfolio_value=ai_elite_portfolio_value,
                    ai_elite_last_rebalance_value=ai_elite_last_rebalance_value,
                )
                strategies_rebalanced_today['AI Elite'] = rebalanced_flag
            except Exception as e:
                print(f"   ⚠️ AI Elite error: {e}")


        # STATIC BH PORTFOLIOS: Initialize on day 1 and optional periodic rebalancing
        # Static BH 1Y, 3M, and 1M are always initialized, then rebalance every N days if configured
        if config.ENABLE_STATIC_BH:
            # Increment days since last rebalance
            static_bh_1y_days_since_rebalance += 1
            static_bh_6m_days_since_rebalance += 1
            static_bh_3m_days_since_rebalance += 1
            static_bh_1m_days_since_rebalance += 1

            # STATIC BH 1Y: Initialize on day 1, then rebalance every N days if configured
            # Always initialize on first day, then only rebalance if REBALANCE_DAYS > 0
            should_init_or_rebalance_1y = (
                (not static_bh_1y_initialized) or  # Always initialize on first day
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and static_bh_1y_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance_1y:
                new_static_bh_1y_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_static_bh_1y_stocks:
                    print(f"   📊 Static BH 1Y Day {day_count}: {new_static_bh_1y_stocks}")
                    if new_static_bh_1y_stocks != current_static_bh_1y_stocks:
                        if not static_bh_1y_initialized:
                            print(f"   🎯 Static BH 1Y: Initializing with top {len(new_static_bh_1y_stocks)} by 1Y performance: {new_static_bh_1y_stocks}")
                        else:
                            print(f"   🔄 Static BH 1Y: Smart rebalancing to top {len(new_static_bh_1y_stocks)} by 1Y performance: {new_static_bh_1y_stocks}")

                        # Use universal smart rebalancing function
                        static_bh_1y_positions, static_bh_1y_cash, current_static_bh_1y_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Static BH 1Y",
                            current_stocks=current_static_bh_1y_stocks,
                            new_stocks=new_static_bh_1y_stocks,
                            positions=static_bh_1y_positions,
                            cash=static_bh_1y_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not static_bh_1y_initialized  # Force initial allocation
                        )
                        strategies_rebalanced_today['Static BH 1Y'] = rebalanced_flag
                        static_bh_transaction_costs += rebalance_costs

                        static_bh_1y_initialized = True
                        static_bh_1y_days_since_rebalance = 0

        # === ADAPTIVE REBALANCING STRATEGIES (based on Static BH 1Y) ===
        from adaptive_rebalancing import (
            check_volatility_trigger, check_performance_trigger,
            check_momentum_trigger, check_atr_trigger, check_hybrid_trigger
        )

        # 1. Volatility-triggered rebalancing
        if config.ENABLE_STATIC_BH_1Y_VOLATILITY:
            should_rebalance_vol = (not static_bh_1y_vol_initialized) or check_volatility_trigger(
                static_bh_1y_vol_positions, ticker_data_grouped, current_date
            )
            if should_rebalance_vol:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y Vol Day {day_count}: {new_stocks}")
                    if not static_bh_1y_vol_initialized:
                        print(f"   🎯 Static BH 1Y Vol: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y Vol: Volatility trigger rebalance")
                    static_bh_1y_vol_positions, static_bh_1y_vol_cash, current_static_bh_1y_vol_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Vol", current_stocks=current_static_bh_1y_vol_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_vol_positions, cash=static_bh_1y_vol_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_vol_initialized
                    )
                    static_bh_1y_vol_transaction_costs += rc
                    static_bh_1y_vol_initialized = True

        # 2. Performance-triggered rebalancing
        if config.ENABLE_STATIC_BH_1Y_PERFORMANCE:
            should_rebalance_perf = (not static_bh_1y_perf_initialized) or check_performance_trigger(
                static_bh_1y_perf_positions, ticker_data_grouped, current_date
            )
            if should_rebalance_perf:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y Perf Day {day_count}: {new_stocks}")
                    if not static_bh_1y_perf_initialized:
                        print(f"   🎯 Static BH 1Y Perf: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y Perf: Performance trigger rebalance")
                    static_bh_1y_perf_positions, static_bh_1y_perf_cash, current_static_bh_1y_perf_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Perf", current_stocks=current_static_bh_1y_perf_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_perf_positions, cash=static_bh_1y_perf_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_perf_initialized
                    )
                    static_bh_1y_perf_transaction_costs += rc
                    static_bh_1y_perf_initialized = True

        # 3. Momentum-triggered rebalancing
        if config.ENABLE_STATIC_BH_1Y_MOMENTUM:
            should_rebalance_mom = (not static_bh_1y_mom_initialized) or check_momentum_trigger(
                static_bh_1y_mom_positions, ticker_data_grouped, current_date
            )
            if should_rebalance_mom:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y Mom Day {day_count}: {new_stocks}")
                    if not static_bh_1y_mom_initialized:
                        print(f"   🎯 Static BH 1Y Mom: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y Mom: Momentum trigger rebalance")
                    static_bh_1y_mom_positions, static_bh_1y_mom_cash, current_static_bh_1y_mom_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Mom", current_stocks=current_static_bh_1y_mom_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_mom_positions, cash=static_bh_1y_mom_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_mom_initialized
                    )
                    static_bh_1y_mom_transaction_costs += rc
                    static_bh_1y_mom_initialized = True

        # 4. ATR-triggered rebalancing
        if config.ENABLE_STATIC_BH_1Y_ATR:
            should_rebalance_atr = (not static_bh_1y_atr_initialized) or check_atr_trigger(
                static_bh_1y_atr_positions, ticker_data_grouped, current_date
            )
            if should_rebalance_atr:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y ATR Day {day_count}: {new_stocks}")
                    if not static_bh_1y_atr_initialized:
                        print(f"   🎯 Static BH 1Y ATR: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y ATR: ATR trigger rebalance")
                    static_bh_1y_atr_positions, static_bh_1y_atr_cash, current_static_bh_1y_atr_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y ATR", current_stocks=current_static_bh_1y_atr_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_atr_positions, cash=static_bh_1y_atr_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_atr_initialized
                    )
                    static_bh_1y_atr_transaction_costs += rc
                    static_bh_1y_atr_initialized = True

        # 5. Hybrid rebalancing (combines all triggers with safety net)
        if config.ENABLE_STATIC_BH_1Y_HYBRID:
            static_bh_1y_hybrid_days_since_rebalance += 1
            should_rebalance_hybrid = (not static_bh_1y_hybrid_initialized) or check_hybrid_trigger(
                static_bh_1y_hybrid_positions, ticker_data_grouped, current_date, static_bh_1y_hybrid_days_since_rebalance
            )
            if should_rebalance_hybrid:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y Hybrid Day {day_count}: {new_stocks}")
                    if not static_bh_1y_hybrid_initialized:
                        print(f"   🎯 Static BH 1Y Hybrid: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y Hybrid: Hybrid trigger rebalance (day {static_bh_1y_hybrid_days_since_rebalance})")
                    static_bh_1y_hybrid_positions, static_bh_1y_hybrid_cash, current_static_bh_1y_hybrid_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Hybrid", current_stocks=current_static_bh_1y_hybrid_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_hybrid_positions, cash=static_bh_1y_hybrid_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_hybrid_initialized
                    )
                    static_bh_1y_hybrid_transaction_costs += rc
                    static_bh_1y_hybrid_initialized = True
                    static_bh_1y_hybrid_days_since_rebalance = 0

        # 6. Volume Filter strategy
        if config.ENABLE_STATIC_BH_1Y_VOLUME_FILTER:
            from enhanced_static_bh_strategies import select_volume_filtered_bh_1y_stocks
            new_stocks = select_volume_filtered_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE, STATIC_BH_1Y_VOLUME_MIN_DAILY
            )
            if new_stocks and (not static_bh_1y_volume_initialized or set(new_stocks) != set(current_static_bh_1y_volume_stocks)):
                print(f"   📊 Static BH 1Y Volume Day {day_count}: {new_stocks}")
                if not static_bh_1y_volume_initialized:
                    print(f"   🎯 Static BH 1Y Volume: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Volume: Volume filter rebalance")
                static_bh_1y_volume_positions, static_bh_1y_volume_cash, current_static_bh_1y_volume_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Volume", current_stocks=current_static_bh_1y_volume_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_volume_positions, cash=static_bh_1y_volume_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_volume_initialized
                )
                static_bh_1y_volume_transaction_costs += rc
                static_bh_1y_volume_initialized = True

        # 7. Sector Rotation strategy
        if config.ENABLE_STATIC_BH_1Y_SECTOR_ROTATION:
            from enhanced_static_bh_strategies import select_sector_rotated_bh_1y_stocks
            new_stocks = select_sector_rotated_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE, STATIC_BH_1Y_SECTOR_MAX_PER_SECTOR
            )
            if new_stocks and (not static_bh_1y_sector_initialized or set(new_stocks) != set(current_static_bh_1y_sector_stocks)):
                print(f"   📊 Static BH 1Y Sector Day {day_count}: {new_stocks}")
                if not static_bh_1y_sector_initialized:
                    print(f"   🎯 Static BH 1Y Sector: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Sector: Sector rotation rebalance")
                static_bh_1y_sector_positions, static_bh_1y_sector_cash, current_static_bh_1y_sector_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Sector", current_stocks=current_static_bh_1y_sector_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_sector_positions, cash=static_bh_1y_sector_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_sector_initialized
                )
                static_bh_1y_sector_transaction_costs += rc
                static_bh_1y_sector_initialized = True

        # 8. Performance Threshold strategy
        if config.ENABLE_STATIC_BH_1Y_PERFORMANCE_THRESHOLD:
            from enhanced_static_bh_strategies import select_performance_threshold_bh_1y_stocks
            new_stocks, should_rebalance = select_performance_threshold_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                current_static_bh_1y_perf_threshold_stocks, PORTFOLIO_SIZE, STATIC_BH_1Y_PERFORMANCE_MIN_IMPROVEMENT
            )
            if should_rebalance and new_stocks:
                print(f"   📊 Static BH 1Y Perf Thresh Day {day_count}: {new_stocks}")
                if not static_bh_1y_perf_threshold_initialized:
                    print(f"   🎯 Static BH 1Y Perf Thresh: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Perf Thresh: Performance threshold rebalance")
                static_bh_1y_perf_threshold_positions, static_bh_1y_perf_threshold_cash, current_static_bh_1y_perf_threshold_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Perf Thresh", current_stocks=current_static_bh_1y_perf_threshold_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_perf_threshold_positions, cash=static_bh_1y_perf_threshold_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_perf_threshold_initialized
                )
                static_bh_1y_perf_threshold_transaction_costs += rc
                static_bh_1y_perf_threshold_initialized = True

        # 9. Market Regime strategy
        if config.ENABLE_STATIC_BH_1Y_MARKET_REGIME:
            from enhanced_static_bh_strategies import select_market_regime_bh_1y_stocks
            static_bh_1y_market_regime_days_since_rebalance += 1

            # Check if it's time to rebalance (or first initialization)
            if not static_bh_1y_market_regime_initialized or static_bh_1y_market_regime_days_since_rebalance >= static_bh_1y_market_regime_next_rebalance_days:
                new_stocks, next_rebalance_days = select_market_regime_bh_1y_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE,
                    STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS, STATIC_BH_1Y_MARKET_REGIME_HIGH_VOL_DAYS,
                    STATIC_BH_1Y_MARKET_REGIME_LOW_VOL_DAYS, STATIC_BH_1Y_MARKET_REGIME_VOL_THRESHOLD
                )
                if new_stocks and (not static_bh_1y_market_regime_initialized or set(new_stocks) != set(current_static_bh_1y_market_regime_stocks)):
                    print(f"   📊 Static BH 1Y Market Regime Day {day_count}: {new_stocks}")
                    if not static_bh_1y_market_regime_initialized:
                        print(f"   🎯 Static BH 1Y Market Regime: Initializing")
                    else:
                        print(f"   🔄 Static BH 1Y Market Regime: Market regime rebalance (next in {next_rebalance_days} days)")
                    static_bh_1y_market_regime_positions, static_bh_1y_market_regime_cash, current_static_bh_1y_market_regime_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Market Regime", current_stocks=current_static_bh_1y_market_regime_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_market_regime_positions, cash=static_bh_1y_market_regime_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                        transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1y_market_regime_initialized
                    )
                    static_bh_1y_market_regime_transaction_costs += rc
                    static_bh_1y_market_regime_initialized = True
                    static_bh_1y_market_regime_days_since_rebalance = 0
                    static_bh_1y_market_regime_next_rebalance_days = next_rebalance_days

        # 10. Momentum Persistence strategy
        if config.ENABLE_STATIC_BH_1Y_MOMENTUM_PERSIST:
            from enhanced_static_bh_strategies import select_momentum_persistence_bh_1y_stocks
            new_stocks, should_rebalance, static_bh_1y_mom_persist_top_stocks_history = select_momentum_persistence_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                static_bh_1y_mom_persist_top_stocks_history, current_static_bh_1y_mom_persist_stocks,
                PORTFOLIO_SIZE, STATIC_BH_1Y_MOMENTUM_PERSIST_DAYS, STATIC_BH_1Y_MOMENTUM_PERSIST_MIN_OVERLAP
            )
            # Force initialization on first run even if stability check fails
            if (not static_bh_1y_mom_persist_initialized or should_rebalance) and new_stocks:
                print(f"   📊 Static BH 1Y Mom Persist Day {day_count}: {new_stocks}")
                if not static_bh_1y_mom_persist_initialized:
                    print(f"   🎯 Static BH 1Y Mom Persist: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Mom Persist: Momentum persistence confirmed, rebalancing")
                static_bh_1y_mom_persist_positions, static_bh_1y_mom_persist_cash, current_static_bh_1y_mom_persist_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Mom Persist", current_stocks=current_static_bh_1y_mom_persist_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_mom_persist_positions, cash=static_bh_1y_mom_persist_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_mom_persist_initialized
                )
                static_bh_1y_mom_persist_transaction_costs += rc
                static_bh_1y_mom_persist_initialized = True

        # 11. Overlap-Based strategy
        if config.ENABLE_STATIC_BH_1Y_OVERLAP:
            from enhanced_static_bh_strategies import select_overlap_based_bh_1y_stocks
            new_stocks, should_rebalance = select_overlap_based_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                current_static_bh_1y_overlap_stocks, PORTFOLIO_SIZE, STATIC_BH_1Y_OVERLAP_THRESHOLD
            )
            # Force initialization on first run
            if (not static_bh_1y_overlap_initialized or should_rebalance) and new_stocks:
                print(f"   📊 Static BH 1Y Overlap Day {day_count}: {new_stocks}")
                if not static_bh_1y_overlap_initialized:
                    print(f"   🎯 Static BH 1Y Overlap: Initializing")
                else:
                    overlap = len(set(current_static_bh_1y_overlap_stocks) & set(new_stocks)) / len(current_static_bh_1y_overlap_stocks) if current_static_bh_1y_overlap_stocks else 0
                    print(f"   🔄 Static BH 1Y Overlap: Overlap dropped to {overlap:.0%}, rebalancing")
                static_bh_1y_overlap_positions, static_bh_1y_overlap_cash, current_static_bh_1y_overlap_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Overlap", current_stocks=current_static_bh_1y_overlap_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_overlap_positions, cash=static_bh_1y_overlap_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_overlap_initialized
                )
                static_bh_1y_overlap_transaction_costs += rc
                static_bh_1y_overlap_initialized = True

        # 12. Rank Drift strategy
        if config.ENABLE_STATIC_BH_1Y_RANK_DRIFT:
            from enhanced_static_bh_strategies import select_rank_drift_bh_1y_stocks
            new_stocks, should_rebalance, static_bh_1y_rank_drift_previous_rankings = select_rank_drift_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                static_bh_1y_rank_drift_previous_rankings, current_static_bh_1y_rank_drift_stocks,
                PORTFOLIO_SIZE, STATIC_BH_1Y_RANK_DRIFT_THRESHOLD
            )
            # Force initialization on first run
            if (not static_bh_1y_rank_drift_initialized or should_rebalance) and new_stocks:
                print(f"   📊 Static BH 1Y Rank Drift Day {day_count}: {new_stocks}")
                if not static_bh_1y_rank_drift_initialized:
                    print(f"   🎯 Static BH 1Y Rank Drift: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Rank Drift: Rank drift detected, rebalancing")
                static_bh_1y_rank_drift_positions, static_bh_1y_rank_drift_cash, current_static_bh_1y_rank_drift_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Rank Drift", current_stocks=current_static_bh_1y_rank_drift_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_rank_drift_positions, cash=static_bh_1y_rank_drift_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_rank_drift_initialized
                )
                static_bh_1y_rank_drift_transaction_costs += rc
                static_bh_1y_rank_drift_initialized = True

        # 13. Drawdown Trigger strategy
        if config.ENABLE_STATIC_BH_1Y_DRAWDOWN:
            from enhanced_static_bh_strategies import select_drawdown_trigger_bh_1y_stocks
            new_stocks, should_rebalance = select_drawdown_trigger_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                current_static_bh_1y_drawdown_stocks, static_bh_1y_drawdown_portfolio_value,
                static_bh_1y_drawdown_peak, PORTFOLIO_SIZE, STATIC_BH_1Y_DRAWDOWN_THRESHOLD
            )
            # Force initialization on first run
            if (not static_bh_1y_drawdown_initialized or should_rebalance) and new_stocks:
                print(f"   📊 Static BH 1Y Drawdown Day {day_count}: {new_stocks}")
                if not static_bh_1y_drawdown_initialized:
                    print(f"   🎯 Static BH 1Y Drawdown: Initializing")
                else:
                    drawdown = (static_bh_1y_drawdown_peak - static_bh_1y_drawdown_portfolio_value) / static_bh_1y_drawdown_peak if static_bh_1y_drawdown_peak > 0 else 0
                    print(f"   🔄 Static BH 1Y Drawdown: Drawdown {drawdown:.1%} exceeded threshold, rebalancing")
                static_bh_1y_drawdown_positions, static_bh_1y_drawdown_cash, current_static_bh_1y_drawdown_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Drawdown", current_stocks=current_static_bh_1y_drawdown_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_drawdown_positions, cash=static_bh_1y_drawdown_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_drawdown_initialized
                )
                static_bh_1y_drawdown_transaction_costs += rc
                static_bh_1y_drawdown_initialized = True

        # 14. Smart Monthly strategy
        if config.ENABLE_STATIC_BH_1Y_SMART_MONTHLY:
            from enhanced_static_bh_strategies import select_smart_monthly_bh_1y_stocks
            new_stocks, should_rebalance, static_bh_1y_smart_monthly_last_rebalance_month = select_smart_monthly_bh_1y_stocks(
                initial_top_tickers, ticker_data_grouped, current_date,
                current_static_bh_1y_smart_monthly_stocks, static_bh_1y_smart_monthly_portfolio_value,
                static_bh_1y_smart_monthly_peak, static_bh_1y_smart_monthly_last_rebalance_month,
                PORTFOLIO_SIZE, STATIC_BH_1Y_SMART_MONTHLY_DRAWDOWN
            )
            # Force initialization on first run
            if (not static_bh_1y_smart_monthly_initialized or should_rebalance) and new_stocks:
                print(f"   📊 Static BH 1Y Smart Monthly Day {day_count}: {new_stocks}")
                if not static_bh_1y_smart_monthly_initialized:
                    print(f"   🎯 Static BH 1Y Smart Monthly: Initializing")
                else:
                    print(f"   🔄 Static BH 1Y Smart Monthly: Monthly/conditional rebalance")
                static_bh_1y_smart_monthly_positions, static_bh_1y_smart_monthly_cash, current_static_bh_1y_smart_monthly_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 1Y Smart Monthly", current_stocks=current_static_bh_1y_smart_monthly_stocks,
                    new_stocks=new_stocks, positions=static_bh_1y_smart_monthly_positions, cash=static_bh_1y_smart_monthly_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_1y_smart_monthly_initialized
                )
                static_bh_1y_smart_monthly_transaction_costs += rc
                static_bh_1y_smart_monthly_initialized = True

        # =====================================================================
        # SMART REBALANCING STRATEGIES (6 new strategies)
        # =====================================================================

        # Get top performers for all smart rebalancing strategies (shared selection)
        smart_rebal_candidates = select_top_performers(
            initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE * 2
        )
        smart_rebal_top_n = smart_rebal_candidates[:PORTFOLIO_SIZE] if smart_rebal_candidates else []

        # 1. BH 1Y Mom Sell - Momentum-Based Sell
        if config.ENABLE_BH_1Y_MOM_SELL:
            from enhanced_static_bh_strategies import should_sell_momentum_based
            bh_1y_mom_sell_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_mom_sell_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_mom_sell_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_top_n:
                # Custom sell logic: only sell if momentum declining
                positions_to_sell = []
                for ticker in list(bh_1y_mom_sell_positions.keys()):
                    if ticker not in set(smart_rebal_candidates[:PORTFOLIO_BUFFER_SIZE]):
                        should_sell, reason = should_sell_momentum_based(
                            ticker, ticker_data_grouped, current_date,
                            BH_1Y_MOM_SELL_LOOKBACK_SHORT, BH_1Y_MOM_SELL_LOOKBACK_LONG
                        )
                        if should_sell:
                            positions_to_sell.append(ticker)

                # Build new stocks list: keep non-sold + add new from top
                kept_stocks = [t for t in current_bh_1y_mom_sell_stocks if t not in positions_to_sell]
                new_stocks_needed = PORTFOLIO_SIZE - len(kept_stocks)
                new_buys = [t for t in smart_rebal_top_n if t not in kept_stocks][:new_stocks_needed]
                final_stocks = kept_stocks + new_buys

                print(f"   📊 BH 1Y Mom Sell Day {day_count}: {final_stocks}")
                if not bh_1y_mom_sell_initialized:
                    print(f"   🎯 BH 1Y Mom Sell: Initializing")
                    final_stocks = smart_rebal_top_n  # Use standard selection for init

                bh_1y_mom_sell_positions, bh_1y_mom_sell_cash, current_bh_1y_mom_sell_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Mom Sell", current_stocks=current_bh_1y_mom_sell_stocks,
                    new_stocks=final_stocks, positions=bh_1y_mom_sell_positions, cash=bh_1y_mom_sell_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_mom_sell_initialized
                )
                bh_1y_mom_sell_transaction_costs += rc
                bh_1y_mom_sell_initialized = True
                bh_1y_mom_sell_days_since_rebalance = 0

        # 2. BH 1Y Rank Sell - Relative Strength Ranking
        if config.ENABLE_BH_1Y_RANK_SELL:
            from enhanced_static_bh_strategies import should_sell_rank_based
            bh_1y_rank_sell_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_rank_sell_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_rank_sell_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_top_n:
                # Update entry ranks for new positions
                for i, ticker in enumerate(smart_rebal_candidates):
                    if ticker not in bh_1y_rank_sell_entry_ranks:
                        bh_1y_rank_sell_entry_ranks[ticker] = i + 1

                # Custom sell logic: sell if rank dropped significantly
                positions_to_sell = []
                for ticker in list(bh_1y_rank_sell_positions.keys()):
                    try:
                        current_rank = smart_rebal_candidates.index(ticker) + 1
                    except ValueError:
                        current_rank = 999  # Not in list
                    entry_rank = bh_1y_rank_sell_entry_ranks.get(ticker, current_rank)

                    should_sell, reason = should_sell_rank_based(
                        ticker, current_rank, entry_rank,
                        BH_1Y_RANK_SELL_DROP_THRESHOLD, BH_1Y_RANK_SELL_MAX_RANK
                    )
                    if should_sell:
                        positions_to_sell.append(ticker)

                kept_stocks = [t for t in current_bh_1y_rank_sell_stocks if t not in positions_to_sell]
                new_stocks_needed = PORTFOLIO_SIZE - len(kept_stocks)
                new_buys = [t for t in smart_rebal_top_n if t not in kept_stocks][:new_stocks_needed]
                final_stocks = kept_stocks + new_buys

                print(f"   📊 BH 1Y Rank Sell Day {day_count}: {final_stocks}")
                if not bh_1y_rank_sell_initialized:
                    print(f"   🎯 BH 1Y Rank Sell: Initializing")
                    final_stocks = smart_rebal_top_n
                    for i, ticker in enumerate(final_stocks):
                        bh_1y_rank_sell_entry_ranks[ticker] = i + 1

                bh_1y_rank_sell_positions, bh_1y_rank_sell_cash, current_bh_1y_rank_sell_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Rank Sell", current_stocks=current_bh_1y_rank_sell_stocks,
                    new_stocks=final_stocks, positions=bh_1y_rank_sell_positions, cash=bh_1y_rank_sell_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_rank_sell_initialized
                )
                bh_1y_rank_sell_transaction_costs += rc
                bh_1y_rank_sell_initialized = True
                bh_1y_rank_sell_days_since_rebalance = 0

        # 3. BH 1Y Trailing Mom - Trailing Stop + Momentum
        if config.ENABLE_BH_1Y_TRAILING_MOM:
            from enhanced_static_bh_strategies import should_sell_trailing_momentum
            bh_1y_trailing_mom_days_since_rebalance += 1

            # Update peak prices daily
            for ticker in bh_1y_trailing_mom_positions:
                if ticker in ticker_data_grouped:
                    data = ticker_data_grouped[ticker].loc[:current_date]
                    if not data.empty:
                        current_price = data['Close'].dropna().iloc[-1]
                        if ticker not in bh_1y_trailing_mom_peaks:
                            bh_1y_trailing_mom_peaks[ticker] = current_price
                        elif current_price > bh_1y_trailing_mom_peaks[ticker]:
                            bh_1y_trailing_mom_peaks[ticker] = current_price

            should_init_or_rebalance = (
                (not bh_1y_trailing_mom_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_trailing_mom_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_top_n:
                positions_to_sell = []
                for ticker in list(bh_1y_trailing_mom_positions.keys()):
                    if ticker not in set(smart_rebal_candidates[:PORTFOLIO_BUFFER_SIZE]):
                        entry_price = bh_1y_trailing_mom_positions[ticker].get('entry_price', 0)
                        peak_price = bh_1y_trailing_mom_peaks.get(ticker, entry_price)

                        should_sell, reason = should_sell_trailing_momentum(
                            ticker, ticker_data_grouped, current_date,
                            entry_price, peak_price,
                            BH_1Y_TRAILING_MOM_SOFT_STOP, BH_1Y_TRAILING_MOM_HARD_STOP
                        )
                        if should_sell:
                            positions_to_sell.append(ticker)

                kept_stocks = [t for t in current_bh_1y_trailing_mom_stocks if t not in positions_to_sell]
                new_stocks_needed = PORTFOLIO_SIZE - len(kept_stocks)
                new_buys = [t for t in smart_rebal_top_n if t not in kept_stocks][:new_stocks_needed]
                final_stocks = kept_stocks + new_buys

                print(f"   📊 BH 1Y Trailing Mom Day {day_count}: {final_stocks}")
                if not bh_1y_trailing_mom_initialized:
                    print(f"   🎯 BH 1Y Trailing Mom: Initializing")
                    final_stocks = smart_rebal_top_n

                bh_1y_trailing_mom_positions, bh_1y_trailing_mom_cash, current_bh_1y_trailing_mom_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Trailing Mom", current_stocks=current_bh_1y_trailing_mom_stocks,
                    new_stocks=final_stocks, positions=bh_1y_trailing_mom_positions, cash=bh_1y_trailing_mom_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_trailing_mom_initialized
                )
                bh_1y_trailing_mom_transaction_costs += rc
                bh_1y_trailing_mom_initialized = True
                bh_1y_trailing_mom_days_since_rebalance = 0

                # Initialize peaks for new positions
                for ticker in current_bh_1y_trailing_mom_stocks:
                    if ticker not in bh_1y_trailing_mom_peaks and ticker in bh_1y_trailing_mom_positions:
                        bh_1y_trailing_mom_peaks[ticker] = bh_1y_trailing_mom_positions[ticker].get('entry_price', 0)

        # 4. BH 1Y Volume Confirm - Volume Confirmation
        if config.ENABLE_BH_1Y_VOLUME_CONFIRM:
            from enhanced_static_bh_strategies import should_sell_volume_confirmed
            bh_1y_volume_confirm_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_volume_confirm_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_volume_confirm_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_top_n:
                buffer_set = set(smart_rebal_candidates[:PORTFOLIO_BUFFER_SIZE])
                positions_to_sell = []
                for ticker in list(bh_1y_volume_confirm_positions.keys()):
                    in_buffer = ticker in buffer_set
                    should_sell, reason = should_sell_volume_confirmed(
                        ticker, ticker_data_grouped, current_date,
                        in_buffer, BH_1Y_VOLUME_CONFIRM_MULTIPLIER
                    )
                    if should_sell:
                        positions_to_sell.append(ticker)

                kept_stocks = [t for t in current_bh_1y_volume_confirm_stocks if t not in positions_to_sell]
                new_stocks_needed = PORTFOLIO_SIZE - len(kept_stocks)
                new_buys = [t for t in smart_rebal_top_n if t not in kept_stocks][:new_stocks_needed]
                final_stocks = kept_stocks + new_buys

                print(f"   📊 BH 1Y Volume Confirm Day {day_count}: {final_stocks}")
                if not bh_1y_volume_confirm_initialized:
                    print(f"   🎯 BH 1Y Volume Confirm: Initializing")
                    final_stocks = smart_rebal_top_n

                bh_1y_volume_confirm_positions, bh_1y_volume_confirm_cash, current_bh_1y_volume_confirm_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Volume Confirm", current_stocks=current_bh_1y_volume_confirm_stocks,
                    new_stocks=final_stocks, positions=bh_1y_volume_confirm_positions, cash=bh_1y_volume_confirm_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_volume_confirm_initialized
                )
                bh_1y_volume_confirm_transaction_costs += rc
                bh_1y_volume_confirm_initialized = True
                bh_1y_volume_confirm_days_since_rebalance = 0

        # 5. BH 1Y Sector Aware - Sector Rotation Awareness
        if config.ENABLE_BH_1Y_SECTOR_AWARE:
            from enhanced_static_bh_strategies import filter_sells_sector_aware
            bh_1y_sector_aware_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_sector_aware_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_sector_aware_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_top_n:
                # Simple sector mapping (can be expanded)
                ticker_to_sector = {}
                for ticker in smart_rebal_candidates:
                    # Use first letter as pseudo-sector for now
                    ticker_to_sector[ticker] = ticker[0].upper() if ticker else 'X'

                buffer_set = set(smart_rebal_candidates[:PORTFOLIO_BUFFER_SIZE])
                current_set = set(current_bh_1y_sector_aware_stocks)
                positions_to_sell_raw = current_set - buffer_set

                # Apply sector-aware filtering
                adjusted_sell, adjusted_keep = filter_sells_sector_aware(
                    positions_to_sell_raw, current_set & buffer_set,
                    ticker_to_sector, smart_rebal_candidates,
                    BH_1Y_SECTOR_AWARE_MIN_PER_SECTOR, BH_1Y_SECTOR_AWARE_MAX_RANK
                )

                kept_stocks = list(adjusted_keep)
                new_stocks_needed = PORTFOLIO_SIZE - len(kept_stocks)
                new_buys = [t for t in smart_rebal_top_n if t not in kept_stocks][:new_stocks_needed]
                final_stocks = kept_stocks + new_buys

                print(f"   📊 BH 1Y Sector Aware Day {day_count}: {final_stocks}")
                if not bh_1y_sector_aware_initialized:
                    print(f"   🎯 BH 1Y Sector Aware: Initializing")
                    final_stocks = smart_rebal_top_n

                bh_1y_sector_aware_positions, bh_1y_sector_aware_cash, current_bh_1y_sector_aware_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Sector Aware", current_stocks=current_bh_1y_sector_aware_stocks,
                    new_stocks=final_stocks, positions=bh_1y_sector_aware_positions, cash=bh_1y_sector_aware_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_sector_aware_initialized
                )
                bh_1y_sector_aware_transaction_costs += rc
                bh_1y_sector_aware_initialized = True
                bh_1y_sector_aware_days_since_rebalance = 0

        # 6. BH 1Y Accel Buy - Accelerating Momentum Buy
        if config.ENABLE_BH_1Y_ACCEL_BUY:
            from enhanced_static_bh_strategies import select_buys_with_acceleration
            bh_1y_accel_buy_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_accel_buy_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_accel_buy_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance and smart_rebal_candidates:
                # Use acceleration-weighted selection for buys
                final_stocks = select_buys_with_acceleration(
                    smart_rebal_candidates, ticker_data_grouped, current_date,
                    PORTFOLIO_SIZE, BH_1Y_ACCEL_BUY_WEIGHT
                )

                print(f"   📊 BH 1Y Accel Buy Day {day_count}: {final_stocks}")
                if not bh_1y_accel_buy_initialized:
                    print(f"   🎯 BH 1Y Accel Buy: Initializing")

                bh_1y_accel_buy_positions, bh_1y_accel_buy_cash, current_bh_1y_accel_buy_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="BH 1Y Accel Buy", current_stocks=current_bh_1y_accel_buy_stocks,
                    new_stocks=final_stocks, positions=bh_1y_accel_buy_positions, cash=bh_1y_accel_buy_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_accel_buy_initialized
                )
                bh_1y_accel_buy_transaction_costs += rc
                bh_1y_accel_buy_initialized = True
                bh_1y_accel_buy_days_since_rebalance = 0

        # Static BH 3M with Acceleration
        if config.ENABLE_STATIC_BH_3M_ACCEL:
            from enhanced_static_bh_strategies import select_static_bh_3m_accel
            static_bh_3m_accel_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not static_bh_3m_accel_initialized) or
                (STATIC_BH_3M_REBALANCE_DAYS > 0 and static_bh_3m_accel_days_since_rebalance >= STATIC_BH_3M_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                # Select stocks using 3M acceleration logic
                final_stocks = select_static_bh_3m_accel(
                    ticker_data_grouped, current_date,
                    PORTFOLIO_SIZE, STATIC_BH_3M_ACCEL_LOOKBACK_SHORT,
                    STATIC_BH_3M_ACCEL_LOOKBACK_LONG, STATIC_BH_3M_ACCEL_WEIGHT
                )

                print(f"   📊 Static BH 3M Accel Day {day_count}: {final_stocks}")
                if not static_bh_3m_accel_initialized:
                    print(f"   🎯 Static BH 3M Accel: Initializing")

                static_bh_3m_accel_positions, static_bh_3m_accel_cash, current_static_bh_3m_accel_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Static BH 3M Accel", current_stocks=current_static_bh_3m_accel_stocks,
                    new_stocks=final_stocks, positions=static_bh_3m_accel_positions, cash=static_bh_3m_accel_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not static_bh_3m_accel_initialized
                )
                static_bh_3m_accel_transaction_costs += rc
                static_bh_3m_accel_initialized = True
                static_bh_3m_accel_days_since_rebalance = 0

        # =====================================================================
        # 10 NEW REBALANCING STRATEGIES (Based on Static BH 1Y stock selection)
        # =====================================================================

        # Helper config dict for rebalancing functions
        _rebal_config = {
            'BH_1Y_VOL_ADJ_REBAL_LOW_VOL_THRESH': BH_1Y_VOL_ADJ_REBAL_LOW_VOL_THRESH,
            'BH_1Y_VOL_ADJ_REBAL_HIGH_VOL_THRESH': BH_1Y_VOL_ADJ_REBAL_HIGH_VOL_THRESH,
            'BH_1Y_VOL_ADJ_REBAL_MIN_DAYS': BH_1Y_VOL_ADJ_REBAL_MIN_DAYS,
            'BH_1Y_VOL_ADJ_REBAL_MAX_DAYS': BH_1Y_VOL_ADJ_REBAL_MAX_DAYS,
            'BH_1Y_CORR_FILTER_THRESH': BH_1Y_CORR_FILTER_THRESH,
            'BH_1Y_CORR_FILTER_LOOKBACK': BH_1Y_CORR_FILTER_LOOKBACK,
            'BH_1Y_REGIME_BULL_MA': BH_1Y_REGIME_BULL_MA,
            'BH_1Y_REGIME_BEAR_MA': BH_1Y_REGIME_BEAR_MA,
            'BH_1Y_REGIME_REBAL_BULL': BH_1Y_REGIME_REBAL_BULL,
            'BH_1Y_REGIME_REBAL_BEAR': BH_1Y_REGIME_REBAL_BEAR,
            'BH_1Y_REGIME_REBAL_SIDEWAYS': BH_1Y_REGIME_REBAL_SIDEWAYS,
            'BH_1Y_RISK_PARITY_VOL_LOOKBACK': BH_1Y_RISK_PARITY_VOL_LOOKBACK,
            'BH_1Y_RISK_PARITY_MIN_WEIGHT': BH_1Y_RISK_PARITY_MIN_WEIGHT,
            'BH_1Y_RISK_PARITY_MAX_WEIGHT': BH_1Y_RISK_PARITY_MAX_WEIGHT,
            'BH_1Y_DRIFT_THRESH_TARGET': BH_1Y_DRIFT_THRESH_TARGET,
            'BH_1Y_DRIFT_THRESH_TRIGGER': BH_1Y_DRIFT_THRESH_TRIGGER,
            'BH_1Y_MOM_QUALITY_MOM_WEIGHT': BH_1Y_MOM_QUALITY_MOM_WEIGHT,
            'BH_1Y_MOM_QUALITY_CONS_WEIGHT': BH_1Y_MOM_QUALITY_CONS_WEIGHT,
            'BH_1Y_MOM_QUALITY_MIN_QUALITY': BH_1Y_MOM_QUALITY_MIN_QUALITY,
            'BH_1Y_LIQUIDITY_AVG_VOL_DAYS': BH_1Y_LIQUIDITY_AVG_VOL_DAYS,
            'BH_1Y_LIQUIDITY_MIN_DOLLAR_VOL': BH_1Y_LIQUIDITY_MIN_DOLLAR_VOL,
            'BH_1Y_LIQUIDITY_MAX_WEIGHT': BH_1Y_LIQUIDITY_MAX_WEIGHT,
            'BH_1Y_EARNINGS_AVOID_DAYS_BEFORE': BH_1Y_EARNINGS_AVOID_DAYS_BEFORE,
            'BH_1Y_EARNINGS_AVOID_DAYS_AFTER': BH_1Y_EARNINGS_AVOID_DAYS_AFTER,
            'BH_1Y_MULTI_FACTOR_MOM_WEIGHT': BH_1Y_MULTI_FACTOR_MOM_WEIGHT,
            'BH_1Y_MULTI_FACTOR_VAL_WEIGHT': BH_1Y_MULTI_FACTOR_VAL_WEIGHT,
            'BH_1Y_MULTI_FACTOR_QUAL_WEIGHT': BH_1Y_MULTI_FACTOR_QUAL_WEIGHT,
            'BH_1Y_TIME_DECAY_EXIT_DAYS': BH_1Y_TIME_DECAY_EXIT_DAYS,
            'BH_1Y_TIME_DECAY_EXIT_PCT_DAILY': BH_1Y_TIME_DECAY_EXIT_PCT_DAILY,
            'PORTFOLIO_SIZE': PORTFOLIO_SIZE,
        }

        # 1. Volatility-Adjusted Rebalancing
        if config.ENABLE_BH_1Y_VOL_ADJ_REBAL:
            from enhanced_static_bh_strategies import get_vol_adjusted_rebalance_days
            bh_1y_vol_adj_rebal_days_since_rebalance += 1

            # Calculate adaptive rebalance days based on volatility
            vol_adjusted_days = get_vol_adjusted_rebalance_days(ticker_data_grouped, current_date, _rebal_config)

            should_init_or_rebalance = (
                (not bh_1y_vol_adj_rebal_initialized) or
                (vol_adjusted_days > 0 and bh_1y_vol_adj_rebal_days_since_rebalance >= vol_adjusted_days)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y VolAdj Day {day_count}: {new_stocks}")
                if not bh_1y_vol_adj_rebal_initialized:
                    print(f"   🎯 Rebal 1Y VolAdj: Initializing")

                bh_1y_vol_adj_rebal_positions, bh_1y_vol_adj_rebal_cash, current_bh_1y_vol_adj_rebal_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y VolAdj", current_stocks=current_bh_1y_vol_adj_rebal_stocks,
                    new_stocks=new_stocks, positions=bh_1y_vol_adj_rebal_positions, cash=bh_1y_vol_adj_rebal_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_vol_adj_rebal_initialized
                )
                bh_1y_vol_adj_rebal_transaction_costs += rc
                bh_1y_vol_adj_rebal_initialized = True
                bh_1y_vol_adj_rebal_days_since_rebalance = 0

        # 1b. Volatility-Adjusted Rebalancing with 1M/3M Ratio ranking
        if config.ENABLE_RATIO_1M3M_VOL_ADJ_REBAL:
            from enhanced_static_bh_strategies import get_vol_adjusted_rebalance_days
            from shared_strategies import select_1y_performers_ranked_by_1m3m_ratio
            ratio_1m3m_vol_adj_rebal_days_since_rebalance += 1

            # Calculate adaptive rebalance days based on volatility (same logic as original)
            vol_adjusted_days = get_vol_adjusted_rebalance_days(ticker_data_grouped, current_date, _rebal_config)

            should_init_or_rebalance = (
                (not ratio_1m3m_vol_adj_rebal_initialized) or
                (vol_adjusted_days > 0 and ratio_1m3m_vol_adj_rebal_days_since_rebalance >= vol_adjusted_days)
            )

            # Force initialization on first day
            if day_count == 1 and not ratio_1m3m_vol_adj_rebal_initialized:
                should_init_or_rebalance = True


            if should_init_or_rebalance:
                # Select top 1Y performers, then rank by 1M/3M ratio (acceleration)
                new_stocks = select_1y_performers_ranked_by_1m3m_ratio(initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1M3M VolAdj Day {day_count}: {new_stocks}")
                if not ratio_1m3m_vol_adj_rebal_initialized:
                    print(f"   🎯 Rebal 1M3M VolAdj: Initializing")

                ratio_1m3m_vol_adj_rebal_positions, ratio_1m3m_vol_adj_rebal_cash, current_ratio_1m3m_vol_adj_rebal_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1M3M VolAdj", current_stocks=current_ratio_1m3m_vol_adj_rebal_stocks,
                    new_stocks=new_stocks, positions=ratio_1m3m_vol_adj_rebal_positions, cash=ratio_1m3m_vol_adj_rebal_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not ratio_1m3m_vol_adj_rebal_initialized
                )
                ratio_1m3m_vol_adj_rebal_transaction_costs += rc
                ratio_1m3m_vol_adj_rebal_initialized = True
                ratio_1m3m_vol_adj_rebal_days_since_rebalance = 0

        # 2. Correlation-Based Filtering
        if config.ENABLE_BH_1Y_CORR_FILTER:
            from enhanced_static_bh_strategies import filter_by_correlation
            bh_1y_corr_filter_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_corr_filter_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_corr_filter_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + 5)
                # Filter by correlation
                new_stocks = filter_by_correlation(new_stocks, ticker_data_grouped, current_date, _rebal_config)

                print(f"   📊 Rebal 1Y CorrFilt Day {day_count}: {new_stocks}")
                if not bh_1y_corr_filter_initialized:
                    print(f"   🎯 Rebal 1Y CorrFilt: Initializing")

                bh_1y_corr_filter_positions, bh_1y_corr_filter_cash, current_bh_1y_corr_filter_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y CorrFilt", current_stocks=current_bh_1y_corr_filter_stocks,
                    new_stocks=new_stocks, positions=bh_1y_corr_filter_positions, cash=bh_1y_corr_filter_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_corr_filter_initialized
                )
                bh_1y_corr_filter_transaction_costs += rc
                bh_1y_corr_filter_initialized = True
                bh_1y_corr_filter_days_since_rebalance = 0

        # 3. Market Regime Detection
        if config.ENABLE_BH_1Y_REGIME_AWARE:
            from enhanced_static_bh_strategies import detect_market_regime
            bh_1y_regime_aware_days_since_rebalance += 1

            # Detect current regime
            new_regime = detect_market_regime(ticker_data_grouped, current_date, _rebal_config)
            regime_changed = new_regime != bh_1y_regime_aware_current_regime
            bh_1y_regime_aware_current_regime = new_regime

            # Get rebalance days based on regime
            if new_regime == 'bull':
                regime_rebal_days = BH_1Y_REGIME_REBAL_BULL
            elif new_regime == 'bear':
                regime_rebal_days = BH_1Y_REGIME_REBAL_BEAR
            else:
                regime_rebal_days = BH_1Y_REGIME_REBAL_SIDEWAYS

            should_init_or_rebalance = (
                (not bh_1y_regime_aware_initialized) or
                (regime_rebal_days > 0 and bh_1y_regime_aware_days_since_rebalance >= regime_rebal_days) or
                regime_changed
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y Regime Day {day_count}: {new_stocks}")
                if not bh_1y_regime_aware_initialized:
                    print(f"   🎯 Rebal 1Y Regime: Initializing ({new_regime})")
                elif regime_changed:
                    print(f"   🔄 Rebal 1Y Regime: Regime changed to {new_regime}, rebalancing")

                bh_1y_regime_aware_positions, bh_1y_regime_aware_cash, current_bh_1y_regime_aware_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y Regime", current_stocks=current_bh_1y_regime_aware_stocks,
                    new_stocks=new_stocks, positions=bh_1y_regime_aware_positions, cash=bh_1y_regime_aware_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_regime_aware_initialized
                )
                bh_1y_regime_aware_transaction_costs += rc
                bh_1y_regime_aware_initialized = True
                bh_1y_regime_aware_days_since_rebalance = 0

        # 4. Risk Parity Allocation
        if config.ENABLE_BH_1Y_RISK_PARITY:
            from enhanced_static_bh_strategies import get_risk_parity_weights
            bh_1y_risk_parity_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_risk_parity_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_risk_parity_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y RiskPar Day {day_count}: {new_stocks}")
                if not bh_1y_risk_parity_initialized:
                    print(f"   🎯 Rebal 1Y RiskPar: Initializing")

                # Get risk parity weights
                risk_weights = get_risk_parity_weights(new_stocks, ticker_data_grouped, current_date, _rebal_config)

                bh_1y_risk_parity_positions, bh_1y_risk_parity_cash, current_bh_1y_risk_parity_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y RiskPar", current_stocks=current_bh_1y_risk_parity_stocks,
                    new_stocks=new_stocks, positions=bh_1y_risk_parity_positions, cash=bh_1y_risk_parity_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_risk_parity_initialized,
                    custom_weights=risk_weights
                )
                bh_1y_risk_parity_transaction_costs += rc
                bh_1y_risk_parity_initialized = True
                bh_1y_risk_parity_days_since_rebalance = 0

        # 5. Adaptive Drift Threshold
        if config.ENABLE_BH_1Y_DRIFT_THRESH:
            from enhanced_static_bh_strategies import calculate_portfolio_drift
            bh_1y_drift_thresh_days_since_rebalance += 1

            # Calculate current drift
            current_drift = calculate_portfolio_drift(bh_1y_drift_thresh_positions, ticker_data_grouped, current_date, _rebal_config)
            drift_trigger = BH_1Y_DRIFT_THRESH_TRIGGER

            should_init_or_rebalance = (
                (not bh_1y_drift_thresh_initialized) or
                (bh_1y_drift_thresh_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS) or
                (current_drift > drift_trigger)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y Drift Day {day_count}: {new_stocks}")
                if not bh_1y_drift_thresh_initialized:
                    print(f"   🎯 Rebal 1Y Drift: Initializing")
                elif current_drift > drift_trigger:
                    print(f"   🔄 Rebal 1Y Drift: Drift {current_drift:.1%} > {drift_trigger:.1%}, rebalancing")

                bh_1y_drift_thresh_positions, bh_1y_drift_thresh_cash, current_bh_1y_drift_thresh_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y Drift", current_stocks=current_bh_1y_drift_thresh_stocks,
                    new_stocks=new_stocks, positions=bh_1y_drift_thresh_positions, cash=bh_1y_drift_thresh_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_drift_thresh_initialized
                )
                bh_1y_drift_thresh_transaction_costs += rc
                bh_1y_drift_thresh_initialized = True
                bh_1y_drift_thresh_days_since_rebalance = 0

        # 6. Momentum Quality Score
        if config.ENABLE_BH_1Y_MOM_QUALITY:
            from enhanced_static_bh_strategies import score_momentum_quality
            bh_1y_mom_quality_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_mom_quality_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_mom_quality_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                candidates = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + 10)

                # Score by momentum quality
                scored = []
                for t in candidates:
                    score = score_momentum_quality(t, ticker_data_grouped, current_date, _rebal_config)
                    if score >= BH_1Y_MOM_QUALITY_MIN_QUALITY:
                        scored.append((t, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                new_stocks = [t for t, s in scored[:PORTFOLIO_SIZE]]

                print(f"   📊 Rebal 1Y MomQual Day {day_count}: {new_stocks}")
                if not bh_1y_mom_quality_initialized:
                    print(f"   🎯 Rebal 1Y MomQual: Initializing")

                bh_1y_mom_quality_positions, bh_1y_mom_quality_cash, current_bh_1y_mom_quality_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y MomQual", current_stocks=current_bh_1y_mom_quality_stocks,
                    new_stocks=new_stocks, positions=bh_1y_mom_quality_positions, cash=bh_1y_mom_quality_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_mom_quality_initialized
                )
                bh_1y_mom_quality_transaction_costs += rc
                bh_1y_mom_quality_initialized = True
                bh_1y_mom_quality_days_since_rebalance = 0

        # 7. Liquidity-Based Sizing
        if config.ENABLE_BH_1Y_LIQUIDITY:
            from enhanced_static_bh_strategies import get_liquidity_weights
            bh_1y_liquidity_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_liquidity_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_liquidity_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y Liquid Day {day_count}: {new_stocks}")
                if not bh_1y_liquidity_initialized:
                    print(f"   🎯 Rebal 1Y Liquid: Initializing")

                # Get liquidity-based weights
                liq_weights = get_liquidity_weights(new_stocks, ticker_data_grouped, current_date, _rebal_config)

                bh_1y_liquidity_positions, bh_1y_liquidity_cash, current_bh_1y_liquidity_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y Liquid", current_stocks=current_bh_1y_liquidity_stocks,
                    new_stocks=new_stocks, positions=bh_1y_liquidity_positions, cash=bh_1y_liquidity_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_liquidity_initialized,
                    custom_weights=liq_weights
                )
                bh_1y_liquidity_transaction_costs += rc
                bh_1y_liquidity_initialized = True
                bh_1y_liquidity_days_since_rebalance = 0

        # 8. Earnings Avoidance
        if config.ENABLE_BH_1Y_EARNINGS_AVOID:
            from enhanced_static_bh_strategies import is_near_earnings
            bh_1y_earnings_avoid_days_since_rebalance += 1

            # Check if any held stocks are near earnings
            near_earnings = False
            for ticker in list(bh_1y_earnings_avoid_positions.keys()):
                if is_near_earnings(ticker, ticker_data_grouped, current_date, _rebal_config):
                    near_earnings = True
                    break

            should_init_or_rebalance = (
                (not bh_1y_earnings_avoid_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_earnings_avoid_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y EarnAvd Day {day_count}: {new_stocks}")
                if not bh_1y_earnings_avoid_initialized:
                    print(f"   🎯 Rebal 1Y EarnAvd: Initializing")

                bh_1y_earnings_avoid_positions, bh_1y_earnings_avoid_cash, current_bh_1y_earnings_avoid_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y EarnAvd", current_stocks=current_bh_1y_earnings_avoid_stocks,
                    new_stocks=new_stocks, positions=bh_1y_earnings_avoid_positions, cash=bh_1y_earnings_avoid_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_earnings_avoid_initialized
                )
                bh_1y_earnings_avoid_transaction_costs += rc
                bh_1y_earnings_avoid_initialized = True
                bh_1y_earnings_avoid_days_since_rebalance = 0

        # 9. Multi-Factor Composite
        if config.ENABLE_BH_1Y_MULTI_FACTOR:
            from enhanced_static_bh_strategies import get_multi_factor_score
            bh_1y_multi_factor_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_multi_factor_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_multi_factor_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                candidates = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + 10)

                # Score by multi-factor
                scored = []
                for t in candidates:
                    score, mom, qual = get_multi_factor_score(t, ticker_data_grouped, current_date, _rebal_config)
                    scored.append((t, score, mom, qual))
                scored.sort(key=lambda x: x[1], reverse=True)
                new_stocks = [t for t, s, m, q in scored[:PORTFOLIO_SIZE]]

                print(f"   📊 Rebal 1Y MultiFact Day {day_count}: {new_stocks}")
                if not bh_1y_multi_factor_initialized:
                    print(f"   🎯 Rebal 1Y MultiFact: Initializing")

                bh_1y_multi_factor_positions, bh_1y_multi_factor_cash, current_bh_1y_multi_factor_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y MultiFact", current_stocks=current_bh_1y_multi_factor_stocks,
                    new_stocks=new_stocks, positions=bh_1y_multi_factor_positions, cash=bh_1y_multi_factor_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_multi_factor_initialized
                )
                bh_1y_multi_factor_transaction_costs += rc
                bh_1y_multi_factor_initialized = True
                bh_1y_multi_factor_days_since_rebalance = 0

        # 10. Time-Decay Holdings
        if config.ENABLE_BH_1Y_TIME_DECAY:
            from enhanced_static_bh_strategies import get_time_decay_exit_pct
            bh_1y_time_decay_days_since_rebalance += 1

            should_init_or_rebalance = (
                (not bh_1y_time_decay_initialized) or
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and bh_1y_time_decay_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )

            if should_init_or_rebalance:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=252, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)

                print(f"   📊 Rebal 1Y TimeDec Day {day_count}: {new_stocks}")
                if not bh_1y_time_decay_initialized:
                    print(f"   🎯 Rebal 1Y TimeDec: Initializing")

                bh_1y_time_decay_positions, bh_1y_time_decay_cash, current_bh_1y_time_decay_stocks, rc, _ = _smart_rebalance_portfolio(
                    strategy_name="Rebal 1Y TimeDec", current_stocks=current_bh_1y_time_decay_stocks,
                    new_stocks=new_stocks, positions=bh_1y_time_decay_positions, cash=bh_1y_time_decay_cash,
                    ticker_data_grouped=ticker_data_grouped, current_date=current_date,
                    transaction_cost=TRANSACTION_COST, portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not bh_1y_time_decay_initialized
                )
                bh_1y_time_decay_transaction_costs += rc
                bh_1y_time_decay_initialized = True
                bh_1y_time_decay_days_since_rebalance = 0
                # Update entry dates
                for ticker in current_bh_1y_time_decay_stocks:
                    if ticker not in bh_1y_time_decay_entry_dates:
                        bh_1y_time_decay_entry_dates[ticker] = current_date

        # STATIC BH 3M: Initialize on day 1, then rebalance every N days if configured
        if config.ENABLE_STATIC_BH:
            should_init_or_rebalance_3m = (
                (not static_bh_3m_initialized) or  # Always initialize on first day
                (STATIC_BH_3M_REBALANCE_DAYS > 0 and static_bh_3m_days_since_rebalance >= STATIC_BH_3M_REBALANCE_DAYS)
            )

            if should_init_or_rebalance_3m:
                new_static_bh_3m_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=90, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_static_bh_3m_stocks:
                    print(f"   📊 Static BH 3M Day {day_count}: {new_static_bh_3m_stocks}")
                    if not static_bh_3m_initialized:
                        print(f"   🎯 Static BH 3M: Initializing with top {len(new_static_bh_3m_stocks)} by 3M performance: {new_static_bh_3m_stocks}")
                    else:
                        print(f"   🔄 Static BH 3M: Smart rebalancing to top {len(new_static_bh_3m_stocks)} by 3M performance: {new_static_bh_3m_stocks}")

                    # Use universal smart rebalancing function
                    static_bh_3m_positions, static_bh_3m_cash, current_static_bh_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 3M",
                        current_stocks=current_static_bh_3m_stocks,
                        new_stocks=new_static_bh_3m_stocks,
                        positions=static_bh_3m_positions,
                        cash=static_bh_3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_3m_initialized  # Force initial allocation
                    )
                    strategies_rebalanced_today['Static BH 3M'] = rebalanced_flag
                    static_bh_transaction_costs += rebalance_costs

                    static_bh_3m_initialized = True
                    static_bh_3m_days_since_rebalance = 0

        # STATIC BH 6M: Initialize on day 1, then rebalance every N days if configured
        if config.ENABLE_STATIC_BH_6M:
            should_init_or_rebalance_6m = (
                (not static_bh_6m_initialized) or  # Always initialize on first day
                (STATIC_BH_6M_REBALANCE_DAYS > 0 and static_bh_6m_days_since_rebalance >= STATIC_BH_6M_REBALANCE_DAYS)
            )

            if should_init_or_rebalance_6m:
                new_static_bh_6m_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=180, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_static_bh_6m_stocks:
                    print(f"   📊 Static BH 6M Day {day_count}: {new_static_bh_6m_stocks}")
                    if not static_bh_6m_initialized:
                        print(f"   🎯 Static BH 6M: Initializing with top {len(new_static_bh_6m_stocks)} by 6M performance: {new_static_bh_6m_stocks}")
                    else:
                        print(f"   🔄 Static BH 6M: Smart rebalancing to top {len(new_static_bh_6m_stocks)} by 6M performance: {new_static_bh_6m_stocks}")

                    # Use universal smart rebalancing function
                    static_bh_6m_positions, static_bh_6m_cash, current_static_bh_6m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 6M",
                        current_stocks=current_static_bh_6m_stocks,
                        new_stocks=new_static_bh_6m_stocks,
                        positions=static_bh_6m_positions,
                        cash=static_bh_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_6m_initialized  # Force initial allocation
                    )
                    strategies_rebalanced_today['Static BH 6M'] = rebalanced_flag
                    static_bh_transaction_costs += rebalance_costs

                    static_bh_6m_initialized = True
                    static_bh_6m_days_since_rebalance = 0

            # STATIC BH 1M: Initialize on day 1, then rebalance every N days if configured
            should_init_or_rebalance_1m = (
                (not static_bh_1m_initialized) or  # Always initialize on first day
                (STATIC_BH_1M_REBALANCE_DAYS > 0 and static_bh_1m_days_since_rebalance >= STATIC_BH_1M_REBALANCE_DAYS)
            )

            if should_init_or_rebalance_1m:
                new_static_bh_1m_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=30, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_static_bh_1m_stocks:
                    print(f"   📊 Static BH 1M Day {day_count}: {new_static_bh_1m_stocks}")
                    if not static_bh_1m_initialized:
                        print(f"   🎯 Static BH 1M: Initializing with top {len(new_static_bh_1m_stocks)} by 1M performance: {new_static_bh_1m_stocks}")
                    else:
                        print(f"   🔄 Static BH 1M: Smart rebalancing to top {len(new_static_bh_1m_stocks)} by 1M performance: {new_static_bh_1m_stocks}")

                    # Use universal smart rebalancing function
                    static_bh_1m_positions, static_bh_1m_cash, current_static_bh_1m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1M",
                        current_stocks=current_static_bh_1m_stocks,
                        new_stocks=new_static_bh_1m_stocks,
                        positions=static_bh_1m_positions,
                        cash=static_bh_1m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_1m_initialized  # Force initial allocation
                    )
                    strategies_rebalanced_today['Static BH 1M'] = rebalanced_flag
                    static_bh_1m_transaction_costs += rebalance_costs

                    static_bh_1m_initialized = True
                    static_bh_1m_days_since_rebalance = 0

        # STATIC BH MONTHLY REBALANCE VARIANTS: Rebalance on first trading day of each month
        is_first_trading_day_of_month = (
            (day_count == 1) or  # First day of backtest
            (day_count > 1 and current_date.month != business_days[day_count - 2].month)  # Month changed from previous trading day
        )

        # Static BH 1Y Monthly
        if config.ENABLE_STATIC_BH_1Y_MONTHLY:
            should_rebalance_1y_monthly = (not static_bh_1y_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_1y_monthly:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1Y Monthly Day {day_count}: {new_stocks}")
                    if not static_bh_1y_monthly_initialized:
                        print(f"   🎯 Static BH 1Y Monthly: Initializing with {new_stocks}")
                    else:
                        print(f"   🔄 Static BH 1Y Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                    static_bh_1y_monthly_positions, static_bh_1y_monthly_cash, current_static_bh_1y_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1Y Mth", current_stocks=current_static_bh_1y_monthly_stocks,
                        new_stocks=new_stocks, positions=static_bh_1y_monthly_positions, cash=static_bh_1y_monthly_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_1y_monthly_initialized)
                    strategies_rebalanced_today['BH 1Y Monthly'] = rebalanced_flag
                    static_bh_1y_monthly_transaction_costs += rc
                    static_bh_1y_monthly_initialized = True
                    static_bh_1y_monthly_last_month = current_date.month

        # Static BH 6M Monthly
        if config.ENABLE_STATIC_BH_6M_MONTHLY:
            should_rebalance_6m_monthly = (not static_bh_6m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_6m_monthly:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=180, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 6M Monthly Day {day_count}: {new_stocks}")
                    if not static_bh_6m_monthly_initialized:
                        print(f"   🎯 Static BH 6M Monthly: Initializing with {new_stocks}")
                    else:
                        print(f"   🔄 Static BH 6M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                    static_bh_6m_monthly_positions, static_bh_6m_monthly_cash, current_static_bh_6m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 6M Mth", current_stocks=current_static_bh_6m_monthly_stocks,
                        new_stocks=new_stocks, positions=static_bh_6m_monthly_positions, cash=static_bh_6m_monthly_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_6m_monthly_initialized)
                    strategies_rebalanced_today['BH 6M Monthly'] = rebalanced_flag
                    static_bh_6m_monthly_transaction_costs += rc
                    static_bh_6m_monthly_initialized = True
                    static_bh_6m_monthly_last_month = current_date.month

        # Static BH 3M Monthly
        if config.ENABLE_STATIC_BH_3M_MONTHLY:
            should_rebalance_3m_monthly = (not static_bh_3m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_3m_monthly:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=90, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 3M Monthly Day {day_count}: {new_stocks}")
                    if not static_bh_3m_monthly_initialized:
                        print(f"   🎯 Static BH 3M Monthly: Initializing with {new_stocks}")
                    else:
                        print(f"   🔄 Static BH 3M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                    static_bh_3m_monthly_positions, static_bh_3m_monthly_cash, current_static_bh_3m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 3M Mth", current_stocks=current_static_bh_3m_monthly_stocks,
                        new_stocks=new_stocks, positions=static_bh_3m_monthly_positions, cash=static_bh_3m_monthly_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_3m_monthly_initialized)
                    strategies_rebalanced_today['BH 3M Monthly'] = rebalanced_flag
                    static_bh_3m_monthly_transaction_costs += rc
                    static_bh_3m_monthly_initialized = True
                    static_bh_3m_monthly_last_month = current_date.month

        # Static BH 1M Monthly
        if config.ENABLE_STATIC_BH_1M_MONTHLY:
            should_rebalance_1m_monthly = (not static_bh_1m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_1m_monthly:
                new_stocks = select_top_performers(initial_top_tickers, ticker_data_grouped, current_date, lookback_days=30, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                if new_stocks:
                    print(f"   📊 Static BH 1M Monthly Day {day_count}: {new_stocks}")
                    if not static_bh_1m_monthly_initialized:
                        print(f"   🎯 Static BH 1M Monthly: Initializing with {new_stocks}")
                    else:
                        print(f"   🔄 Static BH 1M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                    static_bh_1m_monthly_positions, static_bh_1m_monthly_cash, current_static_bh_1m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Static BH 1M Mth", current_stocks=current_static_bh_1m_monthly_stocks,
                        new_stocks=new_stocks, positions=static_bh_1m_monthly_positions, cash=static_bh_1m_monthly_cash,
                        ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_1m_monthly_initialized)
                    strategies_rebalanced_today['BH 1M Monthly'] = rebalanced_flag
                    static_bh_1m_monthly_transaction_costs += rc
                    static_bh_1m_monthly_initialized = True
                    static_bh_1m_monthly_last_month = current_date.month

        # DYNAMIC BH 1Y PORTFOLIO: Rebalance to current top N performers DAILY
        if config.ENABLE_DYNAMIC_BH_1Y:
            print(f"\n🔄 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): Dynamic BH 1Y Rebalancing...")

            try:
                new_dynamic_bh_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                    apply_performance_filter=True, filter_label="Dynamic BH 1Y"
                )

                if new_dynamic_bh_stocks:
                    print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-year): {', '.join(new_dynamic_bh_stocks)}")

                    # Use universal smart rebalancing function
                    dynamic_bh_positions, dynamic_bh_cash, current_dynamic_bh_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Dynamic BH 1Y",
                        current_stocks=current_dynamic_bh_stocks,
                        new_stocks=new_dynamic_bh_stocks,
                        positions=dynamic_bh_positions,
                        cash=dynamic_bh_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_bh_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Dynamic BH 1Y'] = rebalanced_flag
                    dynamic_bh_transaction_costs += rebalance_costs
                    dynamic_bh_last_rebalance_value = _mark_to_market_value(
                        dynamic_bh_positions, dynamic_bh_cash, ticker_data_grouped, current_date
                    )

            except Exception as e:
                print(f"   ⚠️ Dynamic BH 1Y error: {e}")

        # DYNAMIC BH 6M PORTFOLIO: Rebalance to current top N performers DAILY
        if config.ENABLE_DYNAMIC_BH_6M:
            print(f"\n🔄 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): Dynamic BH 6M Rebalancing...")

            try:
                new_dynamic_bh_6m_stocks = select_top_performers(
                    initial_top_tickers, ticker_data_grouped, current_date, lookback_days=180, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                    apply_performance_filter=True, filter_label="Dynamic BH 6M"
                )

                if new_dynamic_bh_6m_stocks:
                    print(f"   🏆 Top {PORTFOLIO_SIZE} performers (6-month): {', '.join(new_dynamic_bh_6m_stocks)}")

                    # Use universal smart rebalancing function
                    dynamic_bh_6m_positions, dynamic_bh_6m_cash, current_dynamic_bh_6m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Dynamic BH 6M",
                        current_stocks=current_dynamic_bh_6m_stocks,
                        new_stocks=new_dynamic_bh_6m_stocks,
                        positions=dynamic_bh_6m_positions,
                        cash=dynamic_bh_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_bh_6m_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Dynamic BH 6M'] = rebalanced_flag
                    dynamic_bh_6m_transaction_costs += rebalance_costs
                    dynamic_bh_6m_last_rebalance_value = _mark_to_market_value(
                        dynamic_bh_6m_positions, dynamic_bh_6m_cash, ticker_data_grouped, current_date
                    )

            except Exception as e:
                print(f"   ⚠️ Dynamic BH 6M error: {e}")

        # DYNAMIC BH 3M PORTFOLIO: Rebalance to current top N performers DAILY
        if config.ENABLE_DYNAMIC_BH_3M:
            new_dynamic_bh_3m_stocks = select_top_performers(
                initial_top_tickers, ticker_data_grouped, current_date, lookback_days=90, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                apply_performance_filter=True, filter_label="Dynamic BH 3M"
            )

            if new_dynamic_bh_3m_stocks:
                print(f"   📊 Dynamic BH 3M Day {day_count}: {new_dynamic_bh_3m_stocks}")

                # Use universal smart rebalancing function
                dynamic_bh_3m_positions, dynamic_bh_3m_cash, current_dynamic_bh_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 3M",
                    current_stocks=current_dynamic_bh_3m_stocks,
                    new_stocks=new_dynamic_bh_3m_stocks,
                    positions=dynamic_bh_3m_positions,
                    cash=dynamic_bh_3m_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_3m_stocks  # Force initial allocation
                )
                strategies_rebalanced_today['Dynamic BH 3M'] = rebalanced_flag
                dynamic_bh_3m_transaction_costs += rebalance_costs
                dynamic_bh_3m_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_3m_positions, dynamic_bh_3m_cash, ticker_data_grouped, current_date
                )

        # DYNAMIC BH 1M PORTFOLIO: Rebalance to current top N performers DAILY
        if config.ENABLE_DYNAMIC_BH_1M:
            new_dynamic_bh_1m_stocks = select_top_performers(
                initial_top_tickers, ticker_data_grouped, current_date, lookback_days=21, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                apply_performance_filter=True, filter_label="Dynamic BH 1M"
            )

            if new_dynamic_bh_1m_stocks:
                print(f"   📊 Dynamic BH 1M Day {day_count}: {new_dynamic_bh_1m_stocks}")
                print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-month): {', '.join(new_dynamic_bh_1m_stocks)}")

                # Use universal smart rebalancing function
                dynamic_bh_1m_positions, dynamic_bh_1m_cash, current_dynamic_bh_1m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1M",
                    current_stocks=current_dynamic_bh_1m_stocks,
                    new_stocks=new_dynamic_bh_1m_stocks,
                    positions=dynamic_bh_1m_positions,
                    cash=dynamic_bh_1m_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1m_stocks  # Force initial allocation
                )
                strategies_rebalanced_today['Dynamic BH 1M'] = rebalanced_flag
                dynamic_bh_1m_transaction_costs += rebalance_costs
                dynamic_bh_1m_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1m_positions, dynamic_bh_1m_cash, ticker_data_grouped, current_date
                )

        # DYNAMIC BH 1Y + VOLATILITY FILTER: Same as Dynamic BH 1Y but with volatility filter
        if config.ENABLE_DYNAMIC_BH_1Y_VOL_FILTER:
            new_dynamic_bh_1y_vol_filter_stocks, stocks_passed_vol_filter, stocks_evaluated = select_top_performers_vol_filtered(
                initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365,
                max_volatility=DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
            )

            if new_dynamic_bh_1y_vol_filter_stocks:
                print(f"   📊 Dynamic BH 1Y+Vol Day {day_count}: {new_dynamic_bh_1y_vol_filter_stocks}")
                print(f"   🎯 Top {PORTFOLIO_SIZE} performers (1Y + Vol Filter): {', '.join(new_dynamic_bh_1y_vol_filter_stocks)}")
                print(f"   🔍 Filter stats: {stocks_passed_vol_filter}/{stocks_evaluated} stocks passed volatility filter")

                # Use universal smart rebalancing function
                dynamic_bh_1y_vol_filter_positions, dynamic_bh_1y_vol_filter_cash, current_dynamic_bh_1y_vol_filter_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1Y+Vol",
                    current_stocks=current_dynamic_bh_1y_vol_filter_stocks,
                    new_stocks=new_dynamic_bh_1y_vol_filter_stocks,
                    positions=dynamic_bh_1y_vol_filter_positions,
                    cash=dynamic_bh_1y_vol_filter_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1y_vol_filter_stocks  # Force initial allocation
                )
                strategies_rebalanced_today['Dynamic BH 1Y+Vol'] = rebalanced_flag
                dynamic_bh_1y_vol_filter_transaction_costs += rebalance_costs
                dynamic_bh_1y_vol_filter_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1y_vol_filter_positions, dynamic_bh_1y_vol_filter_cash, ticker_data_grouped, current_date
                )
            else:
                print(f"   ⚠️ No stocks passed volatility filter (max {DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY:.1f}% annualized)")
                print(f"   🔍 Filter stats: {stocks_passed_vol_filter}/{stocks_evaluated} stocks passed volatility filter")
                # Debug: show what volatilities we saw
                if current_top_performers_vol_filter:
                    vol_debug = [(ticker, f"{vol:.1f}%") for ticker, perf, vol in current_top_performers_vol_filter[:5]]
                    print(f"   🔍 Sample volatilities: {', '.join([f'{t}({v})' for t, v in vol_debug])}")
                else:
                    print(f"   🔍 No stocks had valid volatility calculations")

        # DYNAMIC BH 1Y + TRAILING STOP: Same as Dynamic BH 1Y but with 20% trailing stop
        if config.ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP:
            # First, check trailing stops on existing positions
            positions_to_sell = []
            for ticker in list(current_dynamic_bh_1y_trailing_stop_stocks):
                if ticker in dynamic_bh_1y_trailing_stop_positions:
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is None or ticker_data.empty:
                            continue
                        ticker_data = ticker_data.loc[ticker_data.index == current_date]
                        if len(ticker_data) > 0:
                            current_price = ticker_data['Close'].iloc[0]
                            position = dynamic_bh_1y_trailing_stop_positions[ticker]
                            peak_price = position.get('peak_price', position['entry_price'])

                            # Update peak price if current price is higher
                            if current_price > peak_price:
                                dynamic_bh_1y_trailing_stop_positions[ticker]['peak_price'] = current_price
                                peak_price = current_price

                            # Check if trailing stop triggered (20% drop from peak)
                            stop_price = peak_price * (1 - DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT / 100)
                            if current_price <= stop_price:
                                positions_to_sell.append(ticker)
                                print(f"   🛑 Trailing stop triggered for {ticker}: ${current_price:.2f} <= ${stop_price:.2f} (peak: ${peak_price:.2f})")
                    except Exception as e:
                        continue

            # Sell positions that hit trailing stop
            for ticker in positions_to_sell:
                if ticker in dynamic_bh_1y_trailing_stop_positions:
                    position = dynamic_bh_1y_trailing_stop_positions[ticker]
                    shares = position['shares']
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is None or ticker_data.empty:
                            continue
                        ticker_data = ticker_data.loc[ticker_data.index == current_date]
                        if len(ticker_data) > 0:
                            current_price = ticker_data['Close'].iloc[0]
                            sell_value = shares * current_price
                            sell_cost = sell_value * TRANSACTION_COST
                            dynamic_bh_1y_trailing_stop_transaction_costs += sell_cost
                            dynamic_bh_1y_trailing_stop_cash += (sell_value - sell_cost)
                            del dynamic_bh_1y_trailing_stop_positions[ticker]
                            current_dynamic_bh_1y_trailing_stop_stocks.remove(ticker)
                    except Exception as e:
                        continue

            # Regular rebalancing (same as Dynamic BH 1Y)
            new_dynamic_bh_1y_trailing_stop_stocks = select_top_performers(
                initial_top_tickers, ticker_data_grouped, current_date, lookback_days=365, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                apply_performance_filter=True, filter_label="Dynamic BH 1Y+TS"
            )

            if new_dynamic_bh_1y_trailing_stop_stocks:
                print(f"   📊 Dynamic BH 1Y+TS Day {day_count}: {new_dynamic_bh_1y_trailing_stop_stocks}")
                print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-year + Trailing Stop): {', '.join(new_dynamic_bh_1y_trailing_stop_stocks)}")

                # Use universal smart rebalancing function
                dynamic_bh_1y_trailing_stop_positions, dynamic_bh_1y_trailing_stop_cash, current_dynamic_bh_1y_trailing_stop_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1Y+TS",
                    current_stocks=current_dynamic_bh_1y_trailing_stop_stocks,
                    new_stocks=new_dynamic_bh_1y_trailing_stop_stocks,
                    positions=dynamic_bh_1y_trailing_stop_positions,
                    cash=dynamic_bh_1y_trailing_stop_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1y_trailing_stop_stocks  # Force initial allocation
                )
                strategies_rebalanced_today['Dynamic BH 1Y+TS'] = rebalanced_flag
                dynamic_bh_1y_trailing_stop_transaction_costs += rebalance_costs
                dynamic_bh_1y_trailing_stop_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1y_trailing_stop_positions, dynamic_bh_1y_trailing_stop_cash, ticker_data_grouped, current_date
                )

        # SECTOR ROTATION: Rebalance to top performing sector ETFs
        if config.ENABLE_SECTOR_ROTATION:
            # Check if it's time to rebalance (or if this is initial allocation)
            days_since_rebalance = (current_date - sector_rotation_last_rebalance_date).days if 'sector_rotation_last_rebalance_date' in locals() else AI_REBALANCE_FREQUENCY_DAYS
            is_initial_allocation = not current_sector_rotation_etfs  # Force day 1 investment

            if is_initial_allocation or days_since_rebalance >= AI_REBALANCE_FREQUENCY_DAYS:
                print(f"   🏢 Sector Rotation rebalancing (every {AI_REBALANCE_FREQUENCY_DAYS} days)...")

                # Select top sector ETFs based on momentum
                from shared_strategies import select_sector_rotation_etfs
                new_sector_rotation_etfs = select_sector_rotation_etfs(
                    list(available_tickers_in_data), ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )

                if new_sector_rotation_etfs:
                    print(f"   📊 Sector Rotation Day {day_count}: {new_sector_rotation_etfs}")
                    # Use universal smart rebalancing function
                    sector_rotation_positions, sector_rotation_cash, current_sector_rotation_etfs, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Sector Rotation",
                        current_stocks=current_sector_rotation_etfs,
                        new_stocks=new_sector_rotation_etfs,
                        positions=sector_rotation_positions,
                        cash=sector_rotation_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_sector_rotation_etfs  # Force initial allocation
                    )
                    strategies_rebalanced_today['Sector Rotation'] = rebalanced_flag
                    sector_rotation_transaction_costs += rebalance_costs
                    sector_rotation_last_rebalance_date = current_date
                    sector_rotation_last_rebalance_value = _mark_to_market_value(
                        sector_rotation_positions, sector_rotation_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No sector ETFs selected for rotation")
            else:
                print(f"   ⏭️ Skip Sector Rotation rebalance: {days_since_rebalance} days since last (rebalance every {AI_REBALANCE_FREQUENCY_DAYS} days)")

        # RISK-ADJUSTED MOMENTUM: Rebalance to current top N using shared strategy
        if config.ENABLE_RISK_ADJ_MOM:
            from shared_strategies import select_risk_adj_mom_stocks

            # Use shared strategy for consistent selection
            new_risk_adj_mom_stocks = select_risk_adj_mom_stocks(
                initial_top_tickers,
                ticker_data_grouped,
                current_date,
                top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
            )

            if new_risk_adj_mom_stocks:
                print(f"   📊 Risk-Adj Mom Day {day_count}: {new_risk_adj_mom_stocks}")
                # Use universal smart rebalancing function
                risk_adj_mom_positions, risk_adj_mom_cash, current_risk_adj_mom_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                    strategy_name="Risk-Adj Mom",
                    current_stocks=current_risk_adj_mom_stocks,
                    new_stocks=new_risk_adj_mom_stocks,
                    positions=risk_adj_mom_positions,
                    cash=risk_adj_mom_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_risk_adj_mom_stocks  # Force initial allocation
                )
                strategies_rebalanced_today['Risk-Adj Mom'] = rebalanced_flag
                risk_adj_mom_transaction_costs += rebalance_costs
                risk_adj_mom_last_rebalance_value = _mark_to_market_value(
                    risk_adj_mom_positions, risk_adj_mom_cash, ticker_data_grouped, current_date
                )

        # 3M/1Y RATIO: Rebalance to highest 3M/1Y ratio stocks DAILY
        if config.ENABLE_3M_1Y_RATIO:
            try:
                from shared_strategies import select_3m_1y_ratio_stocks

                print(f"   📊 3M/1Y Ratio Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use shared strategy for consistent selection
                new_ratio_3m_1y_stocks = select_3m_1y_ratio_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,  # Add the current date parameter!
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE  # Use configured portfolio size (default 10)
                )

                if new_ratio_3m_1y_stocks:
                    print(f"   📊 3M/1Y Ratio Day {day_count}: {new_ratio_3m_1y_stocks}")
                    # Use universal smart rebalancing function
                    ratio_3m_1y_positions, ratio_3m_1y_cash, current_ratio_3m_1y_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="3M/1Y Ratio",
                        current_stocks=current_ratio_3m_1y_stocks,
                        new_stocks=new_ratio_3m_1y_stocks,
                        positions=ratio_3m_1y_positions,
                        cash=ratio_3m_1y_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_ratio_3m_1y_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['3M/1Y Ratio'] = rebalanced_flag
                    ratio_3m_1y_transaction_costs += rebalance_costs
                    ratio_3m_1y_last_rebalance_value = _mark_to_market_value(
                        ratio_3m_1y_positions, ratio_3m_1y_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No 3M/1Y Ratio stocks selected")

            except Exception as e:
                print(f"   ⚠️ 3M/1Y Ratio strategy error: {e}")

        # 1M/3M RATIO: Rebalance to highest short-term acceleration stocks DAILY
        if config.ENABLE_1M_3M_RATIO:
            try:
                from shared_strategies import select_1m_3m_ratio_stocks

                print(f"   📊 1M/3M Ratio Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use shared strategy for consistent selection
                new_ratio_1m_3m_stocks = select_1m_3m_ratio_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_ratio_1m_3m_stocks:
                    print(f"   📊 1M/3M Ratio Day {day_count}: {new_ratio_1m_3m_stocks}")
                    # Use universal smart rebalancing function
                    ratio_1m_3m_positions, ratio_1m_3m_cash, current_ratio_1m_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="1M/3M Ratio",
                        current_stocks=current_ratio_1m_3m_stocks,
                        new_stocks=new_ratio_1m_3m_stocks,
                        positions=ratio_1m_3m_positions,
                        cash=ratio_1m_3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_ratio_1m_3m_stocks
                    )
                    strategies_rebalanced_today['1M/3M Ratio'] = rebalanced_flag
                    ratio_1m_3m_transaction_costs += rebalance_costs
                    ratio_1m_3m_last_rebalance_value = _mark_to_market_value(
                        ratio_1m_3m_positions, ratio_1m_3m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No 1M/3M Ratio stocks selected")

            except Exception as e:
                print(f"   ⚠️ 1M/3M Ratio strategy error: {e}")

        # 1Y/3M RATIO: Rebalance to highest 1Y/3M ratio stocks DAILY
        if config.ENABLE_3M_1Y_RATIO:
            try:
                from shared_strategies import select_1y_3m_ratio_stocks

                print(f"   📊 1Y/3M Ratio Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

            # Use shared strategy for consistent selection
                new_ratio_1y_3m_stocks = select_1y_3m_ratio_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,  # Add the current date parameter!
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE  # Use configured portfolio size (default 10)
                )

                if new_ratio_1y_3m_stocks:
                    print(f"   📊 1Y/3M Ratio Day {day_count}: {new_ratio_1y_3m_stocks}")
                    # Use universal smart rebalancing function
                    ratio_1y_3m_positions, ratio_1y_3m_cash, current_ratio_1y_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="1Y/3M Ratio",
                        current_stocks=current_ratio_1y_3m_stocks,
                        new_stocks=new_ratio_1y_3m_stocks,
                        positions=ratio_1y_3m_positions,
                        cash=ratio_1y_3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_ratio_1y_3m_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['3M/1Y Ratio'] = rebalanced_flag
                    ratio_1y_3m_transaction_costs += rebalance_costs
                    ratio_1y_3m_last_rebalance_value = _mark_to_market_value(
                        ratio_1y_3m_positions, ratio_1y_3m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No 1Y/3M Ratio stocks selected")

            except Exception as e:
                print(f"   ⚠️ 1Y/3M Ratio strategy error: {e}")

        # TURNAROUND: Rebalance to best turnaround stocks DAILY
        if config.ENABLE_TURNAROUND:
            try:
                from shared_strategies import select_turnaround_stocks

                print(f"   📊 Turnaround Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use shared strategy for consistent selection
                new_turnaround_stocks = select_turnaround_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,  # Add the current date parameter!
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE  # Use configured portfolio size (default 10)
                )

                if new_turnaround_stocks:
                    print(f"   📊 Turnaround Day {day_count}: {new_turnaround_stocks}")
                    # Use universal smart rebalancing function
                    turnaround_positions, turnaround_cash, current_turnaround_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Turnaround",
                        current_stocks=current_turnaround_stocks,
                        new_stocks=new_turnaround_stocks,
                        positions=turnaround_positions,
                        cash=turnaround_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=min(len(new_turnaround_stocks), 15),  # Allow up to 15 positions
                        force_rebalance=not current_turnaround_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Turnaround'] = rebalanced_flag
                    turnaround_transaction_costs += rebalance_costs
                    turnaround_last_rebalance_value = _mark_to_market_value(
                        turnaround_positions, turnaround_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Turnaround stocks selected")

            except Exception as e:
                print(f"   ⚠️ Turnaround strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID: Rebalance using hybrid strategy DAILY
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_stocks

                print(f"   🎯 Momentum-Volatility Hybrid Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use momentum-volatility hybrid for stock selection
                new_momentum_volatility_hybrid_stocks = select_momentum_volatility_hybrid_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE  # Use configured portfolio size (default 10)
                )

                if new_momentum_volatility_hybrid_stocks:
                    print(f"   📊 Mom-Vol Hybrid Day {day_count}: {new_momentum_volatility_hybrid_stocks}")
                    # Use universal smart rebalancing function
                    momentum_volatility_hybrid_positions, momentum_volatility_hybrid_cash, current_momentum_volatility_hybrid_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Momentum-Vol Hybrid",
                        current_stocks=current_momentum_volatility_hybrid_stocks,
                        new_stocks=new_momentum_volatility_hybrid_stocks,
                        positions=momentum_volatility_hybrid_positions,
                        cash=momentum_volatility_hybrid_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Mom-Vol Hybrid'] = rebalanced_flag
                    momentum_volatility_hybrid_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_positions, momentum_volatility_hybrid_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Momentum-Volatility Hybrid stocks selected")

            except Exception as e:
                print(f"   ⚠️ Momentum-Volatility Hybrid strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 6M: Rebalance using hybrid strategy (6M lookback) DAILY
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_6m_stocks

                print(f"   🎯 Mom-Vol Hybrid 6M Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_momentum_volatility_hybrid_6m_stocks = select_momentum_volatility_hybrid_6m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_momentum_volatility_hybrid_6m_stocks:
                    print(f"   📊 Mom-Vol Hybrid 6M Day {day_count}: {new_momentum_volatility_hybrid_6m_stocks}")
                    momentum_volatility_hybrid_6m_positions, momentum_volatility_hybrid_6m_cash, current_momentum_volatility_hybrid_6m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 6M",
                        current_stocks=current_momentum_volatility_hybrid_6m_stocks,
                        new_stocks=new_momentum_volatility_hybrid_6m_stocks,
                        positions=momentum_volatility_hybrid_6m_positions,
                        cash=momentum_volatility_hybrid_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_6m_stocks
                    )
                    strategies_rebalanced_today['Mom-Vol Hybrid 6M'] = rebalanced_flag
                    momentum_volatility_hybrid_6m_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_6m_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_6m_positions, momentum_volatility_hybrid_6m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 6M stocks selected")

            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 6M strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 1Y: Rebalance using hybrid strategy (1Y lookback) DAILY
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_1y_stocks

                print(f"   🎯 Mom-Vol Hybrid 1Y Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_momentum_volatility_hybrid_1y_stocks = select_momentum_volatility_hybrid_1y_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_momentum_volatility_hybrid_1y_stocks:
                    print(f"   📊 Mom-Vol Hybrid 1Y Day {day_count}: {new_momentum_volatility_hybrid_1y_stocks}")
                    momentum_volatility_hybrid_1y_positions, momentum_volatility_hybrid_1y_cash, current_momentum_volatility_hybrid_1y_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 1Y",
                        current_stocks=current_momentum_volatility_hybrid_1y_stocks,
                        new_stocks=new_momentum_volatility_hybrid_1y_stocks,
                        positions=momentum_volatility_hybrid_1y_positions,
                        cash=momentum_volatility_hybrid_1y_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_1y_stocks
                    )
                    strategies_rebalanced_today['Mom-Vol Hybrid 1Y'] = rebalanced_flag
                    momentum_volatility_hybrid_1y_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_1y_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_1y_positions, momentum_volatility_hybrid_1y_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 1Y stocks selected")

            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 1Y strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 1Y/3M: Rebalance using hybrid strategy (1Y/3M ratio) DAILY
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_1y3m_stocks

                print(f"   🎯 Mom-Vol Hybrid 1Y/3M Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_momentum_volatility_hybrid_1y3m_stocks = select_momentum_volatility_hybrid_1y3m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_momentum_volatility_hybrid_1y3m_stocks:
                    print(f"   📊 Mom-Vol Hybrid 1Y3M Day {day_count}: {new_momentum_volatility_hybrid_1y3m_stocks}")
                    momentum_volatility_hybrid_1y3m_positions, momentum_volatility_hybrid_1y3m_cash, current_momentum_volatility_hybrid_1y3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 1Y/3M",
                        current_stocks=current_momentum_volatility_hybrid_1y3m_stocks,
                        new_stocks=new_momentum_volatility_hybrid_1y3m_stocks,
                        positions=momentum_volatility_hybrid_1y3m_positions,
                        cash=momentum_volatility_hybrid_1y3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_1y3m_stocks
                    )
                    strategies_rebalanced_today['Mom-Vol Hybrid 1Y/3M'] = rebalanced_flag
                    momentum_volatility_hybrid_1y3m_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_1y3m_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_1y3m_positions, momentum_volatility_hybrid_1y3m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 1Y/3M stocks selected")

            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 1Y/3M strategy error: {e}")

        # PRICE ACCELERATION: Rebalance using velocity/acceleration strategy DAILY
        if config.ENABLE_PRICE_ACCELERATION:
            try:
                from shared_strategies import select_price_acceleration_stocks

                print(f"   🚀 Price Acceleration Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use price acceleration for stock selection
                new_price_acceleration_stocks = select_price_acceleration_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE  # Use configured portfolio size (default 10)
                )

                if new_price_acceleration_stocks:
                    print(f"   📊 Price Accel Day {day_count}: {new_price_acceleration_stocks}")
                    # Use universal smart rebalancing function
                    price_acceleration_positions, price_acceleration_cash, current_price_acceleration_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Price Acceleration",
                        current_stocks=current_price_acceleration_stocks,
                        new_stocks=new_price_acceleration_stocks,
                        positions=price_acceleration_positions,
                        cash=price_acceleration_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_price_acceleration_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Price Acceleration'] = rebalanced_flag
                    price_acceleration_transaction_costs += rebalance_costs
                    price_acceleration_last_rebalance_value = _mark_to_market_value(
                        price_acceleration_positions, price_acceleration_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Price Acceleration stocks selected")

            except Exception as e:
                print(f"   ⚠️ Price Acceleration strategy error: {e}")

        # ADAPTIVE ENSEMBLE: Rebalance using meta-ensemble strategy DAILY
        if config.ENABLE_ADAPTIVE_STRATEGY:
            try:
                from adaptive_ensemble import select_adaptive_ensemble_stocks, reset_ensemble_state

                print(f"   📊 Adaptive Ensemble Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Use adaptive ensemble for stock selection
                new_adaptive_ensemble_stocks = select_adaptive_ensemble_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_adaptive_ensemble_stocks:
                    print(f"   📊 Adaptive Ensemble Day {day_count}: {new_adaptive_ensemble_stocks}")
                    # Use universal smart rebalancing function
                    adaptive_ensemble_positions, adaptive_ensemble_cash, current_adaptive_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Adaptive Ensemble",
                        current_stocks=current_adaptive_ensemble_stocks,
                        new_stocks=new_adaptive_ensemble_stocks,
                        positions=adaptive_ensemble_positions,
                        cash=adaptive_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_adaptive_ensemble_stocks  # Force initial allocation on day 1
                    )
                    strategies_rebalanced_today['Adaptive Ensemble'] = rebalanced_flag
                    adaptive_ensemble_transaction_costs += rebalance_costs
                    adaptive_ensemble_last_rebalance_value = _mark_to_market_value(
                        adaptive_ensemble_positions, adaptive_ensemble_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Adaptive Ensemble stocks selected")

            except Exception as e:
                print(f"   ⚠️ Adaptive Ensemble strategy error: {e}")
                import traceback
                traceback.print_exc()

        # VOLATILITY ENSEMBLE: Rebalance using volatility-adjusted strategy DAILY
        if config.ENABLE_VOLATILITY_ENSEMBLE:
            try:
                from volatility_ensemble import select_volatility_ensemble_stocks, reset_vol_ensemble_state

                print(f"   📊 Volatility Ensemble Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_volatility_ensemble_stocks = select_volatility_ensemble_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_volatility_ensemble_stocks:
                    print(f"   📊 Vol Ensemble Day {day_count}: {new_volatility_ensemble_stocks}")
                    # Use universal smart rebalancing function
                    volatility_ensemble_positions, volatility_ensemble_cash, current_volatility_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Volatility Ensemble",
                        current_stocks=current_volatility_ensemble_stocks,
                        new_stocks=new_volatility_ensemble_stocks,
                        positions=volatility_ensemble_positions,
                        cash=volatility_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_volatility_ensemble_stocks  # Force initial allocation on day 1
                    )
                    strategies_rebalanced_today['Volatility Ensemble'] = rebalanced_flag
                    volatility_ensemble_transaction_costs += rebalance_costs
                    volatility_ensemble_last_rebalance_value = _mark_to_market_value(
                        volatility_ensemble_positions, volatility_ensemble_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Volatility Ensemble stocks selected")

            except Exception as e:
                print(f"   ⚠️ Volatility Ensemble strategy error: {e}")

        # CORRELATION ENSEMBLE: Select stocks with low correlation for diversification
        if config.ENABLE_CORRELATION_ENSEMBLE:
            try:
                print(f"   📊 Correlation Ensemble: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # Calculate correlation-filtered stocks (select stocks with low correlation to each other)
                correlation_scores = []

                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]

                        # Get 60-day returns for correlation calculation
                        lookback_start = current_date - timedelta(days=60)
                        data_slice = ticker_data.loc[lookback_start:current_date]

                        if len(data_slice) >= 30:
                            returns = data_slice['Close'].pct_change().dropna()
                            if len(returns) >= 20:
                                # Calculate momentum (positive returns preferred)
                                momentum = (data_slice['Close'].iloc[-1] / data_slice['Close'].iloc[0]) - 1
                                # Calculate volatility
                                volatility = returns.std() * np.sqrt(252)
                                # Score: momentum adjusted by volatility
                                # Only include stocks with positive momentum
                                if volatility > 0 and momentum > 0:
                                    score = momentum / volatility
                                    correlation_scores.append((ticker, score, returns))
                    except Exception:
                        continue

                if correlation_scores:
                    # Sort by score and select top candidates
                    correlation_scores.sort(key=lambda x: x[1], reverse=True)

                    # Select stocks with low correlation to each other
                    selected_stocks = []
                    for ticker, score, returns in correlation_scores:
                        if len(selected_stocks) >= PORTFOLIO_SIZE:
                            break

                        # Check correlation with already selected stocks
                        is_correlated = False
                        for sel_ticker, _, sel_returns in [s for s in correlation_scores if s[0] in selected_stocks]:
                            try:
                                corr = returns.corr(sel_returns)
                                if abs(corr) > 0.7:  # High correlation threshold
                                    is_correlated = True
                                    break
                            except Exception:
                                pass

                        if not is_correlated:
                            selected_stocks.append(ticker)

                    new_correlation_ensemble_stocks = selected_stocks

                    if new_correlation_ensemble_stocks:
                        print(f"   📊 Correlation Ensemble Day {day_count}: {new_correlation_ensemble_stocks}")
                        print(f"   🎯 Correlation Ensemble: Selected {len(new_correlation_ensemble_stocks)} low-correlation stocks")

                        # Use universal smart rebalancing function
                        correlation_ensemble_positions, correlation_ensemble_cash, current_correlation_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Correlation Ensemble",
                            current_stocks=current_correlation_ensemble_stocks,
                            new_stocks=new_correlation_ensemble_stocks,
                            positions=correlation_ensemble_positions,
                            cash=correlation_ensemble_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_correlation_ensemble_stocks  # Force initial allocation on day 1
                        )
                        strategies_rebalanced_today['Correlation Ensemble'] = rebalanced_flag
                        correlation_ensemble_transaction_costs += rebalance_costs
                        correlation_ensemble_last_rebalance_value = _mark_to_market_value(
                            correlation_ensemble_positions, correlation_ensemble_cash, ticker_data_grouped, current_date
                        )
                    else:
                        print(f"   ❌ No low-correlation stocks found")
                else:
                    print(f"   ❌ No correlation data available")

            except Exception as e:
                print(f"   ⚠️ Correlation Ensemble strategy error: {e}")

        # DYNAMIC POOL: Rebalance using dynamic strategy pool DAILY
        if config.ENABLE_DYNAMIC_POOL:
            try:
                from dynamic_pool import select_dynamic_pool_stocks

                print(f"   📊 Dynamic Pool Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_dynamic_pool_stocks = select_dynamic_pool_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_dynamic_pool_stocks:
                    print(f"   📊 Dynamic Pool Day {day_count}: {new_dynamic_pool_stocks}")
                    # Use universal smart rebalancing function
                    dynamic_pool_positions, dynamic_pool_cash, current_dynamic_pool_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Dynamic Pool",
                        current_stocks=current_dynamic_pool_stocks,
                        new_stocks=new_dynamic_pool_stocks,
                        positions=dynamic_pool_positions,
                        cash=dynamic_pool_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_pool_stocks  # Force initial allocation on day 1
                    )
                    strategies_rebalanced_today['Dynamic Pool'] = rebalanced_flag
                    dynamic_pool_transaction_costs += rebalance_costs
                    dynamic_pool_last_rebalance_value = _mark_to_market_value(
                        dynamic_pool_positions, dynamic_pool_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Dynamic Pool stocks selected")

            except Exception as e:
                print(f"   ⚠️ Dynamic Pool strategy error: {e}")

        # RISK-ADJ MOMENTUM SENTIMENT: Rebalance using risk-adjusted momentum + sentiment strategy DAILY
        if config.ENABLE_RISK_ADJ_MOM_SENTIMENT:
            try:
                from risk_adj_mom_sentiment import select_risk_adj_mom_sentiment_stocks

                print(f"   📊 Risk-Adj Mom Sentiment Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                new_risk_adj_mom_sentiment_stocks = select_risk_adj_mom_sentiment_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_risk_adj_mom_sentiment_stocks:
                    print(f"   📊 RiskAdj Sent Day {day_count}: {new_risk_adj_mom_sentiment_stocks}")
                    # Use universal smart rebalancing function
                    risk_adj_mom_sentiment_positions, risk_adj_mom_sentiment_cash, current_risk_adj_mom_sentiment_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="RiskAdj Sent",
                        current_stocks=current_risk_adj_mom_sentiment_stocks,
                        new_stocks=new_risk_adj_mom_sentiment_stocks,
                        positions=risk_adj_mom_sentiment_positions,
                        cash=risk_adj_mom_sentiment_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_BUFFER_SIZE,
                        strategy_stop_loss=STRATEGY_STOP_LOSS_PCT.get('Risk-Adj Mom Sentiment', STOP_LOSS_PCT)
                    )
                    strategies_rebalanced_today['RiskAdj Sent'] = rebalanced_flag

                    risk_adj_mom_sentiment_transaction_costs += rebalance_costs
                    risk_adj_mom_sentiment_last_rebalance_value = risk_adj_mom_sentiment_portfolio_value

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom Sentiment strategy error: {e}")

        # MULTI-HORIZON ENSEMBLE: Rebalance using multi-horizon signals from daily data
        if config.ENABLE_MULTI_TIMEFRAME_ENSEMBLE:
            try:
                from multi_timeframe_ensemble import select_multi_timeframe_stocks

                new_multi_tf_ensemble_stocks = select_multi_timeframe_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_multi_tf_ensemble_stocks:
                    print(f"   📊 Multi-Horizon Ensemble Day {day_count}: {new_multi_tf_ensemble_stocks}")
                    # Use universal smart rebalancing function
                    multi_tf_ensemble_positions, multi_tf_ensemble_cash, current_multi_tf_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Multi-Horizon Ensemble",
                        current_stocks=current_multi_tf_ensemble_stocks,
                        new_stocks=new_multi_tf_ensemble_stocks,
                        positions=multi_tf_ensemble_positions,
                        cash=multi_tf_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_multi_tf_ensemble_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Multi-Horizon Ensemble'] = rebalanced_flag
                    multi_tf_ensemble_transaction_costs += rebalance_costs
                    multi_tf_ensemble_last_rebalance_value = multi_tf_ensemble_portfolio_value
                else:
                    print(f"   ⚠️ Multi-Horizon Ensemble: No stocks selected (no consensus)")

            except Exception as e:
                print(f"   ⚠️ Multi-Horizon Ensemble strategy error: {e}")

        # MULTI-HORIZON INTRADAY: Rebalance using hourly data for medium/short horizons
        if config.ENABLE_MULTI_TIMEFRAME_INTRADAY_ENSEMBLE:
            try:
                from multi_timeframe_intraday_ensemble import select_multi_timeframe_intraday_stocks

                new_multi_tf_intraday_ensemble_stocks = select_multi_timeframe_intraday_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_multi_tf_intraday_ensemble_stocks:
                    print(f"   📊 Multi-Horizon Intraday Day {day_count}: {new_multi_tf_intraday_ensemble_stocks}")
                    multi_tf_intraday_ensemble_positions, multi_tf_intraday_ensemble_cash, current_multi_tf_intraday_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Multi-Horizon Intraday",
                        current_stocks=current_multi_tf_intraday_ensemble_stocks,
                        new_stocks=new_multi_tf_intraday_ensemble_stocks,
                        positions=multi_tf_intraday_ensemble_positions,
                        cash=multi_tf_intraday_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_multi_tf_intraday_ensemble_stocks
                    )
                    strategies_rebalanced_today['Multi-Horizon Intraday'] = rebalanced_flag
                    multi_tf_intraday_ensemble_transaction_costs += rebalance_costs
                    multi_tf_intraday_ensemble_last_rebalance_value = multi_tf_intraday_ensemble_portfolio_value
                else:
                    print(f"   ⚠️ Multi-Horizon Intraday: No stocks selected (no consensus)")

            except Exception as e:
                print(f"   ⚠️ Multi-Horizon Intraday strategy error: {e}")

        # === MEAN REVERSION, QUALITY+MOM, VOL-ADJ MOM STRATEGIES ===
        # These strategies run independently of Dynamic BH performance data

        # MEAN REVERSION: Rebalance to bottom N performers DAILY
        if config.ENABLE_MEAN_REVERSION:
            try:
                # Calculate current bottom N performers based on recent short-term performance
                # Mean reversion: buy stocks that have declined recently (expecting bounce back)
                current_bottom_performers = []

                print(f"   🔍 Mean Reversion: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # ✅ OPTIMIZED: Use pre-grouped data
                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]

                        # Use 1-month performance for mean reversion (opposite of momentum)
                        # ✅ FIX: Use explicit date range like other working strategies
                        perf_start_date_mr = current_date - timedelta(days=30)
                        data_slice = ticker_data.loc[perf_start_date_mr:current_date]
                        if len(data_slice) >= 10:  # Relaxed: at least 10 days of data
                            recent_data = data_slice.tail(21) if len(data_slice) >= 21 else data_slice
                            if len(recent_data) >= 2:
                                start_price = recent_data['Close'].iloc[0]
                                end_price = recent_data['Close'].iloc[-1]
                                if start_price > 0:
                                    monthly_return = ((end_price - start_price) / start_price) * 100
                                    current_bottom_performers.append((ticker, monthly_return))

                    except Exception as e:
                        continue

                print(f"   📊 Mean Reversion: Found {len(current_bottom_performers)} tickers with valid data")

                if current_bottom_performers:
                    current_bottom_performers.sort(key=lambda x: x[1])  # Sort by return (ascending = worst performers)
                    # FIXED: Select stocks with moderate losses (not worst performers) for mean reversion
                    # Filter for stocks with -20% to -5% returns (oversold but not crashing)
                    moderate_losers = [t for t, ret in current_bottom_performers if -20 <= ret <= -5]
                    new_mean_reversion_stocks = [ticker for ticker in moderate_losers[:PORTFOLIO_SIZE]]
                    print(f"   🎯 Mean Reversion: Selected {new_mean_reversion_stocks} (from {len(moderate_losers)} moderate losers)")

                    if new_mean_reversion_stocks:
                        print(f"   📊 Mean Reversion Day {day_count}: {new_mean_reversion_stocks}")
                        # Use universal smart rebalancing function
                        mean_reversion_positions, mean_reversion_cash, current_mean_reversion_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Mean Reversion",
                            current_stocks=current_mean_reversion_stocks,
                            new_stocks=new_mean_reversion_stocks,
                            positions=mean_reversion_positions,
                            cash=mean_reversion_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_mean_reversion_stocks  # Force initial allocation
                        )
                        strategies_rebalanced_today['Mean Reversion'] = rebalanced_flag
                        mean_reversion_transaction_costs += rebalance_costs
                        mean_reversion_last_rebalance_value = _mark_to_market_value(
                            mean_reversion_positions, mean_reversion_cash, ticker_data_grouped, current_date
                        )
                else:
                    print(f"   ⚠️ Mean Reversion: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Mean reversion selection failed: {e}")

        # QUALITY + MOMENTUM: Rebalance to top performers by combined quality+momentum score DAILY
        if config.ENABLE_QUALITY_MOM:
            try:
                # Calculate combined quality + momentum scores
                quality_momentum_scores = []

                print(f"   🔍 Quality+Mom: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # ✅ OPTIMIZED: Use pre-grouped data
                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]

                        # Use 3-month period for both quality and momentum assessment
                        # ✅ FIX: Use explicit date range like other working strategies
                        perf_start_date_qm = current_date - timedelta(days=90)
                        data_slice = ticker_data.loc[perf_start_date_qm:current_date]
                        if len(data_slice) >= 30:  # Relaxed: at least 30 days of data
                            recent_data = data_slice.tail(63) if len(data_slice) >= 63 else data_slice

                            if len(recent_data) >= 10:
                                # MOMENTUM SCORE: 3-month return
                                start_price = recent_data['Close'].iloc[0]
                                end_price = recent_data['Close'].iloc[-1]
                                momentum_score = ((end_price - start_price) / start_price) * 100 if start_price > 0 else -100

                                # QUALITY SCORE: Consistency (low volatility) + trend strength
                                returns = recent_data['Close'].pct_change(fill_method=None).dropna()
                                if len(returns) > 5:
                                    # Volatility (lower = higher quality)
                                    volatility = returns.std() * np.sqrt(252)  # Annualized
                                    quality_volatility = max(0, 50 - volatility * 100)  # Higher score for lower volatility

                                    # Trend consistency (higher = higher quality)
                                    trend_strength = abs(momentum_score) * (1 - volatility)  # Strong trend with low volatility

                                    # Combined score: 70% momentum, 30% quality
                                    combined_score = (momentum_score * 0.7) + (quality_volatility * 0.3)

                                    quality_momentum_scores.append((ticker, combined_score, momentum_score, quality_volatility))

                    except Exception as e:
                        continue

                print(f"   📊 Quality+Mom: Found {len(quality_momentum_scores)} tickers with valid data")

                if quality_momentum_scores:
                    # Sort by combined score (descending)
                    quality_momentum_scores.sort(key=lambda x: x[1], reverse=True)
                    new_quality_momentum_stocks = [ticker for ticker, score, mom, qual in quality_momentum_scores[:PORTFOLIO_SIZE]]
                    print(f"   🎯 Quality+Mom: Selected {new_quality_momentum_stocks}")

                    if new_quality_momentum_stocks:
                        print(f"   📊 Quality+Mom Day {day_count}: {new_quality_momentum_stocks}")
                        # Use universal smart rebalancing function
                        quality_momentum_positions, quality_momentum_cash, current_quality_momentum_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Quality+Mom",
                            current_stocks=current_quality_momentum_stocks,
                            new_stocks=new_quality_momentum_stocks,
                            positions=quality_momentum_positions,
                            cash=quality_momentum_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_quality_momentum_stocks  # Force initial allocation
                        )
                        strategies_rebalanced_today['Quality+Mom'] = rebalanced_flag
                        quality_momentum_transaction_costs += rebalance_costs
                        quality_momentum_last_rebalance_value = _mark_to_market_value(
                            quality_momentum_positions, quality_momentum_cash, ticker_data_grouped, current_date
                        )
                else:
                    print(f"   ⚠️ Quality+Mom: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Quality + momentum selection failed: {e}")

        # VOLATILITY-ADJUSTED MOMENTUM STRATEGY
        if config.ENABLE_VOLATILITY_ADJ_MOM:
            try:
                from shared_strategies import select_volatility_adj_mom_stocks

                new_volatility_adj_mom_stocks = select_volatility_adj_mom_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )

                if new_volatility_adj_mom_stocks:
                    print(f"   📊 Vol-Adj Mom Day {day_count}: {new_volatility_adj_mom_stocks}")
                    # Use universal smart rebalancing function
                    volatility_adj_mom_positions, volatility_adj_mom_cash, current_volatility_adj_mom_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Vol-Adj Mom",
                        current_stocks=current_volatility_adj_mom_stocks,
                        new_stocks=new_volatility_adj_mom_stocks,
                        positions=volatility_adj_mom_positions,
                        cash=volatility_adj_mom_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_volatility_adj_mom_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Vol-Adj Mom'] = rebalanced_flag
                    volatility_adj_mom_transaction_costs += rebalance_costs
                    volatility_adj_mom_last_rebalance_value = _mark_to_market_value(
                        volatility_adj_mom_positions, volatility_adj_mom_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ⚠️ Vol-Adj Mom: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Volatility-adjusted momentum selection failed: {e}")

        # === INVERSE ETF HEDGE STRATEGY ===
        # Standalone strategy: holds inverse ETFs when market is down, cash when market is up
        if config.ENABLE_INVERSE_ETF_HEDGE:
            try:
                from market_regime import get_trailing_market_regime
                from config import (INVERSE_ETF_HEDGE_THRESHOLD_LOW, INVERSE_ETF_HEDGE_THRESHOLD_MED,
                                    INVERSE_ETF_HEDGE_THRESHOLD_HIGH, INVERSE_ETF_HEDGE_BASE_ALLOCATION,
                                    INVERSE_ETF_HEDGE_MAX_ALLOCATION, INVERSE_ETF_HEDGE_PREFERENCE,
                                    INVERSE_ETF_HEDGE_MIN_HOLD_DAYS)

                market_return, is_market_up, proxy = get_trailing_market_regime(
                    ticker_data_grouped,
                    current_date,
                    lookback_days=5,
                )

                # Debug: Show market conditions periodically
                if day_count % 20 == 0:
                    if market_return is None:
                        print(f"   🛡️ Hedge Check: trailing 5d market return unavailable")
                    else:
                        print(f"   🛡️ Hedge Check: trailing 5d market return={market_return:+.1f}% via {proxy}")

                # === HYBRID INVERSE ETF HEDGE LOGIC ===
                # Uses the inverse of the shared market-up regime signal:
                # - trailing 5d return <= 0 starts hedge mode
                # - larger declines scale hedge allocation
                market_decline_pct = abs(market_return) if market_return is not None and market_return <= 0 else 0.0
                low_threshold_pct = INVERSE_ETF_HEDGE_THRESHOLD_LOW * 100
                med_threshold_pct = INVERSE_ETF_HEDGE_THRESHOLD_MED * 100
                high_threshold_pct = INVERSE_ETF_HEDGE_THRESHOLD_HIGH * 100

                # Calculate target hedge allocation
                if market_decline_pct >= high_threshold_pct:
                    target_hedge_pct = INVERSE_ETF_HEDGE_MAX_ALLOCATION  # 80% hedge
                elif market_decline_pct >= med_threshold_pct:
                    target_hedge_pct = 0.50  # 50% hedge
                elif market_return is not None and not is_market_up:
                    target_hedge_pct = INVERSE_ETF_HEDGE_BASE_ALLOCATION  # 20% hedge
                else:
                    target_hedge_pct = 0.0  # No hedge

                # Determine if we should be in hedge mode
                in_hedge_mode = target_hedge_pct > 0

                # Minimum hold days check - prevent whipsaw
                if current_inverse_etf_hedge_stocks and not in_hedge_mode:
                    if inverse_etf_hedge_days_in_hedge < INVERSE_ETF_HEDGE_MIN_HOLD_DAYS:
                        # Force continue hedge for minimum hold period
                        in_hedge_mode = True
                        target_hedge_pct = INVERSE_ETF_HEDGE_BASE_ALLOCATION  # Keep at least 20% hedge
                        if day_count % 20 == 0:
                            print(f"   🛡️ Hedge hold period: {inverse_etf_hedge_days_in_hedge}/{INVERSE_ETF_HEDGE_MIN_HOLD_DAYS} days (preventing whipsaw)")

                if in_hedge_mode and market_return is not None and not is_market_up:
                    # Market is down - select inverse ETFs based on allocation level
                    inverse_etf_hedge_days_in_hedge += 1

                    # Pick the best performing inverse ETFs from preference list
                    inverse_etf_scores = []
                    for etf in INVERSE_ETF_HEDGE_PREFERENCE:
                        if etf in ticker_data_grouped:
                            try:
                                etf_data = ticker_data_grouped[etf]
                                if len(etf_data) >= 30:
                                    recent = etf_data[etf_data.index <= current_date].tail(30)
                                    if len(recent) >= 5:
                                        perf_1m = (recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100
                                        inverse_etf_scores.append((etf, perf_1m))
                            except:
                                pass

                    if inverse_etf_scores:
                        # Sort by performance and pick top 2-4 based on allocation
                        inverse_etf_scores.sort(key=lambda x: x[1], reverse=True)
                        # Scale number of ETFs with allocation level
                        num_etfs = 2 if target_hedge_pct <= 0.30 else 3 if target_hedge_pct <= 0.60 else 4
                        new_inverse_etf_stocks = [etf for etf, _ in inverse_etf_scores[:num_etfs]]

                        if new_inverse_etf_stocks != current_inverse_etf_hedge_stocks:
                            print(f"   🛡️ Hybrid Hedge: Market down {market_return:+.1f}% over trailing 5d -> {target_hedge_pct:.0%} allocation, selecting {new_inverse_etf_stocks}")

                            # Rebalance to inverse ETFs
                            inverse_etf_hedge_positions, inverse_etf_hedge_cash, current_inverse_etf_hedge_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                                strategy_name="Hybrid Inverse ETF Hedge",
                                current_stocks=current_inverse_etf_hedge_stocks,
                                new_stocks=new_inverse_etf_stocks,
                                positions=inverse_etf_hedge_positions,
                                cash=inverse_etf_hedge_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=len(new_inverse_etf_stocks),
                                force_rebalance=not inverse_etf_hedge_initialized
                            )
                            strategies_rebalanced_today['Inverse ETF Hedge'] = rebalanced_flag
                            inverse_etf_hedge_transaction_costs += rebalance_costs
                            inverse_etf_hedge_initialized = True

                            # Log detailed allocation
                            total_invested = sum(
                                p['shares'] * p.get('entry_price', p.get('avg_price', 0))
                                for p in inverse_etf_hedge_positions.values()
                            )
                            print(f"   🛡️ Hedge Active: {target_hedge_pct:.0%} allocation, Cash=${inverse_etf_hedge_cash:,.0f}, Invested=${total_invested:,.0f}, Days in hedge={inverse_etf_hedge_days_in_hedge}")
                            for etf, pos in inverse_etf_hedge_positions.items():
                                avg_price = pos.get('entry_price', pos.get('avg_price', 0))
                                value = pos['shares'] * avg_price
                                pct = (value / (total_invested + inverse_etf_hedge_cash)) * 100 if (total_invested + inverse_etf_hedge_cash) > 0 else 0
                                print(f"      📈 {etf}: {pos['shares']:.0f} shares @ ${avg_price:.2f} = ${value:,.0f} ({pct:.1f}%)")
                else:
                    # Market is up or recovery period - reset hedge days and sell all inverse ETFs
                    if current_inverse_etf_hedge_stocks:
                        recovered_msg = f"{market_return:+.1f}% over trailing 5d" if market_return is not None else "market signal unavailable"
                        print(f"   🛡️ Hybrid Hedge: Market recovered ({recovered_msg}), selling hedge after {inverse_etf_hedge_days_in_hedge} days")
                        inverse_etf_hedge_days_in_hedge = 0  # Reset counter

                        # Sell all positions
                        for ticker, pos in list(inverse_etf_hedge_positions.items()):
                            if ticker in ticker_data_grouped:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    current_price = ticker_data[ticker_data.index <= current_date]['Close'].iloc[-1]
                                    sell_value = pos['shares'] * current_price
                                    sell_cost = sell_value * TRANSACTION_COST
                                    inverse_etf_hedge_cash += sell_value - sell_cost
                                    inverse_etf_hedge_transaction_costs += sell_cost
                                    print(f"   💰 Selling {ticker}: {pos['shares']:.0f} shares @ ${current_price:.2f}")
                                except:
                                    pass
                        inverse_etf_hedge_positions = {}
                        current_inverse_etf_hedge_stocks = []
                        print(f"   🛡️ Hedge closed. Holding cash: ${inverse_etf_hedge_cash:,.0f}")
                        current_inverse_etf_hedge_stocks = []

            except Exception as e:
                print(f"   ⚠️ Inverse ETF Hedge selection failed: {e}")

        # === ANALYST RECOMMENDATION STRATEGY ===
        if config.ENABLE_ANALYST_RECOMMENDATION:
            try:
                # Check if it's time to rebalance (weekly by default)
                should_rebalance_analyst = (
                    not analyst_rec_initialized or
                    (day_count - analyst_rec_last_rebalance_day) >= ANALYST_REBALANCE_DAYS
                )

                if should_rebalance_analyst:
                    from analyst_recommendation_strategy import (
                        fetch_all_analyst_data, select_analyst_recommendation_stocks
                    )

                    # Fetch analyst data if not cached or cache is old
                    if not analyst_data_cache:
                        try:
                            print(f"   📊 Analyst Recommendation: Fetching analyst data for {len(initial_top_tickers)} tickers...")
                        except UnicodeEncodeError:
                            print(f"   [Analyst] Fetching analyst data for {len(initial_top_tickers)} tickers...")
                        analyst_data_cache.update(fetch_all_analyst_data(initial_top_tickers, max_workers=10, show_progress=False))
                        try:
                            print(f"   📊 Analyst data available for {len(analyst_data_cache)} tickers")
                        except UnicodeEncodeError:
                            print(f"   [Analyst] Data available for {len(analyst_data_cache)} tickers")

                    # Select stocks based on analyst recommendations
                    new_analyst_stocks = select_analyst_recommendation_stocks(
                        tickers=initial_top_tickers,
                        ticker_data_grouped=ticker_data_grouped,
                        analyst_data=analyst_data_cache,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                        lookback_days=ANALYST_LOOKBACK_DAYS,
                        min_actions=ANALYST_MIN_ACTIONS
                    )

                    if new_analyst_stocks:
                        print(f"   📊 Analyst Rec Day {day_count}: {new_analyst_stocks}")
                        # Rebalance portfolio
                        analyst_rec_positions, analyst_rec_cash, current_analyst_rec_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Analyst Rec",
                            current_stocks=current_analyst_rec_stocks,
                            new_stocks=new_analyst_stocks,
                            positions=analyst_rec_positions,
                            cash=analyst_rec_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not analyst_rec_initialized
                        )
                        strategies_rebalanced_today['Analyst Rec'] = rebalanced_flag
                        analyst_rec_transaction_costs += rebalance_costs
                        analyst_rec_initialized = True
                        analyst_rec_last_rebalance_day = day_count

            except Exception as e:
                import traceback
                print(f"   [ERROR] Analyst Recommendation selection failed: {e}")
                traceback.print_exc()

        # === ENHANCED VOLATILITY TRADER STRATEGY ===
        if config.ENABLE_ENHANCED_VOLATILITY:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if 'last_enhanced_volatility_rebalance_day' not in locals():
                    last_enhanced_volatility_rebalance_day = 0

                if last_enhanced_volatility_rebalance_day == 0 or (day_count - last_enhanced_volatility_rebalance_day) >= 7:  # Weekly rebalance
                    print(f"\n🔄 Enhanced Volatility Trader: Evaluating portfolio (Day {day_count})...")

                    # Calculate volatility-adjusted momentum scores
                    enhanced_vol_scores = []
                    available_tickers = initial_top_tickers if initial_top_tickers else []

                    for ticker in available_tickers:
                        try:
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_data = ticker_data_grouped[ticker]

                            # Get data for last 30 days
                            end_date = current_date
                            start_date = current_date - timedelta(days=30)
                            data_slice = ticker_data.loc[start_date:end_date]

                            if len(data_slice) >= 20:
                                # Calculate returns
                                returns = data_slice['Close'].pct_change().dropna()

                                if len(returns) > 0:
                                    # Calculate volatility (annualized)
                                    volatility = returns.std() * np.sqrt(252)

                                    # Calculate momentum (20-day return)
                                    momentum = (data_slice['Close'].iloc[-1] / data_slice['Close'].iloc[0]) - 1

                                    # Calculate ATR for stop loss
                                    high_low = data_slice['High'] - data_slice['Low']
                                    high_close = np.abs(data_slice['High'] - data_slice['Close'].shift())
                                    low_close = np.abs(data_slice['Low'] - data_slice['Close'].shift())
                                    atr = np.maximum(high_low, np.maximum(high_close, low_close)).mean()

                                    # Enhanced score: momentum adjusted by volatility (lower volatility = higher score)
                                    if volatility > 0 and momentum > 0:  # Only positive momentum
                                        enhanced_score = momentum / volatility
                                        enhanced_vol_scores.append((ticker, enhanced_score, volatility, atr))

                        except Exception as e:
                            continue

                    if enhanced_vol_scores:
                        # Sort by enhanced score (descending)
                        enhanced_vol_scores.sort(key=lambda x: x[1], reverse=True)
                        top_enhanced_vol_stocks = [(t, s, v, a) for t, s, v, a in enhanced_vol_scores[:PORTFOLIO_SIZE]]

                        print(f"   📈 Top {PORTFOLIO_SIZE} Enhanced Volatility stocks: {[(t, f'{s*100:.2f}', f'{v*100:.1f}%') for t, s, v, a in top_enhanced_vol_stocks]}")

                        # Rebalance portfolio
                        total_value = enhanced_volatility_cash + sum(pos.get('value', 0) for pos in enhanced_volatility_positions.values())
                        capital_per_stock = total_value / PORTFOLIO_SIZE if PORTFOLIO_SIZE > 0 else 0

                        # Sell existing positions
                        for ticker in list(enhanced_volatility_positions.keys()):
                            if ticker not in [t for t, _, _, _ in top_enhanced_vol_stocks]:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        shares = enhanced_volatility_positions[ticker]['shares']
                                        gross_sale = shares * current_price
                                        sell_cost = gross_sale * TRANSACTION_COST
                                        enhanced_volatility_cash += gross_sale - sell_cost
                                        del enhanced_volatility_positions[ticker]
                                except Exception:
                                    continue

                        # Buy new positions with ATR-based stops
                        for ticker, score, volatility, atr in top_enhanced_vol_stocks:
                            if ticker not in enhanced_volatility_positions and capital_per_stock > 0:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            # Use target allocation or remaining cash, whichever is smaller
                                            max_buy = min(capital_per_stock, enhanced_volatility_cash / (1 + TRANSACTION_COST))
                                            shares = int(max_buy / current_price)
                                            if shares > 0:
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                if enhanced_volatility_cash >= buy_value + buy_cost:
                                                    enhanced_volatility_cash -= (buy_value + buy_cost)

                                                    # Set ATR-based stop loss (2x ATR) and take profit (3x ATR)
                                                    stop_loss = current_price - (2 * atr)
                                                    take_profit = current_price + (3 * atr)

                                                    enhanced_volatility_positions[ticker] = {
                                                        'shares': shares,
                                                        'entry_price': current_price,
                                                        'value': buy_value,
                                                        'stop_loss': stop_loss,
                                                        'take_profit': take_profit,
                                                        'atr': atr
                                                    }
                                except Exception:
                                    continue

                        last_enhanced_volatility_rebalance_day = day_count
                        print(f"   ✅ Enhanced Volatility: Rebalanced to {len(enhanced_volatility_positions)} positions")
                        strategies_rebalanced_today['Enhanced Volatility'] = True

                # Check stop losses and take profits daily
                positions_to_close = []
                for ticker, pos in enhanced_volatility_positions.items():
                    try:
                        ticker_data = ticker_data_grouped[ticker]
                        price_data = ticker_data.loc[:current_date]
                        if not price_data.empty:
                            current_price = price_data['Close'].dropna().iloc[-1]

                            # Check stop loss
                            if current_price <= pos['stop_loss']:
                                positions_to_close.append((ticker, current_price, 'Stop Loss'))
                            # Check take profit
                            elif current_price >= pos['take_profit']:
                                positions_to_close.append((ticker, current_price, 'Take Profit'))
                    except Exception:
                        continue

                # Close positions that hit stop loss or take profit
                for ticker, price, reason in positions_to_close:
                    try:
                        shares = enhanced_volatility_positions[ticker]['shares']
                        gross_sale = shares * price
                        sell_cost = gross_sale * TRANSACTION_COST
                        enhanced_volatility_cash += gross_sale - sell_cost
                        del enhanced_volatility_positions[ticker]
                        print(f"   🛑 Enhanced Volatility: Sold {ticker} @ ${price:.2f} ({reason})")
                    except Exception:
                        continue

            except Exception as e:
                print(f"   ⚠️ Enhanced Volatility Trader error: {e}")

        # === AI VOLATILITY ENSEMBLE STRATEGY ===
        if config.ENABLE_AI_VOLATILITY_ENSEMBLE:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if 'last_ai_vol_ensemble_rebalance_day' not in locals():
                    last_ai_vol_ensemble_rebalance_day = 0

                if last_ai_vol_ensemble_rebalance_day == 0 or (day_count - last_ai_vol_ensemble_rebalance_day) >= 7:  # Weekly rebalance
                    print(f"\n🔄 AI Volatility Ensemble: Evaluating portfolio (Day {day_count})...")

                    # Calculate AI-enhanced volatility scores
                    ai_vol_scores = []
                    available_tickers = initial_top_tickers if initial_top_tickers else []

                    for ticker in available_tickers:
                        try:
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_data = ticker_data_grouped[ticker]

                            # Get data for last 60 days
                            end_date = current_date
                            start_date = current_date - timedelta(days=60)
                            data_slice = ticker_data.loc[start_date:end_date]

                            if len(data_slice) >= 30:
                                # Calculate multiple volatility metrics
                                returns = data_slice['Close'].pct_change().dropna()

                                if len(returns) > 5:
                                    # 1. Realized volatility (20-day)
                                    real_vol = returns.tail(20).std() * np.sqrt(252)

                                    # 2. Volatility trend (is volatility increasing or decreasing?)
                                    vol_short = returns.tail(10).std() * np.sqrt(252)
                                    vol_long = returns.head(20).std() * np.sqrt(252)
                                    vol_trend = (vol_short - vol_long) / vol_long if vol_long > 0 else 0

                                    # 3. Price momentum (30 calendar days)
                                    start_30d = current_date - timedelta(days=30)
                                    data_30d = data_slice[data_slice.index >= start_30d]
                                    if len(data_30d) >= 10:
                                        price_momentum = (data_30d['Close'].iloc[-1] / data_30d['Close'].iloc[0]) - 1
                                    else:
                                        price_momentum = 0

                                    # 4. Volume confirmation
                                    volume_ratio = data_slice['Volume'].tail(10).mean() / data_slice['Volume'].head(30).mean()

                                    # AI-enhanced score: combine multiple factors
                                    # Prefer: moderate volatility, decreasing vol trend, positive momentum, high volume
                                    if real_vol > 0:
                                        # Normalize factors
                                        vol_score = 1 / (1 + real_vol)  # Lower volatility = higher score
                                        trend_score = 1 - max(0, vol_trend)  # Decreasing volatility = higher score
                                        momentum_score = max(0, price_momentum)  # Positive momentum = higher score
                                        volume_score = min(2, volume_ratio)  # Higher volume = higher score (capped at 2x)

                                        # Weighted combination (AI-optimized weights)
                                        ai_score = (0.3 * vol_score +
                                                   0.25 * trend_score +
                                                   0.25 * momentum_score +
                                                   0.2 * volume_score)

                                        # Only include stocks with positive momentum
                                        if price_momentum > 0:
                                            ai_vol_scores.append((ticker, ai_score, real_vol, price_momentum, vol_trend))

                        except Exception:
                            continue

                    if ai_vol_scores:
                        # Sort by AI score (descending)
                        ai_vol_scores.sort(key=lambda x: x[1], reverse=True)
                        top_ai_vol_stocks = [(t, s, v, m, tr) for t, s, v, m, tr in ai_vol_scores[:PORTFOLIO_SIZE]]

                        print(f"   🤖 Top {PORTFOLIO_SIZE} AI Volatility stocks: {[(t, f'{s*100:.1f}', f'{v*100:.1f}%', f'{m*100:.1f}%') for t, s, v, m, tr in top_ai_vol_stocks]}")

                        # Rebalance portfolio with volatility caps
                        total_value = ai_volatility_ensemble_cash + sum(pos.get('value', 0) for pos in ai_volatility_ensemble_positions.values())
                        capital_per_stock = total_value / PORTFOLIO_SIZE if PORTFOLIO_SIZE > 0 else 0

                        # Apply volatility cap: max 15% allocation per position
                        max_position_value = total_value * 0.15

                        # Sell existing positions
                        for ticker in list(ai_volatility_ensemble_positions.keys()):
                            if ticker not in [t for t, _, _, _, _ in top_ai_vol_stocks]:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        shares = ai_volatility_ensemble_positions[ticker]['shares']
                                        gross_sale = shares * current_price
                                        sell_cost = gross_sale * TRANSACTION_COST
                                        ai_volatility_ensemble_cash += gross_sale - sell_cost
                                        del ai_volatility_ensemble_positions[ticker]
                                except Exception:
                                    continue

                        # Buy new positions with volatility caps
                        for ticker, ai_score, vol, momentum, vol_trend in top_ai_vol_stocks:
                            if ticker not in ai_volatility_ensemble_positions and capital_per_stock > 0:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            # Apply volatility cap and use remaining cash
                                            position_value = min(capital_per_stock, max_position_value, ai_volatility_ensemble_cash / (1 + TRANSACTION_COST))
                                            shares = int(position_value / current_price)
                                            if shares > 0:
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                if ai_volatility_ensemble_cash >= buy_value + buy_cost:
                                                    ai_volatility_ensemble_cash -= (buy_value + buy_cost)

                                                    ai_volatility_ensemble_positions[ticker] = {
                                                        'shares': shares,
                                                        'entry_price': current_price,
                                                        'value': buy_value,
                                                        'ai_score': ai_score,
                                                        'volatility': vol,
                                                        'momentum': momentum
                                                    }
                                except Exception:
                                    continue

                        last_ai_vol_ensemble_rebalance_day = day_count
                        print(f"   ✅ AI Volatility Ensemble: Rebalanced to {len(ai_volatility_ensemble_positions)} positions")
                        strategies_rebalanced_today['AI Volatility Ensemble'] = True

            except Exception as e:
                print(f"   ⚠️ AI Volatility Ensemble error: {e}")

        # === MOMENTUM + AI HYBRID STRATEGY ===
        if config.ENABLE_MOMENTUM_AI_HYBRID:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if last_momentum_ai_hybrid_rebalance_day == 0 or (day_count - last_momentum_ai_hybrid_rebalance_day) >= AI_REBALANCE_FREQUENCY_DAYS:
                    print(f"\n🔄 Momentum+AI Hybrid: Evaluating portfolio (Day {day_count})...")

                    from shared_strategies import select_momentum_ai_hybrid_stocks
                    top_momentum_stocks = select_momentum_ai_hybrid_stocks(
                        initial_top_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                    )

                    if top_momentum_stocks:
                        momentum_ai_hybrid_positions, momentum_ai_hybrid_cash, current_momentum_ai_hybrid_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Momentum+AI Hybrid",
                            current_stocks=current_momentum_ai_hybrid_stocks,
                            new_stocks=top_momentum_stocks,
                            positions=momentum_ai_hybrid_positions,
                            cash=momentum_ai_hybrid_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=last_momentum_ai_hybrid_rebalance_day == 0  # Force initial allocation
                        )
                        strategies_rebalanced_today['Momentum+AI'] = rebalanced_flag
                        momentum_ai_hybrid_transaction_costs += rebalance_costs

                        last_momentum_ai_hybrid_rebalance_day = day_count

            except Exception as e:
                print(f"   ⚠️ Momentum+AI hybrid failed: {e}")
                import traceback
                traceback.print_exc()

        # === NEW ADVANCED STRATEGIES ===

        # MOMENTUM ACCELERATION STRATEGY
        if config.ENABLE_MOMENTUM_ACCELERATION:
            try:
                from new_strategies import select_momentum_acceleration_stocks

                print(f"   📈 Momentum Acceleration: Analyzing {len(initial_top_tickers)} tickers...")

                new_mom_accel_stocks = select_momentum_acceleration_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_mom_accel_stocks:
                    print(f"   📊 Mom Accel Day {day_count}: {new_mom_accel_stocks}")
                    total_value = mom_accel_cash + sum(pos.get('value', 0) for pos in mom_accel_positions.values())
                    capital_per_stock = total_value / len(new_mom_accel_stocks)

                    # Sell positions not in new selection
                    for ticker in list(mom_accel_positions.keys()):
                        if ticker not in new_mom_accel_stocks:
                            if ticker in ticker_data_grouped:
                                price_data = ticker_data_grouped[ticker].loc[:current_date]
                                if not price_data.empty:
                                    current_price = price_data['Close'].dropna().iloc[-1]
                                    shares = mom_accel_positions[ticker]['shares']
                                    gross_sale = shares * current_price
                                    sell_cost = gross_sale * TRANSACTION_COST
                                    mom_accel_transaction_costs += sell_cost
                                    mom_accel_cash += gross_sale - sell_cost
                            del mom_accel_positions[ticker]

                    # Buy new positions
                    for ticker in new_mom_accel_stocks:
                        if ticker not in mom_accel_positions:
                            if ticker in ticker_data_grouped:
                                price_data = ticker_data_grouped[ticker].loc[:current_date]
                                if not price_data.empty:
                                    current_price = price_data['Close'].dropna().iloc[-1]
                                    if current_price > 0:
                                        # Use target allocation or remaining cash, whichever is smaller
                                        max_buy = min(capital_per_stock, mom_accel_cash / (1 + TRANSACTION_COST))
                                        shares = int(max_buy / current_price)
                                        if shares > 0:
                                            buy_value = shares * current_price
                                            buy_cost = buy_value * TRANSACTION_COST
                                            if mom_accel_cash >= buy_value + buy_cost:
                                                mom_accel_transaction_costs += buy_cost
                                                mom_accel_cash -= (buy_value + buy_cost)
                                                mom_accel_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}

                    old_stocks = current_mom_accel_stocks.copy()
                    current_mom_accel_stocks = new_mom_accel_stocks
                    strategies_rebalanced_today['Mom Acceleration'] = set(new_mom_accel_stocks) != set(old_stocks)

            except Exception as e:
                print(f"   ⚠️ Momentum Acceleration error: {e}")

        # CONCENTRATED 3M STRATEGY
        if config.ENABLE_CONCENTRATED_3M:
            try:
                concentrated_3m_days_since_rebalance += 1

                # Only rebalance monthly
                if concentrated_3m_days_since_rebalance >= CONCENTRATED_3M_REBALANCE_DAYS or not current_concentrated_3m_stocks:
                    from new_strategies import select_concentrated_3m_stocks

                    print(f"   🎯 Concentrated 3M: Analyzing {len(initial_top_tickers)} tickers...")

                    new_concentrated_3m_stocks = select_concentrated_3m_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                    )

                    if new_concentrated_3m_stocks:
                        print(f"   📊 Concentrated 3M Day {day_count}: {new_concentrated_3m_stocks}")
                        # Use universal smart rebalancing function
                        concentrated_3m_positions, concentrated_3m_cash, current_concentrated_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="Concentrated 3M",
                            current_stocks=current_concentrated_3m_stocks,
                            new_stocks=new_concentrated_3m_stocks,
                            positions=concentrated_3m_positions,
                            cash=concentrated_3m_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_concentrated_3m_stocks  # Force initial allocation
                        )
                        strategies_rebalanced_today['Concentrated 3M'] = rebalanced_flag
                        concentrated_3m_transaction_costs += rebalance_costs
                        concentrated_3m_days_since_rebalance = 0

            except Exception as e:
                print(f"   ⚠️ Concentrated 3M error: {e}")

        # DUAL MOMENTUM STRATEGY
        if config.ENABLE_DUAL_MOMENTUM:
            try:
                from new_strategies import select_dual_momentum_stocks

                print(f"   📊 Dual Momentum: Analyzing {len(initial_top_tickers)} tickers...")

                new_dual_mom_stocks, is_risk_on = select_dual_momentum_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                # If risk-off, sell all positions
                if not is_risk_on:
                    for ticker in list(dual_mom_positions.keys()):
                        if ticker in ticker_data_grouped:
                            price_data = ticker_data_grouped[ticker].loc[:current_date]
                            if not price_data.empty:
                                current_price = price_data['Close'].dropna().iloc[-1]
                                shares = dual_mom_positions[ticker]['shares']
                                gross_sale = shares * current_price
                                sell_cost = gross_sale * TRANSACTION_COST
                                dual_mom_transaction_costs += sell_cost
                                dual_mom_cash += gross_sale - sell_cost
                        del dual_mom_positions[ticker]
                    current_dual_mom_stocks = []
                    dual_mom_is_risk_on = False
                elif new_dual_mom_stocks:
                    dual_mom_is_risk_on = True
                    # Use universal smart rebalancing function
                    dual_mom_positions, dual_mom_cash, current_dual_mom_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Dual Momentum",
                        current_stocks=current_dual_mom_stocks,
                        new_stocks=new_dual_mom_stocks,
                        positions=dual_mom_positions,
                        cash=dual_mom_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dual_mom_stocks  # Force initial allocation
                    )
                    strategies_rebalanced_today['Dual Momentum'] = rebalanced_flag
                    dual_mom_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Dual Momentum error: {e}")

        # TREND FOLLOWING ATR STRATEGY
        if config.ENABLE_TREND_FOLLOWING_ATR:
            try:
                from new_strategies import select_trend_following_atr_stocks, reset_trend_atr_state

                # Reset on first day
                if day_count == 1:
                    reset_trend_atr_state()

                print(f"   📈 Trend Following ATR: Analyzing {len(initial_top_tickers)} tickers...")

                stocks_to_buy, stocks_to_sell = select_trend_following_atr_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                # Process sells first
                for ticker in stocks_to_sell:
                    if ticker in trend_atr_positions:
                        if ticker in ticker_data_grouped:
                            price_data = ticker_data_grouped[ticker].loc[:current_date]
                            if not price_data.empty:
                                current_price = price_data['Close'].dropna().iloc[-1]
                                shares = trend_atr_positions[ticker]['shares']
                                gross_sale = shares * current_price
                                sell_cost = gross_sale * TRANSACTION_COST
                                trend_atr_transaction_costs += sell_cost
                                trend_atr_cash += gross_sale - sell_cost
                        del trend_atr_positions[ticker]

                # Process buys
                if stocks_to_buy:
                    total_value = trend_atr_cash + sum(pos.get('value', 0) for pos in trend_atr_positions.values())
                    available_slots = PORTFOLIO_SIZE - len(trend_atr_positions)
                    if available_slots > 0:
                        capital_per_stock = total_value / PORTFOLIO_SIZE

                        for ticker in stocks_to_buy[:available_slots]:
                            if ticker not in trend_atr_positions:
                                if ticker in ticker_data_grouped:
                                    price_data = ticker_data_grouped[ticker].loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            # Use target allocation or remaining cash, whichever is smaller
                                            max_buy = min(capital_per_stock, trend_atr_cash / (1 + TRANSACTION_COST))
                                            shares = int(max_buy / current_price)
                                            if shares > 0:
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                if trend_atr_cash >= buy_value + buy_cost:
                                                    trend_atr_transaction_costs += buy_cost
                                                    trend_atr_cash -= (buy_value + buy_cost)
                                                    trend_atr_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}

                old_stocks = current_trend_atr_stocks.copy()
                current_trend_atr_stocks = list(trend_atr_positions.keys())
                strategies_rebalanced_today['Trend ATR'] = set(current_trend_atr_stocks) != set(old_stocks)

            except Exception as e:
                print(f"   ⚠️ Trend Following ATR error: {e}")

        # ELITE HYBRID STRATEGY (Mom-Vol 6M + 1Y/3M Ratio)
        if config.ENABLE_ELITE_HYBRID:
            try:
                from elite_hybrid_strategy import select_elite_hybrid_stocks

                print(f"   🏆 Elite Hybrid: Analyzing {len(initial_top_tickers)} tickers...")

                new_elite_hybrid_stocks = select_elite_hybrid_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_elite_hybrid_stocks:
                    print(f"   📊 Elite Hybrid Day {day_count}: {new_elite_hybrid_stocks}")
                    elite_hybrid_positions, elite_hybrid_cash, current_elite_hybrid_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Elite Hybrid",
                        current_stocks=current_elite_hybrid_stocks,
                        new_stocks=new_elite_hybrid_stocks,
                        positions=elite_hybrid_positions,
                        cash=elite_hybrid_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_elite_hybrid_stocks
                    )
                    strategies_rebalanced_today['Elite Hybrid'] = rebalanced_flag
                    elite_hybrid_transaction_costs += rebalance_costs
                    elite_hybrid_last_rebalance_value = elite_hybrid_portfolio_value

            except Exception as e:
                print(f"   ⚠️ Elite Hybrid error: {e}")

        # ELITE RISK STRATEGY (Risk-Adj Mom base + Elite Hybrid dip/vol bonuses)
        if config.ENABLE_ELITE_RISK:
            try:
                from elite_risk_strategy import select_elite_risk_stocks

                new_elite_risk_stocks = select_elite_risk_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_elite_risk_stocks:
                    print(f"   📊 Elite Risk Day {day_count}: {new_elite_risk_stocks}")
                    elite_risk_positions, elite_risk_cash, current_elite_risk_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Elite Risk",
                        current_stocks=current_elite_risk_stocks,
                        new_stocks=new_elite_risk_stocks,
                        positions=elite_risk_positions,
                        cash=elite_risk_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_elite_risk_stocks
                    )
                    strategies_rebalanced_today['Elite Risk'] = rebalanced_flag
                    elite_risk_transaction_costs += rebalance_costs
                    elite_risk_last_rebalance_value = elite_risk_portfolio_value

            except Exception as e:
                print(f"   ⚠️ Elite Risk error: {e}")

        # RISK-ADJ MOM 6M STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_6M:
            try:
                from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks

                new_risk_adj_mom_6m_stocks = select_risk_adj_mom_6m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_risk_adj_mom_6m_stocks:
                    print(f"   📊 Risk-Adj Mom 6M Day {day_count}: {new_risk_adj_mom_6m_stocks}")
                    risk_adj_mom_6m_positions, risk_adj_mom_6m_cash, current_risk_adj_mom_6m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 6M",
                        current_stocks=current_risk_adj_mom_6m_stocks,
                        new_stocks=new_risk_adj_mom_6m_stocks,
                        positions=risk_adj_mom_6m_positions,
                        cash=risk_adj_mom_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_6m_stocks
                    )
                    strategies_rebalanced_today['Risk-Adj Mom 6M'] = rebalanced_flag
                    risk_adj_mom_6m_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 6M error: {e}")

        # RISK-ADJ MOM 6M MONTHLY STRATEGY (same scoring, rebalance start of month only)
        if config.ENABLE_RISK_ADJ_MOM_6M_MONTHLY:
            should_rebalance_6m_mom_monthly = (not risk_adj_mom_6m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_6m_mom_monthly:
                try:
                    from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks

                    new_stocks = select_risk_adj_mom_6m_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                    )

                    if new_stocks:
                        print(f"   📊 Risk-Adj Mom 6M Monthly Day {day_count}: {new_stocks}")
                        if not risk_adj_mom_6m_monthly_initialized:
                            print(f"   🎯 Risk-Adj Mom 6M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Risk-Adj Mom 6M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        risk_adj_mom_6m_monthly_positions, risk_adj_mom_6m_monthly_cash, current_risk_adj_mom_6m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="RiskAdj 6M Mth",
                            current_stocks=current_risk_adj_mom_6m_monthly_stocks,
                            new_stocks=new_stocks,
                            positions=risk_adj_mom_6m_monthly_positions,
                            cash=risk_adj_mom_6m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not risk_adj_mom_6m_monthly_initialized)
                        strategies_rebalanced_today['RiskAdj 6M Mth'] = rebalanced_flag
                        risk_adj_mom_6m_monthly_transaction_costs += rc
                        risk_adj_mom_6m_monthly_initialized = True
                        risk_adj_mom_6m_monthly_last_month = current_date.month

                except Exception as e:
                    print(f"   ⚠️ Risk-Adj Mom 6M Monthly error: {e}")

        # RISK-ADJ MOM 3M STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_3M:
            try:
                from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks

                new_risk_adj_mom_3m_stocks = select_risk_adj_mom_3m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_risk_adj_mom_3m_stocks:
                    print(f"   📊 Risk-Adj Mom 3M Day {day_count}: {new_risk_adj_mom_3m_stocks}")
                    risk_adj_mom_3m_positions, risk_adj_mom_3m_cash, current_risk_adj_mom_3m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 3M",
                        current_stocks=current_risk_adj_mom_3m_stocks,
                        new_stocks=new_risk_adj_mom_3m_stocks,
                        positions=risk_adj_mom_3m_positions,
                        cash=risk_adj_mom_3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_3m_stocks
                    )
                    strategies_rebalanced_today['Risk-Adj Mom 3M'] = rebalanced_flag
                    risk_adj_mom_3m_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 3M error: {e}")

        # RISK-ADJ MOM 3M MONTHLY STRATEGY (same scoring, rebalance start of month only)
        if config.ENABLE_RISK_ADJ_MOM_3M_MONTHLY:
            should_rebalance_3m_mom_monthly = (not risk_adj_mom_3m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_3m_mom_monthly:
                try:
                    from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks

                    new_stocks = select_risk_adj_mom_3m_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                    )

                    if new_stocks:
                        print(f"   📊 Risk-Adj Mom 3M Monthly Day {day_count}: {new_stocks}")
                        if not risk_adj_mom_3m_monthly_initialized:
                            print(f"   🎯 Risk-Adj Mom 3M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Risk-Adj Mom 3M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        risk_adj_mom_3m_monthly_positions, risk_adj_mom_3m_monthly_cash, current_risk_adj_mom_3m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="RiskAdj 3M Mth",
                            current_stocks=current_risk_adj_mom_3m_monthly_stocks,
                            new_stocks=new_stocks,
                            positions=risk_adj_mom_3m_monthly_positions,
                            cash=risk_adj_mom_3m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not risk_adj_mom_3m_monthly_initialized)
                        strategies_rebalanced_today['RiskAdj 3M Mth'] = rebalanced_flag
                        risk_adj_mom_3m_monthly_transaction_costs += rc
                        risk_adj_mom_3m_monthly_initialized = True
                        risk_adj_mom_3m_monthly_last_month = current_date.month

                except Exception as e:
                    print(f"   ⚠️ Risk-Adj Mom 3M Monthly error: {e}")

        # RISK-ADJ MOM 3M SENTIMENT STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_3M_SENTIMENT:
            try:
                from risk_adj_mom_3m_sentiment_strategy import select_risk_adj_mom_3m_sentiment_stocks

                new_stocks = select_risk_adj_mom_3m_sentiment_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_stocks:
                    print(f"   📊 RiskAdj 3M Sent Day {day_count}: {new_stocks}")
                    risk_adj_mom_3m_sentiment_positions, risk_adj_mom_3m_sentiment_cash, current_risk_adj_mom_3m_sentiment_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="RiskAdj 3M Sent",
                        current_stocks=current_risk_adj_mom_3m_sentiment_stocks,
                        new_stocks=new_stocks,
                        positions=risk_adj_mom_3m_sentiment_positions,
                        cash=risk_adj_mom_3m_sentiment_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_3m_sentiment_stocks
                    )
                    strategies_rebalanced_today['RiskAdj 3M Sent'] = rebalanced_flag
                    risk_adj_mom_3m_sentiment_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 3M Sentiment error: {e}")

        # VOL-SWEET MOMENTUM STRATEGY
        if config.ENABLE_VOL_SWEET_MOM:
            try:
                from vol_sweet_mom_strategy import select_vol_sweet_mom_stocks

                new_stocks = select_vol_sweet_mom_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_stocks:
                    print(f"   📊 VolSweet Mom Day {day_count}: {new_stocks}")
                    vol_sweet_mom_positions, vol_sweet_mom_cash, current_vol_sweet_mom_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="VolSweet Mom",
                        current_stocks=current_vol_sweet_mom_stocks,
                        new_stocks=new_stocks,
                        positions=vol_sweet_mom_positions,
                        cash=vol_sweet_mom_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_vol_sweet_mom_stocks
                    )
                    strategies_rebalanced_today['VolSweet Mom'] = rebalanced_flag
                    vol_sweet_mom_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Vol-Sweet Momentum error: {e}")

        # Reset daily rebalancing flags for all strategies at start of day
        strategies_rebalanced_today.clear()

        # RISK-ADJ MOM 3M MARKET-UP ONLY STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_3M_MARKET_UP:
            try:
                from risk_adj_mom_3m_market_up_strategy import select_risk_adj_mom_3m_market_up_stocks

                new_stocks = select_risk_adj_mom_3m_market_up_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_stocks:
                    print(f"   📊 RiskAdj 3M Up Day {day_count}: {new_stocks}")
                    # Store previous positions before rebalancing for tracking
                    prev_risk_adj_mom_3m_market_up_stocks = current_risk_adj_mom_3m_market_up_stocks.copy()
                    risk_adj_mom_3m_market_up_positions, risk_adj_mom_3m_market_up_cash, current_risk_adj_mom_3m_market_up_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 3M Market-Up",
                        current_stocks=current_risk_adj_mom_3m_market_up_stocks,
                        new_stocks=new_stocks,
                        positions=risk_adj_mom_3m_market_up_positions,
                        cash=risk_adj_mom_3m_market_up_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_3m_market_up_stocks
                    )
                    strategies_rebalanced_today['RiskAdj 3M Up'] = rebalanced_flag
                    risk_adj_mom_3m_market_up_transaction_costs += rebalance_costs
                    # Check if rebalancing actually changed positions
                    risk_adj_mom_3m_market_up_rebalanced_today = set(current_risk_adj_mom_3m_market_up_stocks) != set(prev_risk_adj_mom_3m_market_up_stocks)

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 3M Market-Up error: {e}")

        # RISK-ADJ MOM 3M WITH STOPS STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS:
            try:
                from risk_adj_mom_3m_with_stops_strategy import select_risk_adj_mom_3m_with_stops_stocks

                new_stocks = select_risk_adj_mom_3m_with_stops_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_stocks:
                    print(f"   📊 RiskAdj 3M Stop Day {day_count}: {new_stocks}")
                    # Store previous positions before rebalancing for tracking
                    prev_risk_adj_mom_3m_with_stops_stocks = current_risk_adj_mom_3m_with_stops_stocks.copy()
                    risk_adj_mom_3m_with_stops_positions, risk_adj_mom_3m_with_stops_cash, current_risk_adj_mom_3m_with_stops_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 3M with Stops",
                        current_stocks=current_risk_adj_mom_3m_with_stops_stocks,
                        new_stocks=new_stocks,
                        positions=risk_adj_mom_3m_with_stops_positions,
                        cash=risk_adj_mom_3m_with_stops_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_3m_with_stops_stocks
                    )
                    strategies_rebalanced_today['RiskAdj 3M Stop'] = rebalanced_flag
                    risk_adj_mom_3m_with_stops_transaction_costs += rebalance_costs
                    # Check if rebalancing actually changed positions
                    risk_adj_mom_3m_with_stops_rebalanced_today = set(current_risk_adj_mom_3m_with_stops_stocks) != set(prev_risk_adj_mom_3m_with_stops_stocks)

                    # Track entry prices for new positions
                    for ticker in new_stocks:
                        if ticker in risk_adj_mom_3m_with_stops_positions:
                            ticker_df = ticker_data_grouped.get(ticker)
                            if ticker_df is not None:
                                from backtesting import _last_valid_close_up_to
                                entry_price = _last_valid_close_up_to(ticker_df, current_date)
                                if entry_price is not None:
                                    risk_adj_mom_3m_with_stops_positions[ticker]['entry_price'] = entry_price
                                    risk_adj_mom_3m_with_stops_positions[ticker]['entry_date'] = current_date

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 3M with Stops error: {e}")

        # RISK-ADJ MOM 1M + VOL-SWEET STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET:
            try:
                from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks

                new_stocks = select_risk_adj_mom_1m_vol_sweet_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_stocks:
                    print(f"   📊 1M VolSweet Day {day_count}: {new_stocks}")
                    risk_adj_mom_1m_vol_sweet_positions, risk_adj_mom_1m_vol_sweet_cash, current_risk_adj_mom_1m_vol_sweet_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="1M VolSweet",
                        current_stocks=current_risk_adj_mom_1m_vol_sweet_stocks,
                        new_stocks=new_stocks,
                        positions=risk_adj_mom_1m_vol_sweet_positions,
                        cash=risk_adj_mom_1m_vol_sweet_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_1m_vol_sweet_stocks
                    )
                    strategies_rebalanced_today['1M VolSweet'] = rebalanced_flag
                    risk_adj_mom_1m_vol_sweet_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ 1M VolSweet error: {e}")

        # BH 1Y VOL-SWEET ACCEL STRATEGY
        if config.ENABLE_BH_1Y_VOL_SWEET_ACCEL:
            try:
                from shared_strategies import select_bh_1y_volsweet_accel_stocks

                new_bh_1y_volsweet_accel_stocks = select_bh_1y_volsweet_accel_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_bh_1y_volsweet_accel_stocks:
                    print(f"   📊 BH 1Y VolSweet Accel Day {day_count}: {new_bh_1y_volsweet_accel_stocks}")
                    bh_1y_volsweet_accel_positions, bh_1y_volsweet_accel_cash, current_bh_1y_volsweet_accel_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="BH 1Y VolSweet Accel",
                        current_stocks=current_bh_1y_volsweet_accel_stocks,
                        new_stocks=new_bh_1y_volsweet_accel_stocks,
                        positions=bh_1y_volsweet_accel_positions,
                        cash=bh_1y_volsweet_accel_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_bh_1y_volsweet_accel_stocks
                    )
                    strategies_rebalanced_today['BH 1Y VolSweet Accel'] = rebalanced_flag
                    bh_1y_volsweet_accel_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ BH 1Y VolSweet Accel error: {e}")

        # BH 1Y DYNAMIC ACCEL STRATEGY (dynamic rebalancing)
        if config.ENABLE_BH_1Y_DYNAMIC_ACCEL:
            try:
                from shared_strategies import select_bh_1y_dynamic_accel_stocks

                # Check if this is initial allocation (no positions yet)
                is_initial = not bh_1y_dynamic_accel_initialized and len(bh_1y_dynamic_accel_positions) == 0

                bh_1y_dynamic_accel_days_since_rebalance += 1

                new_bh_1y_dynamic_accel_stocks, should_rebalance = select_bh_1y_dynamic_accel_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                    days_since_rebalance=0 if is_initial else bh_1y_dynamic_accel_days_since_rebalance,  # Pass 0 for initial
                    min_days=5,
                    max_days=44
                )

                if should_rebalance and new_bh_1y_dynamic_accel_stocks:
                    print(f"   📊 BH 1Y Dynamic Accel Day {day_count}: {new_bh_1y_dynamic_accel_stocks}")
                    if not bh_1y_dynamic_accel_initialized:
                        print(f"   🎯 BH 1Y Dynamic Accel: Initializing")
                        bh_1y_dynamic_accel_initialized = True

                    bh_1y_dynamic_accel_positions, bh_1y_dynamic_accel_cash, current_bh_1y_dynamic_accel_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="BH 1Y Dynamic Accel",
                        current_stocks=current_bh_1y_dynamic_accel_stocks,
                        new_stocks=new_bh_1y_dynamic_accel_stocks,
                        positions=bh_1y_dynamic_accel_positions,
                        cash=bh_1y_dynamic_accel_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not bh_1y_dynamic_accel_initialized
                    )
                    strategies_rebalanced_today['BH 1Y Dynamic Accel'] = rebalanced_flag
                    bh_1y_dynamic_accel_transaction_costs += rebalance_costs
                    bh_1y_dynamic_accel_days_since_rebalance = 0

            except Exception as e:
                print(f"   ⚠️ BH 1Y Dynamic Accel error: {e}")

        # RISK-ADJ MOM 1M STRATEGY
        if config.ENABLE_RISK_ADJ_MOM_1M:
            try:
                from risk_adj_mom_1m_strategy import select_risk_adj_mom_1m_stocks

                new_risk_adj_mom_1m_stocks = select_risk_adj_mom_1m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                )

                if new_risk_adj_mom_1m_stocks:
                    print(f"   📊 Risk-Adj Mom 1M Day {day_count}: {new_risk_adj_mom_1m_stocks}")
                    risk_adj_mom_1m_positions, risk_adj_mom_1m_cash, current_risk_adj_mom_1m_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 1M",
                        current_stocks=current_risk_adj_mom_1m_stocks,
                        new_stocks=new_risk_adj_mom_1m_stocks,
                        positions=risk_adj_mom_1m_positions,
                        cash=risk_adj_mom_1m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_1m_stocks
                    )
                    strategies_rebalanced_today['Risk-Adj Mom 1M'] = rebalanced_flag
                    risk_adj_mom_1m_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 1M error: {e}")

        # RISK-ADJ MOM 1M MONTHLY STRATEGY (same scoring, rebalance start of month only)
        if config.ENABLE_RISK_ADJ_MOM_1M_MONTHLY:
            should_rebalance_1m_mom_monthly = (not risk_adj_mom_1m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_1m_mom_monthly:
                try:
                    from risk_adj_mom_1m_strategy import select_risk_adj_mom_1m_stocks

                    new_stocks = select_risk_adj_mom_1m_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE
                    )

                    if new_stocks:
                        print(f"   📊 Risk-Adj Mom 1M Monthly Day {day_count}: {new_stocks}")
                        if not risk_adj_mom_1m_monthly_initialized:
                            print(f"   🎯 Risk-Adj Mom 1M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Risk-Adj Mom 1M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        risk_adj_mom_1m_monthly_positions, risk_adj_mom_1m_monthly_cash, current_risk_adj_mom_1m_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="RiskAdj 1M Mth",
                            current_stocks=current_risk_adj_mom_1m_monthly_stocks,
                            new_stocks=new_stocks,
                            positions=risk_adj_mom_1m_monthly_positions,
                            cash=risk_adj_mom_1m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not risk_adj_mom_1m_monthly_initialized)
                        strategies_rebalanced_today['RiskAdj 1M Mth'] = rebalanced_flag
                        risk_adj_mom_1m_monthly_transaction_costs += rc
                        risk_adj_mom_1m_monthly_initialized = True
                        risk_adj_mom_1m_monthly_last_month = current_date.month

                except Exception as e:
                    print(f"   ⚠️ Risk-Adj Mom 1M Monthly error: {e}")

        # AI ELITE MONTHLY STRATEGY (same ML scoring, retrain + rebalance start of month only)
        if config.ENABLE_AI_ELITE_MONTHLY:
            should_act_ai_elite_monthly = (not ai_elite_monthly_initialized) or is_first_trading_day_of_month
            if should_act_ai_elite_monthly:
                try:
                    from shared_strategies import select_ai_elite_with_training

                    print(f"   📊 AI Elite Monthly: {'Initializing' if not ai_elite_monthly_initialized else 'Start-of-month'} ({current_date.strftime('%b %Y')})")

                    # Use shared function (handles load/train/select) with monthly model path
                    # AI_ELITE_FORCE_FRESH_TRAIN controls whether to load existing model (incremental) or fresh train
                    new_stocks, ai_elite_monthly_models = select_ai_elite_with_training(
                        all_tickers=initial_top_tickers,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Get 12 candidates for buffer
                        ai_elite_models=ai_elite_monthly_models,
                        force_train=AI_ELITE_FORCE_FRESH_TRAIN,
                        model_path_suffix="_monthly"
                    )

                    if new_stocks:
                        print(f"   📊 AI Elite Monthly Day {day_count}: {new_stocks}")
                        ai_elite_monthly_positions, ai_elite_monthly_cash, current_ai_elite_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name="AI Elite Mth",
                            current_stocks=current_ai_elite_monthly_stocks,
                            new_stocks=new_stocks,
                            positions=ai_elite_monthly_positions,
                            cash=ai_elite_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Keep if in top 12
                            force_rebalance=not ai_elite_monthly_initialized)
                        strategies_rebalanced_today['AI Elite Mth'] = rebalanced_flag
                        ai_elite_monthly_transaction_costs += rc
                        ai_elite_monthly_initialized = True
                        ai_elite_monthly_last_month = current_date.month

                except Exception as e:
                    print(f"   ⚠️ AI Elite Monthly error: {e}")

        # AI ELITE FILTERED STRATEGY (Risk-Adj Mom 3M pre-filter + AI Elite re-rank)
        if config.ENABLE_AI_ELITE_FILTERED:
            try:
                from ai_elite_filtered_strategy import select_ai_elite_filtered_stocks

                # Use the same AI Elite models (shared with daily AI Elite)
                new_ai_elite_filtered_stocks = select_ai_elite_filtered_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Get 12 candidates for buffer
                    per_ticker_models=ai_elite_models,  # Reuse AI Elite models
                    pre_filter_n=PORTFOLIO_SIZE * 2  # Pre-filter to 2x final count
                )

                if new_ai_elite_filtered_stocks:
                    print(f"   📊 AI Elite Filtered Day {day_count}: {new_ai_elite_filtered_stocks}")
                    ai_elite_filtered_positions, ai_elite_filtered_cash, current_ai_elite_filtered_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="AI Elite Flt",
                        current_stocks=current_ai_elite_filtered_stocks,
                        new_stocks=new_ai_elite_filtered_stocks,
                        positions=ai_elite_filtered_positions,
                        cash=ai_elite_filtered_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Keep if in top 12
                        force_rebalance=not ai_elite_filtered_initialized
                    )
                    strategies_rebalanced_today['AI Elite Flt'] = rebalanced_flag
                    ai_elite_filtered_transaction_costs += rc
                    ai_elite_filtered_initialized = True

            except Exception as e:
                print(f"   ⚠️ AI Elite Filtered error: {e}")

        # AI ELITE MARKET-UP STRATEGY (AI Elite only when market is up)
        if config.ENABLE_AI_ELITE_MARKET_UP:
            try:
                from ai_elite_market_up_strategy import select_ai_elite_market_up_stocks

                new_ai_elite_market_up_stocks = select_ai_elite_market_up_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Get 12 candidates for buffer=2
                    per_ticker_models=ai_elite_models,  # Reuse AI Elite models
                )

                if new_ai_elite_market_up_stocks:
                    # Show portfolio before rebalancing
                    print(f"   📊 AI Elite Market-Up Day {day_count} - Before: {current_ai_elite_market_up_stocks}")
                    # Show candidates for rebalancing
                    print(f"   📊 AI Elite Market-Up Day {day_count} - Candidates: {new_ai_elite_market_up_stocks}")

                    # Store previous positions before rebalancing for tracking
                    prev_ai_elite_market_up_stocks = current_ai_elite_market_up_stocks.copy()
                    ai_elite_market_up_positions, ai_elite_market_up_cash, current_ai_elite_market_up_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="AI Elite Mkt-Up",
                        current_stocks=current_ai_elite_market_up_stocks,
                        new_stocks=new_ai_elite_market_up_stocks,
                        positions=ai_elite_market_up_positions,
                        cash=ai_elite_market_up_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Keep if in top 12
                        force_rebalance=not ai_elite_market_up_initialized
                    )
                    # Show final portfolio after rebalancing
                    print(f"   📊 AI Elite Market-Up Day {day_count} - After: {current_ai_elite_market_up_stocks}")
                    strategies_rebalanced_today['AI Elite Mkt-Up'] = rebalanced_flag
                    ai_elite_market_up_transaction_costs += rc
                    ai_elite_market_up_initialized = True
                    # Check if rebalancing actually changed positions
                    ai_elite_market_up_rebalanced_today = set(current_ai_elite_market_up_stocks) != set(prev_ai_elite_market_up_stocks)

            except Exception as e:
                print(f"   ⚠️ AI Elite Market-Up error: {e}")

        # AI CHAMPION / AI REGIME selectors are evaluated later in the day after
        # the full day-end strategy snapshot is available.

        # UNIVERSAL MODEL STRATEGY (single ML model for all tickers)
        if config.ENABLE_UNIVERSAL_MODEL:
            try:
                from universal_model_strategy import UniversalModelStrategy, select_universal_model_stocks

                # Initialize strategy on first use
                if universal_model_strategy is None:
                    universal_model_strategy = UniversalModelStrategy(retrain_days=1, min_samples=200)
                    # Try to load existing model from disk (for faster startup)
                    universal_model_strategy.load_model()

                universal_model_strategy.increment_day()

                # Select stocks using universal model
                new_universal_model_stocks = select_universal_model_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,
                    PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Get 12 candidates for buffer
                    universal_model_strategy,
                    business_days,
                    day_count
                )

                if new_universal_model_stocks:
                    print(f"   📊 Universal Model Day {day_count}: {new_universal_model_stocks}")
                    universal_model_positions, universal_model_cash, current_universal_model_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Universal Model",
                        current_stocks=current_universal_model_stocks,
                        new_stocks=new_universal_model_stocks,
                        positions=universal_model_positions,
                        cash=universal_model_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,  # Keep if in top 12
                        force_rebalance=not universal_model_initialized
                    )
                    strategies_rebalanced_today['Universal Model'] = rebalanced_flag
                    universal_model_transaction_costs += rc
                    universal_model_initialized = True

            except Exception as e:
                print(f"   ⚠️ Universal Model error: {e}")

        # SAVGOL TREND STRATEGY (pooled ML using local polynomial trend features)
        if config.ENABLE_SAVGOL_TREND:
            try:
                from savgol_trend_strategy import SavgolTrendStrategy, select_savgol_trend_stocks

                if savgol_trend_strategy is None:
                    savgol_trend_strategy = SavgolTrendStrategy(
                        retrain_days=config.SAVGOL_TREND_RETRAIN_DAYS,
                        min_samples=config.SAVGOL_TREND_MIN_SAMPLES,
                        lookback_days=config.SAVGOL_TREND_LOOKBACK_DAYS,
                        forward_days=config.SAVGOL_TREND_FORWARD_DAYS,
                    )
                    savgol_trend_strategy.load_model()

                savgol_trend_strategy.increment_day()

                new_savgol_trend_stocks = select_savgol_trend_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,
                    PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                    savgol_trend_strategy,
                    business_days,
                    day_count,
                    current_savgol_trend_stocks,
                )

                if new_savgol_trend_stocks:
                    print(f"   📊 SavGol Trend Day {day_count}: {new_savgol_trend_stocks}")
                    savgol_trend_positions, savgol_trend_cash, current_savgol_trend_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="SavGol Trend",
                        current_stocks=current_savgol_trend_stocks,
                        new_stocks=new_savgol_trend_stocks,
                        positions=savgol_trend_positions,
                        cash=savgol_trend_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                        force_rebalance=not savgol_trend_initialized
                    )
                    strategies_rebalanced_today['SavGol Trend'] = rebalanced_flag
                    savgol_trend_transaction_costs += rc
                    savgol_trend_initialized = True

            except Exception as e:
                print(f"   ⚠️ SavGol Trend error: {e}")

        # META-STRATEGY SELECTORS are evaluated later in the day after the shared
        # day-end strategy snapshot is fully populated. This keeps them aligned
        # with the same completed source universe used by Voting Ensemble.

        # BOLLINGER BANDS STRATEGIES
        # BB Mean Reversion
        if config.ENABLE_BB_MEAN_REVERSION:
            try:
                from bollinger_bands_strategy import select_bb_mean_reversion_stocks
                new_bb_mean_reversion_stocks = select_bb_mean_reversion_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                if new_bb_mean_reversion_stocks:
                    print(f"   📊 BB Mean Reversion Day {day_count}: {new_bb_mean_reversion_stocks}")
                    bb_mean_reversion_positions, bb_mean_reversion_cash, current_bb_mean_reversion_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="BB Mean Reversion",
                        current_stocks=current_bb_mean_reversion_stocks,
                        new_stocks=new_bb_mean_reversion_stocks,
                        positions=bb_mean_reversion_positions,
                        cash=bb_mean_reversion_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not bb_mean_reversion_initialized
                    )
                    bb_mean_reversion_transaction_costs += rc
                    bb_mean_reversion_initialized = True
            except Exception as e:
                print(f"   ⚠️ BB Mean Reversion error: {e}")

        # BB Breakout
        if config.ENABLE_BB_BREAKOUT:
            try:
                from bollinger_bands_strategy import select_bb_breakout_stocks
                new_bb_breakout_stocks = select_bb_breakout_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                if new_bb_breakout_stocks:
                    print(f"   📊 BB Breakout Day {day_count}: {new_bb_breakout_stocks}")
                    bb_breakout_positions, bb_breakout_cash, current_bb_breakout_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="BB Breakout",
                        current_stocks=current_bb_breakout_stocks,
                        new_stocks=new_bb_breakout_stocks,
                        positions=bb_breakout_positions,
                        cash=bb_breakout_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not bb_breakout_initialized
                    )
                    bb_breakout_transaction_costs += rc
                    bb_breakout_initialized = True
            except Exception as e:
                print(f"   ⚠️ BB Breakout error: {e}")

        # BB Squeeze Breakout
        if config.ENABLE_BB_SQUEEZE_BREAKOUT:
            try:
                from bollinger_bands_strategy import select_bb_squeeze_breakout_stocks
                new_bb_squeeze_breakout_stocks = select_bb_squeeze_breakout_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                if new_bb_squeeze_breakout_stocks:
                    print(f"   📊 BB Squeeze Breakout Day {day_count}: {new_bb_squeeze_breakout_stocks}")
                    bb_squeeze_breakout_positions, bb_squeeze_breakout_cash, current_bb_squeeze_breakout_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="BB Squeeze Breakout",
                        current_stocks=current_bb_squeeze_breakout_stocks,
                        new_stocks=new_bb_squeeze_breakout_stocks,
                        positions=bb_squeeze_breakout_positions,
                        cash=bb_squeeze_breakout_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not bb_squeeze_breakout_initialized
                    )
                    bb_squeeze_breakout_transaction_costs += rc
                    bb_squeeze_breakout_initialized = True
            except Exception as e:
                print(f"   ⚠️ BB Squeeze Breakout error: {e}")

        # BB RSI Combo
        if config.ENABLE_BB_RSI_COMBO:
            try:
                from bollinger_bands_strategy import select_bb_rsi_combo_stocks
                new_bb_rsi_combo_stocks = select_bb_rsi_combo_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                if new_bb_rsi_combo_stocks:
                    print(f"   📊 BB RSI Combo Day {day_count}: {new_bb_rsi_combo_stocks}")
                    bb_rsi_combo_positions, bb_rsi_combo_cash, current_bb_rsi_combo_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="BB RSI Combo",
                        current_stocks=current_bb_rsi_combo_stocks,
                        new_stocks=new_bb_rsi_combo_stocks,
                        positions=bb_rsi_combo_positions,
                        cash=bb_rsi_combo_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not bb_rsi_combo_initialized
                    )
                    bb_rsi_combo_transaction_costs += rc
                    bb_rsi_combo_initialized = True
            except Exception as e:
                print(f"   ⚠️ BB RSI Combo error: {e}")

        # TREND BREAKOUT (no ATR selling, uses smart_rebalance)
        if config.ENABLE_TREND_BREAKOUT:
            try:
                from new_strategies import select_trend_breakout_stocks
                new_trend_breakout_stocks = select_trend_breakout_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                if new_trend_breakout_stocks:
                    print(f"   📊 Trend Breakout Day {day_count}: {new_trend_breakout_stocks}")
                    trend_breakout_positions, trend_breakout_cash, current_trend_breakout_stocks, rc, _ = _smart_rebalance_portfolio(
                        strategy_name="Trend Breakout",
                        current_stocks=current_trend_breakout_stocks,
                        new_stocks=new_trend_breakout_stocks,
                        positions=trend_breakout_positions,
                        cash=trend_breakout_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not trend_breakout_initialized,
                        use_atr_trailing_stop=TREND_BREAKOUT_USE_ATR_STOP,
                        atr_multiplier=TREND_BREAKOUT_ATR_MULTIPLIER
                    )
                    trend_breakout_transaction_costs += rc
                    trend_breakout_initialized = True
            except Exception as e:
                print(f"   ⚠️ Trend Breakout error: {e}")

        # Update DYNAMIC BH 1Y portfolio value daily (skip if disabled)
        dynamic_bh_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_1Y:
          # Iterate over actual positions, not just current_dynamic_bh_stocks list
          print(f"   🔧 DEBUG: Dyn BH 1Y - iterating over {len(dynamic_bh_positions)} positions")
          for ticker in list(dynamic_bh_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_positions[ticker]['shares'] * current_price
                                    dynamic_bh_positions[ticker]['value'] = position_value
                                    dynamic_bh_invested_value += position_value
                                else:
                                    dynamic_bh_invested_value += dynamic_bh_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_invested_value += dynamic_bh_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH position for {ticker}: {e}")

        dynamic_bh_portfolio_value = dynamic_bh_invested_value + dynamic_bh_cash
        dynamic_bh_portfolio_history.append(dynamic_bh_portfolio_value)

        # Update DYNAMIC BH 1Y + VOLATILITY FILTER portfolio value daily (skip if disabled)
        dynamic_bh_1y_vol_filter_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_1Y_VOL_FILTER:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1y_vol_filter_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1y_vol_filter_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1y_vol_filter_positions[ticker]['value'] = position_value
                                    dynamic_bh_1y_vol_filter_invested_value += position_value
                                else:
                                    dynamic_bh_1y_vol_filter_invested_value += dynamic_bh_1y_vol_filter_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1y_vol_filter_invested_value += dynamic_bh_1y_vol_filter_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1Y+Vol position for {ticker}: {e}")

        dynamic_bh_1y_vol_filter_portfolio_value = dynamic_bh_1y_vol_filter_invested_value + dynamic_bh_1y_vol_filter_cash
        dynamic_bh_1y_vol_filter_portfolio_history.append(dynamic_bh_1y_vol_filter_portfolio_value)

        # Update DYNAMIC BH 1Y + TRAILING STOP portfolio value daily (skip if disabled)
        dynamic_bh_1y_trailing_stop_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1y_trailing_stop_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1y_trailing_stop_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1y_trailing_stop_positions[ticker]['value'] = position_value
                                    dynamic_bh_1y_trailing_stop_invested_value += position_value
                                else:
                                    dynamic_bh_1y_trailing_stop_invested_value += dynamic_bh_1y_trailing_stop_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1y_trailing_stop_invested_value += dynamic_bh_1y_trailing_stop_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1Y+TS position for {ticker}: {e}")

        dynamic_bh_1y_trailing_stop_portfolio_value = dynamic_bh_1y_trailing_stop_invested_value + dynamic_bh_1y_trailing_stop_cash
        dynamic_bh_1y_trailing_stop_portfolio_history.append(dynamic_bh_1y_trailing_stop_portfolio_value)

        # Update SECTOR ROTATION portfolio value daily (skip if disabled)
        if config.ENABLE_SECTOR_ROTATION:
            sector_rotation_invested_value = 0.0
            for etf in current_sector_rotation_etfs:
                if etf in sector_rotation_positions:
                    try:
                        etf_data = ticker_data_grouped.get(etf)
                        if etf_data and not etf_data.empty:
                            current_price_data = etf_data.loc[etf_data.index == current_date]
                            if len(current_price_data) > 0:
                                valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = sector_rotation_positions[etf]['shares'] * current_price
                                    sector_rotation_positions[etf]['value'] = position_value
                                    sector_rotation_invested_value += position_value
                                else:
                                    sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                            else:
                                sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                        else:
                            sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                    except Exception as e:
                        sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)

            sector_rotation_portfolio_value = sector_rotation_invested_value + sector_rotation_cash
            sector_rotation_portfolio_history.append(sector_rotation_portfolio_value)

        # Update DYNAMIC BH 6-MONTH portfolio value daily (skip if disabled)
        dynamic_bh_6m_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_6M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_6m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_6m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_6m_positions[ticker]['value'] = position_value
                                    dynamic_bh_6m_invested_value += position_value
                                else:
                                    dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)
                except Exception:
                    dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)

        # Update DYNAMIC BH 3-MONTH portfolio value daily (skip if disabled)
        dynamic_bh_3m_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_3M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_3m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_3m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_3m_positions[ticker]['value'] = position_value
                                    dynamic_bh_3m_invested_value += position_value
                                else:
                                    dynamic_bh_3m_invested_value += dynamic_bh_3m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_3m_invested_value += dynamic_bh_3m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 3M position for {ticker}: {e}")

        dynamic_bh_6m_portfolio_value = dynamic_bh_6m_invested_value + dynamic_bh_6m_cash
        dynamic_bh_6m_portfolio_history.append(dynamic_bh_6m_portfolio_value)

        dynamic_bh_3m_portfolio_value = dynamic_bh_3m_invested_value + dynamic_bh_3m_cash
        dynamic_bh_3m_portfolio_history.append(dynamic_bh_3m_portfolio_value)

        # Update dynamic BH 1-month portfolio value (skip if disabled)
        dynamic_bh_1m_invested_value = 0.0
        if config.ENABLE_DYNAMIC_BH_1M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1m_positions[ticker]['value'] = position_value
                                    dynamic_bh_1m_invested_value += position_value
                                else:
                                    dynamic_bh_1m_invested_value += dynamic_bh_1m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1m_invested_value += dynamic_bh_1m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1M position for {ticker}: {e}")

        dynamic_bh_1m_portfolio_value = dynamic_bh_1m_invested_value + dynamic_bh_1m_cash
        dynamic_bh_1m_portfolio_history.append(dynamic_bh_1m_portfolio_value)

        # Update RISK-ADJUSTED MOMENTUM portfolio value daily (skip if disabled)
        risk_adj_mom_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(risk_adj_mom_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = risk_adj_mom_positions[ticker]['shares'] * current_price
                                    risk_adj_mom_positions[ticker]['value'] = position_value
                                    risk_adj_mom_invested_value += position_value
                                else:
                                    risk_adj_mom_invested_value += risk_adj_mom_positions[ticker].get('value', 0.0)
                            else:
                                risk_adj_mom_invested_value += risk_adj_mom_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating risk-adjusted momentum position for {ticker}: {e}")

        risk_adj_mom_portfolio_value = risk_adj_mom_invested_value + risk_adj_mom_cash
        risk_adj_mom_portfolio_history.append(risk_adj_mom_portfolio_value)

        # Update RISK-ADJ MOMENTUM SENTIMENT portfolio value daily (skip if disabled)
        if config.ENABLE_RISK_ADJ_MOM_SENTIMENT:
            risk_adj_mom_sentiment_invested_value = 0.0
            for ticker in list(risk_adj_mom_sentiment_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = risk_adj_mom_sentiment_positions[ticker]['shares'] * current_price
                                    risk_adj_mom_sentiment_positions[ticker]['value'] = position_value
                                    risk_adj_mom_sentiment_invested_value += position_value
                                else:
                                    risk_adj_mom_sentiment_invested_value += risk_adj_mom_sentiment_positions[ticker].get('value', 0.0)
                            else:
                                risk_adj_mom_sentiment_invested_value += risk_adj_mom_sentiment_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating risk adj mom sentiment position for {ticker}: {e}")

            risk_adj_mom_sentiment_portfolio_value = risk_adj_mom_sentiment_invested_value + risk_adj_mom_sentiment_cash
            risk_adj_mom_sentiment_portfolio_history.append(risk_adj_mom_sentiment_portfolio_value)

        # Update 3M/1Y RATIO portfolio value daily (skip if disabled)
        if config.ENABLE_3M_1Y_RATIO:
            ratio_3m_1y_invested_value = 0.0
            for ticker in list(ratio_3m_1y_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ratio_3m_1y_positions[ticker]['shares']
                            position_value = shares * current_price
                            ratio_3m_1y_positions[ticker]['value'] = position_value
                            ratio_3m_1y_invested_value += position_value
                        else:
                            ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)
                    else:
                        ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating 3M/1Y ratio position for {ticker}: {e}")
                    ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)

            ratio_3m_1y_portfolio_value = ratio_3m_1y_invested_value + ratio_3m_1y_cash
            ratio_3m_1y_portfolio_history.append(ratio_3m_1y_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID portfolio value daily
        momentum_volatility_hybrid_invested_value = 0.0
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(momentum_volatility_hybrid_positions.keys()):
                    try:
                        # Try to get current price from grouped data first
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        print(f"   ⚠️ Error updating Momentum-Volatility Hybrid position for {ticker}: {e}")
                        # Use previous value as fallback
                        momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_portfolio_value = momentum_volatility_hybrid_invested_value + momentum_volatility_hybrid_cash
        momentum_volatility_hybrid_portfolio_history.append(momentum_volatility_hybrid_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 6M portfolio value daily
        momentum_volatility_hybrid_6m_invested_value = 0.0
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M:
            for ticker in list(momentum_volatility_hybrid_6m_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_6m_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_6m_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_6m_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_6m_portfolio_value = momentum_volatility_hybrid_6m_invested_value + momentum_volatility_hybrid_6m_cash
        momentum_volatility_hybrid_6m_portfolio_history.append(momentum_volatility_hybrid_6m_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 1Y portfolio value daily
        momentum_volatility_hybrid_1y_invested_value = 0.0
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y:
            for ticker in list(momentum_volatility_hybrid_1y_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_1y_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_1y_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_1y_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_1y_portfolio_value = momentum_volatility_hybrid_1y_invested_value + momentum_volatility_hybrid_1y_cash
        momentum_volatility_hybrid_1y_portfolio_history.append(momentum_volatility_hybrid_1y_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 1Y/3M portfolio value daily
        momentum_volatility_hybrid_1y3m_invested_value = 0.0
        if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M:
            for ticker in list(momentum_volatility_hybrid_1y3m_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_1y3m_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_1y3m_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_1y3m_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_1y3m_portfolio_value = momentum_volatility_hybrid_1y3m_invested_value + momentum_volatility_hybrid_1y3m_cash
        momentum_volatility_hybrid_1y3m_portfolio_history.append(momentum_volatility_hybrid_1y3m_portfolio_value)

        # Update PRICE ACCELERATION portfolio value daily
        price_acceleration_invested_value = 0.0
        if config.ENABLE_PRICE_ACCELERATION:
            for ticker in list(price_acceleration_positions.keys()):
                try:
                    # Get ticker data using the same method as other strategies
                    ticker_data = ticker_data_grouped.get(ticker) if isinstance(ticker_data_grouped, dict) else None
                    if ticker_data is None and hasattr(ticker_data_grouped, 'get_group'):
                        try:
                            ticker_data = ticker_data_grouped.get_group(ticker)
                        except KeyError:
                            ticker_data = None

                    if ticker_data is not None and not ticker_data.empty:
                        current_price = _last_valid_close_up_to(ticker_data, current_date)
                        if current_price is not None:
                            shares = price_acceleration_positions[ticker]['shares']
                            position_value = shares * current_price
                            price_acceleration_positions[ticker]['value'] = position_value
                            price_acceleration_invested_value += position_value
                        else:
                            price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)
                    else:
                        price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Price Acceleration position for {ticker}: {e}")
                    price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)

        price_acceleration_portfolio_value = price_acceleration_invested_value + price_acceleration_cash
        price_acceleration_portfolio_history.append(price_acceleration_portfolio_value)

        # Update 1Y/3M RATIO portfolio value daily
        ratio_1y_3m_invested_value = 0.0
        for ticker in list(ratio_1y_3m_positions.keys()):
            try:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    current_price = _last_valid_close_up_to(ticker_df, current_date)
                    if current_price is not None:
                        shares = ratio_1y_3m_positions[ticker]['shares']
                        position_value = shares * current_price
                        ratio_1y_3m_positions[ticker]['value'] = position_value
                        ratio_1y_3m_invested_value += position_value
                    else:
                        ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)
                else:
                    ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)
            except Exception as e:
                print(f"   ⚠️ Error updating 1Y/3M ratio position for {ticker}: {e}")
                ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)

        ratio_1y_3m_portfolio_value = ratio_1y_3m_invested_value + ratio_1y_3m_cash
        ratio_1y_3m_portfolio_history.append(ratio_1y_3m_portfolio_value)

        # Update 1M/3M RATIO portfolio value daily
        if config.ENABLE_1M_3M_RATIO:
            ratio_1m_3m_invested_value = 0.0
            for ticker in list(ratio_1m_3m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ratio_1m_3m_positions[ticker]['shares']
                            position_value = shares * current_price
                            ratio_1m_3m_positions[ticker]['value'] = position_value
                            ratio_1m_3m_invested_value += position_value
                        else:
                            ratio_1m_3m_invested_value += ratio_1m_3m_positions[ticker].get('value', 0.0)
                    else:
                        ratio_1m_3m_invested_value += ratio_1m_3m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating 1M/3M ratio position for {ticker}: {e}")
                    ratio_1m_3m_invested_value += ratio_1m_3m_positions[ticker].get('value', 0.0)

            ratio_1m_3m_portfolio_value = ratio_1m_3m_invested_value + ratio_1m_3m_cash
            ratio_1m_3m_portfolio_history.append(ratio_1m_3m_portfolio_value)

        # Update TURNAROUND portfolio value daily
        turnaround_invested_value = 0.0
        for ticker in list(turnaround_positions.keys()):
            try:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    current_price = _last_valid_close_up_to(ticker_df, current_date)
                    if current_price is not None:
                        shares = turnaround_positions[ticker]['shares']
                        position_value = shares * current_price
                        turnaround_positions[ticker]['value'] = position_value
                        turnaround_invested_value += position_value
                    else:
                        turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)
                else:
                    turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)
            except Exception as e:
                print(f"   ⚠️ Error updating turnaround position for {ticker}: {e}")
                turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)

        turnaround_portfolio_value = turnaround_invested_value + turnaround_cash
        turnaround_portfolio_history.append(turnaround_portfolio_value)

        # Update ADAPTIVE ENSEMBLE portfolio value daily (skip if disabled)
        if config.ENABLE_ADAPTIVE_STRATEGY:
            adaptive_ensemble_invested_value = 0.0
            for ticker in list(adaptive_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = adaptive_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            adaptive_ensemble_positions[ticker]['value'] = position_value
                            adaptive_ensemble_invested_value += position_value
                        else:
                            adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating adaptive ensemble position for {ticker}: {e}")
                    adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)

            adaptive_ensemble_portfolio_value = adaptive_ensemble_invested_value + adaptive_ensemble_cash
            adaptive_ensemble_portfolio_history.append(adaptive_ensemble_portfolio_value)

        # Update VOLATILITY ENSEMBLE portfolio value daily (skip if disabled)
        volatility_ensemble_invested_value = 0.0
        if config.ENABLE_VOLATILITY_ENSEMBLE:
            for ticker in list(volatility_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = volatility_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            volatility_ensemble_positions[ticker]['value'] = position_value
                            volatility_ensemble_invested_value += position_value
                        else:
                            volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating volatility ensemble position for {ticker}: {e}")
                    volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)

        volatility_ensemble_portfolio_value = volatility_ensemble_invested_value + volatility_ensemble_cash
        volatility_ensemble_portfolio_history.append(volatility_ensemble_portfolio_value)

        # Update ENHANCED VOLATILITY TRADER portfolio value daily (skip if disabled)
        enhanced_volatility_invested_value = 0.0
        if config.ENABLE_ENHANCED_VOLATILITY:
            for ticker in list(enhanced_volatility_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = enhanced_volatility_positions[ticker]['shares']
                            position_value = shares * current_price
                            enhanced_volatility_positions[ticker]['value'] = position_value
                            enhanced_volatility_invested_value += position_value
                        else:
                            enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)
                    else:
                        enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating enhanced volatility position for {ticker}: {e}")
                    enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)

        enhanced_volatility_portfolio_value = enhanced_volatility_invested_value + enhanced_volatility_cash
        enhanced_volatility_portfolio_history.append(enhanced_volatility_portfolio_value)

        # DEBUG: Log Enhanced Volatility state (only when enabled)
        if config.ENABLE_ENHANCED_VOLATILITY and day_count <= 5:
            print(f"   🔧 DEBUG: Enhanced Volatility Day {day_count} - Cash: ${enhanced_volatility_cash:.2f}, Invested: ${enhanced_volatility_invested_value:.2f}, Total: ${enhanced_volatility_portfolio_value:.2f}, Positions: {len(enhanced_volatility_positions)}")

        # Update AI VOLATILITY ENSEMBLE portfolio value daily (skip if disabled)
        ai_volatility_ensemble_invested_value = 0.0
        if config.ENABLE_AI_VOLATILITY_ENSEMBLE:
            for ticker in list(ai_volatility_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_volatility_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_volatility_ensemble_positions[ticker]['value'] = position_value
                            ai_volatility_ensemble_invested_value += position_value
                        else:
                            ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating AI volatility ensemble position for {ticker}: {e}")
                    ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)

        ai_volatility_ensemble_portfolio_value = ai_volatility_ensemble_invested_value + ai_volatility_ensemble_cash
        ai_volatility_ensemble_portfolio_history.append(ai_volatility_ensemble_portfolio_value)

        # Update MULTI-HORIZON ENSEMBLE portfolio value daily (skip if disabled)
        multi_tf_ensemble_invested_value = 0.0
        if config.ENABLE_MULTI_TIMEFRAME_ENSEMBLE:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(multi_tf_ensemble_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = multi_tf_ensemble_positions[ticker]['shares'] * current_price
                                    multi_tf_ensemble_positions[ticker]['value'] = position_value
                                    multi_tf_ensemble_invested_value += position_value
                                else:
                                    multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)
                            else:
                                multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating multi-timeframe ensemble position for {ticker}: {e}")
                    multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)

        multi_tf_ensemble_portfolio_value = multi_tf_ensemble_invested_value + multi_tf_ensemble_cash
        multi_tf_ensemble_portfolio_history.append(multi_tf_ensemble_portfolio_value)

        # Update MULTI-HORIZON INTRADAY portfolio value daily (skip if disabled)
        multi_tf_intraday_ensemble_invested_value = 0.0
        if config.ENABLE_MULTI_TIMEFRAME_INTRADAY_ENSEMBLE:
            for ticker in list(multi_tf_intraday_ensemble_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = multi_tf_intraday_ensemble_positions[ticker]['shares'] * current_price
                                    multi_tf_intraday_ensemble_positions[ticker]['value'] = position_value
                                    multi_tf_intraday_ensemble_invested_value += position_value
                                else:
                                    multi_tf_intraday_ensemble_invested_value += multi_tf_intraday_ensemble_positions[ticker].get('value', 0.0)
                            else:
                                multi_tf_intraday_ensemble_invested_value += multi_tf_intraday_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating multi-timeframe intraday ensemble position for {ticker}: {e}")
                    multi_tf_intraday_ensemble_invested_value += multi_tf_intraday_ensemble_positions[ticker].get('value', 0.0)

        multi_tf_intraday_ensemble_portfolio_value = multi_tf_intraday_ensemble_invested_value + multi_tf_intraday_ensemble_cash
        multi_tf_intraday_ensemble_portfolio_history.append(multi_tf_intraday_ensemble_portfolio_value)

        # Update CORRELATION ENSEMBLE portfolio value daily (skip if disabled)
        correlation_ensemble_invested_value = 0.0
        if config.ENABLE_CORRELATION_ENSEMBLE:
            for ticker in list(correlation_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = correlation_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            correlation_ensemble_positions[ticker]['value'] = position_value
                            correlation_ensemble_invested_value += position_value
                        else:
                            correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating correlation ensemble position for {ticker}: {e}")
                    correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)

        correlation_ensemble_portfolio_value = correlation_ensemble_invested_value + correlation_ensemble_cash
        correlation_ensemble_portfolio_history.append(correlation_ensemble_portfolio_value)

        # Update DYNAMIC POOL portfolio value daily (skip if disabled)
        if config.ENABLE_DYNAMIC_POOL:
            dynamic_pool_invested_value = 0.0
            for ticker in list(dynamic_pool_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = dynamic_pool_positions[ticker]['shares']
                            position_value = shares * current_price
                            dynamic_pool_positions[ticker]['value'] = position_value
                            dynamic_pool_invested_value += position_value
                        else:
                            dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)
                    else:
                        dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic pool position for {ticker}: {e}")
                    dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)

            dynamic_pool_portfolio_value = dynamic_pool_invested_value + dynamic_pool_cash
            dynamic_pool_portfolio_history.append(dynamic_pool_portfolio_value)

        # Voting Ensemble is updated after all other day-end strategy values are known.

        # Update MOMENTUM ACCELERATION portfolio value daily
        mom_accel_invested_value = 0.0
        if config.ENABLE_MOMENTUM_ACCELERATION:
            for ticker in list(mom_accel_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = mom_accel_positions[ticker]['shares']
                            position_value = shares * current_price
                            mom_accel_positions[ticker]['value'] = position_value
                            mom_accel_invested_value += position_value
                        else:
                            mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
                    else:
                        mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
                except Exception:
                    mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
        mom_accel_portfolio_value = mom_accel_invested_value + mom_accel_cash
        mom_accel_portfolio_history.append(mom_accel_portfolio_value)

        # Update CONCENTRATED 3M portfolio value daily
        concentrated_3m_invested_value = 0.0
        if config.ENABLE_CONCENTRATED_3M:
            for ticker in list(concentrated_3m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = concentrated_3m_positions[ticker]['shares']
                            position_value = shares * current_price
                            concentrated_3m_positions[ticker]['value'] = position_value
                            concentrated_3m_invested_value += position_value
                        else:
                            concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
                    else:
                        concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
                except Exception:
                    concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
        concentrated_3m_portfolio_value = concentrated_3m_invested_value + concentrated_3m_cash
        concentrated_3m_portfolio_history.append(concentrated_3m_portfolio_value)

        # Update DUAL MOMENTUM portfolio value daily
        dual_mom_invested_value = 0.0
        if config.ENABLE_DUAL_MOMENTUM:
            for ticker in list(dual_mom_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = dual_mom_positions[ticker]['shares']
                            position_value = shares * current_price
                            dual_mom_positions[ticker]['value'] = position_value
                            dual_mom_invested_value += position_value
                        else:
                            dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
                    else:
                        dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
                except Exception:
                    dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
        dual_mom_portfolio_value = dual_mom_invested_value + dual_mom_cash
        dual_mom_portfolio_history.append(dual_mom_portfolio_value)

        # Update TREND FOLLOWING ATR portfolio value daily
        trend_atr_invested_value = 0.0
        if config.ENABLE_TREND_FOLLOWING_ATR:
            for ticker in list(trend_atr_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = trend_atr_positions[ticker]['shares']
                            position_value = shares * current_price
                            trend_atr_positions[ticker]['value'] = position_value
                            trend_atr_invested_value += position_value
                        else:
                            trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
                    else:
                        trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
                except Exception:
                    trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
        trend_atr_portfolio_value = trend_atr_invested_value + trend_atr_cash
        trend_atr_portfolio_history.append(trend_atr_portfolio_value)

        # Update ELITE HYBRID portfolio value daily
        elite_hybrid_invested_value = 0.0
        if config.ENABLE_ELITE_HYBRID:
            for ticker in list(elite_hybrid_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = elite_hybrid_positions[ticker]['shares']
                            position_value = shares * current_price
                            elite_hybrid_positions[ticker]['value'] = position_value
                            elite_hybrid_invested_value += position_value
                        else:
                            elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
                    else:
                        elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
                except Exception:
                    elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
        elite_hybrid_portfolio_value = elite_hybrid_invested_value + elite_hybrid_cash
        elite_hybrid_portfolio_history.append(elite_hybrid_portfolio_value)

        # Update ELITE RISK portfolio value daily
        elite_risk_invested_value = 0.0
        if config.ENABLE_ELITE_RISK:
            for ticker in list(elite_risk_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = elite_risk_positions[ticker]['shares']
                            position_value = shares * current_price
                            elite_risk_positions[ticker]['value'] = position_value
                            elite_risk_invested_value += position_value
                        else:
                            elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
                    else:
                        elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
                except Exception:
                    elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
        elite_risk_portfolio_value = elite_risk_invested_value + elite_risk_cash
        elite_risk_portfolio_history.append(elite_risk_portfolio_value)

        # Update RISK-ADJ MOM 6M portfolio value daily
        risk_adj_mom_6m_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_6M:
            for ticker in list(risk_adj_mom_6m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_6m_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_6m_positions[ticker]['value'] = position_value
                            risk_adj_mom_6m_invested_value += position_value
                        else:
                            risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
        risk_adj_mom_6m_portfolio_value = risk_adj_mom_6m_invested_value + risk_adj_mom_6m_cash
        risk_adj_mom_6m_portfolio_history.append(risk_adj_mom_6m_portfolio_value)

        # Update RISK-ADJ MOM 6M MONTHLY portfolio value daily
        risk_adj_mom_6m_monthly_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_6M_MONTHLY:
            for ticker in list(risk_adj_mom_6m_monthly_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_6m_monthly_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_6m_monthly_positions[ticker]['value'] = position_value
                            risk_adj_mom_6m_monthly_invested_value += position_value
                        else:
                            risk_adj_mom_6m_monthly_invested_value += risk_adj_mom_6m_monthly_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_6m_monthly_invested_value += risk_adj_mom_6m_monthly_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_6m_monthly_invested_value += risk_adj_mom_6m_monthly_positions[ticker].get('value', 0.0)
        risk_adj_mom_6m_monthly_portfolio_value = risk_adj_mom_6m_monthly_invested_value + risk_adj_mom_6m_monthly_cash
        risk_adj_mom_6m_monthly_portfolio_history.append(risk_adj_mom_6m_monthly_portfolio_value)

        # Update RISK-ADJ MOM 3M portfolio value daily
        risk_adj_mom_3m_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_3M:
            for ticker in list(risk_adj_mom_3m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_invested_value += position_value
                        else:
                            risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_portfolio_value = risk_adj_mom_3m_invested_value + risk_adj_mom_3m_cash
        risk_adj_mom_3m_portfolio_history.append(risk_adj_mom_3m_portfolio_value)

        # Update RISK-ADJ MOM 3M MONTHLY portfolio value daily
        risk_adj_mom_3m_monthly_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_3M_MONTHLY:
            for ticker in list(risk_adj_mom_3m_monthly_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_monthly_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_monthly_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_monthly_invested_value += position_value
                        else:
                            risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_monthly_portfolio_value = risk_adj_mom_3m_monthly_invested_value + risk_adj_mom_3m_monthly_cash
        risk_adj_mom_3m_monthly_portfolio_history.append(risk_adj_mom_3m_monthly_portfolio_value)

        # Update RISK-ADJ MOM 3M SENTIMENT portfolio value daily
        risk_adj_mom_3m_sentiment_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_3M_SENTIMENT:
            for ticker in list(risk_adj_mom_3m_sentiment_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_sentiment_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_sentiment_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_sentiment_invested_value += position_value
                        else:
                            risk_adj_mom_3m_sentiment_invested_value += risk_adj_mom_3m_sentiment_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_sentiment_invested_value += risk_adj_mom_3m_sentiment_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_sentiment_invested_value += risk_adj_mom_3m_sentiment_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_sentiment_portfolio_value = risk_adj_mom_3m_sentiment_invested_value + risk_adj_mom_3m_sentiment_cash
        risk_adj_mom_3m_sentiment_portfolio_history.append(risk_adj_mom_3m_sentiment_portfolio_value)

        # Update RISK-ADJ MOM 3M MARKET-UP ONLY portfolio value daily
        risk_adj_mom_3m_market_up_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_3M_MARKET_UP:
            for ticker in list(risk_adj_mom_3m_market_up_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_market_up_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_market_up_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_market_up_invested_value += position_value
                        else:
                            risk_adj_mom_3m_market_up_invested_value += risk_adj_mom_3m_market_up_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_market_up_invested_value += risk_adj_mom_3m_market_up_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_market_up_invested_value += risk_adj_mom_3m_market_up_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_market_up_portfolio_value = risk_adj_mom_3m_market_up_invested_value + risk_adj_mom_3m_market_up_cash
        risk_adj_mom_3m_market_up_portfolio_history.append(risk_adj_mom_3m_market_up_portfolio_value)

        # Update RISK-ADJ MOM 3M WITH STOPS portfolio value daily
        risk_adj_mom_3m_with_stops_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS:
            for ticker in list(risk_adj_mom_3m_with_stops_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_with_stops_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_with_stops_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_with_stops_invested_value += position_value
                        else:
                            risk_adj_mom_3m_with_stops_invested_value += risk_adj_mom_3m_with_stops_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_with_stops_invested_value += risk_adj_mom_3m_with_stops_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_with_stops_invested_value += risk_adj_mom_3m_with_stops_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_with_stops_portfolio_value = risk_adj_mom_3m_with_stops_invested_value + risk_adj_mom_3m_with_stops_cash
        risk_adj_mom_3m_with_stops_portfolio_history.append(risk_adj_mom_3m_with_stops_portfolio_value)

        # Check custom stops for Risk-Adj Mom 3M with Stops
        if config.ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS:
            from risk_adj_mom_3m_with_stops_strategy import update_risk_adj_mom_3m_with_stops_positions
            updated_positions, stop_costs, sold_positions = update_risk_adj_mom_3m_with_stops_positions(
                risk_adj_mom_3m_with_stops_positions,
                ticker_data_grouped,
                current_date,
                TRANSACTION_COST
            )
            risk_adj_mom_3m_with_stops_positions = updated_positions
            risk_adj_mom_3m_with_stops_transaction_costs += stop_costs

            # Add cash from sold positions
            for ticker, reason, net_value in sold_positions:
                risk_adj_mom_3m_with_stops_cash += net_value

        # Update VOL-SWEET MOMENTUM portfolio value daily
        vol_sweet_mom_invested_value = 0.0
        if config.ENABLE_VOL_SWEET_MOM:
            for ticker in list(vol_sweet_mom_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = vol_sweet_mom_positions[ticker]['shares']
                            position_value = shares * current_price
                            vol_sweet_mom_positions[ticker]['value'] = position_value
                            vol_sweet_mom_invested_value += position_value
                        else:
                            vol_sweet_mom_invested_value += vol_sweet_mom_positions[ticker].get('value', 0.0)
                    else:
                        vol_sweet_mom_invested_value += vol_sweet_mom_positions[ticker].get('value', 0.0)
                except Exception:
                    vol_sweet_mom_invested_value += vol_sweet_mom_positions[ticker].get('value', 0.0)
        vol_sweet_mom_portfolio_value = vol_sweet_mom_invested_value + vol_sweet_mom_cash
        vol_sweet_mom_portfolio_history.append(vol_sweet_mom_portfolio_value)

        # Update RISK-ADJ MOM 1M + VOL-SWEET portfolio value daily
        risk_adj_mom_1m_vol_sweet_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET:
            for ticker in list(risk_adj_mom_1m_vol_sweet_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_1m_vol_sweet_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_1m_vol_sweet_positions[ticker]['value'] = position_value
                            risk_adj_mom_1m_vol_sweet_invested_value += position_value
                        else:
                            risk_adj_mom_1m_vol_sweet_invested_value += risk_adj_mom_1m_vol_sweet_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_1m_vol_sweet_invested_value += risk_adj_mom_1m_vol_sweet_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_1m_vol_sweet_invested_value += risk_adj_mom_1m_vol_sweet_positions[ticker].get('value', 0.0)
        risk_adj_mom_1m_vol_sweet_portfolio_value = risk_adj_mom_1m_vol_sweet_invested_value + risk_adj_mom_1m_vol_sweet_cash
        risk_adj_mom_1m_vol_sweet_portfolio_history.append(risk_adj_mom_1m_vol_sweet_portfolio_value)

        # Update BH 1Y VolSweet Accel portfolio value daily
        bh_1y_volsweet_accel_invested_value = 0.0
        if config.ENABLE_BH_1Y_VOL_SWEET_ACCEL:
            for ticker in list(bh_1y_volsweet_accel_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = bh_1y_volsweet_accel_positions[ticker].get('shares', 0)
                            bh_1y_volsweet_accel_invested_value += shares * current_price
                except Exception:
                    pass
        bh_1y_volsweet_accel_portfolio_value = bh_1y_volsweet_accel_invested_value + bh_1y_volsweet_accel_cash
        bh_1y_volsweet_accel_portfolio_history.append(bh_1y_volsweet_accel_portfolio_value)

        # Update BH 1Y Dynamic Accel portfolio value daily
        bh_1y_dynamic_accel_invested_value = 0.0
        if config.ENABLE_BH_1Y_DYNAMIC_ACCEL:
            for ticker in list(bh_1y_dynamic_accel_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = bh_1y_dynamic_accel_positions[ticker].get('shares', 0)
                            bh_1y_dynamic_accel_invested_value += shares * current_price
                except Exception:
                    pass
        bh_1y_dynamic_accel_portfolio_value = bh_1y_dynamic_accel_invested_value + bh_1y_dynamic_accel_cash
        bh_1y_dynamic_accel_portfolio_history.append(bh_1y_dynamic_accel_portfolio_value)

        # Update RISK-ADJ MOM 1M portfolio value daily
        risk_adj_mom_1m_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_1M:
            for ticker in list(risk_adj_mom_1m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_1m_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_1m_positions[ticker]['value'] = position_value
                            risk_adj_mom_1m_invested_value += position_value
                        else:
                            risk_adj_mom_1m_invested_value += risk_adj_mom_1m_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_1m_invested_value += risk_adj_mom_1m_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_1m_invested_value += risk_adj_mom_1m_positions[ticker].get('value', 0.0)
        risk_adj_mom_1m_portfolio_value = risk_adj_mom_1m_invested_value + risk_adj_mom_1m_cash
        risk_adj_mom_1m_portfolio_history.append(risk_adj_mom_1m_portfolio_value)

        # Update RISK-ADJ MOM 1M MONTHLY portfolio value daily
        risk_adj_mom_1m_monthly_invested_value = 0.0
        if config.ENABLE_RISK_ADJ_MOM_1M_MONTHLY:
            for ticker in list(risk_adj_mom_1m_monthly_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_1m_monthly_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_1m_monthly_positions[ticker]['value'] = position_value
                            risk_adj_mom_1m_monthly_invested_value += position_value
                        else:
                            risk_adj_mom_1m_monthly_invested_value += risk_adj_mom_1m_monthly_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_1m_monthly_invested_value += risk_adj_mom_1m_monthly_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_1m_monthly_invested_value += risk_adj_mom_1m_monthly_positions[ticker].get('value', 0.0)
        risk_adj_mom_1m_monthly_portfolio_value = risk_adj_mom_1m_monthly_invested_value + risk_adj_mom_1m_monthly_cash
        risk_adj_mom_1m_monthly_portfolio_history.append(risk_adj_mom_1m_monthly_portfolio_value)

        # Update AI ELITE portfolio value daily
        ai_elite_invested_value = 0.0
        if config.ENABLE_AI_ELITE:
            for ticker in list(ai_elite_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_elite_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_elite_positions[ticker]['value'] = position_value
                            ai_elite_invested_value += position_value
                        else:
                            ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
                    else:
                        ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
                except Exception:
                    ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
        ai_elite_portfolio_value = ai_elite_invested_value + ai_elite_cash
        ai_elite_portfolio_history.append(ai_elite_portfolio_value)

        # Update AI ELITE MONTHLY portfolio value daily
        ai_elite_monthly_invested_value = 0.0
        if config.ENABLE_AI_ELITE_MONTHLY:
            for ticker in list(ai_elite_monthly_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_elite_monthly_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_elite_monthly_positions[ticker]['value'] = position_value
                            ai_elite_monthly_invested_value += position_value
                        else:
                            ai_elite_monthly_invested_value += ai_elite_monthly_positions[ticker].get('value', 0.0)
                    else:
                        ai_elite_monthly_invested_value += ai_elite_monthly_positions[ticker].get('value', 0.0)
                except Exception:
                    ai_elite_monthly_invested_value += ai_elite_monthly_positions[ticker].get('value', 0.0)
        ai_elite_monthly_portfolio_value = ai_elite_monthly_invested_value + ai_elite_monthly_cash
        ai_elite_monthly_portfolio_history.append(ai_elite_monthly_portfolio_value)

        # TOP5 CONSISTENCY BLEND: use prior-day top-5 consistency to weight current source holdings.
        if config.ENABLE_TOP5_CONSISTENCY_BLEND:
            try:
                source_strategy_data = _build_daily_strategy_data(locals())
                lagged_days = day_count - 1
                new_top5_consistency_blend_stocks, top5_consistency_blend_last_weights, top5_consistency_sources = _select_top5_consistency_blend_stocks(
                    source_strategy_data=source_strategy_data,
                    top5_consistency_counts=top5_consistency_counts,
                    total_days=lagged_days,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                )

                if new_top5_consistency_blend_stocks:
                    weight_summary = ", ".join(
                        f"{name}={weight:.1%}"
                        for name, weight in sorted(
                            top5_consistency_blend_last_weights.items(),
                            key=lambda item: (-item[1], item[0])
                        )
                    )
                    print(f"   📊 Top5 Cons Blend Day {day_count}: {new_top5_consistency_blend_stocks}")
                    print(f"   ⚖️ Top5 Cons Blend weights from {lagged_days} prior day(s): {weight_summary}")

                    top5_consistency_blend_positions, top5_consistency_blend_cash, current_top5_consistency_blend_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Top5 Cons Blend",
                        current_stocks=current_top5_consistency_blend_stocks,
                        new_stocks=new_top5_consistency_blend_stocks,
                        positions=top5_consistency_blend_positions,
                        cash=top5_consistency_blend_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                        force_rebalance=not top5_consistency_blend_initialized
                    )
                    strategies_rebalanced_today['Top5 Cons Blend'] = rebalanced_flag
                    top5_consistency_blend_transaction_costs += rc
                    top5_consistency_blend_initialized = True
                elif lagged_days <= 0:
                    print("   📊 Top5 Cons Blend: Waiting for prior-day consistency data")
            except Exception as e:
                print(f"   ⚠️ Top5 Consistency Blend error: {e}")

        # Update AI ELITE FILTERED portfolio value daily
        ai_elite_filtered_invested_value = 0.0
        if config.ENABLE_AI_ELITE_FILTERED:
            for ticker in list(ai_elite_filtered_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_elite_filtered_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_elite_filtered_positions[ticker]['value'] = position_value
                            ai_elite_filtered_invested_value += position_value
                        else:
                            ai_elite_filtered_invested_value += ai_elite_filtered_positions[ticker].get('value', 0.0)
                    else:
                        ai_elite_filtered_invested_value += ai_elite_filtered_positions[ticker].get('value', 0.0)
                except Exception:
                    ai_elite_filtered_invested_value += ai_elite_filtered_positions[ticker].get('value', 0.0)
        ai_elite_filtered_portfolio_value = ai_elite_filtered_invested_value + ai_elite_filtered_cash
        ai_elite_filtered_portfolio_history.append(ai_elite_filtered_portfolio_value)

        # Update AI ELITE MARKET-UP portfolio value daily
        ai_elite_market_up_invested_value = 0.0
        if config.ENABLE_AI_ELITE_MARKET_UP:
            for ticker in list(ai_elite_market_up_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_elite_market_up_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_elite_market_up_positions[ticker]['value'] = position_value
                            ai_elite_market_up_invested_value += position_value
                        else:
                            ai_elite_market_up_invested_value += ai_elite_market_up_positions[ticker].get('value', 0.0)
                    else:
                        ai_elite_market_up_invested_value += ai_elite_market_up_positions[ticker].get('value', 0.0)
                except Exception:
                    ai_elite_market_up_invested_value += ai_elite_market_up_positions[ticker].get('value', 0.0)
        ai_elite_market_up_portfolio_value = ai_elite_market_up_invested_value + ai_elite_market_up_cash
        ai_elite_market_up_portfolio_history.append(ai_elite_market_up_portfolio_value)

        # Update SAVGOL TREND portfolio value daily
        savgol_trend_invested_value = 0.0
        if config.ENABLE_SAVGOL_TREND:
            for ticker in list(savgol_trend_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = savgol_trend_positions[ticker]['shares']
                            position_value = shares * current_price
                            savgol_trend_positions[ticker]['value'] = position_value
                            savgol_trend_invested_value += position_value
                        else:
                            savgol_trend_invested_value += savgol_trend_positions[ticker].get('value', 0.0)
                    else:
                        savgol_trend_invested_value += savgol_trend_positions[ticker].get('value', 0.0)
                except Exception:
                    savgol_trend_invested_value += savgol_trend_positions[ticker].get('value', 0.0)
        savgol_trend_portfolio_value = savgol_trend_invested_value + savgol_trend_cash
        savgol_trend_portfolio_history.append(savgol_trend_portfolio_value)

        # Update UNIVERSAL MODEL portfolio value daily
        universal_model_invested_value = 0.0
        if config.ENABLE_UNIVERSAL_MODEL:
            for ticker in list(universal_model_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = universal_model_positions[ticker]['shares']
                            position_value = shares * current_price
                            universal_model_positions[ticker]['value'] = position_value
                            universal_model_invested_value += position_value
                        else:
                            universal_model_invested_value += universal_model_positions[ticker].get('value', 0.0)
                    else:
                        universal_model_invested_value += universal_model_positions[ticker].get('value', 0.0)
                except Exception:
                    universal_model_invested_value += universal_model_positions[ticker].get('value', 0.0)
        universal_model_portfolio_value = universal_model_invested_value + universal_model_cash
        universal_model_portfolio_history.append(universal_model_portfolio_value)

        # Update TOP5 CONSISTENCY BLEND portfolio value daily
        top5_consistency_blend_invested_value = 0.0
        if config.ENABLE_TOP5_CONSISTENCY_BLEND:
            for ticker in list(top5_consistency_blend_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = top5_consistency_blend_positions[ticker]['shares']
                            position_value = shares * current_price
                            top5_consistency_blend_positions[ticker]['value'] = position_value
                            top5_consistency_blend_invested_value += position_value
                        else:
                            top5_consistency_blend_invested_value += top5_consistency_blend_positions[ticker].get('value', 0.0)
                    else:
                        top5_consistency_blend_invested_value += top5_consistency_blend_positions[ticker].get('value', 0.0)
                except Exception:
                    top5_consistency_blend_invested_value += top5_consistency_blend_positions[ticker].get('value', 0.0)
        top5_consistency_blend_portfolio_value = top5_consistency_blend_invested_value + top5_consistency_blend_cash
        if config.ENABLE_TOP5_CONSISTENCY_BLEND:
            top5_consistency_blend_portfolio_history.append(top5_consistency_blend_portfolio_value)

        # Update META STRATEGIES portfolio values daily
        def _update_meta_portfolio(positions, cash):
            """Helper to update meta strategy portfolio value."""
            invested = 0.0
            for ticker in list(positions.keys()):
                try:
                    td = ticker_data_grouped.get(ticker)
                    if td is not None and not td.empty:
                        price_data = td.loc[:current_date]
                        if not price_data.empty:
                            valid_prices = price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if current_price > 0:
                                    positions[ticker]['value'] = positions[ticker]['shares'] * current_price
                                    invested += positions[ticker]['value']
                                    continue
                    invested += positions[ticker].get('value', 0.0)
                except Exception:
                    invested += positions[ticker].get('value', 0.0)
            return invested + cash

        # Meta strategy portfolio values are updated later in the day after the
        # meta selectors rebalance from the completed shared strategy snapshot.

        # Update BOLLINGER BANDS strategy portfolio values daily
        if config.ENABLE_BB_MEAN_REVERSION:
            bb_mean_reversion_value = _update_meta_portfolio(bb_mean_reversion_positions, bb_mean_reversion_cash)
            bb_mean_reversion_history.append(bb_mean_reversion_value)

        if config.ENABLE_BB_BREAKOUT:
            bb_breakout_value = _update_meta_portfolio(bb_breakout_positions, bb_breakout_cash)
            bb_breakout_history.append(bb_breakout_value)

        if config.ENABLE_BB_SQUEEZE_BREAKOUT:
            bb_squeeze_breakout_value = _update_meta_portfolio(bb_squeeze_breakout_positions, bb_squeeze_breakout_cash)
            bb_squeeze_breakout_history.append(bb_squeeze_breakout_value)

        if config.ENABLE_BB_RSI_COMBO:
            bb_rsi_combo_value = _update_meta_portfolio(bb_rsi_combo_positions, bb_rsi_combo_cash)
            bb_rsi_combo_history.append(bb_rsi_combo_value)

        # Update TREND BREAKOUT portfolio value daily
        if config.ENABLE_TREND_BREAKOUT:
            trend_breakout_value = _update_meta_portfolio(trend_breakout_positions, trend_breakout_cash)
            trend_breakout_history.append(trend_breakout_value)

        # Update MEAN REVERSION portfolio value daily (skip if disabled)
        mean_reversion_invested_value = 0.0
        if config.ENABLE_MEAN_REVERSION:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(mean_reversion_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            current_price_data = ticker_data.loc[ticker_data.index == current_date]
                            if not current_price_data.empty:
                                current_price = current_price_data['Close'].iloc[0]
                            else:
                                current_price = None
                        else:
                            current_price = None

                        if current_price is not None and current_price > 0:
                            shares = mean_reversion_positions[ticker]['shares']
                            value = shares * current_price
                            mean_reversion_positions[ticker]['value'] = value
                            mean_reversion_invested_value += value
                        else:
                            # Use previous value if current price is unavailable (e.g., non-US stocks on US holidays)
                            mean_reversion_invested_value += mean_reversion_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        # Keep previous value on error
                        mean_reversion_invested_value += mean_reversion_positions.get(ticker, {}).get('value', 0.0)

        mean_reversion_portfolio_value = mean_reversion_invested_value + mean_reversion_cash
        mean_reversion_portfolio_history.append(mean_reversion_portfolio_value)

        # Update QUALITY + MOMENTUM portfolio value daily (skip if disabled)
        quality_momentum_invested_value = 0.0
        if config.ENABLE_QUALITY_MOM:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(quality_momentum_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            current_price_data = ticker_data.loc[ticker_data.index == current_date]
                            if not current_price_data.empty:
                                current_price = current_price_data['Close'].iloc[0]
                            else:
                                current_price = None
                        else:
                            current_price = None

                        if current_price is not None and current_price > 0:
                            shares = quality_momentum_positions[ticker]['shares']
                            value = shares * current_price
                            quality_momentum_positions[ticker]['value'] = value
                            quality_momentum_invested_value += value
                        else:
                            # Use previous value if current price is unavailable (e.g., non-US stocks on US holidays)
                            quality_momentum_invested_value += quality_momentum_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        # Keep previous value on error
                        quality_momentum_invested_value += quality_momentum_positions.get(ticker, {}).get('value', 0.0)

        quality_momentum_portfolio_value = quality_momentum_invested_value + quality_momentum_cash
        quality_momentum_portfolio_history.append(quality_momentum_portfolio_value)

        # Update VOLATILITY-ADJUSTED MOMENTUM portfolio value daily (skip if disabled)
        volatility_adj_mom_invested_value = 0.0
        if config.ENABLE_VOLATILITY_ADJ_MOM:
            for ticker, pos in volatility_adj_mom_positions.items():
                try:
                    # Get current price
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        current_price_data = ticker_data.loc[ticker_data.index == current_date]
                        if not current_price_data.empty:
                            price_data = current_price_data['Close']
                        else:
                            price_data = pd.Series()
                    else:
                        price_data = pd.Series()

                    if not price_data.empty:
                        current_price = price_data.iloc[0]
                        position_value = pos['shares'] * current_price
                        volatility_adj_mom_invested_value += position_value

                        # Update stored position value
                        pos['value'] = position_value
                    else:
                        # Use previous value if current price is invalid
                        volatility_adj_mom_invested_value += pos.get('value', 0.0)
                except Exception:
                    # Keep previous value if price lookup fails
                    volatility_adj_mom_invested_value += pos.get('value', 0.0)

        volatility_adj_mom_portfolio_value = volatility_adj_mom_invested_value + volatility_adj_mom_cash
        volatility_adj_mom_portfolio_history.append(volatility_adj_mom_portfolio_value)

        # Update INVERSE ETF HEDGE portfolio value daily
        if config.ENABLE_INVERSE_ETF_HEDGE:
            inverse_etf_hedge_invested_value = 0.0
            for ticker, pos in inverse_etf_hedge_positions.items():
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        current_price_data = ticker_data.loc[ticker_data.index == current_date]
                        if not current_price_data.empty:
                            current_price = current_price_data['Close'].iloc[0]
                            position_value = pos['shares'] * current_price
                            inverse_etf_hedge_invested_value += position_value
                            pos['value'] = position_value
                        else:
                            inverse_etf_hedge_invested_value += pos.get('value', 0.0)
                    else:
                        inverse_etf_hedge_invested_value += pos.get('value', 0.0)
                except Exception:
                    inverse_etf_hedge_invested_value += pos.get('value', 0.0)

            inverse_etf_hedge_portfolio_value = inverse_etf_hedge_invested_value + inverse_etf_hedge_cash
            inverse_etf_hedge_portfolio_history.append(inverse_etf_hedge_portfolio_value)

        # Update ANALYST RECOMMENDATION portfolio value daily
        if config.ENABLE_ANALYST_RECOMMENDATION:
            analyst_rec_invested_value = 0.0
            for ticker, pos in analyst_rec_positions.items():
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        valid_prices = ticker_data[ticker_data.index <= current_date]['Close'].dropna()
                        if len(valid_prices) > 0:
                            current_price = valid_prices.iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = pos['shares'] * current_price
                                analyst_rec_positions[ticker]['value'] = position_value
                                analyst_rec_invested_value += position_value
                            else:
                                analyst_rec_invested_value += pos.get('value', 0.0)
                        else:
                            analyst_rec_invested_value += pos.get('value', 0.0)
                except Exception:
                    analyst_rec_invested_value += pos.get('value', 0.0)

            analyst_rec_portfolio_value = analyst_rec_invested_value + analyst_rec_cash
            analyst_rec_portfolio_history.append(analyst_rec_portfolio_value)

        # === MOMENTUM + AI HYBRID: Update portfolio value ===
        if config.ENABLE_MOMENTUM_AI_HYBRID:
            momentum_ai_hybrid_invested_value = 0.0
            for ticker in list(momentum_ai_hybrid_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = momentum_ai_hybrid_positions[ticker]['shares']
                            position_value = shares * current_price
                            momentum_ai_hybrid_positions[ticker]['value'] = position_value
                            momentum_ai_hybrid_invested_value += position_value
                        else:
                            momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)
                    else:
                        momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)
                except Exception:
                    momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)

            momentum_ai_hybrid_portfolio_value = momentum_ai_hybrid_invested_value + momentum_ai_hybrid_cash
            momentum_ai_hybrid_portfolio_history.append(momentum_ai_hybrid_portfolio_value)

        # Update STATIC BH 1Y portfolio value daily (skip if disabled)
        static_bh_1y_invested_value = 0.0
        if config.ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_1y_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_positions[ticker]['shares'] * current_price
                                    static_bh_1y_positions[ticker]['value'] = position_value
                                    static_bh_1y_invested_value += position_value
                                else:
                                    static_bh_1y_invested_value += static_bh_1y_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_1y_invested_value += static_bh_1y_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 1Y position for {ticker}: {e}")

        static_bh_1y_portfolio_value = static_bh_1y_invested_value + static_bh_1y_cash
        static_bh_1y_portfolio_history.append(static_bh_1y_portfolio_value)

        # Update ADAPTIVE REBALANCING STRATEGIES portfolio values
        # 1. Volatility-triggered
        if config.ENABLE_STATIC_BH_1Y_VOLATILITY:
            static_bh_1y_vol_invested_value = 0.0
            for ticker in list(static_bh_1y_vol_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_vol_positions[ticker]['shares'] * current_price
                                    static_bh_1y_vol_positions[ticker]['value'] = position_value
                                    static_bh_1y_vol_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_vol_portfolio_value = static_bh_1y_vol_invested_value + static_bh_1y_vol_cash
            static_bh_1y_vol_portfolio_history.append(static_bh_1y_vol_portfolio_value)

        # 2. Performance-triggered
        if config.ENABLE_STATIC_BH_1Y_PERFORMANCE:
            static_bh_1y_perf_invested_value = 0.0
            for ticker in list(static_bh_1y_perf_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_perf_positions[ticker]['shares'] * current_price
                                    static_bh_1y_perf_positions[ticker]['value'] = position_value
                                    static_bh_1y_perf_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_perf_portfolio_value = static_bh_1y_perf_invested_value + static_bh_1y_perf_cash
            static_bh_1y_perf_portfolio_history.append(static_bh_1y_perf_portfolio_value)

        # 3. Momentum-triggered
        if config.ENABLE_STATIC_BH_1Y_MOMENTUM:
            static_bh_1y_mom_invested_value = 0.0
            for ticker in list(static_bh_1y_mom_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_mom_positions[ticker]['shares'] * current_price
                                    static_bh_1y_mom_positions[ticker]['value'] = position_value
                                    static_bh_1y_mom_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_mom_portfolio_value = static_bh_1y_mom_invested_value + static_bh_1y_mom_cash
            static_bh_1y_mom_portfolio_history.append(static_bh_1y_mom_portfolio_value)

        # 4. ATR-triggered
        if config.ENABLE_STATIC_BH_1Y_ATR:
            static_bh_1y_atr_invested_value = 0.0
            for ticker in list(static_bh_1y_atr_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_atr_positions[ticker]['shares'] * current_price
                                    static_bh_1y_atr_positions[ticker]['value'] = position_value
                                    static_bh_1y_atr_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_atr_portfolio_value = static_bh_1y_atr_invested_value + static_bh_1y_atr_cash
            static_bh_1y_atr_portfolio_history.append(static_bh_1y_atr_portfolio_value)

        # 5. Hybrid
        if config.ENABLE_STATIC_BH_1Y_HYBRID:
            static_bh_1y_hybrid_invested_value = 0.0
            for ticker in list(static_bh_1y_hybrid_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_hybrid_positions[ticker]['shares'] * current_price
                                    static_bh_1y_hybrid_positions[ticker]['value'] = position_value
                                    static_bh_1y_hybrid_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_hybrid_portfolio_value = static_bh_1y_hybrid_invested_value + static_bh_1y_hybrid_cash
            static_bh_1y_hybrid_portfolio_history.append(static_bh_1y_hybrid_portfolio_value)

        # 6. Volume Filter
        if config.ENABLE_STATIC_BH_1Y_VOLUME_FILTER:
            static_bh_1y_volume_invested_value = 0.0
            for ticker in list(static_bh_1y_volume_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_volume_positions[ticker]['shares'] * current_price
                                static_bh_1y_volume_positions[ticker]['value'] = position_value
                                static_bh_1y_volume_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_volume_portfolio_value = static_bh_1y_volume_invested_value + static_bh_1y_volume_cash
            static_bh_1y_volume_portfolio_history.append(static_bh_1y_volume_portfolio_value)

        # 7. Sector Rotation
        if config.ENABLE_STATIC_BH_1Y_SECTOR_ROTATION:
            static_bh_1y_sector_invested_value = 0.0
            for ticker in list(static_bh_1y_sector_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_sector_positions[ticker]['shares'] * current_price
                                static_bh_1y_sector_positions[ticker]['value'] = position_value
                                static_bh_1y_sector_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_sector_portfolio_value = static_bh_1y_sector_invested_value + static_bh_1y_sector_cash
            static_bh_1y_sector_portfolio_history.append(static_bh_1y_sector_portfolio_value)

        # 8. Performance Threshold
        if config.ENABLE_STATIC_BH_1Y_PERFORMANCE_THRESHOLD:
            static_bh_1y_perf_threshold_invested_value = 0.0
            for ticker in list(static_bh_1y_perf_threshold_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_perf_threshold_positions[ticker]['shares'] * current_price
                                static_bh_1y_perf_threshold_positions[ticker]['value'] = position_value
                                static_bh_1y_perf_threshold_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_perf_threshold_portfolio_value = static_bh_1y_perf_threshold_invested_value + static_bh_1y_perf_threshold_cash
            static_bh_1y_perf_threshold_portfolio_history.append(static_bh_1y_perf_threshold_portfolio_value)

        # 9. Market Regime
        if config.ENABLE_STATIC_BH_1Y_MARKET_REGIME:
            static_bh_1y_market_regime_invested_value = 0.0
            for ticker in list(static_bh_1y_market_regime_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_market_regime_positions[ticker]['shares'] * current_price
                                static_bh_1y_market_regime_positions[ticker]['value'] = position_value
                                static_bh_1y_market_regime_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_market_regime_portfolio_value = static_bh_1y_market_regime_invested_value + static_bh_1y_market_regime_cash
            static_bh_1y_market_regime_portfolio_history.append(static_bh_1y_market_regime_portfolio_value)

        # 10. Momentum Persistence
        if config.ENABLE_STATIC_BH_1Y_MOMENTUM_PERSIST:
            static_bh_1y_mom_persist_invested_value = 0.0
            for ticker in list(static_bh_1y_mom_persist_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_mom_persist_positions[ticker]['shares'] * current_price
                                static_bh_1y_mom_persist_positions[ticker]['value'] = position_value
                                static_bh_1y_mom_persist_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_mom_persist_portfolio_value = static_bh_1y_mom_persist_invested_value + static_bh_1y_mom_persist_cash
            static_bh_1y_mom_persist_portfolio_history.append(static_bh_1y_mom_persist_portfolio_value)

        # 11. Overlap-Based
        if config.ENABLE_STATIC_BH_1Y_OVERLAP:
            static_bh_1y_overlap_invested_value = 0.0
            for ticker in list(static_bh_1y_overlap_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_overlap_positions[ticker]['shares'] * current_price
                                static_bh_1y_overlap_positions[ticker]['value'] = position_value
                                static_bh_1y_overlap_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_overlap_portfolio_value = static_bh_1y_overlap_invested_value + static_bh_1y_overlap_cash
            static_bh_1y_overlap_portfolio_history.append(static_bh_1y_overlap_portfolio_value)

        # 12. Rank Drift
        if config.ENABLE_STATIC_BH_1Y_RANK_DRIFT:
            static_bh_1y_rank_drift_invested_value = 0.0
            for ticker in list(static_bh_1y_rank_drift_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_rank_drift_positions[ticker]['shares'] * current_price
                                static_bh_1y_rank_drift_positions[ticker]['value'] = position_value
                                static_bh_1y_rank_drift_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_rank_drift_portfolio_value = static_bh_1y_rank_drift_invested_value + static_bh_1y_rank_drift_cash
            static_bh_1y_rank_drift_portfolio_history.append(static_bh_1y_rank_drift_portfolio_value)

        # 13. Drawdown Trigger
        if config.ENABLE_STATIC_BH_1Y_DRAWDOWN:
            static_bh_1y_drawdown_invested_value = 0.0
            for ticker in list(static_bh_1y_drawdown_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_drawdown_positions[ticker]['shares'] * current_price
                                static_bh_1y_drawdown_positions[ticker]['value'] = position_value
                                static_bh_1y_drawdown_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_drawdown_portfolio_value = static_bh_1y_drawdown_invested_value + static_bh_1y_drawdown_cash
            static_bh_1y_drawdown_portfolio_history.append(static_bh_1y_drawdown_portfolio_value)
            # Update peak
            if static_bh_1y_drawdown_portfolio_value > static_bh_1y_drawdown_peak:
                static_bh_1y_drawdown_peak = static_bh_1y_drawdown_portfolio_value

        # 14. Smart Monthly
        if config.ENABLE_STATIC_BH_1Y_SMART_MONTHLY:
            static_bh_1y_smart_monthly_invested_value = 0.0
            for ticker in list(static_bh_1y_smart_monthly_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker].loc[:current_date]
                        if not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if not pd.isna(current_price) and current_price > 0:
                                position_value = static_bh_1y_smart_monthly_positions[ticker]['shares'] * current_price
                                static_bh_1y_smart_monthly_positions[ticker]['value'] = position_value
                                static_bh_1y_smart_monthly_invested_value += position_value
                except Exception:
                    pass
            static_bh_1y_smart_monthly_portfolio_value = static_bh_1y_smart_monthly_invested_value + static_bh_1y_smart_monthly_cash
            static_bh_1y_smart_monthly_portfolio_history.append(static_bh_1y_smart_monthly_portfolio_value)
            # Update peak
            if static_bh_1y_smart_monthly_portfolio_value > static_bh_1y_smart_monthly_peak:
                static_bh_1y_smart_monthly_peak = static_bh_1y_smart_monthly_portfolio_value

        # =====================================================================
        # Update portfolio values for 6 SMART REBALANCING STRATEGIES
        # =====================================================================

        # Helper function to update portfolio value
        def _update_smart_strategy_value(positions, cash, ticker_data_grouped, current_date):
            invested = 0.0
            for ticker in list(positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        data = ticker_data_grouped[ticker].loc[:current_date]
                        if not data.empty:
                            price = data['Close'].dropna().iloc[-1]
                            if not pd.isna(price) and price > 0:
                                positions[ticker]['value'] = positions[ticker]['shares'] * price
                                invested += positions[ticker]['value']
                except Exception:
                    pass
            return invested + cash

        # 1. BH 1Y Mom Sell
        if config.ENABLE_BH_1Y_MOM_SELL:
            bh_1y_mom_sell_portfolio_value = _update_smart_strategy_value(
                bh_1y_mom_sell_positions, bh_1y_mom_sell_cash, ticker_data_grouped, current_date
            )
            bh_1y_mom_sell_portfolio_history.append(bh_1y_mom_sell_portfolio_value)

        # 2. BH 1Y Rank Sell
        if config.ENABLE_BH_1Y_RANK_SELL:
            bh_1y_rank_sell_portfolio_value = _update_smart_strategy_value(
                bh_1y_rank_sell_positions, bh_1y_rank_sell_cash, ticker_data_grouped, current_date
            )
            bh_1y_rank_sell_portfolio_history.append(bh_1y_rank_sell_portfolio_value)

        # 3. BH 1Y Trailing Mom
        if config.ENABLE_BH_1Y_TRAILING_MOM:
            bh_1y_trailing_mom_portfolio_value = _update_smart_strategy_value(
                bh_1y_trailing_mom_positions, bh_1y_trailing_mom_cash, ticker_data_grouped, current_date
            )
            bh_1y_trailing_mom_portfolio_history.append(bh_1y_trailing_mom_portfolio_value)

        # 4. BH 1Y Volume Confirm
        if config.ENABLE_BH_1Y_VOLUME_CONFIRM:
            bh_1y_volume_confirm_portfolio_value = _update_smart_strategy_value(
                bh_1y_volume_confirm_positions, bh_1y_volume_confirm_cash, ticker_data_grouped, current_date
            )
            bh_1y_volume_confirm_portfolio_history.append(bh_1y_volume_confirm_portfolio_value)

        # 5. BH 1Y Sector Aware
        if config.ENABLE_BH_1Y_SECTOR_AWARE:
            bh_1y_sector_aware_portfolio_value = _update_smart_strategy_value(
                bh_1y_sector_aware_positions, bh_1y_sector_aware_cash, ticker_data_grouped, current_date
            )
            bh_1y_sector_aware_portfolio_history.append(bh_1y_sector_aware_portfolio_value)

        # 6. BH 1Y Accel Buy
        if config.ENABLE_BH_1Y_ACCEL_BUY:
            bh_1y_accel_buy_portfolio_value = _update_smart_strategy_value(
                bh_1y_accel_buy_positions, bh_1y_accel_buy_cash, ticker_data_grouped, current_date
            )
            bh_1y_accel_buy_portfolio_history.append(bh_1y_accel_buy_portfolio_value)

        # Static BH 3M with Acceleration
        if config.ENABLE_STATIC_BH_3M_ACCEL:
            static_bh_3m_accel_portfolio_value = _update_smart_strategy_value(
                static_bh_3m_accel_positions, static_bh_3m_accel_cash, ticker_data_grouped, current_date
            )
            static_bh_3m_accel_portfolio_history.append(static_bh_3m_accel_portfolio_value)

        # =====================================================================
        # 10 NEW REBALANCING STRATEGIES - Portfolio Value Updates
        # =====================================================================

        # 1. Volatility-Adjusted Rebalancing
        if config.ENABLE_BH_1Y_VOL_ADJ_REBAL:
            bh_1y_vol_adj_rebal_portfolio_value = _update_smart_strategy_value(
                bh_1y_vol_adj_rebal_positions, bh_1y_vol_adj_rebal_cash, ticker_data_grouped, current_date
            )
            bh_1y_vol_adj_rebal_portfolio_history.append(bh_1y_vol_adj_rebal_portfolio_value)

        # 1b. Volatility-Adjusted Rebalancing with 1M/3M Ratio ranking
        if config.ENABLE_RATIO_1M3M_VOL_ADJ_REBAL:
            ratio_1m3m_vol_adj_rebal_portfolio_value = _update_smart_strategy_value(
                ratio_1m3m_vol_adj_rebal_positions, ratio_1m3m_vol_adj_rebal_cash, ticker_data_grouped, current_date
            )
            ratio_1m3m_vol_adj_rebal_portfolio_history.append(ratio_1m3m_vol_adj_rebal_portfolio_value)

        # 2. Correlation-Based Filtering
        if config.ENABLE_BH_1Y_CORR_FILTER:
            bh_1y_corr_filter_portfolio_value = _update_smart_strategy_value(
                bh_1y_corr_filter_positions, bh_1y_corr_filter_cash, ticker_data_grouped, current_date
            )
            bh_1y_corr_filter_portfolio_history.append(bh_1y_corr_filter_portfolio_value)

        # 3. Market Regime Detection
        if config.ENABLE_BH_1Y_REGIME_AWARE:
            bh_1y_regime_aware_portfolio_value = _update_smart_strategy_value(
                bh_1y_regime_aware_positions, bh_1y_regime_aware_cash, ticker_data_grouped, current_date
            )
            bh_1y_regime_aware_portfolio_history.append(bh_1y_regime_aware_portfolio_value)

        # 4. Risk Parity Allocation
        if config.ENABLE_BH_1Y_RISK_PARITY:
            bh_1y_risk_parity_portfolio_value = _update_smart_strategy_value(
                bh_1y_risk_parity_positions, bh_1y_risk_parity_cash, ticker_data_grouped, current_date
            )
            bh_1y_risk_parity_portfolio_history.append(bh_1y_risk_parity_portfolio_value)

        # 5. Adaptive Drift Threshold
        if config.ENABLE_BH_1Y_DRIFT_THRESH:
            bh_1y_drift_thresh_portfolio_value = _update_smart_strategy_value(
                bh_1y_drift_thresh_positions, bh_1y_drift_thresh_cash, ticker_data_grouped, current_date
            )
            bh_1y_drift_thresh_portfolio_history.append(bh_1y_drift_thresh_portfolio_value)

        # 6. Momentum Quality Score
        if config.ENABLE_BH_1Y_MOM_QUALITY:
            bh_1y_mom_quality_portfolio_value = _update_smart_strategy_value(
                bh_1y_mom_quality_positions, bh_1y_mom_quality_cash, ticker_data_grouped, current_date
            )
            bh_1y_mom_quality_portfolio_history.append(bh_1y_mom_quality_portfolio_value)

        # 7. Liquidity-Based Sizing
        if config.ENABLE_BH_1Y_LIQUIDITY:
            bh_1y_liquidity_portfolio_value = _update_smart_strategy_value(
                bh_1y_liquidity_positions, bh_1y_liquidity_cash, ticker_data_grouped, current_date
            )
            bh_1y_liquidity_portfolio_history.append(bh_1y_liquidity_portfolio_value)

        # 8. Earnings Avoidance
        if config.ENABLE_BH_1Y_EARNINGS_AVOID:
            bh_1y_earnings_avoid_portfolio_value = _update_smart_strategy_value(
                bh_1y_earnings_avoid_positions, bh_1y_earnings_avoid_cash, ticker_data_grouped, current_date
            )
            bh_1y_earnings_avoid_portfolio_history.append(bh_1y_earnings_avoid_portfolio_value)

        # 9. Multi-Factor Composite
        if config.ENABLE_BH_1Y_MULTI_FACTOR:
            bh_1y_multi_factor_portfolio_value = _update_smart_strategy_value(
                bh_1y_multi_factor_positions, bh_1y_multi_factor_cash, ticker_data_grouped, current_date
            )
            bh_1y_multi_factor_portfolio_history.append(bh_1y_multi_factor_portfolio_value)

        # 10. Time-Decay Holdings
        if config.ENABLE_BH_1Y_TIME_DECAY:
            bh_1y_time_decay_portfolio_value = _update_smart_strategy_value(
                bh_1y_time_decay_positions, bh_1y_time_decay_cash, ticker_data_grouped, current_date
            )
            bh_1y_time_decay_portfolio_history.append(bh_1y_time_decay_portfolio_value)

        # Update STATIC BH 6M portfolio value daily (skip if disabled)
        static_bh_6m_invested_value = 0.0
        if config.ENABLE_STATIC_BH_6M:
            for ticker in list(static_bh_6m_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_6m_positions[ticker]['shares'] * current_price
                                    static_bh_6m_positions[ticker]['value'] = position_value
                                    static_bh_6m_invested_value += position_value
                                else:
                                    static_bh_6m_invested_value += static_bh_6m_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_6m_invested_value += static_bh_6m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 6M position for {ticker}: {e}")

        static_bh_6m_portfolio_value = static_bh_6m_invested_value + static_bh_6m_cash
        static_bh_6m_portfolio_history.append(static_bh_6m_portfolio_value)

        # Update STATIC BH 3M portfolio value daily (skip if disabled)
        static_bh_3m_invested_value = 0.0
        if config.ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_3m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_3m_positions[ticker]['shares'] * current_price
                                    static_bh_3m_positions[ticker]['value'] = position_value
                                    static_bh_3m_invested_value += position_value
                                else:
                                    static_bh_3m_invested_value += static_bh_3m_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_3m_invested_value += static_bh_3m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 3M position for {ticker}: {e}")

        static_bh_3m_portfolio_value = static_bh_3m_invested_value + static_bh_3m_cash
        static_bh_3m_portfolio_history.append(static_bh_3m_portfolio_value)

        # Update STATIC BH 1M portfolio value daily (skip if disabled)
        static_bh_1m_invested_value = 0.0
        if config.ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_1m_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            # ticker_data already has date as index
                            current_price_data = ticker_data.loc[:current_date]
                            if not current_price_data.empty:
                                # Drop NaN values to avoid NaN propagation
                                valid_prices = current_price_data['Close'].dropna()
                                if len(valid_prices) > 0:
                                    current_price = valid_prices.iloc[-1]
                                    if not pd.isna(current_price) and current_price > 0:
                                        position_value = static_bh_1m_positions[ticker]['shares'] * current_price
                                        static_bh_1m_positions[ticker]['value'] = position_value
                                        static_bh_1m_invested_value += position_value
                                    else:
                                        static_bh_1m_invested_value += static_bh_1m_positions[ticker].get('value', 0.0)
                                else:
                                    static_bh_1m_invested_value += static_bh_1m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        print(f"   ⚠️ Error updating Static BH 1M position for {ticker}: {e}")

        static_bh_1m_portfolio_value = static_bh_1m_invested_value + static_bh_1m_cash
        static_bh_1m_portfolio_history.append(static_bh_1m_portfolio_value)

        # Update STATIC BH MONTHLY variants portfolio values daily
        for monthly_var in [
            ('1Y', config.ENABLE_STATIC_BH_1Y_MONTHLY, static_bh_1y_monthly_positions, 'static_bh_1y_monthly'),
            ('6M', config.ENABLE_STATIC_BH_6M_MONTHLY, static_bh_6m_monthly_positions, 'static_bh_6m_monthly'),
            ('3M', config.ENABLE_STATIC_BH_3M_MONTHLY, static_bh_3m_monthly_positions, 'static_bh_3m_monthly'),
            ('1M', config.ENABLE_STATIC_BH_1M_MONTHLY, static_bh_1m_monthly_positions, 'static_bh_1m_monthly'),
        ]:
            label, enabled, positions_dict, var_prefix = monthly_var
            if enabled:
                invested = 0.0
                for ticker in list(positions_dict.keys()):
                    try:
                        if ticker in ticker_data_grouped:
                            td = ticker_data_grouped[ticker]
                            price_data = td.loc[:current_date]
                            if not price_data.empty:
                                valid_close = price_data['Close'].dropna()
                                if len(valid_close) > 0:
                                    cp = valid_close.iloc[-1]
                                    positions_dict[ticker]['value'] = positions_dict[ticker]['shares'] * cp
                                    invested += positions_dict[ticker]['value']
                                else:
                                    invested += positions_dict[ticker].get('value', 0.0)
                            else:
                                invested += positions_dict[ticker].get('value', 0.0)
                    except Exception:
                        invested += positions_dict[ticker].get('value', 0.0)
                if var_prefix == 'static_bh_1y_monthly':
                    static_bh_1y_monthly_portfolio_value = invested + static_bh_1y_monthly_cash
                    static_bh_1y_monthly_portfolio_history.append(static_bh_1y_monthly_portfolio_value)
                elif var_prefix == 'static_bh_6m_monthly':
                    static_bh_6m_monthly_portfolio_value = invested + static_bh_6m_monthly_cash
                    static_bh_6m_monthly_portfolio_history.append(static_bh_6m_monthly_portfolio_value)
                elif var_prefix == 'static_bh_3m_monthly':
                    static_bh_3m_monthly_portfolio_value = invested + static_bh_3m_monthly_cash
                    static_bh_3m_monthly_portfolio_history.append(static_bh_3m_monthly_portfolio_value)
                elif var_prefix == 'static_bh_1m_monthly':
                    static_bh_1m_monthly_portfolio_value = invested + static_bh_1m_monthly_cash
                    static_bh_1m_monthly_portfolio_history.append(static_bh_1m_monthly_portfolio_value)

        # Update portfolio value (invested + cash) at end of each day
        invested_value = 0.0
        for ticker in current_portfolio_stocks:
            if ticker in positions and positions[ticker]['shares'] > 0:
                # Get current price
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = positions[ticker]['shares'] * current_price
                                    invested_value += position_value

                                    # ✅ NEW: Track daily contribution for this stock
                                    old_value = positions[ticker].get('value', position_value)
                                    daily_change = position_value - old_value

                                    if ticker not in stock_performance_tracking:
                                        stock_performance_tracking[ticker] = {
                                            'days_held': 0,
                                            'contribution': 0.0,
                                            'max_shares': 0.0,
                                            'entry_value': position_value,
                                            'total_invested': 0.0
                                        }

                                    stock_performance_tracking[ticker]['days_held'] += 1
                                    stock_performance_tracking[ticker]['contribution'] += daily_change
                                    stock_performance_tracking[ticker]['max_shares'] = max(
                                        stock_performance_tracking[ticker]['max_shares'],
                                        positions[ticker]['shares']
                                    )

                                    # Update stored position value
                                    positions[ticker]['value'] = position_value
                                else:
                                    # Use previous value if current price is invalid
                                    invested_value += positions[ticker].get('value', 0.0)
                            else:
                                invested_value += positions[ticker].get('value', 0.0)
                except Exception:
                    # Keep previous value if price lookup fails
                    invested_value += positions[ticker].get('value', 0.0)

        # Validate values before calculation
        if pd.isna(invested_value):
            invested_value = 0.0
        if pd.isna(cash_balance):
            cash_balance = 0.0

        total_portfolio_value = invested_value + cash_balance

        # Debug: Log if portfolio value is 0 (might indicate an issue)
        if day_count == 1 and total_portfolio_value == 0:
            print(f"   ⚠️ DEBUG: Day 1 portfolio value is 0 (invested: {invested_value}, cash: {cash_balance})")

        # Update portfolio value history
        portfolio_values_history.append(total_portfolio_value)

        # AI CHAMPION / AI REGIME: at day end, use the completed shared source
        # snapshot before meta and voting consume the same information.
        try:
            def _update_ai_selector_portfolio(positions, cash):
                invested = 0.0
                for ticker in list(positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                positions[ticker]['value'] = positions[ticker]['shares'] * current_price
                                invested += positions[ticker]['value']
                                continue
                        invested += positions[ticker].get('value', 0.0)
                    except Exception:
                        invested += positions[ticker].get('value', 0.0)
                return invested + cash

            if config.ENABLE_AI_CHAMPION:
                try:
                    from ai_champion_strategy import AIChampionAllocator, select_ai_champion_stocks

                    if ai_champion_allocator is None:
                        ai_champion_allocator = AIChampionAllocator(
                            retrain_days=config.AI_CHAMPION_RETRAIN_DAYS,
                            forward_days=config.AI_CHAMPION_FORWARD_DAYS,
                            confidence_threshold=config.AI_CHAMPION_CONFIDENCE_THRESHOLD,
                            hold_margin=config.AI_CHAMPION_HOLD_MARGIN,
                        )
                        ai_champion_allocator.load_model()

                    strategy_values_for_champion = build_strategy_values_from_locals(
                        locals(),
                        AI_CHAMPION_STRATEGY_SOURCES,
                    )
                    ai_champion_allocator.record_daily_values(strategy_values_for_champion)

                    if ai_champion_allocator.should_retrain():
                        print(f"   🧠 AI Champion: Training model (day {day_count})...")
                        ai_champion_allocator.train_model(ticker_data_grouped, business_days[:day_count])

                    ai_champion_current_strategy = ai_champion_allocator.predict_best_strategy(ticker_data_grouped, current_date)

                    if ai_champion_current_strategy is not None:
                        new_ai_champion_stocks = select_ai_champion_stocks(
                            initial_top_tickers,
                            ticker_data_grouped,
                            current_date,
                            PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                            ai_champion_current_strategy,
                            ai_elite_models=ai_elite_models,
                        )
                    else:
                        new_ai_champion_stocks = []

                    if new_ai_champion_stocks:
                        print(f"   📊 AI Champion Day {day_count}: {new_ai_champion_stocks}")
                        ai_champion_positions, ai_champion_cash, current_ai_champion_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name=f"AI Champion ({str(ai_champion_current_strategy)[:10]})",
                            current_stocks=current_ai_champion_stocks,
                            new_stocks=new_ai_champion_stocks,
                            positions=ai_champion_positions,
                            cash=ai_champion_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                            force_rebalance=not ai_champion_initialized
                        )
                        strategies_rebalanced_today['AI Champion'] = rebalanced_flag
                        ai_champion_transaction_costs += rc
                        ai_champion_initialized = True
                except Exception as e:
                    print(f"   ⚠️ AI Champion error: {e}")

                ai_champion_portfolio_value = _update_ai_selector_portfolio(ai_champion_positions, ai_champion_cash)
                ai_champion_portfolio_history.append(ai_champion_portfolio_value)

            if config.ENABLE_AI_REGIME:
                try:
                    from ai_regime_strategy import AIRegimeAllocator, select_ai_regime_stocks

                    if ai_regime_allocator is None:
                        ai_regime_allocator = AIRegimeAllocator(retrain_days=1, forward_days=1)
                        ai_regime_allocator.load_model()

                    strategy_values_for_regime = build_strategy_values_from_locals(
                        locals(),
                        AI_REGIME_STRATEGY_SOURCES,
                    )
                    ai_regime_allocator.record_daily_values(strategy_values_for_regime)

                    if ai_regime_allocator.should_retrain():
                        print(f"   🧠 AI Regime: Training model (day {day_count})...")
                        ai_regime_allocator.train_model(ticker_data_grouped, business_days[:day_count])

                    ai_regime_current_strategy = ai_regime_allocator.predict_best_strategy(ticker_data_grouped, current_date)

                    if ai_regime_current_strategy is not None:
                        new_ai_regime_stocks = select_ai_regime_stocks(
                            initial_top_tickers,
                            ticker_data_grouped,
                            current_date,
                            PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                            ai_regime_current_strategy,
                            ai_elite_models=ai_elite_models
                        )
                    else:
                        new_ai_regime_stocks = []

                    if new_ai_regime_stocks:
                        print(f"   📊 AI Regime Day {day_count}: {new_ai_regime_stocks}")
                        ai_regime_positions, ai_regime_cash, current_ai_regime_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                            strategy_name=f"AI Regime ({str(ai_regime_current_strategy)[:10]})",
                            current_stocks=current_ai_regime_stocks,
                            new_stocks=new_ai_regime_stocks,
                            positions=ai_regime_positions,
                            cash=ai_regime_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                            force_rebalance=not current_ai_regime_stocks
                        )
                        strategies_rebalanced_today['AI Regime'] = rebalanced_flag
                        ai_regime_transaction_costs += rc
                        ai_regime_initialized = True
                except Exception as e:
                    print(f"   ⚠️ AI Regime error: {e}")

                ai_regime_portfolio_value = _update_ai_selector_portfolio(ai_regime_positions, ai_regime_cash)
                ai_regime_portfolio_history.append(ai_regime_portfolio_value)

            if config.ENABLE_AI_REGIME_MONTHLY:
                try:
                    from ai_regime_strategy import AIRegimeMonthlyAllocator, select_ai_regime_stocks

                    if ai_regime_monthly_allocator is None:
                        ai_regime_monthly_allocator = AIRegimeMonthlyAllocator(forward_days=1)
                        ai_regime_monthly_allocator.load_model()

                    strategy_values_for_regime = build_strategy_values_from_locals(
                        locals(),
                        AI_REGIME_STRATEGY_SOURCES,
                    )
                    ai_regime_monthly_allocator.record_daily_values(strategy_values_for_regime)

                    if ai_regime_monthly_allocator.should_retrain():
                        print(f"   🧠 AI Regime Monthly: Training model (day {day_count})...")
                        ai_regime_monthly_allocator.train_model(ticker_data_grouped, business_days[:day_count])

                    ai_regime_monthly_current_strategy = ai_regime_monthly_allocator.predict_best_strategy(ticker_data_grouped, current_date)

                    if ai_regime_monthly_allocator.should_rebalance(current_date):
                        if ai_regime_monthly_current_strategy is not None:
                            new_ai_regime_monthly_stocks = select_ai_regime_stocks(
                                initial_top_tickers,
                                ticker_data_grouped,
                                current_date,
                                PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                                ai_regime_monthly_current_strategy,
                                ai_elite_models=ai_elite_models
                            )
                        else:
                            new_ai_regime_monthly_stocks = []

                        if new_ai_regime_monthly_stocks:
                            print(f"   📊 AI Regime Monthly Day {day_count}: {new_ai_regime_monthly_stocks}")
                            ai_regime_monthly_positions, ai_regime_monthly_cash, current_ai_regime_monthly_stocks, rc, rebalanced_flag = _smart_rebalance_portfolio(
                                strategy_name=f"AI Regime Mth ({str(ai_regime_monthly_current_strategy)[:10]})",
                                current_stocks=current_ai_regime_monthly_stocks,
                                new_stocks=new_ai_regime_monthly_stocks,
                                positions=ai_regime_monthly_positions,
                                cash=ai_regime_monthly_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                buffer_size=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                                force_rebalance=not ai_regime_monthly_initialized
                            )
                            strategies_rebalanced_today['AI Regime Mth'] = rebalanced_flag
                            ai_regime_monthly_transaction_costs += rc
                            ai_regime_monthly_initialized = True
                except Exception as e:
                    print(f"   ⚠️ AI Regime Monthly error: {e}")

                ai_regime_monthly_portfolio_value = _update_ai_selector_portfolio(ai_regime_monthly_positions, ai_regime_monthly_cash)
                ai_regime_monthly_portfolio_history.append(ai_regime_monthly_portfolio_value)
        except Exception as e:
            print(f"   ⚠️ AI selector snapshot error: {e}")

        # META-STRATEGY SELECTORS: at day end, score the same completed shared
        # source snapshot used by Voting Ensemble before rebalancing meta books.
        try:
            meta_source_strategy_data = _build_daily_strategy_data(locals())
            meta_strategy_values = {
                strategy_key: strategy_data['value']
                for strategy_key, strategy_data in meta_source_strategy_data.items()
                if strategy_key in META_SUB_STRATEGIES
            }

            meta_strategy_manager.record_daily_values(meta_strategy_values)
            meta_selections = meta_strategy_manager.get_all_selections()

            if config.ENABLE_META_WEIGHTED_COMPOSITE:
                try:
                    selected_strategy, _ = meta_selections['meta_weighted_composite']
                    if selected_strategy:
                        meta_weighted_composite_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_weighted_composite_stocks):
                            meta_weighted_composite_positions, meta_weighted_composite_cash, current_meta_weighted_composite_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Weighted ({selected_strategy[:8]})",
                                current_stocks=current_meta_weighted_composite_stocks,
                                new_stocks=new_stocks,
                                positions=meta_weighted_composite_positions,
                                cash=meta_weighted_composite_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_weighted_composite_initialized
                            )
                            meta_weighted_composite_transaction_costs += rc
                            meta_weighted_composite_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Weighted error: {e}")

            if config.ENABLE_META_TIERED_SELECTION:
                try:
                    selected_strategy, _ = meta_selections['meta_tiered_selection']
                    if selected_strategy:
                        meta_tiered_selection_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_tiered_selection_stocks):
                            meta_tiered_selection_positions, meta_tiered_selection_cash, current_meta_tiered_selection_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Tiered ({selected_strategy[:8]})",
                                current_stocks=current_meta_tiered_selection_stocks,
                                new_stocks=new_stocks,
                                positions=meta_tiered_selection_positions,
                                cash=meta_tiered_selection_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_tiered_selection_initialized
                            )
                            meta_tiered_selection_transaction_costs += rc
                            meta_tiered_selection_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Tiered error: {e}")

            if config.ENABLE_META_ENSEMBLE_ALLOC:
                try:
                    selected_strategy, _ = meta_selections['meta_ensemble_alloc']
                    if selected_strategy:
                        meta_ensemble_alloc_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_ensemble_alloc_stocks):
                            meta_ensemble_alloc_positions, meta_ensemble_alloc_cash, current_meta_ensemble_alloc_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Ensemble ({selected_strategy[:8]})",
                                current_stocks=current_meta_ensemble_alloc_stocks,
                                new_stocks=new_stocks,
                                positions=meta_ensemble_alloc_positions,
                                cash=meta_ensemble_alloc_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_ensemble_alloc_initialized
                            )
                            meta_ensemble_alloc_transaction_costs += rc
                            meta_ensemble_alloc_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Ensemble error: {e}")

            if config.ENABLE_META_REGIME_BASED:
                try:
                    selected_strategy, _ = meta_selections['meta_regime_based']
                    if selected_strategy:
                        meta_regime_based_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_regime_based_stocks):
                            meta_regime_based_positions, meta_regime_based_cash, current_meta_regime_based_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Regime ({selected_strategy[:8]})",
                                current_stocks=current_meta_regime_based_stocks,
                                new_stocks=new_stocks,
                                positions=meta_regime_based_positions,
                                cash=meta_regime_based_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_regime_based_initialized
                            )
                            meta_regime_based_transaction_costs += rc
                            meta_regime_based_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Regime error: {e}")

            if config.ENABLE_META_RECENCY_WEIGHTED:
                try:
                    selected_strategy, _ = meta_selections['meta_recency_weighted']
                    if selected_strategy:
                        meta_recency_weighted_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_recency_weighted_stocks):
                            meta_recency_weighted_positions, meta_recency_weighted_cash, current_meta_recency_weighted_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Recency ({selected_strategy[:8]})",
                                current_stocks=current_meta_recency_weighted_stocks,
                                new_stocks=new_stocks,
                                positions=meta_recency_weighted_positions,
                                cash=meta_recency_weighted_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_recency_weighted_initialized
                            )
                            meta_recency_weighted_transaction_costs += rc
                            meta_recency_weighted_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Recency error: {e}")

            if config.ENABLE_META_EFFICIENCY_RATIO:
                try:
                    selected_strategy, _ = meta_selections['meta_efficiency_ratio']
                    if selected_strategy:
                        meta_efficiency_ratio_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_efficiency_ratio_stocks):
                            meta_efficiency_ratio_positions, meta_efficiency_ratio_cash, current_meta_efficiency_ratio_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Efficiency ({selected_strategy[:8]})",
                                current_stocks=current_meta_efficiency_ratio_stocks,
                                new_stocks=new_stocks,
                                positions=meta_efficiency_ratio_positions,
                                cash=meta_efficiency_ratio_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_efficiency_ratio_initialized
                            )
                            meta_efficiency_ratio_transaction_costs += rc
                            meta_efficiency_ratio_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Efficiency error: {e}")

            if config.ENABLE_META_MIN_VARIANCE:
                try:
                    selected_strategy, _ = meta_selections['meta_min_variance']
                    if selected_strategy:
                        meta_min_variance_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_min_variance_stocks):
                            meta_min_variance_positions, meta_min_variance_cash, current_meta_min_variance_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta MinVar ({selected_strategy[:8]})",
                                current_stocks=current_meta_min_variance_stocks,
                                new_stocks=new_stocks,
                                positions=meta_min_variance_positions,
                                cash=meta_min_variance_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_min_variance_initialized
                            )
                            meta_min_variance_transaction_costs += rc
                            meta_min_variance_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta MinVar error: {e}")

            if config.ENABLE_META_BAYESIAN:
                try:
                    selected_strategy, _ = meta_selections['meta_bayesian']
                    if selected_strategy:
                        meta_bayesian_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_bayesian_stocks):
                            meta_bayesian_positions, meta_bayesian_cash, current_meta_bayesian_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Bayesian ({selected_strategy[:8]})",
                                current_stocks=current_meta_bayesian_stocks,
                                new_stocks=new_stocks,
                                positions=meta_bayesian_positions,
                                cash=meta_bayesian_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_bayesian_initialized
                            )
                            meta_bayesian_transaction_costs += rc
                            meta_bayesian_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Bayesian error: {e}")

            if config.ENABLE_META_ADAPTIVE_CONVEX:
                try:
                    selected_strategy, _ = meta_selections['meta_adaptive_convex']
                    if selected_strategy:
                        meta_adaptive_convex_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_adaptive_convex_stocks):
                            meta_adaptive_convex_positions, meta_adaptive_convex_cash, current_meta_adaptive_convex_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Adaptive ({selected_strategy[:8]})",
                                current_stocks=current_meta_adaptive_convex_stocks,
                                new_stocks=new_stocks,
                                positions=meta_adaptive_convex_positions,
                                cash=meta_adaptive_convex_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_adaptive_convex_initialized
                            )
                            meta_adaptive_convex_transaction_costs += rc
                            meta_adaptive_convex_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Adaptive error: {e}")

            if config.ENABLE_META_CONSENSUS:
                try:
                    selected_strategy, _ = meta_selections['meta_consensus']
                    if selected_strategy:
                        meta_consensus_selected_strategy = selected_strategy
                        new_stocks = select_meta_strategy_stocks(selected_strategy, initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE)
                        if new_stocks and set(new_stocks) != set(current_meta_consensus_stocks):
                            meta_consensus_positions, meta_consensus_cash, current_meta_consensus_stocks, rc, _ = _smart_rebalance_portfolio(
                                strategy_name=f"Meta Consensus ({selected_strategy[:8]})",
                                current_stocks=current_meta_consensus_stocks,
                                new_stocks=new_stocks,
                                positions=meta_consensus_positions,
                                cash=meta_consensus_cash,
                                ticker_data_grouped=ticker_data_grouped,
                                current_date=current_date,
                                transaction_cost=TRANSACTION_COST,
                                portfolio_size=PORTFOLIO_SIZE,
                                force_rebalance=not meta_consensus_initialized
                            )
                            meta_consensus_transaction_costs += rc
                            meta_consensus_initialized = True
                except Exception as e:
                    print(f"   ⚠️ Meta Consensus error: {e}")

            if config.ENABLE_META_WEIGHTED_COMPOSITE:
                meta_weighted_composite_value = _update_meta_portfolio(meta_weighted_composite_positions, meta_weighted_composite_cash)
                meta_weighted_composite_history.append(meta_weighted_composite_value)

            if config.ENABLE_META_TIERED_SELECTION:
                meta_tiered_selection_value = _update_meta_portfolio(meta_tiered_selection_positions, meta_tiered_selection_cash)
                meta_tiered_selection_history.append(meta_tiered_selection_value)

            if config.ENABLE_META_ENSEMBLE_ALLOC:
                meta_ensemble_alloc_value = _update_meta_portfolio(meta_ensemble_alloc_positions, meta_ensemble_alloc_cash)
                meta_ensemble_alloc_history.append(meta_ensemble_alloc_value)

            if config.ENABLE_META_REGIME_BASED:
                meta_regime_based_value = _update_meta_portfolio(meta_regime_based_positions, meta_regime_based_cash)
                meta_regime_based_history.append(meta_regime_based_value)

            if config.ENABLE_META_RECENCY_WEIGHTED:
                meta_recency_weighted_value = _update_meta_portfolio(meta_recency_weighted_positions, meta_recency_weighted_cash)
                meta_recency_weighted_history.append(meta_recency_weighted_value)

            if config.ENABLE_META_EFFICIENCY_RATIO:
                meta_efficiency_ratio_value = _update_meta_portfolio(meta_efficiency_ratio_positions, meta_efficiency_ratio_cash)
                meta_efficiency_ratio_history.append(meta_efficiency_ratio_value)

            if config.ENABLE_META_MIN_VARIANCE:
                meta_min_variance_value = _update_meta_portfolio(meta_min_variance_positions, meta_min_variance_cash)
                meta_min_variance_history.append(meta_min_variance_value)

            if config.ENABLE_META_BAYESIAN:
                meta_bayesian_value = _update_meta_portfolio(meta_bayesian_positions, meta_bayesian_cash)
                meta_bayesian_history.append(meta_bayesian_value)

            if config.ENABLE_META_ADAPTIVE_CONVEX:
                meta_adaptive_convex_value = _update_meta_portfolio(meta_adaptive_convex_positions, meta_adaptive_convex_cash)
                meta_adaptive_convex_history.append(meta_adaptive_convex_value)

            if config.ENABLE_META_CONSENSUS:
                meta_consensus_value = _update_meta_portfolio(meta_consensus_positions, meta_consensus_cash)
                meta_consensus_history.append(meta_consensus_value)
        except Exception as e:
            print(f"   ⚠️ Meta strategy snapshot error: {e}")

        # VOTING ENSEMBLE: at day end, vote using the 3 best-performing strategies
        # based on the same in-memory day-end source snapshot family.
        if config.ENABLE_VOTING_ENSEMBLE:
            try:
                voting_source_strategy_data = _build_daily_strategy_data(locals())
                new_voting_ensemble_stocks, voting_sources, ranked_votes, vote_details = _select_voting_ensemble_stocks_from_strategy_data(
                    source_strategy_data=voting_source_strategy_data,
                    top_n=PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE,
                    max_source_strategies=3,
                    excluded_strategy_keys=['voting_ensemble'],
                )

                if voting_sources:
                    print(f"\n   🗳️  Voting Ensemble Strategy: Using top {len(voting_sources)} daily performers on {current_date.strftime('%Y-%m-%d')}")
                    for source_rank, (_, display_name, value, tickers) in enumerate(voting_sources, 1):
                        source_return_pct = ((value - initial_capital_needed) / initial_capital_needed) * 100
                        print(
                            f"      {source_rank}. {display_name:<20} "
                            f"${value:>11,.0f} ({source_return_pct:+6.1f}%) {len(tickers)} ticker(s)"
                        )

                    print(f"\n   📊 {len(voting_sources)} strategies contributed votes")
                    print(f"\n   📊 VOTING RESULTS (Top {len(new_voting_ensemble_stocks)} stocks):")
                    print(f"   {'Rank':<5} {'Ticker':<8} {'Votes':<8} {'Strategies':<50}")
                    print(f"   {'-'*5} {'-'*8} {'-'*8} {'-'*50}")

                    for rank, (stock, votes) in enumerate(ranked_votes[:PORTFOLIO_SIZE + PORTFOLIO_BUFFER_SIZE], 1):
                        voting_strategies = vote_details.get(stock, [])
                        strategies_str = ', '.join(voting_strategies[:3])
                        if len(voting_strategies) > 3:
                            strategies_str += f" (+{len(voting_strategies)-3} more)"
                        vote_bar = "█" * votes + "░" * (len(voting_sources) - votes)
                        print(f"   {rank:<5} {stock:<8} {votes}/{len(voting_sources)} {vote_bar} {strategies_str}")

                    print(f"\n   🗳️  Voting Ensemble selected {len(new_voting_ensemble_stocks)} tickers:")
                    print(f"   ✅ {new_voting_ensemble_stocks}")

                    voting_ensemble_positions, voting_ensemble_cash, current_voting_ensemble_stocks, rebalance_costs, rebalanced_flag = _smart_rebalance_portfolio(
                        strategy_name="Voting Ensemble",
                        current_stocks=current_voting_ensemble_stocks,
                        new_stocks=new_voting_ensemble_stocks,
                        positions=voting_ensemble_positions,
                        cash=voting_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_voting_ensemble_stocks
                    )
                    strategies_rebalanced_today['Voting Ensemble'] = rebalanced_flag
                    voting_ensemble_transaction_costs += rebalance_costs
                    voting_ensemble_last_rebalance_value = voting_ensemble_portfolio_value
                else:
                    print(f"   ⚠️ Voting Ensemble: No eligible source strategies with tickers on {current_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"   ⚠️ Voting Ensemble strategy error: {e}")

            voting_ensemble_invested_value = 0.0
            for ticker in list(voting_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = voting_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            voting_ensemble_positions[ticker]['value'] = position_value
                            voting_ensemble_invested_value += position_value
                        else:
                            voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating voting ensemble position for {ticker}: {e}")
                    voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)

            voting_ensemble_portfolio_value = voting_ensemble_invested_value + voting_ensemble_cash
            voting_ensemble_portfolio_history.append(voting_ensemble_portfolio_value)

        # Periodic progress update
        if day_count % 50 == 0:
            print(f"   📈 Processed {day_count}/{len(business_days)} days, portfolio: {current_portfolio_stocks}")

        benchmark_top_start_portfolio_value = _mark_to_market_value(
            benchmark_top_start_positions,
            benchmark_top_start_cash,
            ticker_data_grouped,
            current_date,
        )
        benchmark_top_start_portfolio_history.append(benchmark_top_start_portfolio_value)

        # === DAILY SUMMARY ===
        # Build strategy data dynamically from local variables using centralized helper
        daily_strat_data = _build_daily_strategy_data(locals())
        daily_strat_data['benchmark_top_start'] = {
            'value': benchmark_top_start_portfolio_value,
            'history': benchmark_top_start_portfolio_history,
            'cash': benchmark_top_start_cash,
            'positions': len(benchmark_top_start_positions),
            'tickers': list(benchmark_top_start_positions.keys()),
            'display_name': benchmark_top_start_name,
        }

        if daily_strat_data:
            print(f"\n📊 DAILY SUMMARY - Day {day_count} ({current_date.strftime('%Y-%m-%d')})")
            print("=" * 80)

            # Sort by value descending
            sorted_strategies = sorted(daily_strat_data.items(), key=lambda x: x[1]['value'], reverse=True)

            # Track Top 5 consistency
            today_top5_names = []
            for rank, (key, sdata) in enumerate(sorted_strategies[:5], 1):
                dname = sdata['display_name']
                today_top5_names.append(dname)
                if dname not in top5_consistency_counts:
                    top5_consistency_counts[dname] = 0
                top5_consistency_counts[dname] += 1

            recent_top5_history.append(today_top5_names)
            if len(recent_top5_history) > 5:
                recent_top5_history = recent_top5_history[-5:]

            # Show ALL strategies with cash and allocation info
            print(f"{'Rank':<5} {'Strategy':<20} {'Value':>12} {'Return':>9} {'StdDev':>8} {'Ann.Ret':>10} {'Cash':>12} {'Pos':>5}")
            print("-" * 95)

            # Build strategy_details and strategy_to_history from the centralized data
            strategy_details = []
            strategy_to_history = {}
            for key, sdata in sorted_strategies:
                dname = sdata['display_name']
                value = sdata['value']
                strat_cash = sdata['cash']
                num_pos = sdata['positions']
                invested = value - strat_cash
                history = sdata['history']

                strategy_details.append((dname, value, strat_cash, num_pos, invested))
                if history:
                    strategy_to_history[dname] = history

            # Display all strategies
            for i, (name, value, strat_cash, num_pos, invested) in enumerate(strategy_details, 1):
                return_pct = ((value - initial_capital_needed) / initial_capital_needed) * 100
                if day_count > 0:
                    total_return_multiplier = value / initial_capital_needed
                    annualized_return = (total_return_multiplier ** (252.0 / day_count) - 1) * 100
                else:
                    annualized_return = 0.0

                std_dev = calculate_std_dev(strategy_to_history.get(name, []))

                # Add visual marker for rebalancing
                display_name = name
                if strategies_rebalanced_today.get(name, False):
                    display_name = f"🔄 {name}"
                print(f"{i:<5} {display_name:<20} ${value:>11,.0f} {return_pct:>+8.1f}% {std_dev:>7.1f}% {annualized_return:>+9.1f}% ${strat_cash:>11,.0f} {num_pos:>5}")

            # Show Top 5 Consistency Score
            if day_count >= 3 and top5_consistency_counts:
                print(f"\n🏆 TOP 5 CONSISTENCY (Days in Top 5 / {day_count} total days):")
                sorted_consistency = sorted(top5_consistency_counts.items(), key=lambda x: x[1], reverse=True)
                for rank, (strat_name, count) in enumerate(sorted_consistency[:10], 1):
                    pct = (count / day_count) * 100
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    print(f"   {rank:<3} {strat_name:<20} {count:>3}/{day_count} days ({pct:>5.1f}%) {bar}")

                recent_window = min(day_count, 5)
                if recent_window > 0 and recent_top5_history:
                    recent_counts = {}
                    for top5_names in recent_top5_history[-recent_window:]:
                        for strat_name in top5_names:
                            recent_counts[strat_name] = recent_counts.get(strat_name, 0) + 1

                    sorted_recent_consistency = sorted(recent_counts.items(), key=lambda x: x[1], reverse=True)
                    print(f"\n🏆 TOP 5 CONSISTENCY - LAST {recent_window} DAYS:")
                    for rank, (strat_name, count) in enumerate(sorted_recent_consistency[:10], 1):
                        pct = (count / recent_window) * 100
                        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                        print(f"   {rank:<3} {strat_name:<20} {count:>3}/{recent_window} days ({pct:>5.1f}%) {bar}")

            # Statistical Significance Testing
            if day_count >= 10 and len(strategy_details) >= 2:
                print(f"\n📊 STATISTICAL SIGNIFICANCE (Top strategy vs others):")
                top_strategy_name = strategy_details[0][0]
                top_history = strategy_to_history.get(top_strategy_name, [])

                for i in range(1, min(5, len(strategy_details))):  # Compare with top 4 others
                    other_name = strategy_details[i][0]
                    other_history = strategy_to_history.get(other_name, [])

                    t_stat, p_value = paired_t_test(top_history, other_history)

                    if p_value < 0.05:
                        significance = "✅ SIGNIFICANT"
                    elif p_value < 0.10:
                        significance = "⚠️  MARGINAL"
                    else:
                        significance = "❌ NOT SIGNIFICANT"

                    print(f"   {top_strategy_name[:15]:<15} vs {other_name[:15]:<15}: p={p_value:.3f} {significance}")

            print("=" * 80)
        else:
            print(f"\n📊 DAILY SUMMARY - Day {day_count} ({current_date.strftime('%Y-%m-%d')})")
            print("=" * 80)
            print(f"   📊 No strategies enabled for daily summary.")
            print("=" * 80)

        # === INVERSE ETF HEDGE SUMMARY ===
        from shared_strategies import get_inverse_etf_hedge_log, get_market_conditions
        hedge_log = get_inverse_etf_hedge_log()
        today_hedges = [h for h in hedge_log if h[0].date() == current_date.date()]
        if today_hedges:
            print(f"\n🛡️ INVERSE ETF HEDGES TODAY:")
            for date, etf, decline in today_hedges:
                print(f"   {etf}: Added as hedge (market down {decline:.1%})")

        # Show market conditions
        market_cond = get_market_conditions(ticker_data_grouped, current_date)
        if market_cond:
            sp500_3m = market_cond.get('sp500_3m', 0)
            nasdaq_3m = market_cond.get('nasdaq_3m', 0)
            print(f"   📉 Market: SPY 3M={sp500_3m:+.1%}, QQQ 3M={nasdaq_3m:+.1%}")

        # === DAILY ALL STRATEGIES SUMMARY TABLE ===
        print("\n" + "="*120)
        print("                     📊 DAILY ALL STRATEGIES PERFORMANCE SUMMARY 📊")
        print("="*120)

        # Collect all strategy data for each day
        strategy_data = []

        # Build (name, history) list for the summary table
        strategy_histories = [
            (benchmark_top_start_name, benchmark_top_start_portfolio_history),
            ("Static BH 1Y",        static_bh_1y_portfolio_history        if config.ENABLE_STATIC_BH else None),
            ("Static BH 6M",        static_bh_6m_portfolio_history        if config.ENABLE_STATIC_BH_6M else None),
            ("Static BH 3M",        static_bh_3m_portfolio_history        if config.ENABLE_STATIC_BH else None),
            ("Dynamic BH 1Y",       dynamic_bh_portfolio_history          if config.ENABLE_DYNAMIC_BH_1Y else None),
            ("Dynamic BH 6M",       dynamic_bh_6m_portfolio_history       if config.ENABLE_DYNAMIC_BH_6M else None),
            ("Dynamic BH 3M",       dynamic_bh_3m_portfolio_history       if config.ENABLE_DYNAMIC_BH_3M else None),
            ("Dynamic BH 1M",       dynamic_bh_1m_portfolio_history       if config.ENABLE_DYNAMIC_BH_1M else None),
            ("Risk-Adj Mom",        risk_adj_mom_portfolio_history        if config.ENABLE_RISK_ADJ_MOM else None),
            ("Mean Reversion",      mean_reversion_portfolio_history      if config.ENABLE_MEAN_REVERSION else None),
            ("Quality+Mom",         quality_momentum_portfolio_history    if config.ENABLE_QUALITY_MOM else None),
            ("Mom-AI Hybrid",       momentum_ai_hybrid_portfolio_history  if config.ENABLE_MOMENTUM_AI_HYBRID else None),
            ("Vol-Adj Mom",         volatility_adj_mom_portfolio_history  if config.ENABLE_VOLATILITY_ADJ_MOM else None),
            ("1Y/3M Ratio",         ratio_1y_3m_portfolio_history         if config.ENABLE_3M_1Y_RATIO else None),
            ("3M/1Y Ratio",         ratio_3m_1y_portfolio_history         if config.ENABLE_3M_1Y_RATIO else None),
            ("1M/3M Ratio",         ratio_1m_3m_portfolio_history         if config.ENABLE_1M_3M_RATIO else None),
            ("Mom-Vol Hybrid",      momentum_volatility_hybrid_portfolio_history    if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID else None),
            ("Mom-Vol Hybrid 6M",   momentum_volatility_hybrid_6m_portfolio_history if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M else None),
            ("Mom-Vol Hybrid 1Y",   momentum_volatility_hybrid_1y_portfolio_history if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y else None),
            ("Mom-Vol Hybrid 1Y/3M",momentum_volatility_hybrid_1y3m_portfolio_history if config.ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M else None),
            ("Enhanced Volatility", enhanced_volatility_portfolio_history if config.ENABLE_ENHANCED_VOLATILITY else None),
            ("Trend ATR",           trend_atr_portfolio_history           if config.ENABLE_TREND_FOLLOWING_ATR else None),
            ("Dual Momentum",       dual_mom_portfolio_history            if config.ENABLE_DUAL_MOMENTUM else None),
            ("Mom Acceleration",    mom_accel_portfolio_history           if config.ENABLE_MOMENTUM_ACCELERATION else None),
            ("Concentrated 3M",     concentrated_3m_portfolio_history     if config.ENABLE_CONCENTRATED_3M else None),
            ("Price Acceleration",  price_acceleration_portfolio_history  if config.ENABLE_PRICE_ACCELERATION else None),
            ("Elite Hybrid",        elite_hybrid_portfolio_history        if config.ENABLE_ELITE_HYBRID else None),
            ("Elite Risk",          elite_risk_portfolio_history          if config.ENABLE_ELITE_RISK else None),
            ("Risk-Adj Mom 6M",     risk_adj_mom_6m_portfolio_history     if config.ENABLE_RISK_ADJ_MOM_6M else None),
            ("RiskAdj 6M Mth",     risk_adj_mom_6m_monthly_portfolio_history if config.ENABLE_RISK_ADJ_MOM_6M_MONTHLY else None),
            ("Risk-Adj Mom 3M",     risk_adj_mom_3m_portfolio_history     if config.ENABLE_RISK_ADJ_MOM_3M else None),
            ("RiskAdj 3M Mth",     risk_adj_mom_3m_monthly_portfolio_history if config.ENABLE_RISK_ADJ_MOM_3M_MONTHLY else None),
            ("RiskAdj 3M Sent",    risk_adj_mom_3m_sentiment_portfolio_history if config.ENABLE_RISK_ADJ_MOM_3M_SENTIMENT else None),
            ("RiskAdj 3M Up",      risk_adj_mom_3m_market_up_portfolio_history if config.ENABLE_RISK_ADJ_MOM_3M_MARKET_UP else None),
            ("RiskAdj 3M Stop",    risk_adj_mom_3m_with_stops_portfolio_history if config.ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS else None),
            ("VolSweet Mom",       vol_sweet_mom_portfolio_history       if config.ENABLE_VOL_SWEET_MOM else None),
            ("1M VolSweet",        risk_adj_mom_1m_vol_sweet_portfolio_history if config.ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET else None),
            ("Risk-Adj Mom 1M",     risk_adj_mom_1m_portfolio_history     if config.ENABLE_RISK_ADJ_MOM_1M else None),
            ("RiskAdj 1M Mth",     risk_adj_mom_1m_monthly_portfolio_history if config.ENABLE_RISK_ADJ_MOM_1M_MONTHLY else None),
            ("AI Elite",            ai_elite_portfolio_history            if config.ENABLE_AI_ELITE else None),
            ("Top5 Cons Blend",    top5_consistency_blend_portfolio_history if config.ENABLE_TOP5_CONSISTENCY_BLEND else None),
            ("BH 1Y Monthly",       static_bh_1y_monthly_portfolio_history if config.ENABLE_STATIC_BH_1Y_MONTHLY else None),
            ("BH 6M Monthly",       static_bh_6m_monthly_portfolio_history if config.ENABLE_STATIC_BH_6M_MONTHLY else None),
            ("BH 3M Monthly",       static_bh_3m_monthly_portfolio_history if config.ENABLE_STATIC_BH_3M_MONTHLY else None),
            ("BH 1M Monthly",       static_bh_1m_monthly_portfolio_history if config.ENABLE_STATIC_BH_1M_MONTHLY else None),
        ]

        # Filter out disabled strategies
        active_strategies = [(name, history) for name, history in strategy_histories if history is not None and len(history) > 0]

        if len(active_strategies) == 0:
            print(f"⚠️ No strategies enabled for daily summary.")
        else:
            # Create dynamic header based on number of strategies
            num_strategies = len(active_strategies)
            header_parts = [f"{'Date':<12}"]

            # Add column for each strategy
            for i, (name, _) in enumerate(active_strategies):
                short_name = name[:15] if len(name) > 15 else name
                header_parts.append(f"{short_name:<15}")

            header_parts.append(f"{'Best':<10} {'Worst':<10}")
            header_line = " ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))

            # Process each day
            max_days = max(len(history) for _, history in active_strategies)
            for day_idx, current_date in enumerate(business_days[:max_days]):
                daily_values = []

                for strategy_name, history in active_strategies:
                    if day_idx < len(history):
                        value = history[day_idx]
                        if not pd.isna(value) and value > 0:
                            if day_idx == 0:
                                daily_return = 0.0
                            else:
                                prev_value = history[day_idx - 1]
                                if prev_value > 0 and not pd.isna(prev_value):
                                    daily_return = ((value - prev_value) / prev_value) * 100
                                else:
                                    daily_return = 0.0
                            daily_values.append((strategy_name, value, daily_return))

                if len(daily_values) > 0:
                    daily_values.sort(key=lambda x: x[1], reverse=True)
                    best_strategy = daily_values[0][0]
                    worst_strategy = daily_values[-1][0]

                    date_str = current_date.strftime('%Y-%m-%d')
                    line_parts = [f"{date_str:<12}"]

                    for strategy_name, history in active_strategies:
                        if day_idx < len(history):
                            value = history[day_idx]
                            if not pd.isna(value) and value > 0:
                                display_value = f"${value:>8,.0f}"
                                if strategy_name == best_strategy:
                                    display_value = "🥇" + display_value[1:]
                                elif strategy_name == worst_strategy:
                                    display_value = "🔻" + display_value[1:]
                                line_parts.append(f"{display_value:<15}")
                            else:
                                line_parts.append(f"{'N/A':<15}")
                        else:
                            line_parts.append(f"{'--':<15}")

                    best_short = best_strategy[:8] if len(best_strategy) > 8 else best_strategy
                    worst_short = worst_strategy[:8] if len(worst_strategy) > 8 else worst_strategy
                    line_parts.append(f"{best_short:<10} {worst_short:<10}")
                    print(" ".join(line_parts))

            print("-" * 120)
            print(f"\n📈 Summary: Showing all strategies with best/worst identification for each trading day")
            print(f"📊 Total strategies tracked: {len(active_strategies)}")
            print(f"📅 Trading days analyzed: {len(business_days)}")
            print(f"🥇 Best strategy each day marked with 🥇 symbol")
            print(f"🔻 Worst strategy each day marked with 🔻 symbol")

        print("="*120)

        # Release heavy AI Elite model objects between backtest days.
        # They are checkpointed to disk and reloaded when needed.
        if config.ENABLE_AI_ELITE and ai_elite_models:
            ai_elite_models = {}
        if config.ENABLE_AI_ELITE_MONTHLY and ai_elite_monthly_models:
            ai_elite_monthly_models = {}
        if config.ENABLE_AI_CHAMPION and ai_champion_allocator is not None and getattr(ai_champion_allocator, 'retrain_days', None) == 1:
            ai_champion_allocator.release_model_artifacts()
        if config.ENABLE_AI_REGIME and ai_regime_allocator is not None and getattr(ai_regime_allocator, 'retrain_days', None) == 1:
            ai_regime_allocator.release_model_artifacts()
        if config.ENABLE_UNIVERSAL_MODEL and universal_model_strategy is not None and getattr(universal_model_strategy, 'retrain_days', None) == 1:
            universal_model_strategy.release_model_artifacts()
        if config.ENABLE_SAVGOL_TREND and savgol_trend_strategy is not None and getattr(savgol_trend_strategy, 'retrain_days', None) == 1:
            savgol_trend_strategy.release_model_artifacts()
        try:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

    # Build results dictionary - replaces the fragile 149-value return tuple
    # Each strategy has: value, history, costs, cash_deployed
    # Adding a new strategy = just add one dict entry here, no main.py changes needed
    def _strat(value, history, costs, cash):
        return {'value': value, 'history': history, 'costs': costs, 'cash_deployed': cash}

    results = {
        'general': {
            'final_strategy_value': total_portfolio_value,
            'portfolio_values_history': portfolio_values_history,
            'processed_tickers': initial_top_tickers,
            'performance_metrics': [],
            'buy_hold_histories': {},
            'day_count': day_count,
        },
        'strategies': {
            'benchmark_top_start':      _strat(benchmark_top_start_portfolio_value, benchmark_top_start_portfolio_history, benchmark_top_start_transaction_costs, benchmark_top_start_cash),
            'static_bh_1y':             _strat(static_bh_1y_portfolio_value, static_bh_1y_portfolio_history, static_bh_transaction_costs, static_bh_1y_cash),
            'static_bh_6m':             _strat(static_bh_6m_portfolio_value, static_bh_6m_portfolio_history, static_bh_6m_transaction_costs, static_bh_6m_cash),
            'static_bh_3m':             _strat(static_bh_3m_portfolio_value, static_bh_3m_portfolio_history, static_bh_3m_transaction_costs, static_bh_3m_cash),
            'static_bh_1m':             _strat(static_bh_1m_portfolio_value, static_bh_1m_portfolio_history, static_bh_1m_transaction_costs, static_bh_1m_cash),
            'dynamic_bh_1y':            _strat(dynamic_bh_portfolio_value, dynamic_bh_portfolio_history, dynamic_bh_transaction_costs, dynamic_bh_cash),
            'dynamic_bh_6m':            _strat(dynamic_bh_6m_portfolio_value, dynamic_bh_6m_portfolio_history, dynamic_bh_6m_transaction_costs, dynamic_bh_6m_cash),
            'dynamic_bh_3m':            _strat(dynamic_bh_3m_portfolio_value, dynamic_bh_3m_portfolio_history, dynamic_bh_3m_transaction_costs, dynamic_bh_3m_cash),
            'dynamic_bh_1m':            _strat(dynamic_bh_1m_portfolio_value, dynamic_bh_1m_portfolio_history, dynamic_bh_1m_transaction_costs, dynamic_bh_1m_cash),
            'dynamic_bh_1y_vol_filter': _strat(dynamic_bh_1y_vol_filter_portfolio_value, dynamic_bh_1y_vol_filter_portfolio_history, dynamic_bh_1y_vol_filter_transaction_costs, dynamic_bh_1y_vol_filter_cash),
            'dynamic_bh_1y_trailing_stop': _strat(dynamic_bh_1y_trailing_stop_portfolio_value, dynamic_bh_1y_trailing_stop_portfolio_history, dynamic_bh_1y_trailing_stop_transaction_costs, dynamic_bh_1y_trailing_stop_cash),
            'risk_adj_mom':             _strat(risk_adj_mom_portfolio_value, risk_adj_mom_portfolio_history, risk_adj_mom_transaction_costs, risk_adj_mom_cash),
            'mean_reversion':           _strat(mean_reversion_portfolio_value, mean_reversion_portfolio_history, mean_reversion_transaction_costs, mean_reversion_cash),
            'quality_momentum':         _strat(quality_momentum_portfolio_value, quality_momentum_portfolio_history, quality_momentum_transaction_costs, quality_momentum_cash),
            'momentum_ai_hybrid':       _strat(momentum_ai_hybrid_portfolio_value, momentum_ai_hybrid_portfolio_history, momentum_ai_hybrid_transaction_costs, momentum_ai_hybrid_cash),
            'volatility_adj_mom':       _strat(volatility_adj_mom_portfolio_value, volatility_adj_mom_portfolio_history, volatility_adj_mom_transaction_costs, volatility_adj_mom_cash),
            'sector_rotation':          _strat(sector_rotation_portfolio_value, sector_rotation_portfolio_history, sector_rotation_transaction_costs, sector_rotation_cash),
            'ratio_3m_1y':              _strat(ratio_3m_1y_portfolio_value, ratio_3m_1y_portfolio_history, ratio_3m_1y_transaction_costs, ratio_3m_1y_cash),
            'ratio_1y_3m':              _strat(ratio_1y_3m_portfolio_value, ratio_1y_3m_portfolio_history, ratio_1y_3m_transaction_costs, ratio_1y_3m_cash),
            'ratio_1m_3m':              _strat(ratio_1m_3m_portfolio_value, ratio_1m_3m_portfolio_history, ratio_1m_3m_transaction_costs, ratio_1m_3m_cash),
            'momentum_volatility_hybrid': _strat(momentum_volatility_hybrid_portfolio_value, momentum_volatility_hybrid_portfolio_history, momentum_volatility_hybrid_transaction_costs, momentum_volatility_hybrid_cash),
            'momentum_volatility_hybrid_6m': _strat(momentum_volatility_hybrid_6m_portfolio_value, momentum_volatility_hybrid_6m_portfolio_history, momentum_volatility_hybrid_6m_transaction_costs, momentum_volatility_hybrid_6m_cash),
            'momentum_volatility_hybrid_1y': _strat(momentum_volatility_hybrid_1y_portfolio_value, momentum_volatility_hybrid_1y_portfolio_history, momentum_volatility_hybrid_1y_transaction_costs, momentum_volatility_hybrid_1y_cash),
            'momentum_volatility_hybrid_1y3m': _strat(momentum_volatility_hybrid_1y3m_portfolio_value, momentum_volatility_hybrid_1y3m_portfolio_history, momentum_volatility_hybrid_1y3m_transaction_costs, momentum_volatility_hybrid_1y3m_cash),
            'price_acceleration':       _strat(price_acceleration_portfolio_value, price_acceleration_portfolio_history, price_acceleration_transaction_costs, price_acceleration_cash),
            'turnaround':               _strat(turnaround_portfolio_value, turnaround_portfolio_history, turnaround_transaction_costs, turnaround_cash),
            'adaptive_ensemble':        _strat(adaptive_ensemble_portfolio_value, adaptive_ensemble_portfolio_history, adaptive_ensemble_transaction_costs, adaptive_ensemble_cash),
            'volatility_ensemble':      _strat(volatility_ensemble_portfolio_value, volatility_ensemble_portfolio_history, volatility_ensemble_transaction_costs, volatility_ensemble_cash),
            'enhanced_volatility':      _strat(enhanced_volatility_portfolio_value, enhanced_volatility_portfolio_history, enhanced_volatility_transaction_costs, enhanced_volatility_cash),
            'ai_volatility_ensemble':   _strat(ai_volatility_ensemble_portfolio_value, ai_volatility_ensemble_portfolio_history, ai_volatility_ensemble_transaction_costs, ai_volatility_ensemble_cash),
            'multi_tf_ensemble':        _strat(multi_tf_ensemble_portfolio_value, multi_tf_ensemble_portfolio_history, multi_tf_ensemble_transaction_costs, multi_tf_ensemble_cash),
            'multi_tf_intraday_ensemble': _strat(multi_tf_intraday_ensemble_portfolio_value, multi_tf_intraday_ensemble_portfolio_history, multi_tf_intraday_ensemble_transaction_costs, multi_tf_intraday_ensemble_cash),
            'correlation_ensemble':     _strat(correlation_ensemble_portfolio_value, correlation_ensemble_portfolio_history, correlation_ensemble_transaction_costs, correlation_ensemble_cash),
            'dynamic_pool':             _strat(dynamic_pool_portfolio_value, dynamic_pool_portfolio_history, dynamic_pool_transaction_costs, dynamic_pool_cash),
            'sentiment_ensemble':       _strat(sentiment_ensemble_portfolio_value, sentiment_ensemble_portfolio_history, sentiment_ensemble_transaction_costs, sentiment_ensemble_cash),
            'voting_ensemble':          _strat(voting_ensemble_portfolio_value, voting_ensemble_portfolio_history, voting_ensemble_transaction_costs, voting_ensemble_cash),
            'mom_accel':                _strat(mom_accel_portfolio_value, mom_accel_portfolio_history, mom_accel_transaction_costs, mom_accel_cash),
            'concentrated_3m':          _strat(concentrated_3m_portfolio_value, concentrated_3m_portfolio_history, concentrated_3m_transaction_costs, concentrated_3m_cash),
            'dual_momentum':            _strat(dual_mom_portfolio_value, dual_mom_portfolio_history, dual_mom_transaction_costs, dual_mom_cash),
            'trend_atr':                _strat(trend_atr_portfolio_value, trend_atr_portfolio_history, trend_atr_transaction_costs, trend_atr_cash),
            'elite_hybrid':             _strat(elite_hybrid_portfolio_value, elite_hybrid_portfolio_history, elite_hybrid_transaction_costs, elite_hybrid_cash),
            'elite_risk':               _strat(elite_risk_portfolio_value, elite_risk_portfolio_history, elite_risk_transaction_costs, elite_risk_cash),
            'risk_adj_mom_6m':          _strat(risk_adj_mom_6m_portfolio_value, risk_adj_mom_6m_portfolio_history, risk_adj_mom_6m_transaction_costs, risk_adj_mom_6m_cash),
            'risk_adj_mom_6m_monthly':  _strat(risk_adj_mom_6m_monthly_portfolio_value, risk_adj_mom_6m_monthly_portfolio_history, risk_adj_mom_6m_monthly_transaction_costs, risk_adj_mom_6m_monthly_cash),
            'risk_adj_mom_3m':          _strat(risk_adj_mom_3m_portfolio_value, risk_adj_mom_3m_portfolio_history, risk_adj_mom_3m_transaction_costs, risk_adj_mom_3m_cash),
            'risk_adj_mom_3m_monthly':  _strat(risk_adj_mom_3m_monthly_portfolio_value, risk_adj_mom_3m_monthly_portfolio_history, risk_adj_mom_3m_monthly_transaction_costs, risk_adj_mom_3m_monthly_cash),
            'risk_adj_mom_3m_sentiment': _strat(risk_adj_mom_3m_sentiment_portfolio_value, risk_adj_mom_3m_sentiment_portfolio_history, risk_adj_mom_3m_sentiment_transaction_costs, risk_adj_mom_3m_sentiment_cash),
            'risk_adj_mom_3m_market_up': _strat(risk_adj_mom_3m_market_up_portfolio_value, risk_adj_mom_3m_market_up_portfolio_history, risk_adj_mom_3m_market_up_transaction_costs, risk_adj_mom_3m_market_up_cash),
            'risk_adj_mom_3m_with_stops': _strat(risk_adj_mom_3m_with_stops_portfolio_value, risk_adj_mom_3m_with_stops_portfolio_history, risk_adj_mom_3m_with_stops_transaction_costs, risk_adj_mom_3m_with_stops_cash),
            'vol_sweet_mom':            _strat(vol_sweet_mom_portfolio_value, vol_sweet_mom_portfolio_history, vol_sweet_mom_transaction_costs, vol_sweet_mom_cash),
            'risk_adj_mom_1m_vol_sweet': _strat(risk_adj_mom_1m_vol_sweet_portfolio_value, risk_adj_mom_1m_vol_sweet_portfolio_history, risk_adj_mom_1m_vol_sweet_transaction_costs, risk_adj_mom_1m_vol_sweet_cash),
            'bh_1y_volsweet_accel': _strat(bh_1y_volsweet_accel_portfolio_value, bh_1y_volsweet_accel_portfolio_history, bh_1y_volsweet_accel_transaction_costs, bh_1y_volsweet_accel_cash),
            'bh_1y_dynamic_accel': _strat(bh_1y_dynamic_accel_portfolio_value, bh_1y_dynamic_accel_portfolio_history, bh_1y_dynamic_accel_transaction_costs, bh_1y_dynamic_accel_cash),
            'risk_adj_mom_1m':          _strat(risk_adj_mom_1m_portfolio_value, risk_adj_mom_1m_portfolio_history, risk_adj_mom_1m_transaction_costs, risk_adj_mom_1m_cash),
            'risk_adj_mom_1m_monthly':  _strat(risk_adj_mom_1m_monthly_portfolio_value, risk_adj_mom_1m_monthly_portfolio_history, risk_adj_mom_1m_monthly_transaction_costs, risk_adj_mom_1m_monthly_cash),
            'ai_elite':                 _strat(ai_elite_portfolio_value, ai_elite_portfolio_history, ai_elite_transaction_costs, ai_elite_cash) if config.ENABLE_AI_ELITE else _strat(0, [], 0, 0),
            'ai_elite_monthly':         _strat(ai_elite_monthly_portfolio_value, ai_elite_monthly_portfolio_history, ai_elite_monthly_transaction_costs, ai_elite_monthly_cash) if config.ENABLE_AI_ELITE_MONTHLY else _strat(0, [], 0, 0),
            'ai_elite_filtered':        _strat(ai_elite_filtered_portfolio_value, ai_elite_filtered_portfolio_history, ai_elite_filtered_transaction_costs, ai_elite_filtered_cash) if config.ENABLE_AI_ELITE_FILTERED else _strat(0, [], 0, 0),
            'ai_elite_market_up':       _strat(ai_elite_market_up_portfolio_value, ai_elite_market_up_portfolio_history, ai_elite_market_up_transaction_costs, ai_elite_market_up_cash) if config.ENABLE_AI_ELITE_MARKET_UP else _strat(0, [], 0, 0),
            'ai_champion':              _strat(ai_champion_portfolio_value, ai_champion_portfolio_history, ai_champion_transaction_costs, ai_champion_cash) if config.ENABLE_AI_CHAMPION else _strat(0, [], 0, 0),
            'ai_regime':                _strat(ai_regime_portfolio_value, ai_regime_portfolio_history, ai_regime_transaction_costs, ai_regime_cash),
            'ai_regime_monthly':        _strat(ai_regime_monthly_portfolio_value, ai_regime_monthly_portfolio_history, ai_regime_monthly_transaction_costs, ai_regime_monthly_cash),
            'universal_model':          _strat(universal_model_portfolio_value, universal_model_portfolio_history, universal_model_transaction_costs, universal_model_cash),
            'savgol_trend':             _strat(savgol_trend_portfolio_value, savgol_trend_portfolio_history, savgol_trend_transaction_costs, savgol_trend_cash) if config.ENABLE_SAVGOL_TREND else _strat(0, [], 0, 0),
            'inverse_etf_hedge':        _strat(inverse_etf_hedge_portfolio_value, inverse_etf_hedge_portfolio_history, inverse_etf_hedge_transaction_costs, inverse_etf_hedge_cash),
            'analyst_rec':              _strat(analyst_rec_portfolio_value, analyst_rec_portfolio_history, analyst_rec_transaction_costs, analyst_rec_cash),
            'risk_adj_mom_sentiment':   _strat(risk_adj_mom_sentiment_portfolio_value, risk_adj_mom_sentiment_portfolio_history, risk_adj_mom_sentiment_transaction_costs, risk_adj_mom_sentiment_cash),
            'bh_1y_monthly':            _strat(static_bh_1y_monthly_portfolio_value, static_bh_1y_monthly_portfolio_history, static_bh_1y_monthly_transaction_costs, static_bh_1y_monthly_cash),
            'bh_6m_monthly':            _strat(static_bh_6m_monthly_portfolio_value, static_bh_6m_monthly_portfolio_history, static_bh_6m_monthly_transaction_costs, static_bh_6m_monthly_cash),
            'bh_3m_monthly':            _strat(static_bh_3m_monthly_portfolio_value, static_bh_3m_monthly_portfolio_history, static_bh_3m_monthly_transaction_costs, static_bh_3m_monthly_cash),
            'bh_1m_monthly':            _strat(static_bh_1m_monthly_portfolio_value, static_bh_1m_monthly_portfolio_history, static_bh_1m_monthly_transaction_costs, static_bh_1m_monthly_cash),
            # Meta-Strategy Selectors (10 proposals) - now with actual positions
            'meta_weighted_composite':  _strat(meta_weighted_composite_value, meta_weighted_composite_history, meta_weighted_composite_transaction_costs, meta_weighted_composite_cash) if config.ENABLE_META_WEIGHTED_COMPOSITE else _strat(0, [], 0, 0),
            'meta_tiered_selection':    _strat(meta_tiered_selection_value, meta_tiered_selection_history, meta_tiered_selection_transaction_costs, meta_tiered_selection_cash) if config.ENABLE_META_TIERED_SELECTION else _strat(0, [], 0, 0),
            'meta_ensemble_alloc':      _strat(meta_ensemble_alloc_value, meta_ensemble_alloc_history, meta_ensemble_alloc_transaction_costs, meta_ensemble_alloc_cash) if config.ENABLE_META_ENSEMBLE_ALLOC else _strat(0, [], 0, 0),
            'meta_regime_based':        _strat(meta_regime_based_value, meta_regime_based_history, meta_regime_based_transaction_costs, meta_regime_based_cash) if config.ENABLE_META_REGIME_BASED else _strat(0, [], 0, 0),
            'meta_recency_weighted':    _strat(meta_recency_weighted_value, meta_recency_weighted_history, meta_recency_weighted_transaction_costs, meta_recency_weighted_cash) if config.ENABLE_META_RECENCY_WEIGHTED else _strat(0, [], 0, 0),
            'meta_efficiency_ratio':    _strat(meta_efficiency_ratio_value, meta_efficiency_ratio_history, meta_efficiency_ratio_transaction_costs, meta_efficiency_ratio_cash) if config.ENABLE_META_EFFICIENCY_RATIO else _strat(0, [], 0, 0),
            'meta_min_variance':        _strat(meta_min_variance_value, meta_min_variance_history, meta_min_variance_transaction_costs, meta_min_variance_cash) if config.ENABLE_META_MIN_VARIANCE else _strat(0, [], 0, 0),
            'meta_bayesian':            _strat(meta_bayesian_value, meta_bayesian_history, meta_bayesian_transaction_costs, meta_bayesian_cash) if config.ENABLE_META_BAYESIAN else _strat(0, [], 0, 0),
            'meta_adaptive_convex':     _strat(meta_adaptive_convex_value, meta_adaptive_convex_history, meta_adaptive_convex_transaction_costs, meta_adaptive_convex_cash) if config.ENABLE_META_ADAPTIVE_CONVEX else _strat(0, [], 0, 0),
            'meta_consensus':           _strat(meta_consensus_value, meta_consensus_history, meta_consensus_transaction_costs, meta_consensus_cash) if config.ENABLE_META_CONSENSUS else _strat(0, [], 0, 0),
            # Bollinger Bands Strategies
            'bb_mean_reversion':        _strat(bb_mean_reversion_value, bb_mean_reversion_history, bb_mean_reversion_transaction_costs, bb_mean_reversion_cash) if config.ENABLE_BB_MEAN_REVERSION else _strat(0, [], 0, 0),
            'bb_breakout':              _strat(bb_breakout_value, bb_breakout_history, bb_breakout_transaction_costs, bb_breakout_cash) if config.ENABLE_BB_BREAKOUT else _strat(0, [], 0, 0),
            'bb_squeeze_breakout':      _strat(bb_squeeze_breakout_value, bb_squeeze_breakout_history, bb_squeeze_breakout_transaction_costs, bb_squeeze_breakout_cash) if config.ENABLE_BB_SQUEEZE_BREAKOUT else _strat(0, [], 0, 0),
            'bb_rsi_combo':             _strat(bb_rsi_combo_value, bb_rsi_combo_history, bb_rsi_combo_transaction_costs, bb_rsi_combo_cash) if config.ENABLE_BB_RSI_COMBO else _strat(0, [], 0, 0),
            'trend_breakout':           _strat(trend_breakout_value, trend_breakout_history, trend_breakout_transaction_costs, trend_breakout_cash) if config.ENABLE_TREND_BREAKOUT else _strat(0, [], 0, 0),
            # Adaptive Rebalancing Strategies (based on Static BH 1Y)
            'static_bh_1y_vol':         _strat(static_bh_1y_vol_portfolio_value, static_bh_1y_vol_portfolio_history, static_bh_1y_vol_transaction_costs, static_bh_1y_vol_cash) if config.ENABLE_STATIC_BH_1Y_VOLATILITY else _strat(0, [], 0, 0),
            'static_bh_1y_perf':        _strat(static_bh_1y_perf_portfolio_value, static_bh_1y_perf_portfolio_history, static_bh_1y_perf_transaction_costs, static_bh_1y_perf_cash) if config.ENABLE_STATIC_BH_1Y_PERFORMANCE else _strat(0, [], 0, 0),
            'static_bh_1y_mom':         _strat(static_bh_1y_mom_portfolio_value, static_bh_1y_mom_portfolio_history, static_bh_1y_mom_transaction_costs, static_bh_1y_mom_cash) if config.ENABLE_STATIC_BH_1Y_MOMENTUM else _strat(0, [], 0, 0),
            'static_bh_1y_atr':         _strat(static_bh_1y_atr_portfolio_value, static_bh_1y_atr_portfolio_history, static_bh_1y_atr_transaction_costs, static_bh_1y_atr_cash) if config.ENABLE_STATIC_BH_1Y_ATR else _strat(0, [], 0, 0),
            'static_bh_1y_hybrid':      _strat(static_bh_1y_hybrid_portfolio_value, static_bh_1y_hybrid_portfolio_history, static_bh_1y_hybrid_transaction_costs, static_bh_1y_hybrid_cash) if config.ENABLE_STATIC_BH_1Y_HYBRID else _strat(0, [], 0, 0),
            # Enhanced Static BH 1Y Strategies
            'static_bh_1y_volume':       _strat(static_bh_1y_volume_portfolio_value, static_bh_1y_volume_portfolio_history, static_bh_1y_volume_transaction_costs, static_bh_1y_volume_cash) if config.ENABLE_STATIC_BH_1Y_VOLUME_FILTER else _strat(0, [], 0, 0),
            'static_bh_1y_sector':      _strat(static_bh_1y_sector_portfolio_value, static_bh_1y_sector_portfolio_history, static_bh_1y_sector_transaction_costs, static_bh_1y_sector_cash) if config.ENABLE_STATIC_BH_1Y_SECTOR_ROTATION else _strat(0, [], 0, 0),
            'static_bh_1y_perf_threshold': _strat(static_bh_1y_perf_threshold_portfolio_value, static_bh_1y_perf_threshold_portfolio_history, static_bh_1y_perf_threshold_transaction_costs, static_bh_1y_perf_threshold_cash) if config.ENABLE_STATIC_BH_1Y_PERFORMANCE_THRESHOLD else _strat(0, [], 0, 0),
            'static_bh_1y_market_regime': _strat(static_bh_1y_market_regime_portfolio_value, static_bh_1y_market_regime_portfolio_history, static_bh_1y_market_regime_transaction_costs, static_bh_1y_market_regime_cash) if config.ENABLE_STATIC_BH_1Y_MARKET_REGIME else _strat(0, [], 0, 0),
            # Momentum Persistence and Overlap-Based Strategies
            'static_bh_1y_mom_persist': _strat(static_bh_1y_mom_persist_portfolio_value, static_bh_1y_mom_persist_portfolio_history, static_bh_1y_mom_persist_transaction_costs, static_bh_1y_mom_persist_cash) if config.ENABLE_STATIC_BH_1Y_MOMENTUM_PERSIST else _strat(0, [], 0, 0),
            'static_bh_1y_overlap': _strat(static_bh_1y_overlap_portfolio_value, static_bh_1y_overlap_portfolio_history, static_bh_1y_overlap_transaction_costs, static_bh_1y_overlap_cash) if config.ENABLE_STATIC_BH_1Y_OVERLAP else _strat(0, [], 0, 0),
            # Rank Drift, Drawdown Trigger, Smart Monthly
            'static_bh_1y_rank_drift': _strat(static_bh_1y_rank_drift_portfolio_value, static_bh_1y_rank_drift_portfolio_history, static_bh_1y_rank_drift_transaction_costs, static_bh_1y_rank_drift_cash) if config.ENABLE_STATIC_BH_1Y_RANK_DRIFT else _strat(0, [], 0, 0),
            'static_bh_1y_drawdown': _strat(static_bh_1y_drawdown_portfolio_value, static_bh_1y_drawdown_portfolio_history, static_bh_1y_drawdown_transaction_costs, static_bh_1y_drawdown_cash) if config.ENABLE_STATIC_BH_1Y_DRAWDOWN else _strat(0, [], 0, 0),
            'static_bh_1y_smart_monthly': _strat(static_bh_1y_smart_monthly_portfolio_value, static_bh_1y_smart_monthly_portfolio_history, static_bh_1y_smart_monthly_transaction_costs, static_bh_1y_smart_monthly_cash) if config.ENABLE_STATIC_BH_1Y_SMART_MONTHLY else _strat(0, [], 0, 0),
            # Smart Rebalancing Strategies
            'bh_1y_mom_sell': _strat(bh_1y_mom_sell_portfolio_value, bh_1y_mom_sell_portfolio_history, bh_1y_mom_sell_transaction_costs, bh_1y_mom_sell_cash) if config.ENABLE_BH_1Y_MOM_SELL else _strat(0, [], 0, 0),
            'bh_1y_rank_sell': _strat(bh_1y_rank_sell_portfolio_value, bh_1y_rank_sell_portfolio_history, bh_1y_rank_sell_transaction_costs, bh_1y_rank_sell_cash) if config.ENABLE_BH_1Y_RANK_SELL else _strat(0, [], 0, 0),
            'bh_1y_trailing_mom': _strat(bh_1y_trailing_mom_portfolio_value, bh_1y_trailing_mom_portfolio_history, bh_1y_trailing_mom_transaction_costs, bh_1y_trailing_mom_cash) if config.ENABLE_BH_1Y_TRAILING_MOM else _strat(0, [], 0, 0),
            'bh_1y_volume_confirm': _strat(bh_1y_volume_confirm_portfolio_value, bh_1y_volume_confirm_portfolio_history, bh_1y_volume_confirm_transaction_costs, bh_1y_volume_confirm_cash) if config.ENABLE_BH_1Y_VOLUME_CONFIRM else _strat(0, [], 0, 0),
            'bh_1y_sector_aware': _strat(bh_1y_sector_aware_portfolio_value, bh_1y_sector_aware_portfolio_history, bh_1y_sector_aware_transaction_costs, bh_1y_sector_aware_cash) if config.ENABLE_BH_1Y_SECTOR_AWARE else _strat(0, [], 0, 0),
            'bh_1y_accel_buy': _strat(bh_1y_accel_buy_portfolio_value, bh_1y_accel_buy_portfolio_history, bh_1y_accel_buy_transaction_costs, bh_1y_accel_buy_cash) if config.ENABLE_BH_1Y_ACCEL_BUY else _strat(0, [], 0, 0),
            'static_bh_3m_accel': _strat(static_bh_3m_accel_portfolio_value, static_bh_3m_accel_portfolio_history, static_bh_3m_accel_transaction_costs, static_bh_3m_accel_cash) if config.ENABLE_STATIC_BH_3M_ACCEL else _strat(0, [], 0, 0),
            # 10 New Rebalancing Strategies
            'bh_1y_vol_adj_rebal': _strat(bh_1y_vol_adj_rebal_portfolio_value, bh_1y_vol_adj_rebal_portfolio_history, bh_1y_vol_adj_rebal_transaction_costs, bh_1y_vol_adj_rebal_cash) if config.ENABLE_BH_1Y_VOL_ADJ_REBAL else _strat(0, [], 0, 0),
            'ratio_1m3m_vol_adj_rebal': _strat(ratio_1m3m_vol_adj_rebal_portfolio_value, ratio_1m3m_vol_adj_rebal_portfolio_history, ratio_1m3m_vol_adj_rebal_transaction_costs, ratio_1m3m_vol_adj_rebal_cash) if config.ENABLE_RATIO_1M3M_VOL_ADJ_REBAL else _strat(0, [], 0, 0),
            'bh_1y_corr_filter': _strat(bh_1y_corr_filter_portfolio_value, bh_1y_corr_filter_portfolio_history, bh_1y_corr_filter_transaction_costs, bh_1y_corr_filter_cash) if config.ENABLE_BH_1Y_CORR_FILTER else _strat(0, [], 0, 0),
            'bh_1y_regime_aware': _strat(bh_1y_regime_aware_portfolio_value, bh_1y_regime_aware_portfolio_history, bh_1y_regime_aware_transaction_costs, bh_1y_regime_aware_cash) if config.ENABLE_BH_1Y_REGIME_AWARE else _strat(0, [], 0, 0),
            'bh_1y_risk_parity': _strat(bh_1y_risk_parity_portfolio_value, bh_1y_risk_parity_portfolio_history, bh_1y_risk_parity_transaction_costs, bh_1y_risk_parity_cash) if config.ENABLE_BH_1Y_RISK_PARITY else _strat(0, [], 0, 0),
            'bh_1y_drift_thresh': _strat(bh_1y_drift_thresh_portfolio_value, bh_1y_drift_thresh_portfolio_history, bh_1y_drift_thresh_transaction_costs, bh_1y_drift_thresh_cash) if config.ENABLE_BH_1Y_DRIFT_THRESH else _strat(0, [], 0, 0),
            'bh_1y_mom_quality': _strat(bh_1y_mom_quality_portfolio_value, bh_1y_mom_quality_portfolio_history, bh_1y_mom_quality_transaction_costs, bh_1y_mom_quality_cash) if config.ENABLE_BH_1Y_MOM_QUALITY else _strat(0, [], 0, 0),
            'bh_1y_liquidity': _strat(bh_1y_liquidity_portfolio_value, bh_1y_liquidity_portfolio_history, bh_1y_liquidity_transaction_costs, bh_1y_liquidity_cash) if config.ENABLE_BH_1Y_LIQUIDITY else _strat(0, [], 0, 0),
            'bh_1y_earnings_avoid': _strat(bh_1y_earnings_avoid_portfolio_value, bh_1y_earnings_avoid_portfolio_history, bh_1y_earnings_avoid_transaction_costs, bh_1y_earnings_avoid_cash) if config.ENABLE_BH_1Y_EARNINGS_AVOID else _strat(0, [], 0, 0),
            'bh_1y_multi_factor': _strat(bh_1y_multi_factor_portfolio_value, bh_1y_multi_factor_portfolio_history, bh_1y_multi_factor_transaction_costs, bh_1y_multi_factor_cash) if config.ENABLE_BH_1Y_MULTI_FACTOR else _strat(0, [], 0, 0),
            'bh_1y_time_decay': _strat(bh_1y_time_decay_portfolio_value, bh_1y_time_decay_portfolio_history, bh_1y_time_decay_transaction_costs, bh_1y_time_decay_cash) if config.ENABLE_BH_1Y_TIME_DECAY else _strat(0, [], 0, 0),
        }
    }

    # === SAVE STRATEGY SELECTIONS TO JSON FOR LIVE TRADING ===
    # This allows live trading to read selections instead of recalculating
    import json
    from datetime import datetime as dt

    strategy_selections = {
        'timestamp': dt.now().isoformat(),
        'backtest_end_date': str(backtest_end_date),
        'strategies': {
            'static_bh_1y': {'tickers': list(current_static_bh_1y_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_positions.items()}},
            'static_bh_6m': {'tickers': list(current_static_bh_6m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_6m_positions.items()}},
            'static_bh_3m': {'tickers': list(current_static_bh_3m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_3m_positions.items()}},
            'static_bh_1m': {'tickers': list(current_static_bh_1m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1m_positions.items()}},
            'dynamic_bh_1y': {'tickers': list(current_dynamic_bh_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_positions.items()}},
            'dynamic_bh_6m': {'tickers': list(current_dynamic_bh_6m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_6m_positions.items()}},
            'dynamic_bh_3m': {'tickers': list(current_dynamic_bh_3m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_3m_positions.items()}},
            'dynamic_bh_1m': {'tickers': list(current_dynamic_bh_1m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_1m_positions.items()}},
            'risk_adj_mom': {'tickers': list(current_risk_adj_mom_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_positions.items()}},
            'risk_adj_mom_1m': {'tickers': list(current_risk_adj_mom_1m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_1m_positions.items()}},
            'risk_adj_mom_1m_vol_sweet': {'tickers': list(current_risk_adj_mom_1m_vol_sweet_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_1m_vol_sweet_positions.items()}},
            'bh_1y_volsweet_accel': {'tickers': list(current_bh_1y_volsweet_accel_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_volsweet_accel_positions.items()}},
            'bh_1y_dynamic_accel': {'tickers': list(current_bh_1y_dynamic_accel_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_dynamic_accel_positions.items()}},
            'risk_adj_mom_3m': {'tickers': list(current_risk_adj_mom_3m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_3m_positions.items()}},
            'risk_adj_mom_6m': {'tickers': list(current_risk_adj_mom_6m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_6m_positions.items()}},
            'momentum_ai_hybrid': {'tickers': list(momentum_ai_hybrid_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in momentum_ai_hybrid_positions.items()}},
            'ai_elite': {'tickers': list(ai_elite_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_elite_positions.items()}},
            'elite_hybrid': {'tickers': list(elite_hybrid_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in elite_hybrid_positions.items()}},
            'elite_risk': {'tickers': list(elite_risk_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in elite_risk_positions.items()}},
            'inverse_etf_hedge': {'tickers': list(current_inverse_etf_hedge_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in inverse_etf_hedge_positions.items()}},
            'momentum_volatility_hybrid_6m': {'tickers': list(momentum_volatility_hybrid_6m_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in momentum_volatility_hybrid_6m_positions.items()}},
            'trend_atr': {'tickers': list(trend_atr_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in trend_atr_positions.items()}},
            'dual_momentum': {'tickers': list(dual_mom_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dual_mom_positions.items()}},
            'sector_rotation': {'tickers': list(current_sector_rotation_etfs), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in sector_rotation_positions.items()}},
            'quality_momentum': {'tickers': list(current_quality_momentum_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in quality_momentum_positions.items()}},
            'mean_reversion': {'tickers': list(current_mean_reversion_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in mean_reversion_positions.items()}},
            'volatility_ensemble': {'tickers': list(current_volatility_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in volatility_ensemble_positions.items()}},
            'turnaround': {'tickers': list(current_turnaround_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in turnaround_positions.items()}},
            'price_acceleration': {'tickers': list(current_price_acceleration_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in price_acceleration_positions.items()}},
            'volatility_adj_mom': {'tickers': list(current_volatility_adj_mom_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in volatility_adj_mom_positions.items()}},
            'enhanced_volatility': {'tickers': list(current_enhanced_volatility_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in enhanced_volatility_positions.items()}},
            'ratio_3m_1y': {'tickers': list(current_ratio_3m_1y_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ratio_3m_1y_positions.items()}},
            'ratio_1y_3m': {'tickers': list(current_ratio_1y_3m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ratio_1y_3m_positions.items()}},
            'concentrated_3m': {'tickers': list(current_concentrated_3m_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in concentrated_3m_positions.items()}},
            'analyst_rec': {'tickers': list(current_analyst_rec_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in analyst_rec_positions.items()}},
            'bh_1y_monthly': {'tickers': list(current_static_bh_1y_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_monthly_positions.items()}},
            'bh_6m_monthly': {'tickers': list(current_static_bh_6m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_6m_monthly_positions.items()}},
            'bh_3m_monthly': {'tickers': list(current_static_bh_3m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_3m_monthly_positions.items()}},
            'bh_1m_monthly': {'tickers': list(current_static_bh_1m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1m_monthly_positions.items()}},
            # Adaptive Rebalancing Strategies (based on Static BH 1Y)
            'static_bh_1y_vol': {'tickers': list(current_static_bh_1y_vol_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_vol_positions.items()}},
            'static_bh_1y_perf': {'tickers': list(current_static_bh_1y_perf_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_perf_positions.items()}},
            'static_bh_1y_mom': {'tickers': list(current_static_bh_1y_mom_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_mom_positions.items()}},
            'static_bh_1y_atr': {'tickers': list(current_static_bh_1y_atr_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_atr_positions.items()}},
            'static_bh_1y_hybrid': {'tickers': list(current_static_bh_1y_hybrid_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_hybrid_positions.items()}},
            # Enhanced Static BH 1Y Strategies
            'static_bh_1y_volume': {'tickers': list(current_static_bh_1y_volume_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_volume_positions.items()}},
            'static_bh_1y_sector': {'tickers': list(current_static_bh_1y_sector_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_sector_positions.items()}},
            'static_bh_1y_perf_threshold': {'tickers': list(current_static_bh_1y_perf_threshold_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_perf_threshold_positions.items()}},
            'static_bh_1y_market_regime': {'tickers': list(current_static_bh_1y_market_regime_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_market_regime_positions.items()}},
            'static_bh_1y_mom_persist': {'tickers': list(current_static_bh_1y_mom_persist_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_mom_persist_positions.items()}},
            'static_bh_1y_overlap': {'tickers': list(current_static_bh_1y_overlap_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_overlap_positions.items()}},
            'static_bh_1y_rank_drift': {'tickers': list(current_static_bh_1y_rank_drift_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_rank_drift_positions.items()}},
            'static_bh_1y_drawdown': {'tickers': list(current_static_bh_1y_drawdown_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_drawdown_positions.items()}},
            'static_bh_1y_smart_monthly': {'tickers': list(current_static_bh_1y_smart_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_1y_smart_monthly_positions.items()}},
            'bh_1y_mom_sell': {'tickers': list(current_bh_1y_mom_sell_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_mom_sell_positions.items()}},
            'bh_1y_rank_sell': {'tickers': list(current_bh_1y_rank_sell_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_rank_sell_positions.items()}},
            'bh_1y_trailing_mom': {'tickers': list(current_bh_1y_trailing_mom_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_trailing_mom_positions.items()}},
            'bh_1y_volume_confirm': {'tickers': list(current_bh_1y_volume_confirm_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_volume_confirm_positions.items()}},
            'bh_1y_sector_aware': {'tickers': list(current_bh_1y_sector_aware_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_sector_aware_positions.items()}},
            'bh_1y_accel_buy': {'tickers': list(current_bh_1y_accel_buy_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_accel_buy_positions.items()}},
            'static_bh_3m_accel': {'tickers': list(current_static_bh_3m_accel_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in static_bh_3m_accel_positions.items()}},
            'bh_1y_vol_adj_rebal': {'tickers': list(current_bh_1y_vol_adj_rebal_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_vol_adj_rebal_positions.items()}},
            'ratio_1m3m_vol_adj_rebal': {'tickers': list(current_ratio_1m3m_vol_adj_rebal_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ratio_1m3m_vol_adj_rebal_positions.items()}},
            'bh_1y_corr_filter': {'tickers': list(current_bh_1y_corr_filter_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_corr_filter_positions.items()}},
            'bh_1y_regime_aware': {'tickers': list(current_bh_1y_regime_aware_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_regime_aware_positions.items()}},
            'bh_1y_risk_parity': {'tickers': list(current_bh_1y_risk_parity_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_risk_parity_positions.items()}},
            'bh_1y_drift_thresh': {'tickers': list(current_bh_1y_drift_thresh_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_drift_thresh_positions.items()}},
            'bh_1y_mom_quality': {'tickers': list(current_bh_1y_mom_quality_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_mom_quality_positions.items()}},
            'bh_1y_liquidity': {'tickers': list(current_bh_1y_liquidity_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_liquidity_positions.items()}},
            'bh_1y_earnings_avoid': {'tickers': list(current_bh_1y_earnings_avoid_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_earnings_avoid_positions.items()}},
            'bh_1y_multi_factor': {'tickers': list(current_bh_1y_multi_factor_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_multi_factor_positions.items()}},
            'bh_1y_time_decay': {'tickers': list(current_bh_1y_time_decay_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bh_1y_time_decay_positions.items()}},
            'multi_tf_ensemble': {'tickers': list(current_multi_tf_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in multi_tf_ensemble_positions.items()}},
            'multi_tf_intraday_ensemble': {'tickers': list(current_multi_tf_intraday_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in multi_tf_intraday_ensemble_positions.items()}},
            'trend_breakout': {'tickers': list(current_trend_breakout_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in trend_breakout_positions.items()}},
            # Additional strategies
            'ai_elite_filtered': {'tickers': list(ai_elite_filtered_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_elite_filtered_positions.items()}},
            'ai_elite_market_up': {'tickers': list(ai_elite_market_up_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_elite_market_up_positions.items()}},
            'ai_elite_monthly': {'tickers': list(ai_elite_monthly_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_elite_monthly_positions.items()}},
            'adaptive_ensemble': {'tickers': list(current_adaptive_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in adaptive_ensemble_positions.items()}},
            'bb_breakout': {'tickers': list(current_bb_breakout_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bb_breakout_positions.items()}},
            'bb_mean_reversion': {'tickers': list(current_bb_mean_reversion_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bb_mean_reversion_positions.items()}},
            'bb_rsi_combo': {'tickers': list(current_bb_rsi_combo_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bb_rsi_combo_positions.items()}},
            'bb_squeeze_breakout': {'tickers': list(current_bb_squeeze_breakout_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in bb_squeeze_breakout_positions.items()}},
            'correlation_ensemble': {'tickers': list(current_correlation_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in correlation_ensemble_positions.items()}},
            'dynamic_bh_1y_trailing_stop': {'tickers': list(current_dynamic_bh_1y_trailing_stop_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_1y_trailing_stop_positions.items()}},
            'dynamic_bh_1y_vol_filter': {'tickers': list(current_dynamic_bh_1y_vol_filter_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_bh_1y_vol_filter_positions.items()}},
            'dynamic_pool': {'tickers': list(current_dynamic_pool_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in dynamic_pool_positions.items()}},
            'momentum_volatility_hybrid': {'tickers': list(momentum_volatility_hybrid_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in momentum_volatility_hybrid_positions.items()}},
            'momentum_volatility_hybrid_1y': {'tickers': list(momentum_volatility_hybrid_1y_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in momentum_volatility_hybrid_1y_positions.items()}},
            'momentum_volatility_hybrid_1y3m': {'tickers': list(momentum_volatility_hybrid_1y3m_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in momentum_volatility_hybrid_1y3m_positions.items()}},
            'risk_adj_mom_3m_market_up': {'tickers': list(current_risk_adj_mom_3m_market_up_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_3m_market_up_positions.items()}},
            'risk_adj_mom_3m_with_stops': {'tickers': list(current_risk_adj_mom_3m_with_stops_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_3m_with_stops_positions.items()}},
            'risk_adj_mom_1m_monthly': {'tickers': list(current_risk_adj_mom_1m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_1m_monthly_positions.items()}},
            'risk_adj_mom_3m_monthly': {'tickers': list(current_risk_adj_mom_3m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_3m_monthly_positions.items()}},
            'risk_adj_mom_3m_sentiment': {'tickers': list(current_risk_adj_mom_3m_sentiment_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_3m_sentiment_positions.items()}},
            'risk_adj_mom_6m_monthly': {'tickers': list(current_risk_adj_mom_6m_monthly_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_6m_monthly_positions.items()}},
            'risk_adj_mom_sentiment': {'tickers': list(current_risk_adj_mom_sentiment_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in risk_adj_mom_sentiment_positions.items()}},
            'universal_model': {'tickers': list(current_universal_model_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in universal_model_positions.items()}},
            'savgol_trend': {'tickers': list(current_savgol_trend_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in savgol_trend_positions.items()}},
            'top5_consistency_blend': {'tickers': list(current_top5_consistency_blend_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in top5_consistency_blend_positions.items()}},
            'vol_sweet_mom': {'tickers': list(current_vol_sweet_mom_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in vol_sweet_mom_positions.items()}},
            'voting_ensemble': {'tickers': list(current_voting_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in voting_ensemble_positions.items()}},
            'ai_champion': {'tickers': list(ai_champion_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_champion_positions.items()}},
            'ai_regime': {'tickers': list(ai_regime_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_regime_positions.items()}},
            'ai_regime_monthly': {'tickers': list(ai_regime_monthly_positions.keys()), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_regime_monthly_positions.items()}},
            'ai_volatility_ensemble': {'tickers': list(current_ai_volatility_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in ai_volatility_ensemble_positions.items()}},
            'mom_accel': {'tickers': list(current_mom_accel_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in mom_accel_positions.items()}},
            'sentiment_ensemble': {'tickers': list(current_sentiment_ensemble_stocks), 'positions': {t: {'shares': p['shares'], 'avg_price': p.get('entry_price', p.get('avg_price', 0))} for t, p in sentiment_ensemble_positions.items()}},
            'meta_weighted_composite': {'tickers': list(meta_weighted_composite_positions.keys()) if 'meta_weighted_composite_positions' in dir() else [], 'positions': {}},
            'meta_tiered_selection': {'tickers': list(meta_tiered_selection_positions.keys()) if 'meta_tiered_selection_positions' in dir() else [], 'positions': {}},
            'meta_ensemble_alloc': {'tickers': list(meta_ensemble_alloc_positions.keys()) if 'meta_ensemble_alloc_positions' in dir() else [], 'positions': {}},
            'meta_regime_based': {'tickers': list(meta_regime_based_positions.keys()) if 'meta_regime_based_positions' in dir() else [], 'positions': {}},
            'meta_recency_weighted': {'tickers': list(meta_recency_weighted_positions.keys()) if 'meta_recency_weighted_positions' in dir() else [], 'positions': {}},
            'meta_efficiency_ratio': {'tickers': list(meta_efficiency_ratio_positions.keys()) if 'meta_efficiency_ratio_positions' in dir() else [], 'positions': {}},
            'meta_min_variance': {'tickers': list(meta_min_variance_positions.keys()) if 'meta_min_variance_positions' in dir() else [], 'positions': {}},
            'meta_bayesian': {'tickers': list(meta_bayesian_positions.keys()) if 'meta_bayesian_positions' in dir() else [], 'positions': {}},
            'meta_adaptive_convex': {'tickers': list(meta_adaptive_convex_positions.keys()) if 'meta_adaptive_convex_positions' in dir() else [], 'positions': {}},
            'meta_consensus': {'tickers': list(meta_consensus_positions.keys()) if 'meta_consensus_positions' in dir() else [], 'positions': {}},
        },
        'performance': {name: {'value': data['value'], 'return_pct': ((data['value'] - INVESTMENT_PER_STOCK * PORTFOLIO_SIZE) / (INVESTMENT_PER_STOCK * PORTFOLIO_SIZE)) * 100} for name, data in results['strategies'].items()}
    }

    # Save to JSON file
    selections_file = Path('logs/strategy_selections.json')
    selections_file.parent.mkdir(parents=True, exist_ok=True)
    with open(selections_file, 'w') as f:
        json.dump(strategy_selections, f, indent=2, default=str)
    print(f"\n📁 Strategy selections saved to {selections_file}")

    # === SAVE LIVE TRADING SELECTIONS (20 tickers per strategy, independent of PORTFOLIO_SIZE) ===
    # This generates fresh selections at backtest end date with top 20 tickers for live trading
    LIVE_TRADING_TOP_N = 20
    print(f"\n📊 Generating live trading selections (top {LIVE_TRADING_TOP_N} per strategy)...")

    from shared_strategies import (
        select_top_performers, select_risk_adj_mom_stocks,
        select_sector_rotation_etfs, select_quality_momentum_stocks,
        select_mean_reversion_stocks, select_volatility_adj_mom_stocks,
        select_price_acceleration_stocks, select_turnaround_stocks,
        select_3m_1y_ratio_stocks, select_1y_3m_ratio_stocks,
        select_top_performers_vol_filtered,
        select_momentum_volatility_hybrid_stocks
    )

    # Use backtest_end_date as current_date for selections
    live_current_date = backtest_end_date

    live_trading_selections = {
        'timestamp': dt.now().isoformat(),
        'backtest_end_date': str(backtest_end_date),
        'top_n': LIVE_TRADING_TOP_N,
        'strategies': {}
    }

    try:
        # Auto-generate selections for ALL strategies using the centralized registry
        # This ensures live trading always has all strategies without manual updates
        for strategy_name in results['strategies'].keys():
            try:
                # Skip strategies that don't have selection functions (e.g., meta strategies)
                if strategy_name in ['meta_strategy_ml', 'meta_strategy_mom', 'top5_consistency_blend', 'voting_ensemble']:
                    continue

                # Skip disabled strategies (especially AI strategies that are slow to run)
                is_enabled = _is_strategy_enabled(strategy_name, config)
                if is_enabled is False:
                    live_trading_selections['strategies'][strategy_name] = []
                    continue

                # Use the centralized strategy registry from shared_strategies.py
                new_stocks = get_strategy_tickers(
                    strategy_name, initial_top_tickers, ticker_data_grouped,
                    live_current_date, LIVE_TRADING_TOP_N
                )
                live_trading_selections['strategies'][strategy_name] = new_stocks

            except Exception as e:
                print(f"   ⚠️ Error generating selections for {strategy_name}: {e}")
                live_trading_selections['strategies'][strategy_name] = []

        print(f"   ✅ Generated selections for {len(live_trading_selections['strategies'])} strategies")

        if config.ENABLE_TOP5_CONSISTENCY_BLEND and 'top5_consistency_blend' in results['strategies']:
            try:
                live_source_strategy_data = {}
                for strategy_key, tickers in live_trading_selections['strategies'].items():
                    if isinstance(tickers, list):
                        live_source_strategy_data[strategy_key] = {
                            'display_name': _get_strategy_display_name(strategy_key),
                            'tickers': tickers,
                        }

                top5_live_stocks, live_weights, _ = _select_top5_consistency_blend_stocks(
                    source_strategy_data=live_source_strategy_data,
                    top5_consistency_counts=top5_consistency_counts,
                    total_days=day_count,
                    top_n=LIVE_TRADING_TOP_N,
                )
                live_trading_selections['strategies']['top5_consistency_blend'] = top5_live_stocks
                if live_weights:
                    live_weight_summary = ", ".join(
                        f"{name}={weight:.1%}"
                        for name, weight in sorted(live_weights.items(), key=lambda item: (-item[1], item[0]))
                    )
                    print(
                        f"   ✅ Top5 Consistency Blend: Selected {len(top5_live_stocks)} stocks "
                        f"using {live_weight_summary}"
                    )
            except Exception as e:
                print(f"   ⚠️ Top5 Consistency Blend live selection error: {e}")
                live_trading_selections['strategies']['top5_consistency_blend'] = []

        if config.ENABLE_VOTING_ENSEMBLE and 'voting_ensemble' in results['strategies']:
            try:
                live_voting_source_strategy_data = {}
                for strategy_key, strategy_result in results['strategies'].items():
                    if strategy_key == 'voting_ensemble':
                        continue
                    tickers = live_trading_selections['strategies'].get(strategy_key, [])
                    if isinstance(tickers, list):
                        live_voting_source_strategy_data[strategy_key] = {
                            'display_name': _get_strategy_display_name(strategy_key),
                            'tickers': tickers,
                            'value': strategy_result.get('value', 0.0),
                        }

                voting_live_stocks, voting_live_sources, _, _ = _select_voting_ensemble_stocks_from_strategy_data(
                    source_strategy_data=live_voting_source_strategy_data,
                    top_n=LIVE_TRADING_TOP_N,
                    max_source_strategies=3,
                    excluded_strategy_keys=['voting_ensemble'],
                )
                live_trading_selections['strategies']['voting_ensemble'] = voting_live_stocks
                if voting_live_sources:
                    voting_source_summary = ", ".join(display_name for _, display_name, _, _ in voting_live_sources)
                    print(
                        f"   ✅ Voting Ensemble: Selected {len(voting_live_stocks)} stocks "
                        f"from top {len(voting_live_sources)} strategies ({voting_source_summary})"
                    )
            except Exception as e:
                print(f"   ⚠️ Voting Ensemble live selection error: {e}")
                live_trading_selections['strategies']['voting_ensemble'] = []

        if config.ENABLE_AI_CHAMPION and 'ai_champion' in results['strategies']:
            try:
                from ai_champion_strategy import AIChampionAllocator, select_ai_champion_stocks

                ai_champion_allocator = AIChampionAllocator(
                    retrain_days=config.AI_CHAMPION_RETRAIN_DAYS,
                    forward_days=config.AI_CHAMPION_FORWARD_DAYS,
                    confidence_threshold=config.AI_CHAMPION_CONFIDENCE_THRESHOLD,
                    hold_margin=config.AI_CHAMPION_HOLD_MARGIN,
                )
                ai_champion_allocator.load_model()
                ai_champion_current_strategy = ai_champion_allocator.predict_best_strategy(ticker_data_grouped, live_current_date)
                if ai_champion_current_strategy is not None:
                    ai_champion_stocks = select_ai_champion_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        live_current_date,
                        LIVE_TRADING_TOP_N,
                        ai_champion_current_strategy,
                    )
                    live_trading_selections['strategies']['ai_champion'] = ai_champion_stocks
                    print(f"   ✅ AI Champion: Selected {len(ai_champion_stocks)} stocks using {str(ai_champion_current_strategy)}")
                else:
                    live_trading_selections['strategies']['ai_champion'] = []
                    print(f"   ⚠️ AI Champion: No prediction available")
            except Exception as e:
                print(f"   ⚠️ AI Champion live selection error: {e}")
                live_trading_selections['strategies']['ai_champion'] = []

        # Special handling for AI Regime (needs model loading)
        if config.ENABLE_AI_REGIME and 'ai_regime' in results['strategies']:
            try:
                from ai_regime_strategy import AIRegimeAllocator, select_ai_regime_stocks
                ai_regime_allocator = AIRegimeAllocator(retrain_days=1, forward_days=1)
                ai_regime_allocator.load_model()
                ai_regime_current_strategy = ai_regime_allocator.predict_best_strategy(ticker_data_grouped, live_current_date)
                if ai_regime_current_strategy is not None:
                    ai_regime_stocks = select_ai_regime_stocks(
                        initial_top_tickers, ticker_data_grouped, live_current_date,
                        LIVE_TRADING_TOP_N, ai_regime_current_strategy
                    )
                    live_trading_selections['strategies']['ai_regime'] = ai_regime_stocks
                    print(f"   ✅ AI Regime: Selected {len(ai_regime_stocks)} stocks using {str(ai_regime_current_strategy)}")
                else:
                    live_trading_selections['strategies']['ai_regime'] = []
                    print(f"   ⚠️ AI Regime: No prediction available")
            except Exception as e:
                print(f"   ⚠️ AI Regime live selection error: {e}")
                live_trading_selections['strategies']['ai_regime'] = []

        if config.ENABLE_AI_REGIME_MONTHLY and 'ai_regime_monthly' in results['strategies']:
            try:
                from ai_regime_strategy import AIRegimeMonthlyAllocator, select_ai_regime_stocks
                ai_regime_monthly_allocator = AIRegimeMonthlyAllocator(forward_days=1)
                ai_regime_monthly_allocator.load_model()
                ai_regime_monthly_current_strategy = ai_regime_monthly_allocator.predict_best_strategy(ticker_data_grouped, live_current_date)
                if ai_regime_monthly_current_strategy is not None:
                    ai_regime_monthly_stocks = select_ai_regime_stocks(
                        initial_top_tickers, ticker_data_grouped, live_current_date,
                        LIVE_TRADING_TOP_N, ai_regime_monthly_current_strategy
                    )
                    live_trading_selections['strategies']['ai_regime_monthly'] = ai_regime_monthly_stocks
                    print(f"   ✅ AI Regime Monthly: Selected {len(ai_regime_monthly_stocks)} stocks using {str(ai_regime_monthly_current_strategy)}")
                else:
                    live_trading_selections['strategies']['ai_regime_monthly'] = []
                    print(f"   ⚠️ AI Regime Monthly: No prediction available")
            except Exception as e:
                print(f"   ⚠️ AI Regime Monthly live selection error: {e}")
                live_trading_selections['strategies']['ai_regime_monthly'] = []

    except Exception as e:
        print(f"   ⚠️ Error generating live trading selections: {e}")

    # Save live trading selections to separate JSON file
    live_selections_file = Path('logs/live_trading_selections.json')
    with open(live_selections_file, 'w') as f:
        json.dump(live_trading_selections, f, indent=2, default=str)
    print(f"📁 Live trading selections saved to {live_selections_file}")

    return results




def _quick_predict_return(ticker: str, df_recent: pd.DataFrame, model, scaler, y_scaler, horizon_days: int) -> float:
    """Quick prediction of return for stock reselection during walk-forward backtest."""
    # Import PyTorch models if available
    if PYTORCH_AVAILABLE:
        from ml_models import TCNRegressor, GRURegressor, LSTMRegressor, LSTMClassifier, GRUClassifier

    try:
        # Wrap entire prediction in timeout
        with prediction_timeout(PREDICTION_TIMEOUT, ticker):
            if model is None:
                print(f"   ⚠️ {ticker}: model is None")
                return -np.inf
            if scaler is None:
                print(f"   ⚠️ {ticker}: scaler is None")
                return -np.inf
            if df_recent.empty:
                print(f"   ⚠️ {ticker}: df_recent is empty")
                return -np.inf

            # ✅ VALIDATION: Check if we have enough data for prediction
            try:
                validate_prediction_data(df_recent, ticker)
            except InsufficientDataError as e:
                print(f"   {str(e)}")
                return -np.inf

            print(f"   🔍 {ticker}: Starting prediction with {len(df_recent)} rows, model type: {type(model).__name__}")

            # Engineer features - same as training
            df_with_features = df_recent.copy()

            print(f"   🔧 {ticker}: Initial features: {list(df_with_features.columns)}")

            # Add financial features that might be in the data (fill with 0 if missing)
            financial_features = [col for col in df_with_features.columns if col.startswith('Fin_')]
            for col in financial_features:
                df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce').fillna(0)

            df_with_features = _calculate_technical_indicators(df_with_features)
            print(f"   🔧 {ticker}: After technical indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} features")

            # Add annualized BH return feature (same as in training)
            if len(df_with_features) > 1:
                start_price = df_with_features["Close"].iloc[0]
                end_price = df_with_features["Close"].iloc[-1]

                # Fix: Handle date calculation properly when index is named 'date'
                try:
                    total_days = (df_with_features.index[-1] - df_with_features.index[0]).days
                except Exception:
                    # Fallback: convert to datetime if needed
                    total_days = (pd.to_datetime(df_with_features.index[-1]) - pd.to_datetime(df_with_features.index[0])).days

                if total_days > 0 and start_price > 0:
                    total_return = (end_price / start_price) - 1.0
                    annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
                    df_with_features["Annualized_BH_Return"] = annualized_return
                else:
                    df_with_features["Annualized_BH_Return"] = 0.0
            else:
                df_with_features["Annualized_BH_Return"] = 0.0

            # Only drop rows with NaN if we have enough rows to spare
            rows_before_dropna = len(df_with_features)
            df_with_features = df_with_features.dropna()
            rows_after_dropna = len(df_with_features)
            print(f"   🔧 {ticker}: After dropna: {rows_after_dropna} rows (dropped {rows_before_dropna - rows_after_dropna})")

            # ✅ VALIDATION: Check if enough rows remain after feature engineering
            if rows_after_dropna == 0:
                print(f"   ❌ {ticker}: All rows dropped during prediction feature engineering!")
                print(f"   💡 This usually means:")
                print(f"      - Not enough historical data for technical indicators")
                print(f"      - Too many NaN/missing values in source data")
                print(f"      - Feature calculation window is too large for available data")
                return -np.inf

            try:
                validate_features_after_engineering(df_with_features, ticker, min_rows=1, context="prediction")
            except InsufficientDataError as e:
                print(f"   {str(e)}")
                return -np.inf

            # Get latest data point
            latest_data = df_with_features.iloc[-1:]
            print(f"   📊 {ticker}: Latest data shape: {latest_data.shape}, features: {list(latest_data.columns)}")

            # Align features to match scaler's expectations
            scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else []
            print(f"   🔧 {ticker}: Scaler expects {len(scaler_features)} features: {scaler_features[:5]}...")
            if scaler_features:
                # Ensure we have all expected features, fill missing ones with 0
                for feature in scaler_features:
                    if feature not in latest_data.columns:
                        latest_data[feature] = 0.0
                # Reorder columns to match scaler expectations
                latest_data = latest_data[scaler_features]
                print(f"   🔧 {ticker}: After alignment: {latest_data.shape}")

            # Scale features
            try:
                # Pass DataFrame directly to preserve feature names and avoid sklearn warning
                features_scaled = scaler.transform(latest_data)
                print(f"   🔧 {ticker}: Features scaled successfully, shape: {features_scaled.shape}")
            except Exception as e:
                print(f"   ❌ {ticker}: Scaling failed: {e}")
                return -np.inf

            # Predict return
            try:
                # Check if model is a PyTorch sequence model (TCN, GRU, LSTM)
                if PYTORCH_AVAILABLE and isinstance(model, (TCNRegressor, GRURegressor, LSTMRegressor, LSTMClassifier, GRUClassifier)):
                    import torch

                    # For sequence models, we need a sequence of data, not just the latest point
                    # Use the last SEQUENCE_LENGTH rows from df_with_features
                    sequence_length = SEQUENCE_LENGTH  # Default is 60

                    if len(df_with_features) < sequence_length:
                        # If not enough data, pad with zeros
                        sequence_data = df_with_features[scaler_features].copy()
                        padding_needed = sequence_length - len(sequence_data)
                        padding_df = pd.DataFrame(
                            np.zeros((padding_needed, len(scaler_features))),
                            columns=scaler_features
                        )
                        sequence_data = pd.concat([padding_df, sequence_data], ignore_index=True)
                    else:
                        # Get the last sequence_length rows
                        sequence_data = df_with_features[scaler_features].tail(sequence_length)

                    # Scale the entire sequence (pass DataFrame to preserve feature names)
                    sequence_scaled = scaler.transform(sequence_data)
                    print(f"   🔧 {ticker}: Sequence scaled, shape: {sequence_scaled.shape}")

                    # Convert to PyTorch tensor with shape (batch_size=1, sequence_length, num_features)
                    X_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

                    # Move to appropriate device
                    from config import FORCE_CPU
                    device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
                    X_tensor = X_tensor.to(device)
                    model.to(device)

                    # Make prediction
                    model.eval()
                    with torch.no_grad():
                        output_tensor = model(X_tensor)
                        # Handle different output shapes - check tensor dim before converting to numpy
                        if output_tensor.dim() > 1:
                            prediction = float(output_tensor.cpu().numpy()[0][0])
                        else:
                            prediction = float(output_tensor.cpu().numpy()[0])
                        print(f"   🤖 {ticker}: PyTorch model prediction successful: {prediction:.4f}")

                elif hasattr(model, 'predict'):
                    # Scikit-learn style models
                    prediction = model.predict(features_scaled)[0]
                    print(f"   🤖 {ticker}: Model.predict() successful: {float(prediction):.4f}")
                else:
                    # Fallback for other model types
                    prediction = model(features_scaled)[0]
                    print(f"   🤖 {ticker}: Model call successful: {float(prediction):.4f}")

                # ✅ FIX: Clip prediction BEFORE inverse transform to prevent extrapolation
                # For models outputting scaled values, clip to [-1, 1] range
                if y_scaler and hasattr(y_scaler, 'inverse_transform'):
                    # Clip to valid scaled range before inverse transform
                    prediction_clipped = np.clip(float(prediction), -1.0, 1.0)
                    if abs(prediction_clipped - float(prediction)) > 0.01:
                        print(f"   ⚠️ {ticker}: Clipped prediction from {float(prediction):.4f} to {prediction_clipped:.4f}")
                    prediction_pct = y_scaler.inverse_transform(np.array([[prediction_clipped]]))[0][0]
                    # ✅ Convert from percentage to decimal (y_scaler returns percentage like 50.0 for 50%)
                    prediction = prediction_pct / 100.0
                    print(f"   🔄 {ticker}: Y-scaler applied: {prediction_pct:.4f}% → {float(prediction):.4f} decimal")

                # ✅ FIX: Final validation - clip to reasonable return range (-100% to +200%)
                # No stock can lose more than 100%, and >200% returns are rare outliers
                final_prediction = np.clip(float(prediction), -1.0, 2.0)
                if abs(final_prediction - float(prediction)) > 0.01:
                    print(f"   ⚠️ {ticker}: Clipped final prediction from {float(prediction)*100:.2f}% to {final_prediction*100:.2f}%")

                print(f"   ✅ {ticker}: Final prediction = {final_prediction*100:.2f}%")
                return float(final_prediction)

            except Exception as e:
                print(f"   ❌ {ticker}: Prediction failed: {e}")
                return -np.inf

    except PredictionTimeoutError as e:
        print(f"   ⏰ {ticker}: {e}")
        return -np.inf
    except Exception as e:
        print(f"   ⚠️ Prediction failed for {ticker}: {type(e).__name__}: {str(e)[:100]}")
        return -np.inf


# -----------------------------------------------------------------------------
# Final Summary Printing
# -----------------------------------------------------------------------------
def print_final_summary(
    sorted_final_results: List[Dict],
    models: Dict,  # Changed from models_buy and models_sell
    scalers: Dict,
    optimized_params_per_ticker: Dict[str, Dict[str, float]],
    final_strategy_value_1y: float,
    final_buy_hold_value_1y: float,
    ai_1y_return: float,
    final_strategy_value_ytd: float,
    final_buy_hold_value_ytd: float,
    ai_ytd_return: float,
    final_strategy_value_3month: float,
    final_buy_hold_value_3month: float,
    ai_3month_return: float,
    initial_balance_used: float,
    num_tickers_analyzed: int,
    final_strategy_value_1month: float,
    ai_1month_return: float,
    final_buy_hold_value_1month: float,
    final_simple_rule_value_1y: float,
    simple_rule_1y_return: float,
    final_simple_rule_value_ytd: float,
    simple_rule_ytd_return: float,
    final_simple_rule_value_3month: float,
    simple_rule_3month_return: float,
    final_simple_rule_value_1month: float,
    simple_rule_1month_return: float,
    performance_metrics_simple_rule_1y: List[Dict],
    performance_metrics_buy_hold_1y: List[Dict],
    top_performers_data: List[Tuple],
    final_dynamic_bh_value_1y: float = None,
    dynamic_bh_1y_return: float = None,
    final_dynamic_bh_3m_value_1y: float = None,
    dynamic_bh_3m_1y_return: float = None,
    **kwargs
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    # === INVERSE ETF HEDGE SUMMARY ===
    from shared_strategies import get_inverse_etf_hedge_log
    hedge_log = get_inverse_etf_hedge_log()
    if hedge_log:
        print("\n🛡️ INVERSE ETF HEDGE SUMMARY:")
        print("-" * 60)
        # Count by ETF
        etf_counts = {}
        for date, etf, decline in hedge_log:
            if etf not in etf_counts:
                etf_counts[etf] = {'count': 0, 'max_decline': 0}
            etf_counts[etf]['count'] += 1
            etf_counts[etf]['max_decline'] = max(etf_counts[etf]['max_decline'], decline)

        print(f"  Total hedge activations: {len(hedge_log)}")
        for etf, data in sorted(etf_counts.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {etf}: {data['count']} times (max decline: {data['max_decline']:.1%})")

        # Show date range
        first_date = min(h[0] for h in hedge_log)
        last_date = max(h[0] for h in hedge_log)
        print(f"  Period: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
        print("-" * 60)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}")
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)")
    print(f"  1-Year Static BH Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    if final_dynamic_bh_value_1y is not None:
        print(f"  1-Year Dynamic BH Value: ${final_dynamic_bh_value_1y:,.2f} ({dynamic_bh_1y_return:+.2f}%)")
    if final_dynamic_bh_3m_value_1y is not None:
        print(f"  1-Year Dynamic BH 3M Value: ${final_dynamic_bh_3m_value_1y:,.2f} ({dynamic_bh_3m_1y_return:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 280)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Shares Before Liquidation':>25}")
    print("-" * 280)
    for res in sorted_final_results:
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        # Probability thresholds removed
        buy_thresh = 0.0  # Disabled
        sell_thresh = 1.0  # Disabled
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', PERIOD_HORIZONS.get("1-Year", 10))  # 10 calendar days
        opt_status = optimized_params.get('optimization_status', 'N/A')

        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('performance', 0.0) - allocated_capital

        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {shares_before_liquidation_str}")
    print("-" * 290)

    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 126)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 126)

    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital

        one_year_perf_benchmark = np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 126)

    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 126)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 126)

    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0

        one_year_perf_benchmark = np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 126)

    print("\n🤖 ML Model Status:")
    for ticker in sorted_final_results:
        t = ticker['ticker']
        model_status = "✅ Trained" if models.get(t) else "❌ Not Trained"
        print(f"  - {t}: TargetReturn Model: {model_status}")
    print("="*80)

    print("\n💡 Next Steps:")
    print("  - Review individual ticker performance and trade logs for deeper insights.")
    print("  - Experiment with different `MARKET_SELECTION` options and `N_TOP_TICKERS`.")
    print("  - Adjust `TARGET_PERCENTAGE` and `RISK_PER_TRADE` for different risk appetites.")
    print("  - Consider enabling `USE_MARKET_FILTER` and `USE_PERFORMANCE_BENCHMARK` for additional filtering.")
    print("  - Explore advanced ML models or feature engineering for further improvements.")
    print("="*80)


# Import shared function
from shared_strategies import calculate_volatility_adjusted_momentum



# Module for backtesting functions - not meant to be run directly
