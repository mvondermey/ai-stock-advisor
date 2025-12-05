"""
Optimization Phase Module
Handles ML parameter optimization for all periods (1-Year, YTD, 3-Month, 1-Month).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import os

from config import (
    MIN_PROBA_BUY, MIN_PROBA_SELL, TARGET_PERCENTAGE, CLASS_HORIZON,
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    TOP_CACHE_PATH, INVESTMENT_PER_STOCK, ENABLE_YTD_TRAINING,
    ENABLE_3MONTH_TRAINING, ENABLE_1MONTH_TRAINING
)
from backtesting import optimize_thresholds_for_portfolio_parallel
from alpha_training import AlphaThresholdConfig
from data_utils import _ensure_dir

# Re-export for convenience
__all__ = ['optimize_thresholds_for_portfolio_parallel']


def run_optimization_for_period(
    period_name: str,
    top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    capital_per_stock: float,
    target_percentage: float,
    class_horizon: int,
    force_thresholds_optimization: bool,
    force_percentage_optimization: bool,
    loaded_optimized_params: Dict,
    optimized_params_file: Path
) -> Tuple[Dict, Dict]:
    """
    Run optimization for a specific period.
    
    Returns:
        Tuple of (optimized_params_per_ticker, all_tested_combinations)
    """
    print(f"\n🔄 Optimizing ML parameters for {period_name} period...")
    optimization_params = []
    
    for ticker in top_tickers:
        if ticker in models_buy and ticker in models_sell and ticker in scalers:
            model_buy_ticker = models_buy[ticker]
            model_sell_ticker = models_sell[ticker]
            
            buy_model_type = type(model_buy_ticker).__name__ if model_buy_ticker else 'None'
            sell_model_type = type(model_sell_ticker).__name__ if model_sell_ticker else 'None'
            
            if model_buy_ticker is None or model_sell_ticker is None:
                print(f"  ⏭️  Skipping {period_name} optimization for {ticker}: Missing model (Buy: {buy_model_type}, Sell: {sell_model_type})")
                continue
            
            print(f"  ✅ Optimizing {ticker} for {period_name}: Buy={buy_model_type}, Sell={sell_model_type}")
            
            current_min_proba_buy_for_opt = loaded_optimized_params.get(ticker, {}).get('min_proba_buy', MIN_PROBA_BUY)
            current_min_proba_sell_for_opt = loaded_optimized_params.get(ticker, {}).get('min_proba_sell', MIN_PROBA_SELL)
            current_target_percentage_for_opt = loaded_optimized_params.get(ticker, {}).get('target_percentage', target_percentage)
            current_class_horizon_for_opt = loaded_optimized_params.get(ticker, {}).get('class_horizon', class_horizon)

            feature_set_for_opt = scalers[ticker].feature_names_in_ if hasattr(scalers[ticker], 'feature_names_in_') else None

            try:
                ticker_train_data = all_tickers_data.loc[train_start:train_end, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                if ticker_train_data.empty:
                    print(f"  ⚠️ Could not get {period_name} training data for {ticker} for optimization. Skipping.")
                    continue
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice {period_name} training data for {ticker} for optimization. Skipping.")
                continue

            optimization_params.append((
                ticker,
                ticker_train_data.copy(),
                capital_per_stock,
                current_target_percentage_for_opt,
                current_class_horizon_for_opt,
                force_thresholds_optimization,
                force_percentage_optimization,
                True,  # USE_ALPHA_THRESHOLD_BUY
                True,  # USE_ALPHA_THRESHOLD_SELL
                AlphaThresholdConfig(rebalance_freq="D", metric="alpha", costs_bps=5.0, slippage_bps=2.0),
                current_min_proba_buy_for_opt,
                current_min_proba_sell_for_opt,
                current_target_percentage_for_opt,
                current_class_horizon_for_opt,
                GRU_TARGET_PERCENTAGE_OPTIONS,
                GRU_CLASS_HORIZON_OPTIONS,
                42,  # SEED
                feature_set_for_opt
            ))
    
    if optimization_params:
        optimized_params_per_ticker, all_tested_combinations = optimize_thresholds_for_portfolio_parallel(optimization_params)
        
        # Print backtest results for each tested combination
        if all_tested_combinations:
            print("\n" + "="*80)
            print(f"📊 Backtest Results for All Tested Optimization Combinations ({period_name})")
            print("="*80)
            for ticker, combinations in all_tested_combinations.items():
                if not combinations:
                    continue
                print(f"\n📈 {ticker} - Tested {len(combinations)} combinations:")
                print("-" * 100)
                sorted_combinations = sorted(combinations, key=lambda x: x.get('revenue', -np.inf), reverse=True)
                print(f"{'Rank':<6} | {'Buy Thresh':<12} | {'Sell Thresh':<12} | {'Target %':<10} | {'Horizon':<8} | {'AI Revenue':<15} | {'B&H Revenue':<15} | {'Difference':<15}")
                print("-" * 100)
                for idx, combo in enumerate(sorted_combinations[:20], 1):
                    revenue = combo.get('revenue', capital_per_stock)
                    buy_hold_revenue = combo.get('buy_hold_revenue', 0.0)
                    revenue_pct = ((revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                    bh_revenue_pct = ((buy_hold_revenue) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                    diff = revenue - buy_hold_revenue
                    diff_pct = revenue_pct - bh_revenue_pct
                    
                    print(f"{idx:<6} | {combo.get('min_proba_buy', 0.0):>11.2f} | {combo.get('min_proba_sell', 0.0):>11.2f} | "
                          f"{combo.get('target_percentage', 0.0):>9.2%} | {combo.get('class_horizon', 0):>7} | "
                          f"${revenue:>13,.2f} ({revenue_pct:>+6.2f}%) | ${buy_hold_revenue:>13,.2f} ({bh_revenue_pct:>+6.2f}%) | "
                          f"${diff:>13,.2f} ({diff_pct:>+6.2f}%)")
                if len(sorted_combinations) > 20:
                    print(f"... and {len(sorted_combinations) - 20} more combinations")
                print("-" * 100)
            print("="*80 + "\n")

        if optimized_params_per_ticker and period_name == "1-Year":
            try:
                with open(optimized_params_file, 'w') as f:
                    json.dump(optimized_params_per_ticker, f, indent=4)
                print(f"✅ Optimized parameters saved to {optimized_params_file}")
            except Exception as e:
                print(f"⚠️ Could not save optimized parameters to file: {e}")
        
        return optimized_params_per_ticker, all_tested_combinations
    else:
        return {}, {}

