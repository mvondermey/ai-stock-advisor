"""
backtesting.py
Final version – includes 1D sequential optimisation, compatible with main.py and accepts extra kwargs (e.g., top_tickers).
"""

import numpy as np
import sys
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
from backtesting_env import RuleTradingEnv
from ml_models import initialize_ml_libraries, train_and_evaluate_models
from data_utils import load_prices, fetch_training_data, _ensure_dir
import logging
from pathlib import Path
from config import (
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS, MIN_PROBA_BUY, MIN_PROBA_SELL,
    USE_MODEL_GATE, TRANSACTION_COST, SEED, INVESTMENT_PER_STOCK,
    BACKTEST_DAYS, TRAIN_LOOKBACK_DAYS, BACKTEST_DAYS_3MONTH, BACKTEST_DAYS_1MONTH,
    N_TOP_TICKERS, USE_PERFORMANCE_BENCHMARK, PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, NUM_PROCESSES
)
from config import (
    ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, TARGET_PERCENTAGE, CLASS_HORIZON,
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, USE_LSTM, USE_GRU # Moved from ml_models
)
from alpha_training import AlphaThresholdConfig, select_threshold_by_alpha
from scipy.stats import uniform, beta
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
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor
            if isinstance(model, (LSTMClassifier, GRUClassifier, GRURegressor)):
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
                        output_size = numpy_state_dict[key].shape[0]
                
                # Get dropout from model's RNN layer
                if hasattr(model_cpu, 'gru'):
                    dropout = model_cpu.gru.dropout if hasattr(model_cpu.gru, 'dropout') else 0.0
                elif hasattr(model_cpu, 'lstm'):
                    dropout = model_cpu.lstm.dropout if hasattr(model_cpu.lstm, 'dropout') else 0.0
                
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
                return model_info
            else:
                # For non-PyTorch models (LightGBM, XGBoost, etc.), return as-is
                return model
        except (ImportError, AttributeError, Exception) as e:
            print(f"  ⚠️ Error preparing model for multiprocessing: {e}")
            return None
    return model  # For non-PyTorch models, return as-is


def _reconstruct_model_from_info(model_info, device='cuda'):
    """Reconstruct a PyTorch model from model_info (with numpy arrays) on the specified device."""
    if model_info is None:
        return None
    if isinstance(model_info, dict) and 'type' in model_info:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor
            
            model_type = model_info['type']
            numpy_state_dict = model_info['state_dict']
            
            # Convert numpy arrays back to tensors on the target device
            state_dict = {}
            for key, value in numpy_state_dict.items():
                if isinstance(value, np.ndarray):
                    # Convert numpy array to tensor and move to device
                    state_dict[key] = torch.from_numpy(value).to(device)
                else:
                    state_dict[key] = value
            
            # Reconstruct model based on type
            if model_type == 'GRUClassifier':
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
            
            # Load state dict (already on device) and set to eval mode
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except (ImportError, AttributeError, Exception) as e:
            return None
    return model_info  # For non-PyTorch models, return as-is


# -----------------------------------------------------------------------------
# Single-ticker optimisation worker
# -----------------------------------------------------------------------------
def optimize_single_ticker_worker(params):
    (
        ticker, train_data, capital_per_stock, target_percentage, class_horizon,
        force_thresholds_optimization, force_percentage_optimization,
        use_alpha_threshold_buy, use_alpha_threshold_sell,
        alpha_config, current_min_proba_buy, current_min_proba_sell,
        initial_target_percentage, initial_class_horizon,
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
            'target_percentage': initial_target_percentage,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no data)"
        }

    # Reconstruct PyTorch models on GPU if they were passed as model_info dicts
    if PYTORCH_AVAILABLE:
        import torch
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)
    
    if model_buy is None or model_sell is None or scaler is None:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Models or scaler not provided. Skipping optimization.\n")
        return {
            'ticker': ticker,
            'min_proba_buy': current_min_proba_buy,
            'min_proba_sell': current_min_proba_sell,
            'target_percentage': initial_target_percentage,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no models)"
        }

    best_alpha = -np.inf
    best_revenue = -np.inf
    best_min_proba_buy = current_min_proba_buy
    best_min_proba_sell = current_min_proba_sell
    best_target_percentage = initial_target_percentage
    best_class_horizon = initial_class_horizon

    # Store all tested combinations for backtesting
    tested_combinations = []  # List of dicts with params and revenue

    # ITERATIVE HILL-CLIMBING OPTIMIZATION
    # Start from current saved thresholds, try one step up/down, move if better
    from config import MIN_PROBA_BUY_OPTIONS, MIN_PROBA_SELL_OPTIONS
    
    buy_options = sorted(MIN_PROBA_BUY_OPTIONS if MIN_PROBA_BUY_OPTIONS else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70])
    sell_options = sorted(MIN_PROBA_SELL_OPTIONS if MIN_PROBA_SELL_OPTIONS else [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    
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
            model_buy=model_buy,
            model_sell=model_sell,
            scaler=scaler,
            per_ticker_min_proba_buy=p_buy,
            per_ticker_min_proba_sell=p_sell,
            use_gate=USE_MODEL_GATE,
            feature_set=feature_set,
            use_simple_rule_strategy=False
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
            strategy_returns = strategy_values.pct_change().dropna()
            
            # Calculate daily returns for buy & hold (price changes)
            close_prices = pd.to_numeric(df_backtest_opt["Close"], errors='coerce').dropna()
            bh_returns = close_prices.pct_change().dropna()
            
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
                    alpha_annualized = alpha_per_day * 252  # Annualize (252 trading days)
                except Exception as e:
                    alpha_annualized = 0.0
            else:
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
        'target_percentage': initial_target_percentage,
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
                'target_percentage': initial_target_percentage,
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
    
    best_target_percentage = initial_target_percentage
    best_class_horizon = initial_class_horizon
    optimization_status = "Optimized" if iteration > 1 else "No Change"
    if not np.isclose(best_min_proba_buy, current_min_proba_buy) or \
       not np.isclose(best_min_proba_sell, current_min_proba_sell) or \
       not np.isclose(best_target_percentage, initial_target_percentage) or \
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
                np.isclose(combo['target_percentage'], best_target_percentage) and
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
                best_target_percentage = best_beating_bh['target_percentage']
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
    print(f"     Target={best_target_percentage:.4f}, Horizon={best_class_horizon}, Buy={best_min_proba_buy:.2f}, Sell={best_min_proba_sell:.2f}")
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
        'target_percentage': best_target_percentage,
        'class_horizon': best_class_horizon,
        'best_revenue': best_revenue,
        'buy_hold_revenue': buy_hold_final_val_best,
        'revenue_beats_bh': final_revenue_beats_bh,  # Add explicit flag
        'optimization_status': optimization_status,
        'tested_combinations': tested_combinations  # Return all tested combinations
    }

def optimize_thresholds_for_portfolio_parallel(optimization_params, num_processes=None):
    num_processes = num_processes or max(1, cpu_count() - 5)
    print(f"\nOptimizing thresholds for {len(optimization_params)} tickers using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(optimize_single_ticker_worker, optimization_params),
            total=len(optimization_params),
            desc="Optimizing Thresholds"
        ))

    optimized_params = {}
    all_tested_combinations = {}  # Store all tested combinations per ticker
    
    for res in results:
        if res and res.get('ticker'):
            optimized_params[res['ticker']] = {
                k: res[k] for k in ['min_proba_buy', 'min_proba_sell', 'target_percentage', 'class_horizon', 'optimization_status', 'revenue_beats_bh']
            }
            if 'tested_combinations' in res and res['tested_combinations']:
                all_tested_combinations[res['ticker']] = res['tested_combinations']
            beats_bh_status = "✅ Beats B&H" if res.get('revenue_beats_bh', False) else "❌ Below B&H"
            print(f"Optimized {res['ticker']}: Buy>{res['min_proba_buy']:.2f}, Sell>{res['min_proba_sell']:.2f}, "
                  f"Target={res['target_percentage']:.3%}, Horizon={res['class_horizon']}d → {res['optimization_status']} | {beats_bh_status}")

    return optimized_params, all_tested_combinations


# -----------------------------------------------------------------------------
# Backtest worker function
# -----------------------------------------------------------------------------
def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy = params
    
    # Initial log to confirm the worker has started for a ticker
    with open("logs/worker_debug.log", "a") as f:
        f.write(f"Worker started for ticker: {ticker}\n")

    # Reconstruct PyTorch models from prepared dict format
    if PYTORCH_AVAILABLE:
        import torch
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)

    if df_backtest.empty:
        print(f"  ⚠️ Skipping backtest for {ticker}: DataFrame is empty.")
        return None
        
    try:
        env = RuleTradingEnv(
            df=df_backtest.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model_buy=model_buy,
            model_sell=model_sell,
            scaler=scaler,
            use_gate=USE_MODEL_GATE,
            feature_set=feature_set,
            per_ticker_min_proba_buy=None,
            per_ticker_min_proba_sell=None,
            use_simple_rule_strategy=use_simple_rule_strategy
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

        # Print prediction summary
        if hasattr(env, 'all_predictions_buy') and len(env.all_predictions_buy) > 0:
            import numpy as np
            preds_buy = np.array(env.all_predictions_buy)
            print(f"\n📊 [{ticker}] BUY Prediction Summary:")
            print(f"   Min: {np.min(preds_buy)*100:.4f}%, Max: {np.max(preds_buy)*100:.4f}%, Mean: {np.mean(preds_buy)*100:.4f}%")
            print(f"   Above 5% threshold: {np.sum(preds_buy >= 0.05)} out of {len(preds_buy)} days ({np.sum(preds_buy >= 0.05)/len(preds_buy)*100:.1f}%)")
            print(f"   Threshold: {min_proba_buy*100:.2f}%")

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob,
            'shares_before_liquidation': shares_before_liquidation,
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

    # Sharpe Ratio (annualized, assuming 252 trading days)
    sharpe_strat = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() > 0 else 0
    sharpe_bh = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if bh_returns.std() > 0 else 0

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
def _run_portfolio_backtest(
    all_tickers_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    capital_per_stock: float,
    target_percentage: float,
    run_parallel: bool,
    period_name: str,
    top_performers_data: List[Tuple],
    use_simple_rule_strategy: bool = False
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]:
    """Helper function to run portfolio backtest for a given period."""
    num_processes = cpu_count() - 5 # Use NUM_PROCESSES from config if available, otherwise default

    backtest_params = []
    for ticker in top_tickers:
        # Use optimized parameters if available, otherwise fall back to global defaults
        min_proba_buy_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_buy', MIN_PROBA_BUY)
        min_proba_sell_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_sell', MIN_PROBA_SELL)
        target_percentage_ticker = optimized_params_per_ticker.get(ticker, {}).get('target_percentage', target_percentage)

        # Ensure feature_set is passed to backtest_worker
        feature_set_for_worker = scalers.get(ticker).feature_names_in_ if scalers.get(ticker) and hasattr(scalers.get(ticker), 'feature_names_in_') else None
        
        # Slice the main DataFrame for the backtest period for this specific ticker
        try:
            ticker_backtest_data = all_tickers_data.loc[start_date:end_date, (slice(None), ticker)]
            ticker_backtest_data.columns = ticker_backtest_data.columns.droplevel(1)
            if ticker_backtest_data.empty:
                print(f"  ⚠️ Sliced backtest data for {ticker} for period {period_name} is empty. Skipping.")
                continue
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice backtest data for {ticker} for period {period_name}. Skipping.")
            continue

        # Prepare PyTorch models for multiprocessing
        model_buy_prepared = _prepare_model_for_multiprocessing(models_buy.get(ticker))
        model_sell_prepared = _prepare_model_for_multiprocessing(models_sell.get(ticker))
        
        backtest_params.append((
            ticker, ticker_backtest_data.copy(), capital_per_stock,
            model_buy_prepared, model_sell_prepared, scalers.get(ticker),
            feature_set_for_worker, min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data, use_simple_rule_strategy
        ))

    portfolio_values = []
    processed_tickers = []
    performance_metrics = []
    buy_hold_histories_per_ticker: Dict[str, List[float]] = {}
    
    total_tickers_to_process = len(top_tickers)
    processed_count = 0

    if run_parallel:
        print(f"📈 Running {period_name} backtest in parallel for {total_tickers_to_process} tickers using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            results = []
            for res in tqdm(pool.imap(backtest_worker, backtest_params), total=total_tickers_to_process, desc=f"Backtesting {period_name}"):
                if res:
                    print(f"  [DEBUG] Ticker: {res['ticker']}, Final Value: {res['final_val']}")
                    portfolio_values.append(res['final_val'])
                    processed_tickers.append(res['ticker'])
                    performance_metrics.append(res)
                    buy_hold_histories_per_ticker[res['ticker']] = res.get('buy_hold_history', [])
                    
                    perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                    for t, p1y, pytd in top_performers_data:
                        if t == res['ticker']:
                            perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                            perf_ytd_benchmark = pytd if np.isfinite(pytd) else np.nan
                            break
                    
                    print(f"\n📈 Individual Stock Performance for {res['ticker']} ({period_name}):")
                    print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                    print(f"  - YTD Performance: {perf_ytd_benchmark:.2f}%" if pd.notna(perf_ytd_benchmark) else "  - YTD Performance: N/A")
                    print(f"  - AI Sharpe Ratio: {res['perf_data']['sharpe_ratio']:.2f}")
                    print(f"  - Last AI Action: {res['last_ai_action']}")
                    print(f"  - Optimized Buy Threshold: {optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY):.2f}")
                    print(f"  - Optimized Sell Threshold: {optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL):.2f}")
                    print(f"  - Optimized Target Percentage: {optimized_params_per_ticker.get(res['ticker'], {}).get('target_percentage', TARGET_PERCENTAGE):.2%}")
                    
                    # ✅ FIX 3: Add diagnostic for 0 trades
                    total_trades = res.get('perf_data', {}).get('total_trades', 0)
                    if total_trades == 0 and res.get('last_ai_action') == 'HOLD':
                        buy_prob = res.get('buy_prob', 0.0)
                        sell_prob = res.get('sell_prob', 0.0)
                        buy_thresh = optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY)
                        sell_thresh = optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL)
                        print(f"\n  ⚠️  WARNING: {res['ticker']} made 0 trades!")
                        print(f"      Last Buy Probability: {buy_prob:.4f} (threshold: {buy_thresh:.2f})")
                        print(f"      Last Sell Probability: {sell_prob:.4f} (threshold: {sell_thresh:.2f})")
                        if buy_prob < buy_thresh and sell_prob < sell_thresh:
                            print(f"      💡 Model probabilities never exceeded thresholds.")
                            print(f"         Consider: 1) Lowering thresholds, 2) Retraining with different targets, or 3) Using Simple Rule strategy")
                    
                    print("-" * 40)
                processed_count += 1
    else:
        print(f"📈 Running {period_name} backtest sequentially for {total_tickers_to_process} tickers...")
        results = []
        for res in tqdm(backtest_params, desc=f"Backtesting {period_name}"):
            worker_result = backtest_worker(res)
            if worker_result:
                print(f"  [DEBUG] Ticker: {worker_result['ticker']}, Final Value: {worker_result['final_val']}")
                portfolio_values.append(worker_result['final_val'])
                processed_tickers.append(worker_result['ticker'])
                performance_metrics.append(worker_result)
                buy_hold_histories_per_ticker[worker_result['ticker']] = worker_result.get('buy_hold_history', [])
                
                perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                for t, p1y, pytd in top_performers_data:
                    if t == worker_result['ticker']:
                        perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                        ytd_perf_benchmark = pytd if np.isfinite(pytd) else np.nan
                        break
                
                print(f"\n📈 Individual Stock Performance for {worker_result['ticker']} ({period_name}):")
                print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                print(f"  - YTD Performance: {ytd_perf_benchmark:.2f}%" if pd.notna(ytd_perf_benchmark) else "  - YTD Performance: N/A")
                print(f"  - AI Sharpe Ratio: {worker_result['perf_data']['sharpe_ratio']:.2f}")
                print(f"  - Last AI Action: {worker_result['last_ai_action']}")
                print(f"  - Optimized Buy Threshold: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY):.2f}")
                print(f"  - Optimized Sell Threshold: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL):.2f}")
                print(f"  - Optimized Target Percentage: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('target_percentage', TARGET_PERCENTAGE):.2%}")
                
                # ✅ FIX 3: Add diagnostic for 0 trades
                total_trades = worker_result.get('perf_data', {}).get('total_trades', 0)
                if total_trades == 0 and worker_result.get('last_ai_action') == 'HOLD':
                    buy_prob = worker_result.get('buy_prob', 0.0)
                    sell_prob = worker_result.get('sell_prob', 0.0)
                    buy_thresh = optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY)
                    sell_thresh = optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL)
                    print(f"\n  ⚠️  WARNING: {worker_result['ticker']} made 0 trades!")
                    print(f"      Last Buy Probability: {buy_prob:.4f} (threshold: {buy_thresh:.2f})")
                    print(f"      Last Sell Probability: {sell_prob:.4f} (threshold: {sell_thresh:.2f})")
                    if buy_prob < buy_thresh and sell_prob < sell_thresh:
                        print(f"      💡 Model probabilities never exceeded thresholds.")
                        print(f"         Consider: 1) Lowering thresholds, 2) Retraining with different targets, or 3) Using Simple Rule strategy")
                
                print("-" * 40)
            processed_count += 1

    valid_portfolio_values = [v for v in portfolio_values if v is not None and np.isfinite(v)]
    
    final_portfolio_value = sum(valid_portfolio_values) + (total_tickers_to_process - len(processed_tickers)) * capital_per_stock
    print(f"✅ {period_name} Backtest complete. Final portfolio value: ${final_portfolio_value:,.2f}\n")
    return final_portfolio_value, portfolio_values, processed_tickers, performance_metrics, buy_hold_histories_per_ticker

# -----------------------------------------------------------------------------
# Final Summary Printing
# -----------------------------------------------------------------------------
def print_final_summary(
    sorted_final_results: List[Dict],
    models_buy: Dict,
    models_sell: Dict,
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
    top_performers_data: List[Tuple]
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}")
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)")
    print(f"  1-Year Buy & Hold Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  YTD AI Strategy Value: ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)")
    print(f"  YTD Simple Rule Value: ${final_simple_rule_value_ytd:,.2f} ({simple_rule_ytd_return:+.2f}%)")
    print(f"  YTD Buy & Hold Value: ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  3-Month AI Strategy Value: ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)")
    print(f"  3-Month Simple Rule Value: ${final_simple_rule_value_3month:,.2f} ({simple_rule_3month_return:+.2f}%)")
    print(f"  3-Month Buy & Hold Value: ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  1-Month AI Strategy Value: ${final_strategy_value_1month:,.2f} ({ai_1month_return:+.2f}%)")
    print(f"  1-Month Simple Rule Value: ${final_simple_rule_value_1month:,.2f} ({simple_rule_1month_return:+.2f}%)")
    print(f"  1-Month Buy & Hold Value: ${final_buy_hold_value_1month:,.2f} ({((final_buy_hold_value_1month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 290)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Shares Before Liquidation':>25}")
    print("-" * 290)
    for res in sorted_final_results:
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', CLASS_HORIZON)
        opt_status = optimized_params.get('optimization_status', 'N/A')

        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('performance', 0.0) - allocated_capital

        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {shares_before_liquidation_str}")
    print("-" * 290)

    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital
        
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 136)

    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
        
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 136)

    print("\n🤖 ML Model Status:")
    for ticker in sorted_final_results:
        t = ticker['ticker']
        buy_model_status = "✅ Trained" if models_buy.get(t) else "❌ Not Trained"
        sell_model_status = "✅ Trained" if models_sell.get(t) else "❌ Not Trained"
        print(f"  - {t}: Buy Model: {buy_model_status}, Sell Model: {sell_model_status}")
    print("="*80)

    print("\n💡 Next Steps:")
    print("  - Review individual ticker performance and trade logs for deeper insights.")
    print("  - Experiment with different `MARKET_SELECTION` options and `N_TOP_TICKERS`.")
    print("  - Adjust `TARGET_PERCENTAGE` and `RISK_PER_TRADE` for different risk appetites.")
    print("  - Consider enabling `USE_MARKET_FILTER` and `USE_PERFORMANCE_BENCHMARK` for additional filtering.")
    print("  - Explore advanced ML models or feature engineering for further improvements.")
    print("="*80)


if __name__ == "__main__":
    print("Run from main.py with FORCE_THRESHOLDS_OPTIMIZATION=True to execute optimisation.")
