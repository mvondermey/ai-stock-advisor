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
from data_utils import load_prices, fetch_training_data, _ensure_dir, _calculate_technical_indicators
import logging
from pathlib import Path
from config import (
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    TRANSACTION_COST, SEED, INVESTMENT_PER_STOCK,
    BACKTEST_DAYS, TRAIN_LOOKBACK_DAYS,
    N_TOP_TICKERS, USE_PERFORMANCE_BENCHMARK, PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, NUM_PROCESSES, SEQUENCE_LENGTH,
    RETRAIN_FREQUENCY_DAYS
)
from config import (
    ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, TARGET_PERCENTAGE, CLASS_HORIZON,
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, USE_LSTM, USE_GRU # Moved from ml_models
)
from alpha_training import AlphaThresholdConfig, select_threshold_by_alpha
from scipy.stats import uniform, beta
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


def _reconstruct_model_from_info(model_info, device='cuda'):
    """Reconstruct a PyTorch model from model_info (with numpy arrays) on the specified device."""
    if model_info is None:
        return None
    if isinstance(model_info, dict) and 'type' in model_info:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor

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
                    state_dict[key] = torch.from_numpy(value)  # keep on CPU for load_state_dict
                else:
                    state_dict[key] = value

            # Reconstruct model based on type (on CPU)
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
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, y_scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy, horizon_days = params
    
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
    y_scalers: Dict,  # ✅ Added y_scalers parameter
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    capital_per_stock: float,
    target_percentage: float,
    run_parallel: bool,
    period_name: str,
    top_performers_data: List[Tuple],
    use_simple_rule_strategy: bool = False,
    horizon_days: int = 20
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]:
    """Helper function to run portfolio backtest for a given period."""
    num_processes = max(1, cpu_count() - 5) # Use NUM_PROCESSES from config if available, otherwise default

    backtest_params = []
    preview_predictions: List[Tuple[str, float]] = []

    def quick_last_prediction(ticker: str, df_slice: pd.DataFrame, model, scaler, y_scaler, feature_set, horizon_days: int):
        """
        Use the trained model to predict returns for ranking.
        Engineers features from raw OHLCV data first.
        """
        try:
            if model is None or scaler is None:
                # For simple rule strategy, allow neutral score so tickers are retained
                if use_simple_rule_strategy:
                    return 0.0
                return -np.inf

            # Check if we have required OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_slice.columns for col in required_cols):
                return -np.inf

            # STEP 1: Engineer features from raw OHLCV data
            df_with_features = df_slice.copy()
            df_with_features = _calculate_technical_indicators(df_with_features)
            df_with_features = df_with_features.dropna()

            # If not enough rows remain after feature calc, bail out early
            if df_with_features.empty:
                return -np.inf

            # Get latest data point
            latest_data = df_with_features.iloc[-1:]

            # Scale features
            if scaler:
                features_scaled = scaler.transform(latest_data.values.reshape(1, -1))
            else:
                features_scaled = latest_data.values.reshape(1, -1)

            # Predict return
            if hasattr(model, 'predict'):
                prediction = model.predict(features_scaled)[0]
            else:
                # Handle different model types
                prediction = model(latest_data.values.reshape(1, -1, -1, -1) if hasattr(model, '__call__') else features_scaled)[0]

            # Unscale if y_scaler exists
            if y_scaler and hasattr(y_scaler, 'inverse_transform'):
                prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

            return float(prediction)

        except Exception as e:
            return -np.inf

    # Prepare backtest data for each ticker
    for ticker in top_tickers:
        try:
            # Get backtest data slice
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                print(f"  ⚠️ No data found for {ticker} in backtest period")
                continue

            ticker_data = ticker_data.set_index('date')
            ticker_backtest_data = ticker_data.loc[start_date:end_date]

            if ticker_backtest_data.empty:
                print(f"  ⚠️ No backtest data for {ticker} in period {period_name}")
                continue

        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice backtest data for {ticker} for period {period_name}. Skipping.")
            continue

        # Quick prediction for ranking (use buy model)
        preview_pred = quick_last_prediction(
            ticker,
            ticker_backtest_data,
            models_buy.get(ticker),
            scalers.get(ticker),
            y_scalers.get(ticker),
            [],  # feature_set_for_worker - empty for now
            horizon_days
        )
        preview_predictions.append((ticker, preview_pred))

        # Prepare PyTorch models for multiprocessing
        model_buy_prepared = _prepare_model_for_multiprocessing(models_buy.get(ticker))
        model_sell_prepared = _prepare_model_for_multiprocessing(models_sell.get(ticker))

        backtest_params.append((
            ticker, ticker_backtest_data.copy(), capital_per_stock,
            model_buy_prepared, model_sell_prepared, scalers.get(ticker), y_scalers.get(ticker),  # ✅ Added y_scaler
            [], min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data, use_simple_rule_strategy, horizon_days
        ))

    # Keep only top 3 tickers by AI-predicted return
    preview_predictions = sorted(preview_predictions, key=lambda x: x[1], reverse=True)
    allowed_tickers = set([t for t, _ in preview_predictions[:3]])
    # DEBUG: show all candidate predictions and the selected top 3
    print(f"\n[DEBUG] Candidate predicted returns for {period_name}:")
    for t, p in preview_predictions:
        print(f"  - {t}: {p:.4f}")
    print(f"[DEBUG] Selected top 3 for backtest: {list(allowed_tickers)}")

    backtest_params = [p for p in backtest_params if p[0] in allowed_tickers]
    top_tickers = [p[0] for p in backtest_params]

    portfolio_values = []
    processed_tickers = []
    performance_metrics = []
    buy_hold_histories_per_ticker: Dict[str, List[float]] = {}

    total_tickers_to_process = len(top_tickers)
    processed_count = 0

    if run_parallel and total_tickers_to_process > 1:
        # Run backtests in parallel
        with Pool(processes=min(num_processes, total_tickers_to_process)) as pool:
            for result in pool.imap(backtest_worker, backtest_params):
                if result:
                    ticker, final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation, buy_hold_history = result

                    processed_tickers.append(ticker)
                    portfolio_values.append(final_val)
                    buy_hold_histories_per_ticker[ticker] = buy_hold_history

                    # Calculate performance metrics
                    perf_metrics = _calculate_performance_metrics(trade_log, buy_hold_history, final_val, capital_per_stock)
                    perf_metrics.update({
                        'ticker': ticker,
                        'last_ai_action': last_ai_action,
                        'last_buy_prob': last_buy_prob,
                        'last_sell_prob': last_sell_prob,
                        'final_shares': shares_before_liquidation
                    })
                    performance_metrics.append(perf_metrics)

                processed_count += 1
                if processed_count % 5 == 0:
                    print(f"  📊 Processed {processed_count}/{total_tickers_to_process} tickers for {period_name}")
    else:
        # Run backtests sequentially
        for params in backtest_params:
            result = backtest_worker(params)
            if result:
                ticker, final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation, buy_hold_history = result

                processed_tickers.append(ticker)
                portfolio_values.append(final_val)
                buy_hold_histories_per_ticker[ticker] = buy_hold_history

                # Calculate performance metrics
                perf_metrics = _calculate_performance_metrics(trade_log, buy_hold_history, final_val, capital_per_stock)
                perf_metrics.update({
                    'ticker': ticker,
                    'last_ai_action': last_ai_action,
                    'last_buy_prob': last_buy_prob,
                    'last_sell_prob': last_sell_prob,
                    'final_shares': shares_before_liquidation
                })
                performance_metrics.append(perf_metrics)

    # Calculate total portfolio value
    total_portfolio_value = sum(portfolio_values) if portfolio_values else capital_per_stock * len(top_tickers)

    return total_portfolio_value, portfolio_values, processed_tickers, performance_metrics, buy_hold_histories_per_ticker


def _run_portfolio_backtest_walk_forward(
    all_tickers_data: pd.DataFrame,
    train_start_date: datetime,
    backtest_start_date: datetime,
    backtest_end_date: datetime,
    initial_top_tickers: List[str],
    initial_models: Dict,  # Single regression models
    initial_scalers: Dict,
    initial_y_scalers: Dict,
    capital_per_stock: float,
    target_percentage: float,
    period_name: str,
    top_performers_data: List[Tuple],
    horizon_days: int = 20
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]:
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
    print(f"   🧠 Model retraining: Every {RETRAIN_FREQUENCY_DAYS} days for all {len(initial_top_tickers)} stocks")
    print(f"   🎯 Daily selection: Pick best 3 from {len(initial_top_tickers)} stocks EVERY DAY using current models")
    print(f"   💰 Rebalance only when portfolio changes (transaction costs minimized)")
    
    # ✅ FIX 7: Add data structure validation
    print(f"\n🔍 Validating input data structure...")
    print(f"   - all_tickers_data type: {type(all_tickers_data)}")
    print(f"   - all_tickers_data shape: {all_tickers_data.shape}")
    print(f"   - all_tickers_data columns: {list(all_tickers_data.columns)}")
    
    if 'ticker' in all_tickers_data.columns:
        print(f"   ✅ 'ticker' column found")
        print(f"   - Unique tickers in data: {len(all_tickers_data['ticker'].unique())}")
        print(f"   - Sample tickers: {list(all_tickers_data['ticker'].unique())[:5]}")
    else:
        print(f"   ❌ 'ticker' column NOT found! This will cause prediction failures.")
        print(f"   - Checking if MultiIndex columns: {isinstance(all_tickers_data.columns, pd.MultiIndex)}")
        
    if 'date' in all_tickers_data.columns:
        print(f"   ✅ 'date' column found")
    else:
        print(f"   ❌ 'date' column NOT found!")

    # Implement day-by-day walk-forward backtesting with daily selection
    from training_phase import train_models_for_period

    # Initialize
    current_models = initial_models.copy()  # Single regression models
    current_scalers = initial_scalers.copy()
    current_y_scalers = initial_y_scalers.copy()

    # Debug: Check initial models/scalers
    print(f"   🔍 Initial models: {len(current_models)} tickers, sample: {list(current_models.keys())[:3]}")
    print(f"   🔍 Initial scalers: {len(current_scalers)} tickers, sample: {list(current_scalers.keys())[:3]}")
    print(f"   🔍 Initial y_scalers: {len(current_y_scalers)} tickers, sample: {list(current_y_scalers.keys())[:3]}")

    # Check if any models are None
    none_models = [t for t, m in current_models.items() if m is None]
    if none_models:
        print(f"   ⚠️ Warning: {len(none_models)} models are None: {none_models[:5]}...")

    none_scalers = [t for t, s in current_scalers.items() if s is None]
    if none_scalers:
        print(f"   ⚠️ Warning: {len(none_scalers)} scalers are None: {none_scalers[:5]}...")

    # Track current portfolio (starts empty)
    current_portfolio_stocks = []
    total_portfolio_value = 0.0  # Start with no capital invested
    portfolio_values_history = [total_portfolio_value]

    # Track actual positions for proper portfolio management
    positions = {}  # ticker -> {'shares': float, 'avg_price': float, 'value': float}
    cash_balance = 0.0  # Available cash

    # Calculate initial capital that should be allocated (3 stocks * capital_per_stock)
    initial_capital_needed = 3 * capital_per_stock
    cash_balance = initial_capital_needed  # Start with cash available for initial purchases

    all_processed_tickers = []
    all_performance_metrics = []
    all_buy_hold_histories = {}

    # Get all trading days in the backtest period
    date_range = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq='D')
    business_days = [d for d in date_range if d.weekday() < 5]  # Filter to weekdays

    print(f"   📅 Total trading days to process: {len(business_days)}")

    day_count = 0
    retrain_count = 0
    rebalance_count = 0
    
    # ✅ NEW: Track consecutive failures for fail-fast
    consecutive_no_predictions = 0
    consecutive_training_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # Abort if 5 days in a row fail
    
    # ✅ NEW: Track daily predictions vs actuals
    daily_prediction_log = []

    for current_date in business_days:
        day_count += 1

        # Check if it's time to retrain (every RETRAIN_FREQUENCY_DAYS)
        should_retrain = (day_count % RETRAIN_FREQUENCY_DAYS == 1)  # Retrain on day 1, 11, 21, etc.

        # ✅ FIX: Train models on Day 1 if initial_models is empty, OR on regular retrain schedule
        needs_training = (day_count == 1 and not current_models) or (should_retrain and day_count > 1)
        
        if needs_training:
            retrain_count += 1
            print(f"\n🧠 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): {'Initial training' if day_count == 1 else 'Retraining'} models...")

            try:
                # Retrain models using data up to previous day
                train_end_date = current_date - timedelta(days=1)

                retraining_results = train_models_for_period(
                    period_name=f"{period_name}_retrain_{retrain_count}",
                    tickers=initial_top_tickers,  # Retrain on all 40 tickers
                    all_tickers_data=all_tickers_data,
                    train_start=train_start_date,
                    train_end=train_end_date,
                    top_performers_data=top_performers_data,
                    feature_set=None
                )

                # Process and update models
                new_models = {}
                new_scalers = {}
                new_y_scalers = {}
                for result in retraining_results:
                    if result and result.get('status') == 'trained':
                        ticker = result['ticker']
                        new_models[ticker] = result['model']
                        new_scalers[ticker] = result['scaler']
                        if result.get('y_scaler'):
                            new_y_scalers[ticker] = result['y_scaler']

                # Update current models
                current_models.update(new_models)
                current_scalers.update(new_scalers)
                current_y_scalers.update(new_y_scalers)

                print(f"   ✅ Retrained models for {len(new_models)} stocks")
                
                # ✅ FIX: Check if training completely failed
                if len(new_models) == 0:
                    consecutive_training_failures += 1
                    print(f"   ⚠️ WARNING: No models successfully trained! ({consecutive_training_failures} consecutive failures)")
                    
                    if consecutive_training_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"\n❌ ABORT: Training has failed {consecutive_training_failures} times in a row!")
                        print(f"   💡 Possible reasons:")
                        print(f"      - Insufficient historical data for features")
                        print(f"      - Data quality issues (too many NaN values)")
                        print(f"      - Training period too short")
                        print(f"   🔧 Solutions:")
                        print(f"      - Increase TRAIN_LOOKBACK_DAYS in config")
                        print(f"      - Check data sources for quality")
                        print(f"      - Reduce number of tickers")
                        raise InsufficientDataError("Training consistently failing - aborting backtest")
                else:
                    consecutive_training_failures = 0  # Reset counter on success

            except InsufficientDataError:
                raise  # Re-raise to abort
            except Exception as e:
                consecutive_training_failures += 1
                print(f"   ⚠️ Retraining failed: {e}. Using existing models.")
                
                if consecutive_training_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n❌ ABORT: Training has failed {consecutive_training_failures} times in a row!")
                    raise InsufficientDataError(f"Training consistently failing: {e}")

        # Daily stock selection: Use current models to pick best 3 from 40 stocks
        try:
            predictions = []

            # Get predictions for all 40 stocks using current models
            valid_predictions = 0
            for ticker in initial_top_tickers:
                print(f"   🔍 Checking {ticker}: in models={ticker in current_models}, model not None={current_models.get(ticker) is not None if ticker in current_models else False}")
                if ticker in current_models and current_models[ticker] is not None:
                    try:
                        # Get data up to current date for prediction
                        ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
                        if not ticker_data.empty:
                            ticker_data = ticker_data.set_index('date')
                            data_slice = ticker_data.loc[:current_date]

                            if len(data_slice) >= 120:  # Need minimum 120 days for features
                                print(f"   📊 {ticker}: Calling prediction with {len(data_slice.tail(120))} rows, model={type(current_models[ticker]).__name__ if current_models[ticker] else None}, scaler={type(current_scalers.get(ticker)).__name__ if current_scalers.get(ticker) else None}")
                                pred = _quick_predict_return(
                                    ticker, data_slice.tail(120),  # Use last 120 days for features
                                    current_models[ticker],  # Single model
                                    current_scalers.get(ticker),
                                    current_y_scalers.get(ticker),
                                    horizon_days
                                )
                                print(f"   📊 {ticker}: Prediction result = {pred}")
                                # ✅ FIX 4: Only add valid predictions
                                if pred != -np.inf:
                                    predictions.append((ticker, pred))
                                    valid_predictions += 1
                            else:
                                print(f"   ⚠️ {ticker}: Only {len(data_slice)} rows available, need >=120 for feature engineering")
                                # ✅ FIX 4: Don't reference undefined 'pred' variable

                    except Exception as e:
                        continue

            # Debug: Show prediction summary
            if day_count == 1 or day_count % 10 == 0:
                print(f"   🔮 Day {day_count}: {valid_predictions} valid predictions from {len(initial_top_tickers)} tickers")

            # ✅ FIX: Check if no predictions are being made
            if valid_predictions == 0:
                consecutive_no_predictions += 1
                if consecutive_no_predictions >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n❌ ABORT: No valid predictions for {consecutive_no_predictions} consecutive days!")
                    print(f"   💡 Possible reasons:")
                    print(f"      - Models are None or not trained")
                    print(f"      - Insufficient data for prediction (need 120+ days)")
                    print(f"      - All predictions returning -inf")
                    print(f"   🔧 Solutions:")
                    print(f"      - Check model training logs above")
                    print(f"      - Verify data availability with diagnostics")
                    print(f"      - Increase data period")
                    raise InsufficientDataError("No predictions for multiple days - aborting backtest")
            else:
                consecutive_no_predictions = 0  # Reset on success
            
            # ✅ NEW: Store predictions with metadata
            day_predictions = {
                'date': current_date,
                'day': day_count,
                'predictions': [(t, p) for t, p in predictions]  # Store all predictions made
            }

            # Initialize selected_stocks variable
            selected_stocks = []

            # Select top 3 by predicted return
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                # Select top 3 stocks (or all if fewer available)
                num_to_select = min(3, len(predictions))
                selected_stocks = [ticker for ticker, _ in predictions[:num_to_select]]

                # Check if portfolio changed (only then incur transaction costs)
                if set(selected_stocks) != set(current_portfolio_stocks):
                    rebalance_count += 1
                    print(f"📊 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): New portfolio: {selected_stocks}")
                    old_portfolio = current_portfolio_stocks.copy()
                    current_portfolio_stocks = selected_stocks

                    # Execute actual trades for rebalancing
                    # Pass predictions for weighted allocation
                    selected_predictions = {t: p for t, p in predictions[:num_to_select]}
                    try:
                        executed_trades = _execute_portfolio_rebalance(
                            old_portfolio, selected_stocks, current_date, all_tickers_data,
                            positions, cash_balance, capital_per_stock, target_percentage,
                            predictions=selected_predictions  # ✅ NEW: Pass predictions for weighted buying
                        )

                        # Update cash balance after trades
                        cash_balance = executed_trades['cash_balance']

                        if old_portfolio:
                            print(f"   🔄 Rebalanced from {old_portfolio} to {selected_stocks}")
                            if executed_trades['sold_stocks']:
                                print(f"      💰 Sold: {', '.join(executed_trades['sold_stocks'])}")
                            if executed_trades['bought_stocks']:
                                print(f"      🛒 Bought: {', '.join(executed_trades['bought_stocks'])}")
                            print(f"      💸 Transaction costs: ${executed_trades['transaction_costs']:.2f}")
                        else:
                            print(f"   🆕 Initial portfolio: {selected_stocks}")
                            if executed_trades['bought_stocks']:
                                print(f"      🛒 Bought: {', '.join(executed_trades['bought_stocks'])}")
                            print(f"      💸 Transaction costs: ${executed_trades['transaction_costs']:.2f}")

                    except Exception as e:
                        print(f"   ⚠️ Rebalancing failed: {e}. Keeping current portfolio.")
            else:
                # No stocks selected - this might happen on early days
                if day_count == 1:
                    print(f"   ⚠️ Day {day_count}: No valid predictions - portfolio remains unallocated")
                elif day_count % 10 == 0:
                    print(f"   📊 Day {day_count}: No portfolio changes needed")

                    # Calculate actual portfolio value based on current positions
                    if selected_stocks:
                        # Calculate current portfolio value from positions
                        invested_value = 0.0
                        individual_returns = []

                        for ticker in selected_stocks:
                            if ticker in positions and positions[ticker]['shares'] > 0:
                                # Get current price
                                try:
                                    ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
                                    if not ticker_data.empty:
                                        ticker_data = ticker_data.set_index('date')
                                        current_price_data = ticker_data.loc[:current_date]
                                        if not current_price_data.empty:
                                            current_price = current_price_data['Close'].iloc[-1]
                                            position_value = positions[ticker]['shares'] * current_price
                                            invested_value += position_value

                                            # Calculate individual stock performance
                                            entry_price = positions[ticker]['avg_price']
                                            if entry_price > 0:
                                                stock_return_pct = (current_price / entry_price - 1) * 100
                                                individual_returns.append({
                                                    'ticker': ticker,
                                                    'total_return_pct': stock_return_pct,
                                                    'current_price': current_price,
                                                    'entry_price': entry_price,
                                                    'shares': positions[ticker]['shares'],
                                                    'value': position_value
                                                })
                                except Exception as e:
                                    print(f"   ⚠️ Could not calculate value for {ticker}: {e}")

                        # Portfolio value = invested value + cash balance
                        total_portfolio_value = invested_value + cash_balance

                        # Print portfolio status
                        print(f"   💼 Portfolio Status: ${total_portfolio_value:,.0f} total (${invested_value:,.0f} invested + ${cash_balance:,.0f} cash)")
                        print(f"   📊 Positions: {len([p for p in positions.values() if p['shares'] > 0])} stocks held")

                        # Print individual stock performance
                        if individual_returns:
                            print(f"   📋 Individual Stock Performance:")
                            for stock in individual_returns:
                                print(f"      • {stock['ticker']}: {stock['total_return_pct']:+.1f}% (${stock['current_price']:.2f} vs ${stock['entry_price']:.2f})")

                    # In real implementation: execute trades here
                    # For simulation: allocate capital to new stocks

                # Even if no rebalancing, portfolio value is updated at the end of the day loop

        except Exception as e:
            print(f"   ⚠️ Day {day_count}: Stock selection failed: {e}")
            # Keep existing portfolio if selection fails

        # Update portfolio value (invested + cash) at end of each day
        invested_value = 0.0
        for ticker in current_portfolio_stocks:
            if ticker in positions and positions[ticker]['shares'] > 0:
                # Get current price
                try:
                    ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
                    if not ticker_data.empty:
                        ticker_data = ticker_data.set_index('date')
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            current_price = current_price_data['Close'].iloc[-1]
                            position_value = positions[ticker]['shares'] * current_price
                            invested_value += position_value
                            # Update stored position value
                            positions[ticker]['value'] = position_value
                except Exception as e:
                    # Keep previous value if price lookup fails
                    invested_value += positions[ticker]['value']

        total_portfolio_value = invested_value + cash_balance

        # Update portfolio value history
        portfolio_values_history.append(total_portfolio_value)

        # ✅ NEW: Calculate actual returns vs predictions at end of each day
        if day_predictions['predictions'] and day_count > 1:
            # Calculate actual returns for the next prediction horizon (e.g., 20 days)
            future_date = current_date + timedelta(days=horizon_days)
            
            prediction_results = []
            for ticker, predicted_return in day_predictions['predictions']:
                try:
                    # Get current and future price
                    ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
                    if not ticker_data.empty:
                        ticker_data = ticker_data.set_index('date')
                        
                        # Current price
                        current_data = ticker_data.loc[:current_date]
                        if not current_data.empty:
                            current_price = current_data['Close'].iloc[-1]
                            
                            # Future price (if available in our data)
                            future_data = ticker_data.loc[:future_date]
                            if not future_data.empty and len(future_data) > len(current_data):
                                future_price = future_data['Close'].iloc[-1]
                                actual_return = (future_price / current_price - 1)
                                
                                # Calculate Buy & Hold return for comparison
                                bh_return = actual_return  # Same as actual
                                
                                prediction_results.append({
                                    'ticker': ticker,
                                    'predicted_return': predicted_return,
                                    'actual_return': actual_return,
                                    'bh_return': bh_return,
                                    'prediction_error': abs(predicted_return - actual_return),
                                    'current_price': current_price,
                                    'future_price': future_price
                                })
                except Exception:
                    continue
            
            # Store results
            if prediction_results:
                day_predictions['results'] = prediction_results
                daily_prediction_log.append(day_predictions)
                
                # Print daily comparison (every 10 days or when portfolio changes)
                if day_count % 10 == 0 or rebalance_count > 0:
                    print(f"\n   📊 Day {day_count} - AI Predictions vs Buy & Hold (Next {horizon_days} days):")
                    print(f"   {'Ticker':<8} {'AI Predicted':<14} {'Buy & Hold':<12} {'Error':<10} {'Direction':<10}")
                    print(f"   {'-'*65}")
                    for res in prediction_results[:5]:  # Show top 5
                        # Check if direction is correct
                        pred_up = res['predicted_return'] > 0
                        actual_up = res['bh_return'] > 0
                        direction = "✓" if pred_up == actual_up else "✗"
                        print(f"   {res['ticker']:<8} {res['predicted_return']:>12.2%}  {res['bh_return']:>10.2%}  "
                              f"{res['prediction_error']:>8.2%}  {direction:^10}")
        
        # Periodic progress update
        if day_count % 50 == 0:
            print(f"   📈 Processed {day_count}/{len(business_days)} days, portfolio: {current_portfolio_stocks}")

    print(f"\n🏁 Daily selection backtest complete!")
    print(f"   📊 Total days processed: {day_count}")
    print(f"   🧠 Model retrains: {retrain_count}")
    print(f"   🔄 Portfolio rebalances: {rebalance_count} (only when stocks change)")
    print(f"   💰 Transaction costs minimized - daily monitoring, trading only when portfolio changes")
    
    # ✅ NEW: Print prediction accuracy summary
    if daily_prediction_log:
        print(f"\n📈 PREDICTION ACCURACY SUMMARY")
        print(f"=" * 80)
        
        all_predictions = []
        for day_log in daily_prediction_log:
            if 'results' in day_log:
                all_predictions.extend(day_log['results'])
        
        if all_predictions:
            # Calculate statistics
            avg_predicted = np.mean([p['predicted_return'] for p in all_predictions])
            avg_bh = np.mean([p['bh_return'] for p in all_predictions])
            avg_error = np.mean([p['prediction_error'] for p in all_predictions])
            
            # Direction accuracy (did we predict up/down correctly?)
            correct_direction = sum(1 for p in all_predictions 
                                   if (p['predicted_return'] > 0 and p['bh_return'] > 0) or 
                                      (p['predicted_return'] < 0 and p['bh_return'] < 0))
            direction_accuracy = (correct_direction / len(all_predictions)) * 100
            
            print(f"Total Predictions Made: {len(all_predictions)}")
            print(f"Average AI Predicted Return: {avg_predicted:.2%}")
            print(f"Average Buy & Hold Return: {avg_bh:.2%}")
            print(f"Average Prediction Error: {avg_error:.2%}")
            print(f"Direction Accuracy: {direction_accuracy:.1f}% ({correct_direction}/{len(all_predictions)})")
            
            # Show best and worst predictions
            sorted_by_error = sorted(all_predictions, key=lambda x: x['prediction_error'])
            print(f"\n🎯 Best Predictions (lowest error):")
            for p in sorted_by_error[:3]:
                print(f"   {p['ticker']}: AI Predicted {p['predicted_return']:.2%}, "
                      f"B&H {p['bh_return']:.2%}, Error {p['prediction_error']:.2%}")
            
            print(f"\n❌ Worst Predictions (highest error):")
            for p in sorted_by_error[-3:]:
                print(f"   {p['ticker']}: AI Predicted {p['predicted_return']:.2%}, "
                      f"B&H {p['bh_return']:.2%}, Error {p['prediction_error']:.2%}")
            
            print(f"=" * 80)
        else:
            print(f"⚠️ No prediction results available (predictions made but actuals not yet known)")
    else:
        print(f"\n⚠️ WARNING: No predictions were logged during backtest!")

    return total_portfolio_value, portfolio_values_history, initial_top_tickers, [], {}


def _execute_portfolio_rebalance(old_portfolio, new_portfolio, current_date, all_tickers_data,
                               positions, cash_balance, capital_per_stock, target_percentage,
                               predictions=None):
    """
    Execute actual portfolio rebalancing by selling removed stocks and buying new ones.
    
    ✅ NEW: Uses prediction-weighted allocation for buying stocks.
    Stocks with higher predicted returns get larger allocations.

    Returns dict with trade execution details.
    """
    transaction_costs = 0.0
    sold_stocks = []
    bought_stocks = []

    # Get current prices for all stocks involved
    stocks_to_trade = set(old_portfolio + new_portfolio)
    current_prices = {}

    for ticker in stocks_to_trade:
        try:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
            if not ticker_data.empty:
                ticker_data = ticker_data.set_index('date')
                # Get the most recent price available up to current_date
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    current_prices[ticker] = price_data['Close'].iloc[-1]
        except Exception as e:
            print(f"   ⚠️ Could not get price for {ticker}: {e}")
            continue

    # Sell stocks that are no longer in portfolio
    stocks_to_sell = set(old_portfolio) - set(new_portfolio)
    for ticker in stocks_to_sell:
        if ticker in positions and positions[ticker]['shares'] > 0:
            if ticker in current_prices:
                sell_price = current_prices[ticker]
                shares_to_sell = positions[ticker]['shares']
                sell_value = shares_to_sell * sell_price

                # Apply transaction cost (simplified - percentage of trade value)
                cost = sell_value * TRANSACTION_COST
                net_sell_value = sell_value - cost
                transaction_costs += cost

                # Update cash and positions
                cash_balance += net_sell_value
                positions[ticker]['shares'] = 0
                positions[ticker]['value'] = 0

                sold_stocks.append(f"{ticker} ({shares_to_sell:.0f} shares @ ${sell_price:.2f})")
                print(f"      💰 Sold {ticker}: {shares_to_sell:.0f} shares @ ${sell_price:.2f} = ${sell_value:.2f} (-${cost:.2f} cost)")

    # Buy stocks that are newly added to portfolio
    stocks_to_buy = set(new_portfolio) - set(old_portfolio)
    if stocks_to_buy:
        # Calculate capital available for buying (cash + proceeds from sales)
        capital_for_new_stocks = cash_balance if cash_balance > 0 else capital_per_stock * len(stocks_to_buy)

        if capital_for_new_stocks > 0:
            # ✅ NEW: Calculate prediction-weighted allocations
            if predictions and len(predictions) > 0:
                # Get predictions for stocks we're buying
                buy_predictions = {t: predictions.get(t, 0) for t in stocks_to_buy}
                
                # Shift predictions to be positive (add offset if any are negative)
                min_pred = min(buy_predictions.values())
                if min_pred < 0:
                    # Add offset so all weights are positive
                    buy_predictions = {t: p - min_pred + 0.01 for t, p in buy_predictions.items()}
                
                # Calculate total for normalization
                total_pred = sum(buy_predictions.values())
                
                if total_pred > 0:
                    # Calculate weights (normalized to sum to 1)
                    weights = {t: p / total_pred for t, p in buy_predictions.items()}
                    print(f"      📊 Prediction-weighted allocation:")
                    for t, w in sorted(weights.items(), key=lambda x: -x[1]):
                        orig_pred = predictions.get(t, 0)
                        print(f"         {t}: {w*100:.1f}% (predicted: {orig_pred*100:.2f}%)")
                else:
                    # Fall back to equal weight
                    weights = {t: 1.0 / len(stocks_to_buy) for t in stocks_to_buy}
            else:
                # No predictions - use equal weight
                weights = {t: 1.0 / len(stocks_to_buy) for t in stocks_to_buy}

            for ticker in stocks_to_buy:
                if ticker in current_prices:
                    buy_price = current_prices[ticker]
                    if buy_price > 0:
                        # Calculate shares to buy based on weighted allocation
                        ticker_allocation = capital_for_new_stocks * weights.get(ticker, 1.0 / len(stocks_to_buy))
                        shares_to_buy = ticker_allocation / buy_price
                        buy_value = shares_to_buy * buy_price

                        # Apply transaction cost
                        cost = buy_value * TRANSACTION_COST
                        total_buy_cost = buy_value + cost
                        transaction_costs += cost

                        # Update cash and positions
                        cash_balance -= total_buy_cost

                        positions[ticker] = {
                            'shares': shares_to_buy,
                            'avg_price': buy_price,
                            'value': buy_value
                        }

                        weight_pct = weights.get(ticker, 0) * 100
                        bought_stocks.append(f"{ticker} ({shares_to_buy:.0f} shares @ ${buy_price:.2f})")
                        print(f"      🛒 Bought {ticker}: {shares_to_buy:.0f} shares @ ${buy_price:.2f} = ${buy_value:.2f} ({weight_pct:.1f}% weight, +${cost:.2f} cost)")

    # Handle stocks that remain in portfolio (no change needed)

    return {
        'cash_balance': cash_balance,
        'transaction_costs': transaction_costs,
        'sold_stocks': sold_stocks,
        'bought_stocks': bought_stocks,
        'positions': positions
    }


def _quick_predict_return(ticker: str, df_recent: pd.DataFrame, model, scaler, y_scaler, horizon_days: int) -> float:
    """Quick prediction of return for stock reselection during walk-forward backtest."""
    try:
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
            total_days = (df_with_features.index[-1] - df_with_features.index[0]).days

            if total_days > 0 and start_price > 0:
                total_return = (end_price / start_price) - 1.0
                annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
                df_with_features["Annualized_BH_Return"] = annualized_return
            else:
                df_with_features["Annualized_BH_Return"] = 0.0
        else:
            df_with_features["Annualized_BH_Return"] = 0.0

        df_with_features = df_with_features.dropna()
        print(f"   🔧 {ticker}: After dropna: {len(df_with_features)} rows")

        # ✅ VALIDATION: Check if enough rows remain after feature engineering
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
            features_scaled = scaler.transform(latest_data.values.reshape(1, -1))
            print(f"   🔧 {ticker}: Features scaled successfully, shape: {features_scaled.shape}")
        except Exception as e:
            print(f"   ❌ {ticker}: Scaling failed: {e}")
            return -np.inf

        # Predict return
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(features_scaled)[0]
                print(f"   🤖 {ticker}: Model.predict() successful: {float(prediction):.4f}")
            else:
                # Handle different model types
                prediction = model(latest_data.values.reshape(1, -1, -1) if hasattr(model, '__call__') else features_scaled)[0]
                print(f"   🤖 {ticker}: Model call successful: {float(prediction):.4f}")

            # Unscale if y_scaler exists
            if y_scaler and hasattr(y_scaler, 'inverse_transform'):
                prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
                print(f"   🔄 {ticker}: Y-scaler applied: {float(prediction):.4f}")

            print(f"   ✅ {ticker}: Final prediction = {float(prediction):.4f}")
            return float(prediction)

        except Exception as e:
            print(f"   ❌ {ticker}: Prediction failed: {e}")
            return -np.inf

    except Exception as e:
        print(f"   ⚠️ Prediction failed for {ticker}: {type(e).__name__}: {str(e)[:100]}")
        return -np.inf


def _run_portfolio_backtest_single_chunk(
    all_tickers_data: pd.DataFrame,
    chunk_start: datetime,
    chunk_end: datetime,
    current_top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    y_scalers: Dict,
    capital_allocation: float,
    target_percentage: float,
    period_name: str,
    top_performers_data: List[Tuple],
    horizon_days: int = 20
) -> Optional[Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]]:
    """Run a single chunk of the walk-forward backtest with pre-selected stocks."""
    num_processes = max(1, cpu_count() - 5)

    # For chunks, we skip the stock selection step and directly backtest the given stocks
    backtest_params = []
    processed_tickers = []
    performance_metrics = []
    buy_hold_histories_per_ticker = {}

    # Prepare backtest data for each pre-selected ticker
    for ticker in current_top_tickers:
        try:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                continue

            ticker_data = ticker_data.set_index('date')
            ticker_backtest_data = ticker_data.loc[chunk_start:chunk_end]

            if ticker_backtest_data.empty or len(ticker_backtest_data) < 5:
                continue

        except (KeyError, IndexError):
            continue

        # Prepare backtest parameters for this chunk
        model_buy_prepared = _prepare_model_for_multiprocessing(models_buy.get(ticker))
        model_sell_prepared = _prepare_model_for_multiprocessing(models_sell.get(ticker))

        backtest_params.append((
            ticker, ticker_backtest_data.copy(), capital_allocation,
            model_buy_prepared, model_sell_prepared, scalers.get(ticker), y_scalers.get(ticker),
            [], -1.0, 1.0, target_percentage,  # Buy immediately, sell never (hold strategy)
            top_performers_data, False, horizon_days
        ))

    # Run backtests for this chunk
    portfolio_values = []
    total_value = 0.0

    if backtest_params:
        if len(backtest_params) > 1 and num_processes > 1:
            # Parallel execution
            with Pool(processes=min(num_processes, len(backtest_params))) as pool:
                results = pool.imap(backtest_worker, backtest_params)
                for result in results:
                    if result:
                        ticker, final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation, buy_hold_history = result
                        processed_tickers.append(ticker)
                        portfolio_values.append(final_val)
                        buy_hold_histories_per_ticker[ticker] = buy_hold_history

                        perf_metrics = _calculate_performance_metrics(trade_log, buy_hold_history, final_val, capital_allocation)
                        perf_metrics.update({
                            'ticker': ticker,
                            'last_ai_action': last_ai_action,
                            'last_buy_prob': last_buy_prob,
                            'last_sell_prob': last_sell_prob,
                            'final_shares': shares_before_liquidation
                        })
                        performance_metrics.append(perf_metrics)
        else:
            # Sequential execution
            for params in backtest_params:
                result = backtest_worker(params)
                if result:
                    ticker, final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation, buy_hold_history = result
                    processed_tickers.append(ticker)
                    portfolio_values.append(final_val)
                    buy_hold_histories_per_ticker[ticker] = buy_hold_history

                    perf_metrics = _calculate_performance_metrics(trade_log, buy_hold_history, final_val, capital_allocation)
                    perf_metrics.update({
                        'ticker': ticker,
                        'last_ai_action': last_ai_action,
                        'last_buy_prob': last_buy_prob,
                        'last_sell_prob': last_sell_prob,
                        'final_shares': shares_before_liquidation
                    })
                    performance_metrics.append(perf_metrics)

        total_value = sum(portfolio_values) if portfolio_values else capital_allocation * len(current_top_tickers)

    return total_value, portfolio_values, processed_tickers, performance_metrics, buy_hold_histories_per_ticker

    backtest_params = []
    preview_predictions: List[Tuple[str, float]] = []

    def quick_last_prediction(ticker: str, df_slice: pd.DataFrame, model, scaler, y_scaler, feature_set, horizon_days: int):
        """
        Use the trained model to predict returns for ranking.
        Engineers features from raw OHLCV data first.
        """
        try:
            if model is None or scaler is None:
                # For simple rule strategy, allow neutral score so tickers are retained
                if use_simple_rule_strategy:
                    return 0.0
                return -np.inf
            
            # Check if we have required OHLCV data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_slice.columns for col in required_cols):
                return -np.inf
            
            # STEP 1: Engineer features from raw OHLCV data
            df_with_features = df_slice.copy()
            df_with_features = _calculate_technical_indicators(df_with_features)
            df_with_features = df_with_features.dropna()

            # If not enough rows remain after feature calc, bail out early
            if df_with_features.empty:
                return -np.inf

            # STEP 2: Prepare feature columns
            if feature_set:
                # Use only features that exist and match what model was trained on
                cols_to_use = [c for c in feature_set if c in df_with_features.columns]
            else:
                # Fallback: use all feature columns (exclude OHLCV and targets)
                cols_to_use = [c for c in df_with_features.columns if c not in 
                              ['Open', 'High', 'Low', 'Close', 'Volume', 'TargetReturnBuy', 'TargetReturnSell', 'Target']]
            
            if not cols_to_use:
                return -np.inf
            
            # STEP 3: Get model type and predict
            model_type = type(model).__name__
            
            if model_type in ['GRURegressor', 'GRUClassifier', 'LSTMClassifier']:
                if len(df_with_features) < SEQUENCE_LENGTH:
                    return -np.inf
                # GRU/LSTM: Use sequence (last SEQUENCE_LENGTH rows)
                df_feats = df_with_features[cols_to_use].tail(SEQUENCE_LENGTH).copy()
                for col in df_feats.columns:
                    df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(0.0)
                if df_feats.empty or len(df_feats) < SEQUENCE_LENGTH:
                    return -np.inf
                X_scaled = scaler.transform(df_feats)
                import torch
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
                    out = model(X_tensor)
                    pred_scaled = float(out.cpu().numpy()[0][0])
                    if y_scaler is not None:
                        pred = y_scaler.inverse_transform([[pred_scaled]])[0][0]
                    else:
                        pred = pred_scaled
                    return float(pred * 100)  # Return as percentage
            else:
                # XGBoost/RandomForest: Use last row only (no sequence length requirement)
                df_feats = df_with_features[cols_to_use].tail(1).copy()
                for col in df_feats.columns:
                    df_feats[col] = pd.to_numeric(df_feats[col], errors='coerce').fillna(0.0)
                if df_feats.empty:
                    return -np.inf
                X_scaled = scaler.transform(df_feats)
                # Preserve feature names when predicting
                X_scaled_df = pd.DataFrame(X_scaled, columns=df_feats.columns)
                pred = float(model.predict(X_scaled_df)[0])
                return float(pred * 100)  # Return as percentage
            
        except Exception as e:
            # Log the error for debugging
            import sys
            sys.stderr.write(f"  ⚠️ quick_last_prediction error for {ticker}: {e}\n")
            sys.stderr.flush()
            return -np.inf
    for ticker in top_tickers:
        # Use optimized parameters if available, otherwise fall back to global defaults
        # Probability thresholds removed - using simplified trading logic
        min_proba_buy_ticker = 0.0  # Disabled
        min_proba_sell_ticker = 1.0  # Disabled
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

        # Quick prediction for ranking (use buy model)
        preview_pred = quick_last_prediction(
            ticker,
            ticker_backtest_data,
            models_buy.get(ticker),
            scalers.get(ticker),
            y_scalers.get(ticker),
            feature_set_for_worker,
            horizon_days
        )
        preview_predictions.append((ticker, preview_pred))

        # Prepare PyTorch models for multiprocessing
        model_buy_prepared = _prepare_model_for_multiprocessing(models_buy.get(ticker))
        model_sell_prepared = _prepare_model_for_multiprocessing(models_sell.get(ticker))
        
        backtest_params.append((
            ticker, ticker_backtest_data.copy(), capital_per_stock,
            model_buy_prepared, model_sell_prepared, scalers.get(ticker), y_scalers.get(ticker),  # ✅ Added y_scaler
            feature_set_for_worker, min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data, use_simple_rule_strategy, horizon_days
        ))

    # Keep only top 3 tickers by AI-predicted return
    preview_predictions = sorted(preview_predictions, key=lambda x: x[1], reverse=True)
    allowed_tickers = set([t for t, _ in preview_predictions[:3]])
    # DEBUG: show all candidate predictions and the selected top 3
    print(f"\n[DEBUG] Candidate predicted returns for {period_name}:")
    for t, p in preview_predictions:
        print(f"  - {t}: {p:.4f}")
    print(f"[DEBUG] Selected top 3 for backtest: {list(allowed_tickers)}")

    backtest_params = [p for p in backtest_params if p[0] in allowed_tickers]
    top_tickers = [p[0] for p in backtest_params]

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
                    print(f"  - AI Sharpe Ratio: {res['perf_data']['sharpe_ratio']:.2f}")
                    print(f"  - Last AI Action: {res['last_ai_action']}")
                    
                    # ✅ FIX 3: Add diagnostic for 0 trades
                    total_trades = res.get('perf_data', {}).get('total_trades', 0)
                    if total_trades == 0 and res.get('last_ai_action') == 'HOLD':
                        buy_prob = res.get('buy_prob', 0.0)
                        sell_prob = res.get('sell_prob', 0.0)
                        print(f"\n  ⚠️  WARNING: {res['ticker']} made 0 trades!")
                        print(f"      Last Buy Probability: {buy_prob:.4f}")
                        print(f"      Last Sell Probability: {sell_prob:.4f}")
                    
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
                        break
                
                print(f"\n📈 Individual Stock Performance for {worker_result['ticker']} ({period_name}):")
                print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                print(f"  - AI Sharpe Ratio: {worker_result['perf_data']['sharpe_ratio']:.2f}")
                print(f"  - Last AI Action: {worker_result['last_ai_action']}")
                
                # ✅ FIX 3: Add diagnostic for 0 trades
                total_trades = worker_result.get('perf_data', {}).get('total_trades', 0)
                if total_trades == 0 and worker_result.get('last_ai_action') == 'HOLD':
                    buy_prob = worker_result.get('buy_prob', 0.0)
                    sell_prob = worker_result.get('sell_prob', 0.0)
                    print(f"\n  ⚠️  WARNING: {worker_result['ticker']} made 0 trades!")
                    print(f"      Last Buy Probability: {buy_prob:.4f}")
                    print(f"      Last Sell Probability: {sell_prob:.4f}")
                
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
        class_horiz = optimized_params.get('class_horizon', CLASS_HORIZON)
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


# Module for backtesting functions - not meant to be run directly
