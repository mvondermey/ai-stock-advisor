"""
Training Phase Module
Handles model training for 1-Year period.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import joblib
import pandas as pd
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, current_process
from tqdm import tqdm

from config import (
    PYTORCH_AVAILABLE, FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    PERIOD_HORIZONS, NUM_PROCESSES
)
from data_utils import fetch_training_data, _ensure_dir
from ml_models import initialize_ml_libraries, train_and_evaluate_models
from data_validation import validate_training_data, validate_features_after_engineering, InsufficientDataError

# Set multiprocessing start method to 'spawn' for CUDA safety
# This must be done before any Pool is created
try:
    if PYTORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            # Only set if not already set
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method('spawn', force=False)
                print("🔧 Set multiprocessing start method to 'spawn' for CUDA compatibility")
            elif multiprocessing.get_start_method() != 'spawn':
                print(f"⚠️  Multiprocessing start method is '{multiprocessing.get_start_method()}', but 'spawn' is recommended for CUDA")
except RuntimeError:
    # Already set, ignore
    pass

# Conditionally import LSTM/GRU classes if PyTorch is available
try:
    from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor  # ✅ Added LSTMRegressor
except ImportError:
    LSTMClassifier = None
    GRUClassifier = None
    LSTMRegressor = None


def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, class_horizon, feature_set = params
    
    # ✅ FIX: Reset CUDA state at start of each worker process to avoid context issues
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Reset CUDA context for this process
            torch.cuda.init()
    except Exception:
        pass  # Ignore if CUDA not available
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_path = models_dir / f"{ticker}_model.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_path = models_dir / f"{ticker}_TargetReturn_gru_optimized_params.json"

    model, scaler, y_scaler_loaded, loaded_gru_hyperparams = None, None, None, None

    # Flag to indicate if we successfully loaded a model to continue training
    loaded_for_retraining = False

    # Attempt to load models and GRU hyperparams if CONTINUE_TRAINING_FROM_EXISTING is True
    y_scaler_path = models_dir / f"{ticker}_y_scaler.joblib"

    if CONTINUE_TRAINING_FROM_EXISTING and model_path.exists() and scaler_path.exists():
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            if y_scaler_path.exists():
                y_scaler_loaded = joblib.load(y_scaler_path)

            if gru_hyperparams_path.exists():
                with open(gru_hyperparams_path, 'r') as f:
                    loaded_gru_hyperparams = json.load(f)

            print(f"  ✅ Loaded existing model and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

    # If FORCE_TRAINING is False and we didn't load for retraining, then we just load and skip training
    if not FORCE_TRAINING and not loaded_for_retraining and model_path.exists() and scaler_path.exists():
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            if y_scaler_path.exists():
                y_scaler_loaded = joblib.load(y_scaler_path)
            
            # Single regression model - no separate buy/sell hyperparams

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} (FORCE_TRAINING is False).")
            # Before returning, ensure PyTorch models are on CPU if they are deep learning models
            if PYTORCH_AVAILABLE:
                if LSTMRegressor is not None and GRURegressor is not None:
                    if isinstance(model, (LSTMRegressor, GRURegressor)):
                        model = model.cpu()
            return {
                'ticker': ticker,
                'model': model,
                'scaler': scaler,
                'y_scaler': y_scaler_loaded,
                'gru_hyperparams': loaded_gru_hyperparams,
                'status': 'loaded',
                'reason': None
            }
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker}: {e}. Training from scratch.")
            # Fall through to training from scratch if loading fails

    print(f"  ⚙️ Training models for {ticker} (FORCE_TRAINING is {FORCE_TRAINING}, CONTINUE_TRAINING_FROM_EXISTING is {CONTINUE_TRAINING_FROM_EXISTING})...")
    print(f"  [DEBUG] {current_process().name} - {ticker}: Initiating feature extraction for training.")
    
    df_train, actual_feature_set = fetch_training_data(ticker, df_train_period, target_percentage, class_horizon)

    if df_train.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data")
        print(f"     📊 DataFrame is empty after fetch_training_data()")
        print(f"     🔍 Check if ticker has sufficient historical data and feature generation worked")
        return {'ticker': ticker, 'model': None, 'scaler': None, 'y_scaler': None}

    # Train BUY model, passing the potentially loaded model and GRU hyperparams
    # Pass the global models_and_params to avoid re-initialization in worker processes
    global_models_and_params = initialize_ml_libraries() # Ensure it's initialized in the worker process too
    
    # Single regression model predicts expected returns
    # Always use regression targets (removed USE_REGRESSION_MODEL flag)
    target_column = "TargetReturn"

    train_result = train_and_evaluate_models(
        df_train, target_column, actual_feature_set, ticker=ticker,
        initial_model=model if loaded_for_retraining else None,  # Reuse existing model if available
        loaded_gru_hyperparams=loaded_gru_hyperparams,  # Reuse hyperparams
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=True,  # enable HP search
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )

    # Handle different return value formats for single regression model
    if train_result is None or len(train_result) == 3:
        # Enhanced diagnostics for failed training
        data_rows = len(df_train) if df_train is not None else 0
        target_exists = target_column in df_train.columns if df_train is not None else False
        target_non_null = df_train[target_column].notna().sum() if target_exists else 0
        features_available = len(actual_feature_set) if actual_feature_set else 0

        print(f"  ❌ {ticker} regression model training failed")
        print(f"     📊 Data available: {data_rows} rows")
        print(f"     🎯 Target '{target_column}': {'EXISTS' if target_exists else 'MISSING'} ({target_non_null} non-null values)")
        print(f"     🔧 Features: {features_available} available")
        print(f"     💡 Reason: {'Missing target column' if not target_exists else 'Insufficient training data'}")

        model, scaler, y_scaler, gru_hyperparams, winner = None, None, None, None, None
    else:
        model, scaler, y_scaler, gru_hyperparams, winner = train_result

    # Single model - no need to choose between buy/sell scalers
    final_scaler = scaler
    final_y_scaler = y_scaler

    if model and final_scaler:
        try:
            # Save single regression model
            joblib.dump(model, model_path)
            joblib.dump(final_scaler, scaler_path)

            # ✅ Save y_scaler if it exists
            if final_y_scaler is not None:
                y_scaler_path = models_dir / f"{ticker}_y_scaler.joblib"
                joblib.dump(final_y_scaler, y_scaler_path)

            if gru_hyperparams:
                with open(gru_hyperparams_path, 'w') as f:
                    json.dump(gru_hyperparams, f, indent=4)

            print(f"  ✅ Single regression model, scaler, y_scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving model or GRU hyperparams for {ticker}: {e}")

        # Before returning, ensure PyTorch models are on CPU if they are deep learning models
        if PYTORCH_AVAILABLE:
            if LSTMRegressor is not None and GRURegressor is not None:
                if isinstance(model, (LSTMRegressor, GRURegressor)):
                    model = model.cpu()

        return {
            'ticker': ticker,
            'model': model,  # Single regression model
            'scaler': final_scaler,
            'y_scaler': final_y_scaler,  # ✅ Return y_scaler
            'gru_hyperparams': gru_hyperparams,  # Single set of hyperparams
            'winner': winner,  # Which model type won
            'status': 'trained',
            'reason': None
        }
    else:
        reason = "Insufficient training data" # Default reason
        if df_train.empty:
            reason = f"Insufficient training data (initial rows: {len(df_train_period)})"
        elif len(df_train) < 50:
            reason = f"Not enough rows after feature prep ({len(df_train)} rows, need >= 50)"
        
        print(f"  ❌ Failed to train models for {ticker}. Reason: {reason}")
        return {'ticker': ticker, 'model': None, 'scaler': None, 'y_scaler': None, 'status': 'failed', 'reason': reason}


def train_models_for_period(
    period_name: str,
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    top_performers_data: List[Tuple[str, float, float]],
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True
) -> List[Dict]:
    """
    Train single regression models for 1-Year period.

    Args:
        period_name: Name of the period ("1-Year")
        tickers: List of ticker symbols to train
        all_tickers_data: DataFrame containing all ticker data
        train_start: Training start date
        train_end: Training end date
        top_performers_data: List of (ticker, 1y_perf, ytd_perf) tuples
        feature_set: List of feature names to use
        run_parallel: Whether to use parallel processing

    Returns:
        List of training result dictionaries with keys: 'status', 'ticker', 'model', 'scaler', 'y_scaler'
    """
    print(f"\n🔍 Step 3: Training AI models for {period_name} backtest...")
    
    models_buy = {}
    models_sell = {}
    scalers = {}
    
    # Calculate period-specific horizon
    # Use configured horizon for all periods
    period_horizon = PERIOD_HORIZONS.get(period_name, 60)
    
    # Prepare training parameters for each ticker
    training_params = []
    
    for ticker in tickers:
        # Load existing GRU hyperparameters if available
        models_dir = Path("logs/models")
        _ensure_dir(models_dir)
        
        gru_hyperparams_path = models_dir / f"{ticker}_{period_name}_TargetReturn_gru_optimized_params.json"
        
        # Single regression model - no separate buy/sell hyperparams
        
        # Single regression model - no separate buy/sell hyperparams to load
        
        # Helper to extract a close series (prefers 'Close', falls back to 'close')
        def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
            if 'Close' in df.columns:
                return df['Close']
            if 'close' in df.columns:
                return df['close']
            return None

        # Get Buy & Hold return for this ticker
        ticker_bh_return = 0.01  # Default 1% if not found
        for t, perf_1y in top_performers_data:
            if t == ticker:
                ticker_bh_return = perf_1y / 100.0
                break
        
        # Scale the B&H return to the training horizon so Target matches the lookahead
        days_in_period = max((train_end - train_start).days, 1)
        horizon_scale = min(1.0, period_horizon / days_in_period)
        period_target_pct = max(abs(ticker_bh_return) * horizon_scale, 0.01)  # keep at least 1%
        
        print(f"    📊 {ticker} {period_name[:2]} Training: Horizon={period_horizon}d, Target={period_target_pct:.2%} (B&H: {ticker_bh_return:.2%}, Scale={horizon_scale:.3f})")
        
        try:
            # ✅ FIX: Handle both long-format and wide-format data
            if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
                # Long format: filter by ticker and date range
                ticker_train_data = all_tickers_data[
                    (all_tickers_data['ticker'] == ticker) &
                    (all_tickers_data['date'] >= train_start) &
                    (all_tickers_data['date'] <= train_end)
                ].copy()
                
                if ticker_train_data.empty:
                    print(f"  ⚠️ No training data found for {ticker} in period {train_start} to {train_end}. Skipping.")
                    continue
                
                # Set date as index for training
                ticker_train_data = ticker_train_data.set_index('date')
                # Remove ticker column
                if 'ticker' in ticker_train_data.columns:
                    ticker_train_data = ticker_train_data.drop('ticker', axis=1)
            else:
                # Wide format: use original slicing logic
                ticker_train_data = all_tickers_data.loc[train_start:train_end, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
            
            # ✅ VALIDATION: Check if we have enough training data
            try:
                validate_training_data(ticker_train_data, ticker, train_start, train_end)
            except InsufficientDataError as e:
                print(f"  {str(e)}")
                continue
            
            training_params.append((
                ticker,
                ticker_train_data.copy(),
                period_target_pct,
                period_horizon,
                feature_set
            ))
        except (KeyError, IndexError) as e:
            print(f"  ⚠️ Could not slice training data for {ticker} for {period_name} period: {e}. Skipping.")
            continue
    
    # Run training in parallel or sequentially
    if run_parallel and len(training_params) > 0:
        print(f"🤖 Training {period_name} models in parallel for {len(tickers)} tickers using {NUM_PROCESSES} processes...")
        with Pool(processes=NUM_PROCESSES) as pool:
            training_results = list(tqdm(
                pool.imap(train_worker, training_params),
                total=len(training_params),
                desc=f"Training {period_name} Models"
            ))
    else:
        print(f"🤖 Training {period_name} models sequentially for {len(tickers)} tickers...")
        training_results = [train_worker(p) for p in tqdm(training_params, desc=f"Training {period_name} Models")]
    
    # Collect results
    y_scalers = {}  # ✅ Initialize y_scalers dictionary
    model_winners = {}  # ✅ Track model selection statistics
    result_list = []  # ✅ NEW: Collect results as list for backtesting compatibility
    
    for res in training_results:
        if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'):
            # Single model - use same model for both buy and sell logic
            models_buy[res['ticker']] = res['model']
            models_sell[res['ticker']] = res['model']  # Same model for both
            scalers[res['ticker']] = res['scaler']
            y_scalers[res['ticker']] = res.get('y_scaler', None)  # ✅ Collect y_scaler
            
            # ✅ NEW: Add to result list for backtest compatibility
            result_list.append(res)
            
            # Track winners for statistics
            if 'winner' in res:
                winner_key = f"{res['ticker']}_Regression"
                model_winners[winner_key] = res['winner']
    
    # Return separate dictionaries as expected by callers
    # Since we want only one model per stock, we use models_buy as the single model dict
    models = models_buy  # Single model per stock (previously models_buy)

    print(f"✅ {period_name} training complete: {len([t for t in tickers if t in models])} models trained/loaded.")

    # Print model selection statistics
    winner_list = [model_winners.get(f"{ticker}_Buy") for ticker in tickers if model_winners.get(f"{ticker}_Buy")]
    if winner_list:
        from collections import Counter
        winner_counts = Counter(winner_list)
        print(f"\n📊 Model Selection Statistics for {period_name}:")
        print(f"{'Model Name':<30} {'Times Selected':>15}")
        print("=" * 50)
        for model_name, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:<30} {count:>15}")
        print()

    # ✅ FIX: Return list of results for backtesting compatibility
    return result_list

