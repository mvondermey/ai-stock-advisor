"""
Training Phase Module
Handles model training for 1-Year period.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing import current_process
from tqdm import tqdm
import time

from config import (
    PYTORCH_AVAILABLE, FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    PERIOD_HORIZONS, NUM_PROCESSES, CUDA_AVAILABLE, GPU_MAX_CONCURRENT_TRAINING_WORKERS,
    GPU_PER_PROCESS_MEMORY_FRACTION,
    GPU_CLEAR_CACHE_ON_WORKER_INIT,
    GPU_CLEAR_CACHE_AFTER_EACH_TICKER,
    TRAINING_POOL_MAXTASKSPERCHILD,
    TRAINING_NUM_PROCESSES,
    USE_UNIFIED_PARALLEL_TRAINING
)
from data_utils import fetch_training_data, _ensure_dir
from ml_models import initialize_ml_libraries, train_and_evaluate_models, LSTMRegressor, GRURegressor
from data_validation import validate_training_data, validate_features_after_engineering, InsufficientDataError

# Use torch.multiprocessing for proper CUDA context handling
# Falls back to regular multiprocessing if torch not available
try:
    if PYTORCH_AVAILABLE:
        import torch
        import torch.multiprocessing as mp
        # Set start method to 'spawn' for CUDA safety
        try:
            mp.set_start_method('spawn', force=True)
            # Only print in main process
            if mp.current_process().name == 'MainProcess':
                print("🔧 Using torch.multiprocessing with 'spawn' for CUDA compatibility")
        except RuntimeError:
            pass  # Already set
        Pool = mp.Pool
    else:
        from multiprocessing import Pool
except (ImportError, RuntimeError):
    from multiprocessing import Pool

# Global semaphore for limiting concurrent GPU training across worker processes
_GPU_TRAIN_SEMAPHORE = None


def _init_pool_worker(gpu_semaphore):
    """Initializer for Pool workers to receive shared semaphore."""
    global _GPU_TRAIN_SEMAPHORE
    _GPU_TRAIN_SEMAPHORE = gpu_semaphore
    # Avoid nested thread explosions inside each spawned worker (common WSL stability issue)
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    except Exception:
        pass
    try:
        if PYTORCH_AVAILABLE:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
    except Exception:
        pass


def _acquire_gpu_slot(ticker: str):
    """Acquire a GPU training slot if CUDA is enabled and semaphore is available."""
    global _GPU_TRAIN_SEMAPHORE
    from config import FORCE_CPU
    # Skip GPU slot management if FORCE_CPU is enabled
    if FORCE_CPU:
        return
    # If single-process training, don't gate GPU usage
    if NUM_PROCESSES <= 1:
        return
    if not CUDA_AVAILABLE or _GPU_TRAIN_SEMAPHORE is None:
        return
    print(f"🐛 DEBUG: {ticker} - Waiting for GPU slot ({GPU_MAX_CONCURRENT_TRAINING_WORKERS} max)...", flush=True)
    _GPU_TRAIN_SEMAPHORE.acquire()
    print(f"🐛 DEBUG: {ticker} - Acquired GPU slot ✅", flush=True)


def _release_gpu_slot(ticker: str):
    global _GPU_TRAIN_SEMAPHORE
    from config import FORCE_CPU
    # Skip GPU slot management if FORCE_CPU is enabled
    if FORCE_CPU:
        return
    if NUM_PROCESSES <= 1:
        return
    if not CUDA_AVAILABLE or _GPU_TRAIN_SEMAPHORE is None:
        return
    try:
        _GPU_TRAIN_SEMAPHORE.release()
        print(f"🐛 DEBUG: {ticker} - Released GPU slot", flush=True)
    except Exception:
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
    ticker, df_train_period, class_horizon, feature_set = params
    
    import sys
    import os
    
    # ✅ Use GPU with proper CUDA context isolation per worker
    print(f"🐛 DEBUG: train_worker started for {ticker}", flush=True)
    sys.stdout.flush()
    
    # Initialize CUDA for this worker process with memory limit
    try:
        import torch
        from config import FORCE_CPU
        # Skip GPU setup if FORCE_CPU is enabled
        if not FORCE_CPU and torch.cuda.is_available():
            torch.cuda.set_device(0)
            # Optional: cap VRAM per process to reduce OOM risk under multiprocessing.
            # Prefer deriving the cap from the *actual GPU concurrency* (GPU_MAX_CONCURRENT_TRAINING_WORKERS),
            # since NUM_PROCESSES can be >1 even when only 1 GPU worker runs at a time.
            try:
                effective_fraction = None

                # Hard override if explicitly configured
                if GPU_PER_PROCESS_MEMORY_FRACTION is not None:
                    effective_fraction = float(GPU_PER_PROCESS_MEMORY_FRACTION)
                else:
                    # Auto-cap only when we can have >1 concurrent GPU trainers
                    if CUDA_AVAILABLE and GPU_MAX_CONCURRENT_TRAINING_WORKERS and GPU_MAX_CONCURRENT_TRAINING_WORKERS > 1:
                        # Leave a little headroom to reduce fragmentation/OOM thrash
                        effective_fraction = min(0.95, 0.95 / float(GPU_MAX_CONCURRENT_TRAINING_WORKERS))

                if effective_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(effective_fraction, device=0)
            except Exception:
                pass
            if GPU_CLEAR_CACHE_ON_WORKER_INIT:
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ {ticker} - GPU init warning: {e}", flush=True)
    sys.stdout.flush()
    
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
            # Handle PyTorch models specially
            if PYTORCH_AVAILABLE and model_path.with_suffix('.info').exists():
                model_info = joblib.load(model_path.with_suffix('.info'))
                if model_info.get('model_class'):
                    # Reconstruct PyTorch model
                    import torch
                    from ml_models import TCNRegressor, GRURegressor, LSTMRegressor

                    model_class_name = model_info['model_class']
                    if model_class_name == 'TCNRegressor':
                        model = TCNRegressor(
                            input_size=model_info.get('input_size', 35),
                            num_filters=32, kernel_size=3, num_levels=2, dropout=0.1
                        )
                    elif model_class_name == 'GRURegressor':
                        model = GRURegressor(
                            input_size=model_info.get('input_size', 35),
                            hidden_size=model_info.get('hidden_size', 64),
                            num_layers=model_info.get('num_layers', 2),
                            output_size=1, dropout_rate=0.5
                        )
                    elif model_class_name == 'LSTMRegressor':
                        model = LSTMRegressor(
                            input_size=model_info.get('input_size', 35),
                            hidden_size=model_info.get('hidden_size', 64),
                            num_layers=model_info.get('num_layers', 2),
                            output_size=1, dropout_rate=0.5
                        )

                    if model:
                        state_dict = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(state_dict)
                        print(f"  ✅ Loaded PyTorch model {model_class_name} from state_dict for {ticker}")
                else:
                    model = joblib.load(model_path)
            else:
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
                try:
                    from ml_models import LSTMRegressor, GRURegressor, TCNRegressor
                    if isinstance(model, (LSTMRegressor, GRURegressor, TCNRegressor)):
                        model = model.cpu()
                except (ImportError, NameError):
                    # If imports fail in worker process, skip CPU conversion
                    pass
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
    
    df_train, actual_feature_set = fetch_training_data(ticker, df_train_period, class_horizon)

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

    # Limit concurrent GPU-heavy training to avoid WSL CUDA deadlocks
    _acquire_gpu_slot(ticker)
    try:
        print(f"🎯 {ticker}: Starting model training (LSTM→TCN→ML models)...", flush=True)
        sys.stdout.flush()
        train_result = train_and_evaluate_models(
            df_train, target_column, actual_feature_set, ticker=ticker,
            initial_model=model if loaded_for_retraining else None,  # Reuse existing model if available
            loaded_gru_hyperparams=loaded_gru_hyperparams,  # Reuse hyperparams
            models_and_params_global=global_models_and_params,
            perform_gru_hp_optimization=True,  # enable HP search
            default_class_horizon=class_horizon # Pass current class_horizon
        )
    finally:
        _release_gpu_slot(ticker)
    print(f"🐛 DEBUG: {ticker} - train_and_evaluate_models completed", flush=True)
    sys.stdout.flush()

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
            # Handle PyTorch models specially - save state_dict instead of full model
            if PYTORCH_AVAILABLE and hasattr(model, 'state_dict'):
                import torch
                torch.save(model.state_dict(), model_path)
                # Also save model class info for reconstruction
                # Save model architecture parameters based on model type
                model_class_name = model.__class__.__name__
                model_info = {
                    'state_dict_path': str(model_path),
                    'model_class': model_class_name,
                }

                if model_class_name == 'TCNRegressor':
                    # For TCN, we need to know the input_size
                    # We can infer it from the first conv layer's input channels
                    if hasattr(model, 'net') and len(model.net) > 0:
                        first_conv = model.net[0]  # First Conv1d layer
                        if hasattr(first_conv, 'in_channels'):
                            model_info['input_size'] = first_conv.in_channels
                    else:
                        model_info['input_size'] = 35  # Default fallback
                elif model_class_name in ['GRURegressor', 'LSTMRegressor', 'LSTMClassifier', 'GRUClassifier']:
                    # For RNN models, get parameters from the rnn layer
                    rnn_layer = getattr(model, 'gru', None) or getattr(model, 'lstm', None)
                    if rnn_layer:
                        model_info['input_size'] = rnn_layer.input_size
                        model_info['hidden_size'] = rnn_layer.hidden_size
                        model_info['num_layers'] = rnn_layer.num_layers
                joblib.dump(model_info, model_path.with_suffix('.info'))
            else:
                # Regular scikit-learn models
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
            sys.stdout.flush()
        except Exception as e:
            print(f"  ⚠️ Error saving model or GRU hyperparams for {ticker}: {e}")
            sys.stdout.flush()

        # Before returning, ensure PyTorch models are on CPU if they are deep learning models
        print(f"🐛 DEBUG: {ticker} - Moving model to CPU before return...", flush=True)
        if PYTORCH_AVAILABLE:
            try:
                from ml_models import LSTMRegressor, GRURegressor, TCNRegressor
                if isinstance(model, (LSTMRegressor, GRURegressor, TCNRegressor)):
                    model = model.cpu()
            except (ImportError, NameError):
                # If imports fail in worker process, skip CPU conversion
                pass

        # ✅ Optional GPU cache clear before returning (can cause brief utilization dips)
        if CUDA_AVAILABLE and GPU_CLEAR_CACHE_AFTER_EACH_TICKER:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"🐛 DEBUG: {current_process().name} - {ticker}: GPU cache cleared", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Failed to clear GPU cache for {ticker}: {e}", flush=True)

        # ✅ Prepare y_scaler_path before deleting objects
        y_scaler_path_str = str(y_scaler_path) if final_y_scaler is not None else None
        
        # ✅ Explicitly delete large objects before returning (critical for multiprocessing performance)
        del model
        del final_scaler
        if final_y_scaler is not None:
            del final_y_scaler
        
        # ✅ Return only metadata (models already saved to disk) to avoid pickling overhead
        result = {
            'ticker': ticker,
            'model': None,  # Don't return model object - load from disk later
            'scaler': None,  # Don't return scaler - load from disk later
            'y_scaler': None,  # Don't return y_scaler - load from disk later
            'gru_hyperparams': gru_hyperparams,  # Small JSON-serializable dict
            'winner': winner,  # Which model type won
            'status': 'trained',
            'reason': None,
            'model_path': str(model_path),  # Path to load model from
            'scaler_path': str(scaler_path),  # Path to load scaler from
            'y_scaler_path': y_scaler_path_str
        }
        from datetime import datetime
        print(f"🐛 DEBUG [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]: {ticker} - Returning result metadata...", flush=True)
        sys.stdout.flush()
        return result
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
    
    # ============================================
    # NEW: Use Unified Parallel Training if enabled
    # ============================================
    if USE_UNIFIED_PARALLEL_TRAINING:
        print(f"   🚀 Using Unified Parallel Training System (model-level parallelization)")
        try:
            from parallel_training import train_all_models_parallel
        except ModuleNotFoundError:
            # Fallback for different import contexts
            from src.parallel_training import train_all_models_parallel
        
        # Use TRAIN_LOOKBACK_DAYS for training horizon (not PERIOD_HORIZONS which is for prediction)
        period_horizon = TRAIN_LOOKBACK_DAYS
        
        # Get Buy & Hold returns for target percentage calculation
        ticker_bh_returns = {}
        for t, perf_1y in top_performers_data:
            ticker_bh_returns[t] = perf_1y / 100.0
        
        # Calculate average target percentage
        days_in_period = max((train_end - train_start).days, 1)
        horizon_scale = min(1.0, period_horizon / days_in_period)
        
        # Use median B&H return for target percentage
        if ticker_bh_returns:
            median_bh = np.median(list(ticker_bh_returns.values()))
            period_target_pct = max(abs(median_bh) * horizon_scale, 0.01)
        else:
            period_target_pct = 0.01
        
        print(f"   📊 Training Horizon: {period_horizon} days, Target: {period_target_pct:.2%}")
        
        # Train all models in parallel (ticker models only, no AI Portfolio here)
        ticker_models, _ = train_all_models_parallel(
            tickers=tickers,
            all_tickers_data=all_tickers_data,
            train_start=train_start,
            train_end=train_end,
            class_horizon=period_horizon,
            feature_set=feature_set,
            include_ai_portfolio=False,  # AI Portfolio trained separately
            ai_portfolio_features=None
        )
        
        # Convert to return format expected by calling code
        # ✅ Return None for models to avoid GPU/CPU memory issues - load from disk instead
        training_results = []
        for ticker, model_dict in ticker_models.items():
            training_results.append({
                'ticker': ticker,
                'model': None,  # Don't pass models directly - load from disk
                'scaler': None,  # Don't pass scalers directly - load from disk
                'y_scaler': None,  # Don't pass y_scalers directly - load from disk
                'gru_hyperparams': None,  # Not used in unified system
                'winner': model_dict['model_type'],
                'status': 'trained',
                'reason': None
            })
        
        # Add failed tickers
        failed_tickers = set(tickers) - set(ticker_models.keys())
        for ticker in failed_tickers:
            training_results.append({
                'ticker': ticker,
                'model': None,
                'scaler': None,
                'y_scaler': None,
                'gru_hyperparams': None,
                'winner': None,
                'status': 'failed',
                'reason': 'Training failed or insufficient data'
            })
        
        return training_results
