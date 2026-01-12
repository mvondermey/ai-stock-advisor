"""
Unified Parallel Training System
Trains models at the model-type level instead of ticker level for better GPU utilization.
"""

import os
import sys
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from tqdm import tqdm

# Import config
from config import (
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, FORCE_CPU, XGBOOST_USE_GPU,
    GPU_MAX_CONCURRENT_TRAINING_WORKERS, TRAINING_NUM_PROCESSES,
    TRAINING_BATCH_SIZE,
    USE_LSTM, USE_GRU, USE_TCN, USE_XGBOOST, USE_RANDOM_FOREST, 
    USE_LIGHTGBM, USE_RIDGE, USE_ELASTIC_NET, USE_SVM, USE_MLP_CLASSIFIER
)

# Import utilities
from data_utils import fetch_training_data, _ensure_dir
from ml_models import train_single_model_type

# GPU semaphore for limiting concurrent GPU training
_GPU_SEMAPHORE = None

def _init_worker(gpu_semaphore):
    """Initialize worker process with GPU semaphore."""
    global _GPU_SEMAPHORE
    _GPU_SEMAPHORE = gpu_semaphore
    
    # Suppress warnings in worker processes
    warnings.filterwarnings("ignore")


def generate_training_tasks(
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    target_percentage: float,
    class_horizon: int,
    feature_set: Optional[List[str]] = None,
    include_ai_portfolio: bool = False,
    ai_portfolio_features: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate individual training tasks for each (ticker, model_type) combination.
    
    Returns:
        Tuple of (ticker_tasks, ai_portfolio_tasks)
    """
    
    # Get enabled model types from config
    enabled_models = []
    if USE_LSTM and PYTORCH_AVAILABLE:
        enabled_models.append('LSTM')
    if USE_GRU and PYTORCH_AVAILABLE:
        enabled_models.append('GRU')
    if USE_TCN and PYTORCH_AVAILABLE:
        enabled_models.append('TCN')
    if USE_XGBOOST:
        enabled_models.append('XGBoost')
    if USE_RANDOM_FOREST:
        enabled_models.append('RandomForest')
    if USE_LIGHTGBM:
        enabled_models.append('LightGBM')
    if USE_RIDGE:
        enabled_models.append('Ridge')
    if USE_ELASTIC_NET:
        enabled_models.append('ElasticNet')
    if USE_SVM:
        enabled_models.append('SVR')
    if USE_MLP_CLASSIFIER:
        enabled_models.append('MLPRegressor')
    
    if not enabled_models:
        print("⚠️ No models enabled in config!")
        return [], []
    
    print(f"📋 Generating tasks for {len(tickers)} tickers...")
    print(f"   Enabled models: {', '.join(enabled_models)}")
    print(f"   Data structure check:")
    print(f"     - Has 'date' column: {'date' in all_tickers_data.columns}")
    print(f"     - Has 'ticker' column: {'ticker' in all_tickers_data.columns}")
    print(f"     - Total rows in all_tickers_data: {len(all_tickers_data)}")
    print(f"     - Unique tickers in data: {all_tickers_data['ticker'].nunique() if 'ticker' in all_tickers_data.columns else 'N/A'}")
    print(f"     - Train period: {train_start} to {train_end}")
    
    # Validate data before processing
    if 'ticker' in all_tickers_data.columns:
        available_tickers = set(all_tickers_data['ticker'].unique())
        print(f"     - Available tickers in data: {len(available_tickers)}")
        missing_tickers = [t for t in tickers if t not in available_tickers]
        if missing_tickers:
            print(f"   ⚠️  WARNING: {len(missing_tickers)} tickers not found in data: {missing_tickers[:5]}{'...' if len(missing_tickers) > 5 else ''}")
    
    # Generate ticker model tasks
    ticker_tasks = []
    
    # ✅ Normalize dates for comparison (handle timezone-aware vs timezone-naive)
    # Convert to pandas Timestamps for proper comparison with DataFrame dates
    train_start_normalized = pd.Timestamp(train_start)
    train_end_normalized = pd.Timestamp(train_end)
    
    if 'date' in all_tickers_data.columns:
        # Check if dates in DataFrame are timezone-aware
        sample_date = all_tickers_data['date'].iloc[0] if len(all_tickers_data) > 0 else None
        if sample_date is not None:
            sample_tz = getattr(sample_date, 'tzinfo', None)
            
            if sample_tz is not None:
                # DataFrame has timezone-aware dates - localize train dates to same timezone
                if train_start_normalized.tzinfo is None:
                    train_start_normalized = train_start_normalized.tz_localize(sample_tz)
                else:
                    train_start_normalized = train_start_normalized.tz_convert(sample_tz)
                if train_end_normalized.tzinfo is None:
                    train_end_normalized = train_end_normalized.tz_localize(sample_tz)
                else:
                    train_end_normalized = train_end_normalized.tz_convert(sample_tz)
            else:
                # DataFrame has timezone-naive dates - ensure train dates are also naive
                if train_start_normalized.tzinfo is not None:
                    train_start_normalized = train_start_normalized.tz_localize(None)
                if train_end_normalized.tzinfo is not None:
                    train_end_normalized = train_end_normalized.tz_localize(None)
        
        print(f"   📅 Normalized date range for comparison:")
        print(f"      - Train start: {train_start_normalized} (tz: {train_start_normalized.tzinfo})")
        print(f"      - Train end: {train_end_normalized} (tz: {train_end_normalized.tzinfo})")
        if sample_date is not None:
            print(f"      - Sample data date: {sample_date} (tz: {getattr(sample_date, 'tzinfo', None)})")
            print(f"      - Data date range: {all_tickers_data['date'].min()} to {all_tickers_data['date'].max()}")
            # Debug: Check if dates overlap
            data_min = pd.Timestamp(all_tickers_data['date'].min())
            data_max = pd.Timestamp(all_tickers_data['date'].max())
            if train_end_normalized < data_min or train_start_normalized > data_max:
                print(f"      ⚠️ WARNING: Train period does NOT overlap with data range!")
                print(f"         Train: {train_start_normalized} to {train_end_normalized}")
                print(f"         Data:  {data_min} to {data_max}")
    
    # Debug: Track first few tickers for detailed diagnostics
    debug_ticker_count = 0
    max_debug_tickers = 3
    
    for ticker in tickers:
        try:
            # ✅ FIX: Handle both long-format and wide-format data
            if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
                # Long format: filter by ticker and date range
                ticker_mask = all_tickers_data['ticker'] == ticker
                ticker_rows_total = ticker_mask.sum()
                
                # Debug first few tickers
                if debug_ticker_count < max_debug_tickers:
                    ticker_data_all = all_tickers_data[ticker_mask]
                    if len(ticker_data_all) > 0:
                        ticker_date_min = ticker_data_all['date'].min()
                        ticker_date_max = ticker_data_all['date'].max()
                        print(f"   🔍 DEBUG {ticker}: {ticker_rows_total} total rows, dates: {ticker_date_min} to {ticker_date_max}")
                    else:
                        print(f"   🔍 DEBUG {ticker}: 0 rows in data")
                    debug_ticker_count += 1
                
                df_train_period = all_tickers_data[
                    ticker_mask &
                    (all_tickers_data['date'] >= train_start_normalized) &
                    (all_tickers_data['date'] <= train_end_normalized)
                ].copy()
            else:
                # Wide format: filter by ticker and index
                ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                if ticker_data.empty:
                    print(f"  ⚠️ No data for {ticker}, skipping")
                    continue
                
                # Filter by index if it's a DatetimeIndex
                if hasattr(ticker_data.index, 'tz_localize'):
                    df_train_period = ticker_data[
                        (ticker_data.index >= train_start_normalized) & (ticker_data.index <= train_end_normalized)
                    ].copy()
                else:
                    # Fallback: no date filtering
                    df_train_period = ticker_data.copy()
            
            if df_train_period.empty:
                if debug_ticker_count <= max_debug_tickers:
                    print(f"  ⚠️ No training data found for {ticker} in period {train_start_normalized} to {train_end_normalized}. Skipping.")
                continue
            
            if len(df_train_period) < 50:
                print(f"  ⚠️ Insufficient data for {ticker} ({len(df_train_period)} rows), skipping")
                continue
        except Exception as e:
            print(f"  ⚠️ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Create one task per model type for this ticker
        for model_type in enabled_models:
            task = {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'df_train_period': df_train_period,
                'target_percentage': target_percentage,
                'class_horizon': class_horizon,
                'feature_set': feature_set
            }
            ticker_tasks.append(task)
    
    print(f"   Generated {len(ticker_tasks)} ticker model tasks ({len(tickers)} tickers × {len(enabled_models)} models)")
    
    # Generate AI Portfolio tasks
    ai_portfolio_tasks = []
    if include_ai_portfolio and ai_portfolio_features is not None:
        X, y = ai_portfolio_features
        
        # AI Portfolio model types
        ai_models = ['RandomForest', 'XGBoost', 'LightGBM']
        if USE_RIDGE:
            ai_models.append('Ridge')
        
        print(f"   Generating {len(ai_models)} AI Portfolio model tasks")
        
        for model_type in ai_models:
            task = {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'X': X,
                'y': y
            }
            ai_portfolio_tasks.append(task)
    
    print(f"✅ Total tasks generated: {len(ticker_tasks) + len(ai_portfolio_tasks)}")
    
    return ticker_tasks, ai_portfolio_tasks


def safe_universal_model_worker(task):
    """
    Safe wrapper for universal_model_worker that prevents worker crashes.
    Catches all exceptions and returns error results instead of crashing.
    """
    try:
        return universal_model_worker(task)
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt to allow graceful shutdown
        raise
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb_str = traceback.format_exc()
        
        # Extract task info for error reporting
        ticker = task.get('ticker', 'unknown')
        model_type = task.get('model_type', 'unknown')
        task_type = task.get('task_type', 'unknown')
        
        print(f"  🛡️ SAFE WORKER: Caught exception for {ticker} {model_type}: {error_msg[:100]}")
        print(f"     Traceback: {tb_str[-300:]}")  # Last 300 chars of traceback
        
        # Return error result instead of crashing
        return {
            'task_type': task_type,
            'ticker': ticker,
            'model_type': model_type,
            'status': 'error',
            'reason': f'Worker protection: {error_msg[:200]}'
        }


def universal_model_worker(task: Dict) -> Dict:
    """
    Worker function that trains one model for one task.
    
    Args:
        task: Dict with task_type ('ticker' or 'ai_portfolio') and training parameters
    
    Returns:
        Dict with training results
    """
    global _GPU_SEMAPHORE
    
    # ✅ IMPLEMENT PER-TICKER TIMEOUT
    import signal
    import time
    from config import PER_TICKER_TIMEOUT
    
    def per_ticker_timeout_handler(signum, frame):
        raise TimeoutError(f"Per-ticker training timeout ({PER_TICKER_TIMEOUT}s)")
    
    # Set up per-ticker timeout if enabled
    timeout_set = False
    if PER_TICKER_TIMEOUT is not None and PER_TICKER_TIMEOUT > 0:
        timeout_set = True
        signal.signal(signal.SIGALRM, per_ticker_timeout_handler)
        signal.alarm(PER_TICKER_TIMEOUT)
    
    # ✅ TRACK PER-TICKER TIMING
    task_start_time = time.time()
    
    task_type = task.get('task_type')
    model_type = task.get('model_type')
    
    # ============================================
    # TICKER MODEL TRAINING
    # ============================================
    if task_type == 'ticker':
        ticker = task.get('ticker')
        df_train_period = task.get('df_train_period')
        target_percentage = task.get('target_percentage')
        class_horizon = task.get('class_horizon')
        feature_set = task.get('feature_set')
        
        try:
            # Acquire GPU semaphore if this is a GPU model
            gpu_acquired = False
            if not FORCE_CPU and model_type in ['LSTM', 'TCN', 'GRU', 'XGBoost']:
                if _GPU_SEMAPHORE is not None:
                    _GPU_SEMAPHORE.acquire()
                    gpu_acquired = True
            
            # Prepare training data
            df_train, actual_feature_set = fetch_training_data(
                ticker, df_train_period, target_percentage, class_horizon
            )
            
            if df_train.empty or len(df_train) < 50:
                return {
                    'task_type': 'ticker',
                    'ticker': ticker,
                    'model_type': model_type,
                    'status': 'skipped',
                    'reason': 'insufficient_data'
                }
            
            # Train the model
            result = train_single_model_type(
                df_train=df_train,
                model_type=model_type,
                ticker=ticker,
                feature_set=actual_feature_set
            )
            
            if result is None:
                return {
                    'task_type': 'ticker',
                    'ticker': ticker,
                    'model_type': model_type,
                    'status': 'failed',
                    'reason': 'training_failed'
                }
            
            # Save model to disk
            models_dir = Path("logs/models")
            _ensure_dir(models_dir)
            
            model_path = models_dir / f"{ticker}_{model_type}_temp_model.joblib"
            scaler_path = models_dir / f"{ticker}_{model_type}_temp_scaler.joblib"
            
            joblib.dump(result['model'], model_path)
            joblib.dump(result['scaler'], scaler_path)
            
            # Save y_scaler if present
            y_scaler_path = None
            if result.get('y_scaler') is not None:
                y_scaler_path = models_dir / f"{ticker}_{model_type}_temp_y_scaler.joblib"
                joblib.dump(result['y_scaler'], y_scaler_path)
            
            # ✅ CALCULATE TRAINING TIME
            task_end_time = time.time()
            training_time = task_end_time - task_start_time
            
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'success',
                'mse': result['mse'],
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'y_scaler_path': str(y_scaler_path) if y_scaler_path else None,
                'training_time': training_time
            }
        
        except TimeoutError as te:
            # ✅ HANDLE PER-TICKER TIMEOUT
            task_end_time = time.time()
            training_time = task_end_time - task_start_time
            print(f"  ⏰ TIMEOUT: {ticker} {model_type} ({PER_TICKER_TIMEOUT}s)")
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'timeout',
                'reason': f'Per-ticker timeout ({PER_TICKER_TIMEOUT}s)',
                'training_time': training_time
            }
        
        except Exception as e:
            import traceback
            task_end_time = time.time()
            training_time = task_end_time - task_start_time
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"  ❌ ERROR {ticker} {model_type}: {error_msg[:100]}")
            print(f"     Traceback: {tb_str[-500:]}")  # Last 500 chars of traceback
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'error',
                'reason': str(e)[:200],
                'training_time': training_time
            }
        
        finally:
            # ✅ CLEANUP: Cancel per-ticker timeout
            if timeout_set:
                signal.alarm(0)  # Cancel the alarm
            
            # Release GPU semaphore
            if gpu_acquired and _GPU_SEMAPHORE is not None:
                _GPU_SEMAPHORE.release()
    
    # ============================================
    # AI PORTFOLIO MODEL TRAINING
    # ============================================
    elif task_type == 'ai_portfolio':
        X = task.get('X')
        y = task.get('y')
        
        try:
            # Acquire GPU semaphore if XGBoost
            gpu_acquired = False
            if not FORCE_CPU and model_type == 'XGBoost' and XGBOOST_USE_GPU:
                if _GPU_SEMAPHORE is not None:
                    _GPU_SEMAPHORE.acquire()
                    gpu_acquired = True
            
            # Train AI Portfolio model
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, KFold
            from config import SEED
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize model
            model = None
            if model_type == 'RandomForest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=10, random_state=SEED, 
                    n_jobs=TRAINING_NUM_PROCESSES
                )
            elif model_type == 'XGBoost':
                import xgboost as xgb
                common_kwargs = {
                    "random_state": SEED,
                    "tree_method": "hist",
                    "nthread": TRAINING_NUM_PROCESSES,
                }
                if XGBOOST_USE_GPU and CUDA_AVAILABLE and not FORCE_CPU:
                    common_kwargs["device"] = "cuda"
                model = xgb.XGBClassifier(n_estimators=200, max_depth=7, **common_kwargs)
            
            elif model_type == 'LightGBM':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200, max_depth=7, random_state=SEED, 
                    verbosity=-1, n_jobs=TRAINING_NUM_PROCESSES
                )
            elif model_type == 'Ridge':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=TRAINING_NUM_PROCESSES)
            
            if model is None:
                return {
                    'task_type': 'ai_portfolio',
                    'model_type': model_type,
                    'status': 'failed',
                    'reason': 'model_not_available'
                }
            
            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=1)
            
            # Train on full data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X_scaled, y)
            
            train_score = model.score(X_scaled, y)
            
            # Save model to disk
            models_dir = Path("logs/models")
            _ensure_dir(models_dir)
            
            model_path = models_dir / f"ai_portfolio_{model_type}_temp_model.joblib"
            scaler_path = models_dir / f"ai_portfolio_{model_type}_temp_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            return {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'status': 'success',
                'cv_score': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_score': train_score,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path)
            }
        
        except TimeoutError as te:
            # ✅ HANDLE PER-TICKER TIMEOUT
            print(f"  ⏰ TIMEOUT: AI Portfolio {model_type} ({PER_TICKER_TIMEOUT}s)")
            return {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'status': 'timeout',
                'reason': f'Per-ticker timeout ({PER_TICKER_TIMEOUT}s)'
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'status': 'error',
                'reason': str(e)[:200]
            }
        
        finally:
            # ✅ CLEANUP: Cancel per-ticker timeout
            if timeout_set:
                signal.alarm(0)  # Cancel the alarm
            
            # Release GPU semaphore
            if gpu_acquired and _GPU_SEMAPHORE is not None:
                _GPU_SEMAPHORE.release()
    
    # ✅ HANDLE UNKNOWN TASK TYPE (outside try-except)
    if task_type not in ['ticker', 'ai_portfolio']:
        # Ensure cleanup on early return
        if timeout_set:
            signal.alarm(0)
        return {
            'task_type': task_type,
            'status': 'error',
            'reason': f'unknown_task_type: {task_type}'
        }


def aggregate_results(
    results: List[Dict],
    tickers: List[str]
) -> Tuple[Dict, Optional[Dict]]:
    """
    Aggregate training results and select best model per ticker.
    
    Args:
        results: List of training results from workers
        tickers: List of ticker symbols
    
    Returns:
        Tuple of (ticker_models_dict, ai_portfolio_model_dict)
    """
    
    print(f"\n📊 AGGREGATING TRAINING RESULTS")
    print("=" * 42)
    
    # Separate ticker and AI portfolio results
    ticker_results = [r for r in results if r.get('task_type') == 'ticker' and r.get('status') == 'success']
    ai_portfolio_results = [r for r in results if r.get('task_type') == 'ai_portfolio' and r.get('status') == 'success']
    
    # Count statuses
    success_count = len([r for r in results if r.get('status') == 'success'])
    skipped_count = len([r for r in results if r.get('status') == 'skipped'])
    timeout_count = len([r for r in results if r.get('status') == 'timeout'])
    error_count = len([r for r in results if r.get('status') in ['failed', 'error']])
    
    print(f"   ✅ Successful: {success_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   ⏰ Timeouts: {timeout_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # ✅ TIMING ANALYSIS
    timing_results = [r for r in results if 'training_time' in r and r.get('training_time') is not None]
    if timing_results:
        training_times = [r['training_time'] for r in timing_results]
        avg_time = sum(training_times) / len(training_times)
        min_time = min(training_times)
        max_time = max(training_times)
        median_time = sorted(training_times)[len(training_times) // 2]
        
        print(f"\n⏱️  TIMING ANALYSIS:")
        print(f"   📊 Average: {avg_time:.2f}s")
        print(f"   ⚡ Fastest: {min_time:.2f}s")
        print(f"   🐌 Slowest: {max_time:.2f}s")
        print(f"   📈 Median: {median_time:.2f}s")
        
        # Find slowest tickers
        slowest_results = sorted(timing_results, key=lambda x: x['training_time'], reverse=True)[:5]
        print(f"\n🐌 SLOWEST 5 TICKERS:")
        for i, result in enumerate(slowest_results, 1):
            ticker = result.get('ticker', 'unknown')
            model_type = result.get('model_type', 'unknown')
            time_taken = result.get('training_time', 0)
            status = result.get('status', 'unknown')
            print(f"   {i}. {ticker} {model_type}: {time_taken:.2f}s ({status})")
        
        # Find fastest tickers
        fastest_results = sorted(timing_results, key=lambda x: x['training_time'])[:5]
        print(f"\n⚡ FASTEST 5 TICKERS:")
        for i, result in enumerate(fastest_results, 1):
            ticker = result.get('ticker', 'unknown')
            model_type = result.get('model_type', 'unknown')
            time_taken = result.get('training_time', 0)
            status = result.get('status', 'unknown')
            print(f"   {i}. {ticker} {model_type}: {time_taken:.2f}s ({status})")
    
    # Log timeout details
    if timeout_count > 0:
        timeout_results = [r for r in results if r.get('status') == 'timeout']
        print(f"\n⏰ TIMEOUT DETAILS:")
        for result in timeout_results:
            ticker = result.get('ticker', 'unknown')
            model_type = result.get('model_type', 'unknown')
            reason = result.get('reason', 'unknown')
            print(f"   - {ticker} {model_type}: {reason}")
        
        # Save timeout log to file
        with open("logs/timeout_log.txt", "a") as f:
            f.write(f"\n=== TIMEOUT LOG ===\n")
            for result in timeout_results:
                ticker = result.get('ticker', 'unknown')
                model_type = result.get('model_type', 'unknown')
                reason = result.get('reason', 'unknown')
                f.write(f"{ticker},{model_type},{reason}\n")
        print(f"   📝 Timeout details saved to logs/timeout_log.txt")
    
    # ============================================
    # AGGREGATE TICKER MODELS
    # ============================================
    ticker_models = {}
    
    if ticker_results:
        print(f"\n📈 Aggregating ticker models for {len(tickers)} tickers...")
        print(f"   📋 Received {len(ticker_results)} successful ticker results")
        
        # Group results by ticker
        ticker_groups = {}
        for result in ticker_results:
            ticker = result['ticker']
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(result)
        
        print(f"   📋 Grouped into {len(ticker_groups)} unique tickers: {list(ticker_groups.keys())}")
        
        # Select best model per ticker (lowest MSE)
        for ticker, group in ticker_groups.items():
            if not group:
                continue
            
            # Find model with lowest MSE
            best_result = min(group, key=lambda x: x['mse'])
            
            # Load the winning model from disk
            try:
                model = joblib.load(best_result['model_path'])
                scaler = joblib.load(best_result['scaler_path'])
                y_scaler = None
                if best_result.get('y_scaler_path'):
                    y_scaler = joblib.load(best_result['y_scaler_path'])
                
                ticker_models[ticker] = {
                    'model': model,
                    'scaler': scaler,
                    'y_scaler': y_scaler,
                    'model_type': best_result['model_type'],
                    'mse': best_result['mse']
                }
                
                # Save with standard naming
                models_dir = Path("logs/models")
                final_model_path = models_dir / f"{ticker}_TargetReturn_model.joblib"
                final_scaler_path = models_dir / f"{ticker}_TargetReturn_scaler.joblib"
                
                joblib.dump(model, final_model_path)
                joblib.dump(scaler, final_scaler_path)
                
                if y_scaler is not None:
                    final_y_scaler_path = models_dir / f"{ticker}_TargetReturn_y_scaler.joblib"
                    joblib.dump(y_scaler, final_y_scaler_path)
                
                # Clean up temporary files
                for result in group:
                    try:
                        os.remove(result['model_path'])
                        os.remove(result['scaler_path'])
                        if result.get('y_scaler_path'):
                            os.remove(result['y_scaler_path'])
                    except:
                        pass
            
            except Exception as e:
                print(f"  ⚠️ Error loading model for {ticker}: {e}")
                continue
        
        print(f"   ✅ Successfully aggregated {len(ticker_models)} ticker models")
    
    # ============================================
    # AGGREGATE AI PORTFOLIO MODELS
    # ============================================
    ai_portfolio_model = None
    
    if ai_portfolio_results:
        print(f"\n🎯 Aggregating AI Portfolio models ({len(ai_portfolio_results)} candidates)...")
        
        # Find best model by CV score
        best_result = max(ai_portfolio_results, key=lambda x: x['cv_score'])
        
        try:
            # Load the winning model
            model = joblib.load(best_result['model_path'])
            scaler = joblib.load(best_result['scaler_path'])
            
            ai_portfolio_model = {
                'model': model,
                'scaler': scaler,
                'model_name': best_result['model_type'],
                'cv_score': best_result['cv_score'],
                'train_score': best_result['train_score']
            }
            
            print(f"   🏆 WINNER {best_result['model_type']}: CV={best_result['cv_score']:.4f} ± {best_result.get('cv_std', 0):.4f}, Train={best_result['train_score']:.4f}")
            
            # Save with standard naming
            models_dir = Path("logs/models")
            final_model_path = models_dir / "ai_portfolio_model.joblib"
            final_scaler_path = models_dir / "ai_portfolio_scaler.joblib"
            
            joblib.dump(model, final_model_path)
            joblib.dump(scaler, final_scaler_path)
            
            # Clean up temporary files
            for result in ai_portfolio_results:
                try:
                    os.remove(result['model_path'])
                    os.remove(result['scaler_path'])
                except:
                    pass
            
            print(f"   ✅ Selected: {best_result['model_type']}")
        
        except Exception as e:
            print(f"  ⚠️ Error loading AI Portfolio model: {e}")
            ai_portfolio_model = None
    
    print(f"\n✅ AGGREGATION COMPLETE")
    print("=" * 42)
    
    return ticker_models, ai_portfolio_model


def train_all_models_parallel(
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    target_percentage: float,
    class_horizon: int,
    feature_set: Optional[List[str]] = None,
    include_ai_portfolio: bool = False,
    ai_portfolio_features: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[Dict, Optional[Dict]]:
    """
    Main entry point for unified parallel training.
    Trains all models in parallel at the model-type level.
    
    Args:
        tickers: List of ticker symbols to train
        all_tickers_data: DataFrame with historical data for all tickers
        train_start: Training period start date
        train_end: Training period end date
        target_percentage: Target return percentage for classification
        class_horizon: Horizon in days for classification
        feature_set: Optional list of features to use
        include_ai_portfolio: Whether to train AI Portfolio models
        ai_portfolio_features: Optional (X, y) tuple for AI Portfolio training
    
    Returns:
        Tuple of (ticker_models_dict, ai_portfolio_model_dict)
        - ticker_models_dict: {ticker: {'model': ..., 'scaler': ..., 'y_scaler': ..., 'model_type': ...}}
        - ai_portfolio_model_dict: {'model': ..., 'scaler': ..., 'model_name': ..., 'cv_score': ...}
    """
    
    print(f"\n🚀 UNIFIED PARALLEL TRAINING SYSTEM")
    print("=" * 42)
    print(f"   Tickers: {len(tickers)}")
    print(f"   Period: {train_start.date()} to {train_end.date()}")
    print(f"   Workers: {TRAINING_NUM_PROCESSES}")
    print(f"   GPU Mode: {'Enabled' if (CUDA_AVAILABLE and not FORCE_CPU) else 'Disabled'}")
    print("=" * 42)
    
    # Generate training tasks
    ticker_tasks, ai_portfolio_tasks = generate_training_tasks(
        tickers=tickers,
        all_tickers_data=all_tickers_data,
        train_start=train_start,
        train_end=train_end,
        target_percentage=target_percentage,
        class_horizon=class_horizon,
        feature_set=feature_set,
        include_ai_portfolio=include_ai_portfolio,
        ai_portfolio_features=ai_portfolio_features
    )
    
    all_tasks = ticker_tasks + ai_portfolio_tasks
    
    if not all_tasks:
        print("⚠️ No tasks to execute!")
        return {}, None
    
    print(f"\n🏃 Executing {len(all_tasks)} training tasks in parallel...")
    
    # ✅ IMPORT TIME AT FUNCTION LEVEL (fixes scope issue)
    import time
    
    # Create GPU semaphore
    gpu_semaphore = None
    if CUDA_AVAILABLE and not FORCE_CPU:
        gpu_semaphore = mp.Semaphore(GPU_MAX_CONCURRENT_TRAINING_WORKERS)
    
    # Execute tasks in parallel
    start_time = time.time()
    
    try:
        # Use multiprocessing pool with enhanced error handling and timeout
        with mp.Pool(
            processes=TRAINING_NUM_PROCESSES,
            initializer=_init_worker,
            initargs=(gpu_semaphore,)
        ) as pool:
            # ✅ REMOVE GLOBAL SIGNAL TIMEOUT (conflicts with per-ticker timeouts)
            # The per-ticker timeouts in workers should handle individual task timeouts
            # Global timeout causes signal conflicts in WSL multiprocessing
            
            # Use imap_unordered for progress tracking with error recovery
            import sys
            
            # Process tasks in smaller batches to prevent cascade failures
            batch_size = min(TRAINING_BATCH_SIZE, len(all_tasks))  # Use configurable batch size
            all_results = []
            
            for i in range(0, len(all_tasks), batch_size):
                batch_tasks = all_tasks[i:i + batch_size]
                print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size} ({len(batch_tasks)} tasks)")
                
                try:
                    # ✅ ADD PROCESS-LEVEL TIMEOUT for stuck workers
                    # Use imap_unordered with timeout to handle completely stuck workers
                    batch_results = []
                    completed_tasks = 0
                    stuck_timeout = 60  # 60 seconds for entire batch if stuck
                    
                    with tqdm(total=len(batch_tasks), desc=f"Batch {i//batch_size + 1}", position=0, leave=True) as pbar:
                        # Use imap_unordered with timeout
                        iterator = pool.imap_unordered(safe_universal_model_worker, batch_tasks)
                        
                        batch_start_time = time.time()
                        last_progress_time = time.time()
                        
                        while completed_tasks < len(batch_tasks):
                            try:
                                # ✅ TIMEOUT CHECK for stuck batch
                                current_time = time.time()
                                if current_time - last_progress_time > stuck_timeout:
                                    print(f"  ⏰ BATCH TIMEOUT: No progress for {stuck_timeout}s - skipping remaining tasks")
                                    break
                                
                                # ✅ ADD TIMEOUT TO NEXT() CALL
                                # Use a timeout to prevent blocking when workers are stuck
                                import signal
                                
                                def next_timeout_handler(signum, frame):
                                    raise TimeoutError("next() timeout - worker may be stuck")
                                
                                # Set timeout for next() call
                                signal.signal(signal.SIGALRM, next_timeout_handler)
                                signal.alarm(30)  # 30 second timeout for next()
                                
                                try:
                                    result = next(iterator, None)
                                    signal.alarm(0)  # Cancel timeout if successful
                                    
                                    if result is None:
                                        break  # No more results
                                    
                                    batch_results.append(result)
                                    completed_tasks += 1
                                    pbar.update(1)
                                    last_progress_time = current_time  # Reset timeout on progress
                                    
                                except TimeoutError:
                                    print(f"  ⏰ NEXT() TIMEOUT: Worker stuck - skipping to next task")
                                    signal.alarm(0)  # Cancel timeout
                                    # Skip this stuck task and continue
                                    completed_tasks += 1  # Count as completed to avoid infinite loop
                                    pbar.update(1)
                                    last_progress_time = time.time()  # Reset timeout
                                    continue
                                
                            except StopIteration:
                                break  # Iterator exhausted
                            except Exception as e:
                                print(f"  ❌ Error getting result from iterator: {e}")
                                break
                    
                    all_results.extend(batch_results)
                    
                except Exception as batch_error:
                    print(f"❌ Batch {i//batch_size + 1} failed: {batch_error}")
                    # Add error results for this batch
                    for task in batch_tasks:
                        all_results.append({
                            'task_type': task.get('task_type', 'unknown'),
                            'ticker': task.get('ticker', 'unknown'),
                            'model_type': task.get('model_type', 'unknown'),
                            'status': 'error',
                            'reason': f'Batch failure: {str(batch_error)[:100]}'
                        })
                    continue
            
            results = all_results
                
            # ✅ NO GLOBAL SIGNAL CLEANUP NEEDED (removed global timeout)
                
    except Exception as e:
        print(f"❌ Error during parallel training: {e}")
        import traceback
        traceback.print_exc()
        return {}, None
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Total training time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"   Average time per task: {elapsed_time/len(all_tasks):.2f}s")
    
    # Aggregate results
    ticker_models, ai_portfolio_model = aggregate_results(results, tickers)
    
    return ticker_models, ai_portfolio_model
