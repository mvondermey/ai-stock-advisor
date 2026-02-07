"""
Unified Parallel Training System
Trains models at the model-type level instead of ticker level for better GPU utilization.
"""

import os
import sys
import time
import tempfile
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from tqdm import tqdm

# Import config
from config import (
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, FORCE_CPU, XGBOOST_USE_GPU,
    GPU_MAX_CONCURRENT_TRAINING_WORKERS, TRAINING_NUM_PROCESSES,
    TRAINING_BATCH_SIZE, PER_TICKER_TIMEOUT,
    USE_LSTM, USE_GRU, USE_TCN, USE_XGBOOST, USE_RANDOM_FOREST, 
    USE_LIGHTGBM, USE_RIDGE, USE_ELASTIC_NET, USE_SVM, USE_MLP_CLASSIFIER
)

# Import utilities
from data_utils import fetch_training_data, _ensure_dir
from ml_models import train_single_model_type

# GPU semaphore for limiting concurrent GPU training
_GPU_SEMAPHORE = None
# Path to temp directory containing per-ticker data files
_SHARED_DATA_DIR = None

def _init_worker(gpu_semaphore, shared_data_dir=None):
    """Initialize worker process with GPU semaphore and path to shared data directory."""
    global _GPU_SEMAPHORE, _SHARED_DATA_DIR
    _GPU_SEMAPHORE = gpu_semaphore
    _SHARED_DATA_DIR = shared_data_dir
    
    # Suppress warnings in worker processes
    warnings.filterwarnings("ignore")

def _load_ticker_data(ticker):
    """Load data for a specific ticker from its temp file (lightweight, ~125KB each)."""
    if _SHARED_DATA_DIR is None:
        return None
    # Sanitize ticker name for filename (replace dots/slashes)
    safe_name = ticker.replace('/', '_').replace('\\', '_').replace('.', '_')
    ticker_file = os.path.join(_SHARED_DATA_DIR, f'{safe_name}.pkl')
    if os.path.exists(ticker_file):
        return joblib.load(ticker_file)
    return None


def generate_training_tasks(
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    class_horizon: int,
    feature_set: Optional[List[str]] = None,
    include_ai_portfolio: bool = False,
    ai_portfolio_features: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Generate individual training tasks for each (ticker, model_type) combination.
    
    Returns:
        Tuple of (ticker_tasks, ai_portfolio_tasks, ticker_data_dict)
        ticker_data_dict maps ticker -> DataFrame, shared with workers to avoid pickling.
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
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)
    
    if 'date' in all_tickers_data.columns:
        # Check if dates in DataFrame are timezone-aware
        sample_date = all_tickers_data['date'].iloc[0] if len(all_tickers_data) > 0 else None
        if sample_date is not None:
            sample_tz = getattr(sample_date, 'tzinfo', None)
            
            if sample_tz is not None:
                # DataFrame has timezone-aware dates - localize train dates to same timezone
                if train_start.tzinfo is None:
                    train_start = train_start.tz_localize(sample_tz)
                else:
                    train_start = train_start.tz_convert(sample_tz)
                if train_end.tzinfo is None:
                    train_end = train_end.tz_localize(sample_tz)
                else:
                    train_end = train_end.tz_convert(sample_tz)
            else:
                # DataFrame has timezone-naive dates - ensure train dates are also naive
                if train_start.tzinfo is not None:
                    train_start = train_start.tz_localize(None)
                if train_end.tzinfo is not None:
                    train_end = train_end.tz_localize(None)
        
        print(f"   📅 Normalized date range for comparison:")
        print(f"      - Train start: {train_start} (tz: {train_start.tzinfo})")
        print(f"      - Train end: {train_end} (tz: {train_end.tzinfo})")
        if sample_date is not None:
            print(f"      - Sample data date: {sample_date} (tz: {getattr(sample_date, 'tzinfo', None)})")
            print(f"      - Data date range: {all_tickers_data['date'].min()} to {all_tickers_data['date'].max()}")
            # Debug: Check if dates overlap
            data_min = pd.Timestamp(all_tickers_data['date'].min())
            data_max = pd.Timestamp(all_tickers_data['date'].max())
            if train_end < data_min or train_start > data_max:
                print(f"      ⚠️ WARNING: Train period does NOT overlap with data range!")
                print(f"         Train: {train_start} to {train_end}")
                print(f"         Data:  {data_min} to {data_max}")
    
    # Pre-split data by ticker into a dict for shared access (avoids pickling per task)
    ticker_data_dict = {}
    skipped_tickers = []
    
    if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
        # Long format: group by ticker once
        grouped = all_tickers_data.groupby('ticker')
        for ticker in tickers:
            try:
                if ticker not in grouped.groups:
                    skipped_tickers.append((ticker, 'not_in_data'))
                    continue
                df_ticker = grouped.get_group(ticker).copy()
                if len(df_ticker) < 50:
                    skipped_tickers.append((ticker, f'insufficient_{len(df_ticker)}'))
                    continue
                ticker_data_dict[ticker] = df_ticker
            except Exception as e:
                skipped_tickers.append((ticker, str(e)[:50]))
                continue
    else:
        for ticker in tickers:
            try:
                df_ticker = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                if df_ticker.empty or len(df_ticker) < 50:
                    skipped_tickers.append((ticker, f'insufficient_{len(df_ticker)}'))
                    continue
                ticker_data_dict[ticker] = df_ticker
            except Exception as e:
                skipped_tickers.append((ticker, str(e)[:50]))
                continue
    
    if skipped_tickers:
        print(f"   ⚠️ Skipped {len(skipped_tickers)} tickers (insufficient data or missing)")
    print(f"   ✅ {len(ticker_data_dict)} tickers have sufficient data")
    
    # Debug: Show first 3 tickers
    for i, (ticker, df) in enumerate(ticker_data_dict.items()):
        if i >= 3:
            break
        print(f"   🔍 DEBUG {ticker}: {len(df)} rows, dates: {df['date'].min()} to {df['date'].max()}")
    
    # Create lightweight tasks (NO DataFrame in task - workers use shared data)
    for ticker in ticker_data_dict.keys():
        for model_type in enabled_models:
            task = {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'class_horizon': class_horizon,
                'train_start': train_start,
                'train_end': train_end,
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
    
    return ticker_tasks, ai_portfolio_tasks, ticker_data_dict


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
    
    # NOTE: signal.alarm() does NOT work in multiprocessing worker processes
    # Timeout enforcement must happen at the pool level (imap_unordered with timeout)
    # We just track timing here for logging
    import time
    task_start_time = time.time()
    
    task_type = task.get('task_type')
    model_type = task.get('model_type')
    
    # ============================================
    # TICKER MODEL TRAINING
    # ============================================
    if task_type == 'ticker':
        ticker = task.get('ticker')
        class_horizon = task.get('class_horizon')
        train_start = task.get('train_start')
        train_end = task.get('train_end')
        feature_set = task.get('feature_set')
        
        # Load data for this specific ticker from temp file (~125KB, not 98MB)
        df_train_period = _load_ticker_data(ticker)
        if df_train_period is None:
            df_train_period = task.get('df_train_period')
        
        if df_train_period is None or len(df_train_period) == 0:
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'skipped',
                'reason': 'no_data_in_shared_dict'
            }
        
        try:
            # Acquire GPU semaphore if this is a GPU model
            gpu_acquired = False
            if not FORCE_CPU and model_type in ['LSTM', 'TCN', 'GRU', 'XGBoost']:
                if _GPU_SEMAPHORE is not None:
                    _GPU_SEMAPHORE.acquire()
                    gpu_acquired = True
            
            # Prepare training data
            df_train, actual_feature_set = fetch_training_data(
                ticker, df_train_period, class_horizon, train_start, train_end
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
            # Release GPU semaphore
            if gpu_acquired and _GPU_SEMAPHORE is not None:
                _GPU_SEMAPHORE.release()
    
    # ✅ HANDLE UNKNOWN TASK TYPE (outside try-except)
    if task_type not in ['ticker', 'ai_portfolio']:
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
                _ensure_dir(models_dir)  # Ensure directory exists before saving
                final_model_path = models_dir / f"{ticker}_TargetReturn_model.joblib"
                final_scaler_path = models_dir / f"{ticker}_TargetReturn_scaler.joblib"
                
                print(f"   💾 Saving model for {ticker} to {final_model_path.absolute()}")
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
                print(f"  ⚠️ Error loading/saving model for {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"   ✅ Successfully aggregated {len(ticker_models)} ticker models")
        
        # Verify models were saved
        saved_count = 0
        for ticker in ticker_models.keys():
            model_file = models_dir / f"{ticker}_TargetReturn_model.joblib"
            if model_file.exists():
                saved_count += 1
        print(f"   💾 Verified {saved_count}/{len(ticker_models)} models saved to disk at {models_dir.absolute()}")
    
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
    ticker_tasks, ai_portfolio_tasks, ticker_data_dict = generate_training_tasks(
        tickers=tickers,
        all_tickers_data=all_tickers_data,
        train_start=train_start,
        train_end=train_end,
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
    print(f"   📦 Shared data: {len(ticker_data_dict)} tickers (pickled once, not per-task)")
    
    # ✅ IMPORT TIME AT FUNCTION LEVEL (fixes scope issue)
    import time
    
    # Create GPU semaphore
    gpu_semaphore = None
    if CUDA_AVAILABLE and not FORCE_CPU:
        gpu_semaphore = mp.Semaphore(GPU_MAX_CONCURRENT_TRAINING_WORKERS)
    
    # Execute tasks in parallel
    start_time = time.time()
    
    # Write per-ticker data files (workers load only what they need - ~125KB each)
    shared_data_dir = os.path.join(tempfile.gettempdir(), f'training_data_{os.getpid()}')
    os.makedirs(shared_data_dir, exist_ok=True)
    print(f"   💾 Writing {len(ticker_data_dict)} per-ticker data files...")
    for ticker, df in ticker_data_dict.items():
        safe_name = ticker.replace('/', '_').replace('\\', '_').replace('.', '_')
        joblib.dump(df, os.path.join(shared_data_dir, f'{safe_name}.pkl'), compress=1)
    total_size_mb = sum(os.path.getsize(os.path.join(shared_data_dir, f)) for f in os.listdir(shared_data_dir)) / (1024 * 1024)
    print(f"   💾 Per-ticker files: {total_size_mb:.1f} MB total (~{total_size_mb/len(ticker_data_dict)*1024:.0f} KB each)")
    
    # Free the in-memory dict now that it's on disk
    del ticker_data_dict
    
    try:
        # Use multiprocessing.Pool with initializer for proper worker setup
        # Set maxtasksperchild to recycle workers after timeouts (prevents stuck workers)
        with mp.Pool(
            processes=TRAINING_NUM_PROCESSES,
            initializer=_init_worker,
            initargs=(gpu_semaphore, shared_data_dir),
            maxtasksperchild=1  # Recycle workers after EVERY task - ensures fresh workers after timeouts
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
                    # ✅ Use apply_async with timeout for each task
                    batch_results = []
                    per_task_timeout = PER_TICKER_TIMEOUT + 30  # Add 30s buffer for GPU models
                    
                    with tqdm(total=len(batch_tasks), desc=f"Batch {i//batch_size + 1}", position=0, leave=True) as pbar:
                        # Submit all tasks asynchronously
                        async_results = []
                        for task in batch_tasks:
                            async_result = pool.apply_async(safe_universal_model_worker, (task,))
                            async_results.append(async_result)
                        
                        # Collect results with timeout
                        for idx, async_result in enumerate(async_results):
                            try:
                                # Wait for result with timeout
                                result = async_result.get(timeout=per_task_timeout)
                                batch_results.append(result)
                            except mp.TimeoutError:
                                # Task exceeded timeout
                                task = batch_tasks[idx]
                                print(f"  ⏰ TIMEOUT: {task.get('ticker', 'unknown')} {task.get('model_type', 'unknown')} exceeded {per_task_timeout}s")
                                batch_results.append({
                                    'task_type': task.get('task_type', 'unknown'),
                                    'ticker': task.get('ticker', 'unknown'),
                                    'model_type': task.get('model_type', 'unknown'),
                                    'status': 'timeout',
                                    'reason': f'exceeded_{per_task_timeout}s'
                                })
                            except Exception as e:
                                # Other error
                                task = batch_tasks[idx]
                                print(f"  ❌ ERROR: {task.get('ticker', 'unknown')} {task.get('model_type', 'unknown')}: {str(e)[:100]}")
                                batch_results.append({
                                    'task_type': task.get('task_type', 'unknown'),
                                    'ticker': task.get('ticker', 'unknown'),
                                    'model_type': task.get('model_type', 'unknown'),
                                    'status': 'error',
                                    'reason': str(e)[:200]
                                })
                            finally:
                                pbar.update(1)
                                pbar.refresh()
                                import sys
                                sys.stdout.flush()
                    
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
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(shared_data_dir):
            shutil.rmtree(shared_data_dir, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Total training time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"   Average time per task: {elapsed_time/len(all_tasks):.2f}s")
    
    # Aggregate results
    ticker_models, ai_portfolio_model = aggregate_results(results, tickers)
    
    return ticker_models, ai_portfolio_model
