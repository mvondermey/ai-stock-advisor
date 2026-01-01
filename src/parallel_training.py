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
    USE_LSTM, USE_GRU, USE_TCN, USE_XGBOOST, USE_RANDOM_FOREST, 
    USE_LIGHTGBM, USE_RIDGE, USE_ELASTIC_NET, USE_SVM, USE_MLP_CLASSIFIER,
    AI_PORTFOLIO_N_JOBS
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
    print(f"   Training period: {train_start.date()} to {train_end.date()}")
    print(f"   Data shape: {all_tickers_data.shape}")
    print(f"   Data columns: {list(all_tickers_data.columns)[:10]}")
    
    # Generate ticker model tasks
    ticker_tasks = []
    for ticker in tickers:
        try:
            # ✅ FIX: Handle both long-format and wide-format data
            if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
                # Long format: filter by ticker and date range
                df_train_period = all_tickers_data[
                    (all_tickers_data['ticker'] == ticker) &
                    (all_tickers_data['date'] >= train_start) &
                    (all_tickers_data['date'] <= train_end)
                ].copy()
                
                # Remove ticker column and set date as index for training
                if not df_train_period.empty:
                    if 'date' in df_train_period.columns:
                        df_train_period = df_train_period.set_index('date')
                    if 'ticker' in df_train_period.columns:
                        df_train_period = df_train_period.drop('ticker', axis=1)
            else:
                # Wide format: use index-based slicing
                print(f"  ⚠️ Wide format data not supported in unified training")
                continue
            
            if df_train_period.empty:
                print(f"  ⚠️ No training data found for {ticker} in period {train_start.date()} to {train_end.date()}. Skipping.")
                continue
            
            if len(df_train_period) < 50:
                print(f"  ⚠️ Insufficient data for {ticker} ({len(df_train_period)} rows < 50), skipping")
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


def universal_model_worker(task: Dict) -> Dict:
    """
    Worker function that trains one model for one task.
    
    Args:
        task: Dict with task_type ('ticker' or 'ai_portfolio') and training parameters
    
    Returns:
        Dict with training results
    """
    global _GPU_SEMAPHORE
    
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
            
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'success',
                'mse': result['mse'],
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'y_scaler_path': str(y_scaler_path) if y_scaler_path else None
            }
        
        except Exception as e:
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'error',
                'reason': str(e)[:200]
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
                    n_jobs=AI_PORTFOLIO_N_JOBS
                )
            elif model_type == 'XGBoost':
                import xgboost as xgb
                common_kwargs = {
                    "random_state": SEED,
                    "tree_method": "hist",
                    "nthread": AI_PORTFOLIO_N_JOBS,
                }
                if XGBOOST_USE_GPU and CUDA_AVAILABLE and not FORCE_CPU:
                    common_kwargs["device"] = "cuda"
                model = xgb.XGBClassifier(n_estimators=200, max_depth=7, **common_kwargs)
            
            elif model_type == 'LightGBM':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200, max_depth=7, random_state=SEED, 
                    verbosity=-1, n_jobs=AI_PORTFOLIO_N_JOBS
                )
            elif model_type == 'Ridge':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=AI_PORTFOLIO_N_JOBS)
            
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
    
    else:
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
    error_count = len([r for r in results if r.get('status') in ['failed', 'error']])
    
    print(f"   ✅ Successful: {success_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # ============================================
    # AGGREGATE TICKER MODELS
    # ============================================
    ticker_models = {}
    
    if ticker_results:
        print(f"\n📈 Aggregating ticker models for {len(tickers)} tickers...")
        
        # Group results by ticker
        ticker_groups = {}
        for result in ticker_results:
            ticker = result['ticker']
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(result)
        
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
                
                # Save with standard naming (matching load_models_for_tickers expectations)
                models_dir = Path("logs/models")
                final_model_path = models_dir / f"{ticker}_model.joblib"
                final_scaler_path = models_dir / f"{ticker}_scaler.joblib"
                
                # Handle PyTorch models - save state_dict instead of full model
                if PYTORCH_AVAILABLE and hasattr(model, 'state_dict'):
                    import torch
                    torch.save(model.state_dict(), final_model_path)
                    # Save model metadata for reconstruction
                    model_info = {
                        'model_class': model.__class__.__name__,
                        'state_dict_path': str(final_model_path)
                    }
                    # Save architecture info for reconstruction
                    if hasattr(model, 'gru') or hasattr(model, 'lstm'):
                        rnn = getattr(model, 'gru', None) or getattr(model, 'lstm', None)
                        if rnn:
                            model_info['input_size'] = rnn.input_size
                            model_info['hidden_size'] = rnn.hidden_size
                            model_info['num_layers'] = rnn.num_layers
                    elif hasattr(model, 'net') and len(model.net) > 0:
                        # TCN model
                        first_conv = model.net[0]
                        if hasattr(first_conv, 'in_channels'):
                            model_info['input_size'] = first_conv.in_channels
                    joblib.dump(model_info, final_model_path.with_suffix('.info'))
                else:
                    joblib.dump(model, final_model_path)
                
                joblib.dump(scaler, final_scaler_path)
                
                if y_scaler is not None:
                    final_y_scaler_path = models_dir / f"{ticker}_y_scaler.joblib"
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
    
    # Create GPU semaphore
    gpu_semaphore = None
    if CUDA_AVAILABLE and not FORCE_CPU:
        gpu_semaphore = mp.Semaphore(GPU_MAX_CONCURRENT_TRAINING_WORKERS)
    
    # Execute tasks in parallel
    start_time = time.time()
    
    try:
        # Use multiprocessing pool
        with mp.Pool(
            processes=TRAINING_NUM_PROCESSES,
            initializer=_init_worker,
            initargs=(gpu_semaphore,)
        ) as pool:
            # Use imap_unordered for progress tracking
            results = list(tqdm(
                pool.imap_unordered(universal_model_worker, all_tasks),
                total=len(all_tasks),
                desc="Training models"
            ))
    
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
