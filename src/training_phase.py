"""
Training Phase Module
Handles model training for all periods (1-Year, YTD, 3-Month, 1-Month).
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import joblib
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, current_process
from tqdm import tqdm

from config import (
    PYTORCH_AVAILABLE, FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    PERIOD_HORIZONS, NUM_PROCESSES
)
from data_utils import fetch_training_data, _ensure_dir
from ml_models import initialize_ml_libraries, train_and_evaluate_models

# Conditionally import LSTM/GRU classes if PyTorch is available
try:
    from ml_models import LSTMClassifier, GRUClassifier, GRURegressor  # ✅ Added GRURegressor
except ImportError:
    LSTMClassifier = None
    GRUClassifier = None


def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    model_sell_path = models_dir / f"{ticker}_model_sell.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_buy_path = models_dir / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
    gru_hyperparams_sell_path = models_dir / f"{ticker}_TargetClassSell_gru_optimized_params.json"

    model_buy, model_sell, scaler = None, None, None
    
    # Flag to indicate if we successfully loaded a model to continue training
    loaded_for_retraining = False

    # Attempt to load models and GRU hyperparams if CONTINUE_TRAINING_FROM_EXISTING is True
    if CONTINUE_TRAINING_FROM_EXISTING and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

    # If FORCE_TRAINING is False and we didn't load for retraining, then we just load and skip training
    if not FORCE_TRAINING and not loaded_for_retraining and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} (FORCE_TRAINING is False).")
            # Before returning, ensure PyTorch models are on CPU if they are deep learning models
            if PYTORCH_AVAILABLE:
                if isinstance(model_buy, (LSTMClassifier, GRUClassifier, GRURegressor)):  # ✅ Added GRURegressor
                    model_buy = model_buy.cpu()
                if isinstance(model_sell, (LSTMClassifier, GRUClassifier, GRURegressor)):  # ✅ Added GRURegressor
                    model_sell = model_sell.cpu()
            return {
                'ticker': ticker,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler,
                'gru_hyperparams_buy': loaded_gru_hyperparams_buy,
                'gru_hyperparams_sell': loaded_gru_hyperparams_sell,
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
        print(f"  ❌ Skipping {ticker}: Insufficient training data.")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'y_scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for BUY target.")
    # Train BUY model, passing the potentially loaded model and GRU hyperparams
    # Pass the global models_and_params to avoid re-initialization in worker processes
    global_models_and_params = initialize_ml_libraries() # Ensure it's initialized in the worker process too
    
    # Choose target columns based on regression vs classification mode
    from config import USE_REGRESSION_MODEL
    buy_target = "TargetReturnBuy" if USE_REGRESSION_MODEL else "TargetClassBuy"
    sell_target = "TargetReturnSell" if USE_REGRESSION_MODEL else "TargetClassSell"
    
    model_buy, scaler_buy, y_scaler_buy, gru_hyperparams_buy = train_and_evaluate_models(
        df_train, buy_target, actual_feature_set, ticker=ticker,
        initial_model=model_buy if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_buy,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )
    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for SELL target.")
    # Train SELL model, passing the potentially loaded model and GRU hyperparams
    model_sell, scaler_sell, y_scaler_sell, gru_hyperparams_sell = train_and_evaluate_models(
        df_train, sell_target, actual_feature_set, ticker=ticker,
        initial_model=model_sell if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_sell,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )

    # For simplicity, we'll use the scaler from the buy model for both if they are different.
    # In a more complex scenario, you might want to ensure feature_set consistency or use separate scalers.
    final_scaler = scaler_buy if scaler_buy else scaler_sell
    final_y_scaler = y_scaler_buy if y_scaler_buy else y_scaler_sell  # ✅ Also choose y_scaler

    if model_buy and model_sell and final_scaler:
        try:
            joblib.dump(model_buy, model_buy_path)
            joblib.dump(model_sell, model_sell_path)
            joblib.dump(final_scaler, scaler_path)
            
            # ✅ Save y_scaler if it exists
            if final_y_scaler is not None:
                y_scaler_path = models_dir / f"{ticker}_y_scaler.joblib"
                joblib.dump(final_y_scaler, y_scaler_path)
            
            if gru_hyperparams_buy:
                with open(gru_hyperparams_buy_path, 'w') as f:
                    json.dump(gru_hyperparams_buy, f, indent=4)
            if gru_hyperparams_sell:
                with open(gru_hyperparams_sell_path, 'w') as f:
                    json.dump(gru_hyperparams_sell, f, indent=4)

            print(f"  ✅ Models, scaler, y_scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving models or GRU hyperparams for {ticker}: {e}")
            
        # Before returning, ensure PyTorch models are on CPU if they are deep learning models
        if PYTORCH_AVAILABLE:
            if isinstance(model_buy, (LSTMClassifier, GRUClassifier, GRURegressor)):  # ✅ Added GRURegressor
                model_buy = model_buy.cpu()
            if isinstance(model_sell, (LSTMClassifier, GRUClassifier, GRURegressor)):  # ✅ Added GRURegressor
                model_sell = model_sell.cpu()

        return {
            'ticker': ticker,
            'model_buy': model_buy,
            'model_sell': model_sell,
            'scaler': final_scaler,
            'y_scaler': final_y_scaler,  # ✅ Return y_scaler
            'gru_hyperparams_buy': gru_hyperparams_buy,
            'gru_hyperparams_sell': gru_hyperparams_sell,
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
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'y_scaler': None, 'status': 'failed', 'reason': reason}


def train_models_for_period(
    period_name: str,
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    top_performers_data: List[Tuple[str, float, float]],
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Train models for a specific period (1-Year, YTD, 3-Month, 1-Month).
    
    Args:
        period_name: Name of the period ("1-Year", "YTD", "3-Month", "1-Month")
        tickers: List of ticker symbols to train
        all_tickers_data: DataFrame containing all ticker data
        train_start: Training start date
        train_end: Training end date
        top_performers_data: List of (ticker, 1y_perf, ytd_perf) tuples
        feature_set: List of feature names to use
        run_parallel: Whether to use parallel processing
        
    Returns:
        Tuple of (models_buy, models_sell, scalers, y_scalers) dictionaries
    """
    print(f"\n🔍 Step 3: Training AI models for {period_name} backtest...")
    
    models_buy = {}
    models_sell = {}
    scalers = {}
    
    # Calculate period-specific horizon
    if period_name == "YTD":
        # Calculate YTD trading days dynamically
        from datetime import timezone
        ytd_start = datetime(datetime.today().year, 1, 1, tzinfo=timezone.utc)
        ytd_days = (train_end - ytd_start).days
        period_horizon = int(ytd_days * 0.7)  # Approximate trading days
        if period_horizon <= 0:
            period_horizon = 252  # Fallback to full year
    else:
        period_horizon = PERIOD_HORIZONS[period_name]
    
    # Prepare training parameters for each ticker
    training_params = []
    
    for ticker in tickers:
        # Load existing GRU hyperparameters if available
        models_dir = Path("logs/models")
        _ensure_dir(models_dir)
        
        gru_hyperparams_buy_path = models_dir / f"{ticker}_{period_name}_TargetClassBuy_gru_optimized_params.json"
        gru_hyperparams_sell_path = models_dir / f"{ticker}_{period_name}_TargetClassSell_gru_optimized_params.json"
        
        loaded_gru_hyperparams_buy = None
        loaded_gru_hyperparams_sell = None
        
        try:
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
        
        try:
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)
        except Exception as e:
            print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")
        
        # Helper to extract a close series (prefers 'Close', falls back to 'close')
        def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
            if 'Close' in df.columns:
                return df['Close']
            if 'close' in df.columns:
                return df['close']
            return None

        # Get Buy & Hold return for this ticker
        ticker_bh_return = 0.01  # Default 1% if not found
        for t, perf_1y, perf_ytd in top_performers_data:
            if t == ticker:
                if period_name == "1-Year":
                    ticker_bh_return = perf_1y / 100.0
                elif period_name == "YTD":
                    ticker_bh_return = perf_ytd / 100.0
                elif period_name == "3-Month":
                    # Extract 3-month performance from data if available
                    try:
                        ticker_data = all_tickers_data.loc[:, (slice(None), ticker)]
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                        train_3m_start = train_end - pd.Timedelta(days=90)
                        ticker_3m = ticker_data.loc[train_3m_start:train_end]
                        close_series = _get_close_series(ticker_3m)
                        if close_series is not None and len(close_series) > 1:
                            ticker_bh_return = (close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0]
                        else:
                            ticker_bh_return = 0.01
                    except:
                        ticker_bh_return = 0.01
                elif period_name == "1-Month":
                    # Extract 1-month performance from data if available
                    try:
                        ticker_data = all_tickers_data.loc[:, (slice(None), ticker)]
                        ticker_data.columns = ticker_data.columns.droplevel(1)
                        train_1m_start = train_end - pd.Timedelta(days=32)
                        ticker_1m = ticker_data.loc[train_1m_start:train_end]
                        close_series = _get_close_series(ticker_1m)
                        if close_series is not None and len(close_series) > 1:
                            ticker_bh_return = (close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0]
                        else:
                            ticker_bh_return = 0.01
                    except:
                        ticker_bh_return = 0.01
                break
        
        # Target is the B&H return for the period
        period_target_pct = abs(ticker_bh_return)
        period_target_pct = max(period_target_pct, 0.01)  # At least 1%
        
        print(f"    📊 {ticker} {period_name[:2]} Training: Horizon={period_horizon}d, Target={period_target_pct:.2%} (B&H: {ticker_bh_return:.2%})")
        
        try:
            # Slice the main DataFrame for the training period
            ticker_train_data = all_tickers_data.loc[train_start:train_end, (slice(None), ticker)]
            ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
            training_params.append((
                ticker,
                ticker_train_data.copy(),
                period_target_pct,
                period_horizon,
                feature_set,
                loaded_gru_hyperparams_buy,
                loaded_gru_hyperparams_sell
            ))
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice training data for {ticker} for {period_name} period. Skipping.")
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
    for res in training_results:
        if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'):
            models_buy[res['ticker']] = res['model_buy']
            models_sell[res['ticker']] = res['model_sell']
            scalers[res['ticker']] = res['scaler']
            y_scalers[res['ticker']] = res.get('y_scaler', None)  # ✅ Collect y_scaler
    
    print(f"✅ {period_name} training complete: {len(models_buy)} models trained/loaded.")
    
    return models_buy, models_sell, scalers, y_scalers  # ✅ Return y_scalers

