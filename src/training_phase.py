"""
Training Phase Module
Handles model training for all periods (1-Year, YTD, 3-Month, 1-Month).
"""

from pathlib import Path
from typing import Tuple, Dict
import json
import joblib
from multiprocessing import current_process

from config import (
    PYTORCH_AVAILABLE, FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING
)
from data_utils import fetch_training_data, _ensure_dir
from ml_models import initialize_ml_libraries, train_and_evaluate_models

# Conditionally import LSTM/GRU classes if PyTorch is available
try:
    from ml_models import LSTMClassifier, GRUClassifier
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
                if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                    model_buy = model_buy.cpu()
                if isinstance(model_sell, (LSTMClassifier, GRUClassifier)):
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
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for BUY target.")
    # Train BUY model, passing the potentially loaded model and GRU hyperparams
    # Pass the global models_and_params to avoid re-initialization in worker processes
    global_models_and_params = initialize_ml_libraries() # Ensure it's initialized in the worker process too
    model_buy, scaler_buy, gru_hyperparams_buy = train_and_evaluate_models(
        df_train, "TargetClassBuy", actual_feature_set, ticker=ticker,
        initial_model=model_buy if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_buy,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )
    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for SELL target.")
    # Train SELL model, passing the potentially loaded model and GRU hyperparams
    model_sell, scaler_sell, gru_hyperparams_sell = train_and_evaluate_models(
        df_train, "TargetClassSell", actual_feature_set, ticker=ticker,
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

    if model_buy and model_sell and final_scaler:
        try:
            joblib.dump(model_buy, model_buy_path)
            joblib.dump(model_sell, model_sell_path)
            joblib.dump(final_scaler, scaler_path)
            
            if gru_hyperparams_buy:
                with open(gru_hyperparams_buy_path, 'w') as f:
                    json.dump(gru_hyperparams_buy, f, indent=4)
            if gru_hyperparams_sell:
                with open(gru_hyperparams_sell_path, 'w') as f:
                    json.dump(gru_hyperparams_sell, f, indent=4)

            print(f"  ✅ Models, scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving models or GRU hyperparams for {ticker}: {e}")
            
        # Before returning, ensure PyTorch models are on CPU if they are deep learning models
        if PYTORCH_AVAILABLE:
            if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                model_buy = model_buy.cpu()
            if isinstance(model_sell, (LSTMClassifier, GRUClassifier)):
                model_sell = model_sell.cpu()

        return {
            'ticker': ticker,
            'model_buy': model_buy,
            'model_sell': model_sell,
            'scaler': final_scaler,
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
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'status': 'failed', 'reason': reason}

