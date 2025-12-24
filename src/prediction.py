"""
Prediction Module
Handles stock prediction and ranking logic used by both backtesting and live trading.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone

from config import SEQUENCE_LENGTH, PYTORCH_AVAILABLE, CUDA_AVAILABLE
from data_utils import fetch_training_data


def predict_return_for_ticker(
    ticker: str,
    data: pd.DataFrame,
    model,
    scaler,
    y_scaler,
    feature_set: List[str],
    horizon_days: int = 10
) -> float:
    """
    Predict future return for a single ticker using the trained model.
    
    Args:
        ticker: Stock ticker symbol
        data: Historical OHLCV data for the ticker
        model_buy: Trained regression model (predicts return)
        scaler: Feature scaler
        y_scaler: Target scaler (for regression models)
        feature_set: List of feature names the model expects
        horizon_days: Prediction horizon (default 10 days)
    
    Returns:
        Predicted return (e.g., 0.05 for 5% return), or -inf if prediction fails
    """
    try:
        if model_buy is None or scaler is None:
            return -np.inf
        
        # Check if we have required OHLCV data
        if data.empty or len(data) < horizon_days:
            return -np.inf
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return -np.inf
        
        # Engineer features from raw OHLCV data
        processed_data, actual_features = fetch_training_data(
            ticker, 
            data, 
            target_percentage=0.006  # Not used for prediction, just for feature engineering
        )
        
        if processed_data.empty or len(processed_data) < 1:
            return -np.inf
        
        # Get the last row for prediction
        latest_data = processed_data.iloc[-1:].copy()
        
        # Ensure all required features are present
        for feature in feature_set:
            if feature not in latest_data.columns:
                latest_data[feature] = 0
        
        latest_data = latest_data[feature_set]
        
        # Handle PyTorch models (LSTM/GRU) separately
        if PYTORCH_AVAILABLE:
            try:
                from ml_models import LSTMClassifier, GRUClassifier, GRURegressor
                
                if isinstance(model_buy, (LSTMClassifier, GRUClassifier, GRURegressor)):
                    # Need sequence data for PyTorch models
                    if len(processed_data) < SEQUENCE_LENGTH:
                        return -np.inf
                    
                    import torch
                    
                    # Get sequence of SEQUENCE_LENGTH days
                    sequence_data = processed_data.iloc[-SEQUENCE_LENGTH:][feature_set].copy()
                    
                    # Ensure numeric and fill NaNs
                    for col in sequence_data.columns:
                        sequence_data[col] = pd.to_numeric(sequence_data[col], errors='coerce').fillna(0.0)
                    
                    # Scale sequence
                    X_scaled = scaler.transform(sequence_data)
                    
                    # Convert to tensor and add batch dimension
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
                    
                    # Move to device
                    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
                    X_tensor = X_tensor.to(device)
                    
                    # Predict
                    model.eval()
                    with torch.no_grad():
                        output = model(X_tensor)
                        prediction = float(output.cpu().numpy()[0][0])
                    
                    # Inverse transform if using regression with y_scaler
                    if y_scaler is not None:
                        # Clip to [-1, 1] before inverse transform to prevent extrapolation
                        prediction_clipped = np.clip(float(prediction), -1.0, 1.0)
                        prediction = y_scaler.inverse_transform([[prediction_clipped]])[0][0]
                    
                    # Clip to reasonable return range (-100% to +200%)
                    prediction = np.clip(float(prediction), -1.0, 2.0)
                    return prediction
            except ImportError:
                pass  # Fall through to traditional ML
        
        # Traditional ML models (XGBoost, LightGBM, RandomForest, etc.)
        scaled_data = scaler.transform(latest_data)
        
        # Always use regression approach - predict returns directly
        prediction = model.predict(scaled_data)[0]
        
        # Inverse transform if we scaled the target
        if y_scaler is not None:
            # Clip to [-1, 1] before inverse transform to prevent extrapolation
            prediction_clipped = np.clip(float(prediction), -1.0, 1.0)
            prediction = y_scaler.inverse_transform([[prediction_clipped]])[0][0]
        
        # Clip to reasonable return range (-100% to +200%)
        prediction = np.clip(float(prediction), -1.0, 2.0)
        return float(prediction)
    
    except Exception as e:
        print(f"  ⚠️ Error predicting for {ticker}: {e}")
        return -np.inf


def rank_tickers_by_predicted_return(
    tickers: List[str],
    all_data: pd.DataFrame,
    models_buy: Dict,
    scalers: Dict,
    y_scalers: Dict,
    feature_set: List[str],
    horizon_days: int = 10,
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Rank tickers by predicted return and return top N.
    
    This is the EXACT logic used in backtesting (src/backtesting.py lines 882-889).
    
    Args:
        tickers: List of ticker symbols to evaluate
        all_data: Multi-index DataFrame with OHLCV data for all tickers
        models_buy: Dict mapping ticker -> trained model
        scalers: Dict mapping ticker -> feature scaler
        y_scalers: Dict mapping ticker -> target scaler
        feature_set: List of feature names
        horizon_days: Prediction horizon
        top_n: Number of top stocks to return (default 3)
    
    Returns:
        List of (ticker, predicted_return) tuples, sorted by predicted return (descending)
    """
    predictions = []
    
    for ticker in tickers:
        try:
            # Extract data for this ticker
            ticker_data = all_data.loc[:, (slice(None), ticker)]
            ticker_data.columns = ticker_data.columns.droplevel(1)
            
            if ticker_data.empty:
                continue
            
            # Predict return
            predicted_return = predict_return_for_ticker(
                ticker,
                ticker_data,
                models_buy.get(ticker),
                scalers.get(ticker),
                y_scalers.get(ticker),
                feature_set,
                horizon_days
            )
            
            predictions.append((ticker, predicted_return))
        
        except Exception as e:
            print(f"  ⚠️ Error processing {ticker}: {e}")
            continue
    
    # Sort by predicted return (descending) and take top N
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    return predictions[:top_n]


def load_models_for_tickers(
    tickers: List[str],
    models_dir: Path = Path("logs/models")
) -> Tuple[Dict, Dict, Dict]:
    """
    Load trained models, scalers, and y_scalers for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        models_dir: Directory containing saved models
    
    Returns:
        Tuple of (models_buy, scalers, y_scalers) dictionaries
    """
    models_buy = {}
    scalers = {}
    y_scalers = {}
    
    for ticker in tickers:
        try:
            model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
            scaler_path = models_dir / f"{ticker}_scaler.joblib"
            y_scaler_path = models_dir / f"{ticker}_y_scaler.joblib"
            
            if model_buy_path.exists():
                models_buy[ticker] = joblib.load(model_buy_path)
            
            if scaler_path.exists():
                scalers[ticker] = joblib.load(scaler_path)
            
            if y_scaler_path.exists():
                y_scalers[ticker] = joblib.load(y_scaler_path)
        
        except Exception as e:
            print(f"  ⚠️ Error loading model for {ticker}: {e}")
            continue
    
    return models_buy, scalers, y_scalers


def get_feature_set_from_saved_model(
    ticker: str,
    models_dir: Path = Path("logs/models")
) -> Optional[List[str]]:
    """
    Extract feature names from a saved scaler.
    
    Args:
        ticker: Stock ticker symbol
        models_dir: Directory containing saved models
    
    Returns:
        List of feature names, or None if not found
    """
    try:
        scaler_path = models_dir / f"{ticker}_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            if hasattr(scaler, 'feature_names_in_'):
                return list(scaler.feature_names_in_)
    except Exception as e:
        print(f"  ⚠️ Error loading feature set for {ticker}: {e}")
    
    return None


