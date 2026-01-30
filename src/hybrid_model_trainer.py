# hybrid_model_trainer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_hybrid_features_for_training(hourly_data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """
    Create hybrid features for model training.
    
    This creates a feature matrix that combines daily and intraday features
    suitable for AI model training.
    
    Args:
        hourly_data: DataFrame with hourly OHLCV data
        config: Configuration dictionary
        
    Returns:
        DataFrame with hybrid features ready for training
    """
    if hourly_data.empty:
        return pd.DataFrame()
    
    try:
        from hybrid_data_processor import aggregate_hourly_to_daily, calculate_intraday_features
    except ImportError:
        print("❌ Hybrid data processor not available")
        return pd.DataFrame()
    
    # Aggregate to daily data
    daily_data = aggregate_hourly_to_daily(hourly_data)
    
    if daily_data.empty:
        return pd.DataFrame()
    
    # Create feature matrix
    features_list = []
    
    # Process each day (skip first few days due to lookback requirements)
    min_daily_days = 30  # Need at least 30 days for daily features
    min_hourly_points = 168  # Need at least 1 week of hourly data
    
    for i in range(min_daily_days, len(daily_data)):
        current_daily_date = daily_data.index[i]
        
        # Get daily features up to current date
        daily_slice = daily_data.iloc[:i+1]
        daily_features = calculate_daily_features_for_training(daily_slice)
        
        # Get intraday features up to current date
        current_date_end = current_daily_date.replace(hour=23, minute=59)
        hourly_slice = hourly_data.loc[hourly_data.index <= current_date_end]
        
        if len(hourly_slice) >= min_hourly_points:
            intraday_features = calculate_intraday_features(hourly_slice, lookback_hours=24)
        else:
            intraday_features = {}
        
        # Combine features
        combined_features = {}
        combined_features.update(daily_features)
        combined_features.update(intraday_features)
        
        # Add target (next day's return)
        if i < len(daily_data) - 1:
            next_day_return = (daily_data.iloc[i+1]['Close'] / daily_data.iloc[i]['Close'] - 1) * 100
            combined_features['target_return'] = next_day_return
        else:
            combined_features['target_return'] = np.nan
        
        combined_features['date'] = current_daily_date
        features_list.append(combined_features)
    
    return pd.DataFrame(features_list)

def calculate_daily_features_for_training(daily_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate daily features for model training."""
    features = {}
    
    if daily_data.empty or len(daily_data) < 30:
        return features
    
    closes = daily_data['Close']
    volumes = daily_data['Volume'] if 'Volume' in daily_data.columns else pd.Series([0]*len(daily_data))
    
    # Price-based features
    if len(closes) >= 2:
        # Returns
        returns = closes.pct_change().dropna()
        features['daily_return_1d'] = returns.iloc[-1] * 100
        features['daily_return_5d'] = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
        features['daily_return_10d'] = (closes.iloc[-1] / closes.iloc[-10] - 1) * 100 if len(closes) >= 10 else 0
        features['daily_return_30d'] = (closes.iloc[-1] / closes.iloc[-30] - 1) * 100 if len(closes) >= 30 else 0
        
        # Volatility
        features['daily_volatility_5d'] = returns.tail(5).std() * np.sqrt(252) * 100 if len(returns) >= 5 else 0
        features['daily_volatility_10d'] = returns.tail(10).std() * np.sqrt(252) * 100 if len(returns) >= 10 else 0
        features['daily_volatility_30d'] = returns.tail(30).std() * np.sqrt(252) * 100 if len(returns) >= 30 else 0
        
        # Moving averages
        if len(closes) >= 20:
            sma_20 = closes.tail(20).mean()
            features['sma_20d'] = sma_20
            features['price_vs_sma_20d'] = (closes.iloc[-1] / sma_20 - 1) * 100
        
        if len(closes) >= 50:
            sma_50 = closes.tail(50).mean()
            features['sma_50d'] = sma_50
            features['price_vs_sma_50d'] = (closes.iloc[-1] / sma_50 - 1) * 100
        
        # RSI
        if len(closes) >= 14:
            features['rsi_14d'] = calculate_rsi(closes, 14)
        
        # Volume features
        if len(volumes) >= 20:
            avg_volume_20d = volumes.tail(20).mean()
            features['volume_ratio_20d'] = volumes.iloc[-1] / avg_volume_20d
            features['volume_trend_5d'] = (volumes.tail(5).mean() / volumes.tail(20).head(5).mean() - 1) * 100
    
    return features

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

def prepare_hybrid_training_data(ticker: str, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare hybrid training data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Tuple of (features_df, targets_df)
    """
    try:
        from data_utils import load_hybrid_features
    except ImportError:
        print("❌ Data utils not available")
        return pd.DataFrame(), pd.DataFrame()
    
    # Load hybrid data
    daily_data, intraday_data = load_hybrid_features(ticker, start_date, end_date)
    
    if intraday_data.empty:
        print(f"❌ No intraday data available for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create hybrid features
    features_df = create_hybrid_features_for_training(intraday_data)
    
    if features_df.empty:
        print(f"❌ Could not create features for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Remove rows with NaN targets
    features_df = features_df.dropna(subset=['target_return'])
    
    if features_df.empty:
        print(f"❌ No valid training samples for {ticker}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Separate features and targets
    feature_columns = [col for col in features_df.columns if col not in ['date', 'target_return']]
    X = features_df[feature_columns]
    y = features_df['target_return']
    
    return X, y

def check_model_compatibility(ticker: str) -> bool:
    """
    Check if existing model is compatible with hybrid features.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        True if model needs retraining, False if compatible
    """
    try:
        import joblib
        from config import DATA_INTERVAL, AGGREGATE_TO_DAILY
        
        # If we're using hybrid data, models need retraining
        if AGGREGATE_TO_DAILY and DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:
            model_path = Path(f"logs/models/{ticker}_TargetReturn_model.joblib")
            if model_path.exists():
                # Model exists but was trained on daily features
                print(f"🔄 {ticker}: Model exists but needs retraining for hybrid features")
                return True
            else:
                print(f"🆕 {ticker}: No model exists, will train with hybrid features")
                return True
        else:
            # Using daily data, existing models are compatible
            return False
            
    except Exception as e:
        print(f"⚠️ Error checking model compatibility for {ticker}: {e}")
        return True

def get_hybrid_feature_names() -> List[str]:
    """
    Get the list of hybrid feature names.
    
    Returns:
        List of feature names
    """
    # This should match the features created in create_hybrid_features_for_training
    feature_names = [
        # Daily features
        'daily_return_1d', 'daily_return_5d', 'daily_return_10d', 'daily_return_30d',
        'daily_volatility_5d', 'daily_volatility_10d', 'daily_volatility_30d',
        'sma_20d', 'price_vs_sma_20d', 'sma_50d', 'price_vs_sma_50d',
        'rsi_14d', 'volume_ratio_20d', 'volume_trend_5d',
        
        # Intraday features (from hybrid_data_processor)
        'hourly_momentum_1h', 'hourly_momentum_4h', 'hourly_momentum_24h',
        'price_acceleration', 'sma_5h', 'price_vs_sma_5h', 'sma_10h', 'price_vs_sma_10h',
        'intraday_volatility_6h', 'intraday_volatility_24h', 'rsi_14h',
        'volume_spike_6h', 'volume_spike_1h', 'volume_trend_6h',
        'hour_of_day', 'day_of_week', 'market_session'
    ]
    
    return feature_names
