# hybrid_data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

def aggregate_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hourly OHLCV data to daily OHLCV data.
    
    This creates perfect daily data with no gaps by aggregating hourly data.
    
    Args:
        hourly_df: DataFrame with hourly OHLCV data
        
    Returns:
        DataFrame with daily OHLCV data
    """
    if hourly_df.empty:
        return pd.DataFrame()
    
    try:
        # Handle duplicate columns (e.g., from MultiIndex flattening)
        if hourly_df.columns.duplicated().any():
            print(f"    ⚠️ Found duplicate columns, deduplicating...")
            # Keep only the first occurrence of each column
            hourly_df = hourly_df.loc[:, ~hourly_df.columns.duplicated()]
        
        # Ensure we have a datetime index
        if not isinstance(hourly_df.index, pd.DatetimeIndex):
            if 'Date' in hourly_df.columns:
                hourly_df = hourly_df.set_index('Date')
            elif 'Datetime' in hourly_df.columns:
                hourly_df = hourly_df.set_index('Datetime')
            else:
                hourly_df.index = pd.to_datetime(hourly_df.index)
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in hourly_df.columns]
        if missing_cols:
            print(f"    ❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Create date series for grouping - more robust approach
        try:
            if hasattr(hourly_df.index, 'date'):
                date_series = hourly_df.index.date
            else:
                # Fallback for different index types
                date_series = pd.to_datetime(hourly_df.index).date
        except Exception as date_error:
            print(f"    ❌ Error creating date series: {date_error}")
            return pd.DataFrame()
        
        # Group by date and aggregate
        daily_data = hourly_df.groupby(date_series).agg({
            'Open': 'first',      # First hour's open
            'High': 'max',        # Maximum of all hourly highs
            'Low': 'min',         # Minimum of all hourly lows  
            'Close': 'last',      # Last hour's close
            'Volume': 'sum'       # Sum of all hourly volumes
        })
        
        # Convert date index back to datetime
        daily_data.index = pd.to_datetime(daily_data.index)
        daily_data.index.name = 'Date'
        
        return daily_data
        
    except Exception as e:
        print(f"  ❌ Error in aggregate_hourly_to_daily: {e}")
        print(f"  📊 DataFrame shape: {hourly_df.shape}")
        print(f"  📊 Index type: {type(hourly_df.index)}")
        print(f"  📊 Index sample: {hourly_df.index[:3] if len(hourly_df.index) > 0 else 'Empty'}")
        print(f"  📊 Columns: {list(hourly_df.columns)}")
        print(f"  📊 Sample data:\n{hourly_df.head(2) if len(hourly_df) > 0 else 'Empty'}")
        # Return empty DataFrame on error to prevent crash
        return pd.DataFrame()

def calculate_intraday_features(hourly_df: pd.DataFrame, lookback_hours: int = 24) -> Dict[str, float]:
    """
    Calculate intraday features from hourly data.
    
    Args:
        hourly_df: DataFrame with hourly OHLCV data
        lookback_hours: Number of hours to look back for calculations
        
    Returns:
        Dictionary of intraday features
    """
    if hourly_df.empty or len(hourly_df) < lookback_hours:
        return {}
    
    recent_data = hourly_df.tail(lookback_hours)
    
    features = {}
    
    # Price-based features
    if 'Close' in recent_data.columns:
        closes = recent_data['Close']
        
        # Hourly momentum
        if len(closes) >= 2:
            features['hourly_momentum_1h'] = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
        if len(closes) >= 4:
            features['hourly_momentum_4h'] = (closes.iloc[-1] / closes.iloc[-4] - 1) * 100
        if len(closes) >= 24:
            features['hourly_momentum_24h'] = (closes.iloc[-1] / closes.iloc[-24] - 1) * 100
        
        # Price acceleration
        if len(closes) >= 4:
            momentum_2h = (closes.iloc[-1] / closes.iloc[-2] - 1)
            momentum_prev_2h = (closes.iloc[-2] / closes.iloc[-3] - 1)
            features['price_acceleration'] = (momentum_2h - momentum_prev_2h) * 100
        
        # Moving averages (shorter periods for intraday)
        if len(closes) >= 5:
            features['sma_5h'] = closes.tail(5).mean()
            features['price_vs_sma_5h'] = (closes.iloc[-1] / features['sma_5h'] - 1) * 100
        if len(closes) >= 10:
            features['sma_10h'] = closes.tail(10).mean()
            features['price_vs_sma_10h'] = (closes.iloc[-1] / features['sma_10h'] - 1) * 100
        
        # Volatility features
        if len(closes) >= 6:
            hourly_returns = closes.pct_change().dropna()
            features['intraday_volatility_6h'] = hourly_returns.tail(6).std() * np.sqrt(24) * 100  # Annualized
        if len(closes) >= 24:
            hourly_returns = closes.pct_change().dropna()
            features['intraday_volatility_24h'] = hourly_returns.tail(24).std() * np.sqrt(24) * 100  # Annualized
        
        # RSI (intraday)
        if len(closes) >= 14:
            features['rsi_14h'] = calculate_rsi(closes, 14)
    
    # Volume-based features
    if 'Volume' in recent_data.columns:
        volumes = recent_data['Volume']
        
        # Volume spike detection
        if len(volumes) >= 24:
            avg_volume_24h = volumes.tail(24).mean()
            features['volume_spike_6h'] = volumes.tail(6).sum() / (avg_volume_24h / 4)  # 6h vs 6h avg
            features['volume_spike_1h'] = volumes.iloc[-1] / (avg_volume_24h / 24)  # 1h vs 1h avg
        
        # Volume trend
        if len(volumes) >= 6:
            volume_trend = (volumes.tail(6).mean() / volumes.tail(24).head(6).mean() - 1) * 100
            features['volume_trend_6h'] = volume_trend
    
    # Time-based features
    if len(recent_data) > 0:
        current_time = recent_data.index[-1]
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        
        # Market session detection (assuming 9:30 AM - 4:00 PM ET)
        hour_et = current_time.hour - 5 if current_time.hour >= 5 else current_time.hour + 19  # Convert UTC to ET
        if 9 <= hour_et < 16:
            features['market_session'] = 1  # Regular session
        elif hour_et < 9:
            features['market_session'] = 0  # Pre-market
        else:
            features['market_session'] = 2  # After-hours
    
    return features

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50.0  # Neutral
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

def create_hybrid_features(hourly_df: pd.DataFrame, daily_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create combined features from hourly and daily data.
    
    Args:
        hourly_df: DataFrame with hourly OHLCV data
        daily_df: Optional DataFrame with daily OHLCV data (will be calculated if None)
        
    Returns:
        DataFrame with combined features for each timestamp
    """
    if hourly_df.empty:
        return pd.DataFrame()
    
    # Calculate daily data if not provided
    if daily_df is None:
        daily_df = aggregate_hourly_to_daily(hourly_df)
    
    features_list = []
    
    # Process each hour (or each day for daily features)
    for i in range(len(hourly_df)):
        current_time = hourly_df.index[i]
        
        # Get intraday features
        hourly_slice = hourly_df.iloc[:i+1]
        intraday_features = calculate_intraday_features(hourly_slice)
        
        # Get daily features for the current day
        current_date = current_time.date()
        if current_date in [d.date() for d in daily_df.index]:
            daily_slice = daily_df.loc[daily_df.index.date == current_date]
            if not daily_slice.empty:
                daily_features = calculate_daily_features(daily_slice)
            else:
                daily_features = {}
        else:
            daily_features = {}
        
        # Combine features
        combined_features = {f'intraday_{k}': v for k, v in intraday_features.items()}
        combined_features.update({f'daily_{k}': v for k, v in daily_features.items()})
        combined_features['timestamp'] = current_time
        
        features_list.append(combined_features)
    
    return pd.DataFrame(features_list)

def calculate_daily_features(daily_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate traditional daily features from daily data.
    
    Args:
        daily_df: DataFrame with daily OHLCV data
        
    Returns:
        Dictionary of daily features
    """
    if daily_df.empty:
        return {}
    
    features = {}
    
    if 'Close' in daily_df.columns:
        closes = daily_df['Close']
        
        # Daily returns
        if len(closes) >= 2:
            daily_returns = closes.pct_change().dropna()
            features['daily_return_1d'] = daily_returns.iloc[-1] * 100
        
        # Moving averages
        if len(closes) >= 20:
            features['sma_20d'] = closes.tail(20).mean()
            features['price_vs_sma_20d'] = (closes.iloc[-1] / features['sma_20d'] - 1) * 100
        if len(closes) >= 50:
            features['sma_50d'] = closes.tail(50).mean()
            features['price_vs_sma_50d'] = (closes.iloc[-1] / features['sma_50d'] - 1) * 100
        
        # Volatility
        if len(closes) >= 30:
            daily_returns = closes.pct_change().dropna()
            features['daily_volatility_30d'] = daily_returns.tail(30).std() * np.sqrt(252) * 100  # Annualized
    
    return features

def process_hybrid_data(ticker: str, hourly_data: pd.DataFrame, config: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process hourly data into both daily and intraday features.
    
    Args:
        ticker: Stock ticker symbol
        hourly_data: DataFrame with hourly OHLCV data
        config: Configuration dictionary
        
    Returns:
        Tuple of (daily_data, features_data)
    """
    if hourly_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Aggregate to daily
    daily_data = aggregate_hourly_to_daily(hourly_data)
    
    # Create combined features
    features_data = create_hybrid_features(hourly_data, daily_data)
    
    return daily_data, features_data
