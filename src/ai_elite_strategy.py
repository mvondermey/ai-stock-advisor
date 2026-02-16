"""
AI Elite Strategy: ML-powered scoring of elite stock candidates

Uses machine learning to learn optimal scoring from:
- 6M momentum
- Volatility (risk)
- Volume (liquidity)
- Dip score (1Y/3M ratio)
- 1Y performance
- 3M performance

The ML model learns which combinations of these features predict future outperformance,
discovering non-linear relationships that fixed formulas miss.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
import pickle
import os
from pathlib import Path

def select_ai_elite_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    model_path: str = None,
    model = None
) -> List[str]:
    """
    AI Elite Strategy: ML-based scoring of momentum + dip opportunities
    
    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        model_path: Path to saved ML model (optional)
        model: Pre-trained model object (optional, takes precedence over model_path)
        
    Returns:
        List of selected ticker symbols
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite"
    )
    
    candidates = []
    
    print(f"   🤖 AI Elite: Analyzing {len(filtered_tickers)} tickers with ML scoring (filtered from {len(all_tickers)})")
    
    # Extract features for all candidates
    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) == 0:
                continue
            
            # Calculate all features
            features = _extract_features(ticker, ticker_data, current_date)
            if features is None:
                continue
            
            candidates.append(features)
            
        except Exception as e:
            continue
    
    if not candidates:
        print(f"   ⚠️ AI Elite: No candidates found")
        return []
    
    # Use provided model or load from path
    if model is None:
        model = _load_or_create_model(model_path)
    
    # Score candidates using ML model
    if model is not None:
        candidates_df = pd.DataFrame(candidates)
        
        # Add engineered features (same as training)
        candidates_df['mom_vol_ratio'] = candidates_df['momentum_6m'] / candidates_df['volatility']
        candidates_df['dip_ratio'] = candidates_df['perf_1y'] / (candidates_df['perf_3m'] + 1)
        
        feature_cols = ['momentum_6m', 'volatility', 'avg_volume', 'dip_score', 'perf_1y', 'perf_3m',
                        'mom_vol_ratio', 'dip_ratio']
        
        # Predict scores
        X = candidates_df[feature_cols].values
        try:
            # Model predicts probability of outperformance
            scores = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
            candidates_df['ai_score'] = scores
        except:
            # Fallback to product-based scoring if ML fails
            print(f"   ⚠️ AI Elite: ML scoring failed, using fallback")
            candidates_df['ai_score'] = _fallback_scoring(candidates_df)
    else:
        # No model available, use fallback
        candidates_df = pd.DataFrame(candidates)
        candidates_df['ai_score'] = _fallback_scoring(candidates_df)
    
    # Sort by AI score
    candidates_df = candidates_df.sort_values('ai_score', ascending=False)
    
    # Debug: show top candidates
    print(f"   ✅ AI Elite: Found {len(candidates_df)} candidates")
    for i, row in candidates_df.head(5).iterrows():
        print(f"      {i+1}. {row['ticker']}: AI Score={row['ai_score']:.3f}, "
              f"6M={row['momentum_6m']:+.1f}%, Vol={row['volatility']:.1f}%, "
              f"1Y={row['perf_1y']:+.1f}%, 3M={row['perf_3m']:+.1f}%, Dip={row['dip_score']:.1f}")
    
    # Return top N tickers
    selected = candidates_df.head(top_n)['ticker'].tolist()
    return selected


def _extract_features(ticker: str, ticker_data: pd.DataFrame, current_date: datetime) -> Optional[Dict]:
    """
    Extract ML features from ticker data using adaptive lookback.
    
    Features:
    - momentum_6m: 6-month return percentage
    - volatility: Annualized volatility
    - avg_volume: Average daily volume
    - dip_score: 1Y performance - 3M performance
    - perf_1y: 1-year return percentage
    - perf_3m: 3-month return percentage
    """
    try:
        # ✅ FIX: Filter data up to current_date to avoid temporal leakage
        if current_date is not None:
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
            ticker_data_filtered = ticker_data.loc[:current_ts]
        else:
            ticker_data_filtered = ticker_data
        
        # Use dropna'd Close series for all calculations (adaptive approach)
        close_prices = ticker_data_filtered['Close'].dropna()
        n_prices = len(close_prices)
        
        if n_prices < 60:  # Minimum 60 days
            return None
        
        # Get latest price
        latest_price = close_prices.iloc[-1]
        if latest_price <= 0:
            return None
        
        # Calculate 6M performance (adaptive: use up to 126 trading days)
        lookback_6m = min(126, n_prices - 1)
        if lookback_6m < 40:
            return None
        price_6m_ago = close_prices.iloc[-lookback_6m]
        if price_6m_ago <= 0:
            return None
        
        # Feature 1: 6M momentum
        momentum_6m = ((latest_price - price_6m_ago) / price_6m_ago) * 100
        
        # Feature 2: Volatility (annualized)
        daily_returns = close_prices.pct_change().dropna()
        from config import MIN_DATA_DAYS_AI_ELITE_VOLATILITY
        if len(daily_returns) < MIN_DATA_DAYS_AI_ELITE_VOLATILITY:
            return None
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # As percentage
        
        # Feature 3: Average volume
        avg_volume = ticker_data['Volume'].dropna().mean() if 'Volume' in ticker_data.columns else 100000
        
        # Calculate 3M performance (adaptive: use up to 63 trading days)
        lookback_3m = min(63, n_prices - 1)
        if lookback_3m < 10:
            return None
        price_3m_ago = close_prices.iloc[-lookback_3m]
        if price_3m_ago <= 0:
            return None
        perf_3m = ((latest_price - price_3m_ago) / price_3m_ago) * 100
        
        # Calculate 1Y performance (adaptive: use up to 252 trading days)
        lookback_1y = min(252, n_prices - 1)
        if lookback_1y < 60:
            return None
        price_1y_ago = close_prices.iloc[-lookback_1y]
        if price_1y_ago <= 0:
            return None
        perf_1y = ((latest_price - price_1y_ago) / price_1y_ago) * 100
        
        # Feature 6: Dip score
        dip_score = max(perf_1y - perf_3m, 0)
        
        # NO HARD FILTERS - let ML decide what's good
        
        return {
            'ticker': ticker,
            'momentum_6m': momentum_6m,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'dip_score': dip_score,
            'perf_1y': perf_1y,
            'perf_3m': perf_3m
        }
        
    except Exception as e:
        return None


def _load_or_create_model(model_path: Optional[str] = None):
    """
    Load existing ML model or return None to use fallback scoring.
    
    In future iterations, this will train a model on historical data.
    For now, we use fallback scoring until we have training data.
    """
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ AI Elite: Loaded ML model from {model_path}")
            return model
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to load model: {e}")
            return None
    
    # No model available yet - will use fallback scoring
    # In future: implement walk-forward training here
    return None


def _fallback_scoring(candidates_df: pd.DataFrame) -> np.ndarray:
    """
    Fallback scoring when ML model is not available.
    Uses Elite Hybrid's proven additive bonus formula.
    """
    # Calculate mom-vol score (momentum/volatility)
    mom_vol_score = candidates_df['momentum_6m'] / candidates_df['volatility']
    
    # Calculate dip ratio (1Y / 3M)
    dip_ratio = np.where(
        candidates_df['perf_3m'] > 0,
        candidates_df['perf_1y'] / candidates_df['perf_3m'],
        np.maximum(candidates_df['perf_1y'] / 10, 0.1)
    )
    
    # Normalize dip_ratio to 0.5-5.0 range
    dip_ratio = np.clip(dip_ratio, 0.5, 5.0)
    
    # Elite Hybrid formula: additive bonus
    scores = mom_vol_score * (1 + dip_ratio * 0.3)
    
    # Bonus for low volatility (< 50%)
    low_vol_bonus = np.where(candidates_df['volatility'] < 50, 1.1, 1.0)
    scores = scores * low_vol_bonus
    
    return scores


def train_ai_elite_model(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    save_path: str = None,
    forward_days: int = 20
):
    """
    Train ML model to predict stock outperformance based on features.
    
    Uses walk-forward approach:
    1. For each date in training period, extract features for all stocks
    2. Label stocks based on their performance over next forward_days
    3. Train GradientBoostingClassifier to predict outperformance
    
    Args:
        ticker_data_grouped: Historical ticker data
        all_tickers: List of tickers to train on
        train_start_date: Start of training period
        train_end_date: End of training period
        save_path: Path to save trained model
        forward_days: Days ahead to predict (default 20)
        
    Returns:
        Trained model or None if training fails
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    
    print(f"   🎓 AI Elite: Training ML model on {train_start_date.date()} to {train_end_date.date()}...")
    
    # Ensure dates are timezone-aware
    if train_start_date.tzinfo is None:
        train_start_date = train_start_date.replace(tzinfo=timezone.utc)
    if train_end_date.tzinfo is None:
        train_end_date = train_end_date.replace(tzinfo=timezone.utc)
    
    # Collect training samples
    training_data = []
    
    # Sample dates from training period (every 5 days to reduce computation)
    current_date = train_start_date
    sample_dates = []
    while current_date <= train_end_date:
        sample_dates.append(current_date)
        current_date += timedelta(days=5)
    
    print(f"   📊 AI Elite: Sampling {len(sample_dates)} dates for training...")
    
    for sample_date in sample_dates:
        # Extract features for all tickers on this date
        for ticker in all_tickers:
            try:
                if ticker not in ticker_data_grouped:
                    continue
                
                ticker_data = ticker_data_grouped[ticker]
                if len(ticker_data) == 0:
                    continue
                
                # Extract features (uses adaptive lookback, min 60 days)
                features = _extract_features(ticker, ticker_data, sample_date)
                if features is None:
                    continue
                
                # Calculate forward return (label)
                forward_return = _calculate_forward_return(
                    ticker_data, sample_date, forward_days
                )
                if forward_return is None:
                    continue
                
                # Store training sample (label will be assigned after collecting all samples)
                training_data.append({
                    'momentum_6m': features['momentum_6m'],
                    'volatility': features['volatility'],
                    'avg_volume': features['avg_volume'],
                    'dip_score': features['dip_score'],
                    'perf_1y': features['perf_1y'],
                    'perf_3m': features['perf_3m'],
                    'forward_return': forward_return,
                    'sample_date': sample_date
                })
                
            except Exception as e:
                continue
    
    from config import MIN_TRAINING_SAMPLES_AI_ELITE
    if len(training_data) < MIN_TRAINING_SAMPLES_AI_ELITE:
        print(f"   ⚠️ AI Elite: Insufficient training data ({len(training_data)} samples), using fallback")
        return None
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    print(f"   📈 AI Elite: Collected {len(train_df)} training samples")
    
    # Assign labels based on ranking within each date
    # Top 20% performers = 1, Bottom 20% = 0, Middle 60% excluded
    labeled_samples = []
    for sample_date in train_df['sample_date'].unique():
        date_samples = train_df[train_df['sample_date'] == sample_date].copy()
        
        # Sort by forward return
        date_samples = date_samples.sort_values('forward_return', ascending=False)
        n_samples = len(date_samples)
        
        # Top 20% get label 1
        top_20_pct = int(n_samples * 0.2)
        date_samples.iloc[:top_20_pct, date_samples.columns.get_loc('forward_return')] = 1
        
        # Bottom 20% get label 0
        bottom_20_pct = int(n_samples * 0.2)
        date_samples.iloc[-bottom_20_pct:, date_samples.columns.get_loc('forward_return')] = 0
        
        # Keep only top and bottom (exclude middle 60%)
        labeled_samples.append(date_samples.iloc[:top_20_pct])
        labeled_samples.append(date_samples.iloc[-bottom_20_pct:])
    
    train_df = pd.concat(labeled_samples, ignore_index=True)
    train_df['label'] = train_df['forward_return'].astype(int)
    
    print(f"   📊 AI Elite: Training on {len(train_df)} samples (top/bottom 20% only)")
    print(f"   📊 AI Elite: Positive labels: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    
    # Add engineered features
    train_df['mom_vol_ratio'] = train_df['momentum_6m'] / train_df['volatility']
    train_df['dip_ratio'] = train_df['perf_1y'] / (train_df['perf_3m'] + 1)  # +1 to avoid div by zero
    
    # Prepare features and labels
    feature_cols = ['momentum_6m', 'volatility', 'avg_volume', 'dip_score', 'perf_1y', 'perf_3m', 
                    'mom_vol_ratio', 'dip_ratio']
    X = train_df[feature_cols].values
    y = train_df['label'].values
    
    # Train model
    try:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        model.fit(X, y)
        
        # Calculate training accuracy
        train_accuracy = model.score(X, y)
        print(f"   ✅ AI Elite: Model trained! Accuracy: {train_accuracy*100:.1f}%")
        
        # Show feature importances
        importances = model.feature_importances_
        for i, col in enumerate(feature_cols):
            print(f"      {col}: {importances[i]:.3f}")
        
        # Save model if path provided
        if save_path:
            try:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 AI Elite: Model saved to {save_path}")
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to save model: {e}")
        
        return model
        
    except Exception as e:
        print(f"   ⚠️ AI Elite: Training failed: {e}")
        return None


def _calculate_forward_return(
    ticker_data: pd.DataFrame,
    current_date: datetime,
    forward_days: int
) -> Optional[float]:
    """
    Calculate forward return for a stock over next N days.
    
    Args:
        ticker_data: Historical price data for ticker
        current_date: Current date
        forward_days: Number of days to look ahead
        
    Returns:
        Forward return percentage or None if insufficient data
    """
    try:
        # Convert current_date to pandas Timestamp with timezone
        current_date_tz = pd.Timestamp(current_date)
        if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
            if current_date_tz.tz is None:
                current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
            else:
                current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)
        
        # Get current price
        current_data = ticker_data[ticker_data.index <= current_date_tz]
        if len(current_data) == 0:
            return None
        current_price = current_data['Close'].iloc[-1]
        
        # Get future price
        future_date = current_date_tz + timedelta(days=forward_days)
        future_data = ticker_data[(ticker_data.index > current_date_tz) & 
                                  (ticker_data.index <= future_date)]
        
        if len(future_data) == 0:
            return None
        
        future_price = future_data['Close'].iloc[-1]
        
        # Calculate return
        forward_return = ((future_price - current_price) / current_price) * 100
        
        return forward_return
        
    except Exception as e:
        return None
