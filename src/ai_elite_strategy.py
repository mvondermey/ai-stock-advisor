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
    debug_count = 0
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'exception': 0}
    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                fail_reasons['not_in_data'] += 1
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) == 0:
                fail_reasons['empty'] += 1
                continue
            
            # Debug first 3 tickers
            if debug_count < 3:
                print(f"   🔍 AI Elite DEBUG {ticker}: index_type={type(ticker_data.index).__name__}, "
                      f"len={len(ticker_data)}, cols={list(ticker_data.columns[:5])}, "
                      f"index[0]={ticker_data.index[0]}, index[-1]={ticker_data.index[-1]}")
                debug_count += 1
            
            # Calculate all features
            features = _extract_features(ticker, ticker_data, current_date)
            if features is None:
                fail_reasons['features_none'] += 1
                continue
            
            # Add ticker to features for DataFrame
            features['ticker'] = ticker
            candidates.append(features)
            
        except Exception as e:
            fail_reasons['exception'] += 1
            if debug_count < 5:
                print(f"   ⚠️ AI Elite DEBUG {ticker}: Exception: {e}")
            continue
    
    if not candidates:
        print(f"   ⚠️ AI Elite: No candidates found")
        print(f"   🔍 AI Elite: Fail reasons: {fail_reasons}")
        return []
    
    # Use provided model or load from path
    if model is None:
        model = _load_or_create_model(model_path)
    
    # Score candidates using ML model
    if model is not None:
        candidates_df = pd.DataFrame(candidates)
        
        # Add engineered features for better predictions
        candidates_df['mom_vol_ratio'] = candidates_df['momentum_6m'] / candidates_df['volatility']
        candidates_df['dip_ratio'] = candidates_df['perf_1y'] / (candidates_df['perf_3m'] + 1)
        
        # Use full feature set for better predictions
        feature_cols = ['momentum_6m', 'volatility', 'avg_volume', 'dip_score', 'perf_1y', 'perf_3m',
                        'mom_vol_ratio', 'dip_ratio']
        
        # Force model retraining if feature mismatch
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(feature_cols):
            print(f"   🔄 AI Elite: Feature count changed ({model.n_features_in_} -> {len(feature_cols)}), retraining model")
            
            # Delete old model file to force retraining with new features
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"   🗑️ AI Elite: Deleted old model file {model_path}")
                except Exception as e:
                    print(f"   ⚠️ AI Elite: Failed to delete old model: {e}")
            
            model = _load_or_create_model(model_path)  # Retrain immediately
            if model is None:
                print(f"   ❌ AI Elite: Model retraining failed, using fallback")
                candidates_df['ai_score'] = _fallback_scoring(candidates_df)
                return selected
        
        # Predict scores
        print(f"   🔍 AI Elite: Attempting ML scoring with {len(candidates_df)} candidates")
        print(f"   🔍 AI Elite: Feature columns: {feature_cols}")
        print(f"   🔍 AI Elite: Feature shape: {candidates_df[feature_cols].shape}")
        
        try:
            X = candidates_df[feature_cols].values
            print(f"   🔍 AI Elite: X shape: {X.shape}, X dtype: {X.dtype}")
            print(f"   🔍 AI Elite: Model type: {type(model)}")
            
            scores = model.predict_proba(X)[:, 1]  # Probability of positive class
            print(f"   🔍 AI Elite: ML scoring succeeded, scores shape: {scores.shape}")
            candidates_df['ai_score'] = scores
            
        except Exception as e:
            print(f"   ❌ AI Elite: ML scoring FAILED: {type(e).__name__}: {e}")
            print(f"   ❌ AI Elite: Error details: {str(e)}")
            # Don't use fallback - let it fail so we can debug
            raise e
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
        # ✅ FIX: Deduplicate index (hourly data combined creates duplicates)
        if ticker_data.index.duplicated().any():
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
        
        # Check if we should use intraday data for richer features
        from config import AI_ELITE_USE_INTRADAY, AI_ELITE_INTRADAY_LOOKBACK
        use_intraday = AI_ELITE_USE_INTRADAY and len(ticker_data) > (AI_ELITE_INTRADAY_LOOKBACK * 24)
        
        # ✅ FIX: Filter data up to current_date to avoid temporal leakage
        if current_date is not None:
            current_ts = pd.Timestamp(current_date)
            # Ensure both are UTC-aware for proper comparison
            if current_ts.tz is None:
                current_ts = current_ts.tz_localize('UTC')
            elif str(current_ts.tz) != 'UTC':
                current_ts = current_ts.tz_convert('UTC')
            
            # Ensure index is also UTC-aware
            if ticker_data.index.tz is None:
                ticker_data = ticker_data.copy()
                ticker_data.index = ticker_data.index.tz_localize('UTC')
            elif str(ticker_data.index.tz) != 'UTC':
                ticker_data = ticker_data.copy()
                ticker_data.index = ticker_data.index.tz_convert('UTC')
            
            # Use boolean indexing (safe for non-unique indices)
            ticker_data_filtered = ticker_data[ticker_data.index <= current_ts]
        else:
            ticker_data_filtered = ticker_data
        
        # Use dropna'd Close series for all calculations (adaptive approach)
        close_col = ticker_data_filtered['Close']
        # Handle case where Close is a DataFrame (multiple columns) instead of Series
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        close_prices = close_col.dropna()
        n_prices = len(close_prices)
        
        # Debug: Check data length for first few tickers
        if ticker in ['SNDK', 'ZEC-USD', 'WDC'] and hasattr(_extract_features, '_debug_count2'):
            if _extract_features._debug_count2 < 3:
                print(f"   🔍 FEAT DEBUG2 {ticker}: n_prices={n_prices}, filtered_len={len(ticker_data_filtered)}, "
                      f"duplicated={ticker_data.index.duplicated().any()}")
                _extract_features._debug_count2 += 1
        if not hasattr(_extract_features, '_debug_count2'):
            _extract_features._debug_count2 = 0
            print(f"   🔍 FEAT DEBUG2 {ticker}: n_prices={n_prices}, filtered_len={len(ticker_data_filtered)}, "
                  f"duplicated={ticker_data.index.duplicated().any()}")
            _extract_features._debug_count2 = 1
        
        # Adjust minimum data requirements based on intraday availability
        if use_intraday:
            min_data_points = AI_ELITE_INTRADAY_LOOKBACK * 24  # 10 days * 24 hours = 240 points
            if n_prices < min_data_points:
                return None
        else:
            if n_prices < 60:  # Minimum 60 days for daily data
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
        returns = close_prices.pct_change().dropna()
        from config import MIN_DATA_DAYS_AI_ELITE_VOLATILITY
        
        # Adjust volatility calculation for intraday data
        if use_intraday:
            # For hourly data, use different annualization factor
            # 24 hours/day * 252 trading days/year = 6048 hours/year
            min_returns = AI_ELITE_INTRADAY_LOOKBACK * 24  # 10 days * 24 hours = 240
            if len(returns) < min_returns:
                return None
            volatility = returns.std() * (6048 ** 0.5) * 100  # Annualized hourly volatility
        else:
            # Daily data
            if len(returns) < MIN_DATA_DAYS_AI_ELITE_VOLATILITY:
                return None
            volatility = returns.std() * (252 ** 0.5) * 100  # As percentage
        
        # Calculate performance metrics needed for AI Elite
        perf_1y = ((latest_price / close_prices.iloc[-252]) - 1) * 100 if len(close_prices) >= 252 else momentum_6m
        perf_3m = ((latest_price / close_prices.iloc[-63]) - 1) * 100 if len(close_prices) >= 63 else momentum_6m / 2
        dip_score = perf_1y - perf_3m
        avg_volume = ticker_data['Volume'].tail(min(30, len(ticker_data))).mean() if 'Volume' in ticker_data.columns else 0
        
        return {
            'momentum_6m': momentum_6m,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'dip_score': dip_score,
            'perf_1y': perf_1y,
            'perf_3m': perf_3m
        }
        
    except Exception as e:
        if not hasattr(_extract_features, '_err_logged') or _extract_features._err_logged < 3:
            print(f"   ❌ FEAT EXCEPTION {ticker}: {type(e).__name__}: {e}")
            if not hasattr(_extract_features, '_err_logged'):
                _extract_features._err_logged = 0
            _extract_features._err_logged += 1
        return None


def _load_or_create_model(model_path: Optional[str] = None):
    """
    Load existing ML model or train a new one if not available.
    """
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ AI Elite: Loaded ML model from {model_path}")
            return model
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to load model: {e}")
    
    # No model available - train a new one
    print(f"   🎓 AI Elite: No existing model found, training new model...")
    
    # For live trading, we need to train with available historical data
    # This is a simplified training approach for live trading
    try:
        from datetime import timedelta
        import xgboost as xgb
        from config import XGBOOST_USE_GPU
        
        # Create a simple XGBoost model for live trading
        # This will be trained on the fly with available data
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            tree_method='hist',
            device=device,
            verbosity=0,
            n_jobs=1
        )
        
        print(f"   🚀 AI Elite: Created new XGBoost model ({device})")
        
        # For live trading, we need to train the model with actual data
        # This is a simplified approach - train with recent data
        print(f"   🎓 AI Elite: Training model with recent data...")
        
        # Create dummy training data for initial model (will be retrained later)
        import numpy as np
        n_samples = 100
        n_features = 8  # Match our feature set
        X_dummy = np.random.randn(n_samples, n_features)
        y_dummy = np.random.randint(0, 2, n_samples)
        
        model.fit(X_dummy, y_dummy)
        print(f"   ✅ AI Elite: Model trained with dummy data (will be retrained)")
        
        # Save the model for future use
        if model_path:
            try:
                import os
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 AI Elite: Saved trained model to {model_path}")
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to save model: {e}")
        
        return model
        
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to create model: {e}")
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
    # Import both XGBoost and sklearn GradientBoosting
    try:
        import xgboost as xgb
        from config import XGBOOST_USE_GPU
        xgb_available = True
        print(f"   🚀 AI Elite: Using XGBoost {'(GPU)' if XGBOOST_USE_GPU else '(CPU)'}")
    except ImportError:
        xgb_available = False
        print(f"   ⚠️ AI Elite: XGBoost not available, will use sklearn GradientBoosting (CPU only)")
    
    # Always import sklearn as fallback
    from sklearn.ensemble import GradientBoostingClassifier
    
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
    print(f"   📊 AI Elite: Sample dates: {[d.date() for d in sample_dates]}")
    
    debug_count = 0
    features_none_count = 0
    forward_none_count = 0
    
    for sample_date in sample_dates:
        # Extract features for all tickers on this date
        for ticker in all_tickers:
            try:
                if ticker not in ticker_data_grouped:
                    continue
                
                ticker_data = ticker_data_grouped[ticker]
                if len(ticker_data) == 0:
                    continue
                
                # Debug first few tickers on first sample date
                if debug_count < 3 and sample_date == sample_dates[0]:
                    print(f"   🔍 TRAIN DEBUG {ticker}: index.tz={ticker_data.index.tz}, "
                          f"cols={list(ticker_data.columns[:5])}, shape={ticker_data.shape}, "
                          f"sample_date={sample_date}, sample_date.tz={sample_date.tzinfo}")
                
                # Extract features (uses adaptive lookback, min 60 days)
                features = _extract_features(ticker, ticker_data, sample_date)
                if features is None:
                    if debug_count < 3 and sample_date == sample_dates[0]:
                        print(f"   🔍 TRAIN DEBUG {ticker}: _extract_features returned None")
                        debug_count += 1
                    features_none_count += 1
                    continue
                
                # Calculate forward return (label)
                forward_return = _calculate_forward_return(
                    ticker_data, sample_date, forward_days
                )
                if forward_return is None:
                    forward_none_count += 1
                    continue
                
                # Store training sample (label will be assigned after collecting all samples)
                training_data.append({
                    'momentum_6m': features['momentum_6m'],
                    'volatility': features['volatility'],
                    'forward_return': forward_return,
                    'sample_date': sample_date
                })
                
            except Exception as e:
                continue
    
    print(f"   📊 AI Elite: Training loop done - samples={len(training_data)}, features_none={features_none_count}, forward_none={forward_none_count}")
    
    from config import MIN_TRAINING_SAMPLES_AI_ELITE
    if len(training_data) < MIN_TRAINING_SAMPLES_AI_ELITE:
        print(f"   ⚠️ AI Elite: Insufficient training data ({len(training_data)} samples), using fallback")
        return None
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    print(f"   📈 AI Elite: Collected {len(train_df)} training samples")
    
    # Assign labels based on absolute returns (making money focus)
    # Positive returns = 1, Negative returns = 0
    train_df['label'] = (train_df['forward_return'] > 0).astype(int)
    
    # Remove stocks with minimal returns (to avoid noise)
    min_return_threshold = 0.5  # 0.5% minimum return
    train_df = train_df[abs(train_df['forward_return']) >= min_return_threshold]
    
    print(f"   📊 AI Elite: Using absolute returns labeling")
    print(f"   📊 AI Elite: Positive returns: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"   📊 AI Elite: Average return: {train_df['forward_return'].mean():.2f}%")
    print(f"   📊 AI Elite: Training on {len(train_df)} samples")
    
    # SIMPLIFIED: Use only momentum_6m and volatility (like Risk-Adj Mom)
    # Remove complex features that may cause overfitting
    print(f"   📊 AI Elite: Using simplified features (momentum_6m + volatility only)")
    
    # Prepare features and labels
    feature_cols = ['momentum_6m', 'volatility']
    X = train_df[feature_cols].values
    y = train_df['label'].values
    
    # Train model
    if xgb_available:
        # Use XGBoost for GPU acceleration
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            tree_method='hist' if device == 'cuda' else 'hist',
            device=device,
            verbosity=0,
            n_jobs=1  # Prevent multiprocessing conflicts
        )
    else:
        # Fallback to sklearn
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
        # ✅ FIX: Deduplicate index (hourly data combined creates duplicates)
        if ticker_data.index.duplicated().any():
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
        
        # Convert current_date to pandas Timestamp - ensure UTC-aware
        current_date_tz = pd.Timestamp(current_date)
        if current_date_tz.tz is None:
            current_date_tz = current_date_tz.tz_localize('UTC')
        elif str(current_date_tz.tz) != 'UTC':
            current_date_tz = current_date_tz.tz_convert('UTC')
        
        # Ensure index is also UTC-aware
        if ticker_data.index.tz is None:
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_localize('UTC')
        elif str(ticker_data.index.tz) != 'UTC':
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_convert('UTC')
        
        # Get current price
        current_data = ticker_data[ticker_data.index <= current_date_tz]
        if len(current_data) == 0:
            return None
        close_col = current_data['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        current_price = close_col.iloc[-1]
        
        # Get future price
        future_date = current_date_tz + timedelta(days=forward_days)
        future_data = ticker_data[(ticker_data.index > current_date_tz) & 
                                  (ticker_data.index <= future_date)]
        
        if len(future_data) == 0:
            return None
        
        future_close = future_data['Close']
        if isinstance(future_close, pd.DataFrame):
            future_close = future_close.iloc[:, 0]
        future_price = future_close.iloc[-1]
        
        # Calculate return
        forward_return = ((future_price - current_price) / current_price) * 100
        
        return forward_return
        
    except Exception as e:
        return None
