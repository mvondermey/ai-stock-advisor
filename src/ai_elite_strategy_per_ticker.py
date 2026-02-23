"""
Per-ticker AI Elite model training function.
Separated from main ai_elite_strategy.py to avoid circular imports.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta, timezone
from typing import Optional


def train_ai_elite_model_per_ticker(
    ticker: str,
    ticker_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    save_path: str = None,
    forward_days: int = 20,
    existing_model: any = None,
    hourly_cache: dict = None
):
    """
    Train ML model for a single ticker (or continue training existing model).
    
    Args:
        ticker: Single ticker symbol
        ticker_data: Historical price data for this ticker
        train_start_date: Start of training period
        train_end_date: End of training period
        save_path: Path to save trained model
        forward_days: Days ahead to predict (default 20)
        existing_model: Pre-trained model to continue training (optional)
        hourly_cache: Pre-loaded hourly data dict (optional, for parallel training)
        
    Returns:
        Trained model or None if training fails
    """
    if ticker_data is None or len(ticker_data) == 0:
        return None
    
    print(f"   🎓 AI Elite: Training model for {ticker}...")
    
    # Import functions from main module
    try:
        from ai_elite_strategy import _extract_features, _calculate_forward_return, _load_hourly_data_direct
        from config import AI_ELITE_INTRADAY_LOOKBACK, MIN_TRAINING_SAMPLES_AI_ELITE, XGBOOST_USE_GPU
    except ImportError as e:
        print(f"   ❌ AI Elite: Import error: {e}")
        return None
    
    # Use the same training logic as the main function but for single ticker
    try:
        # Import both XGBoost and sklearn GradientBoosting
        try:
            import xgboost as xgb
            xgb_available = True
            print(f"   🚀 AI Elite: Using XGBoost {'(GPU)' if XGBOOST_USE_GPU else '(CPU)'}")
        except ImportError:
            xgb_available = False
            print(f"   ⚠️ AI Elite: XGBoost not available, will use sklearn GradientBoosting (CPU only)")
        
        # Always import sklearn as fallback
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Ensure dates are timezone-aware
        if train_start_date.tzinfo is None:
            train_start_date = train_start_date.replace(tzinfo=timezone.utc)
        if train_end_date.tzinfo is None:
            train_end_date = train_end_date.replace(tzinfo=timezone.utc)
        
        # Collect training samples for this ticker
        training_data = []
        
        # Use pre-loaded hourly cache if provided (for parallel training)
        if hourly_cache is None:
            hourly_cache = {ticker: _load_hourly_data_direct(ticker, train_start_date - timedelta(days=AI_ELITE_INTRADAY_LOOKBACK + 5), train_end_date + timedelta(days=forward_days + 2))}
        
        # Sample dates from training period (every 2 days)
        current_date = train_start_date
        sample_dates = []
        while current_date <= train_end_date:
            sample_dates.append(current_date)
            current_date += timedelta(days=2)
        
        for sample_date in sample_dates:
            try:
                # Extract features for this ticker on this date
                hourly_data = hourly_cache.get(ticker)
                
                # Extract features using both data sources
                features = _extract_features(ticker, hourly_data, sample_date, daily_data=ticker_data)
                if features is None:
                    continue
                
                # Calculate forward return (label)
                forward_return = _calculate_forward_return(ticker_data, sample_date, forward_days)
                if forward_return is None:
                    continue
                
                # Store training sample
                training_data.append({
                    'perf_3m':            features['perf_3m'],
                    'perf_6m':            features['perf_6m'],
                    'perf_1y':            features['perf_1y'],
                    'volatility':         features['volatility'],
                    'avg_volume':         features['avg_volume'],
                    'overnight_gap':      features.get('overnight_gap', 0),
                    'intraday_range':     features.get('intraday_range', 0),
                    'last_hour_momentum': features.get('last_hour_momentum', 0),
                    'risk_adj_score':     features.get('risk_adj_score', 0),
                    'dip_score':          features.get('dip_score', 0),
                    'mom_accel':          features.get('mom_accel', 0),
                    'vol_sweet_spot':     features.get('vol_sweet_spot', 0),
                    'volume_ratio':       features.get('volume_ratio', 1.0),
                    'rsi_14':             features.get('rsi_14', 50.0),
                    'short_term_reversal': features.get('short_term_reversal', 0),
                    'volume_sentiment':   features.get('volume_sentiment', 0),
                    'risk_adj_mom_3m':    features.get('risk_adj_mom_3m', 0),
                    'forward_return':     forward_return,
                    'market_return':      0.0,  # Single ticker, no market comparison
                    'sample_date':        sample_date
                })
                
            except Exception as e:
                continue
        
        # Per-ticker needs fewer samples than shared model (100 is for all tickers combined)
        # With 90-day lookback sampling every 2 days, ~45 samples per ticker
        MIN_PER_TICKER_SAMPLES = 10
        if len(training_data) < MIN_PER_TICKER_SAMPLES:
            print(f"   ⚠️ AI Elite: Insufficient training data for {ticker} ({len(training_data)} samples, need {MIN_PER_TICKER_SAMPLES})")
            return None
        
        # Convert to DataFrame
        train_df = pd.DataFrame(training_data)
        
        # Compute risk-adjusted return for ordinal ranking
        vol_floored = train_df['volatility'].clip(lower=5.0)
        train_df['risk_adj_return'] = train_df['forward_return'] / (vol_floored ** 0.5)
        
        # Clip extreme outliers
        mean_ra = train_df['risk_adj_return'].mean()
        std_ra = train_df['risk_adj_return'].std()
        if std_ra > 0:
            train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
                lower=mean_ra - 3 * std_ra, upper=mean_ra + 3 * std_ra
            )
        
        # Create ordinal labels (adaptive bins based on sample count)
        n_bins = min(5, len(train_df) // 3)  # At least 3 samples per bin
        n_bins = max(2, n_bins)  # At least 2 bins
        train_df['label'] = pd.qcut(train_df['risk_adj_return'], 
                                    q=n_bins, 
                                    labels=list(range(n_bins)),
                                    duplicates='drop').astype(int)
        
        # Prepare features and labels
        feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                        'overnight_gap', 'intraday_range', 'last_hour_momentum',
                        'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                        'volume_ratio', 'rsi_14',
                        'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m']
        X = train_df[feature_cols].values
        y = train_df['label'].values
        
        # Build candidate models
        candidates = {}
        
        if xgb_available:
            device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
            candidates['XGBoost'] = xgb.XGBClassifier(
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
        
        candidates['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=0
        )
        
        # Add LightGBM (CPU)
        try:
            import lightgbm as lgb
            candidates['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
        except ImportError:
            pass
        
        # Use existing model if provided, otherwise train new model
        if existing_model is not None:
            print(f"   🔄 AI Elite: Continuing training {ticker} with existing model")
            best_model = existing_model
            best_name = type(existing_model).__name__
            
            # Continue training on new data
            best_model.fit(X, y)
            
            # Measure actual kappa on training data
            from sklearn.metrics import cohen_kappa_score
            y_pred = best_model.predict(X)
            best_score = cohen_kappa_score(y, y_pred, weights='quadratic')
        else:
            print(f"   🆕 AI Elite: Training new model for {ticker}")
            
            # Cross-validate each model and pick the best
            from sklearn.metrics import cohen_kappa_score, make_scorer
            from sklearn.model_selection import cross_val_score
            import warnings
            
            def weighted_kappa(y_true, y_pred):
                return cohen_kappa_score(y_true, y_pred, weights='quadratic')
            kappa_scorer = make_scorer(weighted_kappa)
            
            best_model = None
            best_name = None
            best_score = -1.0
            cv_folds = 2  # Only 2 folds for small per-ticker datasets (~11 samples)
            
            for name, m in candidates.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = cross_val_score(m, X, y, cv=cv_folds, scoring=kappa_scorer, n_jobs=1)
                    mean_score = scores.mean()
                    if mean_score > best_score:
                        best_score = mean_score
                        best_name = name
                        best_model = m
                except Exception as e:
                    continue
            
            if best_model is None:
                best_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=42, verbose=0
                )
                best_name = 'GradientBoosting'
            
            # Fit best model on full training data
            best_model.fit(X, y)
        
        # Save model if path provided
        if save_path:
            try:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"   💾 AI Elite: Saved {ticker} model to {save_path}")
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to save {ticker} model: {e}")
        
        print(f"   ✅ AI Elite: {ticker} model trained! Best: {best_name} (kappa {best_score:.3f})")
        return best_model
        
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to train {ticker} model: {e}")
        return None
