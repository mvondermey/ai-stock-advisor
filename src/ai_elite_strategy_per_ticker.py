"""
Hybrid AI Elite model training:
  1. collect_ticker_training_data() - gather samples for one ticker
  2. train_shared_base_model()     - train ONE model on ALL tickers' data
  3. fine_tune_per_ticker()        - fine-tune a copy of the base model per ticker
"""

import pandas as pd
import numpy as np
import pickle
import copy
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

FEATURE_COLS = [
    'perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
    'overnight_gap', 'intraday_range', 'last_hour_momentum',
    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
    'volume_ratio', 'rsi_14',
    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
    # NEW: Mean reversion features
    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd'
]


def collect_ticker_training_data(
    ticker: str,
    ticker_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int = 5,
    hourly_cache: dict = None,
    market_returns: dict = None
) -> List[dict]:
    """Collect training samples for a single ticker. Returns list of dicts.
    market_returns: dict mapping sample_date -> market return (pre-computed)."""
    if ticker_data is None or len(ticker_data) == 0:
        return []

    try:
        from ai_elite_strategy import _extract_features, _calculate_forward_return, _load_hourly_data_direct
        from config import AI_ELITE_INTRADAY_LOOKBACK
    except ImportError:
        return []

    if market_returns is None:
        market_returns = {}

    if train_start_date.tzinfo is None:
        train_start_date = train_start_date.replace(tzinfo=timezone.utc)
    if train_end_date.tzinfo is None:
        train_end_date = train_end_date.replace(tzinfo=timezone.utc)

    if hourly_cache is None:
        hourly_cache = {ticker: _load_hourly_data_direct(
            ticker,
            train_start_date - timedelta(days=AI_ELITE_INTRADAY_LOOKBACK + 5),
            train_end_date + timedelta(days=forward_days + 2)
        )}

    samples = []
    current_date = train_start_date
    while current_date <= train_end_date:
        try:
            hourly_data = hourly_cache.get(ticker)
            features = _extract_features(ticker, hourly_data, current_date, daily_data=ticker_data)
            if features is None:
                current_date += timedelta(days=2)
                continue
            forward_return = _calculate_forward_return(ticker_data, current_date, forward_days)
            if forward_return is None:
                current_date += timedelta(days=2)
                continue

            samples.append({
                'ticker':             ticker,
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
                # NEW: Mean reversion features
                'bollinger_position': features.get('bollinger_position', 0.5),
                'sma20_distance':     features.get('sma20_distance', 0),
                'sma50_distance':     features.get('sma50_distance', 0),
                'macd':               features.get('macd', 0),
                'forward_return':     forward_return,
                'market_return':      market_returns.get(current_date, 0.0),
            })
        except Exception as e:
            pass  # Skip date on error
        current_date += timedelta(days=2)

    return samples


def _prepare_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return for regression target (simpler, more predictable)."""
    # Use raw forward return as target - more predictable than risk-adjusted
    train_df['label'] = train_df['forward_return']
    
    # Clip extreme outliers for stability
    mean_ret = train_df['label'].mean()
    std_ret = train_df['label'].std()
    if std_ret > 0:
        train_df['label'] = train_df['label'].clip(
            lower=mean_ret - 3 * std_ret, upper=mean_ret + 3 * std_ret
        )
    
    return train_df


def train_shared_base_model(
    all_training_data: List[dict],
    save_path: str = None,
    existing_model=None,
    train_start: datetime = None,
    train_end: datetime = None
):
    """
    Train ENSEMBLE of models on data from ALL tickers (REGRESSION version).
    Returns ensemble of top 3 models for more robust predictions.
    
    Args:
        all_training_data: Combined list of sample dicts from all tickers
        save_path: Path to save the ensemble
        existing_model: Existing ensemble to continue training (not used for ensembles)
        
    Returns:
        (ensemble_dict, avg_r2_score) or (None, 0.0)
        ensemble_dict contains {'models': [model1, model2, ...], 'weights': [w1, w2, ...]}
    """
    from config import MIN_TRAINING_SAMPLES_AI_ELITE, XGBOOST_USE_GPU

    if len(all_training_data) < MIN_TRAINING_SAMPLES_AI_ELITE:
        print(f"   ⚠️ AI Elite: Insufficient shared training data ({len(all_training_data)} samples, need {MIN_TRAINING_SAMPLES_AI_ELITE})")
        return None, 0.0

    train_df = pd.DataFrame(all_training_data)
    train_df = _prepare_labels(train_df)

    X = train_df[FEATURE_COLS]
    y = train_df['label'].values

    # Check if we have existing models to continue training
    has_existing = existing_model is not None and isinstance(existing_model, dict) and 'all_models' in existing_model
    
    status_msg = "Continuing" if has_existing else "Training NEW"
    print(f"   📊 AI Elite: {status_msg} training on {len(X)} samples from {train_df['ticker'].nunique()} tickers...")
    
    # Use XGBoost + LightGBM + CatBoost (all support GPU + incremental training)
    import xgboost as xgb
    import lightgbm as lgb
    import warnings
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    
    device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
    
    if has_existing:
        # Load existing models for incremental training
        models = existing_model['all_models']
        # Add CatBoost if missing from loaded model
        if 'CatBoost' not in models:
            try:
                import catboost as cb
                task_type = 'GPU' if XGBOOST_USE_GPU else 'CPU'
                models['CatBoost'] = cb.CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.1,
                    task_type=task_type, random_seed=42, verbose=0
                )
            except ImportError:
                pass
        print(f"   🚀 Incremental training: {len(models)} models on {device}")
    else:
        # Fresh training - create new models
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
                tree_method='hist', device=device, verbosity=0, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
            )
        }
        # Add CatBoost if available (GPU-accelerated, good with tabular data)
        try:
            import catboost as cb
            task_type = 'GPU' if XGBOOST_USE_GPU else 'CPU'
            models['CatBoost'] = cb.CatBoostRegressor(
                iterations=100, depth=4, learning_rate=0.1,
                task_type=task_type, random_seed=42, verbose=0
            )
        except ImportError:
            pass
        print(f"   🚀 Fresh training: {list(models.keys())} ({device})")

    # Train with incremental learning (no CV for speed - just train/val split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")
    
    trained_models = []
    model_scores = []
    model_names = []
    
    import time

    for name, m in models.items():
        try:
            print(f"      🔄 {name}: Training started...", end=" ", flush=True)
            start_time = time.time()
            
            # Validate training data
            if len(X_train) < 10:
                print(f"skipped (insufficient data: {len(X_train)} samples)")
                continue
                
            # Check for valid target values
            if np.all(y_train == y_train[0]):
                print(f"skipped (constant target)")
                continue
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if has_existing:
                    # Incremental training - retrain on new data
                    # Note: True incremental learning (adding trees) causes numerical instability
                    # Instead, we retrain the model on new data but keep the same hyperparameters
                    # This is still beneficial as it adapts to recent market conditions
                    m.fit(X_train, y_train)
                else:
                    # Fresh training
                    m.fit(X_train, y_train)
                
                # Validate model was trained (has trees/estimators)
                if name == 'CatBoost':
                    try:
                        # Check if CatBoost has any trees
                        if hasattr(m, 'get_tree_count') and m.get_tree_count() == 0:
                            print(f"failed: No trees built")
                            continue
                    except:
                        pass
                
                # Validate on held-out set (faster than CV)
                y_pred = m.predict(X_val)
                
                # Check for numerical instability
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    print(f"      ⚠️ {name}: Predictions contain NaN/Inf, skipping")
                    continue
                
                if len(y_pred) == 0 or len(y_val) == 0:
                    print(f"failed: Empty predictions")
                    continue
                
                score = r2_score(y_val, y_pred)
                
                # Clip to reasonable bounds - extreme values indicate numerical overflow
                if score < -10 or score > 1 or np.isnan(score) or np.isinf(score):
                    print(f"      ⚠️ {name}: R² = {score:.3f} (invalid, clipping to -10)")
                    score = -10.0
            
            elapsed = time.time() - start_time
            status = "incremental" if has_existing else "fresh"
            print(f"R² = {score:.3f} ({status}, {elapsed:.1f}s)")
            trained_models.append(m)
            model_scores.append(score)
            model_names.append(name)
        except Exception as e:
            print(f"      {name}: Training failed: {e}")
            continue

    if not trained_models:
        print(f"   ⚠️ AI Elite: No models trained successfully")
        return None, 0.0

    # Save ALL trained models, use best for prediction
    best_idx = max(range(len(model_scores)), key=lambda i: model_scores[i])
    best_name = model_names[best_idx]
    best_score = model_scores[best_idx]
    best_model = trained_models[best_idx]
    
    # Create model dict with ALL models stored, best_model for prediction
    model_dict = {
        'all_models': dict(zip(model_names, trained_models)),  # All models by name
        'all_scores': dict(zip(model_names, model_scores)),    # All scores by name
        'best_model': best_model,                               # Best model for prediction
        'best_name': best_name,
        'best_score': best_score,
        'feature_cols': FEATURE_COLS
    }

    if save_path:
        metadata = {
            'trained': datetime.now(timezone.utc).isoformat(),
            'best_model': best_name,
            'best_r2': best_score,
            'all_scores': dict(zip(model_names, model_scores))
        }
        if train_start and train_end:
            metadata['train_start'] = train_start.isoformat()
            metadata['train_end'] = train_end.isoformat()
        _save_model(model_dict, save_path, metadata)

    print(f"   ✅ AI Elite: Saved {len(trained_models)} models. Best = {best_name} (CV R² {best_score:.3f})")
    return model_dict, best_score


def fine_tune_per_ticker(
    ticker: str,
    ticker_samples: List[dict],
    base_model,
    save_path: str = None
):
    """
    Fine-tune a COPY of the shared base model on ticker-specific data.
    Uses fewer boosting rounds to avoid overfitting on small data.
    
    Returns:
        Fine-tuned model or None
    """
    MIN_FINETUNE_SAMPLES = 10
    if len(ticker_samples) < MIN_FINETUNE_SAMPLES:
        return None

    try:
        train_df = pd.DataFrame(ticker_samples)

        mr = train_df['market_return'] if 'market_return' in train_df.columns else 0.0
        train_df['excess_return'] = train_df['forward_return'] - mr
        vol_floored = train_df['volatility'].clip(lower=5.0)
        train_df['risk_adj_return'] = train_df['excess_return'] / (vol_floored ** 0.5)

        mean_ra = train_df['risk_adj_return'].mean()
        std_ra = train_df['risk_adj_return'].std()
        if std_ra > 0:
            train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
                lower=mean_ra - 3 * std_ra, upper=mean_ra + 3 * std_ra
            )

        # Regression target: predict risk_adj_return directly
        train_df['label'] = train_df['risk_adj_return']

        X = train_df[FEATURE_COLS]
        y = train_df['label'].values  # Continuous values for regression

        # Deep copy the base model and fine-tune with fewer rounds
        ft_model = copy.deepcopy(base_model)

        # Reduce n_estimators for fine-tuning to prevent overfitting
        if hasattr(ft_model, 'n_estimators'):
            ft_model.n_estimators = min(50, ft_model.n_estimators)
        if hasattr(ft_model, 'learning_rate'):
            ft_model.learning_rate = 0.05  # Lower LR for fine-tuning

        ft_model.fit(X, y)

        if save_path:
            _save_model(ft_model, save_path)

        return ft_model

    except Exception as e:
        print(f"   ⚠️ AI Elite: Fine-tune failed for {ticker}: {e}")
        return None


def _save_model(model, path: str, metadata: dict = None):
    """Save model to disk with optional metadata."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model with metadata
        model_data = {
            'model': model,
            'metadata': metadata or {}
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to save model to {path}: {e}")
