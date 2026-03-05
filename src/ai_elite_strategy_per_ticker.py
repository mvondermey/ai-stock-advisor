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
        except Exception:
            pass
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

    print(f"   📊 AI Elite: Training model ensemble on {len(X)} samples from {train_df['ticker'].nunique()} tickers...")

    # Build ALL available models for ensemble
    models = {}
    try:
        import xgboost as xgb
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
            tree_method='hist', device=device, verbosity=0, n_jobs=-1
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
        )
    except ImportError:
        pass

    # Always include sklearn models
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    
    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, verbose=0
    )
    
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    
    models['Ridge'] = Ridge(alpha=1.0, random_state=42)

    # Train all models and evaluate with cross-validation
    from sklearn.metrics import r2_score, make_scorer
    from sklearn.model_selection import cross_val_score
    import warnings

    r2_scorer = make_scorer(r2_score)
    cv_folds = 3
    
    trained_models = []
    model_scores = []
    model_names = []

    for name, m in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Train on full data
                m.fit(X, y)
                # Cross-validate
                scores = cross_val_score(m, X, y, cv=cv_folds, scoring=r2_scorer, n_jobs=1)
            mean_score = scores.mean()
            print(f"      {name}: CV R² = {mean_score:.3f}")
            trained_models.append(m)
            model_scores.append(mean_score)
            model_names.append(name)
        except Exception as e:
            print(f"      {name}: Training failed: {e}")
            continue

    if not trained_models:
        print(f"   ⚠️ AI Elite: No models trained successfully")
        return None, 0.0

    # Use BEST model only (not ensemble mean)
    # Sort by score descending and pick the best one
    best_idx = max(range(len(model_scores)), key=lambda i: model_scores[i])
    best_model = trained_models[best_idx]
    best_score = model_scores[best_idx]
    best_name = model_names[best_idx]
    
    # Create model dict (single model, not ensemble)
    model_dict = {
        'models': [best_model],
        'weights': [1.0],
        'names': [best_name],
        'feature_cols': FEATURE_COLS
    }

    if save_path:
        metadata = {
            'trained': datetime.now(timezone.utc).isoformat(),
            'best_model': best_name,
            'best_r2': best_score
        }
        if train_start and train_end:
            metadata['train_start'] = train_start.isoformat()
            metadata['train_end'] = train_end.isoformat()
        _save_model(model_dict, save_path, metadata)

    print(f"   ✅ AI Elite: Best model = {best_name} (CV R² {best_score:.3f})")
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
