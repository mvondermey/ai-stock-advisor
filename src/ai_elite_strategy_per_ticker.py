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
    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m'
]


def collect_ticker_training_data(
    ticker: str,
    ticker_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int = 5,
    hourly_cache: dict = None
) -> List[dict]:
    """Collect training samples for a single ticker. Returns list of dicts."""
    if ticker_data is None or len(ticker_data) == 0:
        return []

    try:
        from ai_elite_strategy import _extract_features, _calculate_forward_return, _load_hourly_data_direct
        from config import AI_ELITE_INTRADAY_LOOKBACK
    except ImportError:
        return []

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
                'forward_return':     forward_return,
            })
        except Exception:
            pass
        current_date += timedelta(days=2)

    return samples


def _prepare_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk-adjusted return and ordinal labels."""
    vol_floored = train_df['volatility'].clip(lower=5.0)
    train_df['risk_adj_return'] = train_df['forward_return'] / (vol_floored ** 0.5)

    mean_ra = train_df['risk_adj_return'].mean()
    std_ra = train_df['risk_adj_return'].std()
    if std_ra > 0:
        train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
            lower=mean_ra - 3 * std_ra, upper=mean_ra + 3 * std_ra
        )

    n_bins = 5  # Full 5 quintiles for shared model (thousands of samples)
    train_df['label'] = pd.qcut(
        train_df['risk_adj_return'], q=n_bins,
        labels=list(range(n_bins)), duplicates='drop'
    ).astype(int)
    return train_df


def train_shared_base_model(
    all_training_data: List[dict],
    save_path: str = None,
    existing_model=None
):
    """
    Train ONE shared base model on data from ALL tickers.
    
    Args:
        all_training_data: Combined list of sample dicts from all tickers
        save_path: Path to save the shared base model
        existing_model: Existing base model to continue training
        
    Returns:
        (model, kappa_score) or (None, 0.0)
    """
    from config import MIN_TRAINING_SAMPLES_AI_ELITE, XGBOOST_USE_GPU

    if len(all_training_data) < MIN_TRAINING_SAMPLES_AI_ELITE:
        print(f"   ⚠️ AI Elite: Insufficient shared training data ({len(all_training_data)} samples, need {MIN_TRAINING_SAMPLES_AI_ELITE})")
        return None, 0.0

    train_df = pd.DataFrame(all_training_data)
    train_df = _prepare_labels(train_df)

    X = train_df[FEATURE_COLS]
    y = train_df['label'].values

    print(f"   📊 AI Elite: Shared base model training on {len(X)} samples from {train_df['ticker'].nunique()} tickers...")

    # Build candidate models
    candidates = {}
    try:
        import xgboost as xgb
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        candidates['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
            tree_method='hist', device=device, verbosity=0, n_jobs=1
        )
    except ImportError:
        pass

    from sklearn.ensemble import GradientBoostingClassifier
    candidates['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42, verbose=0
    )

    try:
        import lightgbm as lgb
        candidates['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=-1
        )
    except ImportError:
        pass

    # If continuing from existing model, just refit on new data
    if existing_model is not None:
        print(f"   🔄 AI Elite: Continuing shared base model training")
        existing_model.fit(X, y)
        from sklearn.metrics import cohen_kappa_score
        y_pred = existing_model.predict(X)
        score = cohen_kappa_score(y, y_pred, weights='quadratic')
        if save_path:
            _save_model(existing_model, save_path)
        print(f"   ✅ AI Elite: Shared base model updated (kappa {score:.3f})")
        return existing_model, score

    # Cross-validate to pick best model type
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import cross_val_score
    import warnings

    def weighted_kappa(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    kappa_scorer = make_scorer(weighted_kappa)

    best_model = None
    best_name = None
    best_score = -1.0
    cv_folds = min(5, max(2, len(np.unique(y))))

    for name, m in candidates.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(m, X, y, cv=cv_folds, scoring=kappa_scorer, n_jobs=1)
            mean_score = scores.mean()
            print(f"      {name}: CV kappa = {mean_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_model = m
        except Exception:
            continue

    if best_model is None:
        best_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=0
        )
        best_name = 'GradientBoosting'

    best_model.fit(X, y)

    if save_path:
        _save_model(best_model, save_path)

    print(f"   ✅ AI Elite: Shared base model trained! Best: {best_name} (CV kappa {best_score:.3f})")
    return best_model, best_score


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

        vol_floored = train_df['volatility'].clip(lower=5.0)
        train_df['risk_adj_return'] = train_df['forward_return'] / (vol_floored ** 0.5)

        mean_ra = train_df['risk_adj_return'].mean()
        std_ra = train_df['risk_adj_return'].std()
        if std_ra > 0:
            train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
                lower=mean_ra - 3 * std_ra, upper=mean_ra + 3 * std_ra
            )

        # Use same 5 bins as shared model for label compatibility
        n_bins = 5
        try:
            train_df['label'] = pd.qcut(
                train_df['risk_adj_return'], q=n_bins,
                labels=list(range(n_bins)), duplicates='drop'
            ).astype(int)
        except ValueError:
            # Not enough unique values for 5 bins, use fewer
            n_bins = max(2, train_df['risk_adj_return'].nunique())
            train_df['label'] = pd.qcut(
                train_df['risk_adj_return'], q=n_bins,
                labels=list(range(n_bins)), duplicates='drop'
            ).astype(int)

        X = train_df[FEATURE_COLS]
        y = train_df['label'].values

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


def _save_model(model, path: str):
    """Save model to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to save model to {path}: {e}")
