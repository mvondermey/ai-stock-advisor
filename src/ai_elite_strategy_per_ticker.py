"""
Hybrid AI Elite model training:
  1. collect_ticker_training_data() - gather samples for one ticker
  2. train_shared_base_model()     - train ONE model on ALL tickers' data
  3. fine_tune_per_ticker()        - fine-tune a copy of the base model per ticker
"""

import pandas as pd
import numpy as np
import copy
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

from model_training_safety import (
    restore_native_model_artifacts,
    save_native_model_artifacts,
)

FEATURE_COLS = [
    'perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
    'overnight_gap', 'intraday_range', 'last_hour_momentum',
    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
    'volume_ratio', 'rsi_14',
    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
    # NEW: Mean reversion features
    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd'
]


class IncrementalScaledSGDRegressor:
    """Standardized SGD regressor with true partial_fit continuation."""

    def __init__(
        self,
        *,
        penalty: str,
        alpha: float,
        l1_ratio: float = 0.15,
        random_state: int = 42,
    ):
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss='squared_error',
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=1,
            tol=None,
            learning_rate='invscaling',
            eta0=0.01,
        )
        self._is_fitted = False

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, 'to_numpy'):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def partial_fit(self, X, y):
        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=float)
        self.scaler.partial_fit(X_np)
        X_scaled = self.scaler.transform(X_np)
        self.model.partial_fit(X_scaled, y_np)
        self._is_fitted = True
        return self

    def fit(self, X, y):
        return self.partial_fit(X, y)

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("IncrementalScaledSGDRegressor is not fitted")
        X_np = self._to_numpy(X)
        X_scaled = self.scaler.transform(X_np)
        return self.model.predict(X_scaled)


def _fresh_ensemble_model(name: str, device: str):
    """Create a fresh ensemble member with stable defaults."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from config import AI_ELITE_CATBOOST_USED_RAM_LIMIT

    if name == 'XGBoost':
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', device=device, verbosity=0, n_jobs=-1
        )
    if name == 'LightGBM':
        return lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
        )
    if name == 'CatBoost':
        import catboost as cb

        catboost_params = dict(
            iterations=100, depth=4, learning_rate=0.1,
            task_type='CPU', random_seed=42, verbose=0,
            allow_writing_files=False, thread_count=1
        )
        if AI_ELITE_CATBOOST_USED_RAM_LIMIT:
            catboost_params['used_ram_limit'] = AI_ELITE_CATBOOST_USED_RAM_LIMIT
        return cb.CatBoostRegressor(**catboost_params)
    if name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    if name == 'Ridge':
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )
    if name == 'ElasticNet':
        return make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
        )
    if name == 'SGDRegressor-L2':
        return IncrementalScaledSGDRegressor(
            penalty='l2',
            alpha=0.0001,
            random_state=42,
        )
    if name == 'SGDRegressor-ElasticNet':
        return IncrementalScaledSGDRegressor(
            penalty='elasticnet',
            alpha=0.0001,
            l1_ratio=0.15,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


def _predictions_are_unstable(predictions) -> bool:
    """Detect obviously broken model output after incremental continuation."""
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        return True
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return True
    return np.max(np.abs(preds)) > 1e10


def _configure_catboost_continuation(model):
    """Apply safer runtime settings before CatBoost continuation training."""
    from config import AI_ELITE_CATBOOST_USED_RAM_LIMIT

    if not hasattr(model, '_init_params'):
        return
    model._init_params['task_type'] = 'CPU'
    model._init_params['thread_count'] = 1
    model._init_params['allow_writing_files'] = False
    if AI_ELITE_CATBOOST_USED_RAM_LIMIT:
        model._init_params['used_ram_limit'] = AI_ELITE_CATBOOST_USED_RAM_LIMIT


def _catboost_has_trained_trees(model) -> bool:
    """Continuation requires an already-trained CatBoost model."""
    try:
        # CatBoost >= 1.2 uses tree_count_ property
        if hasattr(model, 'tree_count_'):
            return model.tree_count_ > 0
        # Older CatBoost versions use get_tree_count() method
        if hasattr(model, 'get_tree_count'):
            return model.get_tree_count() > 0
        # Fallback: check is_fitted()
        if hasattr(model, 'is_fitted'):
            return model.is_fitted()
        return False
    except Exception:
        return False


def restore_catboost_sidecar(model, path: str):
    """Reload any native model sidecars saved beside the shared-base checkpoint."""
    return restore_native_model_artifacts(model, path)


def _order_models_for_training(models: Dict[str, object]) -> Dict[str, object]:
    """Train CatBoost first so crashes are easier to attribute."""
    preferred_order = (
        "CatBoost",
        "XGBoost",
        "LightGBM",
        "ExtraTrees",
        "RandomForest",
        "Ridge",
        "ElasticNet",
        "SGDRegressor-L2",
        "SGDRegressor-ElasticNet",
    )
    ordered = {name: models[name] for name in preferred_order if name in models}
    for name, model in models.items():
        if name not in ordered:
            ordered[name] = model
    return ordered


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
    train_df['label'] = train_df['forward_return'].copy()

    # First, hard clip to reasonable bounds (e.g., -100% to +200% forward return)
    # This prevents extreme outliers from crypto/penny stocks from destabilizing training
    train_df['label'] = train_df['label'].clip(lower=-100.0, upper=200.0)

    # Then apply 3-sigma clipping for remaining outliers
    mean_ret = train_df['label'].mean()
    std_ret = train_df['label'].std()
    if std_ret > 0:
        train_df['label'] = train_df['label'].clip(
            lower=mean_ret - 3 * std_ret, upper=mean_ret + 3 * std_ret
        )

    # Replace any NaN/Inf that might have slipped through
    train_df['label'] = train_df['label'].replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=['label'])

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

    # Use a mixed ensemble: native boosters, tree ensembles, and linear baselines.
    # CatBoost stays on CPU because continuation from init_model is not supported on GPU.
    import xgboost as xgb
    import lightgbm as lgb
    import warnings
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    device = 'cuda' if XGBOOST_USE_GPU else 'cpu'

    if has_existing:
        # Load existing models for incremental training
        models = existing_model['all_models']
        for model_name in (
            'CatBoost',
            'RandomForest',
            'ExtraTrees',
            'Ridge',
            'ElasticNet',
            'SGDRegressor-L2',
            'SGDRegressor-ElasticNet',
        ):
            if model_name not in models:
                try:
                    models[model_name] = _fresh_ensemble_model(model_name, device)
                except ImportError:
                    pass
        models = _order_models_for_training(models)
        print(
            f"   🚀 Incremental training: {len(models)} models "
            f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu, SGD=cpu)"
        )
    else:
        # Fresh training - create new models
        # Add CatBoost if available (GPU-accelerated, good with tabular data)
        models = {}
        try:
            models['CatBoost'] = _fresh_ensemble_model('CatBoost', device)
        except ImportError:
            pass
        models['XGBoost'] = _fresh_ensemble_model('XGBoost', device)
        models['LightGBM'] = _fresh_ensemble_model('LightGBM', device)
        models['RandomForest'] = _fresh_ensemble_model('RandomForest', device)
        models['ExtraTrees'] = _fresh_ensemble_model('ExtraTrees', device)
        models['Ridge'] = _fresh_ensemble_model('Ridge', device)
        models['ElasticNet'] = _fresh_ensemble_model('ElasticNet', device)
        models['SGDRegressor-L2'] = _fresh_ensemble_model('SGDRegressor-L2', device)
        models['SGDRegressor-ElasticNet'] = _fresh_ensemble_model('SGDRegressor-ElasticNet', device)
        models = _order_models_for_training(models)
        print(
            f"   🚀 Fresh training: {list(models.keys())} "
            f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu, SGD=cpu)"
        )

    # Train with incremental learning (no CV for speed - just train/val split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")

    trained_models = []
    model_scores = []
    model_names = []

    import time

    for name, m in models.items():
        try:
            print(f"      🔄 {name}: Training...", end=" ", flush=True)
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
                incremental_failed = False
                used_incremental = False
                if has_existing:
                    # True incremental training for supported models
                    try:
                        if name == 'XGBoost':
                            used_incremental = True
                            m.fit(X_train, y_train, xgb_model=m.get_booster())
                        elif name == 'LightGBM':
                            used_incremental = True
                            m.fit(X_train, y_train, init_model=m.booster_)
                        elif name == 'CatBoost':
                            used_incremental = True
                            if _catboost_has_trained_trees(m):
                                _configure_catboost_continuation(m)
                                m.fit(X_train, y_train, init_model=m)
                            else:
                                print("(no saved trees yet, training fresh)...", end=" ", flush=True)
                                incremental_failed = True
                        elif name in ('SGDRegressor-L2', 'SGDRegressor-ElasticNet'):
                            used_incremental = True
                            m.partial_fit(X_train, y_train)
                        if used_incremental:
                            # Quick sanity check for incremental training - detect numerical instability
                            quick_pred = m.predict(X_val[:100])
                            if _predictions_are_unstable(quick_pred):
                                print(f"(incremental unstable, retraining fresh)...", end=" ", flush=True)
                                incremental_failed = True
                    except Exception as e:
                        print(f"(incremental failed: {e}, retraining fresh)...", end=" ", flush=True)
                        incremental_failed = True

                if not used_incremental or incremental_failed:
                    # Fresh training - create new model instance if incremental failed
                    m = _fresh_ensemble_model(name, device)
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
                if _predictions_are_unstable(y_pred):
                    print(f"      ⚠️ {name}: Predictions contain NaN/Inf, skipping")
                    continue

                if len(y_pred) == 0 or len(y_val) == 0:
                    print(f"failed: Empty predictions")
                    continue

                score = r2_score(y_val, y_pred)

                # If an incremental model validates as numerically broken, retry once from scratch.
                if (used_incremental and (score < -10 or score > 1 or np.isnan(score) or np.isinf(score))):
                    print(f"      ⚠️ {name}: R² = {score:.3f} after incremental fit, retraining fresh...", end=" ", flush=True)
                    m = _fresh_ensemble_model(name, device)
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_val)
                    if _predictions_are_unstable(y_pred):
                        print("failed: unstable after fresh retrain")
                        continue
                    score = r2_score(y_val, y_pred)

                if score < -10 or score > 1 or np.isnan(score) or np.isinf(score):
                    print(f"failed: invalid validation score {score:.3f}")
                    continue

            elapsed = time.time() - start_time
            status = "incremental" if used_incremental and not incremental_failed else "fresh"
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
            'all_scores': dict(zip(model_names, model_scores)),
            'catboost_backend': 'cpu'
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
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create backup before overwriting
        if os.path.exists(path):
            backup_path = path + '.backup'
            import shutil
            shutil.copy2(path, backup_path)
            print(f"   📦 AI Elite: Backed up previous model to {backup_path}")

        # Save model with metadata
        model_data = {
            'model': model,
            'metadata': metadata or {}
        }
        joblib.dump(model_data, path)
        save_native_model_artifacts(model, path)
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to save model to {path}: {e}")
