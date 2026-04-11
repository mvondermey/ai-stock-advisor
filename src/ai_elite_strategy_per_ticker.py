"""
Hybrid AI Elite model training:
  1. collect_ticker_training_data() - gather samples for one ticker
  2. train_shared_base_model()     - train ONE model on ALL tickers' data
  3. fine_tune_per_ticker()        - fine-tune a copy of the base model per ticker
"""

import pandas as pd
import numpy as np
import copy
import joblib
import os
from multiprocessing import Pool, get_context
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

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

_AI_ELITE_COLLECTION_CONTEXT_PATH: Optional[str] = None
_AI_ELITE_COLLECTION_CONTEXT: Optional[Dict[str, object]] = None
_AI_ELITE_TRAIN_CONTEXT_PATH: Optional[str] = None
_AI_ELITE_TRAIN_CONTEXT: Optional[Dict[str, object]] = None


def _init_ai_elite_collection_worker(
    context_path: Optional[str],
    context: Optional[Dict[str, object]] = None,
) -> None:
    global _AI_ELITE_COLLECTION_CONTEXT_PATH, _AI_ELITE_COLLECTION_CONTEXT
    _AI_ELITE_COLLECTION_CONTEXT_PATH = context_path
    _AI_ELITE_COLLECTION_CONTEXT = context


def _get_ai_elite_collection_context() -> Dict[str, object]:
    global _AI_ELITE_COLLECTION_CONTEXT
    if _AI_ELITE_COLLECTION_CONTEXT is None:
        if not _AI_ELITE_COLLECTION_CONTEXT_PATH:
            raise ValueError("AI Elite collection context path is not initialized")
        _AI_ELITE_COLLECTION_CONTEXT = joblib.load(_AI_ELITE_COLLECTION_CONTEXT_PATH)
    return _AI_ELITE_COLLECTION_CONTEXT


def _init_ai_elite_train_worker(
    context_path: Optional[str],
    context: Optional[Dict[str, object]] = None,
) -> None:
    global _AI_ELITE_TRAIN_CONTEXT_PATH, _AI_ELITE_TRAIN_CONTEXT
    _AI_ELITE_TRAIN_CONTEXT_PATH = context_path
    _AI_ELITE_TRAIN_CONTEXT = context


def _collect_ai_elite_training_data_worker(ticker: str) -> Tuple[str, List[dict]]:
    context = _get_ai_elite_collection_context()
    samples = collect_ticker_training_data(
        ticker=ticker,
        ticker_data=context['ticker_data_grouped'].get(ticker),
        train_start_date=context['train_start_date'],
        train_end_date=context['train_end_date'],
        forward_days=int(context['forward_days']),
        market_returns=context['market_returns'],
        price_history_cache=context.get('price_history_cache'),
        hourly_history_cache=context.get('hourly_history_cache'),
    )
    return ticker, samples


def collect_training_data_parallel(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int,
    market_returns: Dict[datetime, float],
    n_processes: int,
    price_history_cache=None,
    hourly_history_cache=None,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    all_training_data: List[dict] = []
    ticker_samples_map: Dict[str, List[dict]] = {}
    temp_context_path = None

    try:
        worker_tickers = list(all_tickers)
        n_workers = max(1, min(n_processes, len(worker_tickers))) if worker_tickers else 1
        sample_context = {
            'ticker_data_grouped': ticker_data_grouped,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'forward_days': forward_days,
            'market_returns': market_returns,
            'price_history_cache': price_history_cache,
            'hourly_history_cache': hourly_history_cache,
        }

        if os.name != "nt":
            global _AI_ELITE_COLLECTION_CONTEXT_PATH, _AI_ELITE_COLLECTION_CONTEXT
            _AI_ELITE_COLLECTION_CONTEXT_PATH = None
            _AI_ELITE_COLLECTION_CONTEXT = sample_context
            try:
                with get_context("fork").Pool(
                    processes=n_workers,
                    initializer=_init_ai_elite_collection_worker,
                    initargs=(None, sample_context),
                ) as pool:
                    results = pool.imap_unordered(_collect_ai_elite_training_data_worker, worker_tickers)
                    for ticker, samples in tqdm(
                        results,
                        total=len(worker_tickers),
                        desc="   AI Elite collection",
                        ncols=100,
                        unit="ticker",
                    ):
                        if samples:
                            all_training_data.extend(samples)
                            ticker_samples_map[ticker] = samples
            finally:
                _AI_ELITE_COLLECTION_CONTEXT_PATH = None
                _AI_ELITE_COLLECTION_CONTEXT = None
            return all_training_data, ticker_samples_map

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
            temp_context_path = temp_context_file.name
        joblib.dump(sample_context, temp_context_path)

        with Pool(
            processes=n_workers,
            initializer=_init_ai_elite_collection_worker,
            initargs=(temp_context_path,),
        ) as pool:
            results = pool.imap_unordered(_collect_ai_elite_training_data_worker, worker_tickers)
            for ticker, samples in tqdm(
                results,
                total=len(worker_tickers),
                desc="   AI Elite collection",
                ncols=100,
                unit="ticker",
            ):
                if samples:
                    all_training_data.extend(samples)
                    ticker_samples_map[ticker] = samples
    finally:
        if temp_context_path is not None:
            try:
                os.unlink(temp_context_path)
            except OSError as e:
                print(f"   ⚠️ AI Elite: Failed to remove temp context: {e}")

    return all_training_data, ticker_samples_map


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


def _fresh_ensemble_model(name: str, device: str, n_jobs: int = -1):
    """Create a fresh ensemble member with stable defaults.
    
    Args:
        name: Model name
        device: 'cpu' or 'cuda' for XGBoost
        n_jobs: Number of parallel jobs for training (-1 = all cores, or specify limit)
    """
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
            tree_method='hist', device=device, verbosity=0, n_jobs=n_jobs
        )
    if name == 'LightGBM':
        lgbm_device = _lightgbm_device()
        return lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=-1, n_jobs=n_jobs,
            device=lgbm_device
        )
    if name == 'CatBoost':
        import catboost as cb

        # CatBoost uses thread_count instead of n_jobs
        thread_count = n_jobs if n_jobs > 0 else os.cpu_count() or 1
        catboost_params = dict(
            iterations=100, depth=4, learning_rate=0.1,
            task_type='CPU', random_seed=42, verbose=0,
            allow_writing_files=False, thread_count=thread_count
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
            n_jobs=n_jobs,
        )
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=n_jobs,
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


def _lightgbm_device() -> str:
    """Mirror the native-Linux GPU policy used at startup for LightGBM."""
    kernel_release = ""
    try:
        kernel_release = os.uname().release.lower()
    except AttributeError:
        kernel_release = ""
    is_wsl = bool(os.getenv("WSL_DISTRO_NAME")) or "microsoft" in kernel_release
    return "cpu" if is_wsl else "gpu"


def _model_backend(name: str, xgb_device: str) -> str:
    if name == "XGBoost":
        return xgb_device
    if name == "LightGBM":
        return _lightgbm_device()
    return "cpu"


def _model_worker_mode(name: str, xgb_device: str) -> str:
    return "spawn" if _model_backend(name, xgb_device) in {"cuda", "gpu"} else "fork"


def _format_training_plan(model_names: List[str], xgb_device: str) -> str:
    return ", ".join(
        f"{name}={_model_backend(name, xgb_device)}/{_model_worker_mode(name, xgb_device)}"
        for name in model_names
    )


def _configure_existing_model_backend(name: str, model, xgb_device: str) -> None:
    """Keep restored models aligned with the backend policy used for fresh models."""
    backend = _model_backend(name, xgb_device)
    if name in {"XGBoost", "LightGBM"} and hasattr(model, "set_params"):
        try:
            model.set_params(device=backend)
        except Exception:
            pass
    if name == "CatBoost" and hasattr(model, "_init_params"):
        model._init_params["task_type"] = "CPU"


def _predictions_are_unstable(predictions) -> bool:
    """Detect obviously broken model output after incremental continuation."""
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        return True
    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return True
    return np.max(np.abs(preds)) > 1e10


def _configure_catboost_continuation(model, thread_count: int = 1):
    """Apply safer runtime settings before CatBoost continuation training."""
    from config import AI_ELITE_CATBOOST_USED_RAM_LIMIT

    if not hasattr(model, '_init_params'):
        return
    model._init_params['task_type'] = 'CPU'
    model._init_params['thread_count'] = thread_count
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


def _get_ai_elite_train_context() -> Dict[str, object]:
    global _AI_ELITE_TRAIN_CONTEXT
    if _AI_ELITE_TRAIN_CONTEXT is None:
        if not _AI_ELITE_TRAIN_CONTEXT_PATH:
            raise ValueError("AI Elite training context is not initialized")
        _AI_ELITE_TRAIN_CONTEXT = joblib.load(_AI_ELITE_TRAIN_CONTEXT_PATH)
    return _AI_ELITE_TRAIN_CONTEXT


def _set_model_parallelism(model, n_jobs_limit: int) -> None:
    """Apply a bounded CPU budget to models that support internal threading."""
    if n_jobs_limit == -1:
        return
    model_class_name = model.__class__.__name__
    if "CatBoost" in model_class_name:
        if hasattr(model, '_init_params'):
            model._init_params['thread_count'] = n_jobs_limit
        return
    if hasattr(model, 'n_jobs'):
        model.n_jobs = n_jobs_limit
    if hasattr(model, 'set_params'):
        for param_name in ('n_jobs', 'thread_count'):
            try:
                model.set_params(**{param_name: n_jobs_limit})
            except (ValueError, TypeError):
                continue
    if hasattr(model, '_init_params'):
        model._init_params['thread_count'] = n_jobs_limit


def _train_single_model(
    name: str,
    n_jobs_limit: int = 4,
) -> Tuple[bool, Optional[object], Optional[float], float, str, str]:
    """
    Train a single model in a worker process.
    Returns (ok, model, score, elapsed_time, status, detail).
    
    Args:
        n_jobs_limit: Max threads per model to avoid over-subscription when parallel training
    """
    import time
    import warnings
    from sklearn.metrics import r2_score

    start_time = time.time()
    context = _get_ai_elite_train_context()
    X_train = context['X_train']
    y_train = context['y_train']
    X_val = context['X_val']
    y_val = context['y_val']
    device = context['device']
    has_existing = bool(context['has_existing'])
    existing_model = context['models'].get(name) if has_existing else None

    # Validate training data
    if len(X_train) < 10:
        return (False, None, None, 0.0, "skipped", "insufficient training data")

    # Check for valid target values
    if np.all(y_train == y_train[0]):
        return (False, None, None, 0.0, "skipped", "constant target")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            incremental_failed = False
            used_incremental = False
            incremental_error = None

            m = existing_model
            if has_existing and m is not None:
                # Limit n_jobs for parallel training to avoid over-subscription
                _set_model_parallelism(m, n_jobs_limit)
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
                            _configure_catboost_continuation(
                                m,
                                thread_count=n_jobs_limit if n_jobs_limit > 0 else (os.cpu_count() or 1),
                            )
                            m.fit(X_train, y_train, init_model=m)
                        else:
                            incremental_failed = True
                    elif name in ('SGDRegressor-L2', 'SGDRegressor-ElasticNet'):
                        used_incremental = True
                        m.partial_fit(X_train, y_train)
                    if used_incremental:
                        # Quick sanity check for incremental training
                        quick_pred = m.predict(X_val[:100])
                        if _predictions_are_unstable(quick_pred):
                            incremental_error = "incremental predictions unstable"
                            incremental_failed = True
                except Exception as e:
                    incremental_error = str(e)
                    incremental_failed = True

            if not used_incremental or incremental_failed:
                # Fresh training with limited n_jobs
                m = _fresh_ensemble_model(name, device, n_jobs=n_jobs_limit)
                m.fit(X_train, y_train)

            # Validate model was trained
            if name == 'CatBoost':
                try:
                    if hasattr(m, 'get_tree_count') and m.get_tree_count() == 0:
                        return (False, None, None, time.time() - start_time, "failed", "CatBoost built zero trees")
                except:
                    pass

            # Validate on held-out set
            y_pred = m.predict(X_val)

            if _predictions_are_unstable(y_pred):
                return (False, None, None, time.time() - start_time, "failed", "predictions unstable")

            if len(y_pred) == 0 or len(y_val) == 0:
                return (False, None, None, time.time() - start_time, "failed", "empty predictions or validation target")

            score = r2_score(y_val, y_pred)

            # Retry from scratch if incremental training produced bad results
            if used_incremental and (score < -10 or score > 1 or np.isnan(score) or np.isinf(score)):
                m = _fresh_ensemble_model(name, device, n_jobs=n_jobs_limit)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_val)
                if _predictions_are_unstable(y_pred):
                    return (False, None, None, time.time() - start_time, "failed", "unstable after fresh retrain")
                score = r2_score(y_val, y_pred)

            if score < -10 or score > 1 or np.isnan(score) or np.isinf(score):
                return (False, None, None, time.time() - start_time, "failed", f"invalid validation score {score:.3f}")

            elapsed = time.time() - start_time
            status = "incremental" if used_incremental and not incremental_failed else "fresh"
            detail = "ok"
            if incremental_failed and incremental_error:
                detail = f"incremental fallback: {incremental_error}"
            return (True, m, score, elapsed, status, detail)

    except Exception as e:
        return (False, None, None, time.time() - start_time, "failed", str(e))


def collect_ticker_training_data(
    ticker: str,
    ticker_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    forward_days: int = 5,
    hourly_cache: dict = None,
    market_returns: dict = None,
    price_history_cache=None,
    hourly_history_cache=None,
) -> List[dict]:
    """Collect training samples for a single ticker. Returns list of dicts.
    market_returns: dict mapping sample_date -> market return (pre-computed)."""
    if ticker_data is None or len(ticker_data) == 0:
        return []

    try:
        from ai_elite_strategy import (
            _extract_features,
            _calculate_forward_return,
            _load_hourly_data_direct,
        )
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
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            train_end_date + timedelta(days=forward_days + 2),
            hourly_history_cache=hourly_history_cache,
        )}

    samples = []
    current_date = train_start_date
    while current_date <= train_end_date:
        try:
            hourly_data = hourly_cache.get(ticker)
            features = _extract_features(
                ticker,
                hourly_data,
                current_date,
                daily_data=ticker_data,
                price_history_cache=price_history_cache,
            )
            if features is None:
                current_date += timedelta(days=1)
                continue
            forward_return = _calculate_forward_return(ticker_data, current_date, forward_days)
            if forward_return is None:
                current_date += timedelta(days=1)
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
        current_date += timedelta(days=1)

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
        for model_name, model in models.items():
            _configure_existing_model_backend(model_name, model, device)
        models = _order_models_for_training(models)
        print(
            f"   🚀 Incremental training: {len(models)} models "
            f"({_format_training_plan(list(models.keys()), device)})"
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
            f"({_format_training_plan(list(models.keys()), device)})"
        )

    # Train with incremental learning (no CV for speed - just train/val split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")

    trained_models = []
    model_scores = []
    model_names = []

    import time

    # Prepare training arguments for parallel execution
    model_names_to_train = list(models.keys())
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    spawn_model_names: List[str] = []
    parallel_model_names = list(model_names_to_train)
    for name in list(parallel_model_names):
        if _model_worker_mode(name, device) == "spawn":
            parallel_model_names.remove(name)
            spawn_model_names.append(name)

    # Determine number of parallel training workers
    # Cap workers by available cores so the pool doesn't oversubscribe small machines.
    total_cores = os.cpu_count() or 8
    n_train_workers = max(1, min(3, len(parallel_model_names), total_cores)) if parallel_model_names else 1
    n_jobs_per_model = max(1, total_cores // n_train_workers)

    print(
        f"   🚀 Training {len(model_names_to_train)} models in parallel "
        f"({n_train_workers} fork workers, {n_jobs_per_model} threads/model; "
        f"{len(spawn_model_names)} spawn models)..."
    )
    if spawn_model_names:
        print("   ℹ️ GPU models use spawned workers; CPU models stay on fork workers")
        for name in spawn_model_names:
            print(f"      🔄 {name}: queued (spawn, backend={_model_backend(name, device)})")
    overall_start = time.time()

    global _AI_ELITE_TRAIN_CONTEXT_PATH, _AI_ELITE_TRAIN_CONTEXT
    _AI_ELITE_TRAIN_CONTEXT = {
        'models': models,
        'X_train': X_train_np,
        'y_train': y_train_np,
        'X_val': X_val_np,
        'y_val': y_val_np,
        'device': device,
        'has_existing': has_existing,
    }
    train_context_path = None
    try:
        pending: Dict[str, object] = {}
        fork_pool = None
        spawn_pool = None
        try:
            if os.name != 'nt' and parallel_model_names and n_train_workers > 1:
                # Use fork so workers inherit the training arrays/models once instead of
                # serializing them into every call.
                fork_pool = get_context("fork").Pool(
                    processes=n_train_workers,
                    initializer=_init_ai_elite_train_worker,
                    initargs=(None, _AI_ELITE_TRAIN_CONTEXT),
                )
                for name in parallel_model_names:
                    print(f"      🔄 {name}: queued (fork, backend={_model_backend(name, device)})")
                    pending[name] = fork_pool.apply_async(_train_single_model, (name, n_jobs_per_model))
            else:
                # Sequential fallback for Windows or single-worker runs - use all cores.
                for name in parallel_model_names:
                    print(f"      🔄 {name}: starting (main-process, backend={_model_backend(name, device)})")
                    result = _train_single_model(name, n_jobs_limit=-1)
                    ok, m, score, elapsed, status, detail = result
                    if ok:
                        print(f"      🔄 {name}: R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                        trained_models.append(m)
                        model_scores.append(score)
                        model_names.append(name)
                    else:
                        print(f"      ⚠️ {name}: {detail} ({status}, {elapsed:.1f}s)")

            if spawn_model_names:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_context_file:
                    train_context_path = temp_context_file.name
                joblib.dump(_AI_ELITE_TRAIN_CONTEXT, train_context_path)
                _AI_ELITE_TRAIN_CONTEXT_PATH = train_context_path
                spawn_pool = get_context("spawn").Pool(
                    processes=len(spawn_model_names),
                    initializer=_init_ai_elite_train_worker,
                    initargs=(train_context_path, None),
                )
                for name in spawn_model_names:
                    pending[name] = spawn_pool.apply_async(_train_single_model, (name, 1))

            while pending:
                ready_names = [name for name, handle in pending.items() if handle.ready()]
                if not ready_names:
                    time.sleep(0.5)
                    continue
                for name in ready_names:
                    result_handle = pending.pop(name)
                    try:
                        result = result_handle.get(timeout=0)
                        ok, m, score, elapsed, status, detail = result
                        if ok:
                            print(f"      🔄 {name}: R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                            trained_models.append(m)
                            model_scores.append(score)
                            model_names.append(name)
                        else:
                            print(f"      ⚠️ {name}: {detail} ({status}, {elapsed:.1f}s)")
                    except Exception as e:
                        print(f"      ⚠️ {name}: Training error: {e}")
        finally:
            if fork_pool is not None:
                fork_pool.close()
                fork_pool.join()
            if spawn_pool is not None:
                spawn_pool.close()
                spawn_pool.join()
    finally:
        _AI_ELITE_TRAIN_CONTEXT_PATH = None
        _AI_ELITE_TRAIN_CONTEXT = None
        if train_context_path is not None:
            try:
                os.unlink(train_context_path)
            except OSError as e:
                print(f"   ⚠️ AI Elite: Failed to remove train context: {e}")

    overall_elapsed = time.time() - overall_start
    print(f"   ⏱️ All models trained in {overall_elapsed:.1f}s")

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
