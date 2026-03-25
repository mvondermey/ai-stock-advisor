"""
Universal Model Strategy - Single ML model for all tickers.

Instead of training one model per ticker (AI Elite approach), this strategy
trains ONE model on pooled data from all tickers and uses it to predict
returns for any stock.

Features are normalized/relative so they work across different price scales.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import joblib

# Model save path
MODEL_SAVE_DIR = Path("logs/models")
UNIVERSAL_MODEL_PATH = MODEL_SAVE_DIR / "universal_model.joblib"
UNIVERSAL_SCALER_PATH = MODEL_SAVE_DIR / "universal_scaler.joblib"

# Feature engineering constants
LOOKBACK_DAYS = 60  # Days of history for feature calculation
FORWARD_DAYS = 5    # Prediction horizon (5 days like AI Elite)


def calculate_universal_features(ticker: str, data: pd.DataFrame, current_date: datetime) -> Optional[Dict[str, float]]:
    """
    Calculate normalized features that work across all tickers.
    All features are relative/normalized so they're comparable across stocks.
    """
    try:
        # Filter to current date
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            if current_date.tzinfo is None:
                current_date = current_date.replace(tzinfo=data.index.tz)

        hist = data.loc[:current_date]
        if len(hist) < LOOKBACK_DAYS:
            return None

        close = hist['Close'].values
        volume = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(close))
        high = hist['High'].values if 'High' in hist.columns else close
        low = hist['Low'].values if 'Low' in hist.columns else close

        # Use last LOOKBACK_DAYS
        close = close[-LOOKBACK_DAYS:]
        volume = volume[-LOOKBACK_DAYS:]
        high = high[-LOOKBACK_DAYS:]
        low = low[-LOOKBACK_DAYS:]

        if len(close) < LOOKBACK_DAYS or close[-1] <= 0:
            return None

        # === MOMENTUM FEATURES (relative) ===
        mom_5d = (close[-1] / close[-5] - 1) * 100 if close[-5] > 0 else 0
        mom_10d = (close[-1] / close[-10] - 1) * 100 if close[-10] > 0 else 0
        mom_20d = (close[-1] / close[-20] - 1) * 100 if close[-20] > 0 else 0
        mom_60d = (close[-1] / close[0] - 1) * 100 if close[0] > 0 else 0

        # === VOLATILITY FEATURES ===
        returns = np.diff(close) / close[:-1]
        volatility_20d = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else 0
        volatility_60d = np.std(returns) * np.sqrt(252) * 100

        # === TREND FEATURES ===
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        price_vs_sma20 = (close[-1] / sma_20 - 1) * 100 if sma_20 > 0 else 0
        price_vs_sma50 = (close[-1] / sma_50 - 1) * 100 if sma_50 > 0 else 0
        sma_trend = (sma_20 / sma_50 - 1) * 100 if sma_50 > 0 else 0

        # === VOLUME FEATURES ===
        avg_vol_20 = np.mean(volume[-20:])
        avg_vol_60 = np.mean(volume)
        vol_ratio = avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else 1

        # === RANGE/VOLATILITY FEATURES ===
        atr_values = high[-20:] - low[-20:]
        atr = np.mean(atr_values)
        atr_pct = (atr / close[-1]) * 100 if close[-1] > 0 else 0

        # === MOMENTUM ACCELERATION ===
        mom_accel = mom_5d - mom_10d  # Recent momentum vs older

        # === RISK-ADJUSTED MOMENTUM ===
        risk_adj_mom = mom_20d / volatility_20d if volatility_20d > 0 else 0

        # === DRAWDOWN FROM RECENT HIGH ===
        high_20d = np.max(close[-20:])
        drawdown = (close[-1] / high_20d - 1) * 100 if high_20d > 0 else 0

        # === RSI-LIKE FEATURE ===
        gains = np.maximum(returns[-14:], 0)
        losses = np.abs(np.minimum(returns[-14:], 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        rs = avg_gain / avg_loss if avg_loss > 0 else 1
        rsi = 100 - (100 / (1 + rs))

        return {
            'mom_5d': mom_5d,
            'mom_10d': mom_10d,
            'mom_20d': mom_20d,
            'mom_60d': mom_60d,
            'volatility_20d': volatility_20d,
            'volatility_60d': volatility_60d,
            'price_vs_sma20': price_vs_sma20,
            'price_vs_sma50': price_vs_sma50,
            'sma_trend': sma_trend,
            'vol_ratio': vol_ratio,
            'atr_pct': atr_pct,
            'mom_accel': mom_accel,
            'risk_adj_mom': risk_adj_mom,
            'drawdown': drawdown,
            'rsi': rsi,
        }

    except Exception as e:
        return None


def calculate_forward_return(data: pd.DataFrame, current_date: datetime, forward_days: int = FORWARD_DAYS) -> Optional[float]:
    """Calculate actual forward return for training labels."""
    try:
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            if current_date.tzinfo is None:
                current_date = current_date.replace(tzinfo=data.index.tz)

        hist = data.loc[:current_date]
        if len(hist) == 0:
            return None

        current_price = hist['Close'].iloc[-1]

        future_date = current_date + timedelta(days=forward_days + 5)  # Buffer for weekends
        future_data = data.loc[current_date:future_date]

        if len(future_data) < forward_days // 2:  # Need at least half the days
            return None

        # Get price approximately forward_days later
        future_price = future_data['Close'].iloc[min(forward_days, len(future_data)-1)]

        if current_price > 0 and future_price > 0:
            return (future_price / current_price - 1) * 100
        return None

    except Exception:
        return None


class UniversalModelStrategy:
    """
    Single ML model trained on pooled data from all tickers.
    """

    def __init__(self, retrain_days: int = 1, min_samples: int = 200):
        self.retrain_days = retrain_days
        self.min_samples = min_samples
        self.model = None
        self.all_models = None  # All trained models
        self.all_scores = None  # All model scores
        self.best_name = None   # Name of best model
        self.scaler = None
        self.feature_cols = None
        self.last_train_day = 0
        self.day_count = 0

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days

    def train_model(self, ticker_data_grouped: Dict[str, pd.DataFrame],
                    business_days: List[datetime], current_day_idx: int):
        """
        Train universal model on pooled data from all tickers.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        import warnings

        print(f"   🧠 Universal Model: Training on pooled data...")

        # Collect training samples from ALL tickers
        X_list = []
        y_list = []

        # Get current date from business_days
        if current_day_idx >= len(business_days):
            current_day_idx = len(business_days) - 1
        current_date = business_days[current_day_idx] if current_day_idx >= 0 else business_days[0]

        # Sample from HISTORICAL data (sequential - GIL blocks threading)
        for ticker, data in ticker_data_grouped.items():
            if data is None or len(data) < LOOKBACK_DAYS + FORWARD_DAYS + 10:
                continue
            available_dates = data.index[data.index < current_date - timedelta(days=FORWARD_DAYS + 5)]
            if len(available_dates) < LOOKBACK_DAYS:
                continue
            sample_dates = available_dates[LOOKBACK_DAYS::5]
            for sample_date in sample_dates[-100:]:
                features = calculate_universal_features(ticker, data, sample_date)
                if features is None:
                    continue
                forward_ret = calculate_forward_return(data, sample_date, FORWARD_DAYS)
                if forward_ret is None:
                    continue
                X_list.append(list(features.values()))
                y_list.append(forward_ret)
            if len(X_list) >= 10000:
                break

        if len(X_list) < self.min_samples:
            print(f"   ⚠️ Universal Model: Not enough samples ({len(X_list)} < {self.min_samples})")
            return False

        X = np.array(X_list)
        y = np.array(y_list)

        # Store feature column names from first successful sample
        self.feature_cols = ['mom_5d', 'mom_10d', 'mom_20d', 'mom_60d', 'volatility_20d',
                             'volatility_60d', 'price_vs_sma20', 'price_vs_sma50', 'sma_trend',
                             'vol_ratio', 'atr_pct', 'mom_accel', 'risk_adj_mom', 'drawdown', 'rsi']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train ALL models and pick the best one
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from config import XGBOOST_USE_GPU
        import xgboost as xgb
        import lightgbm as lgb
        import time

        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        has_existing = self.all_models is not None

        # Build models (XGBoost + LightGBM + CatBoost - all support GPU/incremental)
        if has_existing:
            models = self.all_models
            # Add CatBoost if missing from loaded model
            if 'CatBoost' not in models:
                try:
                    import catboost as cb
                    # Note: CatBoost GPU continuation not supported, always use CPU for incremental
                    models['CatBoost'] = cb.CatBoostRegressor(
                        iterations=100, depth=4, learning_rate=0.1,
                        task_type='CPU', random_seed=42, verbose=0
                    )
                except ImportError:
                    pass
            print(f"   📊 Universal Model: Continuing training on {len(X_list)} samples ({device})...")
        else:
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
            # Add CatBoost if available
            try:
                import catboost as cb
                task_type = 'GPU' if XGBOOST_USE_GPU else 'CPU'
                models['CatBoost'] = cb.CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.1,
                    task_type=task_type, random_seed=42, verbose=0
                )
            except ImportError:
                pass
            print(f"   📊 Universal Model: Training NEW {list(models.keys())} on {len(X_list)} samples ({device})...")

        # Train/val split (faster than CV)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train all models and evaluate
        trained_models = {}
        model_scores = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, m in models.items():
                try:
                    print(f"      🔄 {name}: Training...", end=" ", flush=True)
                    start_time = time.time()

                    # Check if incremental training is safe (features must match)
                    can_increment = has_existing
                    if can_increment:
                        old_n_features = m.n_features_in_ if hasattr(m, 'n_features_in_') else 0
                        new_n_features = X_train.shape[1]
                        if old_n_features != new_n_features:
                            print(f"features {old_n_features}→{new_n_features}, retraining from scratch...", end=" ", flush=True)
                            # Recreate model from scratch
                            if name == 'XGBoost':
                                m = xgb.XGBRegressor(
                                    n_estimators=100, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, random_state=42,
                                    tree_method='hist', device=device, verbosity=0, n_jobs=-1
                                )
                            elif name == 'LightGBM':
                                m = lgb.LGBMRegressor(
                                    n_estimators=100, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
                                )
                            elif name == 'CatBoost':
                                import catboost as cb
                                task_type = 'GPU' if XGBOOST_USE_GPU else 'CPU'
                                m = cb.CatBoostRegressor(
                                    iterations=100, depth=5, learning_rate=0.1,
                                    task_type=task_type, random_seed=42, verbose=0
                                )
                            can_increment = False

                    # True incremental training for supported models
                    if can_increment:
                        if name == 'XGBoost':
                            m.fit(X_train, y_train, xgb_model=m.get_booster())
                        elif name == 'LightGBM':
                            m.fit(X_train, y_train, init_model=m.booster_)
                        elif name == 'CatBoost':
                            # CatBoost GPU doesn't support continuation - switch to CPU
                            m._init_params['task_type'] = 'CPU'
                            m.fit(X_train, y_train, init_model=m)
                        else:
                            m.fit(X_train, y_train)
                    else:
                        m.fit(X_train, y_train)

                    # Validate on held-out set
                    y_pred = m.predict(X_val)
                    score = r2_score(y_val, y_pred)

                    elapsed = time.time() - start_time
                    status = "incremental" if has_existing else "fresh"
                    print(f"R² = {score:.3f} ({status}, {elapsed:.1f}s)")
                    trained_models[name] = m
                    model_scores[name] = score
                except Exception as e:
                    print(f"failed: {e}")

        if not trained_models:
            print(f"   ⚠️ Universal Model: No models trained successfully")
            return False

        # Pick best model
        best_name = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_name]

        self.all_models = trained_models
        self.all_scores = model_scores
        self.model = trained_models[best_name]
        self.best_name = best_name
        self.last_train_day = self.day_count
        print(f"   ✅ Universal Model: Saved {len(trained_models)} models. Best = {best_name} (R² {best_score:.3f})")

        # Save model to disk
        self.save_model()

        return True

    def save_model(self):
        """Save all models, scaler, and training state to disk."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            # Create backup before overwriting
            if UNIVERSAL_MODEL_PATH.exists():
                backup_path = UNIVERSAL_MODEL_PATH.with_suffix('.backup.joblib')
                import shutil
                shutil.copy2(UNIVERSAL_MODEL_PATH, backup_path)
                print(f"   📦 Universal Model: Backed up previous model to {backup_path}")

            joblib.dump({
                'all_models': self.all_models,
                'all_scores': self.all_scores,
                'best_name': self.best_name,
                'model': self.model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                # Persist training state for continuous learning
                'day_count': self.day_count,
                'last_train_day': self.last_train_day
            }, UNIVERSAL_MODEL_PATH)
            print(f"   💾 Universal Model: Saved {len(self.all_models)} models, day_count={self.day_count} to {UNIVERSAL_MODEL_PATH}")
        except Exception as e:
            print(f"   ⚠️ Universal Model: Failed to save: {e}")

    def load_model(self) -> bool:
        """Load all models, scaler, and training state from disk."""
        try:
            if UNIVERSAL_MODEL_PATH.exists():
                data = joblib.load(UNIVERSAL_MODEL_PATH)
                # Handle new format (dict with all_models)
                if isinstance(data, dict) and 'all_models' in data:
                    self.all_models = data['all_models']
                    self.all_scores = data['all_scores']
                    self.best_name = data['best_name']
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.feature_cols = data['feature_cols']
                    # Restore training state for continuous learning
                    self.day_count = data.get('day_count', 0)
                    self.last_train_day = data.get('last_train_day', 0)
                    n_models = len(self.all_models) if self.all_models else 1
                    print(f"   📂 Universal Model: Loaded {n_models} models, day_count={self.day_count}")
                else:
                    # Legacy format (single model)
                    self.model = data
                    if UNIVERSAL_SCALER_PATH.exists():
                        self.scaler = joblib.load(UNIVERSAL_SCALER_PATH)
                    self.feature_cols = ['mom_5d', 'mom_10d', 'mom_20d', 'mom_60d', 'volatility_20d',
                                         'volatility_60d', 'price_vs_sma20', 'price_vs_sma50', 'sma_trend',
                                         'vol_ratio', 'atr_pct', 'mom_accel', 'risk_adj_mom', 'drawdown', 'rsi']
                    print(f"   📂 Universal Model: Loaded from disk (legacy format)")
                return True
        except Exception as e:
            print(f"   ⚠️ Universal Model: Failed to load: {e}")
        return False

    def predict_returns(self, tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame],
                        current_date: datetime) -> List[Tuple[str, float]]:
        """
        Predict returns for all tickers using the universal model.
        """
        if self.model is None:
            return []

        predictions = []

        for ticker in tickers:
            if ticker not in ticker_data_grouped:
                continue

            features = calculate_universal_features(ticker, ticker_data_grouped[ticker], current_date)
            if features is None:
                continue

            try:
                import warnings
                X = np.array([list(features.values())])
                X_scaled = self.scaler.transform(X)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='X does not have valid feature names')
                    pred = self.model.predict(X_scaled)[0]
                predictions.append((ticker, pred))
            except Exception:
                continue

        return predictions

    def increment_day(self):
        self.day_count += 1


def select_universal_model_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    model: UniversalModelStrategy,
    business_days: List[datetime],
    current_day_idx: int
) -> List[str]:
    """
    Select top N stocks using universal model predictions.
    """
    # Train/retrain if needed
    if model.should_retrain():
        model.train_model(ticker_data_grouped, business_days, current_day_idx)

    # Get predictions
    predictions = model.predict_returns(all_tickers, ticker_data_grouped, current_date)

    if not predictions:
        return []

    # Sort by predicted return (highest first)
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Return top N tickers
    return [ticker for ticker, _ in predictions[:top_n]]
