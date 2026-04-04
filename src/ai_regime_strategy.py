"""
AI Regime Strategy: ML predicts which strategy to use based on market conditions

Approach:
1. Track performance of multiple sub-strategies daily
2. Extract rich market regime features (multi-timeframe volatility, trend, breadth, dispersion)
3. Train ML model to predict which strategy will perform best in next N days
4. Allocate capital to the predicted best strategy

This is a "meta-strategy" that learns WHEN to use each strategy, not WHAT stocks to pick.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import joblib
from pathlib import Path

from model_training_safety import (
    catboost_has_trained_trees,
    cleanup_training_memory,
    configure_catboost_cpu_continuation,
    ensure_catboost_cpu_metadata,
    reset_legacy_catboost_member,
    restore_native_model_artifacts,
    save_native_model_artifacts,
)
from strategy_universes import AI_REGIME_STRATEGY_SOURCES, get_enabled_strategy_aliases

# Model save paths
MODEL_SAVE_DIR = Path("logs/models")
AI_REGIME_MODEL_PATH = MODEL_SAVE_DIR / "ai_regime_model.joblib"
AI_REGIME_ENCODER_PATH = MODEL_SAVE_DIR / "ai_regime_encoder.joblib"


SUB_STRATEGIES = get_enabled_strategy_aliases(AI_REGIME_STRATEGY_SOURCES)

# Regime features to extract
REGIME_LOOKBACK = 20  # Days to look back for regime features


class AIRegimeAllocator:
    """
    ML-based regime detection and strategy allocation.

    Learns which strategy performs best under different market conditions.
    """

    def __init__(self, retrain_days: int = 1, forward_days: int = 1):
        """
        Args:
            retrain_days: Retrain model every N days (1 = daily)
            forward_days: Predict best strategy for next N days
        """
        self.retrain_days = retrain_days
        self.forward_days = forward_days

        # Strategy performance history: {strategy_name: [daily_values]}
        self.strategy_histories: Dict[str, List[float]] = defaultdict(list)

        # Training data: list of (features, best_strategy_label)
        self.training_data: List[Dict] = []

        # Current ML model
        self.model = None
        self.all_models = None  # All trained models
        self.all_scores = None  # All model scores
        self.best_name = None   # Name of best model
        self.last_train_day = 0

        # Current allocation
        self.current_strategy = None
        self.day_count = 0

    def record_daily_values(self, strategy_values: Dict[str, float]):
        """
        Record daily portfolio values for all sub-strategies.

        Args:
            strategy_values: {strategy_name: portfolio_value}
        """
        for name in SUB_STRATEGIES:
            if name in strategy_values and strategy_values[name] is not None:
                value = strategy_values[name]
                # Ensure value is a number, not a list or other type
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.strategy_histories[name].append(float(value))
                elif isinstance(value, list):
                    # If it's a list, take the last value (common error case)
                    if value and isinstance(value[-1], (int, float)):
                        self.strategy_histories[name].append(float(value[-1]))
                    else:
                        continue  # Skip invalid list
                else:
                    continue  # Skip invalid types
        self.day_count += 1

    def _history_as_of(self, strategy_name: str, day_idx: Optional[int] = None) -> List[float]:
        """Return strategy history truncated to the requested training day."""
        history = self.strategy_histories.get(strategy_name, [])
        if day_idx is None:
            return history
        return history[: day_idx + 1]

    def extract_regime_features(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
        day_idx: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Extract rich market regime features from price data.

        Features:
        - Multi-timeframe market momentum (5d, 10d, 20d, 60d)
        - Multi-timeframe volatility (short-term vs long-term)
        - Market breadth (% stocks above their 20d SMA)
        - Cross-sectional dispersion of returns and volatility
        - Volatility regime change (short vol / long vol ratio)
        - Strategy momentum and Sharpe at multiple timeframes
        """
        try:
            features = {}

            # Collect multi-timeframe returns and volatilities across all stocks
            stock_returns_5d = []
            stock_returns_10d = []
            stock_returns_20d = []
            stock_returns_60d = []
            stock_vol_short = []   # 10-day vol
            stock_vol_long = []    # 60-day vol
            stocks_above_sma20 = 0
            stocks_total = 0

            for ticker, data in ticker_data_grouped.items():
                if data is None or len(data) < 30:
                    continue

                close = data['Close'].dropna()
                if len(close) < 30:
                    continue

                # Filter to current date
                close = close[close.index <= current_date]
                if len(close) < 25:
                    continue

                latest = close.iloc[-1]
                stocks_total += 1

                # Multi-timeframe momentum using calendar days
                # 5 calendar days (~5 trading days)
                start_5d = current_date - timedelta(days=5)
                data_5d = close[close.index >= start_5d]
                if len(data_5d) >= 3 and data_5d.iloc[0] > 0:
                    stock_returns_5d.append((latest / data_5d.iloc[0] - 1) * 100)

                # 10 calendar days (~10 trading days)
                start_10d = current_date - timedelta(days=10)
                data_10d = close[close.index >= start_10d]
                if len(data_10d) >= 5 and data_10d.iloc[0] > 0:
                    stock_returns_10d.append((latest / data_10d.iloc[0] - 1) * 100)

                # 20 calendar days (~20 trading days)
                start_20d = current_date - timedelta(days=20)
                data_20d = close[close.index >= start_20d]
                if len(data_20d) >= 10 and data_20d.iloc[0] > 0:
                    stock_returns_20d.append((latest / data_20d.iloc[0] - 1) * 100)

                # 60 calendar days (~60 trading days)
                start_60d = current_date - timedelta(days=60)
                data_60d = close[close.index >= start_60d]
                if len(data_60d) >= 30 and data_60d.iloc[0] > 0:
                    stock_returns_60d.append((latest / data_60d.iloc[0] - 1) * 100)

                # Multi-timeframe volatility using calendar days
                daily_ret = close.pct_change().dropna()
                if len(daily_ret) >= 10:
                    vol_short = daily_ret.tail(10).std() * np.sqrt(252) * 100
                    stock_vol_short.append(vol_short)
                if len(daily_ret) >= 60:
                    vol_long = daily_ret.tail(60).std() * np.sqrt(252) * 100
                    stock_vol_long.append(vol_long)

                # Market breadth: stock above its 20d SMA (using last 20 trading days)
                if len(close) >= 20:
                    sma20 = close.tail(20).mean()
                    if latest > sma20:
                        stocks_above_sma20 += 1

            if len(stock_returns_20d) < 10:
                return None

            # --- Market-wide momentum features (multi-timeframe) ---
            features['market_return_5d'] = np.mean(stock_returns_5d) if stock_returns_5d else 0
            features['market_return_10d'] = np.mean(stock_returns_10d) if stock_returns_10d else 0
            features['market_return_20d'] = np.mean(stock_returns_20d)
            features['market_return_60d'] = np.mean(stock_returns_60d) if stock_returns_60d else 0

            # --- Volatility features ---
            features['market_vol_short'] = np.mean(stock_vol_short) if stock_vol_short else 0
            features['market_vol_long'] = np.mean(stock_vol_long) if stock_vol_long else 0
            # Vol regime change: rising vol (>1) = risk-off, falling vol (<1) = risk-on
            features['vol_regime_ratio'] = (features['market_vol_short'] / features['market_vol_long']) if features['market_vol_long'] > 0 else 1.0
            features['high_vol_regime'] = 1.0 if features['market_vol_short'] > 30 else 0.0

            # --- Dispersion features ---
            features['return_dispersion_5d'] = np.std(stock_returns_5d) if len(stock_returns_5d) > 1 else 0
            features['return_dispersion_20d'] = np.std(stock_returns_20d)
            features['volatility_dispersion'] = np.std(stock_vol_short) if len(stock_vol_short) > 1 else 0

            # --- Market breadth ---
            features['breadth_pct_above_sma20'] = (stocks_above_sma20 / stocks_total * 100) if stocks_total > 0 else 50.0

            # --- Trend features ---
            # Short vs long momentum (positive = accelerating, negative = decelerating)
            features['momentum_acceleration'] = features['market_return_5d'] - features['market_return_20d'] / 4
            features['trend_consistency'] = features['market_return_10d'] - features['market_return_5d']

            # --- Strategy momentum and Sharpe features (multi-timeframe) ---
            for strat_name in SUB_STRATEGIES:
                hist = self._history_as_of(strat_name, day_idx)

                # 10-day momentum
                if len(hist) >= 10 and hist[-10] > 0:
                    features[f'{strat_name}_mom_10d'] = (hist[-1] - hist[-10]) / hist[-10] * 100
                else:
                    features[f'{strat_name}_mom_10d'] = 0.0

                # 20-day momentum
                if len(hist) >= 20 and hist[-20] > 0:
                    features[f'{strat_name}_mom_20d'] = (hist[-1] - hist[-20]) / hist[-20] * 100
                else:
                    features[f'{strat_name}_mom_20d'] = 0.0

                # 20-day Sharpe (daily returns annualized)
                if len(hist) >= 20:
                    daily_rets = np.diff(hist[-20:]) / np.array(hist[-20:-1])
                    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
                        features[f'{strat_name}_sharpe_20d'] = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(252)
                    else:
                        features[f'{strat_name}_sharpe_20d'] = 0.0
                else:
                    features[f'{strat_name}_sharpe_20d'] = 0.0

            return features

        except Exception as e:
            print(f"   ⚠️ AI Regime: Feature extraction failed: {e}")
            return None

    def _get_best_strategy_forward(self, day_idx: int) -> Optional[str]:
        """
        Determine which strategy performed best over the next forward_days.
        Used for training labels.
        """
        best_strategy = None
        best_return = -np.inf

        for strat_name in SUB_STRATEGIES:
            hist = self.strategy_histories.get(strat_name, [])
            if len(hist) <= day_idx + self.forward_days:
                continue

            start_val = hist[day_idx]
            end_val = hist[day_idx + self.forward_days]

            if start_val > 0:
                ret = (end_val - start_val) / start_val * 100
                if ret > best_return:
                    best_return = ret
                    best_strategy = strat_name

        return best_strategy

    @cleanup_training_memory
    def train_model(self, ticker_data_grouped: Dict[str, pd.DataFrame],
                    business_days: List[datetime]):
        """
        Train ML model to predict best strategy based on regime features.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        import warnings

        # Collect training samples
        training_samples = []

        # We need: day_idx for features, day_idx + forward_days for label
        # So max valid day_idx = day_count - forward_days - 1
        # Start from day 0 (or 1 if we need previous day data)
        min_day = 1  # Start from day 1 (need day 0 for comparison)
        max_day = self.day_count - self.forward_days  # Last day we can get a forward label for

        if max_day <= min_day or self.day_count < 2:
            print(f"   ⚠️ AI Regime: Need at least 2 days of data (have {self.day_count}), skipping training")
            return False

        # Sample every day (or every 2 days if we have lots of data)
        step = 2 if (max_day - min_day) > 20 else 1
        for day_idx in range(min_day, max_day, step):
            # Get features for this day
            if day_idx >= len(business_days):
                continue
            current_date = business_days[day_idx]

            features = self.extract_regime_features(
                ticker_data_grouped,
                current_date,
                day_idx=day_idx,
            )
            if features is None:
                continue

            # Get label (best strategy over next forward_days)
            best_strat = self._get_best_strategy_forward(day_idx)
            if best_strat is None:
                continue

            features['label'] = best_strat
            training_samples.append(features)

        if len(training_samples) < 1:
            if self.day_count <= self.forward_days + 1:
                print(f"   ℹ️ AI Regime: Day {self.day_count} - waiting for {self.forward_days} days of data before training")
            else:
                print(f"   ⚠️ AI Regime: No training samples available - using existing model")
            # Don't return False - keep existing model for predictions
            return True

        # Convert to DataFrame
        train_df = pd.DataFrame(training_samples)

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(train_df['label'])

        # Check we have at least 2 classes (classifier requirement)
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print(f"   ⚠️ AI Regime: Only {n_classes} class in training data, need at least 2")
            return False

        # Feature columns (exclude label)
        feature_cols = [c for c in train_df.columns if c != 'label']
        X = train_df[feature_cols].values

        min_training_samples = 10
        if len(X) < min_training_samples:
            print(
                f"   ℹ️ AI Regime: Only {len(X)} usable samples; need at least "
                f"{min_training_samples} before training"
            )
            return True

        # Train classifiers (XGBoost + LightGBM for GPU + incremental support)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from config import XGBOOST_USE_GPU
        import xgboost as xgb
        import lightgbm as lgb
        import time

        def _fresh_classifier(name: str, device: str):
            if name == 'XGBoost':
                return xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                    tree_method='hist', device=device, verbosity=0, n_jobs=-1,
                    use_label_encoder=False, eval_metric='mlogloss'
                )
            if name == 'LightGBM':
                return lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
                )
            if name == 'CatBoost':
                import catboost as cb
                return cb.CatBoostClassifier(
                    iterations=100, depth=4, learning_rate=0.1,
                    task_type='CPU', random_seed=42, verbose=0,
                    allow_writing_files=False, thread_count=1
                )
            raise ValueError(f"Unknown model name: {name}")

        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        has_existing = self.all_models is not None and len(self.all_models) > 0

        # Require a stratified split with every class present in both train and val.
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min()) if len(counts) > 0 else 0
        if min_class_count < 2:
            print(
                f"   ℹ️ AI Regime: Need at least 2 samples per class before training "
                f"(current min class count={min_class_count})"
            )
            return True

        test_size = max(len(unique), int(round(len(X) * 0.2)))
        test_size = min(test_size, len(X) - len(unique))
        if test_size < len(unique) or (len(X) - test_size) < len(unique):
            print(
                f"   ℹ️ AI Regime: Not enough samples for a stratified train/val split "
                f"across {len(unique)} classes"
            )
            return True

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if len(X_train) < len(unique):
            print(
                f"   ℹ️ AI Regime: Need at least one training sample per class "
                f"(train={len(X_train)}, classes={len(unique)})"
            )
            return True

        # Build models (XGBoost + LightGBM - both support incremental training)
        if has_existing:
            models = dict(self.all_models)
            if 'CatBoost' not in models:
                try:
                    models['CatBoost'] = _fresh_classifier('CatBoost', device)
                except ImportError:
                    pass
            print(
                f"   📊 AI Regime: Continuing training on {len(training_samples)} samples "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu)..."
            )
        else:
            models = {
                'XGBoost': _fresh_classifier('XGBoost', device),
                'LightGBM': _fresh_classifier('LightGBM', device),
            }
            try:
                models['CatBoost'] = _fresh_classifier('CatBoost', device)
            except ImportError:
                pass
            print(
                f"   📊 AI Regime: Training NEW {list(models.keys())} on {len(training_samples)} samples "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu)..."
            )

        # Train all models and evaluate
        trained_models = {}
        model_scores = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, m in models.items():
                try:
                    print(f"      🔄 {name}: Training...", end=" ", flush=True)
                    start_time = time.time()

                    incremental_failed = False
                    used_incremental = False

                    # Check if incremental training is safe (classes and features must match)
                    if has_existing:
                        try:
                            old_classes = set(getattr(m, 'classes_', []))
                            new_classes = set(np.unique(y_train))
                            old_n_features = m.n_features_in_ if hasattr(m, 'n_features_in_') else 0
                            new_n_features = X_train.shape[1]

                            if old_classes != new_classes or old_n_features != new_n_features:
                                reason = []
                                if old_classes != new_classes:
                                    reason.append(f"classes {len(old_classes)}→{len(new_classes)}")
                                if old_n_features != new_n_features:
                                    reason.append(f"features {old_n_features}→{new_n_features}")
                                print(f"{', '.join(reason)}, retraining from scratch...", end=" ", flush=True)
                                incremental_failed = True
                            else:
                                used_incremental = True
                                if name == 'XGBoost':
                                    m.fit(X_train, y_train, xgb_model=m.get_booster())
                                elif name == 'LightGBM':
                                    m.fit(X_train, y_train, init_model=m.booster_)
                                elif name == 'CatBoost':
                                    if catboost_has_trained_trees(m):
                                        configure_catboost_cpu_continuation(m)
                                        m.fit(X_train, y_train, init_model=m)
                                    else:
                                        print("(no saved trees yet, training fresh)...", end=" ", flush=True)
                                        incremental_failed = True
                                else:
                                    m.fit(X_train, y_train)
                        except Exception as e:
                            print(f"(incremental failed: {e}, retraining fresh)...", end=" ", flush=True)
                            incremental_failed = True

                    if not used_incremental or incremental_failed:
                        m = _fresh_classifier(name, device)
                        m.fit(X_train, y_train)

                    # Validate on held-out set
                    y_pred = m.predict(X_val)
                    score = accuracy_score(y_val, y_pred)

                    elapsed = time.time() - start_time
                    status = "incremental" if used_incremental and not incremental_failed else "fresh"
                    print(f"Accuracy = {score:.3f} ({status}, {elapsed:.1f}s)")
                    trained_models[name] = m
                    model_scores[name] = score
                except Exception as e:
                    print(f"failed: {e}")

        if not trained_models:
            print(f"   ⚠️ AI Regime: No models trained successfully")
            return False

        # Pick best model
        best_name = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_name]

        self.all_models = trained_models
        self.all_scores = model_scores
        self.model = trained_models[best_name]
        self.best_name = best_name
        self.label_encoder = le
        self.feature_cols = feature_cols
        self.last_train_day = self.day_count

        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {le.inverse_transform([u])[0]: c for u, c in zip(unique, counts)}
        print(f"   ✅ AI Regime: Saved {len(trained_models)} models. Best = {best_name} (Accuracy {best_score:.3f})")
        print(f"   📊 AI Regime: Class distribution: {class_dist}")

        # Save model to disk
        self.save_model()

        return True

    def save_model(self):
        """Save all models, label encoder, and training state to disk."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

            # Create backup before overwriting
            if AI_REGIME_MODEL_PATH.exists():
                backup_path = AI_REGIME_MODEL_PATH.with_suffix('.backup.joblib')
                import shutil
                shutil.copy2(AI_REGIME_MODEL_PATH, backup_path)
                print(f"   📦 AI Regime: Backed up previous model to {backup_path}")

            payload = ensure_catboost_cpu_metadata({
                'all_models': self.all_models,
                'all_scores': self.all_scores,
                'best_name': self.best_name,
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_cols': self.feature_cols,
                # Persist training state for continuous learning
                'training_data': self.training_data,
                'day_count': self.day_count,
                'last_train_day': self.last_train_day,
                'strategy_histories': dict(self.strategy_histories)
            })
            joblib.dump(payload, AI_REGIME_MODEL_PATH)
            save_native_model_artifacts(payload, AI_REGIME_MODEL_PATH)
            print(f"   💾 AI Regime: Saved {len(self.all_models)} models + {len(self.training_data)} training samples to {AI_REGIME_MODEL_PATH}")
        except Exception as e:
            print(f"   ⚠️ AI Regime: Failed to save: {e}")

    def load_model(self) -> bool:
        """Load all models, label encoder, and training state from disk."""
        try:
            if AI_REGIME_MODEL_PATH.exists():
                data = joblib.load(AI_REGIME_MODEL_PATH)
                data = restore_native_model_artifacts(data, AI_REGIME_MODEL_PATH, is_classifier=True)
                data, reset_catboost = reset_legacy_catboost_member(data)
                if reset_catboost:
                    print("   ♻️ AI Regime: Resetting saved CatBoost member for a clean CPU incremental restart")
                self.model = data['model']
                self.label_encoder = data['label_encoder']
                self.feature_cols = data['feature_cols']
                # Load all models if available (new format)
                self.all_models = data.get('all_models')
                self.all_scores = data.get('all_scores')
                self.best_name = data.get('best_name')
                # Restore training state for continuous learning
                self.training_data = data.get('training_data', [])
                self.day_count = data.get('day_count', 0)
                self.last_train_day = data.get('last_train_day', 0)
                loaded_histories = data.get('strategy_histories', {})
                self.strategy_histories = defaultdict(list, loaded_histories)
                n_models = len(self.all_models) if self.all_models else 1
                n_samples = len(self.training_data)
                print(f"   📂 AI Regime: Loaded {n_models} models, {n_samples} training samples, day_count={self.day_count}")
                return True
        except Exception as e:
            print(f"   ⚠️ AI Regime: Failed to load: {e}")
        return False

    def release_model_artifacts(self):
        """Drop heavy fitted models while preserving training state."""
        self.model = None
        self.all_models = None

    def predict_best_strategy(self, ticker_data_grouped: Dict[str, pd.DataFrame],
                               current_date: datetime) -> Optional[str]:
        """
        Predict which strategy will perform best.

        Returns:
            Strategy name to use, or None if prediction fails
        """
        # Default if no model trained yet or corrupted
        if self.model is None or not hasattr(self.model, 'predict'):
            if self.model is not None:
                print(f"   ⚠️ AI Regime: Model corrupted (type: {type(self.model)})")
            print("   ⚠️ AI Regime: No selection (model unavailable)")
            return None  # No fallback - let strategy handle it

        # Extract current features
        features = self.extract_regime_features(ticker_data_grouped, current_date)
        if features is None or not isinstance(features, dict):
            print("   ⚠️ AI Regime: No selection (feature extraction failed)")
            return None  # No fallback - let strategy handle it

        # Predict
        try:
            X = np.array([[features.get(c, 0) for c in self.feature_cols]])

            # Validate model state before prediction
            if not hasattr(self.model, 'n_features_in_') or self.model.n_features_in_ != X.shape[1]:
                print(f"   ⚠️ AI Regime: Model feature mismatch, resetting model")
                self.model = None
                print("   ⚠️ AI Regime: No selection (model feature mismatch)")
                return None

            pred_idx = self.model.predict(X)[0]
            pred_strategy = self.label_encoder.inverse_transform([pred_idx])[0]

            # Get prediction probabilities for logging
            probs = self.model.predict_proba(X)[0]
            top_prob = max(probs)

            if pred_strategy != self.current_strategy:
                print(f"   🔄 AI Regime: Switching to {pred_strategy} (confidence: {top_prob:.1%})")

            self.current_strategy = pred_strategy
            return pred_strategy

        except Exception as e:
            print(f"   ⚠️ AI Regime: Prediction failed: {e}")
            print("   ⚠️ AI Regime: No selection (prediction failed)")
            return None  # No fallback - let strategy handle it

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        # Need at least 1 day before training is possible (will fail gracefully if not enough data)
        if self.day_count < 1:
            return False
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days


class AIRegimeMonthlyAllocator(AIRegimeAllocator):
    """
    AI Regime Monthly: Trains and rebalances at start of month only.
    Same logic as daily AI Regime but with monthly rebalance schedule.
    """

    def __init__(self, forward_days: int = 1):
        """Initialize with monthly rebalance (retrain_days set to 30)."""
        super().__init__(retrain_days=30, forward_days=forward_days)
        self.last_rebalance_month = None

    def should_rebalance(self, current_date: datetime) -> bool:
        """Check if we're at the start of a new month."""
        current_month = (current_date.year, current_date.month)
        if self.last_rebalance_month is None:
            self.last_rebalance_month = current_month
            return True
        if current_month != self.last_rebalance_month:
            self.last_rebalance_month = current_month
            return True
        return False


def select_ai_regime_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    predicted_strategy: str,
    ai_elite_models: Dict = None
) -> List[str]:
    """
    Select stocks using the predicted best strategy.

    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date
        top_n: Number of stocks to select
        predicted_strategy: Strategy name predicted by AI Regime
        ai_elite_models: AI Elite models dict (needed if ai_elite is predicted)

    Returns:
        List of selected tickers
    """
    # Import shared functions
    from shared_strategies import (
        select_top_performers, select_top_performers_vol_filtered,
        select_volatility_adj_mom_stocks, select_momentum_ai_hybrid_stocks,
        select_ai_elite_with_training, select_risk_adj_mom_stocks,
        select_quality_momentum_stocks, select_3m_1y_ratio_stocks, select_1y_3m_ratio_stocks,
        select_momentum_volatility_hybrid_1y3m_stocks,
        select_momentum_volatility_hybrid_6m_stocks,
        select_momentum_volatility_hybrid_stocks,
    )
    from new_strategies import select_concentrated_3m_stocks, select_dual_momentum_stocks, select_trend_following_atr_stocks
    from bollinger_bands_strategy import (
        select_bb_squeeze_breakout_stocks, select_bb_rsi_combo_stocks,
        select_bb_breakout_stocks, select_bb_mean_reversion_stocks
    )
    from inverse_etf_hedge_strategy import select_inverse_etf_hedge_stocks
    from analyst_recommendation_strategy import select_analyst_recommendation_stocks as select_analyst_rec_stocks

    # Map strategy names to their implementations
    strategy_map = {
        # Risk-Adjusted Momentum (use shared function with different lookbacks)
        'risk_adj_mom_3m': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),
        'risk_adj_mom_3m_monthly': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),
        'risk_adj_mom_3m_up': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),
        'risk_adj_mom_3m_stop': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),
        'risk_adj_mom_6m': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=180),
        'risk_adj_mom_6m_monthly': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=180),
        'risk_adj_mom': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=365),
        'risk_adj_mom_1m': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=30),
        'risk_adj_mom_1m_monthly': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=30),
        'risk_adj_mom_3m_sent': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),
        'risk_adj_sent': lambda: select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=90),

        # AI/ML Strategies (select_ai_elite_with_training returns tuple, extract first element)
        'ai_elite': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'ai_elite_monthly': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'ai_elite_market_up': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'ai_elite_filtered': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'elite_hybrid': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'elite_risk': lambda: select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'momentum_ai_hybrid': lambda: select_momentum_ai_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n),

        # Momentum Strategies
        'momentum_volatility_hybrid_6m': lambda: select_momentum_volatility_hybrid_6m_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'momentum_volatility_hybrid': lambda: select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'momentum_volatility_hybrid_1y_3m': lambda: select_momentum_volatility_hybrid_1y3m_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'vol_adj_mom': lambda: select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'vol_sweet_mom': lambda: select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        '1m_vol_sweet': lambda: select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'price_acceleration': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 90, top_n),

        # Buy & Hold Strategies
        'static_bh_1y': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'static_bh_6m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 180, top_n),
        'static_bh_3m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 90, top_n),
        'static_bh_1m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 30, top_n),
        'bh_1y_monthly': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_6m_monthly': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 180, top_n),
        'bh_3m_monthly': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 90, top_n),
        'bh_1m_monthly': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 30, top_n),

        # Dynamic Buy & Hold Strategies
        'dynamic_bh_1y': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n, apply_performance_filter=True),
        'dynamic_bh_6m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 180, top_n, apply_performance_filter=True),
        'dynamic_bh_3m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 90, top_n, apply_performance_filter=True),
        'dynamic_bh_1m': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 30, top_n, apply_performance_filter=True),
        'dynamic_bh_1y_vol': lambda: select_top_performers_vol_filtered(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=365, max_volatility=0.4),
        'dynamic_bh_1y_ts': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),

        # Enhanced BH Strategies (all use 1Y lookback with different triggers - simplified to base selection)
        'bh_1y_vol_trigger': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_perf_trigger': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_mom_trigger': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_atr_trigger': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_hybrid_trigger': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_volume': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_sector': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_perf_thresh': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_market_regime': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_mom_persist': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_overlap': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_rank_drift': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_drawdown': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),
        'bh_1y_smart_monthly': lambda: select_top_performers(all_tickers, ticker_data_grouped, current_date, 365, top_n),

        # Technical Strategies
        'concentrated_3m': lambda: select_concentrated_3m_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'dual_momentum': lambda: select_dual_momentum_stocks(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'trend_atr': lambda: select_trend_following_atr_stocks(all_tickers, ticker_data_grouped, current_date, top_n)[0],
        'bb_squeeze': lambda: select_bb_squeeze_breakout_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'bb_rsi_combo': lambda: select_bb_rsi_combo_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'bb_breakout': lambda: select_bb_breakout_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'bb_mean_rev': lambda: select_bb_mean_reversion_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'trend_breakout': lambda: select_trend_following_atr_stocks(all_tickers, ticker_data_grouped, current_date, top_n),

        # Ratio Strategies
        '3m_1y_ratio': lambda: select_3m_1y_ratio_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        '1y_3m_ratio': lambda: select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date, top_n),

        # Quality Strategies
        'quality_momentum': lambda: select_quality_momentum_stocks(all_tickers, ticker_data_grouped, current_date, top_n),

        # Other Strategies
        'analyst_rec': lambda: select_analyst_rec_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
        'inverse_etf_hedge': lambda: select_inverse_etf_hedge_stocks(all_tickers, ticker_data_grouped, current_date, top_n),
    }

    # Try to get the strategy function
    strategy_func = strategy_map.get(predicted_strategy)

    if strategy_func:
        try:
            return strategy_func()
        except Exception as e:
            print(f"   ❌ AI Regime error: {predicted_strategy} failed: {e}")
            print("   ⚠️ AI Regime: No selection (strategy execution failed)")
            return []  # Return empty list, no fallback
    else:
        # Strategy not in map
        print(f"   ❌ AI Regime error: Unknown strategy {predicted_strategy}")
        print("   ⚠️ AI Regime: No selection (unknown strategy)")
        return []  # Return empty list, no fallback
