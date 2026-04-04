"""
AI Champion Strategy: choose the likely next-day winner from a small set
of already strong strategies.

This is intentionally narrower than AI Regime. Instead of choosing from the
full strategy universe, it focuses on a handful of consistent leaders and
uses only as-of-date features when generating training samples.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from model_training_safety import (
    catboost_has_trained_trees,
    cleanup_training_memory,
    configure_catboost_cpu_continuation,
    ensure_catboost_cpu_metadata,
    reset_legacy_catboost_member,
    restore_native_model_artifacts,
    save_native_model_artifacts,
)
from config import (
    AI_CHAMPION_CONFIDENCE_THRESHOLD,
    AI_CHAMPION_FORWARD_DAYS,
    AI_CHAMPION_HOLD_MARGIN,
    AI_CHAMPION_RETRAIN_DAYS,
    XGBOOST_USE_GPU,
)
from strategy_universes import (
    AI_CHAMPION_STRATEGY_SOURCES,
    build_pairwise_feature_names,
    get_enabled_strategy_aliases,
)


MODEL_SAVE_DIR = Path("logs/models")
AI_CHAMPION_MODEL_PATH = MODEL_SAVE_DIR / "ai_champion_model.joblib"

CANDIDATE_STRATEGIES = tuple(get_enabled_strategy_aliases(AI_CHAMPION_STRATEGY_SOURCES))
PAIRWISE_FEATURES = build_pairwise_feature_names(CANDIDATE_STRATEGIES)


def _clone_model(name: str, device: str):
    import xgboost as xgb
    import lightgbm as lgb

    if name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            tree_method="hist",
            device=device,
            verbosity=0,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    if name == "CatBoost":
        import catboost as cb

        return cb.CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.1,
            task_type="CPU",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            thread_count=1,
        )
    raise ValueError(f"Unknown model: {name}")


class AIChampionAllocator:
    """ML allocator focused on a small set of leading strategies."""

    def __init__(
        self,
        retrain_days: int = AI_CHAMPION_RETRAIN_DAYS,
        forward_days: int = AI_CHAMPION_FORWARD_DAYS,
        confidence_threshold: float = AI_CHAMPION_CONFIDENCE_THRESHOLD,
        hold_margin: float = AI_CHAMPION_HOLD_MARGIN,
    ):
        self.retrain_days = retrain_days
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold
        self.hold_margin = hold_margin

        self.strategy_histories: Dict[str, List[float]] = defaultdict(list)
        self.active_candidates = set(CANDIDATE_STRATEGIES)
        self.training_data: List[Dict] = []

        self.model = None
        self.all_models = None
        self.all_scores = None
        self.all_label_values = {}
        self.all_use_mapped_labels = {}
        self.best_name = None
        self.label_encoder = None
        self.feature_cols: List[str] = []
        self.model_label_values: List[int] = []
        self.model_uses_mapped_labels = False
        self.current_strategy = None
        self.day_count = 0
        self.last_train_day = 0

    def record_daily_values(self, strategy_values: Dict[str, float]):
        """Record daily values for the active candidate set."""
        current_active = set()
        for name in CANDIDATE_STRATEGIES:
            if name not in strategy_values:
                continue
            value = strategy_values[name]
            if value is None:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.strategy_histories[name].append(float(value))
                current_active.add(name)
            elif isinstance(value, list) and value and isinstance(value[-1], (int, float)):
                self.strategy_histories[name].append(float(value[-1]))
                current_active.add(name)

        if current_active:
            self.active_candidates = current_active
        self.day_count += 1

    def _eligible_candidates(self, min_history: int = 1) -> List[str]:
        return [
            name
            for name in CANDIDATE_STRATEGIES
            if name in self.active_candidates and len(self.strategy_histories.get(name, [])) >= min_history
        ]

    def _history_as_of(self, strategy_name: str, day_idx: Optional[int] = None) -> List[float]:
        history = self.strategy_histories.get(strategy_name, [])
        if day_idx is None:
            return history
        return history[: day_idx + 1]

    @staticmethod
    def _safe_return(history: List[float], lookback: int) -> float:
        if len(history) <= lookback or history[-(lookback + 1)] <= 0:
            return 0.0
        start_val = history[-(lookback + 1)]
        end_val = history[-1]
        return (end_val - start_val) / start_val

    @staticmethod
    def _safe_vol(history: List[float], lookback: int) -> float:
        if len(history) < lookback + 1:
            return 0.0
        series = pd.Series(history[-(lookback + 1) :])
        returns = series.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(252))

    @staticmethod
    def _safe_sharpe(history: List[float], lookback: int) -> float:
        if len(history) < lookback + 1:
            return 0.0
        series = pd.Series(history[-(lookback + 1) :])
        returns = series.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        std = returns.std()
        if std <= 0:
            return 0.0
        return float((returns.mean() / std) * np.sqrt(252))

    @staticmethod
    def _safe_drawdown(history: List[float], lookback: int) -> float:
        if len(history) < 2:
            return 0.0
        window = np.asarray(history[-min(len(history), lookback) :], dtype=float)
        if len(window) == 0:
            return 0.0
        peaks = np.maximum.accumulate(window)
        drawdowns = np.where(peaks > 0, (peaks - window) / peaks, 0.0)
        return float(np.max(drawdowns)) if len(drawdowns) else 0.0

    def _market_proxy_features(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> Dict[str, float]:
        from market_regime import calculate_trailing_return

        features: Dict[str, float] = {}
        market_data = ticker_data_grouped.get("SPY")
        if market_data is None or len(market_data) == 0:
            market_data = ticker_data_grouped.get("QQQ")

        if market_data is not None and len(market_data) > 0:
            for lookback in (3, 5, 10, 20):
                market_ret = calculate_trailing_return(market_data, current_date, lookback)
                features[f"market_return_{lookback}d"] = (market_ret or 0.0) / 100.0

            closes = market_data["Close"].dropna()
            closes = closes[closes.index <= current_date]
            if len(closes) >= 11:
                returns_10d = closes.tail(11).pct_change().dropna()
                features["market_vol_10d"] = float(returns_10d.std() * np.sqrt(252))
            else:
                features["market_vol_10d"] = 0.0

            if len(closes) >= 21:
                sma20 = closes.tail(20).mean()
                features["market_above_sma20"] = 1.0 if closes.iloc[-1] > sma20 else 0.0
            else:
                features["market_above_sma20"] = 0.0
        else:
            for lookback in (3, 5, 10, 20):
                features[f"market_return_{lookback}d"] = 0.0
            features["market_vol_10d"] = 0.0
            features["market_above_sma20"] = 0.0

        stock_returns_5d = []
        stocks_above_sma20 = 0
        stocks_total = 0
        for data in ticker_data_grouped.values():
            if data is None or len(data) < 25:
                continue
            close = data["Close"].dropna()
            close = close[close.index <= current_date]
            if len(close) < 10:
                continue
            stocks_total += 1
            start_5d = current_date - timedelta(days=5)
            close_5d = close[close.index >= start_5d]
            if len(close_5d) >= 2 and close_5d.iloc[0] > 0:
                stock_returns_5d.append((close_5d.iloc[-1] / close_5d.iloc[0]) - 1)
            if len(close) >= 20 and close.iloc[-1] > close.tail(20).mean():
                stocks_above_sma20 += 1

        features["breadth_pct_above_sma20"] = (
            stocks_above_sma20 / stocks_total if stocks_total else 0.5
        )
        features["market_dispersion_5d"] = float(np.std(stock_returns_5d)) if len(stock_returns_5d) > 1 else 0.0
        return features

    def extract_features(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
        day_idx: Optional[int] = None,
    ) -> Dict[str, float]:
        """Build strictly as-of-date features for a given day."""
        features = self._market_proxy_features(ticker_data_grouped, current_date)

        for name in CANDIDATE_STRATEGIES:
            history = self._history_as_of(name, day_idx)
            features[f"{name}_ret_1d"] = self._safe_return(history, 1)
            features[f"{name}_ret_3d"] = self._safe_return(history, 3)
            features[f"{name}_ret_5d"] = self._safe_return(history, 5)
            features[f"{name}_ret_10d"] = self._safe_return(history, 10)
            features[f"{name}_vol_5d"] = self._safe_vol(history, 5)
            features[f"{name}_vol_10d"] = self._safe_vol(history, 10)
            features[f"{name}_sharpe_5d"] = self._safe_sharpe(history, 5)
            features[f"{name}_sharpe_10d"] = self._safe_sharpe(history, 10)
            features[f"{name}_drawdown_10d"] = self._safe_drawdown(history, 10)
            features[f"{name}_is_active"] = 1.0 if name in self.active_candidates else 0.0

        for left, right in PAIRWISE_FEATURES:
            left_history = self._history_as_of(left, day_idx)
            right_history = self._history_as_of(right, day_idx)
            features[f"{left}_minus_{right}_ret_3d"] = self._safe_return(left_history, 3) - self._safe_return(right_history, 3)
            features[f"{left}_minus_{right}_ret_5d"] = self._safe_return(left_history, 5) - self._safe_return(right_history, 5)
            features[f"{left}_minus_{right}_sharpe_5d"] = self._safe_sharpe(left_history, 5) - self._safe_sharpe(right_history, 5)

        return features

    def _get_best_strategy_forward(self, day_idx: int) -> Optional[str]:
        best_strategy = None
        best_return = -np.inf

        for strat_name in self._eligible_candidates():
            history = self.strategy_histories.get(strat_name, [])
            if len(history) <= day_idx + self.forward_days:
                continue

            start_val = history[day_idx]
            end_val = history[day_idx + self.forward_days]
            if start_val <= 0:
                continue

            total_return = (end_val - start_val) / start_val
            if total_return > best_return:
                best_return = total_return
                best_strategy = strat_name

        return best_strategy

    @cleanup_training_memory
    def train_model(self, ticker_data_grouped: Dict[str, pd.DataFrame], business_days: List[datetime]) -> bool:
        """Train or continue training the champion selector."""
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import lightgbm as lgb
        import time
        import warnings
        import xgboost as xgb

        min_day = 1
        max_day = self.day_count - self.forward_days
        if max_day <= min_day or self.day_count < 3:
            print(f"   ⚠️ AI Champion: Need more history before training (have {self.day_count} days)")
            return False

        training_samples = []
        step = 2 if (max_day - min_day) > 40 else 1

        for day_idx in range(min_day, max_day, step):
            if day_idx >= len(business_days):
                continue

            label = self._get_best_strategy_forward(day_idx)
            if label is None:
                continue

            features = self.extract_features(ticker_data_grouped, business_days[day_idx], day_idx=day_idx)
            features["label"] = label
            training_samples.append(features)

        if len(training_samples) < 8:
            print(f"   ⚠️ AI Champion: Only {len(training_samples)} samples available, keeping existing model")
            return True

        train_df = pd.DataFrame(training_samples)
        self.training_data = training_samples

        le = LabelEncoder()
        y = le.fit_transform(train_df["label"])
        if len(np.unique(y)) < 2:
            print("   ⚠️ AI Champion: Only one class present, skipping retrain")
            return False

        feature_cols = [c for c in train_df.columns if c != "label"]
        X = train_df[feature_cols].fillna(0.0).values

        device = "cuda" if XGBOOST_USE_GPU else "cpu"
        has_existing = self.all_models is not None
        if has_existing:
            models = dict(self.all_models)
            if "CatBoost" not in models:
                try:
                    models["CatBoost"] = _clone_model("CatBoost", device)
                except Exception:
                    pass
            print(
                f"   📊 AI Champion: Continuing training on {len(training_samples)} samples "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu)..."
            )
        else:
            models = {
                "XGBoost": _clone_model("XGBoost", device),
                "LightGBM": _clone_model("LightGBM", device),
            }
            try:
                models["CatBoost"] = _clone_model("CatBoost", device)
            except Exception:
                pass
            print(
                f"   📊 AI Champion: Training NEW {list(models.keys())} on {len(training_samples)} samples "
                f"(XGBoost={device}, LightGBM=cpu, CatBoost=cpu)..."
            )

        test_size = min(0.2, max(0.1, len(X) // 3))
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = all(count >= 2 for count in counts) and len(unique) >= 2
        if can_stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        unique_labels_all = np.unique(y_train)
        label_map_all = {old: new for new, old in enumerate(unique_labels_all)}
        y_train_mapped = np.array([label_map_all[label] for label in y_train])
        y_val_mapped = np.array([label_map_all.get(label, 0) for label in y_val])

        trained_models = {}
        model_scores = {}
        model_label_values = {}
        model_use_mapped = {}

        for name, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    print(f"      🔄 {name}: Training...", end=" ", flush=True)
                    start_time = time.time()
                    incremental_failed = False

                    can_increment = has_existing and hasattr(model, "classes_")
                    if can_increment:
                        old_classes = set(model.classes_)
                        old_n_features = getattr(model, "n_features_in_", 0)
                        new_classes = set(np.unique(y_train))
                        if old_classes != new_classes or old_n_features != X_train.shape[1]:
                            can_increment = False
                            print("schema changed...", end=" ", flush=True)

                    if can_increment:
                        try:
                            if name == "XGBoost":
                                model.fit(X_train, y_train, xgb_model=model.get_booster())
                            elif name == "LightGBM":
                                model.fit(X_train, y_train, init_model=model.booster_)
                            elif name == "CatBoost":
                                if catboost_has_trained_trees(model):
                                    configure_catboost_cpu_continuation(model)
                                    model.fit(X_train, y_train, init_model=model)
                                else:
                                    print("(no saved trees yet, training fresh)...", end=" ", flush=True)
                                    incremental_failed = True
                            else:
                                model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            score = accuracy_score(y_val, y_pred)
                            model_label_values[name] = []
                            model_use_mapped[name] = False
                        except Exception as exc:
                            print(f"(incremental failed: {exc}, retraining fresh)...", end=" ", flush=True)
                            incremental_failed = True

                    if not can_increment or incremental_failed:
                        model = _clone_model(name, device)
                        model.fit(X_train, y_train_mapped)
                        y_pred = model.predict(X_val)
                        score = accuracy_score(y_val_mapped, y_pred)
                        model_label_values[name] = list(unique_labels_all)
                        model_use_mapped[name] = True

                    elapsed = time.time() - start_time
                    status = "incremental" if can_increment and not incremental_failed else "fresh"
                    print(f"Accuracy = {score:.3f} ({status}, {elapsed:.1f}s)")
                    trained_models[name] = model
                    model_scores[name] = score
            except Exception as exc:
                print(f"failed: {exc}")

        if not trained_models:
            print("   ⚠️ AI Champion: No models trained successfully")
            return False

        best_name = max(model_scores, key=model_scores.get)
        self.all_models = trained_models
        self.all_scores = model_scores
        self.all_label_values = model_label_values
        self.all_use_mapped_labels = model_use_mapped
        self.best_name = best_name
        self.model = trained_models[best_name]
        self.model_label_values = model_label_values.get(best_name, [])
        self.model_uses_mapped_labels = model_use_mapped.get(best_name, False)
        self.label_encoder = le
        self.feature_cols = feature_cols
        self.last_train_day = self.day_count

        class_dist = {
            le.inverse_transform([label])[0]: count
            for label, count in zip(*np.unique(y, return_counts=True))
        }
        print(f"   ✅ AI Champion: Best model = {best_name} (Accuracy {model_scores[best_name]:.3f})")
        print(f"   📊 AI Champion: Class distribution: {class_dist}")
        self.save_model()
        return True

    def save_model(self):
        """Persist champion model state for continued training."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            if AI_CHAMPION_MODEL_PATH.exists():
                backup_path = AI_CHAMPION_MODEL_PATH.with_suffix(".backup.joblib")
                import shutil

                shutil.copy2(AI_CHAMPION_MODEL_PATH, backup_path)

            payload = ensure_catboost_cpu_metadata(
                {
                    "all_models": self.all_models,
                    "all_scores": self.all_scores,
                    "all_label_values": self.all_label_values,
                    "all_use_mapped_labels": self.all_use_mapped_labels,
                    "best_name": self.best_name,
                    "model": self.model,
                    "model_label_values": self.model_label_values,
                    "model_uses_mapped_labels": self.model_uses_mapped_labels,
                    "label_encoder": self.label_encoder,
                    "feature_cols": self.feature_cols,
                    "training_data": self.training_data,
                    "day_count": self.day_count,
                    "last_train_day": self.last_train_day,
                    "strategy_histories": dict(self.strategy_histories),
                    "active_candidates": sorted(self.active_candidates),
                    "current_strategy": self.current_strategy,
                }
            )
            joblib.dump(payload, AI_CHAMPION_MODEL_PATH)
            save_native_model_artifacts(payload, AI_CHAMPION_MODEL_PATH)
            print(f"   💾 AI Champion: Saved model state to {AI_CHAMPION_MODEL_PATH}")
        except Exception as exc:
            print(f"   ⚠️ AI Champion: Failed to save: {exc}")

    def load_model(self) -> bool:
        """Load saved state if it exists."""
        try:
            if not AI_CHAMPION_MODEL_PATH.exists():
                return False
            data = joblib.load(AI_CHAMPION_MODEL_PATH)
            data = restore_native_model_artifacts(data, AI_CHAMPION_MODEL_PATH, is_classifier=True)
            data, reset_catboost = reset_legacy_catboost_member(data)
            if reset_catboost:
                print("   ♻️ AI Champion: Resetting saved CatBoost member for a clean CPU incremental restart")
            self.model = data.get("model")
            self.all_models = data.get("all_models")
            self.all_scores = data.get("all_scores")
            self.all_label_values = data.get("all_label_values", {})
            self.all_use_mapped_labels = data.get("all_use_mapped_labels", {})
            self.best_name = data.get("best_name")
            self.model_label_values = data.get("model_label_values", [])
            self.model_uses_mapped_labels = data.get("model_uses_mapped_labels", False)
            self.label_encoder = data.get("label_encoder")
            self.feature_cols = data.get("feature_cols", [])
            self.training_data = data.get("training_data", [])
            self.day_count = data.get("day_count", 0)
            self.last_train_day = data.get("last_train_day", 0)
            self.current_strategy = data.get("current_strategy")
            self.active_candidates = set(data.get("active_candidates", CANDIDATE_STRATEGIES))
            loaded_histories = data.get("strategy_histories", {})
            self.strategy_histories = defaultdict(list, loaded_histories)
            print(
                f"   📂 AI Champion: Loaded model, {len(self.training_data)} samples, "
                f"day_count={self.day_count}"
            )
            return True
        except Exception as exc:
            print(f"   ⚠️ AI Champion: Failed to load: {exc}")
            return False

    def release_model_artifacts(self):
        """Drop heavy fitted models while preserving lightweight strategy state."""
        self.model = None
        self.all_models = None

    def _decode_prediction_label(self, raw_label: int) -> Optional[str]:
        if self.label_encoder is None:
            return None

        label_idx = int(raw_label)
        if self.model_uses_mapped_labels:
            if label_idx < 0 or label_idx >= len(self.model_label_values):
                return None
            label_idx = int(self.model_label_values[label_idx])

        try:
            return self.label_encoder.inverse_transform([label_idx])[0]
        except Exception:
            return None

    def _fallback_strategy(self) -> Optional[str]:
        if "ai_elite_market_up" in self.active_candidates:
            return "ai_elite_market_up"
        if "ai_elite" in self.active_candidates:
            return "ai_elite"
        candidates = self._eligible_candidates()
        return candidates[0] if candidates else None

    def predict_best_strategy(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> Optional[str]:
        """Predict the best candidate strategy for the next forward window."""
        if self.model is None or not self.feature_cols:
            return self._fallback_strategy()

        try:
            features = self.extract_features(ticker_data_grouped, current_date)
            X = np.array([[features.get(col, 0.0) for col in self.feature_cols]], dtype=float)

            if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ != X.shape[1]:
                print("   ⚠️ AI Champion: Feature mismatch, falling back")
                return self._fallback_strategy()

            raw_pred = int(self.model.predict(X)[0])
            predicted_strategy = self._decode_prediction_label(raw_pred)
            if predicted_strategy is None:
                return self._fallback_strategy()

            prob_by_strategy: Dict[str, float] = {}
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                classes = getattr(self.model, "classes_", np.arange(len(probs)))
                for raw_class, prob in zip(classes, probs):
                    strategy_name = self._decode_prediction_label(int(raw_class))
                    if strategy_name is not None:
                        prob_by_strategy[strategy_name] = float(prob)

            predicted_prob = prob_by_strategy.get(predicted_strategy, 0.0)
            current_prob = prob_by_strategy.get(self.current_strategy, 0.0) if self.current_strategy else 0.0

            final_strategy = predicted_strategy
            if self.current_strategy in self.active_candidates:
                weak_signal = predicted_prob < self.confidence_threshold
                small_edge = (predicted_prob - current_prob) <= self.hold_margin
                if weak_signal or small_edge:
                    final_strategy = self.current_strategy

            if final_strategy not in self.active_candidates:
                final_strategy = self._fallback_strategy()

            if final_strategy != self.current_strategy:
                print(
                    f"   🔄 AI Champion: Switching to {final_strategy} "
                    f"(predicted {predicted_strategy}, confidence {predicted_prob:.1%})"
                )

            self.current_strategy = final_strategy
            return final_strategy
        except Exception as exc:
            print(f"   ⚠️ AI Champion: Prediction failed: {exc}")
            return self._fallback_strategy()

    def should_retrain(self) -> bool:
        if self.day_count < 3:
            return False
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days


def select_ai_champion_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    predicted_strategy: str,
    ai_elite_models: Dict = None,
) -> List[str]:
    """Dispatch stock selection to the strategy chosen by AI Champion."""
    if not predicted_strategy:
        return []

    if predicted_strategy == "ai_elite":
        from ai_elite_strategy import select_ai_elite_stocks

        return select_ai_elite_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date=current_date,
            top_n=top_n,
            per_ticker_models=ai_elite_models,
        )

    if predicted_strategy == "ai_elite_market_up":
        from ai_elite_market_up_strategy import select_ai_elite_market_up_stocks

        return select_ai_elite_market_up_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date=current_date,
            top_n=top_n,
            per_ticker_models=ai_elite_models,
        )

    if predicted_strategy == "ai_elite_filtered":
        from ai_elite_filtered_strategy import select_ai_elite_filtered_stocks

        return select_ai_elite_filtered_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date=current_date,
            top_n=top_n,
            per_ticker_models=ai_elite_models,
        )

    if predicted_strategy == "multi_tf_ensemble":
        from multi_timeframe_ensemble import select_multi_timeframe_stocks

        return select_multi_timeframe_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date=current_date,
            top_n=top_n,
        )

    print(f"   ⚠️ AI Champion: Unknown predicted strategy '{predicted_strategy}', returning no selection")
    print("   ⚠️ AI Champion: No selection (unknown strategy)")
    return []
