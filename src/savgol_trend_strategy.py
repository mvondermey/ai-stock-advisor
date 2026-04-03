"""
Savitzky-Golay Trend Strategy.

This strategy trains a pooled regression model across all tickers using
locally smoothed price-trend features. It is designed to answer a simple
question for each rebalance date: which tickers have the strongest
next-horizon trend after denoising recent price action?
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

from config import (
    SAVGOL_TREND_FORWARD_DAYS,
    SAVGOL_TREND_LOOKBACK_DAYS,
    SAVGOL_TREND_MIN_SAMPLES,
    SAVGOL_TREND_RETRAIN_DAYS,
    XGBOOST_USE_GPU,
)


MODEL_SAVE_DIR = Path("logs/models")
SAVGOL_TREND_MODEL_PATH = MODEL_SAVE_DIR / "savgol_trend_model.joblib"


def _normalize_current_date(data: pd.DataFrame, current_date: datetime) -> datetime:
    """Align naive dates to the DataFrame timezone when needed."""
    if hasattr(data.index, "tz") and data.index.tz is not None and current_date.tzinfo is None:
        return current_date.replace(tzinfo=data.index.tz)
    return current_date


def _odd_window(length: int, target: int, minimum: int = 5) -> int:
    """Return a valid odd Savitzky-Golay window for the available history."""
    window = min(length, target)
    if window % 2 == 0:
        window -= 1
    return window if window >= minimum else 0


def _safe_pct_change(end_value: float, start_value: float) -> float:
    if start_value <= 0:
        return 0.0
    return (end_value / start_value - 1.0) * 100.0


def calculate_savgol_features(
    ticker: str,
    data: pd.DataFrame,
    current_date: datetime,
    lookback_days: int = SAVGOL_TREND_LOOKBACK_DAYS,
) -> Optional[Dict[str, float]]:
    """Calculate local polynomial trend features for a single ticker."""
    del ticker  # Features are intentionally ticker-agnostic for pooled training.

    try:
        current_date = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date]
        if len(hist) < lookback_days:
            return None

        close = hist["Close"].dropna().values[-lookback_days:]
        if len(close) < lookback_days or close[-1] <= 0:
            return None

        volume = (
            hist["Volume"].fillna(0).values[-lookback_days:]
            if "Volume" in hist.columns
            else np.ones(len(close))
        )
        high = hist["High"].ffill().fillna(hist["Close"]).values[-lookback_days:] if "High" in hist.columns else close
        low = hist["Low"].ffill().fillna(hist["Close"]).values[-lookback_days:] if "Low" in hist.columns else close

        log_close = np.log(np.clip(close.astype(float), 1e-12, None))
        returns = np.diff(close) / np.clip(close[:-1], 1e-12, None)

        short_window = _odd_window(len(log_close), 9)
        long_window = _odd_window(len(log_close), 21)
        if short_window == 0 or long_window == 0:
            return None

        short_poly = 2 if short_window >= 5 else 1
        long_poly = 3 if long_window >= 7 else 2
        if short_window <= short_poly or long_window <= long_poly:
            return None

        sg_short = savgol_filter(log_close, window_length=short_window, polyorder=short_poly, mode="interp")
        sg_long = savgol_filter(log_close, window_length=long_window, polyorder=long_poly, mode="interp")

        short_slope = (sg_short[-1] - sg_short[-2]) * 100.0 if len(sg_short) >= 2 else 0.0
        long_slope = (sg_long[-1] - sg_long[-2]) * 100.0 if len(sg_long) >= 2 else 0.0
        short_curvature = (
            (sg_short[-1] - 2 * sg_short[-2] + sg_short[-3]) * 10000.0
            if len(sg_short) >= 3
            else 0.0
        )
        long_curvature = (
            (sg_long[-1] - 2 * sg_long[-2] + sg_long[-3]) * 10000.0
            if len(sg_long) >= 3
            else 0.0
        )

        short_diffs = np.diff(sg_short[-10:]) if len(sg_short) >= 10 else np.diff(sg_short)
        long_diffs = np.diff(sg_long[-10:]) if len(sg_long) >= 10 else np.diff(sg_long)
        sg_trend_stability = float(np.mean(short_diffs > 0)) if len(short_diffs) > 0 else 0.5
        sg_long_stability = float(np.mean(long_diffs > 0)) if len(long_diffs) > 0 else 0.5

        x = np.arange(min(10, len(sg_long)))
        sg_regression_slope = (
            float(np.polyfit(x, sg_long[-len(x) :], 1)[0] * 100.0)
            if len(x) >= 2
            else 0.0
        )

        residual = log_close - sg_long
        residual_vol_20d = float(np.std(residual[-20:]) * 100.0) if len(residual) >= 20 else 0.0

        high_20 = np.max(close[-20:])
        atr_14 = float(np.mean(high[-14:] - low[-14:])) if len(high) >= 14 and len(low) >= 14 else 0.0
        avg_vol_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        avg_vol_60 = float(np.mean(volume)) if len(volume) > 0 else 1.0

        features = {
            "mom_5d": _safe_pct_change(close[-1], close[-5]),
            "mom_10d": _safe_pct_change(close[-1], close[-10]),
            "mom_20d": _safe_pct_change(close[-1], close[-20]),
            "mom_40d": _safe_pct_change(close[-1], close[-40]),
            "volatility_10d": float(np.std(returns[-10:]) * np.sqrt(252) * 100.0) if len(returns) >= 10 else 0.0,
            "volatility_20d": float(np.std(returns[-20:]) * np.sqrt(252) * 100.0) if len(returns) >= 20 else 0.0,
            "drawdown_20d": _safe_pct_change(close[-1], high_20),
            "atr_pct_14d": (atr_14 / close[-1]) * 100.0 if close[-1] > 0 else 0.0,
            "volume_ratio_20_60": avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else 1.0,
            "price_vs_sg_short": (log_close[-1] - sg_short[-1]) * 100.0,
            "price_vs_sg_long": (log_close[-1] - sg_long[-1]) * 100.0,
            "sg_short_slope": short_slope,
            "sg_long_slope": long_slope,
            "sg_short_curvature": short_curvature,
            "sg_long_curvature": long_curvature,
            "sg_trend_spread": (sg_short[-1] - sg_long[-1]) * 100.0,
            "sg_trend_stability": sg_trend_stability,
            "sg_long_stability": sg_long_stability,
            "sg_regression_slope": sg_regression_slope,
            "sg_residual_vol_20d": residual_vol_20d,
        }
        return features
    except Exception:
        return None


def calculate_forward_return(
    data: pd.DataFrame,
    current_date: datetime,
    forward_days: int = SAVGOL_TREND_FORWARD_DAYS,
) -> Optional[float]:
    """Calculate the realized forward return used as the training label."""
    try:
        current_date = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date]
        if len(hist) == 0:
            return None

        current_price = hist["Close"].iloc[-1]
        future_date = current_date + timedelta(days=forward_days + 7)
        future_data = data.loc[current_date:future_date]
        if len(future_data) < max(2, forward_days // 2):
            return None

        future_idx = min(forward_days, len(future_data) - 1)
        future_price = future_data["Close"].iloc[future_idx]
        if current_price > 0 and future_price > 0:
            return (future_price / current_price - 1.0) * 100.0
        return None
    except Exception:
        return None


def _fresh_model(name: str, device: str):
    """Create a new model instance for the requested backend."""
    import xgboost as xgb
    import lightgbm as lgb

    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            tree_method="hist",
            device=device,
            verbosity=0,
            n_jobs=-1,
        )
    if name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    if name == "CatBoost":
        import catboost as cb

        task_type = "GPU" if XGBOOST_USE_GPU else "CPU"
        return cb.CatBoostRegressor(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type=task_type,
            random_seed=42,
            verbose=0,
        )
    raise ValueError(f"Unknown model backend: {name}")


def _build_model_set():
    """Build the default ensemble used for training and incremental continuation."""
    device = "cuda" if XGBOOST_USE_GPU else "cpu"
    models = {
        "XGBoost": _fresh_model("XGBoost", device),
        "LightGBM": _fresh_model("LightGBM", device),
    }
    try:
        models["CatBoost"] = _fresh_model("CatBoost", device)
    except Exception:
        pass
    return models, device


class SavgolTrendStrategy:
    """Pooled ML strategy driven by Savitzky-Golay trend features."""

    def __init__(
        self,
        retrain_days: int = SAVGOL_TREND_RETRAIN_DAYS,
        min_samples: int = SAVGOL_TREND_MIN_SAMPLES,
        lookback_days: int = SAVGOL_TREND_LOOKBACK_DAYS,
        forward_days: int = SAVGOL_TREND_FORWARD_DAYS,
    ):
        self.retrain_days = retrain_days
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        self.forward_days = forward_days
        self.model = None
        self.all_models: Optional[Dict[str, object]] = None
        self.all_scores: Optional[Dict[str, float]] = None
        self.best_name: Optional[str] = None
        self.feature_cols: List[str] = []
        self.day_count = 0
        self.last_train_day = 0

    def should_retrain(self) -> bool:
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days

    def increment_day(self):
        self.day_count += 1

    def train_model(
        self,
        ticker_data_grouped: Dict[str, pd.DataFrame],
        business_days: List[datetime],
        current_day_idx: int,
    ) -> bool:
        """Train the pooled model on historical ticker samples only."""
        if not business_days:
            return False

        if current_day_idx >= len(business_days):
            current_day_idx = len(business_days) - 1
        current_date = business_days[current_day_idx] if current_day_idx >= 0 else business_days[0]

        print("   🧠 SavGol Trend: Training pooled model...")

        samples: List[Tuple[pd.Timestamp, Dict[str, float], float]] = []
        for ticker, data in ticker_data_grouped.items():
            if data is None or len(data) < self.lookback_days + self.forward_days + 10:
                continue

            cutoff = current_date - timedelta(days=self.forward_days + 7)
            available_dates = data.index[data.index < cutoff]
            if len(available_dates) < self.lookback_days:
                continue

            sample_dates = available_dates[self.lookback_days :: 5]
            for sample_date in sample_dates[-120:]:
                features = calculate_savgol_features(
                    ticker,
                    data,
                    sample_date,
                    lookback_days=self.lookback_days,
                )
                if features is None:
                    continue
                forward_ret = calculate_forward_return(
                    data,
                    sample_date,
                    forward_days=self.forward_days,
                )
                if forward_ret is None:
                    continue
                samples.append((pd.Timestamp(sample_date), features, float(forward_ret)))

            if len(samples) >= 12000:
                break

        if len(samples) < self.min_samples:
            print(f"   ⚠️ SavGol Trend: Not enough samples ({len(samples)} < {self.min_samples})")
            return False

        samples.sort(key=lambda row: row[0])
        split_idx = max(int(len(samples) * 0.8), 1)
        split_idx = min(split_idx, len(samples) - 1)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        self.feature_cols = list(train_samples[0][1].keys())

        X_train = (
            pd.DataFrame([row[1] for row in train_samples], columns=self.feature_cols)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_train = np.array([row[2] for row in train_samples], dtype=float)
        X_val = (
            pd.DataFrame([row[1] for row in val_samples], columns=self.feature_cols)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_val = np.array([row[2] for row in val_samples], dtype=float)

        models, device = _build_model_set()
        has_existing = self.all_models is not None and len(self.all_models) > 0
        if has_existing:
            models = self.all_models
            for name in ("XGBoost", "LightGBM", "CatBoost"):
                if name not in models:
                    try:
                        models[name] = _fresh_model(name, device)
                    except Exception:
                        continue
            print(f"   📊 SavGol Trend: Continuing training on {len(X_train)} samples ({device})...")
        else:
            print(f"   📊 SavGol Trend: Training NEW {list(models.keys())} on {len(X_train)} samples ({device})...")

        trained_models: Dict[str, object] = {}
        model_scores: Dict[str, float] = {}

        import warnings

        for name, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    incremental_failed = False

                    if has_existing:
                        try:
                            if name == "XGBoost":
                                model.fit(X_train, y_train, xgb_model=model.get_booster())
                            elif name == "LightGBM":
                                model.fit(X_train, y_train, init_model=model.booster_)
                            elif name == "CatBoost":
                                model._init_params["task_type"] = "CPU"
                                model.fit(X_train, y_train, init_model=model)
                            else:
                                model.fit(X_train, y_train)

                            quick_pred = np.asarray(model.predict(X_val.iloc[: min(len(X_val), 100)]), dtype=float)
                            if (
                                np.any(np.isnan(quick_pred))
                                or np.any(np.isinf(quick_pred))
                                or (len(quick_pred) > 0 and np.max(np.abs(quick_pred)) > 1e10)
                            ):
                                incremental_failed = True
                        except Exception:
                            incremental_failed = True

                    if not has_existing or incremental_failed:
                        model = _fresh_model(name, device)
                        model.fit(X_train, y_train)

                val_pred = np.asarray(model.predict(X_val), dtype=float)
                if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
                    continue

                score = spearmanr(y_val, val_pred).correlation if len(y_val) > 1 else 0.0
                if score is None or np.isnan(score):
                    score = 0.0

                trained_models[name] = model
                model_scores[name] = float(score)
                mode = "incremental" if has_existing and not incremental_failed else "fresh"
                print(f"      ✅ SavGol Trend {name}: spearman={score:.3f} ({mode})")
            except Exception as exc:
                print(f"      ⚠️ SavGol Trend {name} failed: {exc}")

        if not trained_models:
            print("   ⚠️ SavGol Trend: No models trained successfully")
            return False

        self.best_name = max(model_scores, key=model_scores.get)
        self.model = trained_models[self.best_name]
        self.all_models = trained_models
        self.all_scores = model_scores
        self.last_train_day = self.day_count
        print(
            f"   ✅ SavGol Trend: Trained {len(trained_models)} models. Best = {self.best_name} "
            f"(val spearman={model_scores[self.best_name]:.3f})"
        )
        self.save_model(current_date=current_date, train_start=train_samples[0][0], train_end=train_samples[-1][0])
        return True

    def save_model(
        self,
        current_date: Optional[datetime] = None,
        train_start: Optional[pd.Timestamp] = None,
        train_end: Optional[pd.Timestamp] = None,
    ):
        """Persist model and training state for later reuse."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            metadata = {}
            if current_date is not None:
                metadata["updated"] = pd.Timestamp(current_date).isoformat()
            if train_start is not None:
                metadata["train_start"] = pd.Timestamp(train_start).isoformat()
            if train_end is not None:
                metadata["train_end"] = pd.Timestamp(train_end).isoformat()
            if self.best_name is not None:
                metadata["best_model"] = self.best_name
            if self.all_scores:
                metadata["all_scores"] = self.all_scores
            joblib.dump(
                {
                    "all_models": self.all_models,
                    "all_scores": self.all_scores,
                    "model": self.model,
                    "best_name": self.best_name,
                    "feature_cols": self.feature_cols,
                    "day_count": self.day_count,
                    "last_train_day": self.last_train_day,
                    "retrain_days": self.retrain_days,
                    "min_samples": self.min_samples,
                    "lookback_days": self.lookback_days,
                    "forward_days": self.forward_days,
                    "metadata": metadata,
                },
                SAVGOL_TREND_MODEL_PATH,
            )
            print(f"   💾 SavGol Trend: Saved model to {SAVGOL_TREND_MODEL_PATH}")
        except Exception as exc:
            print(f"   ⚠️ SavGol Trend: Failed to save model: {exc}")

    def load_model(self) -> bool:
        """Load a previously trained model if available."""
        try:
            if not SAVGOL_TREND_MODEL_PATH.exists():
                return False

            saved = joblib.load(SAVGOL_TREND_MODEL_PATH)
            if isinstance(saved, dict) and "all_models" in saved:
                self.all_models = saved.get("all_models")
                self.all_scores = saved.get("all_scores")
                self.model = saved.get("model")
                self.best_name = saved.get("best_name")
                self.feature_cols = saved.get("feature_cols", [])
                self.day_count = saved.get("day_count", 0)
                self.last_train_day = saved.get("last_train_day", 0)
            else:
                self.model = saved.get("model") if isinstance(saved, dict) else saved
                self.best_name = saved.get("best_name") if isinstance(saved, dict) else None
                self.feature_cols = saved.get("feature_cols", []) if isinstance(saved, dict) else []
                self.day_count = saved.get("day_count", 0) if isinstance(saved, dict) else 0
                self.last_train_day = saved.get("last_train_day", 0) if isinstance(saved, dict) else 0
            print(
                f"   📂 SavGol Trend: Loaded {self.best_name or 'model'} "
                f"(day_count={self.day_count})"
            )
            return self.model is not None
        except Exception as exc:
            print(f"   ⚠️ SavGol Trend: Failed to load model: {exc}")
            return False

    def predict_returns(
        self,
        tickers: List[str],
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> List[Tuple[str, float]]:
        """Predict forward returns for the current candidate universe."""
        if self.model is None or not self.feature_cols:
            return []

        predictions: List[Tuple[str, float]] = []
        for ticker in tickers:
            data = ticker_data_grouped.get(ticker)
            if data is None:
                continue

            features = calculate_savgol_features(
                ticker,
                data,
                current_date,
                lookback_days=self.lookback_days,
            )
            if features is None:
                continue

            row = pd.DataFrame(
                [[features.get(col, 0.0) for col in self.feature_cols]],
                columns=self.feature_cols,
            )
            try:
                pred = float(self.model.predict(row)[0])
                predictions.append((ticker, pred))
            except Exception:
                continue

        return predictions


def select_savgol_trend_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    model: SavgolTrendStrategy,
    business_days: List[datetime],
    current_day_idx: int,
) -> List[str]:
    """Select the top tickers based on predicted forward return."""
    if model.should_retrain():
        model.train_model(ticker_data_grouped, business_days, current_day_idx)

    predictions = model.predict_returns(all_tickers, ticker_data_grouped, current_date)
    if not predictions:
        return []

    predictions.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _ in predictions[:top_n]]
