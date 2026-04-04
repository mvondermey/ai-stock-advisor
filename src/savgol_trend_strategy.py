"""
Savitzky-Golay Trend Strategy.

This version keeps the pooled base-model architecture but strengthens the
signal with richer medium-horizon features, explicit market context, and
more conservative selection logic.
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
    SAVGOL_TREND_FALLBACK_TO_MOMENTUM,
    SAVGOL_TREND_FORWARD_DAYS,
    SAVGOL_TREND_HOLD_MARGIN,
    SAVGOL_TREND_LOOKBACK_DAYS,
    SAVGOL_TREND_MIN_MODEL_SPEARMAN,
    SAVGOL_TREND_MIN_PREDICTED_EDGE,
    SAVGOL_TREND_MIN_SAMPLES,
    SAVGOL_TREND_MIN_SCORE_SPREAD,
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


def _series_pct_change(series: pd.Series, periods: int) -> float:
    """Return trailing percent change over a fixed number of observations."""
    if len(series) <= periods:
        return 0.0
    end_value = float(series.iloc[-1])
    start_value = float(series.iloc[-periods - 1])
    return _safe_pct_change(end_value, start_value)


def _series_volatility_pct(series: pd.Series, periods: int) -> float:
    """Return annualized volatility over a trailing window."""
    if len(series) <= periods:
        return 0.0
    returns = series.pct_change().dropna().tail(periods)
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns) * np.sqrt(252) * 100.0)


def _calculate_market_context(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
) -> Dict[str, float]:
    """Build lightweight market context features for the current date."""
    proxy_candidates = ("SPY", "QQQ", "VTI", "DIA")
    proxy_series: Optional[pd.Series] = None

    for ticker in proxy_candidates:
        data = ticker_data_grouped.get(ticker)
        if data is None or "Close" not in data.columns:
            continue
        current_date_norm = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date_norm]
        close = hist["Close"].dropna()
        if len(close) >= 61:
            proxy_series = close
            break

    if proxy_series is not None:
        market_return_5d = _series_pct_change(proxy_series, 5)
        market_return_20d = _series_pct_change(proxy_series, 20)
        market_return_60d = _series_pct_change(proxy_series, 60)
        market_volatility_20d = _series_volatility_pct(proxy_series, 20)
    else:
        market_return_5d = 0.0
        market_return_20d = 0.0
        market_return_60d = 0.0
        market_volatility_20d = 0.0

    breadth_20d_values: List[float] = []
    breadth_60d_values: List[float] = []
    for data in ticker_data_grouped.values():
        if data is None or "Close" not in data.columns:
            continue
        current_date_norm = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date_norm]
        close = hist["Close"].dropna()
        if len(close) >= 21:
            breadth_20d_values.append(_series_pct_change(close, 20))
        if len(close) >= 61:
            breadth_60d_values.append(_series_pct_change(close, 60))
        if len(breadth_20d_values) >= 250 and len(breadth_60d_values) >= 250:
            break

    breadth_20d = float(np.mean(np.array(breadth_20d_values) > 0)) if breadth_20d_values else 0.5
    breadth_60d = float(np.mean(np.array(breadth_60d_values) > 0)) if breadth_60d_values else 0.5

    return {
        "market_return_5d": market_return_5d,
        "market_return_20d": market_return_20d,
        "market_return_60d": market_return_60d,
        "market_volatility_20d": market_volatility_20d,
        "market_breadth_20d": breadth_20d,
        "market_breadth_60d": breadth_60d,
    }


def _calculate_market_forward_return(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    forward_days: int,
) -> float:
    """Estimate the market forward return using a broad proxy, else a basket average."""
    proxy_candidates = ("SPY", "QQQ", "VTI", "DIA")
    for ticker in proxy_candidates:
        data = ticker_data_grouped.get(ticker)
        if data is None:
            continue
        market_return = calculate_forward_return(data, current_date, forward_days)
        if market_return is not None:
            return float(market_return)

    fallback_returns: List[float] = []
    for data in ticker_data_grouped.values():
        if data is None:
            continue
        market_return = calculate_forward_return(data, current_date, forward_days)
        if market_return is not None:
            fallback_returns.append(float(market_return))
        if len(fallback_returns) >= 50:
            break
    return float(np.mean(fallback_returns)) if fallback_returns else 0.0


def _prepare_label(raw_label: float) -> float:
    """Clip extreme labels so noisy tails do not dominate training."""
    return float(np.clip(raw_label, -100.0, 100.0))


def calculate_savgol_features(
    ticker: str,
    data: pd.DataFrame,
    current_date: datetime,
    lookback_days: int = SAVGOL_TREND_LOOKBACK_DAYS,
    market_context: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    """Calculate local polynomial trend and medium-horizon context features."""
    del ticker  # Still pooled: no ticker ID is passed as a model feature.

    try:
        current_date = _normalize_current_date(data, current_date)
        hist = data.loc[:current_date]
        if len(hist) < lookback_days:
            return None

        close_series = hist["Close"].dropna()
        close = close_series.values[-lookback_days:]
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
            float(np.polyfit(x, sg_long[-len(x):], 1)[0] * 100.0)
            if len(x) >= 2
            else 0.0
        )

        residual = log_close - sg_long
        residual_vol_20d = float(np.std(residual[-20:]) * 100.0) if len(residual) >= 20 else 0.0

        high_20 = np.max(close[-20:])
        atr_14 = float(np.mean(high[-14:] - low[-14:])) if len(high) >= 14 and len(low) >= 14 else 0.0
        avg_vol_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        avg_vol_60 = float(np.mean(volume)) if len(volume) > 0 else 1.0

        perf_5d = _series_pct_change(close_series, 5)
        perf_20d = _series_pct_change(close_series, 20)
        perf_3m = _series_pct_change(close_series, 63)
        perf_6m = _series_pct_change(close_series, 126)
        perf_1y = _series_pct_change(close_series, 252)
        daily_volatility = _series_volatility_pct(close_series, 20)
        risk_adj_mom_3m = perf_3m / (np.sqrt(max(daily_volatility, 5.0)) if daily_volatility > 0 else np.sqrt(5.0))
        dip_score = perf_1y - perf_3m
        mom_accel = perf_3m - perf_6m

        rsi_14 = 50.0
        if len(returns) >= 14:
            gains = np.clip(returns, 0.0, None)
            losses = np.clip(-returns, 0.0, None)
            avg_gain = float(np.mean(gains[-14:]))
            avg_loss = float(np.mean(losses[-14:]))
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_14 = 100.0 - (100.0 / (1.0 + rs))
            elif avg_gain > 0:
                rsi_14 = 100.0

        bollinger_position = 0.5
        sma20_distance = 0.0
        sma50_distance = 0.0
        if len(close_series) >= 20:
            sma20 = float(close_series.tail(20).mean())
            std20 = float(close_series.tail(20).std())
            if sma20 > 0 and std20 > 0:
                upper_band = sma20 + 2.0 * std20
                lower_band = sma20 - 2.0 * std20
                if upper_band > lower_band:
                    bollinger_position = float(np.clip((close[-1] - lower_band) / (upper_band - lower_band), 0.0, 1.0))
                sma20_distance = _safe_pct_change(close[-1], sma20)
        if len(close_series) >= 50:
            sma50 = float(close_series.tail(50).mean())
            if sma50 > 0:
                sma50_distance = _safe_pct_change(close[-1], sma50)

        macd = 0.0
        if len(close_series) >= 35:
            ema12 = close_series.ewm(span=12, adjust=False).mean()
            ema26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd = float(macd_line.iloc[-1] - signal_line.iloc[-1])

        context = market_context or {}
        market_return_20d = float(context.get("market_return_20d", 0.0))
        market_return_60d = float(context.get("market_return_60d", 0.0))

        return {
            "mom_5d": perf_5d,
            "mom_10d": _series_pct_change(close_series, 10),
            "mom_20d": perf_20d,
            "mom_40d": _series_pct_change(close_series, 40),
            "perf_3m": perf_3m,
            "perf_6m": perf_6m,
            "perf_1y": perf_1y,
            "risk_adj_mom_3m": risk_adj_mom_3m,
            "dip_score": dip_score,
            "mom_accel": mom_accel,
            "volatility_10d": float(np.std(returns[-10:]) * np.sqrt(252) * 100.0) if len(returns) >= 10 else 0.0,
            "volatility_20d": float(np.std(returns[-20:]) * np.sqrt(252) * 100.0) if len(returns) >= 20 else 0.0,
            "drawdown_20d": _safe_pct_change(close[-1], high_20),
            "atr_pct_14d": (atr_14 / close[-1]) * 100.0 if close[-1] > 0 else 0.0,
            "volume_ratio_20_60": avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else 1.0,
            "rsi_14": rsi_14,
            "bollinger_position": bollinger_position,
            "sma20_distance": sma20_distance,
            "sma50_distance": sma50_distance,
            "macd": macd,
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
            "market_return_5d": float(context.get("market_return_5d", 0.0)),
            "market_return_20d": market_return_20d,
            "market_return_60d": market_return_60d,
            "market_volatility_20d": float(context.get("market_volatility_20d", 0.0)),
            "market_breadth_20d": float(context.get("market_breadth_20d", 0.5)),
            "market_breadth_60d": float(context.get("market_breadth_60d", 0.5)),
            "rel_strength_20d": perf_20d - market_return_20d,
            "rel_strength_60d": _series_pct_change(close_series, 60) - market_return_60d,
        }
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

        return cb.CatBoostRegressor(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="CPU",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            thread_count=1,
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
    """Pooled ML strategy driven by Savitzky-Golay and market-context features."""

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

    @cleanup_training_memory
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
        market_context_cache: Dict[pd.Timestamp, Dict[str, float]] = {}
        market_forward_cache: Dict[pd.Timestamp, float] = {}

        for ticker, data in ticker_data_grouped.items():
            if data is None or len(data) < self.lookback_days + self.forward_days + 10:
                continue

            cutoff = current_date - timedelta(days=self.forward_days + 7)
            available_dates = data.index[data.index < cutoff]
            if len(available_dates) < self.lookback_days:
                continue

            sample_dates = available_dates[self.lookback_days::5]
            for sample_date in sample_dates[-120:]:
                sample_key = pd.Timestamp(sample_date)
                market_context = market_context_cache.get(sample_key)
                if market_context is None:
                    market_context = _calculate_market_context(ticker_data_grouped, sample_date)
                    market_context_cache[sample_key] = market_context

                features = calculate_savgol_features(
                    ticker,
                    data,
                    sample_date,
                    lookback_days=self.lookback_days,
                    market_context=market_context,
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

                market_forward = market_forward_cache.get(sample_key)
                if market_forward is None:
                    market_forward = _calculate_market_forward_return(
                        ticker_data_grouped,
                        sample_date,
                        self.forward_days,
                    )
                    market_forward_cache[sample_key] = market_forward

                excess_return = float(forward_ret) - float(market_forward)
                risk_floor = np.sqrt(max(features.get("volatility_20d", 0.0), 5.0))
                target = _prepare_label(excess_return / risk_floor)
                samples.append((sample_key, features, target))

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

        import time
        import warnings

        models, device = _build_model_set()
        has_existing = self.all_models is not None and len(self.all_models) > 0
        if has_existing:
            models = dict(self.all_models)
            for name in ("XGBoost", "LightGBM", "CatBoost"):
                if name not in models:
                    try:
                        models[name] = _fresh_model(name, device)
                    except Exception:
                        continue

        print(f"   📊 Train/Val split: {len(X_train)} train, {len(X_val)} val samples")

        trained_models: Dict[str, object] = {}
        model_scores: Dict[str, float] = {}

        for name, model in models.items():
            print(f"      🔄 {name}: Training started...", end=" ", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    incremental_failed = False
                    train_start = time.perf_counter()

                    if has_existing:
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
                                    incremental_failed = True
                            else:
                                model.fit(X_train, y_train)

                            quick_pred = np.asarray(model.predict(X_val.iloc[:min(len(X_val), 100)]), dtype=float)
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

                train_elapsed = time.perf_counter() - train_start

                val_pred = np.asarray(model.predict(X_val), dtype=float)
                if np.any(np.isnan(val_pred)) or np.any(np.isinf(val_pred)):
                    continue

                score = spearmanr(y_val, val_pred).correlation if len(y_val) > 1 else 0.0
                if score is None or np.isnan(score):
                    score = 0.0

                trained_models[name] = model
                model_scores[name] = float(score)
                mode = "incremental" if has_existing and not incremental_failed else "fresh"
                print(f"spearman = {score:.3f} ({mode}, {train_elapsed:.1f}s)")
            except Exception as exc:
                print(f"failed: {exc}")

        if not trained_models:
            print("   ⚠️ SavGol Trend: No models trained successfully")
            return False

        self.best_name = max(model_scores, key=model_scores.get)
        self.model = trained_models[self.best_name]
        self.all_models = trained_models
        self.all_scores = model_scores
        self.last_train_day = self.day_count
        save_succeeded = self.save_model(
            current_date=current_date,
            train_start=train_samples[0][0],
            train_end=train_samples[-1][0],
        )
        if save_succeeded:
            print(
                f"   ✅ SavGol Trend: Saved {len(trained_models)} models. Best = {self.best_name} "
                f"(spearman {model_scores[self.best_name]:.3f})"
            )
        else:
            print(
                f"   ⚠️ SavGol Trend: Trained {len(trained_models)} models, but save failed. "
                f"Best = {self.best_name} (spearman {model_scores[self.best_name]:.3f})"
            )
        return True

    def save_model(
        self,
        current_date: Optional[datetime] = None,
        train_start: Optional[pd.Timestamp] = None,
        train_end: Optional[pd.Timestamp] = None,
    ) -> bool:
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
            payload = ensure_catboost_cpu_metadata(
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
                }
            )
            joblib.dump(payload, SAVGOL_TREND_MODEL_PATH)
            save_native_model_artifacts(payload, SAVGOL_TREND_MODEL_PATH)
            print(f"   💾 SavGol Trend: Saved model to {SAVGOL_TREND_MODEL_PATH}")
            return True
        except Exception as exc:
            print(f"   ⚠️ SavGol Trend: Failed to save model: {exc}")
            return False

    def load_model(self) -> bool:
        """Load a previously trained model if available."""
        try:
            if not SAVGOL_TREND_MODEL_PATH.exists():
                return False

            saved = joblib.load(SAVGOL_TREND_MODEL_PATH)
            saved = restore_native_model_artifacts(saved, SAVGOL_TREND_MODEL_PATH)
            if isinstance(saved, dict):
                saved, reset_catboost = reset_legacy_catboost_member(saved)
                if reset_catboost:
                    print("   ♻️ SavGol Trend: Resetting saved CatBoost member for a clean CPU incremental restart")
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

    def release_model_artifacts(self):
        """Drop heavy fitted models while preserving lightweight state."""
        self.model = None
        self.all_models = None

    def predict_returns(
        self,
        tickers: List[str],
        ticker_data_grouped: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Predict scores for the current candidate universe."""
        if self.model is None or not self.feature_cols:
            return []

        market_context = _calculate_market_context(ticker_data_grouped, current_date)
        predictions: List[Tuple[str, float, Dict[str, float]]] = []
        for ticker in tickers:
            data = ticker_data_grouped.get(ticker)
            if data is None:
                continue

            features = calculate_savgol_features(
                ticker,
                data,
                current_date,
                lookback_days=self.lookback_days,
                market_context=market_context,
            )
            if features is None:
                continue

            row = pd.DataFrame(
                [[features.get(col, 0.0) for col in self.feature_cols]],
                columns=self.feature_cols,
            )
            try:
                pred = float(self.model.predict(row)[0])
                predictions.append((ticker, pred, features))
            except Exception:
                continue

        return predictions

    def fallback_selection(
        self,
        predictions: List[Tuple[str, float, Dict[str, float]]],
        top_n: int,
    ) -> List[str]:
        """Fallback to a simpler momentum/risk-adjusted ranking."""
        ranked = sorted(
            predictions,
            key=lambda item: (
                item[2].get("risk_adj_mom_3m", 0.0),
                item[2].get("perf_3m", 0.0),
                item[2].get("sg_long_slope", 0.0),
            ),
            reverse=True,
        )
        return [ticker for ticker, _, _ in ranked[:top_n]]


def select_savgol_trend_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int,
    model: SavgolTrendStrategy,
    business_days: List[datetime],
    current_day_idx: int,
    current_holdings: Optional[List[str]] = None,
) -> List[str]:
    """Select tickers with confidence gating and lower-turnover ranking."""
    if model.should_retrain():
        model.train_model(ticker_data_grouped, business_days, current_day_idx)

    predictions = model.predict_returns(all_tickers, ticker_data_grouped, current_date)
    if not predictions:
        return []

    current_holdings = current_holdings or []
    predictions.sort(key=lambda item: item[1], reverse=True)

    best_model_score = float(model.all_scores.get(model.best_name, 0.0)) if model.all_scores and model.best_name else 0.0
    top_score = float(predictions[0][1])
    cutoff_score = float(predictions[min(top_n - 1, len(predictions) - 1)][1])
    score_spread = top_score - cutoff_score

    adjusted_predictions = []
    for ticker, pred, features in predictions:
        adjusted_pred = pred + (SAVGOL_TREND_HOLD_MARGIN if ticker in current_holdings else 0.0)
        adjusted_predictions.append((ticker, adjusted_pred, pred, features))
    adjusted_predictions.sort(key=lambda item: item[1], reverse=True)

    low_confidence = (
        best_model_score < SAVGOL_TREND_MIN_MODEL_SPEARMAN
        or top_score < SAVGOL_TREND_MIN_PREDICTED_EDGE
        or score_spread < SAVGOL_TREND_MIN_SCORE_SPREAD
    )

    if low_confidence:
        print(
            "   ⚠️ SavGol Trend: Low confidence "
            f"(spearman={best_model_score:.3f}, top={top_score:+.3f}, spread={score_spread:.3f})"
        )
        if current_holdings:
            return []
        if SAVGOL_TREND_FALLBACK_TO_MOMENTUM:
            fallback = model.fallback_selection(predictions, top_n)
            print(f"   🔄 SavGol Trend: Falling back to momentum-style ranking ({len(fallback)} picks)")
            return fallback
        return []

    selected = [ticker for ticker, _, _, _ in adjusted_predictions[:top_n]]
    print(
        f"   ✅ SavGol Trend: Selected {len(selected)} tickers "
        f"(spearman={best_model_score:.3f}, top={top_score:+.3f}, spread={score_spread:.3f})"
    )
    return selected
