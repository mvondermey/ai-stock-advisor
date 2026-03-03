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
        
        # Sample from HISTORICAL data (before backtest start) using ticker data directly
        # This uses data that exists in ticker_data_grouped before current_date
        for ticker, data in ticker_data_grouped.items():
            if data is None or len(data) < LOOKBACK_DAYS + FORWARD_DAYS + 10:
                continue
                
            # Get dates from ticker data that are before current_date - FORWARD_DAYS (to have forward returns)
            available_dates = data.index[data.index < current_date - timedelta(days=FORWARD_DAYS + 5)]
            if len(available_dates) < LOOKBACK_DAYS:
                continue
            
            # Sample every 5th date to reduce size
            sample_dates = available_dates[LOOKBACK_DAYS::5]
            
            for sample_date in sample_dates[-100:]:  # Use last 100 sample points per ticker
                features = calculate_universal_features(ticker, data, sample_date)
                if features is None:
                    continue
                    
                forward_ret = calculate_forward_return(data, sample_date, FORWARD_DAYS)
                if forward_ret is None:
                    continue
                
                X_list.append(list(features.values()))
                y_list.append(forward_ret)
                
            if len(X_list) >= 10000:  # Cap at 10k samples
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
        
        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=0
            )
            self.model.fit(X_scaled, y)
        
        self.last_train_day = self.day_count
        print(f"   ✅ Universal Model: Trained on {len(X_list)} samples from {len(ticker_data_grouped)} tickers")
        
        # Save model to disk
        self.save_model()
        
        return True
    
    def save_model(self):
        """Save model and scaler to disk."""
        try:
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, UNIVERSAL_MODEL_PATH)
            joblib.dump(self.scaler, UNIVERSAL_SCALER_PATH)
            print(f"   💾 Universal Model: Saved to {UNIVERSAL_MODEL_PATH}")
        except Exception as e:
            print(f"   ⚠️ Universal Model: Failed to save: {e}")
    
    def load_model(self) -> bool:
        """Load model and scaler from disk."""
        try:
            if UNIVERSAL_MODEL_PATH.exists() and UNIVERSAL_SCALER_PATH.exists():
                self.model = joblib.load(UNIVERSAL_MODEL_PATH)
                self.scaler = joblib.load(UNIVERSAL_SCALER_PATH)
                self.feature_cols = ['mom_5d', 'mom_10d', 'mom_20d', 'mom_60d', 'volatility_20d', 
                                     'volatility_60d', 'price_vs_sma20', 'price_vs_sma50', 'sma_trend',
                                     'vol_ratio', 'atr_pct', 'mom_accel', 'risk_adj_mom', 'drawdown', 'rsi']
                print(f"   📂 Universal Model: Loaded from disk")
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
                X = np.array([list(features.values())])
                X_scaled = self.scaler.transform(X)
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
