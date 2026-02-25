"""
AI Regime Strategy: ML predicts which strategy to use based on market conditions

Approach:
1. Track performance of multiple sub-strategies daily
2. Extract market regime features (volatility, trend, dispersion)
3. Train ML model to predict which strategy will perform best in next N days
4. Allocate capital to the predicted best strategy

This is a "meta-strategy" that learns WHEN to use each strategy, not WHAT stocks to pick.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
import os


# Sub-strategies to choose from - ALL major strategies
SUB_STRATEGIES = [
    'risk_adj_mom_3m',      # Risk-Adj Mom 3M
    'risk_adj_mom_6m',      # Risk-Adj Mom 6M
    'risk_adj_mom',         # Risk-Adj Mom 1Y
    'elite_hybrid',         # Elite Hybrid
    'elite_risk',           # Elite Risk
    'ai_elite',             # AI Elite
    'momentum_volatility_hybrid_6m',  # Mom-Vol Hybrid 6M
    'trend_atr',            # Trend ATR
    'dual_momentum',        # Dual Momentum
    'static_bh_1y',         # Static BH 1Y
    'static_bh_3m',         # Static BH 3M
]

# Regime features to extract
REGIME_LOOKBACK = 20  # Days to look back for regime features


class AIRegimeAllocator:
    """
    ML-based regime detection and strategy allocation.
    
    Learns which strategy performs best under different market conditions.
    """
    
    def __init__(self, retrain_days: int = 1, forward_days: int = 20):
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
                self.strategy_histories[name].append(strategy_values[name])
        self.day_count += 1
        
    def extract_regime_features(self, ticker_data_grouped: Dict[str, pd.DataFrame], 
                                 current_date: datetime) -> Optional[Dict[str, float]]:
        """
        Extract market regime features from price data.
        
        Features:
        - Market volatility (avg volatility across stocks)
        - Market trend (avg momentum across stocks)
        - Cross-sectional dispersion (std of returns across stocks)
        - Strategy momentum (recent performance of each strategy)
        - Volatility regime (high/low vol environment)
        """
        try:
            features = {}
            
            # Collect returns and volatilities across all stocks
            stock_returns_20d = []
            stock_returns_5d = []
            stock_volatilities = []
            
            for ticker, data in ticker_data_grouped.items():
                if data is None or len(data) < REGIME_LOOKBACK + 5:
                    continue
                    
                close = data['Close'].dropna()
                if len(close) < REGIME_LOOKBACK + 5:
                    continue
                    
                # Filter to current date
                close = close[close.index <= current_date]
                if len(close) < REGIME_LOOKBACK:
                    continue
                
                latest = close.iloc[-1]
                price_20d_ago = close.iloc[-REGIME_LOOKBACK] if len(close) >= REGIME_LOOKBACK else close.iloc[0]
                price_5d_ago = close.iloc[-5] if len(close) >= 5 else close.iloc[0]
                
                if price_20d_ago > 0:
                    ret_20d = (latest - price_20d_ago) / price_20d_ago * 100
                    stock_returns_20d.append(ret_20d)
                    
                if price_5d_ago > 0:
                    ret_5d = (latest - price_5d_ago) / price_5d_ago * 100
                    stock_returns_5d.append(ret_5d)
                
                # Daily volatility
                daily_ret = close.pct_change().dropna().tail(REGIME_LOOKBACK)
                if len(daily_ret) >= 10:
                    vol = daily_ret.std() * np.sqrt(252) * 100  # Annualized
                    stock_volatilities.append(vol)
            
            if len(stock_returns_20d) < 10:
                return None
                
            # Market-wide features
            features['market_return_20d'] = np.mean(stock_returns_20d)
            features['market_return_5d'] = np.mean(stock_returns_5d) if stock_returns_5d else 0
            features['market_volatility'] = np.mean(stock_volatilities) if stock_volatilities else 0
            features['return_dispersion'] = np.std(stock_returns_20d)  # Cross-sectional dispersion
            features['volatility_dispersion'] = np.std(stock_volatilities) if len(stock_volatilities) > 1 else 0
            
            # Trend strength: 5d vs 20d momentum
            features['trend_strength'] = features['market_return_5d'] - features['market_return_20d'] / 4
            
            # Volatility regime: 1 if high vol, 0 if low vol
            features['high_vol_regime'] = 1.0 if features['market_volatility'] > 25 else 0.0
            
            # Strategy momentum features (how each strategy performed recently)
            for strat_name in SUB_STRATEGIES:
                hist = self.strategy_histories.get(strat_name, [])
                if len(hist) >= 20:
                    recent_ret = (hist[-1] - hist[-20]) / hist[-20] * 100 if hist[-20] > 0 else 0
                    features[f'{strat_name}_momentum'] = recent_ret
                else:
                    features[f'{strat_name}_momentum'] = 0.0
                    
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
        
        # Need enough history for both features and forward labels
        min_day = max(REGIME_LOOKBACK, self.warmup_days)
        max_day = self.day_count - self.forward_days - 1
        
        if max_day <= min_day:
            print(f"   ⚠️ AI Regime: Not enough data for training (need {min_day + self.forward_days} days, have {self.day_count})")
            return False
            
        for day_idx in range(min_day, max_day, 2):  # Sample every 2 days
            # Get features for this day
            if day_idx >= len(business_days):
                continue
            current_date = business_days[day_idx]
            
            features = self.extract_regime_features(ticker_data_grouped, current_date)
            if features is None:
                continue
                
            # Get label (best strategy over next forward_days)
            best_strat = self._get_best_strategy_forward(day_idx)
            if best_strat is None:
                continue
                
            features['label'] = best_strat
            training_samples.append(features)
        
        if len(training_samples) < 20:
            print(f"   ⚠️ AI Regime: Insufficient training samples ({len(training_samples)})")
            return False
            
        # Convert to DataFrame
        train_df = pd.DataFrame(training_samples)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(train_df['label'])
        
        # Feature columns (exclude label)
        feature_cols = [c for c in train_df.columns if c != 'label']
        X = train_df[feature_cols].values
        
        # Train classifier
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, verbose=0
            )
            self.model.fit(X, y)
        
        self.label_encoder = le
        self.feature_cols = feature_cols
        self.last_train_day = self.day_count
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {le.inverse_transform([u])[0]: c for u, c in zip(unique, counts)}
        print(f"   ✅ AI Regime: Model trained on {len(training_samples)} samples")
        print(f"   📊 AI Regime: Class distribution: {class_dist}")
        
        return True
    
    def predict_best_strategy(self, ticker_data_grouped: Dict[str, pd.DataFrame],
                               current_date: datetime) -> str:
        """
        Predict which strategy will perform best.
        
        Returns:
            Strategy name to use
        """
        # Default if no model trained yet
        if self.model is None:
            return 'risk_adj_mom_3m'  # Default to best simple strategy
            
        # Extract current features
        features = self.extract_regime_features(ticker_data_grouped, current_date)
        if features is None:
            return self.current_strategy or 'risk_adj_mom_3m'
            
        # Predict
        try:
            X = np.array([[features.get(c, 0) for c in self.feature_cols]])
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
            return self.current_strategy or 'risk_adj_mom_3m'
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.model is None:
            return True
        return (self.day_count - self.last_train_day) >= self.retrain_days


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
    # Import and call the appropriate strategy
    if predicted_strategy == 'risk_adj_mom_3m':
        from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
        return select_risk_adj_mom_3m_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'risk_adj_mom_6m':
        from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks
        return select_risk_adj_mom_6m_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'risk_adj_mom':
        from risk_adj_mom_strategy import select_risk_adj_mom_stocks
        return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'elite_hybrid':
        from elite_hybrid_strategy import select_elite_hybrid_stocks
        return select_elite_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'elite_risk':
        from elite_risk_strategy import select_elite_risk_stocks
        return select_elite_risk_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'ai_elite':
        from ai_elite_strategy import select_ai_elite_stocks
        return select_ai_elite_stocks(all_tickers, ticker_data_grouped, current_date, top_n, per_ticker_models=ai_elite_models)
        
    elif predicted_strategy == 'momentum_volatility_hybrid_6m':
        from momentum_volatility_hybrid_strategy import select_momentum_volatility_hybrid_stocks
        return select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=126)
        
    elif predicted_strategy == 'trend_atr':
        from trend_following_atr_strategy import select_trend_following_atr_stocks
        return select_trend_following_atr_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'dual_momentum':
        from dual_momentum_strategy import select_dual_momentum_stocks
        return select_dual_momentum_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
        
    elif predicted_strategy == 'static_bh_1y':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=252)
        
    elif predicted_strategy == 'static_bh_3m':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date, top_n, lookback_days=63)
        
    else:
        # Fallback to risk_adj_mom_3m
        from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
        return select_risk_adj_mom_3m_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
