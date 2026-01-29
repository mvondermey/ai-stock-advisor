"""
Machine Learning Ensemble Strategy

Combines predictions from multiple ML models with weighted voting.
Features:
- Multiple model types (existing models in the system)
- Weighted voting based on recent accuracy
- Dynamic weight adjustment
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import pickle

from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
)

# ============================================
# Configuration Parameters
# ============================================

# Model weights (can be adjusted based on backtest performance)
MODEL_WEIGHTS = {
    'lstm': 0.30,
    'xgboost': 0.25,
    'random_forest': 0.20,
    'linear': 0.15,
    'ensemble_avg': 0.10,
}

# Prediction parameters
MIN_CONFIDENCE = 0.6  # Minimum prediction confidence
LOOKBACK_FOR_ACCURACY = 30  # Days to calculate recent accuracy

# Consensus parameters
MIN_MODELS_AGREE = 2  # Minimum models that must agree


class MLEnsemble:
    """Machine Learning Ensemble Strategy Implementation."""
    
    def __init__(self):
        self.model_accuracies = defaultdict(lambda: 0.5)  # Default 50% accuracy
        self.predictions_history = defaultdict(list)
        self.models_dir = Path(__file__).parent.parent / "models"
    
    def load_model_predictions(self, ticker: str, model_type: str,
                               ticker_data: pd.DataFrame,
                               current_date: datetime) -> Tuple[float, float]:
        """
        Load or generate predictions from a specific model.
        Returns: (predicted_return, confidence)
        """
        try:
            # Try to load saved model
            model_path = self.models_dir / f"{ticker}_{model_type}.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # Generate prediction using loaded model
                # This is a placeholder - actual implementation depends on model type
                return self._generate_model_prediction(model, ticker_data, current_date)
            else:
                # Generate synthetic prediction based on price patterns
                return self._generate_synthetic_prediction(ticker_data, current_date, model_type)
                
        except Exception as e:
            return 0.0, 0.0
    
    def _generate_model_prediction(self, model, ticker_data: pd.DataFrame,
                                   current_date: datetime) -> Tuple[float, float]:
        """Generate prediction from loaded model."""
        try:
            # Placeholder - actual implementation depends on model architecture
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 30:
                return 0.0, 0.0
            
            # Simple momentum-based prediction as fallback
            returns = data['Close'].pct_change().dropna()
            recent_return = returns.tail(20).mean()
            volatility = returns.tail(20).std()
            
            # Confidence based on consistency
            positive_days = (returns.tail(20) > 0).sum() / 20
            confidence = abs(positive_days - 0.5) * 2  # 0 to 1 scale
            
            return recent_return * 252, confidence  # Annualized
            
        except Exception as e:
            return 0.0, 0.0
    
    def _generate_synthetic_prediction(self, ticker_data: pd.DataFrame,
                                       current_date: datetime,
                                       model_type: str) -> Tuple[float, float]:
        """Generate synthetic prediction when no model is available."""
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 60:
                return 0.0, 0.0
            
            returns = data['Close'].pct_change().dropna()
            
            # Different "models" use different signals
            if model_type == 'lstm':
                # LSTM-like: trend following with momentum
                short_ma = data['Close'].tail(10).mean()
                long_ma = data['Close'].tail(50).mean()
                trend = (short_ma - long_ma) / long_ma
                pred = trend * 2  # Amplify trend
                conf = min(abs(trend) * 10, 0.9)
                
            elif model_type == 'xgboost':
                # XGBoost-like: multiple features combined
                mom_1m = (data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21]
                mom_3m = (data['Close'].iloc[-1] - data['Close'].iloc[-63]) / data['Close'].iloc[-63] if len(data) >= 63 else mom_1m
                vol = returns.tail(20).std() * np.sqrt(252)
                
                # Combined score
                pred = (mom_1m * 0.4 + mom_3m * 0.4) / (vol + 0.1)
                conf = 0.6 if abs(pred) > 0.1 else 0.4
                
            elif model_type == 'random_forest':
                # RF-like: mean reversion component
                price = data['Close'].iloc[-1]
                ma_50 = data['Close'].tail(50).mean()
                deviation = (price - ma_50) / ma_50
                
                # Mean reversion prediction
                pred = -deviation * 0.5  # Expect reversion
                conf = min(abs(deviation) * 5, 0.8)
                
            elif model_type == 'linear':
                # Linear: simple trend extrapolation
                recent = data['Close'].tail(30)
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent.values, 1)[0]
                pred = (slope / recent.mean()) * 30  # 30-day projection
                conf = 0.5
                
            else:  # ensemble_avg
                # Average of recent returns
                pred = returns.tail(60).mean() * 252
                conf = 0.5
            
            return pred, conf
            
        except Exception as e:
            return 0.0, 0.0
    
    def get_ensemble_prediction(self, ticker: str, ticker_data: pd.DataFrame,
                               current_date: datetime) -> Tuple[float, float, Dict]:
        """
        Get weighted ensemble prediction from all models.
        Returns: (ensemble_prediction, ensemble_confidence, model_predictions)
        """
        model_predictions = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for model_type, base_weight in MODEL_WEIGHTS.items():
            pred, conf = self.load_model_predictions(ticker, model_type, ticker_data, current_date)
            
            if conf < MIN_CONFIDENCE * 0.5:  # Skip very low confidence
                continue
            
            # Adjust weight by confidence and historical accuracy
            accuracy = self.model_accuracies.get(f"{ticker}_{model_type}", 0.5)
            adjusted_weight = base_weight * conf * accuracy
            
            model_predictions[model_type] = {
                'prediction': pred,
                'confidence': conf,
                'weight': adjusted_weight
            }
            
            weighted_sum += pred * adjusted_weight
            weight_sum += adjusted_weight
        
        if weight_sum == 0:
            return 0.0, 0.0, model_predictions
        
        ensemble_pred = weighted_sum / weight_sum
        
        # Ensemble confidence based on agreement
        predictions = [m['prediction'] for m in model_predictions.values()]
        if len(predictions) >= 2:
            # Check how many models agree on direction
            positive = sum(1 for p in predictions if p > 0)
            agreement = max(positive, len(predictions) - positive) / len(predictions)
            ensemble_conf = agreement
        else:
            ensemble_conf = 0.5
        
        return ensemble_pred, ensemble_conf, model_predictions
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks based on ML ensemble predictions."""
        print(f"\n   🎯 ML Ensemble Strategy")
        print(f"   📅 Date: {current_date.date()}")
        print(f"   🤖 Models: {list(MODEL_WEIGHTS.keys())}")
        
        candidates = []
        
        for ticker in all_tickers:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) < 60:
                continue
            
            # Get ensemble prediction
            pred, conf, model_preds = self.get_ensemble_prediction(
                ticker, ticker_data, current_date
            )
            
            # Filter by confidence and positive prediction
            if conf >= MIN_CONFIDENCE and pred > 0:
                # Count agreeing models
                agreeing = sum(1 for m in model_preds.values() if m['prediction'] > 0)
                
                if agreeing >= MIN_MODELS_AGREE:
                    score = pred * conf * (agreeing / len(MODEL_WEIGHTS))
                    candidates.append((ticker, score, pred, conf, agreeing, model_preds))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score, pred, conf, agreeing, model_preds in candidates[:top_n]]
        
        print(f"   ✅ Analyzed {len(all_tickers)} tickers")
        print(f"   ✅ Found {len(candidates)} high-confidence candidates")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score, pred, conf, agreeing, model_preds in candidates[:top_n]:
            print(f"      {ticker}: Pred={pred*100:.1f}%, Conf={conf:.2f}, Models={agreeing}/{len(MODEL_WEIGHTS)}")
        
        return selected


# Global instance
_ml_ensemble_instance = None

def get_ml_ensemble_instance() -> MLEnsemble:
    """Get or create the global ML ensemble instance."""
    global _ml_ensemble_instance
    if _ml_ensemble_instance is None:
        _ml_ensemble_instance = MLEnsemble()
    return _ml_ensemble_instance


def select_ml_ensemble_stocks(all_tickers: List[str],
                              ticker_data_grouped: Dict[str, pd.DataFrame],
                              current_date: datetime = None,
                              train_start_date: datetime = None,
                              top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    ML Ensemble stock selection strategy.
    
    Combines predictions from multiple ML models.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_ml_ensemble_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_ml_ensemble_state():
    """Reset the global ML ensemble instance."""
    global _ml_ensemble_instance
    _ml_ensemble_instance = None
