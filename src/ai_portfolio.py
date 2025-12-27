"""
AI Portfolio Strategy - Meta-Learning for Portfolio Selection

Trains an AI model to learn which combinations of 3 stocks perform best together.
Unlike the main AI strategy (which predicts individual returns), this learns:
- Optimal stock combinations
- Diversification benefits
- Correlation effects
- Portfolio-level dynamics

The model selects 3 stocks from candidates that work well together as a portfolio.

DATA SEPARATION (Critical for preventing look-ahead bias):
┌─────────────────────────────────────────────────────────────────┐
│  Historical Data    │    Training Period    │  Backtest Period  │
│  (not used)         │  (AI Portfolio learns)│  (AI Portfolio    │
│                     │                       │   makes decisions)│
├─────────────────────┼───────────────────────┼───────────────────┤
│  ...older data...   │  train_start_date     │  bt_start_date    │
│                     │         ↓             │       ↓           │
│                     │  train_end_date       │  bt_end_date      │
│                     │  (bt_start - 1 day)   │                   │
└─────────────────────┴───────────────────────┴───────────────────┘
                                              ↑
                                    NO OVERLAP - Training ends
                                    BEFORE backtest starts
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def train_ai_portfolio_model(
    all_tickers_data: pd.DataFrame,
    train_start_date: datetime,
    train_end_date: datetime,
    top_tickers: List[str]
) -> Optional[Dict]:
    """
    Train AI Portfolio Selection Model.
    
    Learns which combinations of 3 stocks perform best together as a portfolio.
    
    Training approach:
    1. Generate all possible 3-stock combinations from candidates
    2. Calculate portfolio performance for each combination (historical)
    3. Extract features for each combination:
       - Individual stock metrics (volatility, momentum, etc.)
       - Portfolio metrics (correlation, diversification, etc.)
    4. Train classifier to predict: good portfolio vs bad portfolio
    
    Args:
        all_tickers_data: Long-format DataFrame with all ticker data
        train_start_date: Start of training period
        train_end_date: End of training period
        top_tickers: Candidate tickers for portfolio
    
    Returns:
        Dict with trained model and scaler, or None if training fails
    """
    
    if not SKLEARN_AVAILABLE:
        print("   ⚠️ AI Portfolio: scikit-learn not available, using fallback strategy")
        return None
    
    if len(top_tickers) < 3:
        print(f"   ⚠️ AI Portfolio: Need at least 3 tickers, got {len(top_tickers)}")
        return None
    
    try:
        print(f"   🧠 AI Portfolio: Training model to select best 3-stock combinations...")
        print(f"   📊 Training on {len(top_tickers)} candidates from {train_start_date.date()} to {train_end_date.date()}")
        
        # ✅ CRITICAL: Validate data separation (prevent look-ahead bias)
        # Training must use ONLY historical data BEFORE backtest
        if train_start_date >= train_end_date:
            print(f"   ❌ AI Portfolio: Invalid training dates (start >= end)")
            return None
        
        # ✅ CRITICAL: Filter data to ONLY training period (prevent look-ahead bias)
        # Training data must be completely separate from backtest data
        train_data = all_tickers_data[
            (all_tickers_data['date'] >= train_start_date) & 
            (all_tickers_data['date'] <= train_end_date)
        ].copy()
        
        if train_data.empty:
            print(f"   ⚠️ AI Portfolio: No data available in training period")
            return None
        
        # Verify data boundaries
        actual_train_start = train_data['date'].min()
        actual_train_end = train_data['date'].max()
        print(f"   🔒 Data isolation: Using ONLY {len(train_data)} rows from training period")
        print(f"      Training data range: {actual_train_start.date()} to {actual_train_end.date()}")
        print(f"   ✅ Backtest data will NEVER be used during training (preventing look-ahead bias)")
        
        # Generate training data: historical 3-stock combinations and their performance
        training_samples = []
        training_labels = []
        
        # Get training parameters from config
        try:
            from config import (
                AI_PORTFOLIO_EVALUATION_WINDOW,
                AI_PORTFOLIO_STEP_SIZE,
                AI_PORTFOLIO_PERFORMANCE_THRESHOLD_ANNUAL
            )
            evaluation_window = AI_PORTFOLIO_EVALUATION_WINDOW
            step_size = AI_PORTFOLIO_STEP_SIZE
            annual_threshold = AI_PORTFOLIO_PERFORMANCE_THRESHOLD_ANNUAL
        except ImportError:
            evaluation_window = 30  # Default: 30 days
            step_size = 15  # Default: 15 days
            annual_threshold = 0.50  # Default: 50% annual
        
        # ✅ Convert annualized threshold to evaluation window period
        # Formula: period_return = (1 + annual_return)^(days/365) - 1
        performance_threshold = (1 + annual_threshold) ** (evaluation_window / 365.0) - 1
        
        print(f"   📐 Training parameters:")
        print(f"      Evaluation window: {evaluation_window} days")
        print(f"      Step size: {step_size} days")
        print(f"      Target annual return: {annual_threshold:.1%}")
        print(f"      → Threshold for {evaluation_window} days: {performance_threshold:.2%} (AFTER transaction costs)")
        
        # Sliding window approach: test combinations at different time points
        current_eval_start = train_start_date
        while current_eval_start + timedelta(days=evaluation_window) <= train_end_date:
            eval_end = current_eval_start + timedelta(days=evaluation_window)
            
            # Try different 3-stock combinations
            for combo in combinations(top_tickers, 3):
                try:
                    # Calculate features and performance for this combination
                    # Use filtered train_data to ensure no backtest data leakage
                    features = _extract_portfolio_features(
                        train_data, list(combo), current_eval_start
                    )
                    
                    # ✅ Calculate performance INCLUDING transaction costs
                    performance = _calculate_portfolio_performance(
                        train_data, list(combo), current_eval_start, eval_end
                    )
                    
                    if features is not None and not np.isnan(performance):
                        training_samples.append(features)
                        # Label: 1 if portfolio beats threshold, 0 otherwise
                        training_labels.append(1 if performance > performance_threshold else 0)
                
                except Exception:
                    continue
            
            current_eval_start += timedelta(days=step_size)
        
        if len(training_samples) < 10:
            print(f"   ⚠️ AI Portfolio: Insufficient training data ({len(training_samples)} samples)")
            return None
        
        # Train model
        X = np.array(training_samples)
        y = np.array(training_labels)
        
        print(f"   📈 AI Portfolio: Generated {len(X)} training samples (50 features per sample)")
        print(f"      Positive samples (good portfolios): {np.sum(y == 1)}")
        print(f"      Negative samples (poor portfolios): {np.sum(y == 0)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # === MULTI-MODEL TRAINING & SELECTION ===
        print(f"   🧪 AI Portfolio: Training multiple models and selecting the best...")
        
        # Check GPU availability
        try:
            from config import CUDA_AVAILABLE
            gpu_available = CUDA_AVAILABLE
        except ImportError:
            gpu_available = False
        
        if gpu_available:
            print(f"   🚀 GPU detected - XGBoost and LightGBM will use GPU acceleration")
        else:
            print(f"   💻 Using CPU with parallel processing (n_jobs=-1)")
        
        models_to_try = {}
        
        # 1. Random Forest (CPU parallel)
        models_to_try['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # ✅ Uses all CPU cores
        )
        
        # 2. Extra Trees (CPU parallel, more randomization)
        models_to_try['Extra Trees'] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # ✅ Uses all CPU cores
        )
        
        # 3. Gradient Boosting (CPU only - no parallel in sklearn)
        models_to_try['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # 4. XGBoost (GPU or CPU parallel)
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            if gpu_available:
                # ✅ GPU acceleration
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['predictor'] = 'gpu_predictor'
                xgb_params['gpu_id'] = 0
            else:
                # CPU parallel
                xgb_params['tree_method'] = 'hist'
                xgb_params['n_jobs'] = -1
            
            models_to_try['XGBoost'] = xgb.XGBClassifier(**xgb_params)
        
        # 5. LightGBM (GPU or CPU parallel)
        if LIGHTGBM_AVAILABLE:
            lgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1
            }
            
            if gpu_available:
                # ✅ GPU acceleration
                lgb_params['device'] = 'gpu'
                lgb_params['gpu_platform_id'] = 0
                lgb_params['gpu_device_id'] = 0
            else:
                # CPU parallel
                lgb_params['device'] = 'cpu'
                lgb_params['n_jobs'] = -1
            
            models_to_try['LightGBM'] = lgb.LGBMClassifier(**lgb_params)
        
        # 6. Logistic Regression (CPU parallel, baseline)
        models_to_try['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1  # ✅ Uses all CPU cores
        )
        
        # Cross-validate each model and select the best
        best_model = None
        best_model_name = None
        best_score = -np.inf
        
        print(f"      Evaluating {len(models_to_try)} models with 3-fold cross-validation...")
        
        for model_name, model in models_to_try.items():
            try:
                # Use cross-validation to evaluate model
                cv_scores = cross_val_score(model, X_scaled, y, cv=min(3, len(X)), scoring='accuracy', n_jobs=-1)
                mean_score = cv_scores.mean()
                
                print(f"         {model_name:<20}: {mean_score:.4f} ± {cv_scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = model_name
            
            except Exception as e:
                print(f"         {model_name:<20}: FAILED ({str(e)[:40]})")
                continue
        
        if best_model is None:
            print(f"   ❌ AI Portfolio: All models failed to train")
            return None
        
        # Train the best model on full dataset
        print(f"   🏆 Best model: {best_model_name} (CV score: {best_score:.4f})")
        print(f"      Training {best_model_name} on full dataset...")
        
        best_model.fit(X_scaled, y)
        train_accuracy = best_model.score(X_scaled, y)
        
        print(f"   ✅ AI Portfolio: Model trained successfully")
        print(f"      Final training accuracy: {train_accuracy:.2%}")
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'cv_score': best_score,
            'scaler': scaler,
            'train_date': train_end_date,
            'n_features': X.shape[1]
        }
    
    except Exception as e:
        print(f"   ❌ AI Portfolio training error: {e}")
        return None


def _extract_portfolio_features(
    all_tickers_data: pd.DataFrame,
    tickers: List[str],
    current_date: datetime,
    lookback_days: int = 20
) -> Optional[np.ndarray]:
    """
    Extract ENHANCED features for a portfolio of stocks.
    
    Features include:
    - Individual stock metrics (momentum, volatility, RSI, Sharpe, etc.)
    - Multiple timeframe momentum (short/medium/long)
    - Volume analysis
    - Portfolio-level metrics (correlation, diversification, risk-adjusted returns)
    - Downside risk metrics
    """
    
    try:
        lookback_start = current_date - timedelta(days=lookback_days + 30)
        
        stock_features = []
        returns_data = []
        volume_data = []
        
        for ticker in tickers:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return None
            
            ticker_data = ticker_data.set_index('date')
            recent_data = ticker_data.loc[lookback_start:current_date]
            
            if len(recent_data) < lookback_days:
                return None
            
            # Calculate individual stock features
            close_prices = recent_data['Close'].dropna()
            if len(close_prices) < lookback_days:
                return None
            
            returns = close_prices.pct_change().dropna()
            
            # === MOMENTUM FEATURES (Multiple Timeframes) ===
            # Short-term momentum (5-day)
            momentum_5d = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0
            # Medium-term momentum (10-day)
            momentum_10d = (close_prices.iloc[-1] / close_prices.iloc[-10] - 1) if len(close_prices) >= 10 else 0
            # Long-term momentum (20-day)
            momentum_20d = (close_prices.iloc[-1] / close_prices.iloc[0] - 1)
            
            # === VOLATILITY FEATURES ===
            volatility = returns.std()
            # Downside volatility (only negative returns)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # === RISK-ADJUSTED RETURNS ===
            # Sharpe ratio approximation (assuming risk-free rate = 0)
            sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
            # Sortino ratio (downside risk-adjusted)
            sortino = (returns.mean() / downside_volatility) if downside_volatility > 0 else 0
            
            # === RSI (Relative Strength Index) ===
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            rs = gains / losses if losses > 0 else 2.0
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = rsi / 100.0
            
            # === VOLUME ANALYSIS ===
            if 'Volume' in recent_data.columns:
                volumes = recent_data['Volume'].dropna()
                if len(volumes) >= 5:
                    # Volume trend (recent vs average)
                    avg_volume = volumes.mean()
                    recent_volume = volumes.iloc[-5:].mean()
                    volume_trend = (recent_volume / avg_volume - 1) if avg_volume > 0 else 0
                else:
                    volume_trend = 0
            else:
                volume_trend = 0
            
            # === PRICE PATTERN FEATURES ===
            # Maximum drawdown in period
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Win rate (percentage of positive days)
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5
            
            # === TREND INDICATORS ===
            # Simple moving averages
            if len(close_prices) >= 10:
                sma_5 = close_prices.iloc[-5:].mean()
                sma_10 = close_prices.iloc[-10:].mean()
                price_vs_sma5 = (close_prices.iloc[-1] / sma_5 - 1)
                price_vs_sma10 = (close_prices.iloc[-1] / sma_10 - 1)
            else:
                price_vs_sma5 = 0
                price_vs_sma10 = 0
            
            # Collect features for this stock
            stock_features.extend([
                momentum_5d,           # 1. Short-term momentum
                momentum_10d,          # 2. Medium-term momentum
                momentum_20d,          # 3. Long-term momentum
                volatility,            # 4. Total volatility
                downside_volatility,   # 5. Downside risk
                sharpe,                # 6. Sharpe ratio
                sortino,               # 7. Sortino ratio
                rsi_normalized,        # 8. RSI indicator
                volume_trend,          # 9. Volume trend
                max_drawdown,          # 10. Maximum drawdown
                win_rate,              # 11. Win rate
                price_vs_sma5,         # 12. Price vs 5-day MA
                price_vs_sma10         # 13. Price vs 10-day MA
            ])
            
            returns_data.append(returns.values)
            if 'Volume' in recent_data.columns:
                volume_data.append(recent_data['Volume'].dropna().values)
        
        # === PORTFOLIO-LEVEL FEATURES ===
        # Correlation matrix and diversification
        if len(returns_data) == 3:
            try:
                min_length = min(len(r) for r in returns_data)
                returns_matrix = np.array([r[:min_length] for r in returns_data])
                corr_matrix = np.corrcoef(returns_matrix)
                
                # Average correlation
                avg_correlation = (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
                # Max correlation (worst case for diversification)
                max_correlation = max(abs(corr_matrix[0,1]), abs(corr_matrix[0,2]), abs(corr_matrix[1,2]))
                # Min correlation (best case for diversification)
                min_correlation = min(abs(corr_matrix[0,1]), abs(corr_matrix[0,2]), abs(corr_matrix[1,2]))
            except:
                avg_correlation = 0.5
                max_correlation = 0.5
                min_correlation = 0.5
        else:
            avg_correlation = 0.5
            max_correlation = 0.5
            min_correlation = 0.5
        
        # Diversification score (inverse of correlation)
        diversification_score = 1 - abs(avg_correlation)
        
        # Portfolio concentration risk (how similar are the stocks?)
        concentration_risk = max_correlation
        
        # === AGGREGATE PORTFOLIO METRICS ===
        # Average momentum across stocks
        avg_momentum = np.mean([stock_features[i*13 + 2] for i in range(3)])  # 20-day momentum
        # Average volatility
        avg_volatility = np.mean([stock_features[i*13 + 3] for i in range(3)])
        # Average Sharpe
        avg_sharpe = np.mean([stock_features[i*13 + 5] for i in range(3)])
        # Best stock momentum (max)
        max_momentum = max([stock_features[i*13 + 2] for i in range(3)])
        # Worst stock momentum (min)
        min_momentum = min([stock_features[i*13 + 2] for i in range(3)])
        # Momentum spread (diversity in performance)
        momentum_spread = max_momentum - min_momentum
        
        # Combine all features
        portfolio_features = stock_features + [
            avg_correlation,         # 40. Average correlation
            max_correlation,         # 41. Maximum correlation
            min_correlation,         # 42. Minimum correlation
            diversification_score,   # 43. Diversification score
            concentration_risk,      # 44. Concentration risk
            avg_momentum,            # 45. Average portfolio momentum
            avg_volatility,          # 46. Average portfolio volatility
            avg_sharpe,              # 47. Average Sharpe ratio
            max_momentum,            # 48. Best stock momentum
            min_momentum,            # 49. Worst stock momentum
            momentum_spread          # 50. Momentum spread (diversity)
        ]
        
        return np.array(portfolio_features)
    
    except Exception as e:
        return None


def _calculate_portfolio_performance(
    all_tickers_data: pd.DataFrame,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime
) -> float:
    """
    Calculate equal-weight portfolio performance over a period.
    
    ✅ INCLUDES TRANSACTION COSTS for realistic training.
    Assumes we buy at start and sell at end (2 transactions per stock).
    
    Returns:
        Portfolio return AFTER transaction costs (e.g., 0.50 for 50% return)
    """
    
    try:
        # Get transaction cost from config
        try:
            from config import TRANSACTION_COST
        except ImportError:
            TRANSACTION_COST = 0.01  # Default 1%
        
        portfolio_returns = []
        
        for ticker in tickers:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if ticker_data.empty:
                return np.nan
            
            ticker_data = ticker_data.set_index('date')
            period_data = ticker_data.loc[start_date:end_date]
            
            if len(period_data) < 2:
                return np.nan
            
            close_prices = period_data['Close'].dropna()
            if len(close_prices) < 2:
                return np.nan
            
            # Raw return
            stock_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1)
            
            # ✅ Subtract transaction costs (buy + sell = 2 transactions)
            # Buy cost: TRANSACTION_COST on entry
            # Sell cost: TRANSACTION_COST on exit
            stock_return_after_costs = stock_return - (2 * TRANSACTION_COST)
            
            portfolio_returns.append(stock_return_after_costs)
        
        # Equal-weight portfolio return (after costs)
        if len(portfolio_returns) == 3:
            return np.mean(portfolio_returns)
        else:
            return np.nan
    
    except Exception:
        return np.nan


# Global variable to store trained model
_ai_portfolio_model = None


def get_ai_portfolio_rebalancing_stocks(
    all_tickers_data: pd.DataFrame,
    top_tickers: List[str],
    current_date: datetime,
    current_portfolio: List[str],
    max_stocks: int = 3
) -> List[str]:
    """
    Select stocks for AI Portfolio using trained model.
    
    Uses the trained model to evaluate all possible 3-stock combinations
    and selects the one predicted to perform best.
    
    Args:
        all_tickers_data: Long-format DataFrame with all ticker data
        top_tickers: List of candidate tickers (from 3-month momentum)
        current_date: Current date in backtest
        current_portfolio: Currently held stocks
        max_stocks: Maximum number of stocks to hold (default: 3)
    
    Returns:
        List of selected ticker symbols (exactly 3)
    """
    
    global _ai_portfolio_model
    
    if not top_tickers or len(top_tickers) < 3:
        print(f"   ⚠️ AI Portfolio: Need at least 3 candidates, got {len(top_tickers)}")
        return current_portfolio[:3] if current_portfolio else top_tickers[:3]
    
    # If no model is trained, use fallback strategy
    if _ai_portfolio_model is None or not SKLEARN_AVAILABLE:
        return _fallback_selection(all_tickers_data, top_tickers, current_date, current_portfolio, max_stocks)
    
    try:
        model = _ai_portfolio_model['model']
        scaler = _ai_portfolio_model['scaler']
        
        # ✅ Smart rebalancing: Evaluate current portfolio score first
        current_score = -np.inf
        if current_portfolio and len(current_portfolio) == 3:
            try:
                current_features = _extract_portfolio_features(
                    all_tickers_data, current_portfolio, current_date
                )
                if current_features is not None:
                    current_features_scaled = scaler.transform(current_features.reshape(1, -1))
                    current_score = model.predict_proba(current_features_scaled)[0][1]
            except Exception:
                pass
        
        # Evaluate all possible 3-stock combinations
        best_score = current_score
        best_combo = None
        
        # Limit candidates to reduce computation
        for combo in combinations(top_tickers[:min(len(top_tickers), 6)], 3):  # Limit to top 6 candidates
            try:
                # Extract features for this combination
                features = _extract_portfolio_features(
                    all_tickers_data, list(combo), current_date
                )
                
                if features is not None:
                    # Scale and predict
                    features_scaled = scaler.transform(features.reshape(1, -1))
                    # Get probability of being a "good" portfolio
                    score = model.predict_proba(features_scaled)[0][1]
                    
                    if score > best_score:
                        best_score = score
                        best_combo = list(combo)
            
            except Exception:
                continue
        
        # ✅ COST-BENEFIT ANALYSIS: Check if expected improvement exceeds transaction costs
        if best_combo is not None:
            if current_score > -np.inf and current_portfolio:
                # Calculate how many stocks need to change
                current_set = set(current_portfolio[:3])
                new_set = set(best_combo)
                stocks_to_change = len(current_set.symmetric_difference(new_set))
                
                if stocks_to_change == 0:
                    # No changes needed
                    print(f"   ✔️ AI Portfolio: No changes needed (same portfolio)")
                    return current_portfolio[:3]
                
                # Get transaction cost from config
                try:
                    from config import TRANSACTION_COST, AI_PORTFOLIO_MIN_IMPROVEMENT_THRESHOLD_ANNUAL
                    annual_improvement_threshold = AI_PORTFOLIO_MIN_IMPROVEMENT_THRESHOLD_ANNUAL
                except ImportError:
                    TRANSACTION_COST = 0.01
                    annual_improvement_threshold = 0.05
                
                # Calculate transaction costs
                # Each stock change = 1 sell + 1 buy = 2 transactions
                total_transaction_cost_pct = stocks_to_change * 2 * TRANSACTION_COST
                
                # Calculate expected improvement (probability to return %)
                # The model outputs probability of "good portfolio" (>50% annual return)
                # Score improvement roughly correlates to expected return difference
                # Conservative estimate: probability_improvement × target_return
                score_improvement = best_score - current_score
                
                # Get evaluation window from config
                try:
                    from config import AI_PORTFOLIO_EVALUATION_WINDOW
                    eval_window = AI_PORTFOLIO_EVALUATION_WINDOW
                except ImportError:
                    eval_window = 30
                
                # Convert annual threshold to evaluation window
                period_threshold = (1 + annual_improvement_threshold) ** (eval_window / 365.0) - 1
                
                # Expected improvement = score_improvement × period_threshold
                # This estimates: if score improves by X, expected return improves by ~X × threshold
                expected_improvement_pct = score_improvement * period_threshold
                
                # ✅ TWO-STAGE THRESHOLD CHECK
                
                # THRESHOLD 1: Minimum improvement (quality gate)
                # The new portfolio must be meaningfully better, not just marginally better
                min_score_improvement = annual_improvement_threshold * 0.1  # Convert to probability scale
                
                if score_improvement < min_score_improvement:
                    print(f"   💤 AI Portfolio: Keeping current portfolio")
                    print(f"      Score improvement: +{score_improvement:.4f} < threshold {min_score_improvement:.4f}")
                    print(f"      Reason: New portfolio not significantly better")
                    return current_portfolio[:3]
                
                # THRESHOLD 2: Cost-benefit analysis
                # The expected improvement must exceed transaction costs
                net_benefit = expected_improvement_pct - total_transaction_cost_pct
                
                if net_benefit > 0:
                    # Both thresholds passed - rebalance!
                    print(f"   ✅ AI Portfolio: Rebalancing justified (passed both thresholds)")
                    print(f"      Score: {current_score:.4f} → {best_score:.4f} (+{score_improvement:.4f})")
                    print(f"      Expected improvement: {expected_improvement_pct:.2%}")
                    print(f"      Transaction costs: {total_transaction_cost_pct:.2%} ({stocks_to_change} stocks × 2 trades)")
                    print(f"      Net benefit: {net_benefit:.2%} ✓")
                    return best_combo
                else:
                    # Quality gate passed, but costs too high
                    print(f"   💤 AI Portfolio: Keeping current portfolio")
                    print(f"      Score improvement: +{score_improvement:.4f} ✓ (passes quality threshold)")
                    print(f"      Expected improvement: {expected_improvement_pct:.2%}")
                    print(f"      Transaction costs: {total_transaction_cost_pct:.2%} ({stocks_to_change} stocks)")
                    print(f"      Net benefit: {net_benefit:.2%} ✗ (NEGATIVE - costs exceed benefit)")
                    return current_portfolio[:3]
            else:
                # No current portfolio, use best combo
                print(f"   🆕 AI Portfolio: Initializing with best portfolio (score: {best_score:.4f})")
                return best_combo
        else:
            # Fallback if no combination could be evaluated
            return _fallback_selection(all_tickers_data, top_tickers, current_date, current_portfolio, 3)
    
    except Exception as e:
        print(f"   ⚠️ AI Portfolio selection error: {e}")
        return _fallback_selection(all_tickers_data, top_tickers, current_date, current_portfolio, 3)


def _fallback_selection(
    all_tickers_data: pd.DataFrame,
    top_tickers: List[str],
    current_date: datetime,
    current_portfolio: List[str],
    max_stocks: int
) -> List[str]:
    """
    Fallback selection strategy when AI model is not available.
    
    Uses simple momentum-based selection.
    """
    try:
        stock_scores = []
        
        for ticker in top_tickers:
            try:
                ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                if ticker_data.empty:
                    continue
                
                ticker_data = ticker_data.set_index('date')
                lookback_date = current_date - timedelta(days=10)
                recent_data = ticker_data.loc[lookback_date:current_date]
                
                if len(recent_data) >= 2:
                    close_prices = recent_data['Close'].dropna()
                    if len(close_prices) >= 2:
                        momentum = (close_prices.iloc[-1] / close_prices.iloc[0] - 1)
                        stock_scores.append((ticker, momentum))
            except Exception:
                continue
        
        if stock_scores:
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in stock_scores[:max_stocks]]
        
        # Final fallback
        if current_portfolio:
            return current_portfolio[:max_stocks]
        return top_tickers[:max_stocks]
    
    except Exception:
        return top_tickers[:max_stocks] if top_tickers else []


def set_ai_portfolio_model(model_dict: Optional[Dict]):
    """
    Set the trained AI Portfolio model globally.
    
    Called after training to make the model available for selection.
    """
    global _ai_portfolio_model
    _ai_portfolio_model = model_dict
    
    if model_dict is not None:
        model_name = model_dict.get('model_name', 'Unknown')
        cv_score = model_dict.get('cv_score', 0)
        n_features = model_dict.get('n_features', 0)
        print(f"   📊 AI Portfolio model active: {model_name} (CV: {cv_score:.4f}, Features: {n_features})")


def get_ai_portfolio_stats(
    portfolio_history: List[float],
    initial_capital: float
) -> Dict[str, float]:
    """
    Calculate performance statistics for AI Portfolio.
    
    Args:
        portfolio_history: List of portfolio values over time
        initial_capital: Starting capital
    
    Returns:
        Dictionary with performance metrics
    """
    if not portfolio_history or len(portfolio_history) < 2:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0
        }
    
    try:
        final_value = portfolio_history[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        portfolio_series = pd.Series(portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        cummax = portfolio_series.cummax()
        drawdown = ((portfolio_series - cummax) / cummax)
        max_drawdown = drawdown.min()
        
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }
    
    except Exception:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0
        }
