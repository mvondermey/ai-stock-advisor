"""
AI Elite Strategy: ML-powered scoring of elite stock candidates

Uses machine learning to learn optimal scoring from:
- 6M momentum
- Volatility (risk)
- Volume (liquidity)
- Dip score (1Y/3M ratio)
- 1Y performance
- 3M performance

The ML model learns which combinations of these features predict future outperformance,
discovering non-linear relationships that fixed formulas miss.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
import pickle
import os
from pathlib import Path

def select_ai_elite_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    model_path: str = None,
    model = None
) -> List[str]:
    """
    AI Elite Strategy: ML-based scoring of momentum + dip opportunities
    
    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        model_path: Path to saved ML model (optional)
        model: Pre-trained model object (optional, takes precedence over model_path)
        
    Returns:
        List of selected ticker symbols
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite"
    )
    
    candidates = []
    
    print(f"   🤖 AI Elite: Analyzing {len(filtered_tickers)} tickers with ML scoring (filtered from {len(all_tickers)})")
    
    # Extract features for all candidates
    debug_count = 0
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'exception': 0}
    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                fail_reasons['not_in_data'] += 1
                continue
            
            # Load raw 1-hour cached data (not the daily-converted data from backtesting)
            from datetime import timedelta
            ticker_data = _load_hourly_data_direct(
                ticker, 
                current_date - timedelta(days=180), 
                current_date
            )
            
            # Skip ticker if no hourly data available
            if ticker_data is None or len(ticker_data) == 0:
                fail_reasons['empty'] += 1
                continue
            
            # Debug first 3 tickers
            if debug_count < 3:
                print(f"   🔍 AI Elite DEBUG {ticker}: index_type={type(ticker_data.index).__name__}, "
                      f"len={len(ticker_data)}, cols={list(ticker_data.columns[:5])}, "
                      f"index[0]={ticker_data.index[0]}, index[-1]={ticker_data.index[-1]}")
                debug_count += 1
            
            # Calculate all features (using 1-hour data for real intraday intelligence)
            features = _extract_features(ticker, ticker_data, current_date)
            if features is None:
                fail_reasons['features_none'] += 1
                continue
            
            # Add ticker to features for DataFrame
            features['ticker'] = ticker
            candidates.append(features)
            
        except Exception as e:
            fail_reasons['exception'] += 1
            if debug_count < 5:
                print(f"   ⚠️ AI Elite DEBUG {ticker}: Exception: {e}")
            continue
    
    if not candidates:
        print(f"   ⚠️ AI Elite: No candidates found")
        print(f"   🔍 AI Elite: Fail reasons: {fail_reasons}")
        return []
    
    # Use provided model or load from path
    if model is None:
        model = _load_or_create_model(model_path)
    
    # Score candidates using ML model (trained on 6 features)
    if model is not None:
        candidates_df = pd.DataFrame(candidates)
        
        # Use simplified 6-feature model
        feature_cols = ['perf_forward', 'volatility', 'avg_volume', 
                        'overnight_gap', 'intraday_range', 'last_hour_momentum']
        
        try:
            X = candidates_df[feature_cols].values
            scores = model.predict_proba(X)[:, 1]  # Probability of positive class
            candidates_df['ai_score'] = scores
            
        except Exception as e:
            print(f"   ⚠️ AI Elite error: {e}")
            candidates_df['ai_score'] = _fallback_scoring(candidates_df)
    else:
        # No model available, use fallback
        print(f"   ⚠️ AI Elite: No model available, using fallback scoring")
        candidates_df = pd.DataFrame(candidates)
        candidates_df['ai_score'] = _fallback_scoring(candidates_df)
    
    # Sort by AI score
    candidates_df = candidates_df.sort_values('ai_score', ascending=False)
    
    # Debug: show top candidates
    print(f"   ✅ AI Elite: Found {len(candidates_df)} candidates")
    for i, row in candidates_df.head(5).iterrows():
        print(f"      {i+1}. {row['ticker']}: AI Score={row['ai_score']:.3f}, "
              f"Perf={row['perf_forward']:+.1f}%, Vol={row['volatility']:.1f}%, "
              f"Gap={row['overnight_gap']:+.2f}%, Range={row['intraday_range']:.1f}%")
    
    # Return top N tickers
    selected = candidates_df.head(top_n)['ticker'].tolist()
    return selected


def _extract_features(ticker: str, ticker_data: pd.DataFrame, current_date: datetime) -> Optional[Dict]:
    """
    Extract ML features from ticker data using 1-hour intraday data.
    
    Features:
    - perf_forward: Performance over AI_ELITE_FORWARD_DAYS
    - volatility: Annualized volatility
    - avg_volume: Average trading volume
    - overnight_gap: Overnight gap behavior (intraday)
    - intraday_range: Intraday volatility pattern (intraday)
    - last_hour_momentum: Last hour rally/fade (intraday)
    """
    try:
        # ✅ FIX: Deduplicate index (hourly data combined creates duplicates)
        if ticker_data.index.duplicated().any():
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
        
        from config import AI_ELITE_INTRADAY_LOOKBACK
        
        # ✅ FIX: Filter data up to current_date to avoid temporal leakage
        if current_date is not None:
            current_ts = pd.Timestamp(current_date)
            # Ensure both are UTC-aware for proper comparison
            if current_ts.tz is None:
                current_ts = current_ts.tz_localize('UTC')
            elif str(current_ts.tz) != 'UTC':
                current_ts = current_ts.tz_convert('UTC')
            
            # Ensure index is also UTC-aware
            if ticker_data.index.tz is None:
                ticker_data = ticker_data.copy()
                ticker_data.index = ticker_data.index.tz_localize('UTC')
            elif str(ticker_data.index.tz) != 'UTC':
                ticker_data = ticker_data.copy()
                ticker_data.index = ticker_data.index.tz_convert('UTC')
            
            # Use boolean indexing (safe for non-unique indices)
            ticker_data_filtered = ticker_data[ticker_data.index <= current_ts]
        else:
            ticker_data_filtered = ticker_data
        
        # Use dropna'd Close series for all calculations (adaptive approach)
        close_col = ticker_data_filtered['Close']
        # Handle case where Close is a DataFrame (multiple columns) instead of Series
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        close_prices = close_col.dropna()
        n_prices = len(close_prices)
        
        # Debug: Check data length for first few tickers
        if ticker in ['SNDK', 'ZEC-USD', 'WDC'] and hasattr(_extract_features, '_debug_count2'):
            if _extract_features._debug_count2 < 3:
                print(f"   🔍 FEAT DEBUG2 {ticker}: n_prices={n_prices}, filtered_len={len(ticker_data_filtered)}, "
                      f"duplicated={ticker_data.index.duplicated().any()}")
                _extract_features._debug_count2 += 1
        if not hasattr(_extract_features, '_debug_count2'):
            _extract_features._debug_count2 = 0
            print(f"   🔍 FEAT DEBUG2 {ticker}: n_prices={n_prices}, filtered_len={len(ticker_data_filtered)}, "
                  f"duplicated={ticker_data.index.duplicated().any()}")
            _extract_features._debug_count2 = 1
        
        # Minimum data requirements for 1-hour data
        min_data_points = AI_ELITE_INTRADAY_LOOKBACK * 24  # 10 days * 24 hours = 240 points
        if n_prices < min_data_points:
            return None
        
        # Get latest price
        latest_price = close_prices.iloc[-1]
        if latest_price <= 0:
            return None
        
        # Calculate performance over AI_ELITE_FORWARD_DAYS (main feature for AI Elite)
        from config import AI_ELITE_FORWARD_DAYS
        
        # For hourly data: AI_ELITE_FORWARD_DAYS * 24 hours
        lookback_forward = min(AI_ELITE_FORWARD_DAYS * 24, n_prices - 1)
        min_lookback = AI_ELITE_INTRADAY_LOOKBACK * 24  # Minimum from config
        
        if lookback_forward < min_lookback:
            return None
        price_forward_ago = close_prices.iloc[-lookback_forward]
        if price_forward_ago <= 0:
            return None
        
        # Feature 1: Performance over AI_ELITE_FORWARD_DAYS (main momentum feature)
        perf_forward = ((latest_price - price_forward_ago) / price_forward_ago) * 100
        
        # Feature 2: Volatility (annualized from 1-hour data)
        returns = close_prices.pct_change().dropna()
        
        # 24 hours/day * 252 trading days/year = 6048 hours/year
        min_returns = AI_ELITE_INTRADAY_LOOKBACK * 24  # 10 days * 24 hours = 240
        if len(returns) < min_returns:
            return None
        volatility = returns.std() * (6048 ** 0.5) * 100  # Annualized hourly volatility
        
        # REAL INTRADAY INTELLIGENCE: Time-of-day patterns (1-hour data)
        if len(ticker_data) >= 240:  # Need 10 days of hourly data
            try:
                # Get recent 10 days of hourly data
                recent_data = ticker_data.tail(240)
                
                # Feature 1: Overnight Gap Behavior
                # Compare first hour of each day vs last hour of previous day
                # Positive gap = stock gaps up overnight (bullish)
                daily_opens = recent_data.iloc[::24]['Close'].values  # First hour of each day
                daily_closes = recent_data.iloc[23::24]['Close'].values  # Last hour of each day
                if len(daily_opens) > 1 and len(daily_closes) > 1:
                    # Align arrays (opens[1:] vs closes[:-1])
                    gaps = (daily_opens[1:] - daily_closes[:-1]) / daily_closes[:-1] * 100
                    avg_gap = gaps.mean() if len(gaps) > 0 else 0
                else:
                    avg_gap = 0
                
                # Feature 2: Intraday Volatility Pattern
                # High intraday volatility = more trading opportunity
                # Calculate average intraday range (high-low within each day)
                intraday_ranges = []
                for day_start in range(0, len(recent_data) - 24, 24):
                    day_data = recent_data.iloc[day_start:day_start+24]
                    if len(day_data) >= 24:
                        day_high = day_data['High'].max()
                        day_low = day_data['Low'].min()
                        day_open = day_data['Open'].iloc[0]
                        if day_open > 0:
                            intraday_range = (day_high - day_low) / day_open * 100
                            intraday_ranges.append(intraday_range)
                
                avg_intraday_range = sum(intraday_ranges) / len(intraday_ranges) if intraday_ranges else 0
                
                # Feature 3: Last Hour Momentum
                # Does stock rally or fade in the last hour of trading?
                # Positive = tends to rally into close (institutional accumulation)
                last_hour_moves = []
                for day_start in range(0, len(recent_data) - 24, 24):
                    day_data = recent_data.iloc[day_start:day_start+24]
                    if len(day_data) >= 24:
                        second_last_hour = day_data['Close'].iloc[-2]
                        last_hour = day_data['Close'].iloc[-1]
                        if second_last_hour > 0:
                            last_hour_move = (last_hour - second_last_hour) / second_last_hour * 100
                            last_hour_moves.append(last_hour_move)
                
                avg_last_hour_momentum = sum(last_hour_moves) / len(last_hour_moves) if last_hour_moves else 0
                
            except Exception as e:
                avg_gap = 0
                avg_intraday_range = 0
                avg_last_hour_momentum = 0
        else:
            avg_gap = 0
            avg_intraday_range = 0
            avg_last_hour_momentum = 0
        
        # Average volume
        avg_volume = ticker_data['Volume'].tail(min(30, len(ticker_data))).mean() if 'Volume' in ticker_data.columns else 0
        
        return {
            'perf_forward': perf_forward,  # Performance over AI_ELITE_FORWARD_DAYS
            'volatility': volatility,
            'avg_volume': avg_volume,
            'overnight_gap': avg_gap,  # Overnight gap behavior (intraday)
            'intraday_range': avg_intraday_range,  # Intraday volatility (intraday)
            'last_hour_momentum': avg_last_hour_momentum  # Last hour rally/fade (intraday)
        }
        
    except Exception as e:
        if not hasattr(_extract_features, '_err_logged') or _extract_features._err_logged < 3:
            print(f"   ❌ FEAT EXCEPTION {ticker}: {type(e).__name__}: {e}")
            if not hasattr(_extract_features, '_err_logged'):
                _extract_features._err_logged = 0
            _extract_features._err_logged += 1
        return None


def _load_hourly_data_direct(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """
    Load cached 1-hour data without converting to daily.
    Uses the same cache as load_prices but stops before daily conversion.
    """
    try:
        from data_utils import _RESOLVED_DATA_CACHE_DIR
        from utils import _to_utc
        
        cache_file = _RESOLVED_DATA_CACHE_DIR / f"{ticker}.csv"
        
        if not cache_file.exists():
            return None
        
        cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Convert to UTC if needed
        if cached_df.index.tz is None:
            cached_df.index = cached_df.index.tz_localize('UTC')
        elif str(cached_df.index.tz) != 'UTC':
            cached_df.index = cached_df.index.tz_convert('UTC')
        
        # Filter for requested range
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        result = cached_df.loc[(cached_df.index >= start_utc) & (cached_df.index <= end_utc)].copy()
        
        if len(result) >= 240:  # Need at least 10 days of hourly data
            return result
        
        return None
        
    except Exception as e:
        return None


def _load_or_create_model(model_path: Optional[str] = None):
    """
    Load existing ML model or train a new one if not available.
    """
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ AI Elite: Loaded ML model from {model_path}")
            return model
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to load model: {e}")
    
    # No model available - train a new one
    print(f"   🎓 AI Elite: No existing model found, training new model...")
    
    # For live trading, we need to train with available historical data
    # This is a simplified training approach for live trading
    try:
        from datetime import timedelta
        import xgboost as xgb
        from config import XGBOOST_USE_GPU
        
        # Create a simple XGBoost model for live trading
        # This will be trained on the fly with available data
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            tree_method='hist',
            device=device,
            verbosity=0,
            n_jobs=1
        )
        
        print(f"   🚀 AI Elite: Created new XGBoost model ({device})")
        
        # For live trading, we need to train the model with actual data
        # Use the proper training function instead of dummy data
        print(f"   🎓 AI Elite: No existing model found, training with real data...")
        
        # We need ticker data and date range for proper training
        # For now, create a minimal trained model that will be retrained with real data
        print(f"   🎓 AI Elite: Training model with REAL historical data...")
        
        # Train on ACTUAL historical data from the system (same as other strategies)
        try:
            print(f"   📊 Loading real historical market data (1-hour intraday)...")
            
            # Use the same data loading as backtesting strategies
            from config import AI_ELITE_INTRADAY_INTERVAL, AI_ELITE_INTRADAY_LOOKBACK, ALL_TICKERS
            
            # Load ALL stocks with 1-hour data for training
            print(f"   📊 Loading all available tickers for training ({AI_ELITE_INTRADAY_INTERVAL} data)...")
            all_ticker_data = {}
            
            # Set training date range (use last 6 months for training)
            from datetime import datetime, timedelta
            train_end = datetime.now()
            train_start = train_end - timedelta(days=180)
            
            for ticker in ALL_TICKERS[:50]:  # Limit to 50 tickers for faster training
                try:
                    # Get raw 1-hour data (before daily conversion)
                    data = _load_hourly_data_direct(ticker, train_start, train_end)
                    if data is not None and len(data) >= 240:  # Need at least 10 days of hourly data
                        all_ticker_data[ticker] = data
                except Exception as e:
                    continue
            
            if not all_ticker_data:
                raise Exception("No historical data loaded")
            
            # Convert to grouped format (same as backtesting uses)
            ticker_data_grouped = {}
            for ticker, data in all_ticker_data.items():
                if data is not None and len(data) > 0:
                    ticker_data_grouped[ticker] = data
            
            print(f"   📊 Training on {len(ticker_data_grouped)} real stocks with {AI_ELITE_INTRADAY_INTERVAL} data")
            print(f"   📊 Using {AI_ELITE_INTRADAY_LOOKBACK} days of {AI_ELITE_INTRADAY_INTERVAL} data per stock")
            
            # Extract real features from historical data
            X_real = []
            y_real = []
            
            for ticker, data in ticker_data_grouped.items():
                # Adjust minimum data requirement for intraday
                min_required = AI_ELITE_INTRADAY_LOOKBACK * 24  # 10 days * 24 hours = 240 hours
                if len(data) < min_required:
                    continue
                
                # Sample multiple time periods from each stock
                # For intraday: sample every 5 days (120 hours) to get enough data points
                # For daily: sample every 20 days
                sample_interval = 120  # Sample every 5 days (120 hours)
                start_point = len(data) - min_required
                
                for i in range(start_point, sample_interval, -sample_interval):
                    try:
                        # Get historical slice (use appropriate lookback for intraday vs daily)
                        lookback_points = AI_ELITE_INTRADAY_LOOKBACK * 24
                        historical_slice = data.iloc[i-lookback_points:i]
                        
                        # Extract features using the same function as live trading
                        current_date = data.index[i-1] if i < len(data) else data.index[-1]
                        features = _extract_features(ticker, historical_slice, current_date)
                        
                        if features is not None:
                            # Calculate what actually happened in next 20 days
                            # For intraday: 20 days = 480 hours, for daily: 20 days
                            future_points = 480  # 20 days * 24 hours
                            if i + future_points < len(data):
                                future_price = data.iloc[i+future_points]['Close']
                                current_price = data.iloc[i]['Close']
                                actual_return = (future_price - current_price) / current_price
                                
                                # Label: 1 if stock went up > 5% in next 20 days
                                label = 1 if actual_return > 0.05 else 0
                                
                                # Create feature vector with REAL intraday intelligence (6 features)
                                feature_vector = [
                                    features['perf_forward'],  # Performance over AI_ELITE_FORWARD_DAYS
                                    features['volatility'],
                                    features['avg_volume'],
                                    features.get('overnight_gap', 0),  # Overnight gap behavior
                                    features.get('intraday_range', 0),  # Intraday volatility
                                    features.get('last_hour_momentum', 0)  # Last hour momentum
                                ]
                                
                                X_real.append(feature_vector)
                                y_real.append(label)
                    
                    except Exception as e:
                        continue
            
            if len(X_real) < 50:
                raise Exception(f"Insufficient training samples: {len(X_real)}")
            
            import numpy as np
            X_real = np.array(X_real)
            y_real = np.array(y_real)
            
            print(f"   📊 Training on {len(X_real)} REAL historical samples")
            print(f"   📈 Positive samples: {sum(y_real)} ({sum(y_real)/len(y_real)*100:.1f}%)")
            print(f"   📉 Negative samples: {len(y_real) - sum(y_real)} ({(len(y_real)-sum(y_real))/len(y_real)*100:.1f}%)")
            
            # Train model on REAL historical data
            model.fit(X_real, y_real)
            print(f"   ✅ AI Elite: Model trained on REAL historical market data!")
            
        except Exception as e:
            print(f"   ⚠️ AI Elite: Real data training failed ({e}), using enhanced patterns")
            # Fallback to enhanced patterns (better than old fake patterns)
            import numpy as np
            
            # Create enhanced patterns based on real market statistics (with intraday features)
            n_samples = 200
            X_enhanced = []
            y_enhanced = []
            
            for i in range(n_samples):
                # Realistic ranges based on actual market data
                perf_forward = np.random.normal(5, 15)  # Performance over AI_ELITE_FORWARD_DAYS
                volatility = np.random.normal(30, 15)  # Typical volatility range
                volume = np.random.lognormal(14, 1)  # Log-normal volume distribution
                
                # REAL intraday intelligence features
                overnight_gap = np.random.normal(0.2, 1.5)  # Overnight gap behavior
                intraday_range = np.random.normal(3.0, 2.0)  # Intraday volatility (% range)
                last_hour_momentum = np.random.normal(0.1, 0.5)  # Last hour rally/fade
                
                # Success probability based on forward performance + real intraday
                success_prob = 0.3 + 0.4 * (1 / (1 + np.exp(-perf_forward/10)))  # Sigmoid on forward perf
                success_prob += 0.05 * (1 / (1 + np.exp(-overnight_gap)))  # Bonus for positive gaps
                success_prob += 0.05 * (1 / (1 + np.exp(-last_hour_momentum*10)))  # Bonus for last hour strength
                success_prob = min(success_prob, 0.9)  # Cap at 90%
                
                # Feature vector with REAL intraday intelligence (6 features)
                features = [
                    perf_forward, volatility, volume,
                    overnight_gap, intraday_range, last_hour_momentum
                ]
                
                X_enhanced.append(features)
                y_enhanced.append(1 if np.random.random() < success_prob else 0)
            
            X_enhanced = np.array(X_enhanced)
            y_enhanced = np.array(y_enhanced)
            
            model.fit(X_enhanced, y_enhanced)
            print(f"   🔄 AI Elite: Used enhanced patterns with intraday features ({len(X_enhanced)} samples)")
        
        # Save the model for future use
        if model_path:
            try:
                import os
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 AI Elite: Saved trained model to {model_path}")
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to save model: {e}")
        
        return model
        
    except Exception as e:
        print(f"   ⚠️ AI Elite: Failed to create model: {e}")
        return None


def _fallback_scoring(candidates_df: pd.DataFrame) -> np.ndarray:
    """
    Fallback scoring when ML model is not available.
    Uses simplified performance/volatility ratio with intraday bonuses.
    """
    # Base score: performance / volatility
    base_score = candidates_df['perf_forward'] / (candidates_df['volatility'] + 1)
    
    # Bonus for positive overnight gaps (bullish)
    gap_bonus = np.where(candidates_df['overnight_gap'] > 0, 1.1, 1.0)
    
    # Bonus for positive last hour momentum (institutional accumulation)
    last_hour_bonus = np.where(candidates_df['last_hour_momentum'] > 0, 1.05, 1.0)
    
    # Combined score with intraday intelligence
    scores = base_score * gap_bonus * last_hour_bonus
    
    return scores


def train_ai_elite_model(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    all_tickers: List[str],
    train_start_date: datetime,
    train_end_date: datetime,
    save_path: str = None,
    forward_days: int = 20
):
    """
    Train ML model to predict stock outperformance based on features.
    
    Uses walk-forward approach:
    1. For each date in training period, extract features for all stocks
    2. Label stocks based on their performance over next forward_days
    3. Train GradientBoostingClassifier to predict outperformance
    
    Args:
        ticker_data_grouped: Historical ticker data
        all_tickers: List of tickers to train on
        train_start_date: Start of training period
        train_end_date: End of training period
        save_path: Path to save trained model
        forward_days: Days ahead to predict (default 20)
        
    Returns:
        Trained model or None if training fails
    """
    # Import both XGBoost and sklearn GradientBoosting
    try:
        import xgboost as xgb
        from config import XGBOOST_USE_GPU
        xgb_available = True
        print(f"   🚀 AI Elite: Using XGBoost {'(GPU)' if XGBOOST_USE_GPU else '(CPU)'}")
    except ImportError:
        xgb_available = False
        print(f"   ⚠️ AI Elite: XGBoost not available, will use sklearn GradientBoosting (CPU only)")
    
    # Always import sklearn as fallback
    from sklearn.ensemble import GradientBoostingClassifier
    
    print(f"   🎓 AI Elite: Training ML model on {train_start_date.date()} to {train_end_date.date()}...")
    
    # Ensure dates are timezone-aware
    if train_start_date.tzinfo is None:
        train_start_date = train_start_date.replace(tzinfo=timezone.utc)
    if train_end_date.tzinfo is None:
        train_end_date = train_end_date.replace(tzinfo=timezone.utc)
    
    # Collect training samples
    training_data = []
    
    # Sample dates from training period (every 5 days to reduce computation)
    current_date = train_start_date
    sample_dates = []
    while current_date <= train_end_date:
        sample_dates.append(current_date)
        current_date += timedelta(days=5)
    
    print(f"   📊 AI Elite: Sampling {len(sample_dates)} dates for training...")
    print(f"   📊 AI Elite: Sample dates: {[d.date() for d in sample_dates]}")
    
    debug_count = 0
    features_none_count = 0
    forward_none_count = 0
    
    for sample_date in sample_dates:
        # Extract features for all tickers on this date
        for ticker in all_tickers:
            try:
                if ticker not in ticker_data_grouped:
                    continue
                
                ticker_data = ticker_data_grouped[ticker]
                if len(ticker_data) == 0:
                    continue
                
                # Debug first few tickers on first sample date
                if debug_count < 3 and sample_date == sample_dates[0]:
                    print(f"   🔍 TRAIN DEBUG {ticker}: index.tz={ticker_data.index.tz}, "
                          f"cols={list(ticker_data.columns[:5])}, shape={ticker_data.shape}, "
                          f"sample_date={sample_date}, sample_date.tz={sample_date.tzinfo}")
                
                # Extract features (uses adaptive lookback, min 60 days)
                features = _extract_features(ticker, ticker_data, sample_date)
                if features is None:
                    if debug_count < 3 and sample_date == sample_dates[0]:
                        print(f"   🔍 TRAIN DEBUG {ticker}: _extract_features returned None")
                        debug_count += 1
                    features_none_count += 1
                    continue
                
                # Calculate forward return (label)
                forward_return = _calculate_forward_return(
                    ticker_data, sample_date, forward_days
                )
                if forward_return is None:
                    forward_none_count += 1
                    continue
                
                # Store training sample (label will be assigned after collecting all samples)
                training_data.append({
                    'perf_forward': features['perf_forward'],
                    'volatility': features['volatility'],
                    'avg_volume': features['avg_volume'],
                    'overnight_gap': features.get('overnight_gap', 0),
                    'intraday_range': features.get('intraday_range', 0),
                    'last_hour_momentum': features.get('last_hour_momentum', 0),
                    'forward_return': forward_return,
                    'sample_date': sample_date
                })
                
            except Exception as e:
                continue
    
    print(f"   📊 AI Elite: Training loop done - samples={len(training_data)}, features_none={features_none_count}, forward_none={forward_none_count}")
    
    from config import MIN_TRAINING_SAMPLES_AI_ELITE
    if len(training_data) < MIN_TRAINING_SAMPLES_AI_ELITE:
        print(f"   ⚠️ AI Elite: Insufficient training data ({len(training_data)} samples), using fallback")
        return None
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    print(f"   📈 AI Elite: Collected {len(train_df)} training samples")
    
    # Assign labels based on absolute returns (making money focus)
    # Positive returns = 1, Negative returns = 0
    train_df['label'] = (train_df['forward_return'] > 0).astype(int)
    
    # Remove stocks with minimal returns (to avoid noise)
    min_return_threshold = 0.5  # 0.5% minimum return
    train_df = train_df[abs(train_df['forward_return']) >= min_return_threshold]
    
    print(f"   📊 AI Elite: Using absolute returns labeling")
    print(f"   📊 AI Elite: Positive returns: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    print(f"   📊 AI Elite: Average return: {train_df['forward_return'].mean():.2f}%")
    print(f"   📊 AI Elite: Training on {len(train_df)} samples")
    
    # Use new 6-feature model with real intraday intelligence
    print(f"   📊 AI Elite: Using 6 features with intraday intelligence")
    
    # Prepare features and labels
    feature_cols = ['perf_forward', 'volatility', 'avg_volume', 
                    'overnight_gap', 'intraday_range', 'last_hour_momentum']
    X = train_df[feature_cols].values
    y = train_df['label'].values
    
    # Train model
    if xgb_available:
        # Use XGBoost for GPU acceleration
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            tree_method='hist' if device == 'cuda' else 'hist',
            device=device,
            verbosity=0,
            n_jobs=1  # Prevent multiprocessing conflicts
        )
    else:
        # Fallback to sklearn
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    
    model.fit(X, y)
    
    # Calculate training accuracy
    train_accuracy = model.score(X, y)
    print(f"   ✅ AI Elite: Model trained! Accuracy: {train_accuracy*100:.1f}%")
    
    # Show feature importances
    importances = model.feature_importances_
    for i, col in enumerate(feature_cols):
        print(f"      {col}: {importances[i]:.3f}")
        
        # Save model if path provided
        if save_path:
            try:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   💾 AI Elite: Model saved to {save_path}")
            except Exception as e:
                print(f"   ⚠️ AI Elite: Failed to save model: {e}")
        
        return model


def _calculate_forward_return(
    ticker_data: pd.DataFrame,
    current_date: datetime,
    forward_days: int
) -> Optional[float]:
    """
    Calculate forward return for a stock over next N days.
    
    Args:
        ticker_data: Historical price data for ticker
        current_date: Current date
        forward_days: Number of days to look ahead
        
    Returns:
        Forward return percentage or None if insufficient data
    """
    try:
        # ✅ FIX: Deduplicate index (hourly data combined creates duplicates)
        if ticker_data.index.duplicated().any():
            ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
        
        # Convert current_date to pandas Timestamp - ensure UTC-aware
        current_date_tz = pd.Timestamp(current_date)
        if current_date_tz.tz is None:
            current_date_tz = current_date_tz.tz_localize('UTC')
        elif str(current_date_tz.tz) != 'UTC':
            current_date_tz = current_date_tz.tz_convert('UTC')
        
        # Ensure index is also UTC-aware
        if ticker_data.index.tz is None:
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_localize('UTC')
        elif str(ticker_data.index.tz) != 'UTC':
            ticker_data = ticker_data.copy()
            ticker_data.index = ticker_data.index.tz_convert('UTC')
        
        # Get current price
        current_data = ticker_data[ticker_data.index <= current_date_tz]
        if len(current_data) == 0:
            return None
        close_col = current_data['Close']
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        current_price = close_col.iloc[-1]
        
        # Get future price
        future_date = current_date_tz + timedelta(days=forward_days)
        future_data = ticker_data[(ticker_data.index > current_date_tz) & 
                                  (ticker_data.index <= future_date)]
        
        if len(future_data) == 0:
            return None
        
        future_close = future_data['Close']
        if isinstance(future_close, pd.DataFrame):
            future_close = future_close.iloc[:, 0]
        future_price = future_close.iloc[-1]
        
        # Calculate return
        forward_return = ((future_price - current_price) / current_price) * 100
        
        return forward_return
        
    except Exception as e:
        return None
