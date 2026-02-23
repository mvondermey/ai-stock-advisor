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
            
            # Daily data (always available - from ticker_data_grouped)
            daily_data = ticker_data_grouped[ticker]
            if daily_data is None or len(daily_data) == 0:
                fail_reasons['empty'] += 1
                continue

            # Hourly data (optional - for intraday features)
            hourly_data = _load_hourly_data_direct(
                ticker,
                current_date - timedelta(days=30),
                current_date
            )
            
            # Debug first 3 tickers
            if debug_count < 3:
                has_hourly = hourly_data is not None and len(hourly_data) > 0
                print(f"   🔍 AI Elite DEBUG {ticker}: daily={len(daily_data)} rows, hourly={'yes' if has_hourly else 'no'}")
                debug_count += 1
            
            # Extract features using both data sources
            features = _extract_features(ticker, hourly_data, current_date, daily_data=daily_data)
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
    
    # Score candidates using ML model (17 features: 5 daily + 3 intraday + 6 derived + 2 sentiment proxy + 1 risk_adj_mom_3m)
    feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                    'overnight_gap', 'intraday_range', 'last_hour_momentum',
                    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                    'volume_ratio', 'rsi_14',
                    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m']

    if model is None:
        raise RuntimeError("AI Elite: model is None - training must have failed")

    candidates_df = pd.DataFrame(candidates)
    X = candidates_df[feature_cols].values
    
    # Debug-only ranks (NOT ML features - risk_adj_mom_3m is already in feature_cols from _extract_features)
    candidates_df['momentum_rank'] = candidates_df['perf_3m'].rank(pct=True)
    candidates_df['risk_adj_mom_rank'] = candidates_df['risk_adj_mom_3m'].rank(pct=True)
    
    # Get ML prediction probabilities (ordinal ranking: up to 5 classes)
    proba = model.predict_proba(X)
    n_classes = proba.shape[1]
    # Weighted score: higher probability for higher classes = better stock
    # Use actual model classes (may be fewer than 5 if pd.qcut dropped duplicates or fallback binary model)
    if hasattr(model, 'classes_'):
        class_weights = np.array(model.classes_, dtype=float)
    else:
        class_weights = np.arange(n_classes, dtype=float)
    max_class = class_weights.max() if class_weights.max() > 0 else 1.0
    weighted_class_score = np.dot(proba, class_weights) / max_class  # Normalize to 0-1
    candidates_df['ai_score'] = weighted_class_score
    
    # Pure AI scoring - the model is trained on risk-adjusted return labels,
    # so it already captures momentum + volatility internally
    candidates_df['final_score'] = candidates_df['ai_score']
    
    # Sort by final hybrid score
    candidates_df = candidates_df.sort_values('final_score', ascending=False)
    
    # Debug: show top candidates with momentum rank
    print(f"   ✅ AI Elite: Found {len(candidates_df)} candidates")
    print(f"   📊 AI Elite: Scoring = 100% ML prediction (trained on risk-adjusted return quintiles)")
    for i, row in candidates_df.head(5).iterrows():
        print(f"      {i+1}. {row['ticker']}: Final={row['final_score']:.3f} (AI={row['ai_score']:.3f}, RiskAdjMom={row['risk_adj_mom_rank']:.3f}), "
              f"3M={row['perf_3m']:+.1f}%, Vol={row['volatility']:.1f}%, RiskAdj={row['risk_adj_mom_3m']:.2f})")
    
    # Return top N tickers by final hybrid score
    selected = candidates_df.head(top_n)['ticker'].tolist()
    return selected


def _extract_features(ticker: str, hourly_data: Optional[pd.DataFrame], current_date: datetime,
                      daily_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Extract ML features using BOTH data sources:
    - hourly_data: intraday features (overnight gap, intraday range, last-hour momentum)
    - daily_data:  daily features (3m/6m/1y performance, volatility, volume)
    
    If hourly_data is None, intraday features default to 0.
    daily_data is required - returns None if missing.
    """
    try:
        from config import AI_ELITE_INTRADAY_LOOKBACK

        # --- Normalise current_date to UTC timestamp ---
        current_ts = pd.Timestamp(current_date)
        if current_ts.tz is None:
            current_ts = current_ts.tz_localize('UTC')
        else:
            current_ts = current_ts.tz_convert('UTC')

        # ------------------------------------------------------------------ #
        # DAILY FEATURES  (3m / 6m / 1y performance, volatility, volume)     #
        # ------------------------------------------------------------------ #
        if daily_data is None or len(daily_data) == 0:
            return None

        # Deduplicate and sort
        if daily_data.index.duplicated().any():
            daily_data = daily_data[~daily_data.index.duplicated(keep='last')]
        daily_data = daily_data.sort_index()

        # Ensure UTC
        if daily_data.index.tz is None:
            daily_data = daily_data.copy()
            daily_data.index = daily_data.index.tz_localize('UTC')
        else:
            daily_data = daily_data.copy()
            daily_data.index = daily_data.index.tz_convert('UTC')

        daily_filtered = daily_data[daily_data.index <= current_ts]
        if len(daily_filtered) < 20:
            return None

        close_daily = daily_filtered['Close']
        if isinstance(close_daily, pd.DataFrame):
            close_daily = close_daily.iloc[:, 0]
        close_daily = close_daily.dropna()
        if len(close_daily) < 20:
            return None

        latest_price = close_daily.iloc[-1]
        if latest_price <= 0:
            return None

        def _perf(days):
            idx = min(days, len(close_daily) - 1)
            p = close_daily.iloc[-idx]
            return ((latest_price - p) / p * 100) if p > 0 else 0.0

        perf_3m  = _perf(63)
        perf_6m  = _perf(126)
        perf_1y  = _perf(252)

        # Annualised daily volatility (252 trading days)
        daily_returns = close_daily.pct_change().dropna()
        volatility_daily = daily_returns.std() * (252 ** 0.5) * 100 if len(daily_returns) >= 10 else 0.0
        daily_vol_pct = daily_returns.std() * 100 if len(daily_returns) >= 10 else 0.0

        # Risk-Adj Mom 3M explicit feature (the winning signal): return / sqrt(volatility)
        # Floor volatility at 5% to prevent extreme values
        risk_adj_mom_3m = perf_3m / (max(volatility_daily, 5.0) ** 0.5)

        # Average volume (daily)
        avg_volume = daily_filtered['Volume'].tail(30).mean() if 'Volume' in daily_filtered.columns else 0.0

        # Volume ratio: recent 20d avg vs prior historical avg (rising volume = conviction)
        volume_ratio = 1.0
        if 'Volume' in daily_filtered.columns:
            vol_series = daily_filtered['Volume'].dropna()
            if len(vol_series) >= 40:
                recent_vol = vol_series.tail(20).mean()
                prior_vol = vol_series.iloc[:-20].mean()
                if prior_vol > 0:
                    volume_ratio = recent_vol / prior_vol

        # Risk-Adj Mom core signal: return / sqrt(daily_vol_pct)
        risk_adj_score = perf_1y / (daily_vol_pct ** 0.5 + 0.001) if daily_vol_pct > 0 else 0.0

        # Elite Hybrid dip signal: strong 1Y but weak 3M
        dip_score = perf_1y - perf_3m

        # Momentum acceleration: recent 3M vs prior 3M (6M - 3M)
        mom_accel = perf_3m - perf_6m

        # Volatility sweet-spot: 1 if annualised vol is 20-40% (Elite Hybrid bonus zone)
        vol_sweet_spot = 1.0 if 20.0 <= volatility_daily <= 40.0 else 0.0

        # RSI-14: Relative Strength Index (overbought >70, oversold <30)
        rsi_14 = 50.0  # neutral default
        if len(daily_returns) >= 14:
            gains = daily_returns.clip(lower=0)
            losses = (-daily_returns).clip(lower=0)
            avg_gain = gains.tail(14).mean()
            avg_loss = losses.tail(14).mean()
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_14 = 100.0 - (100.0 / (1.0 + rs))
            elif avg_gain > 0:
                rsi_14 = 100.0  # no losses = max RSI

        # ------------------------------------------------------------------ #
        # SENTIMENT PROXY FEATURES  (price-derived, no API needed)            #
        # ------------------------------------------------------------------ #
        # Short-term reversal: 5-day return minus 20-day return
        # Positive = recent acceleration (bullish sentiment), negative = fading (bearish)
        perf_5d = _perf(5)
        perf_20d = _perf(20)
        short_term_reversal = perf_5d - perf_20d

        # Volume sentiment: volume surge signed by price direction
        # High volume on up-moves = bullish conviction, high volume on down-moves = panic
        volume_sentiment = 0.0
        if 'Volume' in daily_filtered.columns and len(close_daily) >= 20:
            vol_series = daily_filtered['Volume'].dropna()
            if len(vol_series) >= 20:
                recent_5d_vol = vol_series.tail(5).mean()
                avg_20d_vol = vol_series.tail(20).mean()
                vol_surge = (recent_5d_vol / avg_20d_vol) - 1.0 if avg_20d_vol > 0 else 0.0
                # Sign by 5-day price direction
                price_direction = 1.0 if perf_5d > 0 else (-1.0 if perf_5d < 0 else 0.0)
                volume_sentiment = vol_surge * price_direction

        # ------------------------------------------------------------------ #
        # INTRADAY FEATURES  (overnight gap, intraday range, last-hour mom)   #
        # ------------------------------------------------------------------ #
        avg_gap = 0.0
        avg_intraday_range = 0.0
        avg_last_hour_momentum = 0.0

        if hourly_data is not None and len(hourly_data) >= AI_ELITE_INTRADAY_LOOKBACK * 24:
            try:
                if hourly_data.index.duplicated().any():
                    hourly_data = hourly_data[~hourly_data.index.duplicated(keep='last')]
                if hourly_data.index.tz is None:
                    hourly_data = hourly_data.copy()
                    hourly_data.index = hourly_data.index.tz_localize('UTC')
                else:
                    hourly_data = hourly_data.copy()
                    hourly_data.index = hourly_data.index.tz_convert('UTC')

                recent_h = hourly_data[hourly_data.index <= current_ts].tail(AI_ELITE_INTRADAY_LOOKBACK * 24)

                if len(recent_h) >= AI_ELITE_INTRADAY_LOOKBACK * 24:
                    # Overnight gap
                    daily_opens_h  = recent_h.iloc[::24]['Close'].values
                    daily_closes_h = recent_h.iloc[23::24]['Close'].values
                    min_len = min(len(daily_opens_h), len(daily_closes_h))
                    if min_len > 1:
                        gaps = (daily_opens_h[1:min_len] - daily_closes_h[:min_len-1]) / daily_closes_h[:min_len-1] * 100
                        avg_gap = float(gaps.mean()) if len(gaps) > 0 else 0.0

                    # Intraday range
                    ranges = []
                    for ds in range(0, len(recent_h) - 24, 24):
                        day = recent_h.iloc[ds:ds+24]
                        if len(day) >= 24 and day['Open'].iloc[0] > 0:
                            ranges.append((day['High'].max() - day['Low'].min()) / day['Open'].iloc[0] * 100)
                    avg_intraday_range = float(np.mean(ranges)) if ranges else 0.0

                    # Last-hour momentum
                    lh_moves = []
                    for ds in range(0, len(recent_h) - 24, 24):
                        day = recent_h.iloc[ds:ds+24]
                        if len(day) >= 24 and day['Close'].iloc[-2] > 0:
                            lh_moves.append((day['Close'].iloc[-1] - day['Close'].iloc[-2]) / day['Close'].iloc[-2] * 100)
                    avg_last_hour_momentum = float(np.mean(lh_moves)) if lh_moves else 0.0

            except Exception:
                pass  # intraday features stay 0

        return {
            'perf_3m':               perf_3m,
            'perf_6m':               perf_6m,
            'perf_1y':               perf_1y,
            'volatility':            volatility_daily,
            'avg_volume':            avg_volume,
            'overnight_gap':         avg_gap,
            'intraday_range':        avg_intraday_range,
            'last_hour_momentum':    avg_last_hour_momentum,
            'risk_adj_score':        risk_adj_score,
            'dip_score':             dip_score,
            'mom_accel':             mom_accel,
            'vol_sweet_spot':        vol_sweet_spot,
            'volume_ratio':          volume_ratio,
            'rsi_14':                rsi_14,
            'short_term_reversal':   short_term_reversal,
            'volume_sentiment':      volume_sentiment,
            'risk_adj_mom_3m':       risk_adj_mom_3m,
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
        
        if len(result) >= 120:  # Need at least 20 trading days of hourly data (~6 hours/day)
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
                                actual_return = (future_price - current_price) / current_price * 100
                                
                                # Create feature vector (17 features: 5 daily + 3 intraday + 6 derived + 2 sentiment proxy + 1 risk_adj_mom_3m)
                                feature_vector = [
                                    features.get('perf_3m', 0),
                                    features.get('perf_6m', 0),
                                    features.get('perf_1y', 0),
                                    features['volatility'],
                                    features['avg_volume'],
                                    features.get('overnight_gap', 0),
                                    features.get('intraday_range', 0),
                                    features.get('last_hour_momentum', 0),
                                    features.get('risk_adj_score', 0),
                                    features.get('dip_score', 0),
                                    features.get('mom_accel', 0),
                                    features.get('vol_sweet_spot', 0),
                                    features.get('volume_ratio', 1.0),
                                    features.get('rsi_14', 50.0),
                                    features.get('short_term_reversal', 0),
                                    features.get('volume_sentiment', 0),
                                    features.get('risk_adj_mom_3m', 0),
                                ]
                                
                                X_real.append(feature_vector)
                                y_real.append(actual_return)  # Store raw return for ordinal ranking
                    
                    except Exception as e:
                        continue
            
            if len(X_real) < 50:
                raise Exception(f"Insufficient training samples: {len(X_real)}")
            
            import numpy as np
            X_real = np.array(X_real)
            y_real = np.array(y_real)
            
            print(f"   📊 Training on {len(X_real)} REAL historical samples")
            
            # Convert to DataFrame to compute risk-adjusted returns
            X_df = pd.DataFrame(X_real, columns=[
                'perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                'overnight_gap', 'intraday_range', 'last_hour_momentum',
                'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                'volume_ratio', 'rsi_14',
                'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m'
            ])
            y_series = pd.Series(y_real)
            
            # Compute risk-adjusted return: excess_return / sqrt(volatility)
            # For fallback, we use raw return minus a fixed market estimate (~2%)
            # Floor volatility at 5% to prevent extreme outliers
            excess_return = y_series - 2.0  # Rough market return estimate
            vol_floored = X_df['volatility'].clip(lower=5.0)
            risk_adj_return = excess_return / (vol_floored ** 0.5)
            # Clip extreme outliers
            mean_ra = risk_adj_return.mean()
            std_ra = risk_adj_return.std()
            if std_ra > 0:
                risk_adj_return = risk_adj_return.clip(lower=mean_ra - 3*std_ra, upper=mean_ra + 3*std_ra)
            
            # Convert to ordinal labels based on risk-adjusted return
            y_ordinal = pd.qcut(risk_adj_return, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
            y_real = y_ordinal.astype(int).values
            n_classes = len(np.unique(y_real))
            print(f"   📊 Risk-adjusted ordinal labels: {n_classes} classes, distribution: {dict(zip(*np.unique(y_real, return_counts=True)))}")
            
            # Train model on REAL historical data with ordinal labels
            model.fit(X_real, y_real)
            print(f"   ✅ AI Elite: Model trained on REAL historical market data!")
            
        except Exception as e:
            print(f"   ⚠️ AI Elite: Real data training failed ({e}), using enhanced patterns")
            # Fallback to enhanced patterns with 17 features and ordinal labels
            import numpy as np
            
            n_samples = 500
            X_enhanced = []
            returns_enhanced = []
            
            for i in range(n_samples):
                perf_3m  = np.random.normal(5, 15)
                perf_6m  = np.random.normal(10, 20)
                perf_1y  = np.random.normal(15, 30)
                volatility = max(5.0, np.random.normal(30, 15))
                volume = np.random.lognormal(14, 1)
                overnight_gap = np.random.normal(0.2, 1.5)
                intraday_range = np.random.normal(3.0, 2.0)
                last_hour_momentum = np.random.normal(0.1, 0.5)
                daily_vol_pct = volatility / (252 ** 0.5)
                risk_adj_score = perf_1y / (daily_vol_pct ** 0.5 + 0.001) if daily_vol_pct > 0 else 0.0
                dip_score = perf_1y - perf_3m
                mom_accel = perf_3m - perf_6m
                vol_sweet_spot = 1.0 if 20.0 <= volatility <= 40.0 else 0.0
                volume_ratio = max(0.5, np.random.normal(1.0, 0.3))
                rsi_14 = np.clip(np.random.normal(50, 15), 10, 90)
                short_term_reversal = np.random.normal(0, 5)  # 5d vs 20d return diff
                # Volume sentiment: volume surge * price direction
                vol_surge = max(-0.5, np.random.normal(0, 0.3))
                price_dir = 1.0 if perf_3m > 0 else -1.0
                volume_sentiment = vol_surge * price_dir
                # Risk-Adj Mom 3M explicit feature
                risk_adj_mom_3m = perf_3m / (max(volatility, 5.0) ** 0.5)
                
                # Simulate forward return based on features
                simulated_return = (perf_3m * 0.3 + perf_6m * 0.2 + risk_adj_score * 0.1 
                                   + np.random.normal(0, 10))
                
                # Feature vector (17 features: matching select_ai_elite_stocks)
                features = [
                    perf_3m, perf_6m, perf_1y, volatility, volume,
                    overnight_gap, intraday_range, last_hour_momentum,
                    risk_adj_score, dip_score, mom_accel, vol_sweet_spot,
                    volume_ratio, rsi_14,
                    short_term_reversal, volume_sentiment, risk_adj_mom_3m
                ]
                
                X_enhanced.append(features)
                returns_enhanced.append(simulated_return)
            
            X_enhanced = np.array(X_enhanced)
            # Convert to risk-adjusted ordinal labels (same as main training)
            X_df = pd.DataFrame(X_enhanced, columns=[
                'perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                'overnight_gap', 'intraday_range', 'last_hour_momentum',
                'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                'volume_ratio', 'rsi_14',
                'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m'
            ])
            y_series = pd.Series(returns_enhanced)
            # Risk-adjusted: excess return / sqrt(volatility)
            # Floor volatility at 5% to prevent extreme outliers
            excess_return = y_series - 2.0  # Rough market return estimate
            vol_floored = X_df['volatility'].clip(lower=5.0)
            risk_adj_return = excess_return / (vol_floored ** 0.5)
            # Clip extreme outliers
            mean_ra = risk_adj_return.mean()
            std_ra = risk_adj_return.std()
            if std_ra > 0:
                risk_adj_return = risk_adj_return.clip(lower=mean_ra - 3*std_ra, upper=mean_ra + 3*std_ra)
            y_ordinal = pd.qcut(risk_adj_return, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
            y_enhanced = y_ordinal.astype(int).values
            
            model.fit(X_enhanced, y_enhanced)
            print(f"   🔄 AI Elite: Used enhanced patterns with 17 features + risk-adjusted ordinal labels ({len(X_enhanced)} samples)")
        
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

    from config import AI_ELITE_INTRADAY_LOOKBACK
    hourly_cache: Dict[str, Optional[pd.DataFrame]] = {}
    
    # Sample dates from training period (every 2 days for more samples)
    current_date = train_start_date
    sample_dates = []
    while current_date <= train_end_date:
        sample_dates.append(current_date)
        current_date += timedelta(days=2)
    
    print(f"   📊 AI Elite: Sampling {len(sample_dates)} dates for training...")
    print(f"   📊 AI Elite: Sample dates: {[d.date() for d in sample_dates]}")
    
    debug_count = 0
    features_none_count = 0
    forward_none_count = 0
    
    # Pre-calculate market returns for all sample dates
    market_returns = {}
    print(f"   📊 AI Elite: Calculating market returns for relative labeling...")
    for sample_date in sample_dates:
        market_ret = _calculate_market_return(ticker_data_grouped, sample_date, forward_days)
        market_returns[sample_date] = market_ret if market_ret is not None else 0.0
    
    for sample_date in sample_dates:
        # Extract features for all tickers on this date
        for ticker in all_tickers:
            try:
                if ticker not in ticker_data_grouped:
                    continue

                # Daily data (always available)
                daily_data = ticker_data_grouped[ticker]
                if daily_data is None or len(daily_data) == 0:
                    continue

                # Hourly data (optional - load once per ticker into cache)
                if ticker not in hourly_cache:
                    load_start = train_start_date - timedelta(days=AI_ELITE_INTRADAY_LOOKBACK + 5)
                    load_end = train_end_date + timedelta(days=forward_days + 2)
                    hourly_cache[ticker] = _load_hourly_data_direct(ticker, load_start, load_end)
                hourly_data = hourly_cache[ticker]

                # Debug first few tickers on first sample date
                if debug_count < 3 and sample_date == sample_dates[0]:
                    has_h = hourly_data is not None and len(hourly_data) > 0
                    print(f"   🔍 TRAIN DEBUG {ticker}: daily={len(daily_data)} rows, hourly={'yes' if has_h else 'no'}")
                    debug_count += 1

                # Extract features using both data sources
                features = _extract_features(ticker, hourly_data, sample_date, daily_data=daily_data)
                if features is None:
                    features_none_count += 1
                    continue
                
                # Calculate forward return from daily data (label)
                forward_return = _calculate_forward_return(
                    daily_data, sample_date, forward_days
                )
                if forward_return is None:
                    forward_none_count += 1
                    continue
                
                # Get market return for this sample date
                market_return = market_returns.get(sample_date, 0.0)
                
                # Store training sample
                training_data.append({
                    'perf_3m':            features['perf_3m'],
                    'perf_6m':            features['perf_6m'],
                    'perf_1y':            features['perf_1y'],
                    'volatility':         features['volatility'],
                    'avg_volume':         features['avg_volume'],
                    'overnight_gap':      features.get('overnight_gap', 0),
                    'intraday_range':     features.get('intraday_range', 0),
                    'last_hour_momentum': features.get('last_hour_momentum', 0),
                    'risk_adj_score':     features.get('risk_adj_score', 0),
                    'dip_score':          features.get('dip_score', 0),
                    'mom_accel':          features.get('mom_accel', 0),
                    'vol_sweet_spot':     features.get('vol_sweet_spot', 0),
                    'volume_ratio':       features.get('volume_ratio', 1.0),
                    'rsi_14':             features.get('rsi_14', 50.0),
                    'short_term_reversal': features.get('short_term_reversal', 0),
                    'volume_sentiment':   features.get('volume_sentiment', 0),
                    'risk_adj_mom_3m':    features.get('risk_adj_mom_3m', 0),
                    'forward_return':     forward_return,
                    'market_return':      market_return,
                    'sample_date':        sample_date
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
    
    # Compute excess return vs market (needed for ordinal ranking)
    train_df['excess_return'] = train_df['forward_return'] - train_df['market_return']
    
    # IMPROVED: Compute risk-adjusted return (excess_return / sqrt(volatility))
    # This teaches the model to prefer efficient returns per unit of risk
    # Same formula that makes Risk-Adj Mom 3M successful: return / sqrt(volatility)
    # Floor volatility at 5% to prevent extreme outliers when vol=0 (5478x normal)
    vol_floored = train_df['volatility'].clip(lower=5.0)
    train_df['risk_adj_return'] = train_df['excess_return'] / (vol_floored ** 0.5)
    # Clip extreme outliers (>3 std from mean) to prevent quintile corruption
    mean_ra = train_df['risk_adj_return'].mean()
    std_ra = train_df['risk_adj_return'].std()
    if std_ra > 0:
        train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
            lower=mean_ra - 3 * std_ra, upper=mean_ra + 3 * std_ra
        )
    
    print(f"   📊 AI Elite: Average stock return: {train_df['forward_return'].mean():.2f}%")
    print(f"   📊 AI Elite: Average market return: {train_df['market_return'].mean():.2f}%")
    print(f"   📊 AI Elite: Average excess return: {train_df['excess_return'].mean():.2f}%")
    print(f"   📊 AI Elite: Average volatility: {train_df['volatility'].mean():.1f}%")
    print(f"   📊 AI Elite: Average risk-adjusted return: {train_df['risk_adj_return'].mean():.2f}")
    
    # IMPROVED: Use risk-adjusted return for ordinal ranking
    # This captures efficiency of outperformance, not just magnitude
    # Stocks with same excess return but lower volatility get higher labels
    # Top 20% = label 4, next 20% = 3, middle 20% = 2, next 20% = 1, bottom 20% = 0
    train_df['label'] = pd.qcut(train_df['risk_adj_return'], 
                                q=5, 
                                labels=[0, 1, 2, 3, 4],
                                duplicates='drop').astype(int)
    
    n_label_classes = train_df['label'].nunique()
    print(f"   📊 AI Elite: Risk-adjusted quintile labels (0=worst, 4=best), {n_label_classes} classes:")
    for label in sorted(train_df['label'].unique(), reverse=True):
        count = (train_df['label'] == label).sum()
        avg_risk_adj = train_df[train_df['label'] == label]['risk_adj_return'].mean()
        avg_excess = train_df[train_df['label'] == label]['excess_return'].mean()
        avg_vol = train_df[train_df['label'] == label]['volatility'].mean()
        print(f"      Label {label}: {count} samples, risk_adj={avg_risk_adj:.2f}, excess={avg_excess:.1f}%, vol={avg_vol:.1f}%")
    
    # Remove stocks with minimal returns (to avoid noise)
    min_return_threshold = 0.5  # 0.5% minimum return
    train_df = train_df[abs(train_df['forward_return']) >= min_return_threshold]
    
    print(f"   📊 AI Elite: Using 17 features (5 daily + 3 intraday + 6 derived + 2 sentiment proxy + 1 risk_adj_mom_3m)")
    
    # Prepare features and labels
    feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                    'overnight_gap', 'intraday_range', 'last_hour_momentum',
                    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                    'volume_ratio', 'rsi_14',
                    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m']
    X = train_df[feature_cols].values
    y = train_df['label'].values
    
    # Build candidate models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import warnings

    candidates = {}

    if xgb_available:
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        candidates['XGBoost'] = xgb.XGBClassifier(
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

    candidates['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, verbose=0
    )
    
    # Add LightGBM (CPU - LightGBM GPU requires OpenCL, not available on CUDA-only systems)
    try:
        import lightgbm as lgb
        candidates['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=-1  # Suppress LightGBM output
        )
        print(f"   🚀 AI Elite: LightGBM available (CPU)")
    except ImportError:
        print(f"   ⚠️ AI Elite: LightGBM not available")
    candidates['RandomForest'] = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=1
    )

    # Logistic Regression needs scaled features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    candidates['LogisticRegression'] = LogisticRegression(
        max_iter=500, random_state=42, n_jobs=1
    )

    # Cross-validate each model and pick the best
    # IMPROVED: Use kappa scoring for ordinal ranking instead of binary accuracy
    from sklearn.metrics import cohen_kappa_score, make_scorer
    
    # Custom scorer: weighted kappa (accounts for ordinality - being off by 1 is better than off by 4)
    def weighted_kappa(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    kappa_scorer = make_scorer(weighted_kappa)
    
    best_model = None
    best_name = None
    best_score = -1.0
    cv_folds = max(2, min(3, len(np.unique(y))))  # 2-3 folds, respect class count

    print(f"   🏆 AI Elite: Selecting best model via {cv_folds}-fold cross-validation (weighted kappa for ordinal ranking)...")
    for name, m in candidates.items():
        try:
            X_input = X_scaled if name == 'LogisticRegression' else X
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use kappa scorer for ordinal ranking instead of accuracy
                scores = cross_val_score(m, X_input, y, cv=cv_folds, scoring=kappa_scorer, n_jobs=1)
            mean_score = scores.mean()
            print(f"      {name}: CV weighted kappa = {mean_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_model = m
        except Exception as e:
            print(f"      {name}: failed ({e})")

    if best_model is None:
        print(f"   ⚠️ AI Elite: All models failed CV, falling back to GradientBoosting")
        best_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=0
        )
        best_name = 'GradientBoosting'

    print(f"   ✅ AI Elite: Best model = {best_name} (CV weighted kappa {best_score:.3f})")

    # Fit best model on full training data
    X_input = X_scaled if best_name == 'LogisticRegression' else X
    best_model.fit(X_input, y)
    train_kappa = weighted_kappa(y, best_model.predict(X_input))
    print(f"   ✅ AI Elite: Final model trained! Train weighted kappa: {train_kappa:.3f}")

    # Show feature importances (tree models only)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        sorted_feats = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        print(f"   📊 AI Elite: Top features:")
        for col, imp in sorted_feats[:5]:
            print(f"      {col}: {imp:.3f}")

    # Wrap LogisticRegression with scaler so predict_proba works on raw X
    if best_name == 'LogisticRegression':
        from sklearn.pipeline import Pipeline
        best_model = Pipeline([('scaler', scaler), ('clf', best_model)])
        best_model.fit(X, y)  # refit pipeline on raw X

    # Save model if path provided
    if save_path:
        try:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"   💾 AI Elite: Model saved to {save_path}")
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to save model: {e}")

    return best_model


def _calculate_market_return(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    forward_days: int,
    market_ticker: str = 'SPY'  # Use SPY as market proxy
) -> Optional[float]:
    """
    Calculate market return for the same period.
    Uses SPY as market proxy, falls back to equal-weighted average of available stocks.
    """
    try:
        # Try SPY first
        if market_ticker in ticker_data_grouped:
            market_data = ticker_data_grouped[market_ticker]
            market_return = _calculate_forward_return(market_data, current_date, forward_days)
            if market_return is not None:
                return market_return
        
        # Fallback: equal-weighted average of all available stocks
        returns = []
        for ticker, data in ticker_data_grouped.items():
            if len(data) > 0:
                ret = _calculate_forward_return(data, current_date, forward_days)
                if ret is not None:
                    returns.append(ret)
        
        if returns:
            return np.mean(returns)
        return None
        
    except Exception:
        return None


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
