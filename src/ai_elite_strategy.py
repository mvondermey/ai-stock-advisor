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
    per_ticker_models: Dict[str, any] = None
) -> List[str]:
    """
    AI Elite Strategy: ML-based scoring of momentum + dip opportunities
    
    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        per_ticker_models: Dict of ticker -> trained model
        
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
    
    # Score candidates using PER-TICKER models
    feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                    'overnight_gap', 'intraday_range', 'last_hour_momentum',
                    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                    'volume_ratio', 'rsi_14',
                    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
                    # NEW: Mean reversion features
                    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd']

    candidates_df = pd.DataFrame(candidates)
    
    # Debug-only ranks (NOT ML features - risk_adj_mom_3m is already in feature_cols from _extract_features)
    candidates_df['momentum_rank'] = candidates_df['perf_3m'].rank(pct=True)
    candidates_df['risk_adj_mom_rank'] = candidates_df['risk_adj_mom_3m'].rank(pct=True)
    
    # Score each ticker with its own model (REGRESSION - predict risk_adj_return directly)
    ai_scores = []
    for idx, row in candidates_df.iterrows():
        ticker = row['ticker']
        
        # Get per-ticker model
        ticker_model = per_ticker_models.get(ticker) if per_ticker_models else None
        
        if ticker_model is None:
            print(f"   ⚠️ AI Elite: No model available for {ticker}, using score 0")
            ai_scores.append(0.0)
            continue
        
        # Extract features for this ticker (keep as DataFrame for feature name compatibility)
        X_ticker = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        
        # Get ML prediction (REGRESSION - predict continuous value)
        # Handle both new format (best_model) and legacy formats
        try:
            if isinstance(ticker_model, dict) and 'best_model' in ticker_model:
                # New format: use best_model directly
                pred_return = ticker_model['best_model'].predict(X_ticker)[0]
            elif isinstance(ticker_model, dict) and 'models' in ticker_model:
                # Legacy ensemble format: use first model with weight 1.0
                models = ticker_model['models']
                weights = ticker_model.get('weights', [1.0] * len(models))
                ensemble_pred = 0.0
                for model, weight in zip(models, weights):
                    pred = model.predict(X_ticker)[0]
                    ensemble_pred += pred * weight
                pred_return = ensemble_pred
            else:
                # Single model prediction
                pred_return = ticker_model.predict(X_ticker)[0]
            ai_scores.append(float(pred_return))
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to score {ticker}: {e}")
            ai_scores.append(0.0)
    
    candidates_df['ai_score'] = ai_scores
    
    # Pure AI scoring - the model predicts expected forward return directly
    candidates_df['final_score'] = candidates_df['ai_score']
    
    # Sort by final hybrid score
    candidates_df = candidates_df.sort_values('final_score', ascending=False)
    
    # Debug: show top candidates with momentum rank
    print(f"   ✅ AI Elite: Found {len(candidates_df)} candidates")
    print(f"   📊 AI Elite: Scoring = ML regression (predicts forward return)")
    for i, row in candidates_df.head(5).iterrows():
        print(f"      {i+1}. {row['ticker']}: PredReturn={row['final_score']:+.2f}% (RiskAdjMom={row['risk_adj_mom_rank']:.3f}), "
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
        # NEW: BOLLINGER BANDS & SMA DISTANCES (Mean Reversion Indicators)     #
        # ------------------------------------------------------------------ #
        # Bollinger position: 0 = at lower band, 1 = at upper band, 0.5 = at SMA
        bollinger_position = 0.5  # neutral default
        sma20_distance = 0.0  # neutral default (price vs SMA20 as %)
        sma50_distance = 0.0  # neutral default (price vs SMA50 as %)
        
        if len(close_daily) >= 20:
            sma20 = close_daily.tail(20).mean()
            std20 = close_daily.tail(20).std()
            if std20 > 0 and sma20 > 0:
                # Bollinger position: (price - lower) / (upper - lower)
                upper_band = sma20 + 2 * std20
                lower_band = sma20 - 2 * std20
                bollinger_position = (latest_price - lower_band) / (upper_band - lower_band)
                bollinger_position = max(0.0, min(1.0, bollinger_position))  # Clip to [0, 1]
                
                # SMA20 distance: (price - SMA20) / SMA20 * 100
                sma20_distance = ((latest_price - sma20) / sma20) * 100
        
        if len(close_daily) >= 50:
            sma50 = close_daily.tail(50).mean()
            if sma50 > 0:
                # SMA50 distance: (price - SMA50) / SMA50 * 100
                sma50_distance = ((latest_price - sma50) / sma50) * 100

        # MACD: MACD line - Signal line (momentum indicator)
        macd = 0.0  # neutral default
        if len(close_daily) >= 26:
            ema12 = close_daily.tail(26).ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = close_daily.tail(26).ewm(span=26, adjust=False).mean().iloc[-1]
            macd_line = ema12 - ema26
            # Signal line is 9-day EMA of MACD line
            macd_series = close_daily.tail(35).ewm(span=12, adjust=False).mean() - close_daily.tail(35).ewm(span=26, adjust=False).mean()
            signal_line = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
            macd = macd_line - signal_line

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
            # NEW: Mean reversion features
            'bollinger_position':    bollinger_position,
            'sma20_distance':        sma20_distance,
            'sma50_distance':        sma50_distance,
            'macd':                  macd,
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
    """Load existing ML model from disk. Returns None if no model exists."""
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            # Handle both old format (direct model) and new format (model + metadata)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                metadata = model_data.get('metadata', {})
                info_parts = []
                if 'trained' in metadata:
                    info_parts.append(f"trained {metadata['trained'][:10]}")
                elif 'updated' in metadata:
                    info_parts.append(f"updated {metadata['updated'][:10]}")
                if 'train_start' in metadata and 'train_end' in metadata:
                    info_parts.append(f"data {metadata['train_start'][:10]} to {metadata['train_end'][:10]}")
                info_str = f" ({', '.join(info_parts)})" if info_parts else ""
                print(f"   ✅ AI Elite: Loaded ML model from {model_path}{info_str}")
            else:
                model = model_data
                print(f"   ✅ AI Elite: Loaded ML model from {model_path} (legacy format)")
            return model
        except Exception as e:
            print(f"   ⚠️ AI Elite: Failed to load model: {e}")
    
    print(f"   ⚠️ AI Elite: No saved model found at {model_path}. Run backtesting first to train.")
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
    3. Train GradientBoostingRegressor to predict risk-adjusted return (REGRESSION)
    
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
    from sklearn.ensemble import GradientBoostingRegressor
    
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
                    # NEW: Mean reversion features
                    'bollinger_position': features.get('bollinger_position', 0.5),
                    'sma20_distance':     features.get('sma20_distance', 0),
                    'sma50_distance':     features.get('sma50_distance', 0),
                    'macd':               features.get('macd', 0),
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
    
    # Compute excess return vs market
    train_df['excess_return'] = train_df['forward_return'] - train_df['market_return']
    
    # Compute risk-adjusted return (excess_return / sqrt(volatility))
    vol_floored = train_df['volatility'].clip(lower=5.0)
    train_df['risk_adj_return'] = train_df['excess_return'] / (vol_floored ** 0.5)
    
    # Clip extreme outliers
    mean_ra = train_df['risk_adj_return'].mean()
    std_ra = train_df['risk_adj_return'].std()
    if std_ra > 0:
        train_df['risk_adj_return'] = train_df['risk_adj_return'].clip(
            lower=mean_ra - 3*std_ra, upper=mean_ra + 3*std_ra
        )
    
    # REGRESSION: Use risk_adj_return directly as continuous target
    train_df['label'] = train_df['risk_adj_return']
    
    print(f"   📊 AI Elite: Training regression model on {len(train_df)} samples")
    print(f"   📊 AI Elite: Risk-adjusted return stats: mean={mean_ra:.2f}, std={std_ra:.2f}")
    
    # Remove stocks with minimal returns (to avoid noise)
    min_return_threshold = 0.5  # 0.5% minimum return
    train_df = train_df[abs(train_df['forward_return']) >= min_return_threshold]
    
    print(f"   📊 AI Elite: Using 21 features (5 daily + 3 intraday + 6 derived + 2 sentiment proxy + 1 risk_adj_mom_3m + 4 mean reversion)")
    
    # Prepare features and target
    feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                    'overnight_gap', 'intraday_range', 'last_hour_momentum',
                    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                    'volume_ratio', 'rsi_14',
                    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
                    # NEW: Mean reversion features
                    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd']
    X = train_df[feature_cols].values
    y = train_df['label'].values  # Continuous risk-adjusted return values
    
    # Build candidate REGRESSOR models
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import warnings

    candidates = {}

    if xgb_available:
        device = 'cuda' if XGBOOST_USE_GPU else 'cpu'
        candidates['XGBoost'] = xgb.XGBRegressor(
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

    candidates['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42, verbose=0
    )
    
    # Add LightGBM
    try:
        import lightgbm as lgb
        candidates['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        print(f"   🚀 AI Elite: LightGBM available (CPU)")
    except ImportError:
        print(f"   ⚠️ AI Elite: LightGBM not available")
    
    candidates['RandomForest'] = RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=1
    )

    # Ridge regression needs scaled features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    candidates['Ridge'] = Ridge(alpha=1.0, random_state=42)

    # Cross-validate each model (R² for regression)
    from sklearn.metrics import r2_score, make_scorer
    
    r2_scorer = make_scorer(r2_score)
    
    best_model = None
    best_name = None
    best_score = -1.0
    cv_folds = 3

    print(f"   🏆 AI Elite: Selecting best model via {cv_folds}-fold cross-validation (R² for regression)...")
    for name, m in candidates.items():
        try:
            X_input = X_scaled if name == 'Ridge' else X
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(m, X_input, y, cv=cv_folds, scoring=r2_scorer, n_jobs=1)
            mean_score = scores.mean()
            print(f"      {name}: CV R² = {mean_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_model = m
        except Exception as e:
            print(f"      {name}: failed ({e})")

    if best_model is None:
        print(f"   ⚠️ AI Elite: All models failed CV, falling back to GradientBoosting")
        best_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, verbose=0
        )
        best_name = 'GradientBoosting'

    print(f"   ✅ AI Elite: Best model = {best_name} (CV R² {best_score:.3f})")

    # Fit best model on full training data
    X_input = X_scaled if best_name == 'Ridge' else X
    best_model.fit(X_input, y)
    train_r2 = r2_score(y, best_model.predict(X_input))
    print(f"   ✅ AI Elite: Final model trained! Train R²: {train_r2:.3f}")

    # Show feature importances (tree models only)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        sorted_feats = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        print(f"   📊 AI Elite: Top features:")
        for col, imp in sorted_feats[:5]:
            print(f"      {col}: {imp:.3f}")

    # Wrap Ridge with scaler so predict works on raw X
    if best_name == 'Ridge':
        from sklearn.pipeline import Pipeline
        best_model = Pipeline([('scaler', scaler), ('reg', best_model)])
        best_model.fit(X, y)

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
