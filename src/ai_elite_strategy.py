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
    
    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]
    
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "AI Elite"
    )
    
    print(f"   🤖 AI Elite: Analyzing {len(filtered_tickers)} tickers with ML scoring (filtered from {len(all_tickers)})")
    
    # Sequential prediction - avoids nested multiprocessing deadlocks
    # (backtesting already uses multiprocessing, nested Pools cause semaphore leaks)
    import time
    
    start_time = time.time()
    
    ai_scores = {}
    fail_reasons = {'not_in_data': 0, 'empty': 0, 'features_none': 0, 'no_model': 0, 'exception': 0}
    
    for ticker in filtered_tickers:
        ticker_data = ticker_data_grouped.get(ticker)
        ticker_model = per_ticker_models.get(ticker) if per_ticker_models else None
        args = (ticker, ticker_data, current_date, ticker_model)
        
        ticker_result, score, status = _predict_ticker_worker(args)
        if status == 'success':
            ai_scores[ticker_result] = score
        else:
            fail_reasons[status] += 1
            if status == 'no_model':
                ai_scores[ticker_result] = 0.0
    
    elapsed = time.time() - start_time
    print(f"   📊 AI Elite: Predicted {len(ai_scores)} tickers ({elapsed:.1f}s)")
    
    if not ai_scores:
        print(f"   ⚠️ AI Elite: No predictions found")
        print(f"   🔍 AI Elite: Fail reasons: {fail_reasons}")
        return []
    
    # Create candidates DataFrame from successful predictions
    feature_cols = ['perf_3m', 'perf_6m', 'perf_1y', 'volatility', 'avg_volume',
                    'overnight_gap', 'intraday_range', 'last_hour_momentum',
                    'risk_adj_score', 'dip_score', 'mom_accel', 'vol_sweet_spot',
                    'volume_ratio', 'rsi_14',
                    'short_term_reversal', 'volume_sentiment', 'risk_adj_mom_3m',
                    # NEW: Mean reversion features
                    'bollinger_position', 'sma20_distance', 'sma50_distance', 'macd']
    
    # Extract features for successful tickers (sequential - much faster now)
    candidates = []
    debug_count = 0
    for ticker, ai_score in ai_scores.items():
        try:
            daily_data = ticker_data_grouped[ticker]
            hourly_data = _load_hourly_data_direct(ticker, 
                current_date - timedelta(days=30), current_date)
            features = _extract_features(ticker, hourly_data, current_date, daily_data=daily_data)
            if features is None:
                continue
            
            features['ticker'] = ticker
            features['ai_score'] = ai_score
            candidates.append(features)
            
            # Debug first 3 tickers
            if debug_count < 3:
                has_hourly = hourly_data is not None and len(hourly_data) > 0
                print(f"   🔍 AI Elite DEBUG {ticker}: daily={len(daily_data)} rows, hourly={'yes' if has_hourly else 'no'}")
                debug_count += 1
        except Exception:
            continue
    
    candidates_df = pd.DataFrame(candidates)
    
    # Debug-only ranks
    candidates_df['momentum_rank'] = candidates_df['perf_3m'].rank(pct=True)
    candidates_df['risk_adj_mom_rank'] = candidates_df['risk_adj_mom_3m'].rank(pct=True)
    
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


# NOTE: train_ai_elite_model was removed - training now happens in ai_elite_strategy_per_ticker.py
# via train_shared_base_model(), called from shared_strategies.select_ai_elite_with_training()


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


def _predict_ticker_worker(args):
    """Top-level worker for multiprocessing - predicts score for one ticker."""
    ticker, ticker_data, current_date, ticker_model = args
    
    try:
        if ticker_data is None or len(ticker_data) == 0:
            return ticker, None, 'empty'
        
        # Load hourly data
        hourly_data = _load_hourly_data_direct(ticker, 
            current_date - timedelta(days=30),
            current_date + timedelta(days=5))
        
        # Extract features
        features = _extract_features(ticker, hourly_data, current_date, daily_data=ticker_data)
        if features is None:
            return ticker, None, 'features_none'
        
        # Get model and predict
        if ticker_model is None:
            return ticker, 0.0, 'no_model'
        
        # Extract the actual model from ensemble dict
        if isinstance(ticker_model, dict):
            actual_model = ticker_model.get('best_model')
            if actual_model is None:
                return ticker, 0.0, 'no_model'
        else:
            actual_model = ticker_model
        
        # Predict (keep as DataFrame for compatibility)
        from ai_elite_strategy_per_ticker import FEATURE_COLS
        # Use explicit feature ordering to match training
        feature_values = [features.get(col, 0.0) for col in FEATURE_COLS]
        X_ticker = pd.DataFrame([feature_values], columns=FEATURE_COLS)
        ai_score = actual_model.predict(X_ticker)[0]
        
        return ticker, ai_score, 'success'
    except Exception as e:
        import traceback
        print(f"   ⚠️ AI Elite worker exception for {ticker}: {e}")
        traceback.print_exc()
        return ticker, None, 'exception'
