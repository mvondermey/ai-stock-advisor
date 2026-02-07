"""
Shared Strategy Implementations
Used by both backtesting and live trading to ensure identical logic.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Import config for strategy parameters
from config import (
    RISK_ADJ_MOM_PERFORMANCE_WINDOW, RISK_ADJ_MOM_VOLATILITY_WINDOW,
    RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_CONFIRM_SHORT,
    RISK_ADJ_MOM_CONFIRM_MEDIUM, RISK_ADJ_MOM_CONFIRM_LONG, RISK_ADJ_MOM_MIN_CONFIRMATIONS,
    RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION, RISK_ADJ_MOM_VOLUME_WINDOW, RISK_ADJ_MOM_VOLUME_MULTIPLIER,
    RISK_ADJ_MOM_MIN_SCORE, VOLATILITY_ADJ_MOM_LOOKBACK, VOLATILITY_ADJ_MOM_VOL_WINDOW,
    VOLATILITY_ADJ_MOM_MIN_SCORE, PARALLEL_THRESHOLD, DATA_FRESHNESS_MAX_DAYS,
    ENABLE_MULTITASK_LEARNING
)

# Import multi-task learning strategy
if ENABLE_MULTITASK_LEARNING:
    try:
        from multitask_strategy import select_multitask_stocks
        MULTITASK_AVAILABLE = True
    except ImportError:
        MULTITASK_AVAILABLE = False
else:
    MULTITASK_AVAILABLE = False


def calculate_risk_adjusted_momentum_score(ticker_data: pd.DataFrame, current_date: datetime = None, train_start_date: datetime = None) -> tuple:
    """
    Calculate risk-adjusted momentum score for a ticker.
    
    Returns:
        tuple: (score, return_pct, volatility_pct) or (0, 0, 0) if insufficient data
    """
    if len(ticker_data) < 100:
        return 0.0, 0.0, 0.0
    
    # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
    end_date = ticker_data.index.max()
    
    # If current_date is provided, try to use it (must convert to pandas Timestamp first)
    if current_date is not None:
        try:
            # Convert to pandas Timestamp (handles both datetime and Timestamp)
            current_ts = pd.Timestamp(current_date)
            
            # If data has timezone, ensure current_ts matches
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    # current_ts is naive, localize to data's timezone
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    # current_ts has timezone, convert to data's timezone
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)
            
            # Use the minimum of current_ts and data's max (can't use future data)
            end_date = min(current_ts, end_date)
            
            # Check data freshness - warn if data is older than configured limit
            data_age_days = (current_ts - end_date).total_seconds() / 86400
            if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                # Data is stale - raise error to be caught by caller
                raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")
        except ValueError:
            # Re-raise ValueError (stale data error) so caller can handle it
            raise
        except Exception:
            # For other errors (timezone conversion, etc), stick with data's max date
            pass
    
    # Calculate 1-year performance with train_start_date constraint
    start_date = end_date - timedelta(days=RISK_ADJ_MOM_PERFORMANCE_WINDOW)
    if train_start_date:
        start_date = max(train_start_date, start_date)
    
    # ✅ Use boolean indexing instead of .loc[] to avoid KeyError with timezone mismatches
    perf_data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= end_date)]
    
    if len(perf_data) < 50:
        return 0.0, 0.0, 0.0
    
    valid_close = perf_data['Close'].dropna()
    if len(valid_close) < 10:
        return 0.0, 0.0, 0.0
    
    start_price = valid_close.iloc[0]
    end_price = valid_close.iloc[-1]
    
    if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
        return 0.0, 0.0, 0.0
    
    # FIXED: Allow negative returns with good risk-adjusted scores
    # Calculate risk-adjusted score for all returns, not just positive ones
    basic_return = ((end_price - start_price) / start_price) * 100
    
    # Calculate volatility
    daily_returns = valid_close.pct_change().dropna()
    if len(daily_returns) <= 5:
        return 0.0, 0.0, 0.0
    
    # Calculate volatility using full performance window
    volatility = daily_returns.std() * 100
    
    # Risk-adjusted score - allow negative returns but penalize appropriately
    # Use absolute return in numerator but keep sign in score
    risk_adj_score = basic_return / (volatility**0.5 + 0.001)
    
    return risk_adj_score, basic_return, volatility


def check_momentum_confirmation(ticker_data: pd.DataFrame, current_date: datetime = None, train_start_date: datetime = None) -> int:
    """
    Check momentum confirmation across multiple timeframes.
    
    Returns:
        int: Number of timeframes with positive momentum
    """
    if not RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
        return 1  # No confirmation required
    
    momentum_confirmations = 0
    
    # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
    end_date = ticker_data.index.max()
    
    # If current_date is provided, try to use it (must convert to pandas Timestamp first)
    if current_date is not None:
        try:
            # Convert to pandas Timestamp (handles both datetime and Timestamp)
            current_ts = pd.Timestamp(current_date)
            
            # If data has timezone, ensure current_ts matches
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    # current_ts is naive, localize to data's timezone
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    # current_ts has timezone, convert to data's timezone
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)
            
            # Check data freshness - warn if data is older than configured limit
            data_age_days = (current_ts - end_date).total_seconds() / 86400
            if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                # Data is stale - raise error to be caught by caller
                raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")
            
            # Use the minimum of current_ts and data's max (can't use future data)
            end_date = min(current_ts, end_date)
        except ValueError:
            # Re-raise ValueError (stale data error) so caller can handle it
            raise
        except Exception:
            # For other errors (timezone conversion, etc), stick with data's max date
            pass
    
    # 3-month momentum check
    if RISK_ADJ_MOM_CONFIRM_SHORT:
        start_3m = end_date - timedelta(days=90)
        if train_start_date:
            start_3m = max(train_start_date, start_3m)
        
        # ✅ Use boolean indexing instead of .loc[] to avoid KeyError
        data_3m = ticker_data[(ticker_data.index >= start_3m) & (ticker_data.index <= end_date)]
        if len(data_3m) >= 30:
            valid_close = data_3m['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_3m = ((end_price - start_price) / start_price) * 100
                    if return_3m > 0:
                        momentum_confirmations += 1
    
    # 6-month momentum check
    if RISK_ADJ_MOM_CONFIRM_MEDIUM:
        start_6m = end_date - timedelta(days=180)
        if train_start_date:
            start_6m = max(train_start_date, start_6m)
        data_6m = ticker_data[(ticker_data.index >= start_6m) & (ticker_data.index <= end_date)]
        if len(data_6m) >= 60:
            valid_close = data_6m['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_6m = ((end_price - start_price) / start_price) * 100
                    if return_6m > 0:
                        momentum_confirmations += 1
    
    # 1-year momentum check
    if RISK_ADJ_MOM_CONFIRM_LONG:
        start_1y = end_date - timedelta(days=365)
        if train_start_date:
            start_1y = max(train_start_date, start_1y)
        data_1y = ticker_data[(ticker_data.index >= start_1y) & (ticker_data.index <= end_date)]
        if len(data_1y) >= 100:
            valid_close = data_1y['Close'].dropna()
            if len(valid_close) >= 2:
                start_price = valid_close.iloc[0]
                end_price = valid_close.iloc[-1]
                if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
                    return_1y = ((end_price - start_price) / start_price) * 100
                    if return_1y > 0:
                        momentum_confirmations += 1
    
    return momentum_confirmations


def check_volume_confirmation(ticker_data: pd.DataFrame) -> bool:
    """
    Check volume confirmation criteria.
    
    Returns:
        bool: True if volume confirmation passes, False otherwise
    """
    if not RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION:
        return True  # No volume confirmation required
    
    volume_data = ticker_data['Volume'].dropna()
    if len(volume_data) < RISK_ADJ_MOM_VOLUME_WINDOW + 20:
        return True  # Insufficient data, pass by default
    
    recent_volume = volume_data.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
    avg_volume = volume_data.head(len(volume_data) - RISK_ADJ_MOM_VOLUME_WINDOW).mean()
    
    if avg_volume <= 0:
        return True  # Invalid average, pass by default
    
    return recent_volume >= avg_volume * RISK_ADJ_MOM_VOLUME_MULTIPLIER


def select_risk_adj_mom_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                               current_date: datetime = None, train_start_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Risk-Adjusted Momentum stock selection logic.
    Used by both backtesting and live trading.
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for analysis (None for last available)
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    # Use parallel processing for large ticker lists
    from config import PARALLEL_THRESHOLD
    if len(all_tickers) > PARALLEL_THRESHOLD:  # Use parallel only for large lists
        try:
            from parallel_backtest import calculate_parallel_risk_adj_scores
            from config import NUM_PROCESSES
            
            # Calculate scores in parallel
            scores_data = calculate_parallel_risk_adj_scores(
                all_tickers,
                ticker_data_grouped,
                current_date,
                train_start_date
            )
            
            # Apply filters
            current_top_performers = []
            momentum_filtered = 0
            volume_filtered = 0
            data_issues = 0
            
            for ticker, score, return_pct, volatility_pct in scores_data:
                try:
                    ticker_data = ticker_data_grouped[ticker]
                    
                    # Check momentum confirmation
                    momentum_confirmations = check_momentum_confirmation(ticker_data, current_date, train_start_date)
                    
                    if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                        if momentum_confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                            momentum_filtered += 1
                            continue
                    
                    # Check volume confirmation
                    if not check_volume_confirmation(ticker_data):
                        volume_filtered += 1
                        continue
                    
                    if score > RISK_ADJ_MOM_MIN_SCORE:
                        current_top_performers.append((ticker, score, return_pct, volatility_pct))
                        
                except Exception:
                    data_issues += 1
                    continue
            
            analyzed_count = len(scores_data)
            
        except ImportError:
            # Fallback to sequential if parallel module not available
            return select_risk_adj_mom_stocks_sequential(all_tickers, ticker_data_grouped, current_date, train_start_date, top_n)
    else:
        # Use sequential for small lists
        return select_risk_adj_mom_stocks_sequential(all_tickers, ticker_data_grouped, current_date, train_start_date, top_n)
    
    # Sort by risk-adjusted score and get top N
    if current_top_performers:
        current_top_performers.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, ret, vol in current_top_performers[:top_n]]
        
        # Debug info
        confirm_parts = []
        if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
            confirm_parts.append("momentum")
        if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION:
            confirm_parts.append("volume")
        confirm_text = f" (with {' + '.join(confirm_parts)} confirmation)" if confirm_parts else ""
        
        print(f"   📊 Analysis: {analyzed_count} processed (PARALLEL), {momentum_filtered} momentum filtered, {volume_filtered} volume filtered, {data_issues} data issues")
        print(f"   🎯 Selected {len(selected_tickers)} stocks{confirm_text}:")
        for ticker, score, ret, vol in current_top_performers[:top_n]:
            print(f"      {ticker}: score={score:.2f}, return={ret:.1f}%, vol={vol:.1f}%")
        
        return selected_tickers
    else:
        print(f"   ❌ No stocks passed filtering criteria (analyzed: {analyzed_count})")
        return []


def select_risk_adj_mom_stocks_sequential(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                         current_date: datetime = None, train_start_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Sequential version of Risk-Adjusted Momentum stock selection (original implementation).
    """
    import time
    from config import (RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_MIN_CONFIRMATIONS,
                       RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION, RISK_ADJ_MOM_MIN_SCORE)
    start_time = time.time()
    
    current_top_performers = []
    analyzed_count = 0
    momentum_filtered = 0
    volume_filtered = 0
    data_issues = 0
    stale_data_count = 0
    
    # Debug: Show first few tickers and keys
    if len(all_tickers) > 0 and len(ticker_data_grouped) > 0:
        print(f"   🔍 DEBUG: First 3 all_tickers: {all_tickers[:3]}")
        all_keys = list(ticker_data_grouped.keys())
        print(f"   🔍 DEBUG: Total keys in ticker_data_grouped: {len(all_keys)}")
        # Check if first ticker exists
        first_ticker = all_tickers[0]
        print(f"   🔍 DEBUG: '{first_ticker}' in ticker_data_grouped: {first_ticker in all_keys}")
        # Find matching tickers
        matching = [t for t in all_tickers if t in all_keys]
        print(f"   🔍 DEBUG: Matching tickers: {len(matching)} of {len(all_tickers)}")
        if matching:
            df = ticker_data_grouped[matching[0]]
            print(f"   🔍 DEBUG: {matching[0]} data shape: {df.shape}, index: {type(df.index).__name__}, range: {df.index.min()} to {df.index.max()}")
    
    for ticker in all_tickers:
        try:
            analyzed_count += 1
            
            if ticker not in ticker_data_grouped:
                data_issues += 1
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            # Check momentum confirmation
            momentum_confirmations = check_momentum_confirmation(ticker_data, current_date, train_start_date)
            
            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                if momentum_confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    momentum_filtered += 1
                    continue
            
            # Check volume confirmation
            if not check_volume_confirmation(ticker_data):
                volume_filtered += 1
                continue
            
            # Calculate risk-adjusted score
            score, return_pct, volatility_pct = calculate_risk_adjusted_momentum_score(ticker_data, current_date, train_start_date)
            
            # Debug first few tickers
            if analyzed_count <= 3:
                print(f"   🔍 DEBUG: {ticker} score={score:.2f}, return={return_pct:.1f}%, vol={volatility_pct:.1f}%, min_score={RISK_ADJ_MOM_MIN_SCORE}")
            
            if score > RISK_ADJ_MOM_MIN_SCORE:  # Use configurable minimum score
                current_top_performers.append((ticker, score, return_pct, volatility_pct))
        
        except Exception as e:
            # Check if it's a stale data error
            error_str = str(e)
            if "Data too old" in error_str:
                stale_data_count += 1
                if stale_data_count <= 3:
                    print(f"   🔍 DEBUG: {ticker} {error_str}")
            else:
                data_issues += 1
                if data_issues <= 3:
                    print(f"   🔍 DEBUG: {ticker} exception: {type(e).__name__}: {error_str}")
            continue
    
    # Sort by risk-adjusted score and get top N
    elapsed = time.time() - start_time
    if current_top_performers:
        current_top_performers.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, ret, vol in current_top_performers[:top_n]]
        
        # Debug info
        confirm_parts = []
        if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
            confirm_parts.append("momentum")
        if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION:
            confirm_parts.append("volume")
        confirm_text = f" (with {' + '.join(confirm_parts)} confirmation)" if confirm_parts else ""
        
        # Build summary message
        summary_parts = [f"{analyzed_count} processed (SEQUENTIAL) in {elapsed:.2f}s"]
        summary_parts.append(f"{momentum_filtered} momentum filtered")
        summary_parts.append(f"{volume_filtered} volume filtered")
        if stale_data_count > 0:
            summary_parts.append(f"⚠️ {stale_data_count} stale data (>{DATA_FRESHNESS_MAX_DAYS} days old)")
        if data_issues > 0:
            summary_parts.append(f"{data_issues} other data issues")
        
        print(f"   📊 Analysis: {', '.join(summary_parts)}")
        print(f"   🎯 Selected {len(selected_tickers)} stocks{confirm_text}:")
        for ticker, score, ret, vol in current_top_performers[:top_n]:
            print(f"      {ticker}: score={score:.2f}, return={ret:.1f}%, vol={vol:.1f}%")
        
        return selected_tickers
    else:
        # Build failure summary
        summary_parts = [f"analyzed: {analyzed_count}"]
        if stale_data_count > 0:
            summary_parts.append(f"⚠️ {stale_data_count} rejected due to stale data (>{DATA_FRESHNESS_MAX_DAYS} days old)")
        if data_issues > 0:
            summary_parts.append(f"{data_issues} data issues")
        
        print(f"   ❌ No stocks passed filtering criteria ({', '.join(summary_parts)})")
        return []


def select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, period='1y', current_date=None, top_n=20):
    """
    Shared Dynamic Buy & Hold stock selection logic.
    """
    performances = []
    
    # DEBUG: Print current_date and period
    print(f"   🔍 DEBUG select_dynamic_bh_stocks: period={period}, current_date={current_date}")
    
    # Use current date or last available date
    if current_date is None:
        # Find the latest date across all tickers
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []
    
    # Determine lookback period
    if period == '1y':
        lookback_days = 365
    elif period == '6m':
        lookback_days = 180
    elif period == '3m':
        lookback_days = 90
    elif period == '1m':
        lookback_days = 30
    else:
        lookback_days = 365
    
    stale_data_count = 0
    analyzed_count = 0
    
    for ticker in all_tickers:  # Process all tickers
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            analyzed_count += 1
            ticker_data = ticker_data_grouped[ticker]
            
            # ✅ Check data freshness before processing
            data_max_date = ticker_data.index.max()
            current_ts = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_ts.tz is None:
                    current_ts = current_ts.tz_localize(ticker_data.index.tz)
                else:
                    current_ts = current_ts.tz_convert(ticker_data.index.tz)
            
            data_age_days = (current_ts - data_max_date).total_seconds() / 86400
            if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                stale_data_count += 1
                if stale_data_count <= 3:  # Only print first 3
                    print(f"   🔍 DEBUG: {ticker} data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")
                continue  # Skip stale data instead of raising exception
            
            # Calculate start date based on lookback period
            start_date = current_date - timedelta(days=lookback_days)
            
            # DEBUG: Show date calculation for first few tickers
            if analyzed_count <= 3:
                print(f"   🔍 DEBUG {ticker}: current_date={current_date.date()}, lookback_days={lookback_days}, start_date={start_date.date()}")
            
            # Filter data to the exact period - be more flexible with date range
            available_start = ticker_data.index.min()
            available_end = ticker_data.index.max()
            
            # DEBUG: Show available data range
            if analyzed_count <= 3:
                print(f"   🔍 DEBUG {ticker}: available_start={available_start.date()}, available_end={available_end.date()}")
            
            # Use available data if it covers most of the period
            if available_start > start_date:
                days_short = (available_start - start_date).days
                # DEBUG: Show days short calculation
                if analyzed_count <= 3:
                    print(f"   🔍 DEBUG {ticker}: available_start > start_date by {days_short} days")
                # Use whatever data is available (no threshold)
                start_date = available_start
                if analyzed_count <= 3:
                    print(f"   🔍 DEBUG {ticker}: ADJUSTING start_date to {start_date.date()}")
            
            period_data = ticker_data[(ticker_data.index >= start_date) & (ticker_data.index <= current_ts)]
            
            # Reduce minimum data requirement for live trading
            min_data_points = max(5, lookback_days // 30)  # At least 5 points or 1 per month
            if len(period_data) < min_data_points:
                continue
            
            valid_close = period_data['Close'].dropna()
            if len(valid_close) < 2:
                continue
            
            start_price = valid_close.iloc[0]
            end_price = valid_close.iloc[-1]
            
            if start_price > 0:
                performance = ((end_price - start_price) / start_price) * 100
                # Include all stocks (not just positive performance) to ensure we have picks
                performances.append((ticker, performance))
        
        except Exception as e:
            print(f"   🔍 DEBUG: {ticker} error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Check if all data was stale
    if stale_data_count > 0 and stale_data_count == analyzed_count:
        print(f"   ❌ No stocks passed filtering (⚠️ {stale_data_count} rejected due to stale data (>{DATA_FRESHNESS_MAX_DAYS} days old))")
        return []
    
    # Sort by performance and get top N
    if performances:
        performances.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, _ in performances[:top_n]]
        
        print(f"   📊 Top {top_n} performers ({period}): {selected_tickers}")
        for i, (ticker, perf) in enumerate(performances[:top_n], 1):
            print(f"      {i}. {ticker}: {perf:+.1f}%")
        
        return selected_tickers
    else:
        if stale_data_count > 0:
            print(f"   ❌ No valid performance data found for {period} (⚠️ {stale_data_count} stale data)")
        else:
            print(f"   ❌ No valid performance data found for {period}")
        return []


def select_volatility_adj_mom_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                     current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Volatility-Adjusted Momentum stock selection logic.
    """
    current_top_performers = []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < VOLATILITY_ADJ_MOM_LOOKBACK:
                continue
            
            # Calculate momentum return over lookback period
            if len(ticker_data) >= VOLATILITY_ADJ_MOM_LOOKBACK:
                momentum_return = (ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-VOLATILITY_ADJ_MOM_LOOKBACK] - 1)
            else:
                momentum_return = 0.0
            
            # Only include stocks with positive momentum return
            if momentum_return <= 0:
                continue
            
            # Calculate volatility
            daily_returns = ticker_data['Close'].pct_change().dropna()
            if len(daily_returns) >= VOLATILITY_ADJ_MOM_VOL_WINDOW:
                volatility = daily_returns.iloc[-VOLATILITY_ADJ_MOM_VOL_WINDOW:].std()
            else:
                volatility = daily_returns.std()
            
            # Avoid division by zero
            if volatility <= 0:
                continue
            
            # Volatility-adjusted momentum score
            vol_adjusted_score = momentum_return / (volatility ** 0.5)
            
            if vol_adjusted_score >= VOLATILITY_ADJ_MOM_MIN_SCORE:
                current_top_performers.append((ticker, vol_adjusted_score, momentum_return * 100, volatility * 100))
        
        except Exception:
            continue
    
    # Sort by volatility-adjusted score and get top N
    if current_top_performers:
        current_top_performers.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, ret, vol in current_top_performers[:top_n]]
        
        print(f"   📊 Top {top_n} volatility-adjusted momentum: {selected_tickers}")
        for ticker, score, ret, vol in current_top_performers[:top_n]:
            print(f"      {ticker}: score={score:.3f}, return={ret:.1f}%, vol={vol:.1f}%")
        
        return selected_tickers
    else:
        print(f"   ❌ No stocks passed volatility-adjusted momentum criteria")
        return []


def select_mean_reversion_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Mean Reversion stock selection logic.
    Selects oversold stocks based on recent price decline.
    """
    oversold_candidates = []
    
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < 50:  # Need at least 50 days of data
                continue
            
            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)
            
            # Calculate recent performance (last 20 days) using date filtering
            recent_start = current_date_tz - timedelta(days=20)
            recent_data = ticker_data[(ticker_data.index >= recent_start) & (ticker_data.index <= current_date_tz)]
            if len(recent_data) >= 2:
                recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1) * 100
            else:
                recent_return = 0.0
            
            # Calculate longer-term performance (last 100 days) using date filtering
            longer_start = current_date_tz - timedelta(days=100)
            longer_data = ticker_data[(ticker_data.index >= longer_start) & (ticker_data.index <= current_date_tz)]
            if len(longer_data) >= 2:
                longer_return = (longer_data['Close'].iloc[-1] / longer_data['Close'].iloc[0] - 1) * 100
            else:
                longer_return = 0.0
            
            # FIXED: Mean reversion signal - look for oversold conditions, not just recent decline
            # Use RSI-like logic: recent decline but not too severe, with longer-term strength
            if recent_return < -5 and recent_return > -25 and longer_return > -5:  # Moderate recent drop, not severe long-term decline
                # Better scoring: prioritize stocks with larger recent drops but solid long-term performance
                reversion_score = (-recent_return * 0.7) + (longer_return * 0.3)  # Weight recent drop more
                oversold_candidates.append((ticker, reversion_score, recent_return, longer_return))
        
        except Exception:
            continue
    
    # Sort by reversion score and get top N
    if oversold_candidates:
        oversold_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, recent_ret, longer_ret in oversold_candidates[:top_n]]
        
        print(f"   📊 Top {top_n} mean reversion candidates: {selected_tickers}")
        for ticker, score, recent_ret, longer_ret in oversold_candidates[:top_n]:
            print(f"      {ticker}: score={score:.2f}, recent={recent_ret:.1f}%, longer={longer_ret:.1f}%")
        
        return selected_tickers
    else:
        print(f"   ❌ No oversold candidates found")
        return []


def select_quality_momentum_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                   current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Shared Quality + Momentum stock selection logic.
    Combines fundamental quality indicators with momentum.
    """
    quality_momentum_candidates = []
    
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            if len(ticker_data) < 200:  # Need enough data for quality assessment
                continue
            
            # ✅ BETTER FIX: Use data's max date and convert current_date to pandas Timestamp
            current_date_tz = ticker_data.index.max()
            
            # If current_date is provided, try to use it (must convert to pandas Timestamp first)
            if current_date is not None:
                try:
                    # Convert to pandas Timestamp (handles both datetime and Timestamp)
                    current_ts = pd.Timestamp(current_date)
                    
                    # If data has timezone, ensure current_ts matches
                    if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                        if current_ts.tz is None:
                            # current_ts is naive, localize to data's timezone
                            current_ts = current_ts.tz_localize(ticker_data.index.tz)
                        else:
                            # current_ts has timezone, convert to data's timezone
                            current_ts = current_ts.tz_convert(ticker_data.index.tz)
                    
                    # Check data freshness - warn if data is older than configured limit
                    data_age_days = (current_ts - current_date_tz).total_seconds() / 86400
                    if data_age_days > DATA_FRESHNESS_MAX_DAYS:
                        # Data is stale - raise error
                        raise ValueError(f"Data too old: {data_age_days:.1f} days (max {DATA_FRESHNESS_MAX_DAYS} days)")
                    
                    # Use the minimum of current_ts and data's max (can't use future data)
                    current_date_tz = min(current_ts, current_date_tz)
                except ValueError:
                    # Re-raise ValueError (stale data error) so it gets caught in outer exception handler
                    raise
                except Exception:
                    # For other errors (timezone conversion, etc), stick with data's max date
                    pass
            
            # Momentum calculation (1-year) using date filtering
            momentum_start = current_date_tz - timedelta(days=365)  # 1 year for better performance measurement
            momentum_data = ticker_data[(ticker_data.index >= momentum_start) & (ticker_data.index <= current_date_tz)]
            
            if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                print(f"   🔍 DEBUG: {ticker} data range: {ticker_data.index.min()} to {ticker_data.index.max()}")
                print(f"   🔍 DEBUG: {ticker} momentum range: {momentum_start} to {current_date_tz}")
                print(f"   🔍 DEBUG: {ticker} momentum_data points: {len(momentum_data)}")
            if len(momentum_data) >= 2:
                # Drop NaN values from Close prices
                valid_prices = momentum_data['Close'].dropna()
                if len(valid_prices) < 2:
                    momentum_return = 0.0
                else:
                    start_price = valid_prices.iloc[0]
                    end_price = valid_prices.iloc[-1]
                    if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                        momentum_calc = (end_price / start_price - 1) * 100
                        print(f"   🔍 DEBUG: {ticker} start_price={start_price}, end_price={end_price}")
                        print(f"   🔍 DEBUG: {ticker} momentum={momentum_calc:.1f}%")
                    if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
                        momentum_return = 0.0
                    else:
                        momentum_return = (end_price / start_price - 1) * 100
            else:
                momentum_return = 0.0
            
            # Check for NaN values
            if pd.isna(momentum_return):
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum=nan% (NaN value, filtered)")
                continue
            
            # Only include stocks with positive momentum
            if momentum_return <= 0:
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum={momentum_return:.1f}% (<=0, filtered)")
                continue
            
            # Quality indicators (simplified)
            # 1. Price stability (lower volatility is better)
            daily_returns = ticker_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # 2. Trend consistency (positive recent performance) using date filtering
            short_start = current_date_tz - timedelta(days=30)
            short_data = ticker_data[(ticker_data.index >= short_start) & (ticker_data.index <= current_date_tz)]
            if len(short_data) >= 2:
                short_trend = (short_data['Close'].iloc[-1] / short_data['Close'].iloc[0] - 1) * 100
            else:
                short_trend = 0.0
            
            # Quality score: momentum with stability bonus
            stability_bonus = max(0, 50 - volatility) / 50  # Higher bonus for lower volatility
            trend_bonus = max(0, short_trend) / 100  # Bonus for positive short trend
            
            quality_score = momentum_return * (1 + stability_bonus + trend_bonus)
            
            if momentum_return > 0:  # Only consider positive momentum (lowered from 5%)
                quality_momentum_candidates.append((ticker, quality_score, momentum_return, volatility))
            else:
                if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                    print(f"   🔍 DEBUG: {ticker} momentum={momentum_return:.1f}% (<=0%, filtered)")
        
        except Exception as e:
            if ticker in ['SNDK', 'WDC', 'MU', 'SLV', 'STX', 'NEM']:  # Debug first few
                print(f"   🔍 DEBUG: {ticker} exception: {type(e).__name__}: {e}")
            continue
    
    # Sort by quality score and get top N
    if quality_momentum_candidates:
        quality_momentum_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, score, mom, vol in quality_momentum_candidates[:top_n]]
        
        print(f"   📊 Top {top_n} quality + momentum: {selected_tickers}")
        for ticker, score, mom, vol in quality_momentum_candidates[:top_n]:
            print(f"      {ticker}: score={score:.1f}, momentum={mom:.1f}%, vol={vol:.1f}%")
        
        return selected_tickers
    else:
        print(f"   ❌ No quality + momentum candidates found")
        return []


def select_sector_rotation_etfs(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                current_date: datetime, top_n: int = 5) -> List[str]:
    """
    Select top performing sector ETFs based on momentum.
    PROPOSAL 2: Sector Rotation Strategy
    """
    from config import SECTOR_ROTATION_MOMENTUM_WINDOW, SECTOR_ROTATION_MIN_MOMENTUM
    
    print(f"   🔍 Sector Rotation Debug: Looking for ETFs in {len(all_tickers)} available tickers")
    print(f"   🔍 Momentum window: {SECTOR_ROTATION_MOMENTUM_WINDOW} days, Min threshold: {SECTOR_ROTATION_MIN_MOMENTUM}%")
    
    # Check if sector ETFs are in the available tickers
    sector_etfs = [
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB',
        'GDX', 'USO', 'TLT'
    ]
    
    available_sector_etfs = [etf for etf in sector_etfs if etf in all_tickers]
    print(f"   🔍 Available sector ETFs: {available_sector_etfs}")
    
    if not available_sector_etfs:
        print(f"   ❌ No sector ETFs found in ticker list! Strategy cannot execute.")
        return []
    
    sector_performance = []
    found_etfs = []
    
    for etf in available_sector_etfs:
        if etf in all_tickers and etf in ticker_data_grouped:
            found_etfs.append(etf)
            try:
                etf_data = ticker_data_grouped[etf]
                
                # Convert current_date to pandas Timestamp with timezone
                current_date_tz = pd.Timestamp(current_date)
                if hasattr(etf_data.index, 'tz') and etf_data.index.tz is not None:
                    if current_date_tz.tz is None:
                        current_date_tz = current_date_tz.tz_localize(etf_data.index.tz)
                    else:
                        current_date_tz = current_date_tz.tz_convert(etf_data.index.tz)
                
                # Filter data for momentum calculation
                start_date = current_date_tz - timedelta(days=SECTOR_ROTATION_MOMENTUM_WINDOW + 30)
                
                # Use index-based filtering since date is the index
                etf_filtered = etf_data[(etf_data.index >= start_date) & (etf_data.index <= current_date_tz)]
                
                print(f"   🔍 {etf}: {len(etf_filtered)} data points available")
                
                if len(etf_filtered) >= 20:  # Need sufficient data
                    start_price = etf_filtered['Close'].iloc[0]
                    end_price = etf_filtered['Close'].iloc[-1]
                    momentum_pct = ((end_price - start_price) / start_price) * 100
                    
                    print(f"   🔍 {etf}: momentum = {momentum_pct:.1f}% (threshold: {SECTOR_ROTATION_MIN_MOMENTUM}%)")
                    
                    if momentum_pct >= SECTOR_ROTATION_MIN_MOMENTUM:
                        sector_performance.append((etf, momentum_pct))
                else:
                    print(f"   🔍 {etf}: insufficient data (need 20, have {len(etf_filtered)})")
                        
            except Exception as e:
                print(f"   🔍 {etf}: error calculating momentum - {e}")
                continue
    
    print(f"   🔍 Found {len(found_etfs)} sector ETFs: {found_etfs}")
    print(f"   🔍 {len(sector_performance)} ETFs met momentum threshold")
    
    # Sort by momentum and get top N
    if sector_performance:
        sector_performance.sort(key=lambda x: x[1], reverse=True)
        selected_etfs = [etf for etf, momentum in sector_performance[:top_n]]
        
        print(f"   🏢 Top {len(selected_etfs)} sector ETFs by {SECTOR_ROTATION_MOMENTUM_WINDOW}-day momentum:")
        for etf, momentum in sector_performance[:top_n]:
            print(f"      {etf}: {momentum:+.1f}%")
        
        print(f"   ✅ Sector Rotation selected {len(selected_etfs)} ETFs: {selected_etfs}")
        return selected_etfs
    else:
        print(f"   ❌ No sector ETFs met minimum momentum threshold ({SECTOR_ROTATION_MIN_MOMENTUM}%)")
        print(f"   ❌ Total ETFs analyzed: {len(sector_performance)}")
        return []


def select_3m_1y_ratio_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                              current_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    3M/1Y Ratio Strategy: Select tickers with highest ratio of 3-month performance to 1-year performance.
    
    This strategy identifies stocks that are showing strong recent momentum (3M) relative to 
    their longer-term performance (1Y), which can indicate accelerating momentum or 
    reversal from longer-term trends.
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data (with date as index)
        current_date: Current date for analysis (None for last available)
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    ratio_candidates = []
    
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers 
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []
    
    print(f"   📊 3M/1Y Ratio Strategy analyzing {len(all_tickers)} tickers")
    
    analysis_count = 0
    filtered_3m_negative = 0
    filtered_1y_negative = 0
    filtered_ratio_negative = 0
    filtered_ratio_too_high = 0
    data_insufficient = 0
    
    for ticker in all_tickers:
        try:
            analysis_count += 1
            
            if ticker not in ticker_data_grouped:
                data_insufficient += 1
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            # Need at least 1 year of data
            if len(ticker_data) < 250:
                data_insufficient += 1
                continue
            
            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)
            
            # Calculate 3-month performance
            three_month_start = current_date_tz - timedelta(days=90)
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) & 
                                         (ticker_data.index <= current_date_tz)]
            
            if len(three_month_data) < 10:  # Need at least 10 data points
                data_insufficient += 1
                continue
            
            three_month_valid = three_month_data['Close'].dropna()
            if len(three_month_valid) < 2:
                data_insufficient += 1
                continue
            
            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]
            
            if three_month_start_price <= 0 or pd.isna(three_month_start_price) or pd.isna(three_month_end_price):
                data_insufficient += 1
                continue
            
            three_month_performance = ((three_month_end_price - three_month_start_price) / 
                                     three_month_start_price) * 100
            
            # Calculate 1-year performance
            one_year_start = current_date_tz - timedelta(days=365)
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) & 
                                      (ticker_data.index <= current_date_tz)]
            
            if len(one_year_data) < 50:  # Need at least 50 data points
                data_insufficient += 1
                continue
            
            one_year_valid = one_year_data['Close'].dropna()
            if len(one_year_valid) < 2:
                data_insufficient += 1
                continue
            
            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]
            
            if one_year_start_price <= 0 or pd.isna(one_year_start_price) or pd.isna(one_year_end_price):
                data_insufficient += 1
                continue
            
            one_year_performance = ((one_year_end_price - one_year_start_price) / 
                                  one_year_start_price) * 100
            
            # Calculate ratio (handle division by zero and negative 1Y performance)
            if abs(one_year_performance) < 0.1:  # Avoid division by very small numbers
                continue
            
            ratio = three_month_performance / one_year_performance
            
            # Better approach: Annualized 3M performance vs 1Y performance
            # This compares like-for-like annualized rates
            annualized_3m = three_month_performance * (365/90)  # Annualize 3M performance
            momentum_acceleration = annualized_3m - one_year_performance
            
            # Debug first few stocks
            if analysis_count <= 5:
                print(f"   🔍 DEBUG {ticker}: 3M={three_month_performance:+.1f}%, annualized_3M={annualized_3m:+.1f}%, 1Y={one_year_performance:+.1f}%, acceleration={momentum_acceleration:+.1f}%")
            
            # Track filtering reasons
            if three_month_performance <= 0:
                filtered_3m_negative += 1
                continue
            if one_year_performance <= 0:
                filtered_1y_negative += 1
                continue
            
            # Better approach: Strong base + annualized acceleration
            # Require minimum 1Y performance AND positive annualized acceleration - RELAXED
            if (one_year_performance > 5 and  # Reduced: Minimum 5% 1Y performance vs 10%
                momentum_acceleration > 5):  # Reduced: At least 5% annualized acceleration vs 10%
                ratio_candidates.append((ticker, momentum_acceleration, annualized_3m, one_year_performance))
            else:
                # Track why filtered
                if momentum_acceleration <= 5 or one_year_performance <= 5:
                    filtered_ratio_negative += 1  # Reuse this counter for low acceleration/weak 1Y
        
        except Exception as e:
            data_insufficient += 1
            continue
    
    print(f"   📊 Analysis Summary:")
    print(f"      Total analyzed: {analysis_count}")
    print(f"      Data insufficient: {data_insufficient}")
    print(f"      3M negative: {filtered_3m_negative}")
    print(f"      1Y negative: {filtered_1y_negative}")
    print(f"      Weak 1Y/Low acceleration: {filtered_ratio_negative}")
    print(f"      Valid candidates: {len(ratio_candidates)}")
    
    # Sort by momentum acceleration (highest first) and get top N
    if ratio_candidates:
        ratio_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [ticker for ticker, acceleration, annualized_3m, y1_perf in ratio_candidates[:top_n]]
        
        print(f"   📊 Top {top_n} Annualized Acceleration candidates:")
        for ticker, acceleration, annualized_3m, y1_perf in ratio_candidates[:top_n]:
            print(f"      {ticker}: acceleration={acceleration:+.1f}%, annualized_3M={annualized_3m:+.1f}%, 1Y={y1_perf:+.1f}%")
        
        print(f"   ✅ Annualized Acceleration selected {len(selected_tickers)} tickers: {selected_tickers}")
        return selected_tickers
    else:
        print(f"   ❌ No Annualized Acceleration candidates found")
        print(f"   ❌ Analyzed {len(all_tickers)} tickers, found {len(ratio_candidates)} valid candidates")
        return []


def select_multitask_learning_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                                     current_date: datetime = None, train_start_date: datetime = None,
                                     train_end_date: datetime = None, top_n: int = 20) -> List[str]:
    """
    Multi-Task Learning stock selection strategy wrapper.
    
    This strategy uses unified models that learn from all tickers simultaneously,
    enabling knowledge sharing and better generalization.
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data
        current_date: Current date for analysis
        train_start_date: Start date for training
        train_end_date: End date for training
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    
    if not MULTITASK_AVAILABLE:
        print("   ⚠️ Multi-Task Learning not available, using fallback")
        return []
    
    if train_start_date is None or train_end_date is None:
        print("   ⚠️ Multi-Task Learning requires training dates")
        return []
    
    try:
        return select_multitask_stocks(
            all_tickers=all_tickers,
            ticker_data_grouped=ticker_data_grouped,
            current_date=current_date,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            top_n=top_n
        )
    except Exception as e:
        print(f"   ❌ Multi-Task Learning strategy error: {e}")
        return []


def select_turnaround_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Turnaround Strategy: Select stocks with low 3Y performance but high 1Y performance.
    This identifies stocks that may be emerging from a long decline with strong recent momentum.
    """
    turnaround_candidates = []
    data_insufficient = 0
    filtered_3y_positive = 0
    filtered_1y_low = 0
    
    print(f"   🔍 Turnaround: Analyzing {len(all_tickers)} tickers")
    
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is a datetime object
    if isinstance(current_date, str):
        try:
            current_date = pd.to_datetime(current_date)
        except Exception:
            pass

    # Ensure current_date is timezone-aware for comparison
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    elif not hasattr(current_date, 'tzinfo'):
        # Fallback if it's not a datetime-like object
        return []
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)
            
            # Calculate 1-year and 3-year start dates
            one_year_start = current_date_tz - timedelta(days=365)
            three_year_start = current_date_tz - timedelta(days=1095)  # 3 years
            
            # Get 1-year data
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) & (ticker_data.index <= current_date_tz)]
            one_year_valid = one_year_data['Close'].dropna()
            
            # Get 3-year data
            three_year_data = ticker_data[(ticker_data.index >= three_year_start) & (ticker_data.index <= current_date_tz)]
            three_year_valid = three_year_data['Close'].dropna()
            
            # Check data sufficiency (reduced requirements for available data)
            if len(one_year_valid) < 100 or len(three_year_valid) < 100:  # Reduced requirements
                data_insufficient += 1
                continue
            
            # Calculate performances
            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]
            three_year_start_price = three_year_valid.iloc[0]
            three_year_end_price = three_year_valid.iloc[-1]
            
            if any(price <= 0 or pd.isna(price) for price in [one_year_start_price, one_year_end_price, three_year_start_price, three_year_end_price]):
                data_insufficient += 1
                continue
            
            one_year_performance = ((one_year_end_price - one_year_start_price) / one_year_start_price) * 100
            three_year_performance = ((three_year_end_price - three_year_start_price) / three_year_start_price) * 100
            
            # Debug first few stocks
            if len(turnaround_candidates) < 5:
                print(f"   🔍 DEBUG {ticker}: 3Y={three_year_performance:+.1f}%, 1Y={one_year_performance:+.1f}%")
            
            # Turnaround criteria (relaxed for more candidates):
            # 1. Poor 3Y performance (negative or low - below 30%)
            # 2. Strong 1Y performance (positive - minimum 10%)
            # 3. Recovery ratio: 1Y performance should be better than 3Y annualized
            if three_year_performance > 30:  # 3Y should be below 30% (relaxed from 0%)
                filtered_3y_positive += 1
                continue
            
            if one_year_performance < 10:  # Need positive 1Y performance (relaxed from 20%)
                filtered_1y_low += 1
                continue
            
            # Calculate recovery score: 1Y performance - (3Y performance / 3)
            # This compares 1Y performance to average annual performance over 3 years
            three_year_annual = three_year_performance / 3
            recovery_score = one_year_performance - three_year_annual
            
            # Add to candidates if recovery is positive (relaxed from 30%)
            if recovery_score > 10:  # 1Y should be at least 10% better than 3Y annual average
                turnaround_candidates.append((ticker, recovery_score, one_year_performance, three_year_performance))
        
        except Exception as e:
            data_insufficient += 1
            continue
    
    # Sort by recovery score (highest first)
    turnaround_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if turnaround_candidates:
        print(f"   📊 Turnaround: Selected {len(turnaround_candidates)} candidates")
        print(f"   📊 Filter breakdown: {filtered_3y_positive} filtered (3Y positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        print(f"   🎯 Selected {min(len(turnaround_candidates), top_n)} turnaround stocks:")
        for ticker, score, one_year, three_year in turnaround_candidates[:top_n]:
            print(f"      {ticker}: recovery={score:.1f}%, 1Y={one_year:+.1f}%, 3Y={three_year:+.1f}%")
        
        return [ticker for ticker, _, _, _ in turnaround_candidates[:top_n]]
    else:
        print(f"   ❌ No turnaround candidates found")
        print(f"   📊 Filter breakdown: {filtered_3y_positive} filtered (3Y positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        return []


def select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    1Y/3M Ratio Strategy: Select stocks with strong 1Y performance but weak 3M performance.
    This identifies stocks in long-term uptrends that recently pulled back (buy on dip).
    """
    ratio_candidates = []
    data_insufficient = 0
    filtered_3m_positive = 0
    filtered_1y_low = 0
    
    print(f"   🔍 1Y/3M Ratio: Analyzing {len(all_tickers)} tickers")
    
    # Use current date or last available date
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
    
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    
    for ticker in all_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            
            # Convert current_date to pandas Timestamp with timezone
            current_date_tz = pd.Timestamp(current_date)
            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(ticker_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(ticker_data.index.tz)
            
            # Calculate 3-month and 1-year start dates
            three_month_start = current_date_tz - timedelta(days=90)
            one_year_start = current_date_tz - timedelta(days=365)
            
            # Get 3-month data
            three_month_data = ticker_data[(ticker_data.index >= three_month_start) & 
                                         (ticker_data.index <= current_date_tz)]
            
            # Get 1-year data
            one_year_data = ticker_data[(ticker_data.index >= one_year_start) & 
                                      (ticker_data.index <= current_date_tz)]
            
            # Check data sufficiency
            if len(three_month_data) < 10 or len(one_year_data) < 200:
                data_insufficient += 1
                continue
            
            three_month_valid = three_month_data['Close'].dropna()
            one_year_valid = one_year_data['Close'].dropna()
            
            if len(three_month_valid) < 2 or len(one_year_valid) < 2:
                data_insufficient += 1
                continue
            
            # Calculate performances
            three_month_start_price = three_month_valid.iloc[0]
            three_month_end_price = three_month_valid.iloc[-1]
            one_year_start_price = one_year_valid.iloc[0]
            one_year_end_price = one_year_valid.iloc[-1]
            
            if any(price <= 0 or pd.isna(price) for price in [three_month_start_price, three_month_end_price, one_year_start_price, one_year_end_price]):
                data_insufficient += 1
                continue
            
            three_month_performance = ((three_month_end_price - three_month_start_price) / 
                                      three_month_start_price) * 100
            one_year_performance = ((one_year_end_price - one_year_start_price) / 
                                  one_year_start_price) * 100
            
            # Debug first few stocks
            if len(ratio_candidates) < 5:
                print(f"   🔍 DEBUG {ticker}: 3M={three_month_performance:+.1f}%, 1Y={one_year_performance:+.1f}%")
            
            # Buy on dip criteria - RELAXED:
            # 1. Strong 1Y performance (positive and significant)
            # 2. Weak or negative 3M performance (pullback)
            # 3. Dip ratio: 1Y performance should be much better than 3M
            if three_month_performance > 30:  # Relaxed: 3M should be weak or negative (30% vs 20%)
                filtered_3m_positive += 1
                continue
            
            if one_year_performance < 10:  # Relaxed: Need strong 1Y performance (10% vs 15%)
                filtered_1y_low += 1
                continue
            
            # Calculate dip opportunity score: 1Y performance - 3M performance
            # Higher score means stronger 1Y trend with bigger recent pullback
            dip_score = one_year_performance - three_month_performance
            
            # Add to candidates if dip opportunity is strong - RELAXED
            if dip_score > 15:  # Relaxed: 1Y should be at least 15% better than 3M (15% vs 20%)
                ratio_candidates.append((ticker, dip_score, one_year_performance, three_month_performance))
        
        except Exception as e:
            data_insufficient += 1
            continue
    
    # Sort by dip score (highest first)
    ratio_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if ratio_candidates:
        print(f"   📊 1Y/3M Ratio: Selected {len(ratio_candidates)} dip candidates")
        print(f"   📊 Filter breakdown: {filtered_3m_positive} filtered (3M positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        print(f"   🎯 Selected {min(len(ratio_candidates), top_n)} buy-on-dip stocks:")
        for ticker, score, one_year, three_month in ratio_candidates[:top_n]:
            print(f"      {ticker}: dip={score:.1f}%, 1Y={one_year:+.1f}%, 3M={three_month:+.1f}%")
        
        return [ticker for ticker, _, _, _ in ratio_candidates[:top_n]]
    else:
        print(f"   ❌ No buy-on-dip candidates found")
        print(f"   📊 Filter breakdown: {filtered_3m_positive} filtered (3M positive), {filtered_1y_low} filtered (1Y low), {data_insufficient} insufficient data")
        return []


def select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Hybrid Momentum-Volatility Strategy: Combines strong momentum with controlled volatility.
    """
    if current_date is None:
        current_date = datetime.now()
    
    candidates = []
    
    for ticker in all_tickers:
        try:
            # Get ticker data - handle both dict and GroupBy objects
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None
                
            if ticker_data is None or len(ticker_data) == 0:
                continue
            
            # Use simple iloc-based approach like other strategies
            if len(ticker_data) < 60:
                continue
            
            # Get latest price (last row)
            latest_price = ticker_data['Close'].dropna().iloc[-1] if len(ticker_data['Close'].dropna()) > 0 else None
            if latest_price is None or latest_price <= 0:
                continue
            
            # Calculate 3M performance (approx 63 trading days)
            lookback_3m = min(63, len(ticker_data) - 1)
            if lookback_3m < 20:
                continue
            price_3m_ago = ticker_data['Close'].dropna().iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue
            performance_3m = (latest_price - price_3m_ago) / price_3m_ago
            annualized_3m = (1 + performance_3m) ** (252 / lookback_3m) - 1
            
            # Calculate 1Y performance (approx 252 trading days)
            lookback_1y = min(252, len(ticker_data) - 1)
            if lookback_1y < 60:
                continue
            price_1y_ago = ticker_data['Close'].dropna().iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago
            
            # Calculate volatility (using daily returns)
            daily_returns = ticker_data['Close'].pct_change().dropna()
            if len(daily_returns) < 30:
                continue
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
            
            # Calculate average volume
            avg_volume = ticker_data['Volume'].mean() if 'Volume' in ticker_data.columns else 100000
            
            # Apply filters - RELAXED criteria
            if (annualized_3m > 0.0 and  # Any positive 3M momentum
                performance_1y > -0.3 and  # Allow up to 30% loss in 1Y
                volatility < 3.0 and  # Volatility < 300%
                avg_volume > 10000):  # Low volume threshold
                
                # Calculate composite score
                momentum_score = annualized_3m * 0.6 + max(performance_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)
                
                candidates.append({
                    'ticker': ticker,
                    'score': composite_score,
                    'annualized_3m': annualized_3m,
                    'performance_1y': performance_1y,
                    'volatility': volatility
                })
                
        except Exception as e:
            continue
    
    # Sort by composite score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Debug output
    if candidates:
        print(f"   🎯 Momentum-Volatility Hybrid: Found {len(candidates)} candidates")
        if len(candidates) >= 3:
            print(f"   Top 3: {candidates[0]['ticker']} ({candidates[0]['score']:.3f}), "
                  f"{candidates[1]['ticker']} ({candidates[1]['score']:.3f}), "
                  f"{candidates[2]['ticker']} ({candidates[2]['score']:.3f})")
    else:
        print(f"   ⚠️ Momentum-Volatility Hybrid: No candidates found (checked {len(all_tickers)} tickers)")
    
    return [c['ticker'] for c in candidates[:top_n]]


def select_price_acceleration_stocks(all_tickers, ticker_data_grouped, current_date=None, top_n=20):
    """
    Price Acceleration Strategy: Uses velocity (price change) and acceleration (velocity change)
    to identify stocks with increasing momentum.
    
    Physics-inspired approach:
    - Velocity = price.pct_change() (daily return rate)
    - Acceleration = velocity.diff() (change in velocity)
    - Buy when acceleration is positive (momentum increasing)
    """
    if current_date is None:
        current_date = datetime.now()
    
    candidates = []
    data_insufficient = 0
    low_velocity = 0
    negative_acceleration = 0
    
    print(f"   🚀 Price Acceleration: Analyzing {len(all_tickers)} tickers")
    print(f"   📐 Formula: velocity = price.pct_change(), acceleration = velocity.diff()")
    
    for ticker in all_tickers:
        try:
            # Get ticker data
            if isinstance(ticker_data_grouped, dict):
                ticker_data = ticker_data_grouped.get(ticker)
            elif hasattr(ticker_data_grouped, 'get_group'):
                ticker_data = ticker_data_grouped.get_group(ticker) if ticker in ticker_data_grouped.groups else None
            else:
                ticker_data = None
                
            if ticker_data is None or len(ticker_data) < 30:
                data_insufficient += 1
                continue
            
            # Calculate velocity (daily returns)
            prices = ticker_data['Close'].dropna()
            if len(prices) < 30:
                data_insufficient += 1
                continue
            
            velocity = prices.pct_change().dropna()
            if len(velocity) < 20:
                data_insufficient += 1
                continue
            
            # Calculate acceleration (change in velocity)
            acceleration = velocity.diff().dropna()
            if len(acceleration) < 10:
                data_insufficient += 1
                continue
            
            # Get recent metrics (last 10 days)
            recent_velocity = velocity.tail(10).mean()
            recent_acceleration = acceleration.tail(5).mean()  # Last 5 days acceleration
            latest_acceleration = acceleration.iloc[-1]  # Most recent day
            
            # Calculate trend consistency (how many recent days have positive acceleration)
            recent_accel_series = acceleration.tail(5)
            positive_accel_days = (recent_accel_series > 0).sum()
            consistency_score = positive_accel_days / 5  # 0.0 to 1.0
            
            # Debug first few stocks
            if len(candidates) < 3:
                print(f"   🔍 DEBUG {ticker}: velocity={recent_velocity:.4f}, accel={recent_acceleration:.6f}, "
                      f"latest={latest_acceleration:.6f}, consistency={consistency_score:.1%}")
            
            # Selection criteria:
            # 1. Positive recent velocity (price going up)
            # 2. Positive acceleration (momentum increasing)
            # 3. Recent consistency (at least 3 of 5 days with positive acceleration)
            # 4. Strong latest acceleration signal
            
            if recent_velocity <= 0.001:  # At least 0.1% average daily gain
                low_velocity += 1
                continue
            
            if recent_acceleration <= 0:  # Must have positive acceleration
                negative_acceleration += 1
                continue
            
            # Calculate composite acceleration score
            # Weights: 40% avg acceleration, 40% latest acceleration, 20% consistency
            accel_score = (recent_acceleration * 0.4 + 
                          latest_acceleration * 0.4 + 
                          consistency_score * recent_acceleration * 0.2)
            
            # Scale by velocity (stronger acceleration matters more with higher velocity)
            final_score = accel_score * (1 + recent_velocity * 100)
            
            candidates.append({
                'ticker': ticker,
                'score': final_score,
                'velocity': recent_velocity,
                'acceleration': recent_acceleration,
                'latest_accel': latest_acceleration,
                'consistency': consistency_score
            })
            
        except Exception as e:
            data_insufficient += 1
            continue
    
    # Sort by acceleration score (highest first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"   📊 Analysis Summary:")
    print(f"      Total analyzed: {len(all_tickers)}")
    print(f"      Data insufficient: {data_insufficient}")
    print(f"      Low velocity: {low_velocity}")
    print(f"      Negative acceleration: {negative_acceleration}")
    print(f"      Valid candidates: {len(candidates)}")
    
    if candidates:
        print(f"   📊 Top {min(len(candidates), top_n)} Price Acceleration candidates:")
        for i, c in enumerate(candidates[:top_n], 1):
            print(f"      {i}. {c['ticker']}: score={c['score']:.4f}, "
                  f"velocity={c['velocity']:.4f}, accel={c['acceleration']:.6f}, "
                  f"consistency={c['consistency']:.0%}")
        
        selected = [c['ticker'] for c in candidates[:top_n]]
        print(f"   ✅ Price Acceleration selected {len(selected)} tickers: {selected}")
        return selected
    else:
        print(f"   ❌ No Price Acceleration candidates found")
        return []
