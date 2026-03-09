"""
Analyst Recommendation Strategy
Selects stocks based on analyst upgrades, downgrades, and price targets.
Uses Yahoo Finance historical upgrade/downgrade data.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import PORTFOLIO_SIZE


# Scoring weights
UPGRADE_TO_BUY_POINTS = 3
UPGRADE_TO_HOLD_POINTS = 1
INITIATE_BUY_POINTS = 2
INITIATE_HOLD_POINTS = 0
MAINTAIN_BUY_POINTS = 1
DOWNGRADE_POINTS = -3
PRICE_TARGET_WEIGHT = 10  # Multiplier for price target upside


def fetch_analyst_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch analyst upgrade/downgrade history for a ticker.
    
    Returns DataFrame with columns:
    - Firm, ToGrade, FromGrade, Action, currentPriceTarget, priorPriceTarget
    - Index: GradeDate (datetime)
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.upgrades_downgrades
        if data is not None and not data.empty:
            return data
        return None
    except Exception as e:
        return None


def fetch_all_analyst_data(
    tickers: List[str],
    max_workers: int = 10,
    show_progress: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch analyst data for all tickers in parallel.
    
    Returns dict: ticker -> DataFrame of analyst actions
    """
    analyst_data = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_analyst_data, ticker): ticker for ticker in tickers}
        
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(tickers), desc="Fetching analyst data")
        
        for future in iterator:
            ticker = futures[future]
            try:
                data = future.result()
                if data is not None:
                    analyst_data[ticker] = data
            except Exception:
                pass
    
    return analyst_data


def calculate_analyst_score(
    analyst_df: pd.DataFrame,
    current_date: datetime,
    current_price: float,
    lookback_days: int = 60
) -> Tuple[float, int]:
    """
    Calculate analyst recommendation score for a stock.
    
    Args:
        analyst_df: DataFrame of analyst actions (from yfinance)
        current_date: Date to calculate score as of
        current_price: Current stock price
        lookback_days: Days to look back for analyst actions
        
    Returns:
        Tuple of (score, num_actions)
        
    Note: Yahoo Finance only provides recent analyst data (last few months).
    For backtesting, we use the most recent N days of available data regardless
    of the backtest date. This introduces forward-looking bias but allows
    testing the strategy logic. For live trading, dates will match correctly.
    """
    if analyst_df is None or analyst_df.empty:
        return 0.0, 0
    
    # Get the most recent analyst actions (last N days of available data)
    # This is necessary because Yahoo only provides current data, not historical
    if len(analyst_df) > 0:
        most_recent_date = analyst_df.index.max()
        cutoff_date = most_recent_date - timedelta(days=lookback_days)
        
        mask = analyst_df.index >= cutoff_date
        recent_actions = analyst_df[mask]
    else:
        recent_actions = analyst_df
    
    if recent_actions.empty:
        return 0.0, 0
    
    score = 0.0
    num_actions = len(recent_actions)
    
    for _, row in recent_actions.iterrows():
        action = str(row.get('Action', '')).lower()
        to_grade = str(row.get('ToGrade', '')).lower()
        from_grade = str(row.get('FromGrade', '')).lower()
        
        # Upgrade points
        if 'upgrade' in action:
            if 'buy' in to_grade or 'outperform' in to_grade or 'overweight' in to_grade:
                score += UPGRADE_TO_BUY_POINTS
            else:
                score += UPGRADE_TO_HOLD_POINTS
        
        # Initiate coverage
        elif 'init' in action:
            if 'buy' in to_grade or 'outperform' in to_grade or 'overweight' in to_grade:
                score += INITIATE_BUY_POINTS
            else:
                score += INITIATE_HOLD_POINTS
        
        # Maintain rating
        elif 'maintain' in action or 'reiterate' in action:
            if 'buy' in to_grade or 'outperform' in to_grade or 'overweight' in to_grade:
                score += MAINTAIN_BUY_POINTS
        
        # Downgrade
        elif 'downgrade' in action:
            score += DOWNGRADE_POINTS
        
        # Price target bonus
        try:
            price_target = float(row.get('currentPriceTarget', 0))
            if price_target > 0 and current_price > 0:
                upside = (price_target / current_price - 1)
                if upside > 0:
                    score += min(upside * PRICE_TARGET_WEIGHT, 5)  # Cap at 5 points
        except (ValueError, TypeError):
            pass
    
    return score, num_actions


def select_analyst_recommendation_stocks(
    tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    analyst_data: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = None,
    lookback_days: int = 60,
    min_actions: int = 1
) -> List[str]:
    """
    Select top stocks based on analyst recommendation scores.
    
    Args:
        tickers: List of ticker symbols to consider
        ticker_data_grouped: Dict of ticker -> price DataFrame
        analyst_data: Dict of ticker -> analyst actions DataFrame
        current_date: Current backtest date
        top_n: Number of stocks to select (default: PORTFOLIO_SIZE)
        lookback_days: Days to look back for analyst actions
        min_actions: Minimum analyst actions required
        
    Returns:
        List of selected ticker symbols
    """
    if top_n is None:
        top_n = PORTFOLIO_SIZE
    
    scores = []
    
    for ticker in tickers:
        if ticker not in analyst_data:
            continue
        
        # Get current price
        if ticker not in ticker_data_grouped:
            continue
        
        price_df = ticker_data_grouped[ticker]
        try:
            # Filter to data before current_date
            if hasattr(price_df.index, 'tz') and price_df.index.tz is not None:
                current_date_tz = pd.Timestamp(current_date).tz_localize(price_df.index.tz)
                valid_prices = price_df[price_df.index <= current_date_tz]
            else:
                valid_prices = price_df[price_df.index <= current_date]
            
            if valid_prices.empty:
                continue
            
            current_price = valid_prices['Close'].iloc[-1]
            if pd.isna(current_price) or current_price <= 0:
                continue
        except Exception:
            continue
        
        # Calculate score
        score, num_actions = calculate_analyst_score(
            analyst_data[ticker],
            current_date,
            current_price,
            lookback_days
        )
        
        if num_actions >= min_actions:
            scores.append((ticker, score, num_actions))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    selected = [ticker for ticker, score, _ in scores[:top_n]]
    
    if selected:
        print(f"   [Analyst] Top {len(selected)} by score")
        for ticker, score, num_actions in scores[:min(5, len(scores))]:
            print(f"      {ticker}: score={score:.1f} ({num_actions} actions)")
    
    return selected


def get_analyst_consensus(
    ticker: str,
    analyst_df: pd.DataFrame,
    current_date: datetime,
    lookback_days: int = 90
) -> Dict[str, int]:
    """
    Get current analyst consensus (count of each rating type).
    
    Returns dict with keys: buy, hold, sell, total
    """
    if analyst_df is None or analyst_df.empty:
        return {'buy': 0, 'hold': 0, 'sell': 0, 'total': 0}
    
    cutoff_date = current_date - timedelta(days=lookback_days)
    
    # Handle timezone
    if analyst_df.index.tz is not None:
        current_date = pd.Timestamp(current_date).tz_localize(analyst_df.index.tz)
        cutoff_date = pd.Timestamp(cutoff_date).tz_localize(analyst_df.index.tz)
    
    mask = (analyst_df.index <= current_date) & (analyst_df.index >= cutoff_date)
    recent = analyst_df[mask]
    
    buy_count = 0
    hold_count = 0
    sell_count = 0
    
    # Get most recent action per firm
    firms_seen = set()
    for idx in recent.index[::-1]:  # Most recent first
        row = recent.loc[idx]
        firm = row.get('Firm', '')
        if firm in firms_seen:
            continue
        firms_seen.add(firm)
        
        to_grade = str(row.get('ToGrade', '')).lower()
        
        if any(x in to_grade for x in ['buy', 'outperform', 'overweight', 'positive']):
            buy_count += 1
        elif any(x in to_grade for x in ['sell', 'underperform', 'underweight', 'negative']):
            sell_count += 1
        else:
            hold_count += 1
    
    return {
        'buy': buy_count,
        'hold': hold_count,
        'sell': sell_count,
        'total': buy_count + hold_count + sell_count
    }
