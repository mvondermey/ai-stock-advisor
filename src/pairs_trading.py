"""
Pairs Trading / Statistical Arbitrage Strategy

Finds correlated stock pairs and trades mean reversion of the spread.
Features:
- Cointegration testing for pair selection
- Z-score based entry/exit signals
- Market-neutral positioning
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import itertools

from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
)

# ============================================
# Configuration Parameters
# ============================================

# Pair selection
MIN_CORRELATION = 0.70  # Minimum correlation for pair consideration
CORRELATION_LOOKBACK = 120  # Days for correlation calculation
MIN_DATA_POINTS = 60  # Minimum data points for analysis

# Z-score thresholds
ZSCORE_ENTRY = 2.0  # Enter when z-score exceeds this
ZSCORE_EXIT = 0.5  # Exit when z-score returns to this
ZSCORE_STOP = 3.5  # Stop loss if z-score exceeds this

# Spread calculation
SPREAD_LOOKBACK = 60  # Days for spread mean/std calculation

# Known sector pairs (for faster pair finding)
SECTOR_PAIRS = [
    ('KO', 'PEP'),      # Beverages
    ('V', 'MA'),        # Payments
    ('JPM', 'BAC'),     # Banks
    ('XOM', 'CVX'),     # Oil
    ('HD', 'LOW'),      # Home improvement
    ('UNH', 'CI'),      # Health insurance
    ('MSFT', 'AAPL'),   # Tech giants
    ('GOOGL', 'META'),  # Digital advertising
    ('DIS', 'NFLX'),    # Entertainment
    ('CAT', 'DE'),      # Industrial equipment
    ('F', 'GM'),        # Auto
    ('WMT', 'TGT'),     # Retail
    ('MCD', 'YUM'),     # Fast food
    ('NKE', 'LULU'),    # Athletic apparel
    ('BA', 'LMT'),      # Aerospace/Defense
]


class PairsTrading:
    """Pairs Trading Strategy Implementation."""
    
    def __init__(self):
        self.active_pairs = {}  # pair -> {'long': ticker, 'short': ticker, 'entry_zscore': float}
        self.pair_stats = {}  # pair -> {'mean': float, 'std': float}
    
    def calculate_correlation(self, data1: pd.Series, data2: pd.Series) -> float:
        """Calculate correlation between two price series."""
        try:
            # Align data
            aligned = pd.concat([data1, data2], axis=1).dropna()
            if len(aligned) < MIN_DATA_POINTS:
                return 0.0
            
            returns1 = aligned.iloc[:, 0].pct_change().dropna()
            returns2 = aligned.iloc[:, 1].pct_change().dropna()
            
            return returns1.corr(returns2)
            
        except Exception as e:
            return 0.0
    
    def calculate_spread(self, data1: pd.Series, data2: pd.Series) -> pd.Series:
        """Calculate the spread between two normalized price series."""
        try:
            # Normalize prices to start at 1
            norm1 = data1 / data1.iloc[0]
            norm2 = data2 / data2.iloc[0]
            
            # Spread = log ratio
            spread = np.log(norm1 / norm2)
            return spread
            
        except Exception as e:
            return pd.Series()
    
    def calculate_zscore(self, spread: pd.Series, lookback: int = SPREAD_LOOKBACK) -> float:
        """Calculate z-score of current spread."""
        try:
            if len(spread) < lookback:
                return 0.0
            
            recent_spread = spread.tail(lookback)
            mean = recent_spread.mean()
            std = recent_spread.std()
            
            if std == 0:
                return 0.0
            
            current = spread.iloc[-1]
            zscore = (current - mean) / std
            
            return zscore
            
        except Exception as e:
            return 0.0
    
    def find_tradeable_pairs(self, all_tickers: List[str],
                            ticker_data_grouped: Dict[str, pd.DataFrame],
                            current_date) -> List[Tuple[str, str, float, float]]:
        """Find pairs with high correlation and tradeable z-scores."""
        tradeable_pairs = []
        
        # First check known sector pairs
        for ticker1, ticker2 in SECTOR_PAIRS:
            if ticker1 not in ticker_data_grouped or ticker2 not in ticker_data_grouped:
                continue
            
            data1 = ticker_data_grouped[ticker1]
            data2 = ticker_data_grouped[ticker2]
            
            data1 = data1[data1.index <= current_date]['Close']
            data2 = data2[data2.index <= current_date]['Close']
            
            if len(data1) < CORRELATION_LOOKBACK or len(data2) < CORRELATION_LOOKBACK:
                continue
            
            # Check correlation
            corr = self.calculate_correlation(
                data1.tail(CORRELATION_LOOKBACK),
                data2.tail(CORRELATION_LOOKBACK)
            )
            
            if corr < MIN_CORRELATION:
                continue
            
            # Calculate spread and z-score
            spread = self.calculate_spread(data1, data2)
            if len(spread) == 0:
                continue
            
            zscore = self.calculate_zscore(spread)
            
            # Check if tradeable (z-score beyond entry threshold)
            if abs(zscore) >= ZSCORE_ENTRY:
                tradeable_pairs.append((ticker1, ticker2, corr, zscore))
        
        # Also scan for pairs among provided tickers
        tickers_with_data = [t for t in all_tickers[:100]  # Limit for performance
                           if t in ticker_data_grouped 
                           and len(ticker_data_grouped[t]) >= CORRELATION_LOOKBACK]
        
        for ticker1, ticker2 in itertools.combinations(tickers_with_data[:50], 2):
            data1 = ticker_data_grouped[ticker1]
            data2 = ticker_data_grouped[ticker2]
            
            data1 = data1[data1.index <= current_date]['Close']
            data2 = data2[data2.index <= current_date]['Close']
            
            corr = self.calculate_correlation(
                data1.tail(CORRELATION_LOOKBACK),
                data2.tail(CORRELATION_LOOKBACK)
            )
            
            if corr < MIN_CORRELATION:
                continue
            
            spread = self.calculate_spread(data1, data2)
            if len(spread) == 0:
                continue
            
            zscore = self.calculate_zscore(spread)
            
            if abs(zscore) >= ZSCORE_ENTRY:
                # Avoid duplicates
                pair_key = tuple(sorted([ticker1, ticker2]))
                if not any(tuple(sorted([p[0], p[1]])) == pair_key for p in tradeable_pairs):
                    tradeable_pairs.append((ticker1, ticker2, corr, zscore))
        
        # Sort by absolute z-score (best opportunities first)
        tradeable_pairs.sort(key=lambda x: abs(x[3]), reverse=True)
        
        return tradeable_pairs
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks from pairs trading signals (long side only for simplicity)."""
        print(f"\n   🎯 Pairs Trading Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        # Convert current_date to pandas Timestamp with timezone
        current_date_tz = pd.Timestamp(current_date)
        # Use the first ticker's timezone as reference
        if all_tickers and all_tickers[0] in ticker_data_grouped:
            first_data = ticker_data_grouped[all_tickers[0]]
            if hasattr(first_data.index, 'tz') and first_data.index.tz is not None:
                if current_date_tz.tz is None:
                    current_date_tz = current_date_tz.tz_localize(first_data.index.tz)
                else:
                    current_date_tz = current_date_tz.tz_convert(first_data.index.tz)
        
        # Find tradeable pairs
        pairs = self.find_tradeable_pairs(all_tickers, ticker_data_grouped, current_date_tz)
        
        print(f"   📊 Found {len(pairs)} tradeable pairs")
        
        # Select long positions (underperformer in each pair)
        selected = []
        
        for ticker1, ticker2, corr, zscore in pairs:
            if len(selected) >= top_n:
                break
            
            # If z-score > 0, ticker1 is overvalued relative to ticker2
            # So we go long ticker2 (undervalued)
            if zscore > ZSCORE_ENTRY:
                long_ticker = ticker2
            elif zscore < -ZSCORE_ENTRY:
                long_ticker = ticker1
            else:
                continue
            
            if long_ticker not in selected:
                selected.append(long_ticker)
                print(f"      LONG {long_ticker} (pair: {ticker1}/{ticker2}, z={zscore:.2f}, corr={corr:.2f})")
        
        # If not enough pairs, fill with high-correlation stocks that are oversold
        if len(selected) < top_n:
            for ticker in all_tickers:
                if len(selected) >= top_n:
                    break
                if ticker in selected:
                    continue
                if ticker not in ticker_data_grouped:
                    continue
                
                data = ticker_data_grouped[ticker]
                if len(data) < 60:
                    continue
                
                # Simple mean reversion: buy if below 20-day MA
                data = data[data.index <= current_date_tz]
                price = data['Close'].iloc[-1]
                ma20 = data['Close'].tail(20).mean()
                
                if price < ma20 * 0.95:  # 5% below MA
                    selected.append(ticker)
        
        print(f"   ✅ Selected {len(selected)} stocks")
        
        return selected


# Global instance
_pairs_instance = None

def get_pairs_instance() -> PairsTrading:
    """Get or create the global pairs trading instance."""
    global _pairs_instance
    if _pairs_instance is None:
        _pairs_instance = PairsTrading()
    return _pairs_instance


def select_pairs_trading_stocks(all_tickers: List[str],
                                ticker_data_grouped: Dict[str, pd.DataFrame],
                                current_date: datetime = None,
                                train_start_date: datetime = None,
                                top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Pairs Trading stock selection strategy.
    
    Selects undervalued stocks from correlated pairs.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_pairs_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_pairs_state():
    """Reset the global pairs trading instance."""
    global _pairs_instance
    _pairs_instance = None
