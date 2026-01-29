"""
Insider Trading Signal Strategy

Tracks insider buying/selling patterns as trading signals.
Features:
- Insider buying cluster detection
- Officer vs director distinction
- Transaction size weighting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from config import (
    TRANSACTION_COST,
    PORTFOLIO_SIZE,
)

# ============================================
# Configuration Parameters
# ============================================

# Insider signal parameters
INSIDER_LOOKBACK_DAYS = 90  # Look for insider activity in last 90 days
MIN_INSIDER_BUYS = 2  # Minimum insider buys to trigger signal
MIN_BUY_VALUE = 50000  # Minimum total buy value ($)

# Signal weighting
CEO_WEIGHT = 3.0  # CEO/CFO buys weighted higher
DIRECTOR_WEIGHT = 1.5
OTHER_WEIGHT = 1.0

# Cluster detection
CLUSTER_WINDOW_DAYS = 14  # Multiple buys within 14 days = cluster


class InsiderTradingSignal:
    """Insider Trading Signal Strategy Implementation."""
    
    def __init__(self):
        # Simulated insider data (in production, fetch from SEC EDGAR or data provider)
        self.insider_data = self._generate_mock_insider_data()
    
    def _generate_mock_insider_data(self) -> Dict[str, List[Dict]]:
        """
        Generate mock insider trading data.
        In production, this would fetch from SEC EDGAR API or Quiver Quant.
        """
        # Mock data structure: ticker -> list of transactions
        mock_data = {
            'AAPL': [
                {'date': '2025-12-15', 'type': 'buy', 'shares': 10000, 'price': 175, 'insider_type': 'director'},
                {'date': '2025-12-18', 'type': 'buy', 'shares': 5000, 'price': 178, 'insider_type': 'officer'},
            ],
            'MSFT': [
                {'date': '2025-12-10', 'type': 'buy', 'shares': 8000, 'price': 380, 'insider_type': 'ceo'},
            ],
            'GOOGL': [
                {'date': '2025-12-20', 'type': 'buy', 'shares': 3000, 'price': 140, 'insider_type': 'director'},
                {'date': '2025-12-22', 'type': 'buy', 'shares': 2000, 'price': 142, 'insider_type': 'director'},
                {'date': '2025-12-23', 'type': 'buy', 'shares': 4000, 'price': 141, 'insider_type': 'officer'},
            ],
            'NVDA': [
                {'date': '2025-12-05', 'type': 'buy', 'shares': 1000, 'price': 480, 'insider_type': 'cfo'},
                {'date': '2025-12-08', 'type': 'buy', 'shares': 500, 'price': 485, 'insider_type': 'director'},
            ],
            'META': [
                {'date': '2025-12-12', 'type': 'sell', 'shares': 50000, 'price': 350, 'insider_type': 'ceo'},
            ],
            'AMZN': [
                {'date': '2025-12-01', 'type': 'buy', 'shares': 2000, 'price': 185, 'insider_type': 'director'},
            ],
        }
        return mock_data
    
    def get_insider_weight(self, insider_type: str) -> float:
        """Get weight based on insider type."""
        insider_type = insider_type.lower()
        if insider_type in ['ceo', 'cfo', 'president']:
            return CEO_WEIGHT
        elif insider_type in ['director', 'board']:
            return DIRECTOR_WEIGHT
        else:
            return OTHER_WEIGHT
    
    def calculate_insider_score(self, ticker: str, current_date: datetime) -> Tuple[float, int, float]:
        """
        Calculate insider trading score for a ticker.
        Returns: (score, num_buys, total_value)
        """
        if ticker not in self.insider_data:
            return 0.0, 0, 0.0
        
        transactions = self.insider_data[ticker]
        lookback_start = current_date - timedelta(days=INSIDER_LOOKBACK_DAYS)
        
        buy_score = 0.0
        sell_score = 0.0
        num_buys = 0
        total_buy_value = 0.0
        
        for txn in transactions:
            txn_date = datetime.strptime(txn['date'], '%Y-%m-%d')
            if txn_date < lookback_start or txn_date > current_date:
                continue
            
            value = txn['shares'] * txn['price']
            weight = self.get_insider_weight(txn['insider_type'])
            
            if txn['type'] == 'buy':
                buy_score += weight * np.log1p(value / 10000)  # Log scale
                num_buys += 1
                total_buy_value += value
            else:
                sell_score += weight * np.log1p(value / 10000)
        
        # Net score (buys - sells)
        net_score = buy_score - (sell_score * 0.5)  # Sells weighted less
        
        return net_score, num_buys, total_buy_value
    
    def detect_buying_cluster(self, ticker: str, current_date: datetime) -> bool:
        """Detect if there's a cluster of insider buys."""
        if ticker not in self.insider_data:
            return False
        
        transactions = self.insider_data[ticker]
        cluster_start = current_date - timedelta(days=CLUSTER_WINDOW_DAYS)
        
        recent_buys = [
            txn for txn in transactions
            if txn['type'] == 'buy'
            and datetime.strptime(txn['date'], '%Y-%m-%d') >= cluster_start
            and datetime.strptime(txn['date'], '%Y-%m-%d') <= current_date
        ]
        
        return len(recent_buys) >= MIN_INSIDER_BUYS
    
    def estimate_insider_activity_from_price(self, ticker_data: pd.DataFrame,
                                             current_date: datetime) -> float:
        """
        Estimate insider activity from price patterns when no direct data available.
        Unusual accumulation patterns may indicate insider buying.
        """
        try:
            data = ticker_data[ticker_data.index <= current_date]
            if len(data) < 60:
                return 0.0
            
            recent = data.tail(30)
            
            # Look for accumulation pattern:
            # 1. Price relatively flat
            # 2. Volume increasing
            # 3. Positive close bias
            
            price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
            
            if 'Volume' in recent.columns:
                vol_trend = recent['Volume'].tail(10).mean() / recent['Volume'].head(10).mean()
            else:
                vol_trend = 1.0
            
            # Positive close bias (closing near high)
            if 'High' in recent.columns and 'Low' in recent.columns:
                close_position = (recent['Close'] - recent['Low']) / (recent['High'] - recent['Low'] + 0.01)
                close_bias = close_position.mean()
            else:
                close_bias = 0.5
            
            # Accumulation score
            if abs(price_change) < 0.05 and vol_trend > 1.2 and close_bias > 0.6:
                return 0.5  # Moderate accumulation signal
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def select_stocks(self, all_tickers: List[str],
                     ticker_data_grouped: Dict[str, pd.DataFrame],
                     current_date: datetime,
                     train_start_date: datetime = None,
                     top_n: int = PORTFOLIO_SIZE) -> List[str]:
        """Select stocks with positive insider trading signals."""
        print(f"\n   🎯 Insider Trading Signal Strategy")
        print(f"   📅 Date: {current_date.date()}")
        
        candidates = []
        
        for ticker in all_tickers:
            # Get insider score from data
            score, num_buys, total_value = self.calculate_insider_score(ticker, current_date)
            
            # Check for buying cluster
            has_cluster = self.detect_buying_cluster(ticker, current_date)
            
            # Estimate from price if no direct data
            if score == 0 and ticker in ticker_data_grouped:
                price_signal = self.estimate_insider_activity_from_price(
                    ticker_data_grouped[ticker], current_date
                )
                score = price_signal
            
            # Boost score for clusters
            if has_cluster:
                score *= 1.5
            
            if score > 0:
                candidates.append((ticker, score, num_buys, total_value, has_cluster))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected = [ticker for ticker, score, num_buys, total_value, has_cluster in candidates[:top_n]]
        
        print(f"   ✅ Found {len(candidates)} insider signal candidates")
        print(f"   ✅ Selected {len(selected)} stocks:")
        for ticker, score, num_buys, total_value, has_cluster in candidates[:top_n]:
            cluster_str = " [CLUSTER]" if has_cluster else ""
            if num_buys > 0:
                print(f"      {ticker}: Score={score:.2f}, Buys={num_buys}, Value=${total_value:,.0f}{cluster_str}")
            else:
                print(f"      {ticker}: Score={score:.2f} (price pattern)")
        
        return selected


# Global instance
_insider_instance = None

def get_insider_instance() -> InsiderTradingSignal:
    """Get or create the global insider trading instance."""
    global _insider_instance
    if _insider_instance is None:
        _insider_instance = InsiderTradingSignal()
    return _insider_instance


def select_insider_trading_stocks(all_tickers: List[str],
                                  ticker_data_grouped: Dict[str, pd.DataFrame],
                                  current_date: datetime = None,
                                  train_start_date: datetime = None,
                                  top_n: int = PORTFOLIO_SIZE) -> List[str]:
    """
    Insider Trading Signal stock selection strategy.
    
    Selects stocks with positive insider buying signals.
    """
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max()
                       for t in all_tickers
                       if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    instance = get_insider_instance()
    return instance.select_stocks(
        all_tickers, ticker_data_grouped, current_date, train_start_date, top_n
    )


def reset_insider_state():
    """Reset the global insider trading instance."""
    global _insider_instance
    _insider_instance = None
