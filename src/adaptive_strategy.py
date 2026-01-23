"""
Adaptive Strategy System
Rotates between strategies based on market conditions and performance
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from shared_strategies import (
    get_3m_1y_ratio_tickers,
    get_volatility_adj_mom_tickers,
    get_quality_mom_tickers,
    get_dynamic_bh_3m_tickers,
    get_risk_adj_mom_tickers
)
from config import TOP_N_STOCKS

class AdaptiveStrategy:
    """Adaptive strategy that rotates between different strategies based on market conditions"""
    
    def __init__(self):
        self.current_strategy = "3m_1y_ratio"
        self.strategy_performance = {}  # Track performance of each strategy
        self.last_rotation_day = 0
        self.rotation_frequency = 63  # Quarterly (63 trading days)
        
        # Strategy mapping by quarter (will be updated based on backtest results)
        self.quarterly_strategies = {
            1: "3m_1y_ratio",        # Q1: Momentum acceleration
            2: "volatility_adj_mom",  # Q2: Summer volatility
            3: "quality_mom",         # Q3: Earnings season
            4: "dynamic_bh_3m"        # Q4: Year-end rally
        }
        
        # Market condition strategies
        self.market_condition_strategies = {
            "bull_market": "3m_1y_ratio",
            "bear_market": "volatility_adj_mom", 
            "volatile": "risk_adj_mom",
            "sideways": "quality_mom"
        }
    
    def get_current_quarter(self, current_date: datetime) -> int:
        """Get the quarter number (1-4) from a date"""
        return (current_date.month - 1) // 3 + 1
    
    def get_market_condition(self, market_data: pd.DataFrame) -> str:
        """Determine market condition based on recent performance and volatility"""
        if len(market_data) < 30:
            return "bull_market"  # Default assumption
        
        # Calculate 30-day return and volatility
        recent_data = market_data.tail(30)
        returns = recent_data['Close'].pct_change().dropna()
        
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Determine market condition
        if avg_return > 0.005:  # >0.5% daily average = strong bull
            return "bull_market"
        elif avg_return < -0.003:  # <-0.3% daily average = bear
            return "bear_market"
        elif volatility > 0.03:  # >3% daily volatility = volatile
            return "volatile"
        else:
            return "sideways"
    
    def should_rotate_strategy(self, day_count: int, current_date: datetime) -> bool:
        """Check if it's time to rotate strategies"""
        # Rotate quarterly
        if day_count - self.last_rotation_day >= self.rotation_frequency:
            return True
        
        # Also rotate if it's a new quarter
        current_quarter = self.get_current_quarter(current_date)
        # This would need to be tracked across days
        
        return False
    
    def get_strategy_for_quarter(self, quarter: int) -> str:
        """Get the recommended strategy for a specific quarter"""
        return self.quarterly_strategies.get(quarter, "3m_1y_ratio")
    
    def get_strategy_for_market_condition(self, market_condition: str) -> str:
        """Get the recommended strategy for market conditions"""
        return self.market_condition_strategies.get(market_condition, "3m_1y_ratio")
    
    def get_adaptive_tickers(self, all_tickers: List[str], current_date: datetime, 
                           market_data: Optional[pd.DataFrame] = None, 
                           day_count: int = 0) -> List[str]:
        """Get tickers using the adaptive strategy"""
        
        # Check if we should rotate strategies
        if self.should_rotate_strategy(day_count, current_date):
            quarter = self.get_current_quarter(current_date)
            self.current_strategy = self.get_strategy_for_quarter(quarter)
            self.last_rotation_day = day_count
            print(f"🔄 Adaptive Strategy: Rotating to {self.current_strategy} for Q{quarter}")
        
        # Get tickers based on current strategy
        strategy_functions = {
            "3m_1y_ratio": get_3m_1y_ratio_tickers,
            "volatility_adj_mom": get_volatility_adj_mom_tickers,
            "quality_mom": get_quality_mom_tickers,
            "dynamic_bh_3m": get_dynamic_bh_3m_tickers,
            "risk_adj_mom": get_risk_adj_mom_tickers
        }
        
        if self.current_strategy in strategy_functions:
            return strategy_functions[self.current_strategy](all_tickers)
        else:
            # Fallback to 3m_1y_ratio
            return get_3m_1y_ratio_tickers(all_tickers)
    
    def update_strategy_performance(self, strategy: str, performance: float):
        """Update performance tracking for a strategy"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(performance)
    
    def get_best_strategy_recent(self, lookback_days: int = 30) -> str:
        """Get the best performing strategy in recent days"""
        best_strategy = "3m_1y_ratio"
        best_performance = -999
        
        for strategy, performances in self.strategy_performance.items():
            if len(performances) >= lookback_days:
                recent_avg = sum(performances[-lookback_days:]) / lookback_days
                if recent_avg > best_performance:
                    best_performance = recent_avg
                    best_strategy = strategy
        
        return best_strategy

# Global adaptive strategy instance
adaptive_strategy = AdaptiveStrategy()

def get_adaptive_strategy_tickers(all_tickers: List[str], current_date: datetime,
                                market_data: Optional[pd.DataFrame] = None,
                                day_count: int = 0) -> List[str]:
    """Main function to get adaptive strategy tickers"""
    return adaptive_strategy.get_adaptive_tickers(all_tickers, current_date, market_data, day_count)
