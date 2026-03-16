#!/usr/bin/env python3
"""
Quick Strategy Check - Run a single strategy for today's recommendations.
Usage: python quick_strategy_check.py [strategy_name] [top_n]

Available strategies:
  - risk_adj_mom_3m (default)
  - mom_vol_hybrid_1y3m
  - trend_atr
  - trend_breakout
  - dual_momentum
  - ai_elite
  - dynamic_bh_1y
  - static_bh_1y
  - price_acceleration
  - quality_mom
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from typing import Dict, List
import pandas as pd

from config import PORTFOLIO_SIZE
from data_utils import load_all_market_data


def get_strategy_recommendations(strategy_name: str, top_n: int = 10) -> List[str]:
    """Get stock recommendations for a single strategy."""
    
    # Load market data
    print(f"\n📊 Loading market data...")
    from ticker_selection import get_all_tickers
    all_tickers = get_all_tickers()
    print(f"   Found {len(all_tickers)} tickers")
    
    # Download data
    print(f"   Downloading price data (uses cache if available)...")
    all_tickers_data = load_all_market_data(all_tickers)
    
    if all_tickers_data.empty:
        print("❌ No data loaded!")
        return []
    
    # Convert to ticker_data_grouped format (same as backtesting.py)
    print(f"   🔧 Pre-grouping data by ticker...")
    ticker_data_grouped = {}
    grouped = all_tickers_data.groupby('ticker')
    
    for ticker in all_tickers_data['ticker'].unique():
        try:
            ticker_df = grouped.get_group(ticker).copy()
            if 'date' in ticker_df.columns:
                ticker_df = ticker_df.set_index('date')
            ticker_df = ticker_df.drop('ticker', axis=1, errors='ignore')
            ticker_data_grouped[ticker] = ticker_df
        except KeyError:
            pass
    
    print(f"   ✅ Got data for {len(ticker_data_grouped)} tickers")
    
    current_date = datetime.now(timezone.utc)
    
    # Run the selected strategy
    print(f"\n🎯 Running {strategy_name} strategy...")
    
    if strategy_name == 'risk_adj_mom_3m':
        from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
        return select_risk_adj_mom_3m_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)
    
    elif strategy_name == 'mom_vol_hybrid_1y3m':
        from momentum_volatility_hybrid_strategy import select_momentum_volatility_hybrid_stocks
        return select_momentum_volatility_hybrid_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date,
            lookback_days=365, secondary_lookback_days=90, top_n=top_n
        )
    
    elif strategy_name == 'trend_atr':
        from new_strategies import select_trend_following_atr_stocks
        return select_trend_following_atr_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n
        )
    
    elif strategy_name == 'dual_momentum':
        from dual_momentum_strategy import select_dual_momentum_stocks
        return select_dual_momentum_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)
    
    elif strategy_name == 'ai_elite':
        from shared_strategies import select_ai_elite_with_training
        stocks, _ = select_ai_elite_with_training(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n=top_n
        )
        return stocks
    
    elif strategy_name == 'dynamic_bh_1y':
        from shared_strategies import select_top_performers
        return select_top_performers(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date,
            lookback_days=365, top_n=top_n, apply_performance_filter=True
        )
    
    elif strategy_name == 'static_bh_1y':
        from shared_strategies import select_top_performers
        return select_top_performers(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date,
            lookback_days=365, top_n=top_n, apply_performance_filter=False
        )
    
    elif strategy_name == 'price_acceleration':
        from new_strategies import select_price_acceleration_stocks
        return select_price_acceleration_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)
    
    elif strategy_name == 'quality_mom':
        from quality_momentum_strategy import select_quality_momentum_stocks
        return select_quality_momentum_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)
    
    elif strategy_name == 'trend_breakout':
        from new_strategies import select_trend_breakout_stocks
        return select_trend_breakout_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n
        )
    
    else:
        print(f"❌ Unknown strategy: {strategy_name}")
        print("Available: risk_adj_mom_3m, mom_vol_hybrid_1y3m, trend_atr, dual_momentum, ai_elite, dynamic_bh_1y, static_bh_1y, price_acceleration, quality_mom, trend_breakout")
        return []


def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else 'risk_adj_mom_3m'
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else PORTFOLIO_SIZE
    
    print(f"\n{'='*60}")
    print(f"🚀 Quick Strategy Check: {strategy}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Top N: {top_n}")
    print(f"{'='*60}")
    
    recommendations = get_strategy_recommendations(strategy, top_n)
    
    if recommendations:
        print(f"\n✅ {strategy.upper()} RECOMMENDATIONS ({len(recommendations)} stocks):")
        print("-" * 40)
        for i, ticker in enumerate(recommendations, 1):
            print(f"   {i:2}. {ticker}")
        print("-" * 40)
    else:
        print(f"\n❌ No recommendations from {strategy}")
    
    return recommendations


if __name__ == "__main__":
    main()
