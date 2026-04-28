#!/usr/bin/env python3
"""
Quick Strategy Check - Run a single strategy for today's recommendations.
Usage: python quick_strategy_check.py [strategy_name] [top_n]

Available strategies:
  - risk_adj_mom_3m (default)
  - risk_adj_mom_1m / 1m_volsweet
  - risk_adj_mom_6m
  - mom_vol_hybrid_1y3m
  - mom_vol_hybrid_6m
  - trend_atr
  - trend_breakout
  - dual_momentum
  - ai_elite
  - elite_hybrid
  - elite_risk
  - dynamic_bh_1y / dynamic_bh_6m / dynamic_bh_3m / dynamic_bh_1m
  - static_bh_1y / static_bh_6m / static_bh_3m / static_bh_1m
  - bh_1y_weekly
  - quality_mom
  - turnaround
  - ratio_3m_1y
  - ratio_1y_3m
  - sector_rotation
  - mean_reversion
  - volatility_adj_mom
  - concentrated_3m
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
        from shared_strategies import select_momentum_volatility_hybrid_1y3m_stocks
        return select_momentum_volatility_hybrid_1y3m_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n
        )

    elif strategy_name == 'trend_atr':
        from new_strategies import select_trend_following_atr_stocks
        return select_trend_following_atr_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n
        )[0]

    elif strategy_name == 'dual_momentum':
        from new_strategies import select_dual_momentum_stocks
        return select_dual_momentum_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)[0]

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
            lookback_days=365, top_n=top_n, apply_performance_filter=True
        )

    elif strategy_name == 'quality_mom':
        from quality_momentum_strategy import select_quality_momentum_stocks
        return select_quality_momentum_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'trend_breakout':
        from new_strategies import select_trend_breakout_stocks
        return select_trend_breakout_stocks(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n
        )

    # Risk-Adj Mom variants
    elif strategy_name in ['risk_adj_mom_1m', '1m_volsweet']:
        from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
        return select_risk_adj_mom_1m_vol_sweet_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'risk_adj_mom_6m':
        from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks
        return select_risk_adj_mom_6m_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    # Mom-Vol Hybrid variants
    elif strategy_name == 'mom_vol_hybrid_6m':
        from shared_strategies import select_momentum_volatility_hybrid_stocks
        return select_momentum_volatility_hybrid_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=180, top_n=top_n)

    # Elite strategies
    elif strategy_name == 'elite_hybrid':
        from elite_hybrid_strategy import select_elite_hybrid_stocks
        return select_elite_hybrid_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'elite_risk':
        from elite_risk_strategy import select_elite_risk_stocks
        return select_elite_risk_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    # Dynamic BH variants
    elif strategy_name == 'dynamic_bh_6m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=180, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'dynamic_bh_3m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=90, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'dynamic_bh_1m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=30, top_n=top_n, apply_performance_filter=True)

    # Static BH variants
    elif strategy_name == 'static_bh_6m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=180, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'static_bh_6m_perf':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=180, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'static_bh_9m_perf':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=270, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'bh_1y_1m_rank':
        from shared_strategies import select_bh_1y_1m_rank_stocks
        return select_bh_1y_1m_rank_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'bh_1y_6m_rank':
        from shared_strategies import select_bh_1y_6m_rank_stocks
        return select_bh_1y_6m_rank_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'bh_1y_6m_blend':
        from shared_strategies import select_bh_1y_6m_blend_stocks
        return select_bh_1y_6m_blend_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'bh_1y_weekly':
        from shared_strategies import select_top_performers
        return select_top_performers(
            list(ticker_data_grouped.keys()), ticker_data_grouped, current_date,
            lookback_days=365, top_n=top_n, apply_performance_filter=True
        )

    elif strategy_name == 'early_leader_accel':
        from shared_strategies import select_early_leader_accel_stocks
        return select_early_leader_accel_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'bh_1y_sma200':
        from shared_strategies import select_bh_1y_sma200_stocks
        return select_bh_1y_sma200_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'bh_1y_fcf_rank':
        from shared_strategies import select_bh_1y_fcf_rank_stocks
        return select_bh_1y_fcf_rank_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date=current_date, top_n=top_n)

    elif strategy_name == 'foresight_mimic':
        from new_strategies import select_foresight_mimic_stocks
        return select_foresight_mimic_stocks(
            list(ticker_data_grouped.keys()),
            ticker_data_grouped,
            current_date=current_date,
            top_n=top_n,
        )

    elif strategy_name == 'static_bh_3m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=90, top_n=top_n, apply_performance_filter=True)

    elif strategy_name == 'static_bh_1m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=30, top_n=top_n, apply_performance_filter=True)

    # Other strategies
    elif strategy_name == 'turnaround':
        from shared_strategies import select_turnaround_stocks
        return select_turnaround_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'ratio_3m_1y':
        from shared_strategies import select_3m_1y_ratio_stocks
        return select_3m_1y_ratio_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'ratio_1y_3m':
        from shared_strategies import select_1y_3m_ratio_stocks
        return select_1y_3m_ratio_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'sector_rotation':
        from shared_strategies import select_sector_rotation_etfs
        return select_sector_rotation_etfs(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'mean_reversion':
        from shared_strategies import select_mean_reversion_stocks
        return select_mean_reversion_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'volatility_adj_mom':
        from shared_strategies import select_volatility_adj_mom_stocks
        return select_volatility_adj_mom_stocks(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, top_n)

    elif strategy_name == 'concentrated_3m':
        from shared_strategies import select_top_performers
        return select_top_performers(list(ticker_data_grouped.keys()), ticker_data_grouped, current_date, lookback_days=90, top_n=top_n, apply_performance_filter=True)

    else:
        print(f"❌ Unknown strategy: {strategy_name}")
        print("Available: risk_adj_mom_3m, risk_adj_mom_1m, 1m_volsweet, risk_adj_mom_6m, mom_vol_hybrid_1y3m, mom_vol_hybrid_6m,")
        print("          trend_atr, trend_breakout, dual_momentum, ai_elite, elite_hybrid, elite_risk,")
        print("          dynamic_bh_1y/6m/3m/1m, static_bh_1y/6m/3m/1m, bh_1y_weekly, quality_mom,")
        print("          turnaround, ratio_3m_1y, ratio_1y_3m, sector_rotation, mean_reversion, volatility_adj_mom, concentrated_3m")
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
