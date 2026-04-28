#!/usr/bin/env python3
"""
Multi-Strategy with Acceleration Ranking
Combines multiple strategies and ranks selections by acceleration score
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def _execute_strategy_live(strategy: str, all_tickers: list, ticker_data_grouped: dict, portfolio_size: int) -> list:
    """Execute a strategy directly with current data (for --live-run mode)."""
    current_date = datetime.now(timezone.utc)

    if strategy == 'static_bh_1y':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=365, top_n=portfolio_size)

    elif strategy == 'static_bh_6m':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=180, top_n=portfolio_size)

    elif strategy == 'static_bh_6m_perf':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=180, top_n=portfolio_size)

    elif strategy == 'static_bh_9m_perf':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=270, top_n=portfolio_size)

    elif strategy == 'bh_1y_1m_rank':
        from shared_strategies import select_bh_1y_1m_rank_stocks
        return select_bh_1y_1m_rank_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_6m_rank':
        from shared_strategies import select_bh_1y_6m_rank_stocks
        return select_bh_1y_6m_rank_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_6m_blend':
        from shared_strategies import select_bh_1y_6m_blend_stocks
        return select_bh_1y_6m_blend_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_weekly':
        from shared_strategies import select_top_performers
        return select_top_performers(
            all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=365, top_n=portfolio_size
        )

    elif strategy == 'early_leader_accel':
        from shared_strategies import select_early_leader_accel_stocks
        return select_early_leader_accel_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_sma200':
        from shared_strategies import select_bh_1y_sma200_stocks
        return select_bh_1y_sma200_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_fcf_rank':
        from shared_strategies import select_bh_1y_fcf_rank_stocks
        return select_bh_1y_fcf_rank_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'foresight_mimic':
        from new_strategies import select_foresight_mimic_stocks
        return select_foresight_mimic_stocks(
            all_tickers,
            ticker_data_grouped,
            current_date=current_date,
            top_n=portfolio_size,
        )

    elif strategy == 'static_bh_3m':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=90, top_n=portfolio_size)

    elif strategy == 'static_bh_1m':
        from shared_strategies import select_top_performers
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, lookback_days=30, top_n=portfolio_size)

    elif strategy == 'risk_adj_mom_1m_vol_sweet' or strategy == '1m_volsweet':
        from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
        return select_risk_adj_mom_1m_vol_sweet_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'risk_adj_mom_1m':
        from risk_adj_mom_1m_strategy import select_risk_adj_mom_1m_stocks
        return select_risk_adj_mom_1m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'risk_adj_mom_3m':
        from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
        return select_risk_adj_mom_3m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'risk_adj_mom_6m':
        from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks
        return select_risk_adj_mom_6m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'risk_adj_mom':
        from shared_strategies import select_risk_adj_mom_stocks
        return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_volsweet_accel':
        from shared_strategies import select_bh_1y_volsweet_accel_stocks
        return select_bh_1y_volsweet_accel_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size)

    elif strategy == 'bh_1y_dynamic_accel':
        from shared_strategies import select_bh_1y_dynamic_accel_stocks
        selected, _ = select_bh_1y_dynamic_accel_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=portfolio_size, days_since_rebalance=0, min_days=0, max_days=44)
        return selected

    else:
        print(f"   ⚠️ Strategy '{strategy}' not supported for live execution")
        return []


def calculate_acceleration_score(
    ticker: str,
    ticker_df: pd.DataFrame,
    current_date: datetime = None
) -> Tuple[float, Dict]:
    """
    Calculate acceleration score for a ticker.

    Returns:
        Tuple of (final_score, details_dict)
    """
    if ticker_df is None or len(ticker_df) < 60:
        return 0.0, {}

    close = ticker_df['Close'].dropna()
    if len(close) < 60:
        return 0.0, {}

    # Calculate returns at different windows
    returns_1d = close.pct_change().dropna()
    returns_5d = close.pct_change(5).dropna()
    returns_21d = close.pct_change(21).dropna()

    if len(returns_21d) < 20:
        return 0.0, {}

    # Recent velocity (average of last 10 days)
    recent_velocity = returns_5d.iloc[-10:].mean()

    # Acceleration metrics
    # Recent acceleration (last 10 days vs previous 10 days)
    recent_accel = returns_5d.iloc[-10:].mean() - returns_5d.iloc[-20:-10].mean()

    # Latest acceleration (last 5 days vs previous 5 days)
    latest_accel = returns_5d.iloc[-5:].mean() - returns_5d.iloc[-10:-5].mean()

    # Consistency score (how often acceleration was positive in last 20 days)
    accel_series = returns_5d.rolling(5).mean().diff().iloc[-20:]
    consistency_score = (accel_series > 0).mean() if len(accel_series) > 0 else 0

    # Composite acceleration score (same weights as shared_strategies.py)
    # 40% avg acceleration, 40% latest acceleration, 20% consistency
    accel_score = (recent_accel * 0.4 +
                  latest_accel * 0.4 +
                  consistency_score * recent_accel * 0.2)

    # Scale by velocity (stronger acceleration matters more with higher velocity)
    final_score = accel_score * (1 + recent_velocity * 100)

    details = {
        'velocity': recent_velocity,
        'acceleration': recent_accel,
        'latest_acceleration': latest_accel,
        'consistency': consistency_score,
        'accel_score': accel_score
    }

    return final_score, details


def combine_strategies_with_acceleration(
    strategy_names: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    num_stocks: int = 12,
    execute_live: bool = False
) -> List[Tuple[str, str, float, Dict]]:
    """
    Combine multiple strategies and rank by acceleration score.

    Args:
        strategy_names: List of strategy names to combine
        ticker_data_grouped: Ticker data dictionary
        current_date: Current date for calculations
        num_stocks: Total number of stocks to select
        execute_live: If True, execute strategies directly; if False, read from JSON

    Returns:
        List of (ticker, strategy, acceleration_score, details) tuples
    """
    print(f"\n🚀 Combining strategies with acceleration ranking...")
    print(f"   Strategies: {strategy_names}")
    print(f"   Total stocks to select: {num_stocks}")

    # Get selections from each strategy
    all_selections = {}
    for strategy in strategy_names:
        try:
            if execute_live:
                tickers = _execute_strategy_live(strategy, list(ticker_data_grouped.keys()), ticker_data_grouped, num_stocks)
            else:
                from live_trading import get_strategy_tickers
                tickers = get_strategy_tickers(strategy, list(ticker_data_grouped.keys()), ticker_data_grouped)
            all_selections[strategy] = tickers
            print(f"   {strategy}: {len(tickers)} tickers")
        except Exception as e:
            print(f"   ⚠️ Error in {strategy}: {e}")
            all_selections[strategy] = []

    # Combine all unique tickers
    combined_tickers = set()
    for tickers in all_selections.values():
        combined_tickers.update(tickers)

    print(f"\n📊 Calculating acceleration scores for {len(combined_tickers)} unique tickers...")

    # Calculate acceleration scores
    scored_tickers = []
    for ticker in combined_tickers:
        try:
            score, details = calculate_acceleration_score(ticker, ticker_data_grouped.get(ticker), current_date)

            # Find which strategy selected this ticker
            source_strategy = "Unknown"
            for strategy, tickers in all_selections.items():
                if ticker in tickers:
                    source_strategy = strategy
                    break

            scored_tickers.append((ticker, source_strategy, score, details))

        except Exception as e:
            print(f"   ⚠️ Error scoring {ticker}: {e}")
            continue

    # Sort by acceleration score (highest first)
    scored_tickers.sort(key=lambda x: x[2], reverse=True)

    # Take top N
    final_selections = scored_tickers[:num_stocks]

    print(f"\n✅ Final selections ranked by acceleration:")
    print("-" * 80)
    for i, (ticker, strategy, score, details) in enumerate(final_selections, 1):
        print(f"{i:2}. {ticker:8} | {strategy:15} | Score: {score:8.4f}")
        print(f"    Velocity: {details['velocity']:+.4f}, "
              f"Accel: {details['acceleration']:+.6f}, "
              f"Latest: {details['latest_acceleration']:+.6f}, "
              f"Consistency: {details['consistency']:.0%}")

    return final_selections


def main():
    """Main function for testing"""
    import sys
    from data_utils import load_all_market_data
    from ticker_selection import get_all_tickers

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python multi_strategy_acceleration.py strategy1,strategy2,... [num_stocks]")
        print("Example: python multi_strategy_acceleration.py 1m_volsweet,static_bh_1y 12")
        return

    strategies = sys.argv[1].split(',')
    num_stocks = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    # Load data
    print("Loading market data...")
    all_tickers = get_all_tickers()
    all_tickers_data = load_all_market_data(all_tickers)

    # Convert to ticker_data_grouped format
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

    print(f"Loaded data for {len(ticker_data_grouped)} tickers")

    # Run combined strategy with acceleration
    current_date = datetime.now(timezone.utc)
    selections = combine_strategies_with_acceleration(
        strategies, ticker_data_grouped, current_date, num_stocks
    )

    # Print final ticker list
    final_tickers = [s[0] for s in selections]
    print(f"\n🎯 Final ticker list: {final_tickers}")


def select_multi_strategy_acceleration_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 20,
) -> List[str]:
    """Select stocks using multi-strategy acceleration approach"""
    # Default strategies to combine
    default_strategies = ["static_bh_1y", "risk_adj_mom_3m", "concentrated_3m"]
    
    # Use default strategies if none specified
    strategies = default_strategies
    
    # Combine strategies with acceleration scoring
    selections = combine_strategies_with_acceleration(
        strategies, ticker_data_grouped, current_date or datetime.now(timezone.utc), top_n
    )
    
    # Return just the ticker symbols
    return [ticker for ticker, _, _, _ in selections]


if __name__ == "__main__":
    main()
