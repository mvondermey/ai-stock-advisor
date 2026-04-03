"""
Rebalance Horizon Optimizer

Tests multiple rebalance horizons for static strategies and identifies
the best performing horizon for each strategy type.

Optimized for speed using vectorized operations and pre-computed price matrices.
"""

import pandas as pd
import numpy as np
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config import (
    PORTFOLIO_SIZE, TRANSACTION_COST,
    REBALANCE_HORIZON_MIN, REBALANCE_HORIZON_MAX
)


def _simulate_strategy_vectorized(
    price_matrix: np.ndarray,
    dates: np.ndarray,
    tickers: List[str],
    strategy_type: str,
    rebalance_days: int,
    initial_capital: float,
    portfolio_size: int
) -> Tuple[float, float]:
    """
    Vectorized simulation of a static buy & hold strategy.

    Args:
        price_matrix: 2D array of prices [dates x tickers]
        dates: Array of normalized daily dates
        tickers: List of ticker names
        strategy_type: '1Y', '6M', '3M', or '1M'
        rebalance_days: Days between rebalances
        initial_capital: Starting capital
        portfolio_size: Number of stocks to hold

    Returns:
        Tuple of (final_value, total_txn_cost)
    """
    n_dates, n_tickers = price_matrix.shape
    dates = np.asarray(dates, dtype='datetime64[D]')

    if n_dates < 20:
        return initial_capital, 0.0

    # Lookback period based on strategy
    lookback_map = {'1Y': 365, '6M': 180, '3M': 90, '1M': 30}
    lookback_days = lookback_map.get(strategy_type, 90)

    # Initialize
    cash = initial_capital
    holdings = np.zeros(n_tickers)  # shares held per ticker
    total_txn_cost = 0.0
    days_since_rebalance = rebalance_days  # Force initial rebalance

    for day_idx in range(n_dates):
        days_since_rebalance += 1

        # Check if we should rebalance
        if days_since_rebalance >= rebalance_days:
            current_prices = price_matrix[day_idx]

            # Find lookback start index using actual calendar dates.
            target_date = dates[day_idx] - np.timedelta64(lookback_days, 'D')
            lookback_idx = int(np.searchsorted(dates, target_date, side='left'))

            # Skip if not enough history
            if day_idx - lookback_idx < 20:
                continue

            start_prices = price_matrix[lookback_idx]

            # Calculate returns (vectorized)
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = (current_prices / start_prices - 1) * 100

            # Mask invalid returns
            valid_mask = np.isfinite(returns) & (start_prices > 0) & (current_prices > 0)
            returns = np.where(valid_mask, returns, -np.inf)

            # Get top N performers
            top_indices = np.argsort(returns)[-portfolio_size:][::-1]
            top_indices = top_indices[returns[top_indices] > -np.inf]

            if len(top_indices) == 0:
                continue

            # Sell current holdings not in top picks
            for i in range(n_tickers):
                if holdings[i] > 0 and i not in top_indices:
                    if current_prices[i] > 0:
                        sell_value = holdings[i] * current_prices[i]
                        txn_cost = sell_value * TRANSACTION_COST
                        cash += sell_value - txn_cost
                        total_txn_cost += txn_cost
                    holdings[i] = 0

            # Calculate current portfolio value
            portfolio_value = cash
            for i in range(n_tickers):
                if holdings[i] > 0 and current_prices[i] > 0:
                    portfolio_value += holdings[i] * current_prices[i]

            # Buy new positions (equal weight)
            stocks_to_buy = [i for i in top_indices if holdings[i] == 0]
            if stocks_to_buy and cash > 100:  # Min cash threshold
                capital_per_stock = cash / (len(stocks_to_buy) * (1 + TRANSACTION_COST))
                for i in stocks_to_buy:
                    if current_prices[i] > 0:
                        shares = capital_per_stock / current_prices[i]
                        txn_cost = capital_per_stock * TRANSACTION_COST
                        holdings[i] = shares
                        cash -= capital_per_stock + txn_cost
                        total_txn_cost += txn_cost

            days_since_rebalance = 0

    # Calculate final portfolio value
    final_prices = price_matrix[-1]
    final_value = cash
    for i in range(n_tickers):
        if holdings[i] > 0 and final_prices[i] > 0:
            final_value += holdings[i] * final_prices[i]

    return final_value, total_txn_cost


def optimize_rebalance_horizons(
    all_tickers_data: pd.DataFrame,
    backtest_start: datetime,
    backtest_end: datetime,
    initial_capital: float,
    portfolio_size: int = 3,
    strategy_types: List[str] = ['1Y', '6M', '3M', '1M']
) -> Dict[str, Dict]:
    """
    Test rebalance horizons from REBALANCE_HORIZON_MIN to REBALANCE_HORIZON_MAX
    for each strategy type and find the best performing horizon.

    Uses vectorized operations for speed - runs sequentially but very fast.
    """
    # Ensure minimum optimization period
    MIN_OPTIMIZATION_DAYS = REBALANCE_HORIZON_MAX * 3
    actual_days = (backtest_end - backtest_start).days

    if actual_days < MIN_OPTIMIZATION_DAYS:
        optimization_start = backtest_end - timedelta(days=MIN_OPTIMIZATION_DAYS)
        print(f"\n🔄 REBALANCE HORIZON OPTIMIZATION", flush=True)
        print("=" * 50, flush=True)
        print(f"   ⚠️ Backtest period too short ({actual_days} days)", flush=True)
        print(f"   📅 Using extended period: {optimization_start.date()} to {backtest_end.date()} ({MIN_OPTIMIZATION_DAYS} days)", flush=True)
    else:
        optimization_start = backtest_start
        print(f"\n🔄 REBALANCE HORIZON OPTIMIZATION", flush=True)
        print("=" * 50, flush=True)

    print(f"   Testing horizons: {REBALANCE_HORIZON_MIN} to {REBALANCE_HORIZON_MAX} days", flush=True)
    print(f"   Strategy types: {strategy_types}", flush=True)
    print(f"   Optimization period: {optimization_start.date()} to {backtest_end.date()}", flush=True)

    # Build price matrix (vectorized)
    print(f"   📊 Building price matrix...", flush=True)

    # Filter data to optimization period
    if 'date' in all_tickers_data.columns:
        filtered_data = all_tickers_data.copy()
        filtered_data['date'] = pd.to_datetime(filtered_data['date'], utc=True).dt.normalize()

        optimization_start_ts = pd.Timestamp(optimization_start)
        backtest_end_ts = pd.Timestamp(backtest_end)
        if optimization_start_ts.tzinfo is None:
            optimization_start_ts = optimization_start_ts.tz_localize('UTC')
        else:
            optimization_start_ts = optimization_start_ts.tz_convert('UTC')
        if backtest_end_ts.tzinfo is None:
            backtest_end_ts = backtest_end_ts.tz_localize('UTC')
        else:
            backtest_end_ts = backtest_end_ts.tz_convert('UTC')

        optimization_start_ts = optimization_start_ts.normalize()
        backtest_end_ts = backtest_end_ts.normalize()

        mask = (
            (filtered_data['date'] >= optimization_start_ts) &
            (filtered_data['date'] <= backtest_end_ts)
        )
        filtered_data = filtered_data[mask].copy()
    else:
        filtered_data = all_tickers_data.copy()

    # Pivot to get price matrix
    close_col = 'Close' if 'Close' in filtered_data.columns else 'close'
    pivot_df = filtered_data.pivot_table(
        index='date',
        columns='ticker',
        values=close_col,
        aggfunc='last'
    )

    # Forward fill missing prices (stock didn't trade that day)
    pivot_df = pivot_df.ffill()

    # Convert to numpy for speed
    price_matrix = pivot_df.values.astype(np.float64)
    dates = pivot_df.index.values
    tickers = list(pivot_df.columns)

    print(f"   📊 Price matrix: {price_matrix.shape[0]} days × {price_matrix.shape[1]} tickers", flush=True)

    # Test horizons
    horizons = list(range(REBALANCE_HORIZON_MIN, REBALANCE_HORIZON_MAX + 1))
    total_sims = len(horizons) * len(strategy_types)

    print(f"   🚀 Running {total_sims} simulations (vectorized, sequential)...", flush=True)

    results = {}
    for st in strategy_types:
        results[st] = {'all_results': [], 'best_horizon': None, 'best_return': -float('inf'), 'best_txn_cost': 0}

    # Run simulations with progress
    from tqdm import tqdm
    completed = 0

    with tqdm(total=total_sims, desc="   Optimizing horizons", ncols=100) as pbar:
        for strategy_type in strategy_types:
            for horizon in horizons:
                final_value, txn_cost = _simulate_strategy_vectorized(
                    price_matrix=price_matrix,
                    dates=dates,
                    tickers=tickers,
                    strategy_type=strategy_type,
                    rebalance_days=horizon,
                    initial_capital=initial_capital,
                    portfolio_size=portfolio_size
                )

                return_pct = ((final_value / initial_capital) - 1) * 100
                results[strategy_type]['all_results'].append((horizon, return_pct, txn_cost))

                if return_pct > results[strategy_type]['best_return']:
                    results[strategy_type]['best_return'] = return_pct
                    results[strategy_type]['best_horizon'] = horizon
                    results[strategy_type]['best_txn_cost'] = txn_cost

                pbar.update(1)

    # Print summary
    print(f"\n   ✅ OPTIMIZATION COMPLETE", flush=True)
    print("-" * 50, flush=True)
    for strategy_type in strategy_types:
        r = results[strategy_type]
        print(f"   Static BH {strategy_type}:", flush=True)
        print(f"      Best horizon: {r['best_horizon']} days", flush=True)
        print(f"      Best return: {r['best_return']:+.1f}%", flush=True)
        print(f"      Transaction costs: ${r['best_txn_cost']:.0f}", flush=True)

        # Show top 5 horizons
        all_sorted = sorted(r['all_results'], key=lambda x: x[1], reverse=True)[:5]
        horizon_str = ", ".join([f"{h}d={ret:+.1f}%" for h, ret, _ in all_sorted])
        print(f"      Top 5: {horizon_str}", flush=True)
    print("-" * 50, flush=True)

    # Cleanup
    del price_matrix, pivot_df, filtered_data
    gc.collect()

    return results


def get_best_horizons(optimization_results: Dict) -> Dict[str, int]:
    """
    Extract the best horizon for each strategy type from optimization results.
    """
    return {
        st: results['best_horizon']
        for st, results in optimization_results.items()
        if results['best_horizon'] is not None
    }
