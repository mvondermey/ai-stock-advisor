"""
Rebalance Horizon Optimizer

Tests multiple rebalance horizons (30-90 days) for static strategies in parallel
and identifies the best performing horizon for each strategy type.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from config import (
    TRAINING_NUM_PROCESSES, PORTFOLIO_SIZE, TRANSACTION_COST,
    REBALANCE_HORIZON_MIN, REBALANCE_HORIZON_MAX
)


def _simulate_static_strategy(args) -> Tuple[str, int, float, float]:
    """
    Simulate a static buy & hold strategy with a specific rebalance horizon.
    
    Args:
        args: Tuple of (strategy_type, rebalance_days, ticker_data_dict, top_performers, 
              backtest_start, backtest_end, initial_capital, portfolio_size)
    
    Returns:
        Tuple of (strategy_type, rebalance_days, final_value, total_txn_cost)
    """
    (strategy_type, rebalance_days, ticker_data_dict, top_performers,
     backtest_start, backtest_end, initial_capital, portfolio_size) = args
    
    try:
        # Initialize portfolio
        positions = {}  # ticker -> {'shares': float, 'entry_price': float}
        cash = initial_capital
        days_since_rebalance = 0
        initialized = False
        total_txn_cost = 0.0
        
        # Get trading days
        all_dates = set()
        for ticker, data in ticker_data_dict.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index.tolist())
            elif 'date' in data.columns:
                all_dates.update(data['date'].tolist())
        
        trading_days = sorted([d for d in all_dates if backtest_start <= d <= backtest_end])
        
        if not trading_days:
            return (strategy_type, rebalance_days, initial_capital, 0.0)
        
        # Simulate each trading day
        for current_date in trading_days:
            days_since_rebalance += 1
            
            # Check if we should rebalance
            should_rebalance = (
                (not initialized) or
                (rebalance_days > 0 and days_since_rebalance >= rebalance_days)
            )
            
            if should_rebalance:
                # Get current top performers for this strategy type
                if strategy_type == '1Y':
                    # Use 1-year performance
                    lookback_days = 365
                elif strategy_type == '3M':
                    # Use 3-month performance
                    lookback_days = 90
                else:  # 1M
                    # Use 1-month performance
                    lookback_days = 30
                
                # Calculate performance for each ticker
                perf_list = []
                lookback_start = current_date - timedelta(days=lookback_days)
                
                for ticker, data in ticker_data_dict.items():
                    try:
                        if isinstance(data.index, pd.DatetimeIndex):
                            period_data = data.loc[lookback_start:current_date]
                        else:
                            period_data = data[(data['date'] >= lookback_start) & (data['date'] <= current_date)]
                        
                        if len(period_data) < 20:
                            continue
                        
                        close_col = 'Close' if 'Close' in period_data.columns else 'close'
                        if close_col not in period_data.columns:
                            continue
                        
                        prices = period_data[close_col].dropna()
                        if len(prices) < 2:
                            continue
                        
                        perf = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
                        current_price = prices.iloc[-1]
                        perf_list.append((ticker, perf, current_price))
                    except:
                        continue
                
                # Sort by performance and get top N
                perf_list.sort(key=lambda x: x[1], reverse=True)
                new_stocks = [t[0] for t in perf_list[:portfolio_size]]
                price_map = {t[0]: t[2] for t in perf_list}
                
                if new_stocks:
                    # Sell positions not in new_stocks
                    for ticker in list(positions.keys()):
                        if ticker not in new_stocks:
                            if ticker in price_map:
                                sell_value = positions[ticker]['shares'] * price_map[ticker]
                                txn_cost = sell_value * TRANSACTION_COST
                                cash += sell_value - txn_cost
                                total_txn_cost += txn_cost
                            del positions[ticker]
                    
                    # Buy new positions
                    stocks_to_buy = [t for t in new_stocks if t not in positions]
                    if stocks_to_buy and cash > 0:
                        capital_per_stock = cash / len(stocks_to_buy)
                        for ticker in stocks_to_buy:
                            if ticker in price_map and price_map[ticker] > 0:
                                shares = capital_per_stock / price_map[ticker]
                                txn_cost = capital_per_stock * TRANSACTION_COST
                                positions[ticker] = {
                                    'shares': shares,
                                    'entry_price': price_map[ticker]
                                }
                                cash -= capital_per_stock
                                total_txn_cost += txn_cost
                    
                    initialized = True
                    days_since_rebalance = 0
        
        # Calculate final portfolio value
        final_value = cash
        final_date = trading_days[-1] if trading_days else backtest_end
        
        for ticker, pos in positions.items():
            try:
                data = ticker_data_dict.get(ticker)
                if data is None:
                    continue
                
                if isinstance(data.index, pd.DatetimeIndex):
                    recent = data.loc[:final_date].tail(5)
                else:
                    recent = data[data['date'] <= final_date].tail(5)
                
                close_col = 'Close' if 'Close' in recent.columns else 'close'
                if close_col in recent.columns and len(recent) > 0:
                    final_price = recent[close_col].dropna().iloc[-1]
                    final_value += pos['shares'] * final_price
            except:
                continue
        
        return (strategy_type, rebalance_days, final_value, total_txn_cost)
    
    except Exception as e:
        return (strategy_type, rebalance_days, initial_capital, 0.0)


def optimize_rebalance_horizons(
    all_tickers_data: pd.DataFrame,
    backtest_start: datetime,
    backtest_end: datetime,
    initial_capital: float,
    portfolio_size: int = 3,
    strategy_types: List[str] = ['1Y', '3M', '1M']
) -> Dict[str, Dict]:
    """
    Test all rebalance horizons from REBALANCE_HORIZON_MIN to REBALANCE_HORIZON_MAX
    for each strategy type and find the best performing horizon.
    
    Args:
        all_tickers_data: DataFrame with all ticker data (long format with 'ticker' column)
        backtest_start: Start date for backtesting
        backtest_end: End date for backtesting
        initial_capital: Starting capital
        portfolio_size: Number of stocks to hold
        strategy_types: List of strategy types to test ('1Y', '3M', '1M')
    
    Returns:
        Dict with results for each strategy type:
        {
            '1Y': {'best_horizon': 45, 'best_return': 55.6, 'all_results': [(30, 50.2), (31, 51.1), ...]},
            '3M': {...},
            '1M': {...}
        }
    """
    print(f"\n🔄 REBALANCE HORIZON OPTIMIZATION", flush=True)
    print("=" * 50, flush=True)
    print(f"   Testing horizons: {REBALANCE_HORIZON_MIN} to {REBALANCE_HORIZON_MAX} days", flush=True)
    print(f"   Strategy types: {strategy_types}", flush=True)
    print(f"   Backtest period: {backtest_start.date()} to {backtest_end.date()}", flush=True)
    
    # Convert data to dict format for faster access in workers
    print(f"   📊 Preparing data for parallel processing...", flush=True)
    ticker_data_dict = {}
    for ticker in all_tickers_data['ticker'].unique():
        ticker_df = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
        if 'date' in ticker_df.columns:
            ticker_df = ticker_df.set_index('date')
        ticker_data_dict[ticker] = ticker_df
    
    print(f"   📊 Prepared data for {len(ticker_data_dict)} tickers", flush=True)
    
    # Generate all tasks
    horizons = list(range(REBALANCE_HORIZON_MIN, REBALANCE_HORIZON_MAX + 1))
    tasks = []
    
    for strategy_type in strategy_types:
        for horizon in horizons:
            tasks.append((
                strategy_type, horizon, ticker_data_dict, None,
                backtest_start, backtest_end, initial_capital, portfolio_size
            ))
    
    total_tasks = len(tasks)
    print(f"   🚀 Running {total_tasks} simulations in parallel ({len(horizons)} horizons × {len(strategy_types)} strategies)", flush=True)
    
    # Run in parallel using NUM_PROCESSES (rebalance optimization is not training)
    from config import NUM_PROCESSES
    n_workers = min(NUM_PROCESSES, total_tasks)
    print(f"   🔧 Using {n_workers} workers (NUM_PROCESSES - not GPU training)", flush=True)
    
    results = {}
    for st in strategy_types:
        results[st] = {'all_results': [], 'best_horizon': None, 'best_return': -float('inf'), 'best_txn_cost': 0}
    
    # Process in batches
    batch_size = n_workers * 4
    total_batches = (total_tasks + batch_size - 1) // batch_size
    
    completed = 0
    with mp.Pool(processes=n_workers) as pool:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_tasks = tasks[batch_start:batch_end]
            
            batch_results = pool.map(_simulate_static_strategy, batch_tasks)
            
            for strategy_type, horizon, final_value, txn_cost in batch_results:
                return_pct = ((final_value / initial_capital) - 1) * 100
                results[strategy_type]['all_results'].append((horizon, return_pct, txn_cost))
                
                if return_pct > results[strategy_type]['best_return']:
                    results[strategy_type]['best_return'] = return_pct
                    results[strategy_type]['best_horizon'] = horizon
                    results[strategy_type]['best_txn_cost'] = txn_cost
            
            completed += len(batch_tasks)
            print(f"   📊 Progress: {completed}/{total_tasks} ({completed*100//total_tasks}%)", flush=True)
    
    # Print summary
    print(f"\n   ✅ OPTIMIZATION COMPLETE", flush=True)
    print("-" * 50, flush=True)
    for strategy_type in strategy_types:
        r = results[strategy_type]
        print(f"   Static BH {strategy_type}:", flush=True)
        print(f"      Best horizon: {r['best_horizon']} days", flush=True)
        print(f"      Best return: {r['best_return']:+.1f}%", flush=True)
        print(f"      Transaction costs: ${r['best_txn_cost']:.0f}", flush=True)
    print("-" * 50, flush=True)
    
    return results


def get_best_horizons(optimization_results: Dict) -> Dict[str, int]:
    """
    Extract the best horizon for each strategy type from optimization results.
    
    Returns:
        Dict mapping strategy type to best horizon days
    """
    return {
        st: results['best_horizon'] 
        for st, results in optimization_results.items()
        if results['best_horizon'] is not None
    }
