#!/usr/bin/env python3
"""
Portfolio Parameter Test - Runs ACTUAL backtesting with different portfolio sizes and buffers.
Tests: Dynamic BH 1Y/6M/3M/1M, BH 1Y/6M/3M Monthly, Multi-TF Ensemble
Uses same data loading as main.py
"""
import sys
sys.path.insert(0, '/home/mvondermey/ai-stock-advisor/src')

import config

# Disable ALL ENABLE_ flags dynamically
for attr in dir(config):
    if attr.startswith('ENABLE_') and isinstance(getattr(config, attr), bool):
        setattr(config, attr, False)

# Explicitly disable the master STATIC_BH flag
config.ENABLE_STATIC_BH = False

# Explicitly disable ratio strategies
config.ENABLE_3M_1Y_RATIO = False
config.ENABLE_1M_3M_RATIO = False

# Explicitly disable all AI strategies
config.ENABLE_AI_ELITE = False
config.ENABLE_AI_ELITE_MONTHLY = False
config.ENABLE_AI_ELITE_FILTERED = False
config.ENABLE_AI_ELITE_MARKET_UP = False

# Enable ONLY the strategies we want to test
config.ENABLE_DYNAMIC_BH_1Y = True
config.ENABLE_DYNAMIC_BH_6M = True
config.ENABLE_DYNAMIC_BH_3M = True
config.ENABLE_DYNAMIC_BH_1M = True
config.ENABLE_STATIC_BH_1Y_MONTHLY = True
config.ENABLE_STATIC_BH_6M_MONTHLY = True
config.ENABLE_STATIC_BH_3M_MONTHLY = True
config.ENABLE_MULTI_TIMEFRAME_ENSEMBLE = True

# Keep essential non-strategy flags enabled
config.ENABLE_PRICE_CACHE = True
config.ENABLE_PARALLEL_STRATEGIES = True

# Debug: Print key config values to verify they're set correctly
print(f"DEBUG: ENABLE_STATIC_BH = {config.ENABLE_STATIC_BH}")
print(f"DEBUG: ENABLE_3M_1Y_RATIO = {config.ENABLE_3M_1Y_RATIO}")
print(f"DEBUG: ENABLE_1M_3M_RATIO = {config.ENABLE_1M_3M_RATIO}")
print(f"DEBUG: ENABLE_DYNAMIC_BH_1Y = {config.ENABLE_DYNAMIC_BH_1Y}")
print(f"DEBUG: ENABLE_STATIC_BH_1Y_MONTHLY = {config.ENABLE_STATIC_BH_1Y_MONTHLY}")

from datetime import datetime, timedelta, timezone
from multiprocessing import cpu_count

# (size, buffer) combinations to test
PARAM_COMBINATIONS = [
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4), (2, 5),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9),
    (5, 6), (5, 7), (10, 12)
]

# Global data cache (loaded once)
_cached_data = None
_cached_tickers = None

def load_data_once():
    """Load ticker data from cache only (no download)"""
    global _cached_data, _cached_tickers
    if _cached_data is not None:
        return _cached_data, _cached_tickers

    from pathlib import Path
    import pandas as pd

    cache_dir = Path('/home/mvondermey/ai-stock-advisor/data_cache')

    print("Loading ticker data from cache (no download)...")
    dfs = []
    cache_files = list(cache_dir.glob('*.csv'))

    for f in cache_files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'Close' in df.columns and len(df) > 50:
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize('UTC')
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'date'})
                df['ticker'] = f.stem
                dfs.append(df)
        except:
            pass

    # Combine into single long-format DataFrame (what backtesting expects)
    all_tickers_data = pd.concat(dfs, ignore_index=True)
    initial_top_tickers = list(all_tickers_data['ticker'].unique())
    print(f"Loaded {len(initial_top_tickers)} tickers from cache ({len(all_tickers_data)} rows)")

    _cached_data = all_tickers_data
    _cached_tickers = initial_top_tickers
    return all_tickers_data, initial_top_tickers

def run_backtest_with_params(portfolio_size, buffer_size, all_tickers_data, initial_top_tickers):
    """Run actual backtesting with specific portfolio size and buffer"""
    from backtesting import _run_portfolio_backtest_walk_forward
    import config

    # Debug: Print config values seen by backtesting
    print(f"DEBUG IN BACKTESTING: ENABLE_STATIC_BH = {config.ENABLE_STATIC_BH}")
    print(f"DEBUG IN BACKTESTING: ENABLE_3M_1Y_RATIO = {config.ENABLE_3M_1Y_RATIO}")
    print(f"DEBUG IN BACKTESTING: ENABLE_1M_3M_RATIO = {config.ENABLE_1M_3M_RATIO}")

    # Override config with test parameters
    config.PORTFOLIO_SIZE = portfolio_size
    config.PORTFOLIO_BUFFER_SIZE = buffer_size
    config.INVESTMENT_PER_STOCK = config.TOTAL_CAPITAL / portfolio_size

    # Use max CPU cores since no AI training
    config.NUM_PROCESSES = cpu_count()
    config.TRAINING_NUM_PROCESSES = cpu_count()

    # Prepare backtest parameters
    bt_end = datetime.now(timezone.utc)
    bt_start = bt_end - timedelta(days=config.BACKTEST_DAYS)

    # Get top performers data (same as main.py)
    top_performers_data = [(t, 0.0) for t in initial_top_tickers]

    capital_per_stock = config.INVESTMENT_PER_STOCK

    # Run backtesting
    result = _run_portfolio_backtest_walk_forward(
        all_tickers_data=all_tickers_data,
        backtest_start_date=bt_start,
        backtest_end_date=bt_end,
        initial_top_tickers=initial_top_tickers,
        capital_per_stock=capital_per_stock,
        period_name="Test",
        top_performers_data=top_performers_data,
    )

    return result

def extract_strategy_results(result, initial_capital):
    """Extract strategy values from backtest result dict"""
    strategies = result.get('strategies', {})
    extracted = {}

    strategy_map = {
        'dynamic_bh_1y': 'Dynamic BH 1Y',
        'dynamic_bh_6m': 'Dynamic BH 6M',
        'dynamic_bh_3m': 'Dynamic BH 3M',
        'dynamic_bh_1m': 'Dynamic BH 1M',
        'bh_1y_monthly': 'BH 1Y Mth',
        'bh_6m_monthly': 'BH 6M Mth',
        'bh_3m_monthly': 'BH 3M Mth',
        'multi_tf_ensemble': 'Multi-TF',
    }

    for key, display_name in strategy_map.items():
        if key in strategies and strategies[key]['value']:
            value = strategies[key]['value']
            ret = ((value - initial_capital) / initial_capital) * 100
            costs = strategies[key].get('costs', 0)
            extracted[display_name] = {
                'value': value,
                'return': ret,
                'costs': costs
            }

    return extracted

def main():
    import config
    initial_capital = config.TOTAL_CAPITAL

    print("=" * 120)
    print("PORTFOLIO PARAMETER TEST - Using ACTUAL Backtesting")
    print("Strategies: Dynamic BH 1Y/6M/3M/1M, BH 1Y/6M/3M Monthly, Multi-TF Ensemble")
    print("=" * 120)

    all_results = []

    # Load data once (same as main.py)
    all_tickers_data, initial_top_tickers = load_data_once()

    for size, buffer in PARAM_COMBINATIONS:
        print(f"\n{'='*60}")
        print(f"Testing: Portfolio Size={size}, Buffer={buffer}")
        print(f"{'='*60}")

        try:
            result = run_backtest_with_params(size, buffer, all_tickers_data, initial_top_tickers)
            strategies = extract_strategy_results(result, initial_capital)

            for strat_name, strat_data in strategies.items():
                all_results.append({
                    'strategy': strat_name,
                    'size': size,
                    'buffer': buffer,
                    'value': strat_data['value'],
                    'return': strat_data['return'],
                    'costs': strat_data['costs']
                })
                print(f"  {strat_name}: ${strat_data['value']:,.0f} ({strat_data['return']:+.1f}%)")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY - Sorted by Return")
    print("=" * 120)
    print(f"{'Strategy':<15} {'Size':<6} {'Buffer':<8} {'Value':<15} {'Return':<12} {'Costs'}")
    print("-" * 120)

    for r in sorted(all_results, key=lambda x: x['return'], reverse=True)[:30]:
        print(f"{r['strategy']:<15} {r['size']:<6} {r['buffer']:<8} ${r['value']:>12,.0f} {r['return']:>+10.1f}% ${r['costs']:>8,.0f}")

    # Best per strategy
    print("\n" + "=" * 120)
    print("BEST CONFIGURATION PER STRATEGY")
    print("=" * 120)

    strategy_names = set(r['strategy'] for r in all_results)
    for strat in sorted(strategy_names):
        strat_results = [r for r in all_results if r['strategy'] == strat]
        if strat_results:
            best = max(strat_results, key=lambda x: x['return'])
            print(f"{strat:<15}: Size={best['size']}, Buffer={best['buffer']} → {best['return']:+.1f}%")

    # Overall winner
    if all_results:
        winner = max(all_results, key=lambda x: x['return'])
        print(f"\n🏆 OVERALL BEST: {winner['strategy']} (Size={winner['size']}, Buffer={winner['buffer']}) → {winner['return']:+.1f}%")

if __name__ == "__main__":
    main()
