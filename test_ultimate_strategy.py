#!/usr/bin/env python3
"""
Test Ultimate Strategy - Run ONLY the Ultimate strategy and compare with top performers.
Uses same data loading and backtest period as main.py
"""
import sys
sys.path.insert(0, '/home/mvondermey/ai-stock-advisor/src')

import config

# Disable ALL strategies
for attr in dir(config):
    if attr.startswith('ENABLE_') and isinstance(getattr(config, attr), bool):
        setattr(config, attr, False)

# Enable ONLY Ultimate strategy and a few comparison strategies
config.ENABLE_ULTIMATE = True
config.ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET = True  # 1M VolSweet for comparison
config.ENABLE_VOL_SWEET_MOM = True  # VolSweet Mom for comparison

# Keep essential non-strategy flags enabled
config.ENABLE_PRICE_CACHE = True
config.ENABLE_PARALLEL_STRATEGIES = True

from datetime import datetime, timedelta, timezone
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import threading

# Backtest parameters
BACKTEST_DAYS = 51  # Same as output22.03.log (51 trading days)
PORTFOLIO_SIZE = 10
INITIAL_CAPITAL = 300000
TRANSACTION_COST = 0.001  # 0.1% per trade

_cached_data = None
_cached_tickers = None

def load_data_once():
    """Load ticker data from cache only (no download)"""
    global _cached_data, _cached_tickers
    if _cached_data is not None:
        return _cached_data, _cached_tickers

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

    all_tickers_data = pd.concat(dfs, ignore_index=True)
    initial_top_tickers = list(all_tickers_data['ticker'].unique())
    print(f"Loaded {len(initial_top_tickers)} tickers from cache ({len(all_tickers_data)} rows)")

    _cached_data = all_tickers_data
    _cached_tickers = initial_top_tickers
    return all_tickers_data, initial_top_tickers


def run_simple_backtest(strategy_name, select_func, all_tickers_data, initial_top_tickers,
                        backtest_start, backtest_end, portfolio_size, initial_capital,
                        verbose=False, ticker_data_grouped=None, trading_days=None):
    """Run a simple daily rebalancing backtest for a single strategy."""
    import sys

    # Use pre-grouped data if provided (much faster for parallel execution)
    if ticker_data_grouped is None:
        # Group data by ticker for fast lookups - use pandas groupby (MUCH faster)
        grouped = all_tickers_data.groupby('ticker')

        ticker_data_grouped = {}
        for ticker in initial_top_tickers:
            if ticker in grouped.groups:
                ticker_df = grouped.get_group(ticker).copy()
                if len(ticker_df) > 0:
                    ticker_df = ticker_df.set_index('date').sort_index()
                    # Resample to daily (use last price of each day)
                    ticker_df = ticker_df.resample('D').last().dropna(subset=['Close'])
                    if len(ticker_df) > 0:
                        ticker_data_grouped[ticker] = ticker_df

    # Use pre-calculated trading days if provided
    if trading_days is None:
        all_dates = set()
        for ticker, df in ticker_data_grouped.items():
            all_dates.update(df.index.tolist())
        trading_days = sorted([d for d in all_dates if backtest_start <= d <= backtest_end])

    # Initialize portfolio
    cash = initial_capital
    positions = {}  # ticker -> {'shares': int, 'entry_price': float}
    portfolio_history = [initial_capital]
    transaction_costs = 0.0

    for day_idx, current_date in enumerate(trading_days):
        # Verbose on day 1, 10, 20, 30, etc.
        is_verbose_day = (day_idx == 0) or ((day_idx + 1) % 10 == 0)

        # Get current stock selections
        try:
            new_stocks = select_func(
                initial_top_tickers, ticker_data_grouped, current_date, portfolio_size,
                verbose=is_verbose_day
            )
        except TypeError:
            # Fallback for strategies that don't support verbose parameter
            new_stocks = select_func(
                initial_top_tickers, ticker_data_grouped, current_date, portfolio_size
            )
        except Exception as e:
            new_stocks = list(positions.keys())  # Keep current positions

        # Print progress
        if is_verbose_day:
            portfolio_value = cash + sum(
                positions[t]['shares'] * ticker_data_grouped.get(t, {}).get('Close', pd.Series([0])).iloc[-1]
                for t in positions if t in ticker_data_grouped
            )
            ret = (portfolio_value / initial_capital - 1) * 100 if initial_capital > 0 else 0
            print(f"   📅 {strategy_name} Day {day_idx + 1}: ${portfolio_value:,.0f} ({ret:+.1f}%) Holdings: {list(positions.keys())[:5]}", flush=True)
            sys.stdout.flush()

        current_stocks = list(positions.keys())

        # Rebalance: Sell stocks not in new list
        for ticker in list(positions.keys()):
            if ticker not in new_stocks:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    valid_prices = ticker_df[ticker_df.index <= current_date]['Close'].dropna()
                    if len(valid_prices) > 0:
                        sell_price = valid_prices.iloc[-1]
                        shares = positions[ticker]['shares']
                        proceeds = shares * sell_price
                        cost = proceeds * TRANSACTION_COST
                        cash += proceeds - cost
                        transaction_costs += cost
                        del positions[ticker]

        # Buy new stocks
        stocks_to_buy = [s for s in new_stocks if s not in positions]
        if stocks_to_buy and cash > 1000:
            capital_per_stock = cash / len(stocks_to_buy)
            for ticker in stocks_to_buy:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    valid_prices = ticker_df[ticker_df.index <= current_date]['Close'].dropna()
                    if len(valid_prices) > 0:
                        buy_price = valid_prices.iloc[-1]
                        if buy_price > 0:
                            shares = int(capital_per_stock / buy_price)
                            if shares > 0:
                                cost = shares * buy_price * TRANSACTION_COST
                                positions[ticker] = {
                                    'shares': shares,
                                    'entry_price': buy_price
                                }
                                cash -= (shares * buy_price + cost)
                                transaction_costs += cost

        # Calculate portfolio value
        invested_value = 0.0
        for ticker, pos in positions.items():
            ticker_df = ticker_data_grouped.get(ticker)
            if ticker_df is not None:
                valid_prices = ticker_df[ticker_df.index <= current_date]['Close'].dropna()
                if len(valid_prices) > 0:
                    current_price = valid_prices.iloc[-1]
                    invested_value += pos['shares'] * current_price

        portfolio_value = cash + invested_value
        portfolio_history.append(portfolio_value)

    # Final results
    final_value = portfolio_history[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate volatility (std dev of daily returns)
    daily_returns = pd.Series(portfolio_history).pct_change().dropna()
    volatility = daily_returns.std() * 100  # Daily std dev as %

    print(f"\n✅ {strategy_name} Results:")
    print(f"   Final Value: ${final_value:,.0f}")
    print(f"   Return: {total_return:+.1f}%")
    print(f"   Volatility: {volatility:.1f}%")
    print(f"   Transaction Costs: ${transaction_costs:,.0f}")
    print(f"   Final Positions: {list(positions.keys())}")

    return {
        'strategy': strategy_name,
        'final_value': final_value,
        'return': total_return,
        'volatility': volatility,
        'costs': transaction_costs,
        'history': portfolio_history,
        'positions': positions,
    }


def main():
    import sys
    print("=" * 80, flush=True)
    print("ULTIMATE STRATEGY TEST", flush=True)
    print("=" * 80, flush=True)

    # Load data
    all_tickers_data, initial_top_tickers = load_data_once()

    # Determine backtest period (same as output22.03.log)
    bt_end = datetime(2026, 3, 20, tzinfo=timezone.utc)
    bt_start = bt_end - timedelta(days=70)  # ~51 trading days

    print(f"\nBacktest period: {bt_start.date()} to {bt_end.date()}", flush=True)
    print(f"Portfolio size: {PORTFOLIO_SIZE}", flush=True)
    print(f"Initial capital: ${INITIAL_CAPITAL:,}", flush=True)
    sys.stdout.flush()

    # Import strategy functions
    print("\nImporting strategies...", flush=True)
    import sys
    sys.stdout.flush()

    print("  - Importing ultimate_strategy...", flush=True)
    from ultimate_strategy import select_ultimate_stocks
    print("  - Importing risk_adj_mom_1m_vol_sweet_strategy...", flush=True)
    from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
    print("  - Importing vol_sweet_mom_strategy...", flush=True)
    from vol_sweet_mom_strategy import select_vol_sweet_mom_stocks
    print("  - Importing multi_timeframe_ensemble...", flush=True)
    from multi_timeframe_ensemble import select_multi_timeframe_stocks
    print("  - Importing sector_rotation...", flush=True)
    from shared_strategies import select_sector_rotation_etfs
    print("  - Importing bb_squeeze...", flush=True)
    from bollinger_bands_strategy import select_bb_squeeze_breakout_stocks
    print("  - Importing risk_adj_mom_3m_market_up...", flush=True)
    from risk_adj_mom_3m_market_up_strategy import select_risk_adj_mom_3m_market_up_stocks
    print("  - Importing inverse_etf_hedge...", flush=True)
    from inverse_etf_hedge_strategy import select_inverse_etf_hedge_stocks
    print("  - Importing select_top_performers (BH 1Y)...", flush=True)
    from shared_strategies import select_top_performers
    print("✅ All strategies imported!", flush=True)

    # Wrapper for BH 1Y (365-day lookback)
    def select_bh_1y_stocks(all_tickers, ticker_data_grouped, current_date, top_n, verbose=True):
        stocks = select_top_performers(all_tickers, ticker_data_grouped, current_date,
                                       lookback_days=365, top_n=top_n)
        if verbose:
            print(f"   📊 BH 1Y: Selected {stocks[:5]}...")
        return stocks

    strategies = [
        ("Ultimate", select_ultimate_stocks),
        ("Multi-TF Ensemble", select_multi_timeframe_stocks),
        ("1M VolSweet", select_risk_adj_mom_1m_vol_sweet_stocks),
        ("VolSweet Mom", select_vol_sweet_mom_stocks),
        ("Sector Rotation", select_sector_rotation_etfs),
        ("BB Squeeze", select_bb_squeeze_breakout_stocks),
        ("RiskAdj 3M Up", select_risk_adj_mom_3m_market_up_stocks),
        ("Inv ETF Hedge", select_inverse_etf_hedge_stocks),
        ("BH 1Y", select_bh_1y_stocks),
    ]

    # PRE-GROUP DATA ONCE (major speedup - avoid doing this 9x in parallel)
    import sys
    print("\n📊 Pre-grouping ticker data (one-time)...", flush=True)
    sys.stdout.flush()

    grouped = all_tickers_data.groupby('ticker')
    ticker_data_grouped = {}
    for ticker in initial_top_tickers:
        if ticker in grouped.groups:
            ticker_df = grouped.get_group(ticker).copy()
            if len(ticker_df) > 0:
                ticker_df = ticker_df.set_index('date').sort_index()
                # Resample to daily (use last price of each day)
                ticker_df = ticker_df.resample('D').last().dropna(subset=['Close'])
                if len(ticker_df) > 0:
                    ticker_data_grouped[ticker] = ticker_df

    # Pre-calculate trading days
    all_dates = set()
    for ticker, df in ticker_data_grouped.items():
        all_dates.update(df.index.tolist())
    trading_days = sorted([d for d in all_dates if bt_start <= d <= bt_end])

    print(f"   ✅ Grouped {len(ticker_data_grouped)} tickers, {len(trading_days)} trading days", flush=True)
    sys.stdout.flush()

    # Run strategies in PARALLEL for speed
    results = []
    print_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable counter in closure

    started_count = [0]

    def run_strategy_wrapper(strategy_tuple):
        """Wrapper to run a single strategy and handle output."""
        name, func = strategy_tuple
        with print_lock:
            started_count[0] += 1
            print(f"   🔄 Starting {name}... ({started_count[0]}/{len(strategies)})", flush=True)
            sys.stdout.flush()
        try:
            result = run_simple_backtest(
                name, func, all_tickers_data, initial_top_tickers,
                bt_start, bt_end, PORTFOLIO_SIZE, INITIAL_CAPITAL,
                ticker_data_grouped=ticker_data_grouped,
                trading_days=trading_days
            )
            with print_lock:
                completed_count[0] += 1
                print(f"   ✅ [{completed_count[0]}/{len(strategies)}] {name} complete: {result['return']:+.1f}%", flush=True)
                sys.stdout.flush()
            return result
        except Exception as e:
            import traceback
            with print_lock:
                completed_count[0] += 1
                print(f"   ❌ [{completed_count[0]}/{len(strategies)}] {name} FAILED: {e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
            return {
                'strategy': name,
                'final_value': INITIAL_CAPITAL,
                'return': 0.0,
                'volatility': 0.0,
                'costs': 0.0,
            }

    # Determine number of parallel workers (limit to avoid memory issues)
    max_workers = min(len(strategies), 4)  # Max 4 parallel backtests

    print(f"\n🚀 Starting backtests for {len(strategies)} strategies in PARALLEL ({max_workers} workers)...", flush=True)
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all strategies
        futures = {executor.submit(run_strategy_wrapper, s): s[0] for s in strategies}

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            sys.stdout.flush()

    print(f"\n✅ All {len(results)} strategies completed!", flush=True)
    sys.stdout.flush()

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Final Value':<15} {'Return':<12} {'Volatility':<12} {'Costs'}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['return'], reverse=True):
        print(f"{r['strategy']:<20} ${r['final_value']:>12,.0f} {r['return']:>+10.1f}% {r['volatility']:>10.1f}% ${r['costs']:>8,.0f}")

    # Winner
    winner = max(results, key=lambda x: x['return'])
    print(f"\n🏆 BEST STRATEGY: {winner['strategy']} → {winner['return']:+.1f}%")


if __name__ == "__main__":
    main()
