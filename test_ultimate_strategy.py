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
from pathlib import Path
import pandas as pd

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
                        backtest_start, backtest_end, portfolio_size, initial_capital):
    """Run a simple daily rebalancing backtest for a single strategy."""

    # Group data by ticker for fast lookups
    ticker_data_grouped = {}
    for ticker in initial_top_tickers:
        ticker_df = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
        if len(ticker_df) > 0:
            ticker_df = ticker_df.set_index('date').sort_index()
            ticker_data_grouped[ticker] = ticker_df

    # Get trading days
    all_dates = set()
    for ticker, df in ticker_data_grouped.items():
        all_dates.update(df.index.tolist())

    trading_days = sorted([d for d in all_dates if backtest_start <= d <= backtest_end])
    print(f"\n📊 {strategy_name}: Running {len(trading_days)} trading days")

    # Initialize portfolio
    cash = initial_capital
    positions = {}  # ticker -> {'shares': int, 'entry_price': float}
    portfolio_history = [initial_capital]
    transaction_costs = 0.0

    for day_idx, current_date in enumerate(trading_days):
        # Get current stock selections
        try:
            new_stocks = select_func(
                initial_top_tickers, ticker_data_grouped, current_date, portfolio_size
            )
        except Exception as e:
            print(f"   ⚠️ Error on day {day_idx + 1}: {e}")
            new_stocks = list(positions.keys())  # Keep current positions

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

        # Print progress every 10 days
        if (day_idx + 1) % 10 == 0 or day_idx == len(trading_days) - 1:
            ret = (portfolio_value / initial_capital - 1) * 100
            print(f"   Day {day_idx + 1}: ${portfolio_value:,.0f} ({ret:+.1f}%) - Holdings: {list(positions.keys())[:5]}...")

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
    print("=" * 80)
    print("ULTIMATE STRATEGY TEST")
    print("=" * 80)

    # Load data
    all_tickers_data, initial_top_tickers = load_data_once()

    # Determine backtest period (same as output22.03.log)
    bt_end = datetime(2026, 3, 20, tzinfo=timezone.utc)
    bt_start = bt_end - timedelta(days=70)  # ~51 trading days

    print(f"\nBacktest period: {bt_start.date()} to {bt_end.date()}")
    print(f"Portfolio size: {PORTFOLIO_SIZE}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}")

    # Import strategy functions
    from ultimate_strategy import select_ultimate_stocks
    from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
    from vol_sweet_mom_strategy import select_vol_sweet_mom_stocks

    strategies = [
        ("Ultimate", select_ultimate_stocks),
        ("1M VolSweet", select_risk_adj_mom_1m_vol_sweet_stocks),
        ("VolSweet Mom", select_vol_sweet_mom_stocks),
    ]

    results = []
    for name, func in strategies:
        result = run_simple_backtest(
            name, func, all_tickers_data, initial_top_tickers,
            bt_start, bt_end, PORTFOLIO_SIZE, INITIAL_CAPITAL
        )
        results.append(result)

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
