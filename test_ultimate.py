#!/usr/bin/env python3
"""
Test Ultimate Strategy - Simple backtest to compare against top performers.
Comparing against: Multi-TF Ensemble (+41.2%), VolSweet Mom (+34.2%), BH 3M Monthly (+23.5%),
                   Rebal 1Y VolAdj (+20.6%), 1M VolSweet (+28.3%)
"""
import sys
sys.path.insert(0, '/home/mvondermey/ai-stock-advisor/src')

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# Parameters from output22.03.log
PORTFOLIO_SIZE = 10
BACKTEST_DAYS = 90  # ~65 trading days
INITIAL_CAPITAL = 300000
TRANSACTION_COST = 0.001  # 0.1%

def load_data():
    """Load ticker data from cache."""
    cache_dir = Path('/home/mvondermey/ai-stock-advisor/data_cache')
    print("Loading data from cache...")

    ticker_data = {}
    for f in cache_dir.glob('*.csv'):
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'Close' in df.columns and len(df) > 100:
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize('UTC')
                ticker_data[f.stem] = df
        except:
            pass

    print(f"Loaded {len(ticker_data)} tickers")
    return ticker_data

def is_first_trading_day_of_month(current_date, prev_date):
    """Check if current_date is the first trading day of a new month."""
    if prev_date is None:
        return True
    return current_date.month != prev_date.month

def get_vol_adjusted_rebalance_days(ticker_data, current_date):
    """Calculate rebalance frequency based on market volatility."""
    market_ticker = None
    for t in ['SPY', 'QQQ', 'VOO', 'IVV']:
        if t in ticker_data:
            market_ticker = t
            break

    if not market_ticker:
        return 20

    try:
        data = ticker_data[market_ticker].loc[:current_date]
        if len(data) < 20:
            return 20

        returns = data['Close'].pct_change().dropna()
        if len(returns) < 20:
            return 20

        vol = returns.std() * np.sqrt(252)

        # Config values from config.py
        low_thresh = 0.15
        high_thresh = 0.35
        min_days = 5
        max_days = 30

        if vol <= low_thresh:
            return max_days  # Low vol = rebalance less often
        elif vol >= high_thresh:
            return min_days  # High vol = rebalance more often
        else:
            ratio = (vol - low_thresh) / (high_thresh - low_thresh)
            return int(max_days - ratio * (max_days - min_days))
    except:
        return 20

def run_backtest(strategy_name, select_func, ticker_data, bt_start, bt_end, rebal_freq='daily'):
    """
    Run backtest with configurable rebalancing frequency.

    Args:
        rebal_freq: 'daily', 'monthly', or integer for days between rebalancing
    """
    all_tickers = list(ticker_data.keys())

    # Get trading days
    all_dates = set()
    for df in ticker_data.values():
        for idx in df.index:
            if hasattr(idx, 'date'):
                all_dates.add(idx.date())
            else:
                all_dates.add(pd.Timestamp(idx).date())

    start_date = bt_start.date() if hasattr(bt_start, 'date') else bt_start
    end_date = bt_end.date() if hasattr(bt_end, 'date') else bt_end
    trading_days = sorted([d for d in all_dates if start_date <= d <= end_date])

    print(f"\n{'='*60}")
    print(f"📊 {strategy_name}: {len(trading_days)} trading days (rebal: {rebal_freq})")
    print(f"{'='*60}")

    cash = INITIAL_CAPITAL
    positions = {}
    history = [INITIAL_CAPITAL]
    costs = 0.0
    current_stocks = []
    last_rebal_day = None
    days_since_rebal = 0

    for day_idx, current_date in enumerate(trading_days):
        current_dt = pd.Timestamp(current_date).tz_localize('UTC')
        prev_date = trading_days[day_idx - 1] if day_idx > 0 else None

        # Determine if we should rebalance
        should_rebalance = False
        if rebal_freq == 'daily':
            should_rebalance = True
        elif rebal_freq == 'monthly':
            should_rebalance = is_first_trading_day_of_month(current_date, prev_date)
        elif rebal_freq == 'vol_adjusted':
            days_since_rebal += 1
            rebal_days = get_vol_adjusted_rebalance_days(ticker_data, current_dt)
            if last_rebal_day is None or days_since_rebal >= rebal_days:
                should_rebalance = True
        elif isinstance(rebal_freq, int):
            days_since_rebal += 1
            if last_rebal_day is None or days_since_rebal >= rebal_freq:
                should_rebalance = True

        # Get new stock selection only on rebalance days
        if should_rebalance:
            try:
                new_stocks = select_func(all_tickers, ticker_data, current_dt, PORTFOLIO_SIZE)
                if new_stocks:
                    current_stocks = new_stocks
                    last_rebal_day = current_date
                    days_since_rebal = 0
            except Exception as e:
                if day_idx == 0:
                    print(f"   ⚠️ Selection error: {e}")

        new_stocks = current_stocks

        # Sell stocks not in new list
        for ticker in list(positions.keys()):
            if ticker not in new_stocks:
                df = ticker_data.get(ticker)
                if df is not None:
                    prices = df[df.index.date <= current_date]['Close'].dropna()
                    if len(prices) > 0:
                        price = prices.iloc[-1]
                        proceeds = positions[ticker]['shares'] * price
                        cost = proceeds * TRANSACTION_COST
                        cash += proceeds - cost
                        costs += cost
                        del positions[ticker]

        # Buy new stocks
        to_buy = [s for s in new_stocks if s not in positions]
        if to_buy and cash > 1000:
            per_stock = cash / len(to_buy)
            for ticker in to_buy:
                df = ticker_data.get(ticker)
                if df is not None:
                    prices = df[df.index.date <= current_date]['Close'].dropna()
                    if len(prices) > 0:
                        price = prices.iloc[-1]
                        if price > 0:
                            shares = int(per_stock / price)
                            if shares > 0:
                                cost = shares * price * TRANSACTION_COST
                                positions[ticker] = {'shares': shares, 'price': price}
                                cash -= (shares * price + cost)
                                costs += cost

        # Calculate value
        invested = sum(
            pos['shares'] * ticker_data[t][ticker_data[t].index.date <= current_date]['Close'].dropna().iloc[-1]
            for t, pos in positions.items()
            if t in ticker_data and len(ticker_data[t][ticker_data[t].index.date <= current_date]['Close'].dropna()) > 0
        )
        value = cash + invested
        history.append(value)

        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            ret = (value / INITIAL_CAPITAL - 1) * 100
            date_str = str(current_date)
            print(f"   Day {day_idx+1}/{len(trading_days)} ({date_str}): ${value:,.0f} ({ret:+.1f}%)")

    final = history[-1]
    ret = (final / INITIAL_CAPITAL - 1) * 100
    vol = pd.Series(history).pct_change().dropna().std() * 100

    print(f"\n✅ {strategy_name} FINAL: ${final:,.0f} ({ret:+.1f}%) Vol:{vol:.1f}%")
    print(f"   Holdings: {list(positions.keys())}")

    return {'name': strategy_name, 'value': final, 'return': ret, 'vol': vol, 'costs': costs}

def main():
    print("=" * 70)
    print("ULTIMATE STRATEGY TEST")
    print("Target: Beat Multi-TF Ensemble (+41.2%)")
    print("=" * 70)

    ticker_data = load_data()

    # Same period as output22.03.log
    bt_end = datetime(2026, 3, 20, tzinfo=timezone.utc)
    bt_start = bt_end - timedelta(days=BACKTEST_DAYS)

    print(f"\nPeriod: {bt_start.date()} to {bt_end.date()}")
    print(f"Portfolio: {PORTFOLIO_SIZE} stocks, Capital: ${INITIAL_CAPITAL:,}")

    # Import strategies
    from ultimate_strategy import select_ultimate_stocks
    from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
    from vol_sweet_mom_strategy import select_vol_sweet_mom_stocks
    from multi_timeframe_ensemble import select_multi_timeframe_stocks
    from shared_strategies import select_top_performers

    results = []

    # Test Ultimate - daily rebalancing
    results.append(run_backtest("Ultimate", select_ultimate_stocks, ticker_data, bt_start, bt_end, rebal_freq='daily'))

    # Test competitors - daily rebalancing strategies
    results.append(run_backtest("1M VolSweet", select_risk_adj_mom_1m_vol_sweet_stocks, ticker_data, bt_start, bt_end, rebal_freq='daily'))
    results.append(run_backtest("VolSweet Mom", select_vol_sweet_mom_stocks, ticker_data, bt_start, bt_end, rebal_freq='daily'))
    results.append(run_backtest("Multi-TF", select_multi_timeframe_stocks, ticker_data, bt_start, bt_end, rebal_freq='daily'))

    # Monthly rebalancing strategies (only rebalance at start of month)
    results.append(run_backtest("BH 3M Monthly", lambda t, d, dt, n: select_top_performers(t, d, dt, 90, n), ticker_data, bt_start, bt_end, rebal_freq='monthly'))

    # Rebal 1Y VolAdj - volatility-adjusted rebalancing (5-30 days based on market vol)
    results.append(run_backtest("Rebal 1Y VolAdj", lambda t, d, dt, n: select_top_performers(t, d, dt, 252, n), ticker_data, bt_start, bt_end, rebal_freq='vol_adjusted'))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<18} {'Value':<15} {'Return':<12} {'Vol':<10} {'Costs'}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x['return'], reverse=True):
        print(f"{r['name']:<18} ${r['value']:>12,.0f} {r['return']:>+10.1f}% {r['vol']:>8.1f}% ${r['costs']:>8,.0f}")

    winner = max(results, key=lambda x: x['return'])
    print(f"\n🏆 WINNER: {winner['name']} → {winner['return']:+.1f}%")

    # Check if Ultimate beats all other strategies
    ultimate = next(r for r in results if r['name'] == 'Ultimate')
    other_strategies = [r for r in results if r['name'] != 'Ultimate']
    best_other = max(other_strategies, key=lambda x: x['return'])

    if ultimate['return'] >= best_other['return']:
        print(f"✅ SUCCESS! Ultimate ({ultimate['return']:+.1f}%) beats {best_other['name']} ({best_other['return']:+.1f}%)")
    else:
        print(f"❌ Need improvement: Ultimate ({ultimate['return']:+.1f}%) vs {best_other['name']} ({best_other['return']:+.1f}%)")
        print(f"   Gap: {best_other['return'] - ultimate['return']:.1f}%")

if __name__ == "__main__":
    main()
