#!/usr/bin/env python3
"""
Portfolio Parameter Test - BH 1M, BH 3M, BH 1Y with Transaction Costs and Buffer
"""
import sys
sys.path.insert(0, '/home/mvondermey/ai-stock-advisor/src')

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path('/home/mvondermey/ai-stock-advisor/data_cache')
BACKTEST_DAYS = 90
TX_COST_PCT = 0.011  # 1.1% per trade

# (size, buffer) combinations
PARAM_COMBINATIONS = [(3, 4), (3, 5), (5, 6), (5, 7), (10, 12)]

STRATEGIES = [
    ("BH 1Y", 365, 30),   # Monthly rebalance
    ("BH 3M", 90, 14),    # Bi-weekly rebalance
    ("BH 1M", 30, 7),     # Weekly rebalance
]

def load_data():
    ticker_data = {}
    cache_files = list(CACHE_DIR.glob('*.csv'))
    print(f"Loading {len(cache_files)} cached tickers...")
    for f in cache_files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'Close' in df.columns and len(df) > 50:
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize('UTC')
                ticker_data[f.stem] = df
        except:
            pass
    print(f"Loaded {len(ticker_data)} tickers")
    return ticker_data

def run_test(ticker_data, size, buffer, lookback_days, rebal_days):
    end_date = datetime.now(timezone.utc)
    bt_start = end_date - timedelta(days=BACKTEST_DAYS)

    # Get top performers at start
    perfs = []
    for ticker, df in ticker_data.items():
        df_b = df[df.index <= bt_start]
        if len(df_b) < 30:
            continue
        idx = df_b.index.searchsorted(bt_start - timedelta(days=lookback_days))
        if idx >= len(df_b) - 10:
            continue
        p = (df_b['Close'].iloc[-1] - df_b['Close'].iloc[idx]) / df_b['Close'].iloc[idx]
        perfs.append((ticker, p))

    perfs.sort(key=lambda x: x[1], reverse=True)
    selected = [t[0] for t in perfs[:size]]

    if not selected:
        return {'net_return': 0, 'sharpe': 0, 'trades': 0, 'tx_cost': 0}

    # Calculate returns
    returns, daily_rets = [], []
    for ticker in selected:
        df = ticker_data[ticker]
        df_bt = df[(df.index >= bt_start) & (df.index <= end_date)]
        if len(df_bt) > 5:
            ret = (df_bt['Close'].iloc[-1] - df_bt['Close'].iloc[0]) / df_bt['Close'].iloc[0]
            returns.append(ret)
            daily_rets.append(df_bt['Close'].pct_change().dropna())

    if not returns:
        return {'net_return': 0, 'sharpe': 0, 'trades': 0, 'tx_cost': 0}

    gross_return = np.mean(returns) * 100

    # Transaction cost calculation WITH BUFFER EFFECT
    num_rebalances = BACKTEST_DAYS // rebal_days
    # Buffer reduces turnover: higher buffer ratio = less turnover
    buffer_ratio = buffer / size
    # Turnover estimate: if buffer = size (ratio=1), ~50% turnover; if buffer = 2*size (ratio=2), ~10% turnover
    turnover = max(0.1, 0.5 - (buffer_ratio - 1) * 0.4)
    trades = size + num_rebalances * turnover * size * 2
    tx_cost = trades * TX_COST_PCT
    net_return = gross_return - tx_cost

    # Sharpe & MaxDD
    if daily_rets:
        combined = pd.concat(daily_rets, axis=1).mean(axis=1).dropna()
        if len(combined) > 10:
            sharpe = (combined.mean() / combined.std()) * np.sqrt(252) if combined.std() > 0 else 0
            cumul = (1 + combined).cumprod()
            max_dd = ((cumul - cumul.expanding().max()) / cumul.expanding().max()).min() * 100
        else:
            sharpe, max_dd = 0, 0
    else:
        sharpe, max_dd = 0, 0

    return {
        'gross_return': gross_return, 'net_return': net_return,
        'sharpe': sharpe, 'max_dd': max_dd,
        'trades': int(trades), 'tx_cost': tx_cost, 'top3': selected[:3]
    }

def main():
    print("=" * 110)
    print("PORTFOLIO TEST: BH 1Y vs BH 3M vs BH 1M (with 1.1% transaction costs + BUFFER)")
    print("=" * 110)

    ticker_data = load_data()
    results = []

    for name, lookback, rebal in STRATEGIES:
        print(f"Testing {name} (rebalance every {rebal} days)...")
        for size, buffer in PARAM_COMBINATIONS:
            r = run_test(ticker_data, size, buffer, lookback, rebal)
            r['strategy'], r['size'], r['buffer'] = name, size, buffer
            results.append(r)

    # All results
    print("\n" + "=" * 110)
    print("ALL RESULTS - sorted by NET RETURN")
    print("=" * 110)
    print(f"{'Strategy':<10} {'Size':<6} {'Buffer':<8} {'Gross':<10} {'TxCost':<10} {'NET':<10} {'Sharpe':<8} {'Trades':<8} {'MaxDD'}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: x.get('net_return', 0), reverse=True):
        print(f"{r['strategy']:<10} {r['size']:<6} {r['buffer']:<8} {r.get('gross_return', 0):>+7.1f}%   {r.get('tx_cost', 0):>6.1f}%    {r.get('net_return', 0):>+7.1f}%   {r.get('sharpe', 0):>6.2f}   {r.get('trades', 0):<8} {r.get('max_dd', 0):>6.1f}%")

    # ============ SUMMARY TABLE ============
    print("\n")
    print("=" * 110)
    print("                           SUMMARY COMPARISON TABLE")
    print("=" * 110)
    print(f"{'Strategy':<12} | {'Best Size':<10} | {'Buffer':<8} | {'Gross Return':<14} | {'Tx Cost':<10} | {'NET Return':<12} | {'Sharpe':<8} | {'Trades'}")
    print("-" * 110)

    summary = []
    for name, _, _ in STRATEGIES:
        best = max([r for r in results if r['strategy'] == name], key=lambda x: x.get('net_return', 0))
        summary.append(best)
        print(f"{best['strategy']:<12} | Size={best['size']:<5} | {best['buffer']:<8} | {best.get('gross_return', 0):>+10.1f}%   | {best.get('tx_cost', 0):>7.1f}%  | {best.get('net_return', 0):>+9.1f}%   | {best.get('sharpe', 0):>6.2f}   | {best.get('trades', 0)}")

    print("-" * 110)

    # Winner
    winner = max(summary, key=lambda x: x.get('net_return', 0))
    best_sharpe = max(summary, key=lambda x: x.get('sharpe', 0))

    print(f"\n💰 BEST NET RETURN:  {winner['strategy']} (Size={winner['size']}, Buffer={winner['buffer']}) → NET {winner.get('net_return', 0):+.1f}%")
    print(f"🏆 BEST SHARPE:      {best_sharpe['strategy']} (Size={best_sharpe['size']}, Buffer={best_sharpe['buffer']}) → Sharpe {best_sharpe.get('sharpe', 0):.2f}")
    print("=" * 110)

if __name__ == "__main__":
    main()
