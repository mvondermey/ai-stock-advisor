#!/usr/bin/env python3
"""Corrected analysis of performance data."""

import json

print('=' * 80)
print('📊 CORRECTED PERFORMANCE ANALYSIS')
print('=' * 80)

# Load the saved data
with open('all_1y_performances.json', 'r') as f:
    data = json.load(f)

performances = data['performance']

# Sort by actual performance
sorted_perf = sorted(performances.items(), key=lambda x: x[1], reverse=True)

print(f"\n🏆 ACTUAL TOP 10 PERFORMERS:")
print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12}")
print("-" * 35)

for i, (ticker, ret) in enumerate(sorted_perf[:10], 1):
    print(f"{i:<6} {ticker:<10} {ret:<12.1f}%")

print(f"\n❌ CORRECTION NEEDED:")
print(f"   Previously stated APP (316.4%) was best performer")
print(f"   Actually CVNA (609.7%) is the best performer")
print(f"   APP is #3, not #1")

print(f"\n📊 TRUE TOP 3:")
print(f"   1. CVNA: {sorted_perf[0][1]:.1f}%")
print(f"   2. MSTR: {sorted_perf[1][1]:.1f}%")
print(f"   3. APP: {sorted_perf[2][1]:.1f}%")

# Load your stocks data
with open('your_stocks_performance.json', 'r') as f:
    your_data = json.load(f)

print(f"\n📊 YOUR STOCKS (CORRECTED RANKING):")
print(f"{'Name':<35} {'Ticker':<10} {'1Y Return':<12} {'Market Rank':<12}")
print("-" * 75)

# Get market rank for each of your stocks
market_ranks = {ticker: i+1 for i, (ticker, _) in enumerate(sorted_perf)}

for name, stock_data in your_data.items():
    ticker = stock_data['ticker']
    return_pct = stock_data['return_1y']
    rank = market_ranks.get(ticker, 'N/A')
    rank_str = f"#{rank}" if rank != 'N/A' else 'N/A'
    print(f"{name:<35} {ticker:<10} {return_pct:<12.1f}% {rank_str:<12}")

# Correct statistics
your_returns = [data['return_1y'] for data in your_data.values()]
avg_return = sum(your_returns) / len(your_returns)
best_your = max(your_data.items(), key=lambda x: x[1]['return_1y'])
worst_your = min(your_data.items(), key=lambda x: x[1]['return_1y'])

print(f"\n📈 Your Stocks Statistics:")
print(f"   Average: {avg_return:.1f}%")
print(f"   Best: {best_your[0]} ({best_your[1]['return_1y']:.1f}%)")
print(f"   Worst: {worst_your[0]} ({worst_your[1]['return_1y']:.1f}%)")

print(f"\n🎯 MARKET COMPARISON:")
print(f"   CVNA (market #1): {performances['CVNA']:.1f}%")
print(f"   Your best (APP): {best_your[1]['return_1y']:.1f}%")
print(f"   Difference: {performances['CVNA'] - best_your[1]['return_1y']:.1f}%")
