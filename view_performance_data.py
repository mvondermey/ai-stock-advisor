#!/usr/bin/env python3
"""View the saved performance data."""

import json
from datetime import datetime

print('=' * 80)
print('📊 VIEWING SAVED PERFORMANCE DATA')
print('=' * 80)

# Load all performances
try:
    with open('all_1y_performances.json', 'r') as f:
        all_data = json.load(f)
    
    print(f"\n📅 Data Range: {all_data['data_range']['start']} to {all_data['data_range']['end']}")
    print(f"📊 Total Stocks: {all_data['total_stocks']}")
    print(f"💾 Saved At: {all_data['saved_at']}")
    
    # Show top 20 performers
    performances = all_data['performance']
    sorted_perf = sorted(performances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 20 PERFORMERS:")
    print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12}")
    print("-" * 35)
    for i, (ticker, ret) in enumerate(sorted_perf[:20], 1):
        print(f"{i:<6} {ticker:<10} {ret:<12.1f}%")
    
    # Show bottom 10
    if len(sorted_perf) > 20:
        print(f"\n📉 BOTTOM 10 PERFORMERS:")
        print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12}")
        print("-" * 35)
        for i, (ticker, ret) in enumerate(sorted_perf[-10:], len(sorted_perf)-9):
            print(f"{i:<6} {ticker:<10} {ret:<12.1f}%")
    
except FileNotFoundError:
    print("❌ all_1y_performances.json not found")
    print("   Run save_and_retrieve_performance.py first")

print("\n" + "=" * 80)

# Load your stocks performance
try:
    with open('your_stocks_performance.json', 'r') as f:
        your_data = json.load(f)
    
    print("📊 YOUR STOCKS PERFORMANCE:")
    print(f"{'Name':<35} {'Ticker':<10} {'1Y Return':<12}")
    print("-" * 65)
    
    for name, data in your_data.items():
        print(f"{name:<35} {data['ticker']:<10} {data['return_1y']:<12.1f}%")
    
    # Statistics
    returns = [data['return_1y'] for data in your_data.values()]
    avg_return = sum(returns) / len(returns)
    best = max(your_data.items(), key=lambda x: x[1]['return_1y'])
    worst = min(your_data.items(), key=lambda x: x[1]['return_1y'])
    
    print(f"\n📈 Your Stocks Statistics:")
    print(f"   Average: {avg_return:.1f}%")
    print(f"   Best: {best[0]} ({best[1]['return_1y']:.1f}%)")
    print(f"   Worst: {worst[0]} ({worst[1]['return_1y']:.1f}%)")
    
except FileNotFoundError:
    print("❌ your_stocks_performance.json not found")
    print("   Run save_and_retrieve_performance.py first")

print("\n" + "=" * 80)
print("💡 USAGE:")
print("=" * 80)
print("""
You can now:
1. Access all performances from all_1y_performances.json
2. Access your specific stocks from your_stocks_performance.json
3. Use this data for analysis, comparison, or reporting

The data is saved in JSON format for easy integration with other tools.
""")
