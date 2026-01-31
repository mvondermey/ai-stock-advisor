#!/usr/bin/env python3
"""View the latest live trading performance data."""

import json
from datetime import datetime

print('=' * 80)
print('📊 VIEWING LIVE TRADING PERFORMANCE DATA')
print('=' * 80)

try:
    # Load the latest live trading performances
    with open('latest_live_trading_performances.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n📅 Data Range: {data['data_range']['start']} to {data['data_range']['end']}")
    print(f"📊 Total Stocks: {data['total_stocks']}")
    print(f"📊 Processed: {data['processed']}")
    print(f"❌ Errors: {data['errors']}")
    print(f"🎯 Strategy: {data['strategy']}")
    print(f"💾 Saved At: {data['saved_at']}")
    
    # Get performances
    performances = data['performance']
    detailed = data['detailed_performance']
    
    # Sort by performance
    sorted_perf = sorted(performances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 20 PERFORMERS:")
    print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12} {'Score':<8} {'Volatility':<12}")
    print("-" * 55)
    
    for i, (ticker, return_pct) in enumerate(sorted_perf[:20], 1):
        details = detailed[ticker]
        print(f"{i:<6} {ticker:<10} {return_pct:<12.1f}% {details['score']:<8.2f} {details['volatility']:<12.1f}%")
    
    # Bottom performers
    if len(sorted_perf) > 20:
        print(f"\n📉 BOTTOM 10 PERFORMERS:")
        print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12} {'Score':<8} {'Volatility':<12}")
        print("-" * 55)
        
        for i, (ticker, return_pct) in enumerate(sorted_perf[-10:], len(sorted_perf)-9):
            details = detailed[ticker]
            print(f"{i:<6} {ticker:<10} {return_pct:<12.1f}% {details['score']:<8.2f} {details['volatility']:<12.1f}%")
    
    # Statistics
    returns = list(performances.values())
    if returns:
        avg_return = sum(returns) / len(returns)
        positive = len([r for r in returns if r > 0])
        negative = len([r for r in returns if r < 0])
        
        print(f"\n📊 STATISTICS:")
        print(f"   Average return: {avg_return:.1f}%")
        print(f"   Positive returns: {positive} ({positive/len(returns)*100:.1f}%)")
        print(f"   Negative returns: {negative} ({negative/len(returns)*100:.1f}%)")
        print(f"   Best performer: {sorted_perf[0][0]} ({sorted_perf[0][1]:.1f}%)")
        print(f"   Worst performer: {sorted_perf[-1][0]} ({sorted_perf[-1][1]:.1f}%)")
    
    # Check for your specific stocks
    your_stocks = ['HOOD', 'AVGO', 'CRZBY', 'WBD', 'HOT.DE', 'APP', 'ADBE', 'RWE.DE', 'RHM.DE', 'URTH',
                   'STX', 'MU', 'ENR.DE', 'META', 'PLTR', 'HWM', 'NEM', 'BIL.DE', 'TPR', 'NDX1.DE',
                   'LRCX', 'WDC', 'ATRO', 'EXS2.DE']
    
    print(f"\n📊 YOUR STOCKS PERFORMANCE:")
    print(f"{'Name':<35} {'Ticker':<10} {'1Y Return':<12} {'Status':<10}")
    print("-" * 75)
    
    found_count = 0
    for ticker in your_stocks:
        if ticker in performances:
            return_pct = performances[ticker]
            status = "✅ Found"
            print(f"{ticker:<35} {ticker:<10} {return_pct:<12.1f}% {status:<10}")
            found_count += 1
        else:
            print(f"{ticker:<35} {ticker:<10} {'No data':<12} {'❌ Missing':<10}")
    
    print(f"\n📈 Your Stocks Summary:")
    print(f"   Found: {found_count}/{len(your_stocks)}")
    
    if found_count > 0:
        your_returns = [performances[t] for t in your_stocks if t in performances]
        your_avg = sum(your_returns) / len(your_returns)
        your_best = max([(t, performances[t]) for t in your_stocks if t in performances], key=lambda x: x[1])
        your_worst = min([(t, performances[t]) for t in your_stocks if t in performances], key=lambda x: x[1])
        
        print(f"   Average: {your_avg:.1f}%")
        print(f"   Best: {your_best[0]} ({your_best[1]:.1f}%)")
        print(f"   Worst: {your_worst[0]} ({your_worst[1]:.1f}%)")

except FileNotFoundError:
    print("❌ No live trading performance data found")
    print("   Run live trading first: python src/main.py --live-trading --strategy risk_adj_mom")
except Exception as e:
    print(f"❌ Error loading data: {e}")

print("\n" + "=" * 80)
print("💡 USAGE:")
print("=" * 80)
print("""
The live trading performance data is saved with timestamp:
- latest_live_trading_performances.json (always the latest)
- live_trading_1y_performances_YYYYMMDD_HHMMSS.json (timestamped)

Run this script after live trading to see the latest performance data.
""")
