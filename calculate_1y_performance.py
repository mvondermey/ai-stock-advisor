#!/usr/bin/env python3
"""Calculate 1-year performance for all holdings."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datetime import datetime, timedelta
from data_fetcher import load_prices_robust

# All holdings (both lists combined)
all_holdings = [
    # From first list
    'AFRM', 'DHI', 'DASH', 'GE', 'HLT', 'NKE', 'CHTR', 'NVDA', 'PANW', 'PEP', 
    'RIO', 'RCL', 'AVGO', 'SEM', 'SPY', 'STX', 'WFC', 'ENB', 'DSCSY', 'FTNT',
    # From second list
    'HOOD', 'CRZBY', 'WBD', 'HOT.DE', 'APP', 'ADBE', 'RWE.DE', 'RHM.DE', 'URTH',
    'MU', 'ENR.DE', 'META', 'PLTR', 'HWM', 'NEM', 'BIL.DE', 'TPR', 'NDX1.DE',
    'LRCX', 'WDC', 'ATRO', 'EXS2.DE'
]

# Remove duplicates
all_holdings = list(set(all_holdings))

print('=' * 80)
print('📊 1-YEAR PERFORMANCE CALCULATION')
print('=' * 80)

# Download data for all holdings
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
print(f"\n📥 Calculating 1Y returns from {start_date.date()} to {end_date.date()}...")

results = []
for ticker in all_holdings:
    try:
        data = load_prices_robust(ticker, start_date, end_date)
        if not data.empty:
            # Find price column
            price_col = None
            for col in ['Close', 'Adj Close', 'Adj close', 'close']:
                if col in data.columns:
                    price_col = col
                    break
            
            if price_col and len(data) >= 10:
                start_price = data[price_col].iloc[0]
                end_price = data[price_col].iloc[-1]
                
                if start_price > 0:
                    return_pct = ((end_price - start_price) / start_price) * 100
                    results.append({
                        'ticker': ticker,
                        'return': return_pct,
                        'start_price': start_price,
                        'end_price': end_price,
                        'data_points': len(data)
                    })
                    print(f"   ✅ {ticker}: {return_pct:.1f}% (${start_price:.2f} → ${end_price:.2f})")
                else:
                    print(f"   ❌ {ticker}: Invalid price data")
            else:
                print(f"   ❌ {ticker}: No price column or insufficient data")
        else:
            print(f"   ❌ {ticker}: No data")
    except Exception as e:
        print(f"   ❌ {ticker}: Error - {e}")

# Sort by return
results.sort(key=lambda x: x['return'], reverse=True)

print("\n" + "=" * 80)
print("📈 PERFORMANCE RANKING")
print("=" * 80)

print(f"\n{'Rank':<5} {'Ticker':<10} {'1Y Return':<12} {'Start':<10} {'End':<10} {'Data':<6}")
print("-" * 80)

for i, r in enumerate(results, 1):
    print(f"{i:<5} {r['ticker']:<10} {r['return']:<12.1f} {r['start_price']:<10.2f} {r['end_price']:<10.2f} {r['data_points']:<6}")

# Summary statistics
if results:
    returns = [r['return'] for r in results]
    avg_return = sum(returns) / len(returns)
    positive = len([r for r in returns if r > 0])
    negative = len([r for r in returns if r < 0])
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal stocks analyzed: {len(results)}")
    print(f"Average return: {avg_return:.1f}%")
    print(f"Positive returns: {positive} ({positive/len(results)*100:.1f}%)")
    print(f"Negative returns: {negative} ({negative/len(results)*100:.1f}%)")
    print(f"Best performer: {results[0]['ticker']} ({results[0]['return']:.1f}%)")
    print(f"Worst performer: {results[-1]['ticker']} ({results[-1]['return']:.1f}%)")
    
    # Top 10 performers
    print(f"\n🏆 TOP 10 PERFORMERS:")
    for i, r in enumerate(results[:10], 1):
        print(f"   {i:2d}. {r['ticker']:<10} {r['return']:>8.1f}%")
    
    # Bottom 10 performers
    if len(results) > 10:
        print(f"\n📉 BOTTOM 10 PERFORMERS:")
        for i, r in enumerate(results[-10:], len(results)-9):
            print(f"   {i:2d}. {r['ticker']:<10} {r['return']:>8.1f}%")
