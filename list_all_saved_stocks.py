#!/usr/bin/env python3
"""List all stocks saved in the performance data."""

import json
from datetime import datetime

print('=' * 80)
print('📊 ALL SAVED STOCKS IN PERFORMANCE DATA')
print('=' * 80)

try:
    # Load the saved performance data
    with open('all_1y_performances.json', 'r') as f:
        data = json.load(f)
    
    performances = data['performance']
    
    print(f"\n📅 Data Range: {data['data_range']['start']} to {data['data_range']['end']}")
    print(f"📊 Total Stocks: {data['total_stocks']}")
    print(f"💾 Saved At: {data['saved_at']}")
    
    # Sort alphabetically for easy viewing
    sorted_stocks = sorted(performances.items(), key=lambda x: x[0])
    
    print(f"\n📋 ALL SAVED STOCKS:")
    print(f"{'Ticker':<12} {'1Y Return':<12} {'Category':<20}")
    print("-" * 50)
    
    # Categorize stocks
    categories = {
        'Tech': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'CSCO', 'NFLX'],
        'Semiconductor': ['NVDA', 'AMD', 'INTC', 'MU', 'LRCX', 'AMAT', 'KLAC', 'TER', 'SMCI'],
        'Software': ['MSFT', 'ORCL', 'SAP', 'ADBE', 'CRM', 'INTU', 'NOW'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL'],
        'Industrial': ['GE', 'HON', 'MMM', 'UPS', 'CAT', 'DE'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'MRK'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX'],
        'ETF': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VGT', 'FAS', 'TQQQ', 'UPRO', 'SOXL', 'TECL'],
        'German': ['.DE'],
        'Crypto': ['COIN'],
        'Other': []
    }
    
    def categorize_stock(ticker):
        for category, stocks in categories.items():
            if category == 'German' and '.DE' in ticker:
                return category
            elif category == 'ETF' and any(etf in ticker for etf in ['SPY', 'QQQ', 'FAS', 'TQQQ', 'UPRO', 'SOXL', 'TECL']):
                return category
            elif category == 'Crypto' and ticker == 'COIN':
                return category
            elif ticker in stocks:
                return category
        return 'Other'
    
    # Display all stocks with categories
    for ticker, return_pct in sorted_stocks:
        category = categorize_stock(ticker)
        return_str = f"{return_pct:.1f}%" if return_pct > 0 else f"{return_pct:.1f}%"
        print(f"{ticker:<12} {return_str:<12} {category:<20}")
    
    # Count by category
    print(f"\n📊 STOCKS BY CATEGORY:")
    print("-" * 40)
    category_counts = {}
    for ticker in performances.keys():
        cat = categorize_stock(ticker)
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat:<20}: {count} stocks")
    
    # Top performers
    sorted_by_return = sorted(performances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 TOP 10 PERFORMERS:")
    print(f"{'Rank':<6} {'Ticker':<12} {'1Y Return':<12}")
    print("-" * 35)
    for i, (ticker, return_pct) in enumerate(sorted_by_return[:10], 1):
        print(f"{i:<6} {ticker:<12} {return_pct:<12.1f}%")
    
    # Bottom performers
    print(f"\n📉 BOTTOM 10 PERFORMERS:")
    print(f"{'Rank':<6} {'Ticker':<12} {'1Y Return':<12}")
    print("-" * 35)
    for i, (ticker, return_pct) in enumerate(sorted_by_return[-10:], len(sorted_by_return)-9):
        print(f"{i:<6} {ticker:<12} {return_pct:<12.1f}%")
    
    # Save just the tickers list
    all_tickers = list(performances.keys())
    with open('all_saved_tickers.txt', 'w') as f:
        for ticker in all_tickers:
            f.write(f"{ticker}\n")
    
    print(f"\n💾 Saved {len(all_tickers)} tickers to all_saved_tickers.txt")
    
    # Create a CSV for easy import
    with open('all_performances.csv', 'w') as f:
        f.write("Ticker,Return_1Y,Category\n")
        for ticker, return_pct in sorted_stocks:
            category = categorize_stock(ticker)
            f.write(f"{ticker},{return_pct:.1f},{category}\n")
    
    print(f"💾 Saved performance data to all_performances.csv")
    
except FileNotFoundError:
    print("❌ No saved performance data found")
    print("   Run save_and_retrieve_performance.py first")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("📁 FILES CREATED:")
print("=" * 80)
print("   all_saved_tickers.txt - Simple list of all tickers")
print("   all_performances.csv - CSV with ticker, return, and category")
