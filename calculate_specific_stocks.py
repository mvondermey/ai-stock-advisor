#!/usr/bin/env python3
"""Calculate 1-year performance for specific stocks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datetime import datetime, timedelta
import subprocess

# Stock names and their ticker symbols
stocks = {
    'ROBINH.MKTS CL.A DL-,0001': 'HOOD',
    'BROADCOM INC.     DL-,001': 'AVGO',
    'COMMERZBANK AG': 'CRZBY',
    'WB DISCOVERY SER.A DL-,01': 'WBD',
    'HOCHTIEF AG': 'HOT.DE',
    'APPLOVIN CORP.A  -,00003': 'APP',
    'AD.BIOTECH.CORP. DL-,0001': 'ADBE',
    'RWE AG   INH O.N.': 'RWE.DE',
    'RHEINMETALL AG': 'RHM.DE',
    'X(IE)-MSCI WO.IN.TE. 1CDL': 'URTH',
    'SEAGATE TEC.HLD.DL-,00001': 'STX',
    'MICRON TECHN. INC. DL-,10': 'MU',
    'SIEMENS ENERGY AG NA O.N.': 'ENR.DE',
    'META PLATF.  A DL-,000006': 'META',
    'PALANTIR TECHNOLOGIES INC': 'PLTR',
    'HOWMET AEROSPACE   DL-,01': 'HWM',
    'NEWMONT CORP.     DL 1,60': 'NEM',
    'BILFINGER SE O.N.': 'BIL.DE',
    'TAPESTRY INC.      DL-,01': 'TPR',
    'NORDEX SE O.N.': 'NDX1.DE',
    'LAM RESEARCH CORP. NEW': 'LRCX',
    'WESTN DIGITAL      DL-,01': 'WDC',
    'ASTRONICS CORP.    DL-,01': 'ATRO',
    'MUF-AMU.EOSTXX50 2XLEV.AC': 'EXS2.DE'
}

print('=' * 80)
print('📊 1-YEAR PERFORMANCE CALCULATION')
print('=' * 80)

# Try to get performance data using the existing backtest infrastructure
try:
    # Try to extract from existing backtest output first
    print("📋 Checking existing backtest data...")
    
    # Read the backtest output to extract performance data
    with open('output.log', 'r') as f:
        content = f.read()
    
    # Extract performance data for our tickers
    performance_data = {}
    for line in content.split('\n'):
        if 'return=' in line and any(ticker in line for ticker in stocks.values()):
            # Extract ticker and return value
            parts = line.split('return=')
            if len(parts) > 1:
                return_part = parts[1].split('%')[0]
                try:
                    return_value = float(return_part)
                    # Find which ticker this is
                    for ticker in stocks.values():
                        if ticker in line:
                            performance_data[ticker] = return_value
                            break
                except:
                    pass
    
    # Display results
    print(f"\n📈 1-Year Performance (from backtest data):")
    print(f"{'Name':<35} {'Ticker':<10} {'1Y Return':<12}")
    print("-" * 65)
    
    for name, ticker in stocks.items():
        if ticker in performance_data:
            print(f"{name:<35} {ticker:<10} {performance_data[ticker]:<12.1f}%")
        else:
            print(f"{name:<35} {ticker:<10} {'No data':<12}")
    
    # Calculate statistics
    if performance_data:
        returns = list(performance_data.values())
        avg_return = sum(returns) / len(returns)
        best = max(performance_data.items(), key=lambda x: x[1])
        worst = min(performance_data.items(), key=lambda x: x[1])
        
        print("\n" + "=" * 80)
        print("📊 STATISTICS")
        print("=" * 80)
        print(f"Total stocks: {len(performance_data)}")
        print(f"Average return: {avg_return:.1f}%")
        print(f"Best performer: {best[0]} ({best[1]:.1f}%)")
        print(f"Worst performer: {worst[0]} ({worst[1]:.1f}%)")
        
        # Sort by performance
        sorted_stocks = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 10 PERFORMERS:")
        for i, (ticker, ret) in enumerate(sorted_stocks[:10], 1):
            # Find the full name
            full_name = next((name for name, t in stocks.items() if t == ticker), ticker)
            print(f"   {i:2d}. {full_name:<25} {ticker:<8} {ret:>8.1f}%")
        
        print(f"\n📉 BOTTOM 5 PERFORMERS:")
        for i, (ticker, ret) in enumerate(sorted_stocks[-5:], len(sorted_stocks)-4):
            full_name = next((name for name, t in stocks.items() if t == ticker), ticker)
            print(f"   {i:2d}. {full_name:<25} {ticker:<8} {ret:>8.1f}%")
    
except FileNotFoundError:
    print("❌ No backtest data found in output.log")
    print("   Run 'python src/main.py' first to generate performance data")
except Exception as e:
    print(f"❌ Error reading backtest data: {e}")

print("\n" + "=" * 80)
print("💡 NOTE")
print("=" * 80)
print("""
This data is from the backtest which uses historical data.
For current real-time performance, you would need:
1. Access to real-time market data (Yahoo Finance, Alpha Vantage, etc.)
2. yfinance library installed: pip install yfinance
3. Internet connection to fetch current prices

The backtest data shows performance over the period:
Data Range: 2025-01-08 to 2026-01-08 (365 days)
""")
