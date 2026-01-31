#!/usr/bin/env python3
"""Analyze user holdings with live data against strategy picks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datetime import datetime, timedelta
from data_fetcher import load_prices_robust
from shared_strategies import (
    calculate_risk_adjusted_momentum_score,
    check_momentum_confirmation,
    check_volume_confirmation,
    select_dynamic_bh_stocks
)
from config import RISK_ADJ_MOM_MIN_SCORE, RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_MIN_CONFIRMATIONS

# Your holdings (converted from ISINs)
your_holdings = ['AFRM', 'DHI', 'DASH', 'GE', 'HLT', 'NKE', 'CHTR', 'NVDA', 'PANW', 'PEP', 
                 'RIO', 'RCL', 'AVGO', 'SEM', 'SPY', 'STX', 'WFC', 'ENB', 'DSCSY', 'FTNT']

print('=' * 80)
print('📊 LIVE ANALYSIS - YOUR HOLDINGS')
print('=' * 80)

# Download data for all holdings
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
print(f"\n📥 Downloading data from {start_date.date()} to {end_date.date()}...")

ticker_data = {}
for ticker in your_holdings:
    try:
        data = load_prices_robust(ticker, start_date, end_date)
        if not data.empty:
            ticker_data[ticker] = data
            print(f"   ✅ {ticker}: {len(data)} rows")
        else:
            print(f"   ❌ {ticker}: No data")
    except Exception as e:
        print(f"   ❌ {ticker}: Error - {e}")

print(f"\n📊 Analyzing {len(ticker_data)} tickers with data...")
print("-" * 80)

# Analyze each holding
results = []
current_date = datetime.now()
train_start_date = current_date - timedelta(days=365)

for ticker, data in ticker_data.items():
    try:
        # Risk-Adjusted Momentum Score
        score, return_pct, volatility_pct = calculate_risk_adjusted_momentum_score(data, current_date, train_start_date)
        
        # Momentum confirmation
        momentum_conf = check_momentum_confirmation(data, current_date, train_start_date)
        momentum_ok = not RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION or momentum_conf >= RISK_ADJ_MOM_MIN_CONFIRMATIONS
        
        # Volume confirmation
        volume_ok = check_volume_confirmation(data)
        
        # Dynamic BH 1Y return
        bh_1y_return = select_dynamic_bh_stocks([ticker], {ticker: data}, '1y', current_date, 1)
        bh_selected = ticker in bh_1y_return
        
        results.append({
            'ticker': ticker,
            'score': score,
            'return': return_pct,
            'volatility': volatility_pct,
            'momentum_conf': momentum_conf,
            'momentum_ok': momentum_ok,
            'volume_ok': volume_ok,
            'bh_1y_selected': bh_selected,
            'risk_adj_mom_selected': score > RISK_ADJ_MOM_MIN_SCORE and momentum_ok and volume_ok
        })
        
    except Exception as e:
        print(f"   ⚠️ {ticker}: Analysis error - {e}")

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print detailed results
print(f"\n{'Ticker':<8} {'Score':<8} {'1Y Ret':<8} {'Vol':<8} {'Mom':<6} {'Vol':<6} {'RiskAdj':<8} {'BH1Y':<6}")
print("-" * 80)
for r in results:
    mom = "✓" if r['momentum_ok'] else "✗"
    vol = "✓" if r['volume_ok'] else "✗"
    risk = "✓" if r['risk_adj_mom_selected'] else "✗"
    bh = "✓" if r['bh_1y_selected'] else "✗"
    print(f"{r['ticker']:<8} {r['score']:<8.2f} {r['return']:<8.1f} {r['volatility']:<8.1f} {mom:<6} {vol:<6} {risk:<8} {bh:<6}")

# Recommendations
print("\n" + "=" * 80)
print("📋 RECOMMENDATIONS")
print("=" * 80)

keep = [r['ticker'] for r in results if r['risk_adj_mom_selected'] or r['bh_1y_selected']]
sell = [r['ticker'] for r in results if not r['risk_adj_mom_selected'] and not r['bh_1y_selected']]

print(f"\n✅ KEEP ({len(keep)}): {keep}")
print(f"❌ SELL ({len(sell)}): {sell}")

print(f"\n📈 Top 5 by Risk-Adjusted Score:")
for r in results[:5]:
    print(f"   {r['ticker']}: {r['score']:.2f} (1Y: {r['return']:.1f}%, Vol: {r['volatility']:.1f}%)")
