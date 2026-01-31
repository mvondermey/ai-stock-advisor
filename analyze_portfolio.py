#!/usr/bin/env python3
"""
Standalone portfolio analysis script.
Usage: python analyze_portfolio.py A3CVQC A2JG9Z CBK100 ...
"""

import sys
import json
from datetime import datetime, timedelta

# WKN/ISIN to ticker mappings
WKN_ISIN_TO_TICKER = {
    # German WKNs
    '843706': 'SAP.DE',
    '606840': 'SIE.DE',
    '723610': 'ALV.DE',
    '520625': 'BAS.DE',
    '555750': 'BMW.DE',
    '803200': 'VOW3.DE',
    '805200': 'VNA.DE',
    '840400': 'MBG.DE',
    '514000': 'DBK.DE',
    '555200': 'DPW.DE',
    '666200': 'IFX.DE',
    '575200': 'LHA.DE',
    '766403': 'MRK.DE',
    '710000': 'RWE.DE',
    '704000': 'SHL.DE',
    '850663': 'TKA.DE',
    '823212': 'HOT.DE',
    '843002': 'ENR.DE',
    'A0JQ9W': 'NDA.DE',
    'A0ERL2': 'GBF.DE',
    # Your specific WKNs
    'A3CVQC': 'SAP.DE',
    'A2JG9Z': 'SIE.DE',
    'CBK100': 'DBK.DE',
    'A3DJQZ': 'ALV.DE',
    '607000': 'BMW.DE',
    'A2QR0K': 'VOW3.DE',
    '703712': 'MBG.DE',
    '703000': 'DPW.DE',
    'A113FM': 'IFX.DE',
    'A3CQU7': 'LHA.DE',
    '869020': 'MRK.DE',
    'ENER6Y': 'RWE.DE',
    'A1JWVX': 'SHL.DE',
    'A2QA4J': 'TKA.DE',
    'A2PZ2D': 'BAS.DE',
    '853823': 'BAYN.DE',
    '590900': 'HOT.DE',
    'A2JSR1': 'RHM.DE',
    'A0D655': 'ENR.DE',
    'A40L1V': 'NDA.DE',
    '863060': 'MTK.DE',
    '867880': 'VNA.DE',
    'LYX0BZ': 'GBF.DE',
    # ISINs
    'US03831W1080': 'AFRM',
    'US2056842022': 'DHI',
    'US23834J2015': 'DASH',
    'US3696043013': 'GE',
    'US4432011082': 'HLT',
    'US6516391066': 'NKE',
    'US1710774076': 'CHTR',
    'US67079U3068': 'NVDA',
    'US69608A1088': 'PANW',
    'US7141671039': 'PEP',
    'US7665597024': 'RIO',
    'US78435P1057': 'SPY',
    'US8631111007': 'STX',
    'US9581021055': 'WFC',
    'CA28617B6061': 'ENB',
    'JP3289800009': 'DSCSY',
    'LR0008862868': 'RCL',
    'IE00BKVD2N49': 'AVGO',
    'US8170705011': 'SEM',
    'US36118L1061': 'FTNT',
    # Direct tickers
    'HOOD': 'HOOD',
    'AVGO': 'AVGO',
    'APP': 'APP',
    'ADBE': 'ADBE',
    'META': 'META',
    'PLTR': 'PLTR',
    'NVDA': 'NVDA',
    'STX': 'STX',
    'MU': 'MU',
}

def convert_to_tickers(portfolio_items):
    """Convert WKN/ISIN to ticker symbols."""
    portfolio_tickers = {}
    
    for item in portfolio_items:
        item = item.strip().upper()
        if not item:
            continue
        
        # Check if it's already a ticker
        if item in WKN_ISIN_TO_TICKER:
            ticker = WKN_ISIN_TO_TICKER[item]
            portfolio_tickers[ticker] = 1.0
            print(f"   ✅ {item} -> {ticker}")
        elif item.isupper() and (len(item) <= 5 or '.' in item):
            # Assume it's already a ticker
            portfolio_tickers[item] = 1.0
            print(f"   ✅ {item} -> {item}")
        else:
            print(f"   ❌ {item} -> Unknown")
    
    return portfolio_tickers

def load_cached_performances():
    """Load cached performance data from previous runs."""
    # Try to load from latest live trading performances
    try:
        with open('latest_live_trading_performances.json', 'r') as f:
            data = json.load(f)
            print(f"   ✅ Loaded from latest_live_trading_performances.json")
            return data['performance'], data['data_range']
    except FileNotFoundError:
        pass
    
    # Try to load from all_1y_performances.json
    try:
        with open('all_1y_performances.json', 'r') as f:
            data = json.load(f)
            print(f"   ✅ Loaded from all_1y_performances.json")
            return data['performance'], data['data_range']
    except FileNotFoundError:
        pass
    
    return None, None

def analyze_portfolio(portfolio_items):
    """Analyze portfolio using cached data."""
    print('=' * 80)
    print('📊 CURRENT PORTFOLIO ANALYSIS')
    print('=' * 80)
    
    # Convert to tickers
    print(f"\n🔄 Converting WKN/ISIN to tickers...")
    portfolio_tickers = convert_to_tickers(portfolio_items)
    
    if not portfolio_tickers:
        print("\n❌ No valid tickers found in portfolio")
        return
    
    print(f"\n📋 Portfolio: {len(portfolio_tickers)} tickers")
    print(f"   {'Ticker':<10}")
    print("-" * 15)
    for ticker in portfolio_tickers.keys():
        print(f"   {ticker:<10}")
    
    # Load cached performance data
    print(f"\n📥 Loading cached performance data...")
    all_performances, data_range = load_cached_performances()
    
    if all_performances is None:
        print("\n❌ No cached performance data found")
        print("💡 Run live trading first to generate performance data:")
        print("   python src/main.py --live-trading --strategy risk_adj_mom")
        return
    
    # Calculate portfolio performance
    print(f"\n📊 1-Year Performance Analysis:")
    if data_range:
        print(f"📅 Data Range: {data_range['start']} to {data_range['end']}")
    print(f"{'Ticker':<10} {'1Y Return':<12} {'Status':<15}")
    print("-" * 45)
    
    performances = {}
    for ticker in portfolio_tickers.keys():
        if ticker in all_performances:
            return_pct = all_performances[ticker]
            performances[ticker] = return_pct
            
            # Status based on performance
            if return_pct > 50:
                status = "🟢 Strong"
            elif return_pct > 0:
                status = "🟡 Positive"
            else:
                status = "🔴 Negative"
            
            print(f"   {ticker:<10} {return_pct:<12.1f}% {status:<15}")
        else:
            print(f"   {ticker:<10} {'No data':<12} {'❌ Missing':<15}")
    
    if performances:
        returns = list(performances.values())
        avg_return = sum(returns) / len(returns)
        positive = len([r for r in returns if r > 0])
        negative = len([r for r in returns if r < 0])
        
        print(f"\n📈 Portfolio Statistics:")
        print(f"   Average return: {avg_return:.1f}%")
        print(f"   Positive returns: {positive}/{len(performances)} ({positive/len(performances)*100:.1f}%)")
        print(f"   Negative returns: {negative}/{len(performances)} ({negative/len(performances)*100:.1f}%)")
        
        # Best and worst performers
        sorted_perf = sorted(performances.items(), key=lambda x: x[1], reverse=True)
        print(f"   Best performer: {sorted_perf[0][0]} ({sorted_perf[0][1]:.1f}%)")
        print(f"   Worst performer: {sorted_perf[-1][0]} ({sorted_perf[-1][1]:.1f}%)")
    else:
        print(f"\n📊 Portfolio Statistics:")
        print(f"   No performance data available for your tickers")
    
    # Market comparison
    print(f"\n🎯 Market Comparison:")
    sorted_all = sorted(all_performances.items(), key=lambda x: x[1], reverse=True)
    top_performers = sorted_all[:10]
    
    if top_performers:
        print(f"\n📊 Top 10 Market Performers:")
        print(f"{'Rank':<6} {'Ticker':<10} {'1Y Return':<12} {'In Portfolio':<15}")
        print("-" * 50)
        
        for i, (ticker, perf) in enumerate(top_performers, 1):
            in_portfolio = "✅ Yes" if ticker in portfolio_tickers else "❌ No"
            print(f"   {i:<6} {ticker:<10} {perf:<12.1f}% {in_portfolio:<15}")
        
        # Check overlap
        portfolio_in_top = [t for t, _ in top_performers if t in portfolio_tickers]
        print(f"\n📊 Portfolio in Top 10: {len(portfolio_in_top)}/10")
        if portfolio_in_top:
            print(f"   {portfolio_in_top}")
        
        # Show where portfolio stocks rank
        if performances:
            print(f"\n📊 Your Portfolio Ranking:")
            for ticker in portfolio_tickers.keys():
                if ticker in all_performances:
                    rank = [i for i, (t, _) in enumerate(sorted_all, 1) if t == ticker]
                    if rank:
                        perf = all_performances[ticker]
                        print(f"   {ticker:<10} Rank #{rank[0]:<4} ({perf:.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_portfolio.py WKN1 WKN2 ISIN1 TICKER1 ...")
        print("Example: python analyze_portfolio.py A3CVQC A2JG9Z CBK100 HOOD APP")
        sys.exit(1)
    
    analyze_portfolio(sys.argv[1:])
