#!/usr/bin/env python3
"""Save all 1-year performances to file and retrieve for specific stocks."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from datetime import datetime, timedelta

# Your specific stocks
your_stocks = {
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

def save_all_performances():
    """Save all 1-year performances from backtest to file."""
    print('=' * 80)
    print('💾 SAVING ALL 1-YEAR PERFORMANCES')
    print('=' * 80)
    
    try:
        # Read the backtest output
        with open('output.log', 'r') as f:
            content = f.read()
        
        # Extract all performance data
        all_performance = {}
        lines = content.split('\n')
        
        for line in lines:
            if 'return=' in line and '%' in line:
                # Extract ticker and return value
                parts = line.split('return=')
                if len(parts) > 1:
                    return_part = parts[1].split('%')[0]
                    try:
                        return_value = float(return_part)
                        # Extract ticker from the line
                        # Look for ticker pattern (usually at start of line or after a space)
                        ticker = None
                        line_parts = line.strip().split()
                        for part in line_parts:
                            if ':' in part and len(part) > 3:
                                ticker_candidate = part.split(':')[0].strip()
                                if ticker_candidate.isupper() or '.' in ticker_candidate:
                                    ticker = ticker_candidate
                                    break
                            elif part.isupper() and len(part) > 1 and '%' not in part:
                                ticker = part
                                break
                        
                        if ticker:
                            all_performance[ticker] = return_value
                    except:
                        pass
        
        # Save to file with metadata
        performance_data = {
            'data_range': {
                'start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d'),
                'description': '1-year performance from backtest data'
            },
            'performance': all_performance,
            'total_stocks': len(all_performance),
            'saved_at': datetime.now().isoformat()
        }
        
        with open('all_1y_performances.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"✅ Saved {len(all_performance)} stock performances to all_1y_performances.json")
        print(f"📅 Data range: {performance_data['data_range']['start']} to {performance_data['data_range']['end']}")
        
        return all_performance
        
    except FileNotFoundError:
        print("❌ No backtest data found in output.log")
        print("   Run 'python src/main.py' first to generate performance data")
        return {}
    except Exception as e:
        print(f"❌ Error saving performances: {e}")
        return {}

def retrieve_your_stocks(performance_data):
    """Retrieve performance for your specific stocks."""
    print('\n' + '=' * 80)
    print('📊 RETRIEVING PERFORMANCE FOR YOUR STOCKS')
    print('=' * 80)
    
    print(f"\n📈 1-Year Performance for Your Stocks:")
    print(f"{'Name':<35} {'Ticker':<10} {'1Y Return':<12} {'Status':<10}")
    print("-" * 75)
    
    found_count = 0
    total_count = len(your_stocks)
    
    for name, ticker in your_stocks.items():
        if ticker in performance_data:
            return_pct = performance_data[ticker]
            status = "✅ Found"
            print(f"{name:<35} {ticker:<10} {return_pct:<12.1f}% {status:<10}")
            found_count += 1
        else:
            print(f"{name:<35} {ticker:<10} {'No data':<12} {'❌ Missing':<10}")
    
    print(f"\n📊 Summary:")
    print(f"   Total stocks: {total_count}")
    print(f"   Found in data: {found_count}")
    print(f"   Missing: {total_count - found_count}")
    
    if found_count > 0:
        # Calculate statistics for found stocks
        found_returns = [performance_data[t] for t in your_stocks.values() if t in performance_data]
        avg_return = sum(found_returns) / len(found_returns)
        best = max(found_returns)
        worst = min(found_returns)
        
        print(f"\n📈 Statistics (for {found_count} stocks with data):")
        print(f"   Average return: {avg_return:.1f}%")
        print(f"   Best performer: {best:.1f}%")
        print(f"   Worst performer: {worst:.1f}%")
        
        # Sort your stocks by performance
        sorted_stocks = []
        for name, ticker in your_stocks.items():
            if ticker in performance_data:
                sorted_stocks.append((name, ticker, performance_data[ticker]))
        
        sorted_stocks.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🏆 Your Stocks Ranked by Performance:")
        for i, (name, ticker, ret) in enumerate(sorted_stocks, 1):
            print(f"   {i:2d}. {name:<25} {ticker:<8} {ret:>8.1f}%")
    
    return found_count

def main():
    """Main function to save and retrieve performances."""
    # Save all performances
    all_performance = save_all_performances()
    
    if all_performance:
        # Retrieve for your specific stocks
        retrieve_your_stocks(all_performance)
        
        # Save your stocks performance separately
        your_performance = {}
        for name, ticker in your_stocks.items():
            if ticker in all_performance:
                your_performance[name] = {
                    'ticker': ticker,
                    'return_1y': all_performance[ticker]
                }
        
        with open('your_stocks_performance.json', 'w') as f:
            json.dump(your_performance, f, indent=2)
        
        print(f"\n💾 Saved your stocks performance to your_stocks_performance.json")
    
    print('\n' + '=' * 80)
    print('📁 FILES CREATED:')
    print('=' * 80)
    print('   all_1y_performances.json - All stock performances')
    print('   your_stocks_performance.json - Your specific stocks only')

if __name__ == '__main__':
    main()
