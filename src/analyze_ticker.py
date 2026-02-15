#!/usr/bin/env python3
"""
Analyze a specific ticker against Mom-Vol Hybrid strategy criteria.
Usage: python src/analyze_ticker.py SNDK
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import fetch_ticker_data
from config import DATA_PROVIDER

def analyze_ticker_for_mom_vol(ticker: str):
    """Analyze a ticker against Mom-Vol Hybrid 6M criteria."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {ticker} FOR MOM-VOL HYBRID 6M STRATEGY")
    print(f"{'='*80}\n")
    
    # Fetch data
    print(f"📊 Fetching data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)  # Get ~1 year of data
    
    try:
        ticker_data = fetch_ticker_data(ticker, start_date, end_date, DATA_PROVIDER)
        if ticker_data is None or len(ticker_data) == 0:
            print(f"❌ No data available for {ticker}")
            return
        
        print(f"✅ Retrieved {len(ticker_data)} days of data\n")
        
        # Use dropna'd Close series for all calculations (handles NaN-padded DataFrames)
        close_prices = ticker_data['Close'].dropna()
        n_prices = len(close_prices)
        
        # Get latest price
        latest_price = close_prices.iloc[-1]
        print(f"💰 Current Price: ${latest_price:.2f}")
        print(f"📊 Valid data points: {n_prices}")
        
        # Calculate 6M performance
        lookback_6m = min(126, n_prices - 1)
        if lookback_6m >= 40:
            price_6m_ago = close_prices.iloc[-lookback_6m]
            performance_6m = (latest_price - price_6m_ago) / price_6m_ago
            annualized_6m = (1 + performance_6m) ** (252 / lookback_6m) - 1
            
            print(f"\n📈 6-Month Performance:")
            print(f"   Price {lookback_6m} days ago: ${price_6m_ago:.2f}")
            print(f"   Raw return: {performance_6m:+.2%}")
            print(f"   Annualized: {annualized_6m:+.2%}")
            print(f"   ✅ PASS" if annualized_6m > 0.0 else f"   ❌ FAIL (need > 0%, got {annualized_6m:+.2%})")
        else:
            print(f"\n❌ Insufficient data for 6M calculation (need 40 days, have {lookback_6m})")
            annualized_6m = None
        
        # Calculate 1Y performance
        lookback_1y = min(252, n_prices - 1)
        if lookback_1y >= 60:
            price_1y_ago = close_prices.iloc[-lookback_1y]
            performance_1y = (latest_price - price_1y_ago) / price_1y_ago
            
            print(f"\n📊 1-Year Performance:")
            print(f"   Price {lookback_1y} days ago: ${price_1y_ago:.2f}")
            print(f"   Return: {performance_1y:+.2%}")
            print(f"   ✅ PASS" if performance_1y > -0.3 else f"   ❌ FAIL (need > -30%, got {performance_1y:+.2%})")
        else:
            print(f"\n❌ Insufficient data for 1Y calculation (need 60 days, have {lookback_1y})")
            performance_1y = None
        
        # Calculate volatility
        daily_returns = ticker_data['Close'].pct_change().dropna()
        if len(daily_returns) >= 30:
            volatility = daily_returns.std() * (252 ** 0.5)
            
            print(f"\n📉 Volatility:")
            print(f"   Daily std dev: {daily_returns.std():.4f}")
            print(f"   Annualized: {volatility:.2%}")
            print(f"   ✅ PASS" if volatility < 3.0 else f"   ❌ FAIL (need < 300%, got {volatility:.2%})")
        else:
            print(f"\n❌ Insufficient data for volatility calculation (need 30 days, have {len(daily_returns)})")
            volatility = None
        
        # Calculate average volume
        avg_volume = ticker_data['Volume'].mean() if 'Volume' in ticker_data.columns else 0
        print(f"\n📦 Volume:")
        print(f"   Average daily volume: {avg_volume:,.0f} shares")
        print(f"   ✅ PASS" if avg_volume > 10000 else f"   ❌ FAIL (need > 10,000, got {avg_volume:,.0f})")
        
        # Overall verdict
        print(f"\n{'='*80}")
        print("VERDICT:")
        print(f"{'='*80}")
        
        passes = []
        fails = []
        
        if annualized_6m is not None:
            if annualized_6m > 0.0:
                passes.append("6M momentum > 0%")
            else:
                fails.append(f"6M momentum {annualized_6m:+.2%} (need > 0%)")
        
        if performance_1y is not None:
            if performance_1y > -0.3:
                passes.append("1Y performance > -30%")
            else:
                fails.append(f"1Y performance {performance_1y:+.2%} (need > -30%)")
        
        if volatility is not None:
            if volatility < 3.0:
                passes.append(f"Volatility {volatility:.1%} < 300%")
            else:
                fails.append(f"Volatility {volatility:.1%} (need < 300%)")
        
        if avg_volume > 10000:
            passes.append(f"Volume {avg_volume:,.0f} > 10,000")
        else:
            fails.append(f"Volume {avg_volume:,.0f} (need > 10,000)")
        
        if passes:
            print("\n✅ PASSES:")
            for p in passes:
                print(f"   • {p}")
        
        if fails:
            print("\n❌ FAILS:")
            for f in fails:
                print(f"   • {f}")
        
        if not fails:
            print(f"\n🎉 {ticker} QUALIFIES for Mom-Vol Hybrid 6M strategy!")
            
            # Calculate composite score
            if annualized_6m is not None and performance_1y is not None and volatility is not None:
                momentum_score = annualized_6m * 0.6 + max(performance_1y, 0) * 0.4
                volatility_penalty = min(volatility, 1.0)
                composite_score = momentum_score * (1 - volatility_penalty * 0.3)
                print(f"   Composite Score: {composite_score:.4f}")
        else:
            print(f"\n❌ {ticker} DOES NOT QUALIFY for Mom-Vol Hybrid 6M strategy")
            print(f"   Reason: {len(fails)} filter(s) failed")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/analyze_ticker.py TICKER")
        print("Example: python src/analyze_ticker.py SNDK")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    analyze_ticker_for_mom_vol(ticker)
