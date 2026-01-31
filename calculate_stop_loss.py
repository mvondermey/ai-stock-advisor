#!/usr/bin/env python3
"""
Quick Stop Loss Calculator for Your Positions
Usage: python calculate_stop_loss.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def calculate_atr_stop_loss(ticker, period=14, multiplier=2.0):
    """Calculate ATR-based stop loss"""
    try:
        # Get last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) < period:
            print(f"Not enough data for {ticker}")
            return None, None, None
        
        # Calculate True Range
        data['high_low'] = data['High'] - data['Low']
        data['high_close'] = abs(data['High'] - data['Close'].shift(1))
        data['low_close'] = abs(data['Low'] - data['Close'].shift(1))
        data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate ATR
        data['atr'] = data['tr'].rolling(window=period).mean()
        
        current_price = data['Close'].iloc[-1]
        current_atr = data['atr'].iloc[-1]
        stop_loss = current_price - (multiplier * current_atr)
        
        return current_price, current_atr, stop_loss
        
    except Exception as e:
        print(f"Error calculating for {ticker}: {e}")
        return None, None, None

def main():
    print("STOP LOSS CALCULATOR")
    print("=" * 50)
    
    # Your positions
    positions = ['MRNA', 'ALB', 'AIXA.DE']
    
    print(f"{'Ticker':<10} {'Price':<8} {'ATR':<8} {'Stop Loss':<10} {'% Down':<8}")
    print("-" * 50)
    
    for ticker in positions:
        price, atr, stop_loss = calculate_atr_stop_loss(ticker)
        
        if price:
            percent_down = ((price - stop_loss) / price) * 100
            print(f"{ticker:<10} ${price:<7.2f} ${atr:<7.2f} ${stop_loss:<9.2f} {percent_down:<7.1f}%")
        else:
            print(f"{ticker:<10} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<8}")

if __name__ == "__main__":
    main()
