import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Replicate the updated test data generation
def create_sample_data():
    """Create sample price data for testing."""
    # Start data early enough to have 1 year of history before first test date
    dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    data = {}
    for ticker_idx, ticker in enumerate(tickers):
        # Create realistic price data with trends
        prices = []
        base_price = 100.0
        
        # Create different performance scenarios for different tickers
        if ticker_idx < 2:  # First 2 tickers: Strong performers
            trend = 0.0003  # 0.03% daily trend (about 7.5% annual)
            volatility = 0.015  # 1.5% daily volatility
        elif ticker_idx < 4:  # Next 2 tickers: Moderate performers
            trend = 0.0002  # 0.02% daily trend (about 5% annual)
            volatility = 0.02  # 2% daily volatility
        else:  # Last ticker: Weak performer
            trend = -0.0001  # Slight negative trend
            volatility = 0.025  # Higher volatility
        
        # Set different random seeds for variety
        np.random.seed(42 + ticker_idx)
        
        for i, date in enumerate(dates):
            # Use geometric Brownian motion for realistic price movements
            daily_return = trend + volatility * np.random.normal(0, 1)
            
            if i == 0:
                price = base_price
            else:
                # Geometric Brownian motion
                price = prices[-1] * np.exp(daily_return)
            
            # Ensure price stays reasonable
            price = max(min(price, 10000), 1)  # Between $1 and $10,000
            prices.append(price)
        
        data[ticker] = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    return data

# Test the strategy
from shared_strategies import select_3m_1y_ratio_stocks
from config import (MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M, 
                     MIN_PERFORMANCE_1Y, MIN_PERFORMANCE_6M, MIN_PERFORMANCE_3M)

data = create_sample_data()
test_date = datetime(2025, 12, 15)

print(f'Minimum performance requirements:')
print(f'  1Y: {MIN_PERFORMANCE_1Y:.1%}')
print(f'  6M: {MIN_PERFORMANCE_6M:.1%}')
print(f'  3M: {MIN_PERFORMANCE_3M:.1%}')
print()

# Check performance for each ticker
for ticker in data.keys():
    ticker_data = data[ticker]
    data_up_to_current = ticker_data.loc[:test_date]
    close_prices = data_up_to_current['Close'].dropna()
    
    if len(close_prices) >= MIN_DATA_DAYS_1Y:
        price_1y_ago = close_prices.iloc[-MIN_DATA_DAYS_1Y]
        price_current = close_prices.iloc[-1]
        perf_1y = (price_current - price_1y_ago) / price_1y_ago
        
        if len(close_prices) >= MIN_DATA_DAYS_6M:
            price_6m_ago = close_prices.iloc[-MIN_DATA_DAYS_6M]
            perf_6m = (price_current - price_6m_ago) / price_6m_ago
        else:
            perf_6m = None
        
        if len(close_prices) >= MIN_DATA_DAYS_3M:
            price_3m_ago = close_prices.iloc[-MIN_DATA_DAYS_3M]
            perf_3m = (price_current - price_3m_ago) / price_3m_ago
        else:
            perf_3m = None
        
        print(f'{ticker}:')
        status_1y = "PASS" if perf_1y >= MIN_PERFORMANCE_1Y else "FAIL"
        print(f'  1Y: {perf_1y:.1%} ({status_1y})')
        if perf_6m is not None:
            status_6m = "PASS" if perf_6m >= MIN_PERFORMANCE_6M else "FAIL"
            print(f'  6M: {perf_6m:.1%} ({status_6m})')
        if perf_3m is not None:
            status_3m = "PASS" if perf_3m >= MIN_PERFORMANCE_3M else "FAIL"
            print(f'  3M: {perf_3m:.1%} ({status_3m})')
        print()

# Test the actual strategy
print('Testing strategy...')
try:
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        stocks = select_3m_1y_ratio_stocks(list(data.keys()), data, test_date, top_n=3)
    
    print(f'Strategy returned: {stocks}')
    print(f'Number of stocks selected: {len(stocks)}')
    
    # Check captured output for key messages
    output = f.getvalue()
    if 'Data insufficient' in output:
        print('ERROR: Still getting data insufficient errors')
    elif 'No candidates found' in output:
        print('WARNING: Data is sufficient but no candidates pass filters')
    elif 'Annualized Acceleration selected' in output:
        print('SUCCESS: Strategy found candidates!')
    else:
        print('UNCLEAR: Strategy status unclear')
        
except Exception as e:
    print(f'Error: {e}')
