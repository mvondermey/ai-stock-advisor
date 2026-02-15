import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Replicate the updated test data generation
def create_sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    data = {}
    for ticker_idx, ticker in enumerate(tickers):
        prices = []
        base_price = 100.0
        
        # Create different performance scenarios for different tickers
        if ticker_idx < 2:  # First 2 tickers: Strong performers
            trend = 0.0008  # 0.08% daily trend (about 22% annual)
            volatility = 0.015  # 1.5% daily volatility
        elif ticker_idx < 4:  # Next 2 tickers: Moderate performers
            trend = 0.0006  # 0.06% daily trend (about 16% annual)
            volatility = 0.02  # 2% daily volatility
        else:  # Last ticker: Weak performer
            trend = -0.0001  # Slight negative trend
            volatility = 0.025  # Higher volatility
        
        np.random.seed(42 + ticker_idx)
        
        for i, date in enumerate(dates):
            daily_return = trend + volatility * np.random.normal(0, 1)
            
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * np.exp(daily_return)
            
            price = max(min(price, 10000), 1)
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

print('Performance check:')
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
        
        passes_1y = perf_1y >= MIN_PERFORMANCE_1Y
        passes_6m = perf_6m >= MIN_PERFORMANCE_6M if perf_6m is not None else False
        passes_3m = perf_3m >= MIN_PERFORMANCE_3M if perf_3m is not None else False
        passes_all = passes_1y and passes_6m and passes_3m
        
        print(f'{ticker}: 1Y={perf_1y:.1%}, 6M={perf_6m:.1%}, 3M={perf_3m:.1%} -> {"PASS" if passes_all else "FAIL"}')

print()
print('Testing strategy...')
try:
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        stocks = select_3m_1y_ratio_stocks(list(data.keys()), data, test_date, top_n=3)
    
    print(f'Strategy returned: {stocks}')
    print(f'Number of stocks selected: {len(stocks)}')
    
    if len(stocks) > 0:
        print('SUCCESS: Strategy found candidates!')
    else:
        print('FAIL: Strategy returned no stocks')
        
except Exception as e:
    print(f'Error: {e}')
