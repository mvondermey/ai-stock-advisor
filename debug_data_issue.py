import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

data = {}
for ticker in tickers:
    prices = []
    base_price = 100 + np.random.randint(50, 200)
    
    for i, date in enumerate(dates):
        trend = 0.001 * i
        volatility = np.random.normal(0, 0.02)
        price_change = trend + volatility
        
        if i == 0:
            price = base_price
        else:
            price = max(prices[-1] * (1 + price_change), 1)
        
        prices.append(price)
    
    data[ticker] = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

# Test the strategy
from shared_strategies import select_3m_1y_ratio_stocks
from config import MIN_DATA_DAYS_1Y

test_date = datetime(2025, 12, 15)
print(f'MIN_DATA_DAYS_1Y: {MIN_DATA_DAYS_1Y}')
print(f'Test date: {test_date.date()}')

# Check data filtering for one ticker
ticker = 'AAPL'
ticker_data = data[ticker]
data_up_to_current = ticker_data.loc[:test_date]

print(f'Total data points: {len(ticker_data)}')
print(f'Data up to current_date: {len(data_up_to_current)}')
print(f'Data up to current_date >= MIN_DATA_DAYS_1Y: {len(data_up_to_current) >= MIN_DATA_DAYS_1Y}')

# Check close prices
close_prices = data_up_to_current['Close'].dropna()
print(f'Close prices (no NaN): {len(close_prices)}')
print(f'Close prices >= MIN_DATA_DAYS_1Y: {len(close_prices) >= MIN_DATA_DAYS_1Y}')

if len(close_prices) >= MIN_DATA_DAYS_1Y:
    price_1y_ago = close_prices.iloc[-MIN_DATA_DAYS_1Y]
    price_current = close_prices.iloc[-1]
    print(f'Price 1Y ago: {price_1y_ago:.2f}')
    print(f'Current price: {price_current:.2f}')
    perf_1y = (price_current - price_1y_ago) / price_1y_ago * 100
    print(f'1Y Performance: {perf_1y:.2f}%')
else:
    print('Insufficient close prices for 1Y calculation')
