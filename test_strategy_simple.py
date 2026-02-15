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

test_date = datetime(2025, 12, 15)
print(f'Testing with data from {data["AAPL"].index[0].date()} to {data["AAPL"].index[-1].date()}')
print(f'Test date: {test_date.date()}')
print(f'Days available: {len(data["AAPL"])}')
print(f'Days before test date: {(data["AAPL"].index <= test_date).sum()}')

# Temporarily redirect stdout to avoid Unicode issues
import io
import contextlib

f = io.StringIO()
with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
    try:
        stocks = select_3m_1y_ratio_stocks(list(data.keys()), data, test_date, top_n=3)
        print(f'Strategy returned: {stocks}')
        print(f'Number of stocks selected: {len(stocks)}')
    except Exception as e:
        print(f'Error: {e}')

# Check the captured output
output = f.getvalue()
if 'Data insufficient' in output:
    print('Still getting data insufficient errors')
elif 'No candidates found' in output:
    print('Data is sufficient but no candidates pass filters')
else:
    print('Strategy appears to be working')
