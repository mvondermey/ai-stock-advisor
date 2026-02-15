import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Test the Risk-Adj Mom strategy
from shared_strategies import select_risk_adj_mom_stocks

# Create minimal test data
dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL']

data = {}
for ticker in tickers:
    prices = []
    base_price = 100.0
    
    for i, date in enumerate(dates):
        # Create some upward trend
        trend = 0.001  # 0.1% daily trend
        volatility = 0.02  # 2% daily volatility
        daily_return = trend + volatility * np.random.normal(0, 1)
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_return)
        
        prices.append(price)
    
    data[ticker] = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

# Test the strategy
test_date = datetime(2025, 12, 15)
print(f'Testing Risk-Adj Mom strategy with {len(data)} tickers')
print(f'Test date: {test_date}')

try:
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        stocks = select_risk_adj_mom_stocks(list(data.keys()), data, test_date, top_n=3)
    
    print(f'Strategy returned: {stocks}')
    print(f'Number of stocks selected: {len(stocks)}')
    
    # Show some output
    output = f.getvalue()
    print('Output (last 1000 chars):')
    print(output[-1000:])
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
