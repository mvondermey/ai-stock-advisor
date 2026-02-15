import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Set random seed for reproducible results
np.random.seed(42)

# Create realistic test data
dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

data = {}
for ticker in tickers:
    prices = []
    base_price = 100.0  # Start with reasonable base price
    
    for i, date in enumerate(dates):
        # Use geometric Brownian motion for realistic price movements
        drift = 0.00005  # Small positive drift (about 1.3% annual)
        volatility = 0.015  # 1.5% daily volatility (reasonable for stocks)
        
        # Random walk with drift
        random_shock = np.random.normal(0, 1)
        daily_return = drift + volatility * random_shock
        
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

# Test the strategy
from shared_strategies import select_3m_1y_ratio_stocks
from config import MIN_DATA_DAYS_1Y

test_date = datetime(2025, 12, 15)
print(f'MIN_DATA_DAYS_1Y: {MIN_DATA_DAYS_1Y}')
print(f'Test date: {test_date.date()}')

# Check data for one ticker
ticker = 'AAPL'
ticker_data = data[ticker]
data_up_to_current = ticker_data.loc[:test_date]

print(f'Total data points: {len(ticker_data)}')
print(f'Data up to current_date: {len(data_up_to_current)}')

# Check close prices
close_prices = data_up_to_current['Close'].dropna()
print(f'Close prices: {len(close_prices)}')

if len(close_prices) >= MIN_DATA_DAYS_1Y:
    price_1y_ago = close_prices.iloc[-MIN_DATA_DAYS_1Y]
    price_current = close_prices.iloc[-1]
    print(f'Price 1Y ago: ${price_1y_ago:.2f}')
    print(f'Current price: ${price_current:.2f}')
    perf_1y = (price_current - price_1y_ago) / price_1y_ago * 100
    print(f'1Y Performance: {perf_1y:.2f}%')
    
    # Also check 6M and 3M
    from config import MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M
    if len(close_prices) >= MIN_DATA_DAYS_6M:
        price_6m_ago = close_prices.iloc[-MIN_DATA_DAYS_6M]
        perf_6m = (price_current - price_6m_ago) / price_6m_ago * 100
        print(f'6M Performance: {perf_6m:.2f}%')
    
    if len(close_prices) >= MIN_DATA_DAYS_3M:
        price_3m_ago = close_prices.iloc[-MIN_DATA_DAYS_3M]
        perf_3m = (price_current - price_3m_ago) / price_3m_ago * 100
        print(f'3M Performance: {perf_3m:.2f}%')
else:
    print('Insufficient close prices for 1Y calculation')

# Test the actual strategy
print('\nTesting strategy...')
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
        print('❌ Still getting data insufficient errors')
    elif 'No candidates found' in output:
        print('⚠️ Data is sufficient but no candidates pass filters')
    elif 'Annualized Acceleration selected' in output:
        print('✅ Strategy found candidates!')
    else:
        print('? Strategy status unclear')
        
except Exception as e:
    print(f'Error: {e}')
