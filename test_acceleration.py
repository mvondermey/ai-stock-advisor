import sys
import os
sys.path.append('src')
from datetime import datetime
import pandas as pd
import numpy as np

# Create the same data generation
def create_sample_data():
    dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    data = {}
    for ticker_idx, ticker in enumerate(tickers):
        prices = []
        base_price = 100.0
        
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

# Test the acceleration calculation
data = create_sample_data()
test_date = datetime(2025, 12, 15)

# Check MSFT specifically
ticker = 'MSFT'
ticker_data = data[ticker]
data_up_to_current = ticker_data.loc[:test_date]
close_prices = data_up_to_current['Close'].dropna()

# Calculate performance metrics
from config import MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M

if len(close_prices) >= MIN_DATA_DAYS_1Y:
    price_1y_ago = close_prices.iloc[-MIN_DATA_DAYS_1Y]
    price_current = close_prices.iloc[-1]
    perf_1y = (price_current - price_1y_ago) / price_1y_ago * 100
    
    if len(close_prices) >= MIN_DATA_DAYS_6M:
        price_6m_ago = close_prices.iloc[-MIN_DATA_DAYS_6M]
        perf_6m = (price_current - price_6m_ago) / price_6m_ago * 100
    else:
        perf_6m = None
    
    if len(close_prices) >= MIN_DATA_DAYS_3M:
        price_3m_ago = close_prices.iloc[-MIN_DATA_DAYS_3M]
        perf_3m = (price_current - price_3m_ago) / price_3m_ago * 100
    else:
        perf_3m = None
    
    # Calculate momentum acceleration (same as strategy)
    annualized_3m = perf_3m * (365/90) if perf_3m is not None else None
    momentum_acceleration = annualized_3m - perf_1y if annualized_3m is not None else None
    
    print(f'{ticker} Analysis:')
    print(f'  1Y Performance: {perf_1y:.1f}%')
    print(f'  6M Performance: {perf_6m:.1f}%')
    print(f'  3M Performance: {perf_3m:.1f}%')
    print(f'  Annualized 3M: {annualized_3m:.1f}%')
    print(f'  Momentum Acceleration: {momentum_acceleration:.1f}%')
    print()
    print(f'  Requirements:')
    print(f'    1Y > 5%: {"PASS" if perf_1y > 5 else "FAIL"}')
    print(f'    Acceleration > 5%: {"PASS" if momentum_acceleration > 5 else "FAIL"}')
    print(f'    Overall: {"PASS" if perf_1y > 5 and momentum_acceleration > 5 else "FAIL"}')

# Test the actual strategy
print()
print('Testing strategy...')
try:
    import io
    import contextlib
    
    from shared_strategies import select_3m_1y_ratio_stocks
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        stocks = select_3m_1y_ratio_stocks(list(data.keys()), data, test_date, top_n=3)
    
    print(f'Strategy returned: {stocks}')
    
    # Show some debug output
    output = f.getvalue()
    print('Debug output (last 1000 chars):')
    print(output[-1000:])
        
except Exception as e:
    print(f'Error: {e}')
