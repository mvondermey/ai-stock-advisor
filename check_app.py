import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load the backtest data to check APP's performance
try:
    df = pd.read_pickle('/home/mvondermey/ai-stock-advisor/data_cache/all_tickers_data.pkl')
    if 'ticker' in df.columns:
        app_data = df[df['ticker'] == 'APP'].copy()
        if not app_data.empty:
            app_data = app_data.set_index('date')
            app_data.index = pd.to_datetime(app_data.index)
            
            # Ensure timezone-aware
            if app_data.index.tzinfo is None:
                app_data.index = app_data.index.tz_localize('UTC')
            else:
                app_data.index = app_data.index.tz_convert('UTC')
            
            today = datetime.now(timezone.utc)
            
            # Calculate 3M performance
            start_date = today - timedelta(days=90)
            mask = (app_data.index >= start_date) & (app_data.index <= today)
            app_3m = app_data.loc[mask]
            
            if not app_3m.empty and len(app_3m) >= 2:
                prices = pd.to_numeric(app_3m['Close'], errors='coerce').ffill().bfill().dropna()
                if not prices.empty:
                    start_price = prices.iloc[0]
                    end_price = prices.iloc[-1]
                    perf_3m = ((end_price / start_price) - 1) * 100
                    print(f'APP 3M Performance: {perf_3m:.2f}%')
                    
                    # Also check 1Y
                    start_date_1y = today - timedelta(days=365)
                    mask_1y = (app_data.index >= start_date_1y) & (app_data.index <= today)
                    app_1y = app_data.loc[mask_1y]
                    
                    if not app_1y.empty and len(app_1y) >= 2:
                        prices_1y = pd.to_numeric(app_1y['Close'], errors='coerce').ffill().bfill().dropna()
                        if not prices_1y.empty:
                            start_price_1y = prices_1y.iloc[0]
                            end_price_1y = prices_1y.iloc[-1]
                            perf_1y = ((end_price_1y / start_price_1y) - 1) * 100
                            print(f'APP 1Y Performance: {perf_1y:.2f}%')
                else:
                    print('APP: No valid price data')
            else:
                print('APP: Insufficient data for 3M')
        else:
            print('APP: No data found')
    else:
        print('No ticker column in data')
except Exception as e:
    print(f'Error: {e}')
