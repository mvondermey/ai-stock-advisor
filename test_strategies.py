import pandas as pd
from datetime import datetime, timedelta
from src.backtesting import run_walk_forward_backtest
from src.config import *

# Create mock data
dates = pd.date_range(end=datetime.today(), periods=90, freq='D')
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
data = []
for ticker in tickers:
    for date in dates:
        data.append({
            'ticker': ticker,
            'date': date,
            'Close': 100 + (hash(ticker) % 50),
            'Volume': 1000000
        })

df = pd.DataFrame(data)

# Run backtest
results = run_walk_forward_backtest(
    period_name="TEST",
    all_tickers_data=df,
    train_start_date=datetime.now() - timedelta(days=365),
    backtest_start_date=datetime.now() - timedelta(days=90),
    backtest_end_date=datetime.now(),
    initial_top_tickers=tickers,
    initial_models={},
    initial_scalers={},
    initial_y_scalers={},
    enable_ai_strategy=False
)

# Print portfolio values
print(f"Mean Reversion: ${results['mean_reversion_portfolio_value']:.2f}")
print(f"Quality+Mom: ${results['quality_momentum_portfolio_value']:.2f}")
print(f"Vol-Adj Mom: ${results['volatility_adj_mom_portfolio_value']:.2f}")
