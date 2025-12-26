"""
Quick test script to verify AI predictions work for a single stock (NVDA)
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Add src to path
sys.path.insert(0, 'src')

from config import *
from training_phase import train_models_for_period
from backtesting import _quick_predict_return

print("=" * 80)
print("TESTING AI PREDICTIONS FOR NVDA")
print("=" * 80)

# Test parameters
ticker = "NVDA"
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=365)  # 1 year of data
train_end = end_date - timedelta(days=10)    # Train on first ~355 days
train_start = start_date

print(f"\nDate Range:")
print(f"   Start: {start_date.date()}")
print(f"   Train End: {train_end.date()}")
print(f"   End: {end_date.date()}")

# Step 1: Fetch data
print(f"\nStep 1: Fetching data for {ticker}...")
try:
    import yfinance as yf
    print(f"   Using yfinance...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df is None or df.empty:
        raise Exception("yfinance returned empty data")
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure timezone-aware index
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
except Exception as e:
    print(f"ERROR: Failed to fetch data: {e}")
    sys.exit(1)

if df is None or df.empty:
    print(f"ERROR: Failed to fetch data for {ticker}")
    sys.exit(1)

print(f"SUCCESS: Fetched {len(df)} rows of data")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Convert to long format (as main.py does)
df_long = df.copy()
df_long['date'] = df_long.index
df_long['ticker'] = ticker
df_long = df_long.reset_index(drop=True)

print(f"\nData shape (long format): {df_long.shape}")
print(f"   Columns: {list(df_long.columns)}")

# Step 2: Train model
print(f"\nStep 2: Training model for {ticker}...")

# Calculate performance for NVDA
perf_start = df['Close'].iloc[0]
perf_end = df['Close'].iloc[-1]
perf_1y = ((perf_end / perf_start) - 1) * 100

top_performers_data = [(ticker, perf_1y)]
print(f"   {ticker} performance: {perf_1y:.2f}%")

# Train the model
try:
    training_results = train_models_for_period(
        period_name="Test",
        tickers=[ticker],
        all_tickers_data=df_long,
        train_start=train_start,
        train_end=train_end,
        top_performers_data=top_performers_data,
        feature_set=None,
        run_parallel=False  # Don't use multiprocessing for single stock
    )
    
    if not training_results or not training_results[0]:
        print("ERROR: Training failed - no results returned")
        sys.exit(1)
    
    result = training_results[0]
    if result.get('status') != 'trained':
        print(f"ERROR: Training failed - status: {result.get('status')}")
        sys.exit(1)
    
    model = result['model']
    scaler = result['scaler']
    y_scaler = result.get('y_scaler')
    
    print(f"SUCCESS: Model trained!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Scaler type: {type(scaler).__name__}")
    print(f"   Y-scaler: {type(y_scaler).__name__ if y_scaler else 'None'}")
    
except Exception as e:
    print(f"ERROR: Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Make prediction
print(f"\nStep 3: Making prediction for {ticker}...")

# Get recent data for prediction (use more data for features)
prediction_date = train_end + timedelta(days=5)  # Predict 5 days after training
# Use last 120 days to ensure enough data after feature engineering
df_recent = df.loc[:prediction_date].tail(120)

print(f"   Using {len(df_recent)} days of recent data")
print(f"   Date range: {df_recent.index.min()} to {df_recent.index.max()}")

try:
    prediction = _quick_predict_return(
        ticker=ticker,
        df_recent=df_recent,
        model=model,
        scaler=scaler,
        y_scaler=y_scaler,
        horizon_days=20
    )
    
    print(f"\n{'=' * 80}")
    if prediction != -np.inf:
        print(f"SUCCESS! Prediction for {ticker}: {prediction:.4f}")
        print(f"{'=' * 80}")
        print("\nAI predictions are working correctly!")
    else:
        print(f"FAILED! Prediction returned -inf")
        print(f"{'=' * 80}")
        print("\nPredictions are not working - check debug output above")
        sys.exit(1)
        
except Exception as e:
    print(f"\nERROR: Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest completed successfully!")
