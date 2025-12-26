"""
Test script to demonstrate the new data validation system
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

sys.path.insert(0, 'src')

from data_validation import (
    validate_training_data,
    validate_prediction_data,
    validate_features_after_engineering,
    get_data_summary,
    print_data_diagnostics,
    InsufficientDataError
)

print("=" * 100)
print("TESTING DATA VALIDATION SYSTEM")
print("=" * 100)

# Create test data with different scenarios
def create_test_data(num_days: int, ticker: str):
    """Create sample stock data"""
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=num_days, freq='D')
    df = pd.DataFrame({
        'Close': np.random.uniform(100, 150, num_days),
        'High': np.random.uniform(100, 150, num_days),
        'Low': np.random.uniform(100, 150, num_days),
        'Open': np.random.uniform(100, 150, num_days),
        'Volume': np.random.uniform(1000000, 5000000, num_days)
    }, index=dates)
    return df

# Test 1: Sufficient training data (should pass)
print("\n\nTest 1: SUFFICIENT TRAINING DATA (300 days)")
print("-" * 100)
try:
    df_good = create_test_data(300, "AAPL")
    start = df_good.index.min()
    end = df_good.index.max()
    validate_training_data(df_good, "AAPL", start, end)
    print("RESULT: PASS - Validation successful\n")
except InsufficientDataError as e:
    print(f"RESULT: FAIL - {e}\n")

# Test 2: Insufficient training data (should fail)
print("\nTest 2: INSUFFICIENT TRAINING DATA (50 days)")
print("-" * 100)
try:
    df_bad = create_test_data(50, "TSLA")
    start = df_bad.index.min()
    end = df_bad.index.max()
    validate_training_data(df_bad, "TSLA", start, end)
    print("RESULT: FAIL - Should have raised error\n")
except InsufficientDataError as e:
    print(f"RESULT: PASS - Correctly caught insufficient data")
    print(f"ERROR MESSAGE:\n{e}\n")

# Test 3: Sufficient prediction data (should pass)
print("\nTest 3: SUFFICIENT PREDICTION DATA (150 days)")
print("-" * 100)
try:
    df_pred_good = create_test_data(150, "MSFT")
    validate_prediction_data(df_pred_good, "MSFT")
    print("RESULT: PASS - Validation successful\n")
except InsufficientDataError as e:
    print(f"RESULT: FAIL - {e}\n")

# Test 4: Insufficient prediction data (should fail)
print("\nTest 4: INSUFFICIENT PREDICTION DATA (30 days)")
print("-" * 100)
try:
    df_pred_bad = create_test_data(30, "NVDA")
    validate_prediction_data(df_pred_bad, "NVDA")
    print("RESULT: FAIL - Should have raised error\n")
except InsufficientDataError as e:
    print(f"RESULT: PASS - Correctly caught insufficient data")
    print(f"ERROR MESSAGE:\n{e}\n")

# Test 5: Features after engineering (should fail)
print("\nTest 5: EMPTY DATAFRAME AFTER FEATURES")
print("-" * 100)
try:
    df_empty = pd.DataFrame()
    validate_features_after_engineering(df_empty, "GOOG", context="training")
    print("RESULT: FAIL - Should have raised error\n")
except InsufficientDataError as e:
    print(f"RESULT: PASS - Correctly caught empty dataframe")
    print(f"ERROR MESSAGE:\n{e}\n")

# Test 6: Data diagnostics summary
print("\nTest 6: DATA DIAGNOSTICS SUMMARY (Multiple Tickers)")
print("-" * 100)
test_tickers = [
    ("AAPL", 300),  # Good
    ("MSFT", 250),  # Good
    ("TSLA", 150),  # Warning
    ("NVDA", 50),   # Insufficient
    ("AMZN", 0),    # Empty
]

summaries = []
for ticker, days in test_tickers:
    if days > 0:
        df = create_test_data(days, ticker)
    else:
        df = pd.DataFrame()
    summary = get_data_summary(df, ticker)
    summaries.append(summary)

print_data_diagnostics(summaries)

print("\n" + "=" * 100)
print("ALL TESTS COMPLETED")
print("=" * 100)
print("\nSUMMARY:")
print("  - The validation system will now catch insufficient data BEFORE attempting training/prediction")
print("  - Clear error messages explain WHY data is insufficient and HOW to fix it")
print("  - Diagnostics table shows data quality for all tickers at once")
print("  - Script will fail fast instead of silently returning bad predictions")
print("=" * 100)

