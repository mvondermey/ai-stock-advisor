#!/usr/bin/env python3
"""Test model loading with multiple naming patterns."""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from prediction import _load_single_model

models_dir = Path("logs/models")

# Test tickers that were missing
test_tickers = ["SLV", "LRCX", "ALB", "AMAT", "GOOGL", "INTC", "LLY", "WDC", "AAPL"]

print("Testing model loading with multiple naming patterns...")
print("-" * 60)

for ticker in test_tickers:
    result = _load_single_model((ticker, models_dir))
    ticker, model, scaler, y_scaler, model_class = result
    status = "✅" if model is not None else "❌"
    model_type = type(model).__name__ if model else "None"
    print(f"{status} {ticker}: model={model_type}, scaler={scaler is not None}, y_scaler={y_scaler is not None}, class={model_class}")

print("-" * 60)
print("Done!")
