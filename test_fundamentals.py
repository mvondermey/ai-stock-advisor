#!/usr/bin/env python3
"""Test fundamentals data for specific symbols"""

from src.data_utils import get_fundamentals
import json

symbols_to_test = [
    'BAS.DE',  # German stock
    'VOW3.DE', # German stock
    'TLT',     # ETF
    'BTC-USD', # Crypto
    'AAPL',    # US stock (should work)
    'MSFT',    # US stock (should work)
]

for symbol in symbols_to_test:
    print(f"\n=== Testing {symbol} ===")
    data = get_fundamentals(symbol)
    if data:
        # Show first few keys
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            print(f"  Data keys: {keys}")
        else:
            print(f"  Data type: {type(data)}")
    else:
        print("  No fundamentals data")
