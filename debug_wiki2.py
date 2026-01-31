#!/usr/bin/env python3
"""Debug Wikipedia DAX/MDAX table parsing using the torch environment."""

import sys
sys.path.insert(0, 'src')

import requests
import pandas as pd
from io import StringIO

headers = {'User-Agent': 'Mozilla/5.0'}

# Check DAX
print("=" * 60)
print("CHECKING DAX WIKIPEDIA PAGE")
print("=" * 60)

url_dax = 'https://en.wikipedia.org/wiki/DAX'
response = requests.get(url_dax, headers=headers)
print(f"Status: {response.status_code}")

tables = pd.read_html(StringIO(response.text))
print(f"Found {len(tables)} tables")

for i, table in enumerate(tables):
    cols = list(table.columns)
    print(f"\nTable {i}: {len(table)} rows, columns = {cols[:8]}")
    if 'Ticker' in cols:
        print("  *** FOUND TICKER TABLE ***")
        print(table[['Ticker']].head(10))

# Check MDAX
print("\n" + "=" * 60)
print("CHECKING MDAX WIKIPEDIA PAGE")
print("=" * 60)

url_mdax = 'https://en.wikipedia.org/wiki/MDAX'
response = requests.get(url_mdax, headers=headers)
print(f"Status: {response.status_code}")

tables = pd.read_html(StringIO(response.text))
print(f"Found {len(tables)} tables")

for i, table in enumerate(tables):
    cols = list(table.columns)
    print(f"\nTable {i}: {len(table)} rows, columns = {cols[:8]}")
    if 'Ticker' in cols or 'Symbol' in cols or 'Ticker symbol' in cols:
        print("  *** FOUND TICKER TABLE ***")
        print(table.head(5))
