#!/usr/bin/env python3
"""Debug Wikipedia DAX/MDAX table parsing."""

import requests
from io import StringIO

headers = {'User-Agent': 'Mozilla/5.0'}

# Check DAX
print("=" * 60)
print("CHECKING DAX WIKIPEDIA PAGE")
print("=" * 60)

url_dax = 'https://en.wikipedia.org/wiki/DAX'
response = requests.get(url_dax, headers=headers)
print(f"Status: {response.status_code}")

# Parse tables
try:
    import pandas as pd
    tables = pd.read_html(StringIO(response.text))
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        cols = list(table.columns)
        print(f"\nTable {i}: {len(table)} rows, columns = {cols[:8]}")
        if any('ticker' in str(c).lower() or 'symbol' in str(c).lower() for c in cols):
            print(f"  *** POTENTIAL TICKER TABLE ***")
            print(table.head(5))
except ImportError:
    print("pandas not available, using BeautifulSoup")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    print(f"Found {len(tables)} wikitables")
    
    for i, table in enumerate(tables):
        headers_row = table.find('tr')
        if headers_row:
            headers_cells = headers_row.find_all(['th', 'td'])
            header_text = [h.get_text(strip=True) for h in headers_cells]
            print(f"\nTable {i}: headers = {header_text[:8]}")
            
            # Check for ticker column
            if any('ticker' in h.lower() or 'symbol' in h.lower() for h in header_text):
                print(f"  *** POTENTIAL TICKER TABLE ***")
                rows = table.find_all('tr')[1:6]
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    cell_text = [c.get_text(strip=True) for c in cells]
                    print(f"    {cell_text[:5]}")

# Check MDAX
print("\n" + "=" * 60)
print("CHECKING MDAX WIKIPEDIA PAGE")
print("=" * 60)

url_mdax = 'https://en.wikipedia.org/wiki/MDAX'
response = requests.get(url_mdax, headers=headers)
print(f"Status: {response.status_code}")

try:
    import pandas as pd
    tables = pd.read_html(StringIO(response.text))
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        cols = list(table.columns)
        print(f"\nTable {i}: {len(table)} rows, columns = {cols[:8]}")
        if any('ticker' in str(c).lower() or 'symbol' in str(c).lower() for c in cols):
            print(f"  *** POTENTIAL TICKER TABLE ***")
            print(table.head(5))
except ImportError:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    print(f"Found {len(tables)} wikitables")
    
    for i, table in enumerate(tables):
        headers_row = table.find('tr')
        if headers_row:
            headers_cells = headers_row.find_all(['th', 'td'])
            header_text = [h.get_text(strip=True) for h in headers_cells]
            print(f"\nTable {i}: headers = {header_text[:8]}")
            
            if any('ticker' in h.lower() or 'symbol' in h.lower() for h in header_text):
                print(f"  *** POTENTIAL TICKER TABLE ***")
                rows = table.find_all('tr')[1:6]
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    cell_text = [c.get_text(strip=True) for c in cells]
                    print(f"    {cell_text[:5]}")
