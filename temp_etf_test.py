import re
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
url_etf = 'https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'
response_etf = requests.get(url_etf, headers=headers)
soup = BeautifulSoup(response_etf.text, 'html.parser')

etf_tickers = set()
for li in soup.find_all('li'):
    text = li.get_text()
    match = re.search(r'\((?:NYSE\sArca|NASDAQ|Nasdaq)[\s:|]+([A-Z0-9]+)\)', text, re.IGNORECASE)
    if match:
        ticker = match.group(1).strip().upper()
        ticker = ticker.replace('.', '-')
        etf_tickers.add(ticker)

print(f'Total ETFs found: {len(etf_tickers)}')

sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB']
found = [e for e in sector_etfs if e in etf_tickers]
missing = [e for e in sector_etfs if e not in etf_tickers]
print(f'Sector ETFs found ({len(found)}): {found}')
print(f'Sector ETFs missing ({len(missing)}): {missing}')
print(f'Sample of all tickers: {sorted(list(etf_tickers))[:20]}')
