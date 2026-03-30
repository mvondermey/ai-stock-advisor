import re

test_strings = [
    '- Consumer Discretionary Select Sector SPDR (NYSE Arca XLY)',
    '- Clean Edge Green Energy (NASDAQ|QCLN)',
    '- Technology Select Sector SPDR (NYSE Arca XLK)',
    '- iShares MSCI ACWI Index (Nasdaq: ACWI)',
    '- PowerShares QQQ (NASDAQ|QQQ)',
    '- Financial Select Sector SPDR (NYSE Arca XLF)',
    '- Energy Select Sector SPDR (NYSE Arca XLE)',
    '- Health Care Select Sector SPDR (NYSE Arca XLV)',
    '- Industrial Select Sector SPDR (NYSE Arca XLI)',
    '- Consumer Staples Select Sector SPDR (NYSE Arca XLP)',
    '- Utilities Select Sector SPDR (NYSE Arca XLU)',
    '- Materials Select Sector SPDR (NYSE Arca XLB)',
]

pattern = r'\((?:NYSE\sArca|NASDAQ|Nasdaq)[\s:|]+([A-Z0-9]+)\)'

for s in test_strings:
    match = re.search(pattern, s, re.IGNORECASE)
    if match:
        print(f'MATCH: {match.group(1)}')
    else:
        print(f'NO MATCH: {s}')
