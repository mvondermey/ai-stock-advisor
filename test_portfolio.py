#!/usr/bin/env python3
"""Test the portfolio conversion with your WKN list."""

# Your WKN list
portfolio_wkns = [
    'A3CVQC', 'A2JG9Z', 'CBK100', 'A3DJQZ', '607000', 'A2QR0K', 'A2PLR5', '703712',
    '703000', 'A113FM', 'A3CQU7', '869020', 'ENER6Y', 'A1JWVX', 'A2QA4J',
    'A2PZ2D', '853823', '590900', 'A2JSR1', 'A0D655', 'A40L1V', '863060',
    '867880', 'LYX0BZ'
]

# Extended WKN to ticker mappings
wkn_to_ticker = {
    'A3CVQC': 'SAP.DE',
    'A2JG9Z': 'SIE.DE',
    'CBK100': 'DBK.DE',
    'A3DJQZ': 'ALV.DE',
    '607000': 'BMW.DE',
    'A2QR0K': 'VOW3.DE',
    'A2PLR5': 'VNA.DE',
    '703712': 'MBG.DE',
    '703000': 'DPW.DE',
    'A113FM': 'IFX.DE',
    'A3CQU7': 'LHA.DE',
    '869020': 'MRK.DE',
    'ENER6Y': 'RWE.DE',
    'A1JWVX': 'SHL.DE',
    'A2QA4J': 'TKA.DE',
    'A2PZ2D': 'BAS.DE',
    '853823': 'BAYN.DE',
    '590900': 'HOT.DE',
    'A2JSR1': 'RHM.DE',
    'A0D655': 'ENR.DE',
    'A40L1V': 'NDA.DE',
    '863060': 'MTK.DE',
    '867880': 'VNA.DE',
    'LYX0BZ': 'GBF.DE'
}

print('=' * 80)
print('📊 TESTING PORTFOLIO CONVERSION')
print('=' * 80)

print(f"\n📋 Your WKN List ({len(portfolio_wkns)} items):")
for i, wkn in enumerate(portfolio_wkns, 1):
    print(f"   {i:2d}. {wkn}")

print(f"\n🔄 Converting to Tickers:")
print(f"{'WKN':<10} {'Ticker':<10} {'Status':<10}")
print("-" * 35)

converted = {}
unknown = []

for wkn in portfolio_wkns:
    ticker = wkn_to_ticker.get(wkn, None)
    if ticker:
        converted[wkn] = ticker
        print(f"   {wkn:<10} {ticker:<10} {'✅':<10}")
    else:
        unknown.append(wkn)
        print(f"   {wkn:<10} {'Unknown':<10} {'❌':<10}")

print(f"\n📊 Summary:")
print(f"   Converted: {len(converted)}/{len(portfolio_wkns)}")
print(f"   Unknown: {len(unknown)}")

if unknown:
    print(f"\n❌ Unknown WKNs: {unknown}")

if converted:
    print(f"\n✅ Command to run:")
    cmd_args = ' '.join(converted.keys())
    print(f"   python src/main.py --current-portfolio {cmd_args}")
    
    print(f"\n📊 As tickers:")
    for wkn, ticker in converted.items():
        print(f"   {wkn} -> {ticker}")
