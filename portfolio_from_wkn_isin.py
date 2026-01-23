#!/usr/bin/env python3
"""Convert WKN/ISIN list to ticker symbols for portfolio."""

# WKN/ISIN to ticker mappings
wkn_isin_to_ticker = {
    # German WKNs
    '843706': 'SAP.DE',
    '606840': 'SIE.DE',
    '723610': 'ALV.DE',
    '520625': 'BAS.DE',
    '555750': 'BMW.DE',
    '803200': 'VOW3.DE',
    '805200': 'VNA.DE',
    '840400': 'MBG.DE',
    '514000': 'DBK.DE',
    '555200': 'DPW.DE',
    '666200': 'IFX.DE',
    '575200': 'LHA.DE',
    '766403': 'MRK.DE',
    '710000': 'RWE.DE',
    '704000': 'SHL.DE',
    '850663': 'TKA.DE',
    '555750': 'BMW.DE',
    '823212': 'HOT.DE',
    '703000': 'RHM.DE',
    '843002': 'ENR.DE',
    'A0JQ9W': 'NDA.DE',
    'A0ERL2': 'GBF.DE',
    
    # ISINs (some examples)
    'US03831W1080': 'AFRM',
    'US2056842022': 'DHI',
    'US23834J2015': 'DASH',
    'US3696043013': 'GE',
    'US4432011082': 'HLT',
    'US6516391066': 'NKE',
    'US1710774076': 'CHTR',
    'US67079U3068': 'NVDA',
    'US69608A1088': 'PANW',
    'US7141671039': 'PEP',
    'US7665597024': 'RIO',
    'US78435P1057': 'SPY',
    'US8631111007': 'STX',
    'US9581021055': 'WFC',
    'CA28617B6061': 'ENB',
    'JP3289800009': 'DSCSY',
    'LR0008862868': 'RCL',
    'IE00BKVD2N49': 'AVGO',
    'US8170705011': 'SEM',
    'US36118L1061': 'FTNT',
    
    # Additional mappings from your lists
    'HOOD': 'HOOD',
    'AVGO': 'AVGO',
    'CRZBY': 'CRZBY',
    'WBD': 'WBD',
    'HOT.DE': 'HOT.DE',
    'APP': 'APP',
    'ADBE': 'ADBE',
    'RWE.DE': 'RWE.DE',
    'RHM.DE': 'RHM.DE',
    'URTH': 'URTH',
    'STX': 'STX',
    'MU': 'MU',
    'ENR.DE': 'ENR.DE',
    'META': 'META',
    'PLTR': 'PLTR',
    'HWM': 'HWM',
    'NEM': 'NEM',
    'BIL.DE': 'BIL.DE',
    'TPR': 'TPR',
    'NDX1.DE': 'NDX1.DE',
    'LRCX': 'LRCX',
    'WDC': 'WDC',
    'ATRO': 'ATRO',
    'EXS2.DE': 'EXS2.DE'
}

def convert_portfolio(wkn_isin_list):
    """Convert a list of WKN/ISIN/tickers to ticker symbols."""
    portfolio = {}
    
    for item in wkn_isin_list:
        item = item.strip()
        if not item:
            continue
            
        # Check if it's already a ticker
        if item.isupper() and (len(item) <= 5 or '.' in item):
            ticker = item
        else:
            # Try to find in mappings
            ticker = wkn_isin_to_ticker.get(item.upper(), None)
        
        if ticker:
            portfolio[ticker] = 1.0  # Default quantity
            print(f"   ✅ {item} -> {ticker}")
        else:
            print(f"   ❌ {item} -> Unknown")
    
    return portfolio

# Example usage
if __name__ == "__main__":
    # Example portfolio from your lists
    example_portfolio = [
        'US03831W1080',  # AFRM
        'US2056842022',  # DHI
        '823212',        # HOT.DE (WKN)
        '843002',        # ENR.DE (WKN)
        'HOOD',          # Direct ticker
        'APP',
        'PLTR'
    ]
    
    print("📊 Converting Portfolio from WKN/ISIN:")
    print("=" * 50)
    
    portfolio = convert_portfolio(example_portfolio)
    
    print(f"\n📋 Resulting Portfolio:")
    for ticker, qty in portfolio.items():
        print(f"   {ticker}: {qty} shares")
