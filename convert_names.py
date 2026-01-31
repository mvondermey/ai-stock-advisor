#!/usr/bin/env python3
"""Convert stock names to ticker symbols."""

# Name to ticker mappings
name_to_ticker = {
    'ROBINH.MKTS CL.A DL-,0001': 'HOOD',
    'BROADCOM INC.     DL-,001': 'AVGO',
    'COMMERZBANK AG': 'CRZBY',
    'WB DISCOVERY SER.A DL-,01': 'WBD',
    'HOCHTIEF AG': 'HOT.DE',
    'APPLOVIN CORP.A  -,00003': 'APP',
    'AD.BIOTECH.CORP. DL-,0001': 'ADBE',
    'RWE AG   INH O.N.': 'RWE.DE',
    'RHEINMETALL AG': 'RHM.DE',
    'X(IE)-MSCI WO.IN.TE. 1CDL': 'URTH',
    'SEAGATE TEC.HLD.DL-,00001': 'STX',
    'MICRON TECHN. INC. DL-,10': 'MU',
    'SIEMENS ENERGY AG NA O.N.': 'ENR.DE',
    'META PLATF.  A DL-,000006': 'META',
    'PALANTIR TECHNOLOGIES INC': 'PLTR',
    'HOWMET AEROSPACE   DL-,01': 'HWM',
    'NEWMONT CORP.     DL 1,60': 'NEM',
    'BILFINGER SE O.N.': 'BIL.DE',
    'TAPESTRY INC.      DL-,01': 'TPR',
    'NORDEX SE O.N.': 'NDX1.DE',
    'LAM RESEARCH CORP. NEW': 'LRCX',
    'WESTN DIGITAL      DL-,01': 'WDC',
    'ASTRONICS CORP.    DL-,01': 'ATRO',
    'MUF-AMU.EOSTXX50 2XLEV.AC': 'EXS2.DE'
}

names = '''ROBINH.MKTS CL.A DL-,0001
BROADCOM INC.     DL-,001
COMMERZBANK AG
WB DISCOVERY SER.A DL-,01
HOCHTIEF AG
APPLOVIN CORP.A  -,00003
AD.BIOTECH.CORP. DL-,0001
RWE AG   INH O.N.
RHEINMETALL AG
X(IE)-MSCI WO.IN.TE. 1CDL
SEAGATE TEC.HLD.DL-,00001
MICRON TECHN. INC. DL-,10
SIEMENS ENERGY AG NA O.N.
META PLATF.  A DL-,000006
PALANTIR TECHNOLOGIES INC
HOWMET AEROSPACE   DL-,01
NEWMONT CORP.     DL 1,60
BILFINGER SE O.N.
TAPESTRY INC.      DL-,01
NORDEX SE O.N.
LAM RESEARCH CORP. NEW
WESTN DIGITAL      DL-,01
ASTRONICS CORP.    DL-,01
MUF-AMU.EOSTXX50 2XLEV.AC'''.strip().split('\n')

print('NAME -> TICKER mapping:')
print('=' * 60)
tickers = []
for name in names:
    ticker = name_to_ticker.get(name, 'UNKNOWN')
    tickers.append(ticker)
    print(f'{name:<35} -> {ticker}')

print('\n' + '=' * 60)
print(f'TICKER LIST: {tickers}')
