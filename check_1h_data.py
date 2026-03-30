import yfinance as yf
from datetime import datetime, timedelta

etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB', 'SPY', 'QQQ', 'GDX', 'TLT']

end = datetime.now()
start = end - timedelta(days=30)

print("Checking 1h data availability for ETFs on yfinance:")
print("=" * 60)

for etf in etfs:
    try:
        df = yf.download(etf, start=start, end=end, interval='1h', progress=False)
        if df is not None and not df.empty:
            print(f"{etf}: ✅ {len(df)} rows of 1h data")
        else:
            print(f"{etf}: ❌ No 1h data")
    except Exception as e:
        print(f"{etf}: ❌ Error - {str(e)[:50]}")
