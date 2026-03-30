#!/usr/bin/env python3
import yfinance as yf
from datetime import datetime, timedelta

tickers = ["AGG", "GLD", "SPY", "QQQ", "DIA", "NESN.SW", "AAPL", "PLTR", "EEM", "BND"]

end = datetime.now()
start = end - timedelta(days=30)

with open("quick_check_results.txt", "w") as f:
    f.write("Ticker     | 1h rows | 1d rows\n")
    f.write("-" * 40 + "\n")

    for t in tickers:
        try:
            df_1h = yf.download(t, start=start, end=end, interval="1h", progress=False)
            rows_1h = len(df_1h) if df_1h is not None and not df_1h.empty else 0
        except:
            rows_1h = 0

        try:
            df_1d = yf.download(t, start=start, end=end, interval="1d", progress=False)
            rows_1d = len(df_1d) if df_1d is not None and not df_1d.empty else 0
        except:
            rows_1d = 0

        line = f"{t:10} | {rows_1h:7} | {rows_1d:7}\n"
        f.write(line)
        print(line, end="")

print("Done - check quick_check_results.txt")
