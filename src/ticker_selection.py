import os
import sys
import json
import re
import time
import gc
import threading
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

# Global lock for yfinance calls - yfinance has threading bugs that cause data corruption
_yfinance_lock = threading.Lock()
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import financial data fetching function
try:
    from data_fetcher import _fetch_financial_data
except ImportError:
    _fetch_financial_data = None

# Import from config and data_fetcher
from config import (
    DATA_PROVIDER, N_TOP_TICKERS, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES,
    PAUSE_BETWEEN_YF_CALLS, MARKET_SELECTION, USE_PERFORMANCE_BENCHMARK,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, TOP_CACHE_PATH, VALID_TICKERS_CACHE_PATH,
    ALPACA_STOCKS_LIMIT, ALPACA_STOCKS_EXCHANGES, NUM_PROCESSES
)
from data_utils import load_prices_robust, _download_batch_robust
from utils import _ensure_dir, _normalize_symbol, _to_utc

# Optional Alpaca provider for asset listing
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

def get_all_tickers() -> List[str]:
    """
    Gets a list of tickers from the markets selected in the configuration.
    """
    all_tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    if MARKET_SELECTION.get("ALPACA_STOCKS"):
        if ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
                search_params = GetAssetsRequest(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE
                )
                assets = trading_client.get_all_assets(search_params)

                # ✅ Use Alpaca's built-in attributes to filter common stocks
                # fractionable=True typically indicates common stocks (not warrants/preferred/rights)
                # If fractionable attribute doesn't exist, assume it's NOT a common stock (safer)
                tradable_assets = [
                    a for a in assets
                    if a.tradable
                    and getattr(a, 'fractionable', False)  # ✅ Default False = exclude if unknown
                ]

                # Filter by exchange if specified in config
                if ALPACA_STOCKS_EXCHANGES:
                    tradable_assets = [a for a in tradable_assets if a.exchange in ALPACA_STOCKS_EXCHANGES]

                # Get symbols
                alpaca_tickers = [asset.symbol for asset in tradable_assets]
                print(f"   📊 After fractionable filter: {len(alpaca_tickers)} tickers")

                # ✅ Additional filtering: Remove foreign ADRs and special securities
                def is_us_common_stock(symbol: str) -> bool:
                    """Filter out foreign ADRs (ending in Y) and other special securities"""
                    symbol_upper = symbol.upper()
                    # Exclude ADRs (5-letter tickers ending in Y, e.g., BTVCY, ASAZY)
                    if len(symbol_upper) == 5 and symbol_upper.endswith('Y'):
                        return False
                    # Exclude very long symbols (usually special securities)
                    if len(symbol_upper) > 5:
                        return False
                    # Exclude symbols with special characters (except hyphen for class shares like BRK-A)
                    if '$' in symbol_upper or '/' in symbol_upper or '_' in symbol_upper:
                        return False
                    return True

                alpaca_tickers_before = len(alpaca_tickers)
                alpaca_tickers = [t for t in alpaca_tickers if is_us_common_stock(t)]
                filtered_count = alpaca_tickers_before - len(alpaca_tickers)
                print(f"   📊 After ADR/special filter: {len(alpaca_tickers)} tickers (filtered out {filtered_count})")
                if filtered_count > 0:
                    print(f"   ✅ Removed {filtered_count} foreign ADRs/special securities (e.g., symbols ending in Y)")


                # Apply ALPACA_STOCKS_LIMIT to prevent downloading too many stocks
                exchange_filter_desc = f" ({', '.join(ALPACA_STOCKS_EXCHANGES)} only)" if ALPACA_STOCKS_EXCHANGES else ""
                if len(alpaca_tickers) > ALPACA_STOCKS_LIMIT:
                    alpaca_tickers = alpaca_tickers[:ALPACA_STOCKS_LIMIT]
                    print(f"[LIMITED] Fetched {len(alpaca_tickers)} tradable US equity tickers from Alpaca{exchange_filter_desc} (limited to {ALPACA_STOCKS_LIMIT}).")
                else:
                    print(f"[SUCCESS] Fetched {len(alpaca_tickers)} tradable US equity tickers from Alpaca{exchange_filter_desc}.")

                all_tickers.update(alpaca_tickers)
            except Exception as e:
                print(f"[WARNING] Could not fetch asset list from Alpaca ({e}).")
        else:
            print("⚠️ Alpaca stock selection is enabled, but SDK/API keys are not available.")

    if MARKET_SELECTION.get("NASDAQ_ALL"):
        try:
            url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
            df = pd.read_csv(url, sep='|')
            df_clean = df.iloc[:-1]
            # ✅ Filter: Test Issue = 'N' (not test), and exclude delisted (ETF column should be 'N' for stocks)
            # Also check if there's a delisting indicator
            nasdaq_tickers = df_clean[
                (df_clean['Test Issue'] == 'N') &  # Not a test issue
                (df_clean.get('Financial Status', 'N') != 'D')  # Not delisted (if column exists)
            ]['Symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            print(f"✅ Fetched {len(nasdaq_tickers)} active NASDAQ tickers (delisted excluded).")
        except Exception as e:
            print(f"⚠️ Could not fetch full NASDAQ list ({e}).")

    if MARKET_SELECTION.get("NASDAQ_100"):
        try:
            url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response_nasdaq = requests.get(url_nasdaq, headers=headers)
            response_nasdaq.raise_for_status()
            table_nasdaq = pd.read_html(StringIO(response_nasdaq.text))[4]
            nasdaq_100_tickers = [s.replace('.', '-') for s in table_nasdaq['Ticker'].tolist()]
            all_tickers.update(nasdaq_100_tickers)
            print(f"✅ Fetched {len(nasdaq_100_tickers)} tickers from NASDAQ 100.")
        except Exception as e:
            print(f"⚠️ Could not fetch NASDAQ 100 list ({e}). Using fallback list.")
            # Fallback list of popular NASDAQ 100 stocks
            nasdaq_100_tickers = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD',
                'INTC', 'CMCSA', 'PEP', 'COST', 'ADBE', 'AVGO', 'TXN', 'QCOM', 'HON', 'AMGN',
                'SBUX', 'INTU', 'AMD', 'ISRG', 'BKNG', 'MDLZ', 'GILD', 'REGN', 'VRTX', 'ILMN',
                'IDXX', 'LRCX', 'KLAC', 'AMAT', 'MU', 'WBD', 'PLTR', 'APP', 'SHOP', 'AZN',
                'ASML', 'MNST', 'CSCO', 'EA', 'TTWO', 'ADI', 'AEP', 'AMGN', 'CCEP', 'EXC'
            ]
            all_tickers.update(nasdaq_100_tickers)
            print(f"✅ Using fallback list with {len(nasdaq_100_tickers)} NASDAQ 100 tickers.")

    if MARKET_SELECTION.get("SP500"):
        try:
            url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response_sp500 = requests.get(url_sp500, headers=headers)
            response_sp500.raise_for_status()
            table_sp500 = pd.read_html(StringIO(response_sp500.text))[0]
            col = "Symbol" if "Symbol" in table_sp500.columns else table_sp500.columns[0]
            sp500_tickers = [s.replace('.', '-') for s in table_sp500[col].tolist()]
            all_tickers.update(sp500_tickers)
            print(f"✅ Fetched {len(sp500_tickers)} tickers from S%26P 500.")
        except Exception as e:
            print(f"⚠️ Could not fetch S%26P 500 list ({e}).")

    if MARKET_SELECTION.get("DOW_JONES"):
        try:
            url_dow = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            response_dow = requests.get(url_dow, headers=headers)
            response_dow.raise_for_status()
            tables_dow = pd.read_html(StringIO(response_dow.text))
            table_dow = None
            for table in tables_dow:
                if 'Symbol' in table.columns:
                    table_dow = table
                    break
            if table_dow is None:
                raise ValueError("Could not find the ticker table on the Dow Jones Wikipedia page.")
            col = "Symbol"
            dow_tickers = [str(s).replace('.', '-') for s in table_dow[col].tolist()]
            all_tickers.update(dow_tickers)
            print(f"✅ Fetched {len(dow_tickers)} tickers from Dow Jones. ")
        except Exception as e:
            print(f"⚠️ Could not fetch Dow Jones list ({e}).")

    if MARKET_SELECTION.get("POPULAR_ETFS"):
        try:
            url_etf = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
            response_etf = requests.get(url_etf, headers=headers)
            response_etf.raise_for_status()

            soup = BeautifulSoup(response_etf.text, 'html.parser')
            etf_tickers = set()

            for li in soup.find_all('li'):
                text = li.get_text()
                # Match formats like "(NYSE Arca XLK)" or "(NYSE Arca:XLK)" or "(NASDAQ|QQQ)" or "(Nasdaq:ACWI)"
                match = re.search(r'\((?:NYSE\sArca|NASDAQ|Nasdaq)[\s:|]+([A-Z0-9]+)\)', text, re.IGNORECASE)
                if match:
                    ticker = match.group(1).strip().upper()
                    ticker = ticker.replace('.', '-')
                    etf_tickers.add(ticker)

            if not etf_tickers:
                raise ValueError("No ETF tickers found on the page.")

            # Add sector ETFs required for Sector Rotation strategy
            # These may not be scraped from Wikipedia but are essential
            sector_etfs = [
                'XLK',   # Technology
                'XLF',   # Financials
                'XLE',   # Energy
                'XLV',   # Healthcare
                'XLI',   # Industrials
                'XLP',   # Consumer Staples
                'XLY',   # Consumer Discretionary
                'XLU',   # Utilities
                'XLRE',  # Real Estate
                'XLC',   # Communication Services
                'XLB',   # Materials
                'GDX',   # Gold Miners
                'USO',   # Oil
                'TLT',   # Long-term Treasuries
            ]
            etf_tickers.update(sector_etfs)

            # Add market proxy ETFs required for regime detection strategies
            # (Defensive Momentum, Adaptive Ensemble, etc.)
            market_proxy_etfs = [
                'SPY',   # S&P 500 (primary market proxy)
                'QQQ',   # NASDAQ 100 (tech-heavy market proxy)
                'IWM',   # Russell 2000 (small-cap proxy)
                'DIA',   # Dow Jones Industrial Average
                'VTI',   # Total US Stock Market
            ]
            etf_tickers.update(market_proxy_etfs)

            all_tickers.update(etf_tickers)
            print(f"✅ Fetched {len(etf_tickers)} tickers from Popular ETFs list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Popular ETFs list ({e}).")

    if MARKET_SELECTION.get("CRYPTO"):
        try:
            url_crypto = "https://en.wikipedia.org/wiki/List_of_cryptocurrencies"
            response_crypto = requests.get(url_crypto, headers=headers)
            response_crypto.raise_for_status()
            tables_crypto = pd.read_html(StringIO(response_crypto.text))
            table_crypto = None
            for table in tables_crypto:
                if 'Symbol' in table.columns:
                    table_crypto = table
                    break
            if table_crypto is None:
                raise ValueError("Could not find the ticker table on the Cryptocurrency Wikipedia page.")
            if 'Symbol' in table_crypto.columns:
                crypto_tickers = set()
                for s in table_crypto['Symbol'].tolist():
                    if isinstance(s, str):
                        match = re.match(r'([A-Z]+)', s)
                        if match:
                            crypto_tickers.add(f"{match.group(1)}-USD")
                all_tickers.update(crypto_tickers)
                print(f"✅ Fetched {len(crypto_tickers)} tickers from Cryptocurrency list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Cryptocurrency list ({e}).")

    if MARKET_SELECTION.get("DAX"):
        try:
            url_dax = "https://en.wikipedia.org/wiki/DAX"
            response_dax = requests.get(url_dax, headers=headers)
            response_dax.raise_for_status()
            tables_dax = pd.read_html(StringIO(response_dax.text))
            table_dax = None
            for table in tables_dax:
                if 'Ticker' in table.columns:
                    table_dax = table
                    break
            if table_dax is None:
                raise ValueError("Could not find the ticker table on the DAX Wikipedia page.")
            dax_tickers = [s if '.' in s else f"{s}.DE" for s in table_dax['Ticker'].tolist()]
            all_tickers.update(dax_tickers)
            print(f"✅ Fetched {len(dax_tickers)} tickers from DAX.")
        except Exception as e:
            print(f"⚠️ Could not fetch DAX list ({e}).")

    if MARKET_SELECTION.get("MDAX"):
        try:
            url_mdax = "https://en.wikipedia.org/wiki/MDAX"
            response_mdax = requests.get(url_mdax, headers=headers)
            response_mdax.raise_for_status()
            tables_mdax = pd.read_html(StringIO(response_mdax.text))
            table_mdax = None
            for table in tables_mdax:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    table_mdax = table
                    break
            if table_mdax is None:
                raise ValueError("Could not find the ticker table on the MDAX Wikipedia page.")
            ticker_col = 'Ticker' if 'Ticker' in table_mdax.columns else 'Symbol'
            mdax_tickers = [s if '.' in s else f"{s}.DE" for s in table_mdax[ticker_col].tolist()]
            all_tickers.update(mdax_tickers)
            print(f"✅ Fetched {len(mdax_tickers)} tickers from MDAX.")
        except Exception as e:
            print(f"⚠️ Could not fetch MDAX list ({e}).")

    if MARKET_SELECTION.get("SMI"):
        try:
            url_smi = "https://en.wikipedia.org/wiki/Swiss_Market_Index"
            response_smi = requests.get(url_smi, headers=headers)
            response_smi.raise_for_status()
            tables_smi = pd.read_html(StringIO(response_smi.text))
            table_smi = None
            for table in tables_smi:
                if 'Ticker' in table.columns:
                    table_smi = table
                    break
            if table_smi is None:
                raise ValueError("Could not find the ticker table on the SMI Wikipedia page.")
            smi_tickers = [s if '.' in s else f"{s}.SW" for s in table_smi['Ticker'].tolist()]
            all_tickers.update(smi_tickers)
            print(f"✅ Fetched {len(smi_tickers)} tickers from SMI.")
        except Exception as e:
            print(f"⚠️ Could not fetch SMI list ({e}).")

    if MARKET_SELECTION.get("FTSE_MIB"):
        try:
            url_mib = "https://en.wikipedia.org/wiki/FTSE_MIB"
            response_mib = requests.get(url_mib, headers=headers)
            response_mib.raise_for_status()
            tables_mib = pd.read_html(StringIO(response_mib.text))
            table_mib = None
            for table in tables_mib:
                if 'Ticker' in table.columns:
                    table_mib = table
                    break
            if table_mib is None:
                raise ValueError("Could not find the ticker table on the FTSE MIB Wikipedia page.")
            ticker_col = 'Ticker'
            mib_tickers = [s if '.' in s else f"{s}.MI" for s in table_mib[ticker_col].tolist()]
            all_tickers.update(mib_tickers)
            print(f"✅ Fetched {len(mib_tickers)} tickers from FTSE MIB.")
        except Exception as e:
            print(f"⚠️ Could not fetch FTSE MIB list ({e}).")

    # S&P 400 MidCap
    if MARKET_SELECTION.get("SP400_MIDCAP"):
        try:
            url_sp400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
            response_sp400 = requests.get(url_sp400, headers=headers)
            response_sp400.raise_for_status()
            tables_sp400 = pd.read_html(StringIO(response_sp400.text))
            table_sp400 = None
            for table in tables_sp400:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    table_sp400 = table
                    break
            if table_sp400 is None:
                raise ValueError("Could not find the ticker table on the S&P 400 Wikipedia page.")
            ticker_col = 'Symbol' if 'Symbol' in table_sp400.columns else 'Ticker'
            sp400_tickers = [_normalize_symbol(str(s), DATA_PROVIDER) for s in table_sp400[ticker_col].tolist() if pd.notna(s)]
            all_tickers.update(sp400_tickers)
            print(f"✅ Fetched {len(sp400_tickers)} tickers from S&P 400 MidCap.")
        except Exception as e:
            print(f"⚠️ Could not fetch S&P 400 MidCap list ({e}).")

    # S&P 600 SmallCap
    if MARKET_SELECTION.get("SP600_SMALLCAP"):
        try:
            url_sp600 = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
            response_sp600 = requests.get(url_sp600, headers=headers)
            response_sp600.raise_for_status()
            tables_sp600 = pd.read_html(StringIO(response_sp600.text))
            table_sp600 = None
            for table in tables_sp600:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    table_sp600 = table
                    break
            if table_sp600 is None:
                raise ValueError("Could not find the ticker table on the S&P 600 Wikipedia page.")
            ticker_col = 'Symbol' if 'Symbol' in table_sp600.columns else 'Ticker'
            sp600_tickers = [_normalize_symbol(str(s), DATA_PROVIDER) for s in table_sp600[ticker_col].tolist() if pd.notna(s)]
            all_tickers.update(sp600_tickers)
            print(f"✅ Fetched {len(sp600_tickers)} tickers from S&P 600 SmallCap.")
        except Exception as e:
            print(f"⚠️ Could not fetch S&P 600 SmallCap list ({e}).")

    # CAC 40 (French market)
    if MARKET_SELECTION.get("CAC_40"):
        try:
            url_cac = "https://en.wikipedia.org/wiki/CAC_40"
            response_cac = requests.get(url_cac, headers=headers)
            response_cac.raise_for_status()
            tables_cac = pd.read_html(StringIO(response_cac.text))
            table_cac = None
            for table in tables_cac:
                if 'Ticker' in table.columns:
                    table_cac = table
                    break
            if table_cac is None:
                raise ValueError("Could not find the ticker table on the CAC 40 Wikipedia page.")
            cac_tickers = [s if '.' in s else f"{s}.PA" for s in table_cac['Ticker'].tolist() if pd.notna(s)]
            all_tickers.update(cac_tickers)
            print(f"✅ Fetched {len(cac_tickers)} tickers from CAC 40.")
        except Exception as e:
            print(f"⚠️ Could not fetch CAC 40 list ({e}).")

    # IBEX 35 (Spanish market)
    if MARKET_SELECTION.get("IBEX_35"):
        try:
            url_ibex = "https://en.wikipedia.org/wiki/IBEX_35"
            response_ibex = requests.get(url_ibex, headers=headers)
            response_ibex.raise_for_status()
            tables_ibex = pd.read_html(StringIO(response_ibex.text))
            table_ibex = None
            for table in tables_ibex:
                if 'Ticker' in table.columns:
                    table_ibex = table
                    break
            if table_ibex is None:
                raise ValueError("Could not find the ticker table on the IBEX 35 Wikipedia page.")
            ibex_tickers = [s if '.' in s else f"{s}.MC" for s in table_ibex['Ticker'].tolist() if pd.notna(s)]
            all_tickers.update(ibex_tickers)
            print(f"✅ Fetched {len(ibex_tickers)} tickers from IBEX 35.")
        except Exception as e:
            print(f"⚠️ Could not fetch IBEX 35 list ({e}).")

    # Swiss MTI (Swiss market - broader than SMI)
    if MARKET_SELECTION.get("SWISS_MTI"):
        try:
            url_mti = "https://en.wikipedia.org/wiki/Swiss_Performance_Index"
            response_mti = requests.get(url_mti, headers=headers)
            response_mti.raise_for_status()
            tables_mti = pd.read_html(StringIO(response_mti.text))
            table_mti = None
            for table in tables_mti:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    table_mti = table
                    break
            if table_mti is not None:
                ticker_col = 'Ticker' if 'Ticker' in table_mti.columns else 'Symbol'
                mti_tickers = [s if '.' in s else f"{s}.SW" for s in table_mti[ticker_col].tolist() if pd.notna(s)]
                all_tickers.update(mti_tickers)
                print(f"✅ Fetched {len(mti_tickers)} tickers from Swiss MTI.")
            else:
                print(f"⚠️ Could not find ticker table on Swiss MTI Wikipedia page.")
        except Exception as e:
            print(f"⚠️ Could not fetch Swiss MTI list ({e}).")

    if not all_tickers:
        print("⚠️ No tickers fetched. Returning empty list.")
        return []

    string_tickers = {str(s) for s in all_tickers if pd.notna(s)}

    final_tickers = set()
    for ticker in string_tickers:
        s_ticker = ticker.strip()
        if '$' in s_ticker:
            continue

        if s_ticker.endswith(('.DE', '.MI', '.SW', '.PA', '.AS', '.HE', '.LS', '.BR', '.MC')):
            final_tickers.add(s_ticker)
        else:
            final_tickers.add(s_ticker.replace('.', '-'))

    # ✅ Always include benchmark tickers to ensure they're cached
    final_tickers.update(['QQQ', 'SPY'])

    # ✅ Include inverse ETFs for hedging strategies (from config.INVERSE_ETFS)
    try:
        from config import INVERSE_ETFS, ENABLE_INVERSE_ETFS
        if ENABLE_INVERSE_ETFS and INVERSE_ETFS:
            final_tickers.update(INVERSE_ETFS)
            print(f"✅ Added {len(INVERSE_ETFS)} inverse ETFs for hedging strategies.")
    except ImportError:
        pass

    # ❌ Remove known delisted/renamed ETFs and stocks
    DELISTED_TICKERS = {
        # Delisted leveraged/inverse ETFs
        'BGU', 'BGZ',  # Direxion Large Cap Bull/Bear 3X (delisted 2013)
        'CIU', 'CSJ',  # iShares Credit Bond ETFs (ticker changed)
        'CYB',         # WisdomTree Chinese Yuan (delisted)
        'DTN',         # WisdomTree Dividend ex-Financials (delisted)
        'EEB',         # Guggenheim BRIC ETF (delisted)
        'FBGX',        # UBS FI Enhanced Large Cap Growth ETN (delisted)
        'GAZ',         # iPath Bloomberg Natural Gas (delisted)
        'GVT',         # Grail American Beacon Large Cap Value (delisted)
        'GXF', 'GXG',  # Global X Nordic/Colombia ETFs (delisted)
        'HYLD',        # High Yield ETF (delisted)
        'IEIL', 'IEIS',  # iShares ETFs (delisted)
        'FLGE', 'FTLB',  # Credit Suisse ETNs (delisted)
        'GMMB', 'GMTB',  # iPath ETNs (delisted)
        'CHNA',        # Loncar China BioPharma ETF (delisted)
        'HOLD',        # AdvisorShares Sage Balanced Income (delisted)
        # Delisted/merged US stocks
        'AND',         # Andrea Electronics (delisted)
        'BBT',         # BB&T merged with SunTrust -> TFC (2019)
        'ATGE',        # Adtalem Global Education (no hourly data)
        'AXL',         # American Axle (no hourly data)
        # Delisted Swiss stocks (confirmed by Yahoo: "possibly delisted")
        'CSGN.SW',     # Credit Suisse (merged with UBS 2023)
        'ACHI.SW', 'AFP.SW', 'AIRE.SW', 'ARON.SW', 'BALN.SW', 'BLS.SW',
        'BOBNN.SW', 'BPDG.SW', 'CIE.SW', 'CLXN.SW', 'DUFN.SW', 'FI-N.SW',
        'GUR.SW', 'HELN.SW', 'HOCN.SW', 'HREN.SW',
        # Delisted German stocks
        'ECV.DE',
        # Delisted crypto
        'GRC-USD',     # GridCoin (no data)
    }
    final_tickers -= DELISTED_TICKERS
    if DELISTED_TICKERS & set(string_tickers):
        removed = DELISTED_TICKERS & set(string_tickers)
        print(f"   ℹ️ Removed {len(removed)} delisted/problematic tickers")

    print(f"Total unique tickers found: {len(final_tickers)}")
    return sorted(list(final_tickers))

def get_tickers_for_backtest(n: int = 10) -> List[str]:
    """Gets a list of n random tickers from the S&P 500."""
    fallback = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "LRCX"]
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in table[col].tolist()]
    except Exception as e:
        print(f"⚠️ Could not fetch S%26P 500 list ({e}). Using static fallback.")
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in fallback]

    import random
    # random.seed(SEED) # SEED is in config, not directly accessible here
    if len(tickers_all) > n:
        selected_tickers = random.sample(tickers_all, n)
    else:
        selected_tickers = tickers_all

    print(f"Randomly selected {n} tickers: {', '.join(selected_tickers)}")
    return selected_tickers

def _get_yahoo_1y_return(ticker: str, end_date: datetime) -> Optional[float]:
    """
    Fetch actual 1-year return from Yahoo Finance for comparison.
    Returns the actual 1Y% from Yahoo's live data.
    """
    try:
        start_date = end_date - timedelta(days=365)
        # CRITICAL: Use global lock - yfinance has threading bugs that cause
        # data corruption when called concurrently
        with _yfinance_lock:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=start_date, end=end_date + timedelta(days=1))

        if hist.empty or len(hist) < 10:
            return None

        close_prices = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if close_prices.empty or len(close_prices) < 2:
            return None

        start_price = float(close_prices.iloc[0])
        end_price = float(close_prices.iloc[-1])

        if start_price <= 0 or not np.isfinite(start_price) or not np.isfinite(end_price):
            return None

        return_pct = ((end_price - start_price) / start_price) * 100
        if not np.isfinite(return_pct):
            return None
        return float(return_pct)
    except Exception:
        return None

def _prepare_ticker_data_worker(args: Tuple) -> Optional[Tuple[str, pd.DataFrame]]:
    """Worker function to prepare data for a single ticker (for parallel processing)."""
    ticker, ticker_data_slice = args

    try:
        # Find Close column
        close_col = None
        for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
            if attr in ticker_data_slice.columns:
                close_col = attr
                break

        if close_col is None:
            if ticker == 'SNDK':
                print(f"   🔍 DEBUG SNDK PREP: No close column in {list(ticker_data_slice.columns)}")
            return None

        # Create time series
        s = pd.to_numeric(ticker_data_slice[close_col], errors='coerce')
        s = s.ffill().bfill()

        if s.dropna().shape[0] < 2:
            if ticker == 'SNDK':
                print(f"   🔍 DEBUG SNDK PREP: < 2 rows after cleanup (shape={s.dropna().shape[0]})")
            return None

        ticker_df = pd.DataFrame({'Close': s})

        if ticker == 'SNDK':
            print(f"   🔍 DEBUG SNDK PREP: OK - {len(ticker_df)} rows, first={s.iloc[0]:.2f}, last={s.iloc[-1]:.2f}")

        return (ticker, ticker_df)

    except Exception as e:
        if ticker == 'SNDK':
            print(f"   🔍 DEBUG SNDK PREP: Exception {e}")
        return None


def _calculate_performance_worker(params: Tuple[str, pd.DataFrame]) -> Optional[Tuple[str, float, pd.DataFrame]]:
    """Robust 1Y performance: tolerate gaps, weekends, and column variants."""
    ticker, df_1y = params
    try:
        candidates = ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']
        price_col = next((c for c in candidates if c in df_1y.columns), None)
        if price_col is None:
            lower = {c.lower(): c for c in df_1y.columns}
            for key in ['close', 'adj close']:
                if key in lower:
                    price_col = lower[key]
                    break
        if price_col is None:
            if ticker == 'SNDK':
                print(f"   🔍 DEBUG SNDK: No price column found in {list(df_1y.columns)}")
            return None

        s = pd.to_numeric(df_1y[price_col], errors='coerce').ffill().bfill()
        s = s.dropna()
        if s.empty or len(s) < 2:
            if ticker == 'SNDK':
                print(f"   🔍 DEBUG SNDK: Empty or < 2 rows after cleanup (len={len(s)})")
            return None

        start_price = s.iloc[0]
        end_price = s.iloc[-1]

        # Debug SNDK
        if ticker == 'SNDK':
            print(f"   🔍 DEBUG SNDK: rows={len(s)}, start=${start_price:.2f}, end=${end_price:.2f}, perf={(end_price/start_price-1)*100:.1f}%")

        if start_price <= 0:
            return None

        perf_1y = (end_price / start_price - 1.0) * 100.0

        if np.isfinite(perf_1y):
            out = df_1y.copy()
            if price_col != 'Close':
                out = out.rename(columns={price_col: 'Close'})
            return (ticker, float(perf_1y), out)
    except Exception as e:
        if ticker == 'SNDK':
            print(f"   🔍 DEBUG SNDK: Exception {e}")
        return None
    return None

def find_top_performers(
    all_available_tickers: List[str],
    all_tickers_data: pd.DataFrame,
    return_tickers: bool = False,
    n_top: int = N_TOP_TICKERS,
    fcf_min_threshold: float = 0.0,
    ebitda_min_threshold: float = 0.0,
    performance_end_date: datetime = None
):
    """
    Screens pre-fetched data for the top N performers and returns a list of (ticker, performance) tuples.
    """
    if all_tickers_data.empty:
        print("❌ No ticker data provided to find_top_performers. Exiting.")
        return []

    # Use provided performance end date, or fall back to data's max date
    if performance_end_date is not None:
        end_date = performance_end_date
    else:
        # ✅ FIX: Handle both long-format (with 'date' column) and wide-format (DatetimeIndex)
        if 'date' in all_tickers_data.columns:
            # Long format: dates are in 'date' column
            end_date = pd.to_datetime(all_tickers_data['date']).max()
        else:
            # Wide format: dates are in index
            end_date = all_tickers_data.index.max()

    start_date = end_date - timedelta(days=365)
    ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    final_benchmark_perf = -np.inf
    ytd_benchmark_perf = -np.inf
    if USE_PERFORMANCE_BENCHMARK:
        print("- Calculating 1-Year Performance Benchmarks...")
        benchmark_perfs = {}

        # ✅ Use pre-fetched data from all_tickers_data instead of re-downloading
        for bench_ticker in ['QQQ', 'SPY']:
            try:
                # Extract benchmark data from all_tickers_data (long format)
                if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
                    # Check if ticker exists in dataset
                    ticker_check = all_tickers_data[all_tickers_data['ticker'] == bench_ticker]
                    if ticker_check.empty:
                        print(f"  ⚠️ {bench_ticker}: Not in dataset (available tickers: {sorted(all_tickers_data['ticker'].unique())[:10]}...)")
                        continue

                    bench_data = all_tickers_data[
                        (all_tickers_data['ticker'] == bench_ticker) &
                        (all_tickers_data['date'] >= start_date) &
                        (all_tickers_data['date'] <= end_date)
                    ].sort_values('date')

                    if bench_data.empty:
                        print(f"  ⚠️ {bench_ticker}: No data in date range {start_date.date()} to {end_date.date()}")
                        print(f"      Available date range: {ticker_check['date'].min().date()} to {ticker_check['date'].max().date()}")
                        continue

                    if 'Close' not in bench_data.columns:
                        print(f"  ⚠️ {bench_ticker}: 'Close' column not found (columns: {list(bench_data.columns)})")
                        continue

                    # Drop NaN values
                    valid_prices = bench_data['Close'].dropna()
                    if len(valid_prices) < 2:
                        print(f"  ⚠️ {bench_ticker}: Insufficient valid prices ({len(valid_prices)} non-NaN values)")
                        continue

                    start_price = valid_prices.iloc[0]
                    end_price = valid_prices.iloc[-1]

                    if pd.isna(start_price) or pd.isna(end_price):
                        print(f"  ⚠️ {bench_ticker}: NaN prices (start={start_price}, end={end_price})")
                        continue

                    if start_price > 0:
                        perf = ((end_price - start_price) / start_price) * 100
                        benchmark_perfs[bench_ticker] = perf
                        print(f"  ✅ {bench_ticker} 1-Year Performance: {perf:.2f}% (${start_price:.2f} → ${end_price:.2f})")
                    else:
                        print(f"  ⚠️ {bench_ticker}: Invalid start price ({start_price})")
                else:
                    # Fallback to old method if data is in wide format
                    df = load_prices_robust(bench_ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        start_price = df['Close'].iloc[0]
                        end_price = df['Close'].iloc[-1]
                        if start_price > 0:
                            perf = ((end_price - start_price) / start_price) * 100
                            benchmark_perfs[bench_ticker] = perf
                            print(f"  ✅ {bench_ticker} 1-Year Performance: {perf:.2f}%")
            except Exception as e:
                print(f"  ⚠️ Could not calculate {bench_ticker} performance: {e}")
                import traceback
                traceback.print_exc()

        if not benchmark_perfs:
            print("❌ Could not calculate any benchmark performance. Cannot proceed.")
            return []

        final_benchmark_perf = max(benchmark_perfs.values())
        print(f"  📈 Using final 1-Year performance benchmark of {final_benchmark_perf:.2f}%")
    else:
        print("ℹ️ Performance benchmark is disabled. All tickers will be considered.")

    print("🔍 Calculating 1-Year performance from pre-fetched data...")

    # ✅ FIX: Handle both long-format and wide-format data
    if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
        # Long format: use groupby for FAST splitting (much faster than filtering in a loop!)
        print(f"   📊 Filtering data for period {start_date.date()} to {end_date.date()}...", flush=True)
        all_data = all_tickers_data[
            (all_tickers_data['date'] >= start_date) &
            (all_tickers_data['date'] <= end_date)
        ].copy()

        # Use groupby to split data in ONE operation (instead of 5644 filter operations!)
        print(f"   📋 Splitting data by ticker using groupby (fast)...", flush=True)
        sys.stdout.flush()

        grouped = all_data.groupby('ticker')
        valid_tickers = list(grouped.groups.keys())
        print(f"   📊 Found {len(valid_tickers)} tickers with data", flush=True)

        # Debug: Check if SNDK is in the valid tickers
        if 'SNDK' in valid_tickers:
            sndk_data = grouped.get_group('SNDK')
            print(f"   🔍 DEBUG SNDK: IN valid_tickers, rows={len(sndk_data)}, dates={sndk_data['date'].min()} to {sndk_data['date'].max()}")
        else:
            print(f"   🔍 DEBUG SNDK: NOT in valid_tickers (not in 1Y period)")

        # Build prep_args using pre-grouped data (very fast!)
        print(f"   🔧 Building parameter list...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        prep_args = []
        for ticker in tqdm(valid_tickers, desc="Building params", ncols=100):
            try:
                ticker_data = grouped.get_group(ticker).copy()
                ticker_data = ticker_data.set_index('date')
                prep_args.append((ticker, ticker_data))
            except KeyError:
                pass

        # Parallelize actual data preparation (finding Close column, cleaning data)
        num_prep_workers = min(NUM_PROCESSES, len(prep_args)) if prep_args else 1
        prep_chunksize = max(1, len(prep_args) // (num_prep_workers * 2))

        print(f"   🚀 Processing {len(prep_args)} tickers with {num_prep_workers} workers (chunksize={prep_chunksize})", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        params = []
        try:
            with Pool(processes=num_prep_workers, maxtasksperchild=20) as pool:
                prep_results = list(tqdm(
                    pool.imap(_prepare_ticker_data_worker, prep_args, chunksize=prep_chunksize),
                    total=len(prep_args),
                    desc="Processing ticker data",
                    ncols=100
                ))
                params = [r for r in prep_results if r is not None]
        except Exception as e:
            print(f"   ⚠️ Parallel processing failed ({e}), falling back to sequential...", flush=True)
            params = []
            for args in tqdm(prep_args, desc="Processing ticker data (sequential)", ncols=100):
                result = _prepare_ticker_data_worker(args)
                if result is not None:
                    params.append(result)
        gc.collect()  # Release semaphores after Pool closes (WSL fix)
    else:
        # Wide format: use original logic
        print(f"   📊 Filtering data for period {start_date.date()} to {end_date.date()}...", flush=True)
        all_data = all_tickers_data.loc[start_date:end_date]
        valid_tickers = list(all_data.columns.get_level_values(1).unique())

        print(f"   📋 Building parameters for {len(valid_tickers)} tickers...", flush=True)

        # Prepare data slices (wide format is already column-based, so this is fast)
        prep_args = []
        for ticker in tqdm(valid_tickers, desc="Building params", ncols=100):
            try:
                # Extract close column
                close_key = None
                for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
                    if (attr, ticker) in all_data.columns:
                        close_key = (attr, ticker)
                        break

                if close_key is not None:
                    s = all_data.loc[:, close_key]
                    ticker_df = pd.DataFrame({'Close': s})
                    ticker_df.index = all_data.index
                    prep_args.append((ticker, ticker_df))
            except KeyError:
                pass

        # Parallelize data processing
        num_prep_workers = min(NUM_PROCESSES, len(prep_args)) if prep_args else 1
        prep_chunksize = max(1, len(prep_args) // (num_prep_workers * 2))

        print(f"   🚀 Processing {len(prep_args)} tickers with {num_prep_workers} workers (chunksize={prep_chunksize})", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        params = []
        try:
            with Pool(processes=num_prep_workers, maxtasksperchild=20) as pool:
                prep_results = list(tqdm(
                    pool.imap(_prepare_ticker_data_worker, prep_args, chunksize=prep_chunksize),
                    total=len(prep_args),
                    desc="Processing ticker data",
                    ncols=100
                ))
                params = [r for r in prep_results if r is not None]
        except Exception as e:
            print(f"   ⚠️ Parallel processing failed ({e}), falling back to sequential...", flush=True)
            params = []
            for args in tqdm(prep_args, desc="Processing ticker data (sequential)", ncols=100):
                result = _prepare_ticker_data_worker(args)
                if result is not None:
                    params.append(result)
        gc.collect()  # Release semaphores after Pool closes (WSL fix)

    # Prepare for parallel processing
    if not params:
        print("   ⚠️ No valid tickers found for performance calculation")
        return []

    print(f"   📊 Prepared {len(params)} tickers for performance calculation", flush=True)

    all_tickers_performance_with_df = []
    # Use configured number of processes for optimal performance
    num_workers = min(NUM_PROCESSES, len(params)) if params else 1
    chunksize = max(1, len(params) // (num_workers * 4)) if params else 1  # Optimal chunking

    print(f"   🚀 Starting parallel calculation with {num_workers} workers (chunksize={chunksize})", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        with Pool(processes=num_workers, maxtasksperchild=20) as pool:
            results = pool.imap(_calculate_performance_worker, params, chunksize=chunksize)
            for res in tqdm(
                results,
                total=len(params),
                desc="Calculating 1Y Performance",
                ncols=100
            ):
                if res:
                    all_tickers_performance_with_df.append(res)
    except Exception as e:
        print(f"   ⚠️ Parallel calculation failed ({e}), falling back to sequential...", flush=True)
        for p in tqdm(params, desc="Calculating 1Y Performance (sequential)", ncols=100):
            result = _calculate_performance_worker(p)
            if result:
                all_tickers_performance_with_df.append(result)
    gc.collect()  # Release semaphores after Pool closes (WSL fix)

    print(f"   ✅ Performance calculation complete! Processed {len(all_tickers_performance_with_df)}/{len(params)} tickers")

    if not all_tickers_performance_with_df:
        print("❌ No tickers with valid 1-Year performance found. Aborting.")
        return []

    sorted_all_tickers_performance_with_df = sorted(all_tickers_performance_with_df, key=lambda item: item[1], reverse=True)

    # Note: Removed arbitrary 1000% cap - legitimate high performers should not be excluded

    if n_top > 0:
        final_performers_for_selection = sorted_all_tickers_performance_with_df[:n_top]
        print(f"\n✅ Selected top {len(final_performers_for_selection)} tickers based on 1-Year performance.")
    else:
        final_performers_for_selection = sorted_all_tickers_performance_with_df
        print(f"\n✅ Analyzing all {len(final_performers_for_selection)} tickers (N_TOP_TICKERS is {n_top}).")

    print(f"🔍 Applying performance benchmarks for selected tickers in parallel...")

    finalize_params = [
        (ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, USE_PERFORMANCE_BENCHMARK)
        for ticker, perf_1y, df_1y in final_performers_for_selection
    ]
    performance_data = []

    # Use configured number of processes
    num_workers_bench = min(NUM_PROCESSES, len(finalize_params)) if finalize_params else 1
    chunksize_bench = max(1, len(finalize_params) // (num_workers_bench * 4)) if finalize_params else 1

    print(f"   Using {num_workers_bench} parallel workers with chunksize={chunksize_bench}")

    with Pool(processes=num_workers_bench, maxtasksperchild=50) as pool:
        results = list(tqdm(
            pool.imap(_finalize_single_ticker_performance, finalize_params, chunksize=chunksize_bench),
            total=len(finalize_params),
            desc="Finalizing Top Performers",
            ncols=100
        ))
        for res in results:
            if res:
                performance_data.append(res)
    gc.collect()  # Release semaphores after Pool closes (WSL fix)

    if USE_PERFORMANCE_BENCHMARK:
        print(f"\n✅ Found {len(performance_data)} stocks that passed the performance benchmarks.")
    else:
        print(f"\n✅ Found {len(performance_data)} stocks for analysis (performance benchmark disabled).")

    if not performance_data:
        return []

    final_performers = performance_data

    if fcf_min_threshold is not None or ebitda_min_threshold is not None:
        print(f"  🔍 Screening {len(final_performers)} strong performers for fundamental metrics in parallel...")

        fundamental_screen_params = [
            (ticker, perf_1y, fcf_min_threshold, ebitda_min_threshold)
            for ticker, perf_1y in final_performers
        ]
        screened_performers = []

        # Use configured number of processes
        num_workers_fund = min(NUM_PROCESSES, len(fundamental_screen_params)) if fundamental_screen_params else 1
        chunksize_fund = max(1, len(fundamental_screen_params) // (num_workers_fund * 4)) if fundamental_screen_params else 1

        print(f"   Using {num_workers_fund} parallel workers with chunksize={chunksize_fund}")

        with Pool(processes=num_workers_fund, maxtasksperchild=50) as pool:
            results = list(tqdm(
                pool.imap(_apply_fundamental_screen_worker, fundamental_screen_params, chunksize=chunksize_fund),
                total=len(fundamental_screen_params),
                desc="Applying fundamental screens",
                ncols=100
            ))
            for res in results:
                if res:
                    screened_performers.append(res)
        gc.collect()  # Release semaphores after Pool closes (WSL fix)

        print(f"  ✅ Found {len(screened_performers)} stocks passing the fundamental screens.")
        final_performers = screened_performers

    # ✅ Fetch actual Yahoo Finance 1-year returns for comparison (parallel)
    print(f"\n  📊 Fetching actual Yahoo 1Y returns for comparison (top {min(50, len(final_performers))} tickers)...")
    yahoo_returns = {}

    # Only fetch for top 50 to avoid rate limiting
    tickers_to_check = [ticker for ticker, _ in final_performers[:50]]

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_get_yahoo_1y_return, ticker, end_date): ticker
            for ticker in tickers_to_check
        }

        for future in tqdm(as_completed(future_to_ticker), total=len(tickers_to_check), desc="Yahoo comparison", ncols=100):
            ticker = future_to_ticker[future]
            try:
                yahoo_return = future.result()
                if yahoo_return is not None:
                    yahoo_returns[ticker] = yahoo_return
            except Exception:
                pass

    # ✅ ALWAYS display comparison table (even when return_tickers=True)
    print(f"\n\n🏆 Top Performers with Yahoo Finance Comparison 🏆")
    print("-" * 110)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Cached 1Y':>12} | {'Yahoo 1Y':>12} | {'Diff %':>12} | {'Status':>15}")
    print("-" * 110)

    # Show top 25 for comparison
    for i, (ticker, perf) in enumerate(final_performers[:25], 1):
        yahoo_perf = yahoo_returns.get(ticker)
        if yahoo_perf is not None:
            if abs(yahoo_perf) > 1e-9:
                diff_pct = ((perf - yahoo_perf) / abs(yahoo_perf)) * 100.0
            else:
                diff_pct = 0.0 if abs(perf) <= 1e-9 else float('inf')

            # Flag suspicious discrepancies when cached result differs materially from Yahoo.
            status = "⚠️ LARGE DIFF" if abs(diff_pct) > 10 else "✅ Match"
            diff_display = f"{diff_pct:+11.2f}%" if np.isfinite(diff_pct) else f"{'inf':>11}%"
            print(f"{i:<5} | {ticker:<10} | {perf:11.2f}% | {yahoo_perf:11.2f}% | {diff_display} | {status:>15}")
        else:
            print(f"{i:<5} | {ticker:<10} | {perf:11.2f}% | {'N/A':>12} | {'N/A':>12} | {'No Yahoo data':>15}")

    if len(final_performers) > 25:
        print(f"   ... and {len(final_performers) - 25} more tickers")

    print("-" * 110)

    # Summary stats
    if yahoo_returns:
        matched_tickers = [t for t in tickers_to_check if t in yahoo_returns]
        if matched_tickers:
            avg_cached = np.mean([perf for ticker, perf in final_performers if ticker in matched_tickers])
            avg_yahoo = np.mean([yahoo_returns[t] for t in matched_tickers])
            avg_diff_pct = (
                ((avg_cached - avg_yahoo) / abs(avg_yahoo)) * 100.0
                if abs(avg_yahoo) > 1e-9
                else 0.0 if abs(avg_cached) <= 1e-9 else float('inf')
            )
            avg_diff_display = f"{avg_diff_pct:+.1f}%" if np.isfinite(avg_diff_pct) else "inf%"
            print(f"\n📊 Summary ({len(matched_tickers)} tickers): Avg Cached = {avg_cached:.1f}%, Avg Yahoo = {avg_yahoo:.1f}%, Avg Diff = {avg_diff_display}")

            # Count large relative discrepancies
            perf_dict = {ticker: perf for ticker, perf in final_performers}
            large_diffs = 0
            for t in matched_tickers:
                yahoo_perf = yahoo_returns[t]
                cached_perf = perf_dict.get(t, 0)
                if abs(yahoo_perf) > 1e-9:
                    rel_diff_pct = ((cached_perf - yahoo_perf) / abs(yahoo_perf)) * 100.0
                else:
                    rel_diff_pct = 0.0 if abs(cached_perf) <= 1e-9 else float('inf')
                if abs(rel_diff_pct) > 10:
                    large_diffs += 1
            if large_diffs > 0:
                print(f"⚠️  WARNING: {large_diffs} tickers have >10% relative discrepancy - consider clearing cache and re-downloading!")

    if return_tickers:
        return final_performers

    print(f"\n\n🏆 Stocks Outperforming {final_benchmark_perf:.2f}% (Full List) 🏆")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Performance':>15}")
    print("-" * 60)

    for i, (ticker, perf) in enumerate(final_performers, 1):
        print(f"{i:<5} | {ticker:<10} | {perf:14.2f}%")

    print("-" * 60)

    return [ticker for ticker, perf in final_performers]

def _finalize_single_ticker_performance(params: Tuple) -> Optional[Tuple[str, float]]:
    """Worker to apply performance benchmarks."""
    ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, use_performance_benchmark = params

    if use_performance_benchmark and perf_1y < final_benchmark_perf:
        return None

    # YTD performance calculation removed since YTD support was removed
    return (ticker, perf_1y)

def _apply_fundamental_screen_worker(params: Tuple) -> Optional[Tuple[str, float]]:
    """Worker to apply fundamental screens using yfinance with proper fallback."""
    ticker, perf_1y, fcf_min_threshold, ebitda_min_threshold = params

    # Use yfinance directly for fundamental data - same as _fetch_financial_data but simplified
    try:
        import yfinance as yf
        import time

        # Small delay to be respectful to APIs
        time.sleep(0.1)  # Shorter delay for multiprocessing

        yf_ticker = yf.Ticker(ticker)
        financial_data = {}

        # Get income statement data (EBITDA)
        try:
            income_stmt = yf_ticker.quarterly_income_stmt
            if not income_stmt.empty and 'EBITDA' in income_stmt.index:
                latest_ebitda = income_stmt.loc['EBITDA'].iloc[-1]
                if not pd.isna(latest_ebitda):
                    financial_data['EBITDA'] = float(latest_ebitda)
        except Exception as e:
            pass  # Silently continue if this fails

        # Get cash flow data (Free Cash Flow)
        try:
            cash_flow = yf_ticker.quarterly_cash_flow
            if not cash_flow.empty and 'Free Cash Flow' in cash_flow.index:
                latest_fcf = cash_flow.loc['Free Cash Flow'].iloc[-1]
                if not pd.isna(latest_fcf):
                    financial_data['FCF'] = float(latest_fcf)
        except Exception as e:
            pass  # Silently continue if this fails

        # Apply thresholds (fcf_min_threshold and ebitda_min_threshold are 0.0 by default)
        should_exclude = False

        if 'EBITDA' in financial_data and ebitda_min_threshold is not None:
            if financial_data['EBITDA'] < ebitda_min_threshold:
                should_exclude = True

        if 'FCF' in financial_data and fcf_min_threshold is not None:
            if financial_data['FCF'] < fcf_min_threshold:
                should_exclude = True

        if should_exclude:
            return None  # Exclude stocks that don't meet financial criteria

        # Include stocks that pass financial screening or have no data available
        return (ticker, perf_1y)

    except Exception as e:
        # If anything fails, include the stock by default (fail-open approach)
        return (ticker, perf_1y)
