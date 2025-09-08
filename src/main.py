# -*- coding: utf-8 -*-
"""
Trading AI — Improved Rule-Based System with Optional ML Gate
- Headless-safe Matplotlib (Agg)
- Stooq-first data ('.US' fallback), optional Yahoo fallback
- UTC-safe timestamps; local CSV cache
- YTD top-picker (kept simple)
- Strategy: SMA crossover + ATR trailing stop + take-profit (multiples)
- Position sizing by risk (1% of capital, ATR-based)
- Optional ML classification gate (5-day horizon) to filter entries
"""

from __future__ import annotations

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
import sys
import codecs
from io import StringIO
from multiprocessing import Pool, cpu_count
import joblib # Added for model saving/loading

# --- Force UTF-8 output on Windows ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ---------- Matplotlib (headless-safe) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

# Optional Stooq provider
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# ============================
# Configuration / Hyperparams
# ============================

SEED                    = 42
np.random.seed(SEED)

# --- Provider & caching
DATA_PROVIDER           = 'yahoo'    # 'stooq' or 'yahoo'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
CACHE_DAYS              = 7

# --- Universe / selection
MARKET_SELECTION = {
    "NASDAQ_ALL": True,
    "NASDAQ_100": True,
    "SP500": True,
    "DOW_JONES": True,
    "POPULAR_ETFS": False,
    "CRYPTO": False,
    "DAX": True,
    "MDAX": True,
    "SMI": False,
    "FTSE_MIB": False,
}
N_TOP_TICKERS           = 10          # Set to 0 to disable the limit and run on all performers
BATCH_DOWNLOAD_SIZE     = 500
PAUSE_BETWEEN_BATCHES   = 5.0

# --- Backtest & training windows
BACKTEST_DAYS           = 365        # 1 year for backtest
TRAIN_LOOKBACK_DAYS     = 360        # more data for model (e.g., 1 year)

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 20
STRAT_SMA_LONG          = 100
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 3.5
ATR_MULT_TP             = 0.0        # 0 disables hard TP; rely on trailing
RISK_PER_TRADE          = 0.01       # 1% of capital
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_BUY           = 0.5       # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.5       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.01       # 1% target for buy/sell classification
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # re-enable market filter
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = False   # Set to True to enable benchmark filtering

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = False

# ============================
# Helpers
# ============================

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _to_utc(ts):
    """Return a pandas UTC-aware Timestamp for any datetime-like input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')

def _fetch_from_stooq(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV from Stooq. Try both 'TICKER' and 'TICKER.US'."""
    if pdr is None:
        return pd.DataFrame()
    try:
        df = pdr.DataReader(ticker, "stooq", start, end)
        if (df is None or df.empty) and not ticker.upper().endswith('.US'):
            try:
                df = pdr.DataReader(f"{ticker}.US", "stooq", start, end)
            except Exception:
                pass
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_index()
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()

# ============================
# Data access
# ============================

def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """A wrapper for load_prices that handles rate limiting with retries and other common API errors."""
    import time
    import random
    max_retries = 5
    base_wait_time = 10  # seconds, increased for more tolerance

    for attempt in range(max_retries):
        try:
            return load_prices(ticker, start, end)
        except Exception as e:
            error_str = str(e).lower()
            # Handle YFTzMissingError for delisted stocks gracefully
            if "yftzmissingerror" in error_str or "no timezone found" in error_str:
                print(f"  ℹ️ Skipping {ticker}: Data not available (possibly delisted).")
                return pd.DataFrame()
            
            # Handle rate limiting with exponential backoff
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⚠️ Rate limited trying to fetch {ticker}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # For other unexpected errors, log it and fail for this ticker
                print(f"  ⚠️ An unexpected error occurred for {ticker}: {e}. Skipping.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to load data for {ticker} after {max_retries} retries due to persistent rate limiting.")
    return pd.DataFrame()

def _download_batch_robust(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Wrapper for yf.download for batches with retry logic."""
    import time
    import random
    max_retries = 5
    base_wait_time = 10  # seconds

    for attempt in range(max_retries):
        try:
            # Set threads to False to avoid potential conflicts with yfinance's internal threading
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True, threads=20, keepna=False)
            
            # Critical check: If the dataframe is empty or all values are NaN, it's a failed download.
            if data.empty or data.isnull().all().all():
                # This will be caught by the except block and trigger a retry
                raise ValueError("Batch download failed: DataFrame is empty or all-NaN.")
                
            return data
        except Exception as e:
            error_str = str(e).lower()
            print(f"  ⚠️ Error {error_str}")
            # Catch common yfinance multi-ticker failure messages
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "failed download" in error_str or "batch download failed" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                print(f"  ⚠️ Rate limited on batch download. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ An unexpected error occurred during batch download: {e}. Skipping batch.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to download batch data after {max_retries} retries.")
    return pd.DataFrame()

def _fetch_financial_data(ticker: str) -> pd.DataFrame:
    """Fetch key financial metrics from yfinance and prepare them for merging."""
    yf_ticker = yf.Ticker(ticker)
    
    financial_data = {}
    
    # Fetch income statement (quarterly)
    try:
        income_statement = yf_ticker.quarterly_income_stmt
        if not income_statement.empty:
            # Select relevant metrics and transpose
            metrics = ['Total Revenue', 'Net Income', 'EBITDA']
            for metric in metrics:
                if metric in income_statement.index:
                    financial_data[metric] = income_statement.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch income statement for {ticker}: {e}")

    # Fetch balance sheet (quarterly)
    try:
        balance_sheet = yf_ticker.quarterly_balance_sheet
        if not balance_sheet.empty:
            metrics = ['Total Assets', 'Total Liabilities']
            for metric in metrics:
                if metric in balance_sheet.index:
                    financial_data[metric] = balance_sheet.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch balance sheet for {ticker}: {e}")

    # Fetch cash flow (quarterly)
    try:
        cash_flow = yf_ticker.quarterly_cash_flow
        if not cash_flow.empty:
            metrics = ['Free Cash Flow']
            for metric in metrics:
                if metric in cash_flow.index:
                    financial_data[metric] = cash_flow.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch cash flow for {ticker}: {e}")

    if not financial_data:
        return pd.DataFrame()

    df_financial = pd.DataFrame(financial_data).T
    df_financial.index.name = 'Metric'
    df_financial = df_financial.T # Transpose back to have dates as index
    df_financial.index = pd.to_datetime(df_financial.index, utc=True)
    df_financial.index.name = "Date"
    
    # Rename columns to be more feature-friendly
    df_financial = df_financial.rename(columns={
        'Total Revenue': 'Fin_Revenue',
        'Net Income': 'Fin_NetIncome',
        'Total Assets': 'Fin_TotalAssets',
        'Total Liabilities': 'Fin_TotalLiabilities',
        'Free Cash Flow': 'Fin_FreeCashFlow',
        'EBITDA': 'Fin_EBITDA'
    })
    
    # Ensure all financial columns are numeric
    for col in df_financial.columns:
        df_financial[col] = pd.to_numeric(df_financial[col], errors='coerce')

    return df_financial.sort_index()


def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and clean data from the selected provider, with an improved local caching mechanism."""
    _ensure_dir(DATA_CACHE_DIR)
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
    financial_cache_file = DATA_CACHE_DIR / f"{ticker}_financials.csv"
    
    # --- Check price cache first ---
    price_df = pd.DataFrame()
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):
            try:
                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')
                price_df = cached_df.loc[(cached_df.index >= _to_utc(start)) & (cached_df.index <= _to_utc(end))].copy()
            except Exception as e:
                print(f"⚠️ Could not read or slice price cache file for {ticker}: {e}. Refetching prices.")

    # --- If not in price cache or cache is old, fetch a broad range of data ---
    if price_df.empty:
        fetch_start = datetime.now(timezone.utc) - timedelta(days=1000) # Fetch a generous amount of data
        fetch_end = datetime.now(timezone.utc)
        start_utc = _to_utc(fetch_start)
        end_utc   = _to_utc(fetch_end)
        
        provider = DATA_PROVIDER.lower()
        
        if provider == 'stooq':
            stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
            if not stooq_df.empty:
                price_df = stooq_df.copy()
            elif USE_YAHOO_FALLBACK:
                try:
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                    if downloaded_df is not None:
                        price_df = downloaded_df.dropna()
                except Exception as e:
                    raise e
        else:
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None:
                    price_df = downloaded_df.dropna()
            except Exception as e:
                raise e
            if price_df.empty and pdr is not None:
                stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
                if stooq_df.empty and not ticker.upper().endswith('.US'):
                    stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
                if not stooq_df.empty:
                    price_df = stooq_df.copy()

        if price_df.empty:
            return pd.DataFrame()

        # Clean and normalize the downloaded data
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df.columns = [str(col).capitalize() for col in price_df.columns]
        if "Close" not in price_df.columns and "Adj close" in price_df.columns:
            price_df = price_df.rename(columns={"Adj close": "Close"})

        if "Close" not in price_df.columns:
            return pd.DataFrame()

        price_df.index = pd.to_datetime(price_df.index, utc=True)
        price_df.index.name = "Date"
        price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")
        price_df = price_df.dropna(subset=["Close"])
        price_df = price_df.ffill().bfill()

        # --- Save the entire fetched price data to cache ---
        if not price_df.empty:
            try:
                price_df.to_csv(cache_file)
            except Exception as e:
                print(f"⚠️ Could not write price cache file for {ticker}: {e}")
                
    # --- Fetch and merge financial data ---
    financial_df = pd.DataFrame()
    if financial_cache_file.exists():
        file_mod_time = datetime.fromtimestamp(financial_cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=CACHE_DAYS * 4): # Financials update less frequently
            try:
                financial_df = pd.read_csv(financial_cache_file, index_col='Date', parse_dates=True)
                if financial_df.index.tzinfo is None:
                    financial_df.index = financial_df.index.tz_localize('UTC')
                else:
                    financial_df.index = financial_df.index.tz_convert('UTC')
            except Exception as e:
                print(f"⚠️ Could not read financial cache file for {ticker}: {e}. Refetching financials.")
    
    if financial_df.empty:
        financial_df = _fetch_financial_data(ticker)
        if not financial_df.empty:
            try:
                financial_df.to_csv(financial_cache_file)
            except Exception as e:
                print(f"⚠️ Could not write financial cache file for {ticker}: {e}")

    if not financial_df.empty and not price_df.empty:
        # Merge financial data by forward-filling the latest available financial report
        # Reindex financial_df to cover the full date range of price_df
        full_date_range = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='D', tz='UTC')
        financial_df_reindexed = financial_df.reindex(full_date_range)
        financial_df_reindexed = financial_df_reindexed.ffill()
        
        # Merge with price data
        final_df = price_df.merge(financial_df_reindexed, left_index=True, right_index=True, how='left')
        # Fill any remaining NaNs in financial features (e.g., at the very beginning)
        final_df.fillna(0, inplace=True)
    else:
        final_df = price_df.copy()

    # Return the specifically requested slice
    return final_df.loc[(final_df.index >= _to_utc(start)) & (final_df.index <= _to_utc(end))].copy()

# ============================
# Ticker discovery
# ============================

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
    random.seed(SEED)
    if len(tickers_all) > n:
        selected_tickers = random.sample(tickers_all, n)
    else:
        selected_tickers = tickers_all
    
    print(f"Randomly selected {n} tickers: {', '.join(selected_tickers)}")
    return selected_tickers

def get_all_tickers() -> List[str]:
    """
    Gets a list of tickers from the markets selected in the configuration.
    """
    all_tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    # --- US Tickers ---
    if MARKET_SELECTION.get("NASDAQ_ALL"):
        try:
            url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
            df = pd.read_csv(url, sep='|')
            df_clean = df.iloc[:-1]
            # Include ETFs by removing the 'ETF' == 'N' filter
            nasdaq_tickers = df_clean[df_clean['Test Issue'] == 'N']['Symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            print(f"✅ Fetched {len(nasdaq_tickers)} tickers from NASDAQ (including ETFs).")
        except Exception as e:
            print(f"⚠️ Could not fetch full NASDAQ list ({e}).")

    if MARKET_SELECTION.get("NASDAQ_100"):
        try:
            import requests
            url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response_nasdaq = requests.get(url_nasdaq, headers=headers)
            response_nasdaq.raise_for_status()
            table_nasdaq = pd.read_html(StringIO(response_nasdaq.text))[4]
            nasdaq_100_tickers = [s.replace('.', '-') for s in table_nasdaq['Ticker'].tolist()]
            all_tickers.update(nasdaq_100_tickers)
            print(f"✅ Fetched {len(nasdaq_100_tickers)} tickers from NASDAQ 100.")
        except Exception as e:
            print(f"⚠️ Could not fetch NASDAQ 100 list ({e}).")

    if MARKET_SELECTION.get("SP500"):
        try:
            import requests
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
            import requests
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
            print(f"✅ Fetched {len(dow_tickers)} tickers from Dow Jones.")
        except Exception as e:
            print(f"⚠️ Could not fetch Dow Jones list ({e}).")

    if MARKET_SELECTION.get("POPULAR_ETFS"):
        try:
            import requests
            from bs4 import BeautifulSoup
            url_etf = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
            response_etf = requests.get(url_etf, headers=headers)
            response_etf.raise_for_status()
            
            soup = BeautifulSoup(response_etf.text, 'html.parser')
            etf_tickers = set()
            
            # Find all list items, which contain the ETF info
            for li in soup.find_all('li'):
                text = li.get_text()
                # Use regex to find patterns like (NYSE Arca: ITOT) or (NASDAQ|QQQ)
                match = re.search(r'\((?:NYSE\sArca|NASDAQ)[^)]*:([^)]+)\)', text)
                if match:
                    ticker = match.group(1).strip()
                    # Clean up the ticker symbol
                    ticker = ticker.replace('.', '-')
                    etf_tickers.add(ticker)

            if not etf_tickers:
                raise ValueError("No ETF tickers found on the page.")

            all_tickers.update(etf_tickers)
            print(f"✅ Fetched {len(etf_tickers)} tickers from Popular ETFs list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Popular ETFs list ({e}).")

    if MARKET_SELECTION.get("CRYPTO"):
        try:
            import requests
            url_crypto = "https://en.wikipedia.org/wiki/List_of_cryptocurrencies"
            response_crypto = requests.get(url_crypto, headers=headers)
            response_crypto.raise_for_status()
            tables_crypto = pd.read_html(StringIO(response_crypto.text))
            # The first table on the page is the one with active cryptocurrencies
            if tables_crypto:
                df_crypto = tables_crypto[0]
                # The 'Symbol' column contains the ticker
                if 'Symbol' in df_crypto.columns:
                    # Extract the primary ticker symbol and append '-USD'
                    crypto_tickers = set()
                    for s in df_crypto['Symbol'].tolist():
                        if isinstance(s, str):
                            # Use regex to find the first ticker-like symbol (e.g., BTC)
                            match = re.match(r'([A-Z]+)', s)
                            if match:
                                crypto_tickers.add(f"{match.group(1)}-USD")
                    all_tickers.update(crypto_tickers)
                    print(f"✅ Fetched {len(crypto_tickers)} tickers from Cryptocurrency list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Cryptocurrency list ({e}).")

    # --- German Tickers ---
    if MARKET_SELECTION.get("DAX"):
        try:
            import requests
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
            import requests
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

    # --- Swiss Tickers ---
    if MARKET_SELECTION.get("SMI"):
        try:
            import requests
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

    # --- Italian Tickers ---
    if MARKET_SELECTION.get("FTSE_MIB"):
        try:
            import requests
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
            mib_tickers = [s if '.' in s else f"{s}.MI" for s in table_mib['Ticker'].tolist()]
            all_tickers.update(mib_tickers)
            print(f"✅ Fetched {len(mib_tickers)} tickers from FTSE MIB.")
        except Exception as e:
            print(f"⚠️ Could not fetch FTSE MIB list ({e}).")

    if not all_tickers:
        print("⚠️ No tickers fetched. Returning empty list.")
        return []

    string_tickers = {str(s) for s in all_tickers if pd.notna(s)}
    
    final_tickers = set()
    for ticker in string_tickers:
        s_ticker = ticker.strip()
        if '$' in s_ticker:
            continue
        
        # European tickers already have the correct suffix for Yahoo Finance
        if s_ticker.endswith(('.DE', '.MI', '.SW', '.PA', '.AS', '.HE', '.LS', '.BR', '.MC')):
            final_tickers.add(s_ticker)
        else:
            # Normalize US tickers (e.g., BRK.B -> BRK-B)
            final_tickers.add(s_ticker.replace('.', '-'))

    print(f"Total unique tickers to analyze: {len(final_tickers)}")
    final_list = sorted(list(final_tickers))
    print(f"  Tickers: {', '.join(final_list)}")
    return final_list

# ============================
# Feature prep & model
# ============================

def fetch_training_data(ticker: str, start: Optional[datetime] = None, end: Optional[datetime] = None, target_percentage: float = 0.05) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch prices and compute ML features. Default window is TRAIN_LOOKBACK_DAYS up to 'end' (now if None)."""
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=TRAIN_LOOKBACK_DAYS)

    df = load_prices(ticker, start, end)
    if df.empty or len(df) < FEAT_SMA_LONG + 10:
        print(f"⚠️ Insufficient data for {ticker} from {start.date()} to {end.date()}. Returning empty DataFrame.")
        return pd.DataFrame(), []

    df = df.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    # Fix for missing OHLC columns (Yahoo fallback)
    if "High" not in df.columns and "Close" in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns and "Close" in df.columns:
        df["Low"] = df["Close"]
    if "Open" not in df.columns and "Close" in df.columns:
        df["Open"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    # Ensure 'Close' is numeric and drop rows with NaN in 'Close'
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Fill missing values in other columns
    df = df.ffill().bfill()

    df["Returns"]    = df["Close"].pct_change(fill_method=None)
    df["SMA_F_S"]    = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    df["SMA_F_L"]    = df["Close"].rolling(FEAT_SMA_LONG).mean()
    df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()

    # --- Additional Features ---
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    rs = gain / loss
    df['RSI_feat'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_mid'] = df["Close"].rolling(window=20).mean()
    df['BB_std'] = df["Close"].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)
    
    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]
    
    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Target"]     = df["Close"].shift(-1)

    # Classification label for BUY model: 5-day forward > +target_percentage
    fwd = df["Close"].shift(-CLASS_HORIZON)
    df["TargetClassBuy"] = ((fwd / df["Close"] - 1.0) > target_percentage).astype(float)

    # Classification label for SELL model: 5-day forward < -target_percentage
    df["TargetClassSell"] = ((fwd / df["Close"] - 1.0) < -target_percentage).astype(float)

    # Progress info
    req_cols = ["Close","Returns","SMA_F_S","SMA_F_L","Volatility", "RSI_feat", "MACD", "BB_upper"] + financial_features + ["Target", "TargetClassBuy", "TargetClassSell"]
    
    # Filter req_cols to only include those present in df.columns
    available_req_cols = [col for col in req_cols if col in df.columns]
    
    ready = df[available_req_cols].dropna()
    
    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell"]]

    print(f"   ↳ rows after features available: {len(ready)}")
    return ready, final_training_features

def train_and_evaluate_models(df: pd.DataFrame, target_col: str = "TargetClassBuy", feature_set: Optional[List[str]] = None):
    """Train and compare multiple classifiers for a given target, returning the best one."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")
        LGBMClassifier = None

    df = df.copy()
    # Feature generation (same as before)
    if "Returns" not in df.columns and "Close" in df.columns:
        df["Returns"] = df["Close"].pct_change()
    if "SMA_F_S" not in df.columns and "Close" in df.columns:
        df["SMA_F_S"] = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    if "SMA_F_L" not in df.columns and "Close" in df.columns:
        df["SMA_F_L"] = df["Close"].rolling(FEAT_SMA_LONG).mean()
    if "Volatility" not in df.columns and "Returns" in df.columns:
        df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()
    
    # --- Ensure additional features are present ---
    if 'RSI_feat' not in df.columns:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs = gain / loss
        df['RSI_feat'] = 100 - (100 / (1 + rs))
    if 'MACD' not in df.columns:
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
    if 'BB_upper' not in df.columns:
        df['BB_mid'] = df["Close"].rolling(window=20).mean()
        df['BB_std'] = df["Close"].rolling(window=20).std()
        df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)

    # --- Ensure additional financial features are present and numeric ---
    financial_features_present = [col for col in df.columns if col.startswith('Fin_')]
    for col in financial_features_present:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if target_col not in df.columns:
        print(f"⚠️ Target column '{target_col}' not found in DataFrame. Skipping model training.")
        return None

    req = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper"] + financial_features_present + [target_col]
    if any(c not in df.columns for c in req):
        print(f"⚠️ Missing columns for model comparison (target: {target_col}). Skipping.")
        return None

    d = df[req].dropna()
    if len(d) < 50:  # Increased requirement for cross-validation
        print("⚠️ Not enough rows after feature prep to compare models (need ≥ 50). Skipping.")
        return None

    # --- Check for class balance ---
    if d[target_col].nunique() < 2:
        print(f"⚠️ Not enough class diversity for training on '{target_col}'. Skipping model.")
        return None

    # Use the provided feature_set directly, as it's already filtered and ready
    if feature_set is None:
        # Fallback if feature_set is unexpectedly None, should not happen with new train_worker
        final_feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper"]
    else:
        final_feature_names = feature_set
    
    X_df = d[final_feature_names]
    y = d[target_col].values

    # Scale features for models that are sensitive to scale (like Logistic Regression and SVC)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_df.index)
    
    # Store feature names for consistent use during prediction
    scaler.feature_names_in_ = list(final_feature_names) 

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced"),
        "SVM": SVC(probability=True, random_state=SEED, class_weight="balanced")
    }

    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation):")
    for name, model in models.items():
        try:
            # Set n_jobs=1 to avoid multiprocessing issues on Windows
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
            results[name] = np.mean(scores)
            print(f"    - {name}: {results[name]:.4f} (std: {np.std(scores):.4f})")
        except Exception as e:
            print(f"    - {name}: Failed evaluation. Error: {e}")
            results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best model: {best_model_name} with AUC = {results[best_model_name]:.4f}")

    # Train the best model on all available data and return it
    best_model_instance = models[best_model_name]
    best_model_instance.fit(X, y)
    
    # We need to return the scaler as well to process live data
    return best_model_instance, scaler

# ============================
# Rule-based backtester (ATR & ML gate)
# ============================

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Optional ML gate to allow buys."""
    def __init__(self, df: pd.DataFrame, initial_balance: float, transaction_cost: float,
                 model_buy=None, model_sell=None, scaler=None, min_proba_buy: float = MIN_PROBA_BUY, min_proba_sell: float = MIN_PROBA_SELL, use_gate: bool = USE_MODEL_GATE,
                 market_data: Optional[pd.DataFrame] = None, use_market_filter: bool = USE_MARKET_FILTER, feature_set: Optional[List[str]] = None,
                 per_ticker_min_proba_buy: Optional[float] = None, per_ticker_min_proba_sell: Optional[float] = None):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model_buy = model_buy
        self.model_sell = model_sell
        self.scaler = scaler
        # Use per-ticker thresholds if provided, otherwise fallback to global/default
        self.min_proba_buy = float(per_ticker_min_proba_buy if per_ticker_min_proba_buy is not None else min_proba_buy)
        self.min_proba_sell = float(per_ticker_min_proba_sell if per_ticker_min_proba_sell is not None else min_proba_sell)
        self.use_gate = bool(use_gate) and (scaler is not None)
        self.market_data = market_data
        self.use_market_filter = use_market_filter and market_data is not None
        # Dynamically determine the full feature set including financial features
        # This will be passed from the training worker
        self.feature_set = feature_set if feature_set is not None else ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper", 
                                                                        'Fin_Revenue', 'Fin_NetIncome', 'Fin_TotalAssets', 'Fin_TotalLiabilities', 'Fin_FreeCashFlow', 'Fin_EBITDA']

        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.entry_price: Optional[float] = None
        self.highest_since_entry: Optional[float] = None
        self.entry_atr: Optional[float] = None
        self.holding_bars = 0
        self.portfolio_history: List[float] = [self.initial_balance]
        self.trade_log: List[Tuple] = []
        self.ticker = self.df.iloc[0]['ticker'] if 'ticker' in self.df.columns else "UNKNOWN" # Store ticker for logging
        self.last_ai_action: str = "HOLD" # New: Track last AI action

        close = self.df["Close"]
        
        # --- Strategy Indicators ---
        # 1. Trend Filter: 200-day SMA
        self.df['SMA_200'] = close.rolling(window=200).mean()

        # 2. Crossover SMAs
        self.df['SMA_S'] = close.rolling(window=STRAT_SMA_SHORT).mean()
        self.df['SMA_L'] = close.rolling(window=STRAT_SMA_LONG).mean()

        # --- Other Indicators (for reference or potential future use) ---
        # Momentum: 14-day RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # 3. Volume: On-Balance Volume (OBV) and its SMAs
        self.df['OBV'] = (np.sign(close.diff()) * self.df['Volume']).fillna(0).cumsum()
        self.df['OBV_SMA_S'] = self.df['OBV'].rolling(window=10).mean()
        self.df['OBV_SMA_L'] = self.df['OBV'].rolling(window=30).mean()
        # ------------------------------------

        # ATR for risk management (unchanged)
        high = self.df["High"] if "High" in self.df.columns else None
        low  = self.df["Low"]  if "Low" in self.df.columns else None
        prev_close = close.shift(1)
        if high is not None and low is not None:
            hl = (high - low).abs()
            h_pc = (high - prev_close).abs()
            l_pc = (low  - prev_close).abs()
            tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
            self.df["ATR"] = tr.rolling(ATR_PERIOD).mean()
        else:
            ret = close.pct_change(fill_method=None)
            self.df["ATR"] = (ret.rolling(ATR_PERIOD).std() * close).rolling(2).mean()

        # Low-volatility filter reference: rolling median ATR
        self.df['ATR_MED'] = self.df['ATR'].rolling(50).median()

        # --- Features for ML Gate ---
        self.df["Returns"]    = close.pct_change(fill_method=None)
        self.df["SMA_F_S"]    = close.rolling(FEAT_SMA_SHORT).mean()
        self.df["SMA_F_L"]    = close.rolling(FEAT_SMA_LONG).mean()
        self.df["Volatility"] = self.df["Returns"].rolling(FEAT_VOL_WINDOW).std()
        
        # RSI for features
        delta_feat = close.diff()
        gain_feat = (delta_feat.where(delta_feat > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss_feat = (-delta_feat.where(delta_feat < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs_feat = gain_feat / loss_feat
        self.df['RSI_feat'] = 100 - (100 / (1 + rs_feat))

        # MACD for features
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_12 - ema_26
        
        # Bollinger Bands for features
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.df['BB_upper'] = bb_mid + (bb_std * 2)

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

    def _get_model_prediction(self, i: int, model) -> float:
        """Helper to get a single model's prediction probability."""
        if not self.use_gate or model is None:
            return 0.0
        row = self.df.loc[i]
        
        # Use the feature names that the scaler was fitted with
        model_feature_names = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else self.feature_set
        
        # Create a dictionary for the current row's feature values
        # Fill missing features with 0 or a suitable default
        feature_values = {f: row.get(f, 0.0) for f in model_feature_names}
        
        # Create DataFrame with all expected features
        X_df = pd.DataFrame([feature_values], columns=model_feature_names)
        
        # Ensure all values are numeric, fill any remaining NaNs (e.g., from get(f, 0.0) if f was not in row)
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0) # Coerce to numeric and fill any NaNs

        # Check if any feature column is entirely NaN after processing (should not happen with fillna(0.0))
        if X_df.isnull().all().any():
            # This indicates a serious issue where a feature column is all NaN even after filling
            print(f"  [{self.ticker}] Critical: Feature column is all NaN after fillna at step {i}. Skipping prediction.")
            return 0.0

        try:
            X_scaled_np = self.scaler.transform(X_df)
            X = pd.DataFrame(X_scaled_np, columns=model_feature_names) # Use model_feature_names for columns
            
            return float(model.predict_proba(X)[0][1])
        except Exception as e:
            print(f"  [{self.ticker}] Error in model prediction at step {i}: {e}")
            return 0.0

    def _allow_buy_by_model(self, i: int) -> bool: # Removed feature_set from signature
        return self._get_model_prediction(i, self.model_buy) >= self.min_proba_buy

    def _allow_sell_by_model(self, i: int) -> bool: # Removed feature_set from signature
        return self._get_model_prediction(i, self.model_sell) >= self.min_proba_sell

    def _position_size_from_atr(self, price: float, atr: float) -> int:
        if atr is None or np.isnan(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollars = self.initial_balance * RISK_PER_TRADE
        per_share_risk = ATR_MULT_TRAIL * atr
        qty = int(risk_dollars / per_share_risk)
        return max(qty, 0)

    def _buy(self, price: float, atr: Optional[float], date: str):
        if self.cash <= 0:
            return
        qty = self._position_size_from_atr(price, atr if atr is not None else np.nan)
        if qty <= 0:
            return
        fee = price * qty * self.transaction_cost
        cost = price * qty + fee
        if cost > self.cash:
            qty = int(self.cash / (price * (1 + self.transaction_cost)))
            if qty <= 0:
                return
            fee = price * qty * self.transaction_cost
            cost = price * qty + fee
            if cost > self.cash:
                return

        self.cash -= cost
        self.shares += qty
        self.entry_price = price
        self.entry_atr = atr if atr is not None and not np.isnan(atr) else None
        self.highest_since_entry = price
        self.holding_bars = 0
        self.trade_log.append((date, "BUY", price, qty, self.ticker, {"fee": fee}, fee))
        self.last_ai_action = "BUY" # Update last AI action
        # print(f"  [{self.ticker}] BUY: {date}, Price: {price:.2f}, Qty: {qty}, Cash: {self.cash:,.2f}, Shares: {self.shares:.2f}")

    def _sell(self, price: float, date: str):
        if self.shares <= 0:
            return
        qty = int(self.shares)
        proceeds = price * qty
        fee = proceeds * self.transaction_cost
        self.cash += proceeds - fee
        self.shares -= qty
        self.entry_price = None
        self.entry_atr = None
        self.highest_since_entry = None
        self.holding_bars = 0
        self.trade_log.append((date, "SELL", price, qty, self.ticker, {"fee": fee}, fee))
        self.last_ai_action = "SELL" # Update last AI action
        # print(f"  [{self.ticker}] SELL: {price:.2f}, Qty: {qty}, Cash: {self.cash:,.2f}, Shares: {self.shares:.2f}")

    def step(self):
        if self.current_step < 1: # Need previous row for signal
            self.current_step += 1
            self.portfolio_history.append(self.initial_balance)
            return False

        if self.current_step >= len(self.df):
            return True

        # Current and previous data rows
        row = self.df.iloc[self.current_step]
        # prev_row = self.df.iloc[self.current_step - 1] # Not used, can remove if not needed elsewhere
        
        price = float(row["Close"])
        date = self._date_at(self.current_step)
        atr = float(row.get("ATR", np.nan)) if pd.notna(row.get("ATR", np.nan)) else None

        # --- Market Filter ---
        if self.use_market_filter:
            current_date = row['Date'].normalize()
            # Use asof to find the latest market data point on or before the current date
            market_slice = self.market_data.loc[:current_date]
            if not market_slice.empty:
                latest_market_data = market_slice.iloc[-1]
                market_close = latest_market_data['Close']
                market_sma = latest_market_data['SMA_L_MKT']
                if pd.notna(market_close) and pd.notna(market_sma) and market_close < market_sma:
                    # Market is in a downtrend. Sell any open position and do not open new ones.
                    if self.shares > 0:
                        self._sell(price, date)
                    
                    port_val = self.cash + self.shares * price
                    self.portfolio_history.append(port_val)
                    self.current_step += 1
                    return self.current_step >= len(self.df)

        # --- Entry Signal ---
        # Condition: AI model is now the primary buy signal generator.
        if self.shares == 0 and self._allow_buy_by_model(self.current_step):
            self._buy(price, atr, date)
        
        # --- AI-driven Exit Signal ---
        elif self.shares > 0 and self._allow_sell_by_model(self.current_step): # Changed to elif to prioritize buy
            self._sell(price, date)
        else:
            self.last_ai_action = "HOLD" # No action taken by AI

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple], str]: # Added str for last_ai_action
        done = False
        while not done:
            done = self.step()
        if self.shares > 0:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log, self.last_ai_action # Return last_ai_action

# ============================
# Analytics
# ============================

def analyze_performance(
    trade_log: List[tuple],
    strategy_history: List[float],
    buy_hold_history: List[float],
    ticker: str
) -> Dict[str, float]:
    """Analyzes trades and calculates key performance metrics."""
    # --- Trade Analysis ---
    buys = [t for t in trade_log if t[1] == "BUY"]
    sells = [t for t in trade_log if t[1] == "SELL"]
    profits = []
    n = min(len(buys), len(sells))
    for i in range(n):
        pb, sb = float(buys[i][2]), float(sells[i][2])
        qb, qs = float(buys[i][3]), float(sells[i][3])
        qty = min(qb, qs)
        fee_b = float(buys[i][6]) if len(buys[i]) > 6 else 0.0
        fee_s = float(sells[i][6]) if len(sells[i]) > 6 else 0.0
        profits.append((sb - pb) * qty - (fee_b + fee_s))

    total_pnl = float(sum(profits))
    win_rate = (sum(1 for p in profits if p > 0) / len(profits)) if profits else 0.0
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  - Trades: {n}, Win Rate: {win_rate:.2%}")
    print(f"  - Total PnL: ${total_pnl:,.2f}")

    # --- Performance Metrics ---
    strat_returns = pd.Series(strategy_history).pct_change().dropna()
    bh_returns = pd.Series(buy_hold_history).pct_change().dropna()

    # Sharpe Ratio (annualized, assuming 252 trading days)
    sharpe_strat = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() > 0 else 0
    sharpe_bh = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if bh_returns.std() > 0 else 0

    # Max Drawdown
    strat_series = pd.Series(strategy_history)
    strat_cummax = strat_series.cummax()
    strat_drawdown = ((strat_series - strat_cummax) / strat_cummax).min()

    bh_series = pd.Series(buy_hold_history)
    bh_cummax = bh_series.cummax()
    bh_drawdown = ((bh_series - bh_cummax) / bh_cummax).min()

    print(f"\n📈 {ticker} Performance Metrics:")
    print(f"  | Metric         | Strategy      | Buy & Hold    |")
    print(f"  |----------------|---------------|---------------|")
    print(f"  | Sharpe Ratio   | {sharpe_strat:13.2f} | {sharpe_bh:13.2f} |")
    print(f"  | Max Drawdown   | {strat_drawdown:12.2%} | {bh_drawdown:12.2%} |")

    return {
        "trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe_ratio": sharpe_strat, "max_drawdown": strat_drawdown
    }

# ============================
# Top Performer Analysis
# ============================

def find_top_performers(return_tickers: bool = False, n_top: int = N_TOP_TICKERS, fcf_min_threshold: float = 0.0, ebitda_min_threshold: float = 0.0):
    """
    Fetches S&P 500 & NASDAQ tickers, screens for the top N performers,
    and returns a list of (ticker, performance) tuples.
    """
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        print("❌ No tickers to process. Exiting.")
        return []

    # Apply N_TOP_TICKERS limit here
    if n_top > 0 and len(all_available_tickers) > n_top:
        import random
        random.seed(SEED)
        tickers = random.sample(all_available_tickers, n_top)
        print(f"✅ Selected top {n_top} tickers for analysis: {', '.join(tickers)}")
    else:
        tickers = all_available_tickers
        print(f"✅ Analyzing all {len(tickers)} available tickers.")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)
    
    if USE_PERFORMANCE_BENCHMARK:
        # --- Step 1: Calculate Benchmark Performances ---
        print("- Calculating 1-Year Performance Benchmarks...")
        benchmark_perfs = {}
        for bench_ticker in ['QQQ', 'SPY']:
            try:
                df = load_prices_robust(bench_ticker, start_date, end_date)
                if df is not None and not df.empty:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    if start_price > 0:
                        perf = ((end_price - start_price) / start_price) * 100
                        benchmark_perfs[bench_ticker] = perf
                        print(f"  ✅ {bench_ticker} 1-Year Performance: {perf:.2f}%")
            except Exception as e:
                print(f"⚠️ Could not calculate {bench_ticker} performance: {e}.")
            time.sleep(2) # Add a pause to avoid rate limiting
        
        if not benchmark_perfs:
            print("❌ Could not calculate any benchmark performance. Cannot proceed.")
            return []
            
        # Determine the higher of the two benchmarks
        market_benchmark_perf = max(benchmark_perfs.values()) if benchmark_perfs else 0.0
        final_benchmark_perf = market_benchmark_perf
        print(f"  📈 Using final 1-Year performance benchmark of {final_benchmark_perf:.2f}%")

        # --- Step 2: Calculate YTD Benchmarks ---
        print("- Calculating YTD Performance Benchmarks...")
        ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)
        ytd_benchmark_perfs = {}
        for bench_ticker in ['QQQ', 'SPY']:
            try:
                df = load_prices_robust(bench_ticker, ytd_start_date, end_date)
                if df is not None and not df.empty:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    if start_price > 0:
                        perf = ((end_price - start_price) / start_price) * 100
                        ytd_benchmark_perfs[bench_ticker] = perf
                        print(f"  ✅ {bench_ticker} YTD Performance: {perf:.2f}%")
            except Exception as e:
                print(f"⚠️ Could not calculate {bench_ticker} YTD performance: {e}.")
            time.sleep(2) # Add a pause to avoid rate limiting
        
        if not ytd_benchmark_perfs:
            print("❌ Could not calculate any YTD benchmark performance. Cannot proceed.")
            return []
        ytd_benchmark_perf = max(ytd_benchmark_perfs.values())
        print(f"  📈 Using YTD performance benchmark of {ytd_benchmark_perf:.2f}%")
    else:
        print("ℹ️ Performance benchmark is disabled. All tickers will be considered.")
        final_benchmark_perf = -np.inf
        ytd_benchmark_perf = -np.inf
        ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    # --- Step 3: Find stocks that beat both benchmarks ---
    performance_data = {}
    num_batches = (len(tickers) + BATCH_DOWNLOAD_SIZE - 1) // BATCH_DOWNLOAD_SIZE
    
    for i in range(num_batches):
        start_idx = i * BATCH_DOWNLOAD_SIZE
        end_idx = start_idx + BATCH_DOWNLOAD_SIZE
        batch_tickers = tickers[start_idx:end_idx]
        
        if not batch_tickers:
            continue

        print(f"\n--- Processing Batch {i+1}/{num_batches} ---")
        
        # Batch download for 1-Year Performance
        data_1y = _download_batch_robust(batch_tickers, start=start_date, end=end_date)

        # Batch download for YTD Performance
        data_ytd = _download_batch_robust(batch_tickers, start=ytd_start_date, end=end_date)

        for ticker in tqdm(batch_tickers, desc=f"Analyzing Batch {i+1}/{num_batches}"):
            try:
                # 1-Year Performance
                perf_1y = -np.inf
                if not data_1y.empty:
                    close_series = None
                    if len(batch_tickers) > 1:
                        # Multi-ticker download, columns are MultiIndex
                        if 'Close' in data_1y.columns and ticker in data_1y['Close'].columns:
                            close_series = data_1y['Close'][ticker]
                    else:
                        # Single-ticker download, columns are flat
                        if 'Close' in data_1y.columns:
                            close_series = data_1y['Close']
                    
                    if close_series is not None:
                        df_1y_close = close_series.dropna()
                        if not df_1y_close.empty and len(df_1y_close) > 200:
                            start_price = df_1y_close.iloc[0]
                            end_price = df_1y_close.iloc[-1]
                            if start_price > 0:
                                perf_1y = ((end_price - start_price) / start_price) * 100
                
                # YTD Performance
                perf_ytd = -np.inf
                if not data_ytd.empty:
                    close_series = None
                    if len(batch_tickers) > 1:
                        if 'Close' in data_ytd.columns and ticker in data_ytd['Close'].columns:
                            close_series = data_ytd['Close'][ticker]
                    else:
                        if 'Close' in data_ytd.columns:
                            close_series = data_ytd['Close']

                    if close_series is not None:
                        df_ytd_close = close_series.dropna()
                        if not df_ytd_close.empty:
                            start_price = df_ytd_close.iloc[0]
                            end_price = df_ytd_close.iloc[-1]
                            if start_price > 0:
                                perf_ytd = ((end_price - start_price) / start_price) * 100

                # Only add to performance_data if performance is valid
                if perf_1y != -np.inf and perf_ytd != -np.inf:
                    if USE_PERFORMANCE_BENCHMARK:
                        if perf_1y > final_benchmark_perf and perf_ytd > ytd_benchmark_perf:
                            performance_data[ticker] = perf_1y
                    else:
                        performance_data[ticker] = perf_1y # Add all valid performers when benchmark is disabled
            except Exception:
                pass
        
        if i < num_batches - 1:
            print(f"--- Pausing for {PAUSE_BETWEEN_BATCHES} seconds before next batch ---")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    if USE_PERFORMANCE_BENCHMARK:
        print(f"\n✅ Found {len(performance_data)} stocks that passed the performance benchmarks.")
    else:
        print(f"\n✅ Found {len(performance_data)} stocks for analysis (performance benchmark disabled).")
        
    if not performance_data:
        return []

    # Filter for stocks that beat the high benchmark
    strong_performers = performance_data
    
    sorted_strong_performers = sorted(strong_performers.items(), key=lambda item: item[1], reverse=True)
    
    # The n_top limit is already applied to the 'tickers' list, so this check is redundant here
    # if n_top > 0 and len(sorted_strong_performers) > n_top:
    #     print(f"  ✅ Found {len(sorted_strong_performers)} stocks outperforming the benchmark. Selecting top {n_top}.")
    #     sorted_strong_performers = sorted_strong_performers[:n_top]
    # else:
    #     print(f"  ✅ Found {len(sorted_strong_performers)} stocks outperforming the benchmark.")

    final_performers = sorted_strong_performers
    
    # Add YTD performance to the final list
    final_performers_with_ytd = []
    for ticker, perf_1y in final_performers:
        df_ytd = load_prices_robust(ticker, ytd_start_date, end_date)
        perf_ytd = -np.inf
        if df_ytd is not None and not df_ytd.empty:
            start_price = df_ytd['Close'].iloc[0]
            end_price = df_ytd['Close'].iloc[-1]
            if start_price > 0:
                perf_ytd = ((end_price - start_price) / start_price) * 100
        final_performers_with_ytd.append((ticker, perf_1y, perf_ytd))
    
    final_performers = final_performers_with_ytd

    # --- Step 3: Fundamental Screen (Optional Free Cash Flow & EBITDA for the last fiscal year) ---
    if fcf_min_threshold is not None or ebitda_min_threshold is not None:
        print(f"  🔍 Screening {len(final_performers)} strong performers for fundamental metrics...")
        screened_performers = []
        
        pbar = tqdm(final_performers, desc="Applying fundamental screens")
        
        for ticker, perf_1y, perf_ytd in pbar:
            try:
                yf_ticker = yf.Ticker(ticker)
                
                # FCF Check
                fcf_ok = True
                if fcf_min_threshold is not None:
                    cashflow = yf_ticker.cashflow
                    if not cashflow.empty:
                        latest_cashflow = cashflow.iloc[:, 0]
                        fcf_keys = ['Free Cash Flow', 'freeCashflow']
                        fcf = None
                        for key in fcf_keys:
                            if key in latest_cashflow.index:
                                fcf = latest_cashflow[key]
                                break
                        if fcf is not None and fcf <= fcf_min_threshold:
                            fcf_ok = False
                
                # EBITDA Check
                ebitda_ok = True
                if ebitda_min_threshold is not None:
                    financials = yf_ticker.financials
                    if not financials.empty:
                        latest_financials = financials.iloc[:, 0]
                        ebitda_keys = ['EBITDA', 'ebitda']
                        ebitda = None
                        for key in ebitda_keys:
                            if key in latest_financials.index:
                                ebitda = latest_financials[key]
                                break
                        if ebitda is not None and ebitda <= ebitda_min_threshold:
                            ebitda_ok = False

                if fcf_ok and ebitda_ok:
                    screened_performers.append((ticker, perf_1y, perf_ytd))

            except Exception:
                # If there's any error fetching financials, we let it pass
                pbar.set_description(f"Applying screens ({ticker}: fetch error)")
                screened_performers.append((ticker, perf_1y, perf_ytd))

        print(f"  ✅ Found {len(screened_performers)} stocks passing the fundamental screens.")
        final_performers = screened_performers

    if return_tickers:
        return final_performers
    
    # If not returning for backtest, just print the list
    print(f"\n\n🏆 Stocks Outperforming {final_benchmark_perf:.2f}%) 🏆") # Removed high_benchmark_ticker
    print("-" * 60)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Performance':>15}")
    print("-" * 60)
    
    for i, (ticker, perf, _) in enumerate(final_performers, 1): # Iterate over final_performers which now includes YTD
        print(f"{i:<5} | {ticker:<10} | {perf:14.2f}%")
    
    print("-" * 60)
    return list(final_tickers)


# ============================
# Main
# ============================

def train_worker(params: Tuple) -> Dict:
    """Worker function to train models for a single ticker."""
    ticker, train_start, train_end, target_percentage, feature_set = params # Removed models_buy, models_sell, scalers, market_data, capital_per_stock
    
    training_data_df, final_training_features = fetch_training_data(ticker, train_start, train_end, target_percentage=target_percentage)
    if training_data_df.empty or not final_training_features:
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}
    
    model_buy, model_sell, scaler = None, None, None

    model_buy_and_scaler = train_and_evaluate_models(training_data_df, target_col="TargetClassBuy", feature_set=final_training_features)
    if model_buy_and_scaler is not None:
        model_buy, scaler = model_buy_and_scaler

    model_sell_and_scaler = train_and_evaluate_models(training_data_df, target_col="TargetClassSell", feature_set=final_training_features)
    if model_sell_and_scaler is not None:
        model_sell = model_sell_and_scaler[0]
        if scaler is None:
            scaler = model_sell_and_scaler[1]
            
    return {'ticker': ticker, 'model_buy': model_buy, 'model_sell': model_sell, 'scaler': scaler}

def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function to run backtest for a single ticker."""
    ticker, bt_start, bt_end, capital_per_stock, model_buy, model_sell, scaler, market_data, feature_set, min_proba_buy, min_proba_sell = params
    
    warmup_days = max(STRAT_SMA_LONG, 200) + 50
    data_start = bt_start - timedelta(days=warmup_days)
    df = load_prices_robust(ticker, data_start, bt_end)
    
    if df.empty or len(df.loc[bt_start:]) < STRAT_SMA_SHORT + 5:
        return None
    if df.isna().all().all() or "Close" not in df.columns or df["Close"].isna().any():
        return None

    env = RuleTradingEnv(df, initial_balance=capital_per_stock, transaction_cost=TRANSACTION_COST,
                         model_buy=model_buy, model_sell=model_sell, scaler=scaler, 
                         min_proba_buy=min_proba_buy, min_proba_sell=min_proba_sell, 
                         use_gate=USE_MODEL_GATE,
                         market_data=market_data, use_market_filter=USE_MARKET_FILTER,
                         feature_set=feature_set)
    final_val, log, last_ai_action = env.run() # Capture last_ai_action
    
    df_backtest = df.loc[df.index >= bt_start]
    strategy_history = env.portfolio_history[-len(df_backtest):]
    start_price = float(df_backtest["Close"].iloc[0])
    shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
    cash_bh = capital_per_stock - shares_bh * start_price
    buy_hold_history = (cash_bh + shares_bh * df_backtest["Close"]).tolist()
    bh_val = buy_hold_history[-1]

    made_trades = any(t[1] in ["BUY", "SELL"] for t in log)
    strategy_history_for_analysis = strategy_history if made_trades else [capital_per_stock] * len(df_backtest)
    if not made_trades:
        final_val = capital_per_stock

    perf_data = analyze_performance(log, strategy_history_for_analysis, buy_hold_history, ticker)
    
    # Calculate individual buy and hold return for the backtest period
    individual_bh_return = ((bh_val - capital_per_stock) / capital_per_stock) * 100 if capital_per_stock > 0 else 0

    return {'ticker': ticker, 'final_val': final_val, 'bh_val': bh_val, 'perf_data': perf_data, 'individual_bh_return': individual_bh_return, 'last_ai_action': last_ai_action}

def optimize_thresholds_worker(params: Tuple) -> Dict:
    """Worker function to optimize thresholds for a single ticker."""
    ticker, train_start, train_end, target_percentage, feature_set, models_buy, models_sell, scalers, market_data, capital_per_stock = params

    best_sharpe = -np.inf
    best_min_proba_buy = MIN_PROBA_BUY
    best_min_proba_sell = MIN_PROBA_SELL

    # Load models and scaler for this ticker
    model_buy = models_buy.get(ticker)
    model_sell = models_sell.get(ticker)
    scaler = scalers.get(ticker)

    if model_buy is None or scaler is None:
        return {'ticker': ticker, 'min_proba_buy': MIN_PROBA_BUY, 'min_proba_sell': MIN_PROBA_SELL, 'sharpe': -np.inf}

    # Fetch data for optimization period (same as training data)
    df_opt = load_prices_robust(ticker, train_start, train_end)
    if df_opt.empty or len(df_opt) < STRAT_SMA_LONG + 5:
        return {'ticker': ticker, 'min_proba_buy': MIN_PROBA_BUY, 'min_proba_sell': MIN_PROBA_SELL, 'sharpe': -np.inf}

    # Define a range of thresholds to test
    buy_thresholds = np.arange(0.5, 0.9, 0.05)
    sell_thresholds = np.arange(0.5, 0.9, 0.05)

    for buy_t in buy_thresholds:
        for sell_t in sell_thresholds:
            env = RuleTradingEnv(df_opt.copy(), initial_balance=capital_per_stock, transaction_cost=TRANSACTION_COST,
                                 model_buy=model_buy, model_sell=model_sell, scaler=scaler,
                                 min_proba_buy=buy_t, min_proba_sell=sell_t,
                                 use_gate=USE_MODEL_GATE,
                                 market_data=market_data, use_market_filter=USE_MARKET_FILTER,
                                 feature_set=feature_set)
            final_val, log, _ = env.run() # Don't need last_ai_action for optimization

            if len(env.portfolio_history) > 1:
                strat_returns = pd.Series(env.portfolio_history).pct_change().dropna()
                if strat_returns.std() > 0:
                    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)
                else:
                    sharpe = 0.0 # No volatility, potentially good if return is positive
            else:
                sharpe = -np.inf # Not enough data for meaningful Sharpe

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_min_proba_buy = buy_t
                best_min_proba_sell = sell_t
    
    print(f"  ✅ Optimized {ticker}: Buy={best_min_proba_buy:.2f}, Sell={best_min_proba_sell:.2f}, Sharpe={best_sharpe:.2f}")
    return {'ticker': ticker, 'min_proba_buy': best_min_proba_buy, 'min_proba_sell': best_min_proba_sell, 'sharpe': best_sharpe}

def optimize_thresholds_for_portfolio(
    top_tickers: List[str],
    train_start: datetime,
    train_end: datetime,
    target_percentage: float,
    feature_set: Optional[List[str]],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    market_data: Optional[pd.DataFrame],
    capital_per_stock: float,
    run_parallel: bool
) -> Dict[str, Dict[str, float]]:
    """Orchestrates parallel optimization of thresholds for all tickers."""
    print("\n🔍 Step 2.5: Optimizing ML thresholds for each ticker...")
    num_processes = max(1, cpu_count() - 2)

    optimization_params = []
    for ticker in top_tickers:
        optimization_params.append((ticker, train_start, train_end, target_percentage, feature_set, models_buy, models_sell, scalers, market_data, capital_per_stock))

    optimized_results = {}
    if run_parallel:
        print(f"⚙️ Optimizing thresholds in parallel for {len(top_tickers)} tickers using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(optimize_thresholds_worker, optimization_params), total=len(optimization_params), desc="Optimizing Thresholds"))
    else:
        print(f"⚙️ Optimizing thresholds sequentially for {len(top_tickers)} tickers...")
        results = [optimize_thresholds_worker(p) for p in tqdm(optimization_params, desc="Optimizing Thresholds")]

    for res in results:
        if res and res['sharpe'] != -np.inf: # Only store if optimization was successful
            optimized_results[res['ticker']] = {
                'min_proba_buy': res['min_proba_buy'],
                'min_proba_sell': res['min_proba_sell']
            }
    
    print(f"✅ Optimization complete. Found optimized thresholds for {len(optimized_results)} tickers.")
    return optimized_results

def main(
    fcf_threshold: float = 0.0,
    ebitda_threshold: float = 0.0,
    min_proba_buy: float = MIN_PROBA_BUY,
    min_proba_sell: float = MIN_PROBA_SELL,
    target_percentage: float = TARGET_PERCENTAGE,
    top_performers_data=None,
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True,
    single_ticker: Optional[str] = None,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float]]:
    
    end_date = datetime.now(timezone.utc)
    bt_end = end_date
    
    # --- Handle single ticker case for initial performance calculation ---
    if single_ticker:
        print(f"🔍 Running analysis for single ticker: {single_ticker}")
        start_date_1y = end_date - timedelta(days=365)
        ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)
        
        df_1y = load_prices_robust(single_ticker, start_date_1y, end_date)
        perf_1y = -np.inf
        if df_1y is not None and not df_1y.empty:
            start_price = df_1y['Close'].iloc[0]
            end_price = df_1y['Close'].iloc[-1]
            if start_price > 0:
                perf_1y = ((end_price - start_price) / start_price) * 100

        df_ytd = load_prices_robust(single_ticker, ytd_start_date, end_date)
        perf_ytd = -np.inf
        if df_ytd is not None and not df_ytd.empty:
            start_price = df_ytd['Close'].iloc[0]
            end_price = df_ytd['Close'].iloc[-1]
            if start_price > 0:
                perf_ytd = ((end_price - start_price) / start_price) * 100
        
        top_performers_data = [(single_ticker, perf_1y, perf_ytd)]
    
    # --- Identify top performers if not provided ---
    if top_performers_data is None:
        if pdr is None and DATA_PROVIDER.lower() == 'stooq':
            print("⚠️ pandas-datareader not installed; run: pip install pandas-datareader")
        
        title = "🚀 AI-Powered Momentum & Trend Strategy"
        filters = []
        if fcf_threshold is not None:
            filters.append(f"FCF > ${fcf_threshold:,.0f}")
        if ebitda_threshold is not None:
            filters.append(f"EBITDA > ${ebitda_threshold:,.0f}")
        if filters:
            title += f" ({', '.join(filters)})"
        print(title + "\n" + "="*50 + "\n")

        print("🔍 Step 1: Identifying stocks outperforming market benchmarks...")
        top_performers_data = find_top_performers(return_tickers=True, fcf_min_threshold=fcf_threshold, ebitda_min_threshold=ebitda_threshold)
    
    if not top_performers_data:
        print("❌ Could not identify top tickers. Aborting backtest.")
        return None, None, None, None, None, None, None, None, None, None, None # Return 11 Nones
    
    top_tickers = [ticker for ticker, _, _ in top_performers_data]
    print(f"\n✅ Identified {len(top_tickers)} stocks for backtesting.\n")

    # --- Training Models (for 1-Year Backtest) ---
    print("🔍 Step 2: Training AI models for 1-Year backtest...")
    bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)
    train_end_1y = bt_start_1y - timedelta(days=1)
    train_start_1y = train_end_1y - timedelta(days=TRAIN_LOOKBACK_DAYS)
    num_processes = max(1, cpu_count() - 2)

    training_params_1y = [(ticker, train_start_1y, train_end_1y, target_percentage, feature_set) for ticker in top_tickers]
    models_buy, models_sell, scalers = {}, {}, {}

    if run_parallel:
        print(f"🤖 Training 1-Year models in parallel for {len(top_tickers)} tickers using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            training_results_1y = list(tqdm(pool.imap(train_worker, training_params_1y), total=len(training_params_1y), desc="Training 1-Year Models"))
    else:
        print(f"🤖 Training 1-Year models sequentially for {len(top_tickers)} tickers...")
        training_results_1y = [train_worker(p) for p in tqdm(training_params_1y, desc="Training 1-Year Models")]

    for res in training_results_1y:
        if res and res.get('model_buy'):
            models_buy[res['ticker']] = res['model_buy']
            models_sell[res['ticker']] = res['model_sell']
            scalers[res['ticker']] = res['scaler']

    if not models_buy and USE_MODEL_GATE:
        print("⚠️ No models were trained for 1-Year backtest. Model-gating will be disabled for this run.\n")

    # --- Fetch Market Data (if enabled) ---
    market_data = None
    if USE_MARKET_FILTER:
        print(f"🔄 Fetching market data for filter ({MARKET_FILTER_TICKER})...")
        market_start = train_start_1y - timedelta(days=MARKET_FILTER_SMA)
        market_data = load_prices_robust(MARKET_FILTER_TICKER, market_start, bt_end)
        if not market_data.empty:
            market_data['SMA_L_MKT'] = market_data['Close'].rolling(MARKET_FILTER_SMA).mean()
            print("✅ Market data prepared.\n")
        else:
            print(f"⚠️ Could not load market data for {MARKET_FILTER_TICKER}. Filter will be disabled.\n")

    capital_per_stock = INITIAL_BALANCE / max(len(top_tickers), 1)

    # --- OPTIMIZE THRESHOLDS ---
    optimized_params_per_ticker = optimize_thresholds_for_portfolio(
        top_tickers=top_tickers,
        train_start=train_start_1y, # Use training data for optimization
        train_end=train_end_1y,
        target_percentage=target_percentage,
        feature_set=feature_set,
        models_buy=models_buy,
        models_sell=models_sell,
        scalers=scalers,
        market_data=market_data,
        capital_per_stock=capital_per_stock,
        run_parallel=run_parallel
    )

    # --- Run 1-Year Backtest ---
    final_strategy_value_1y, strategy_results_1y, processed_tickers_1y, performance_metrics_1y = _run_portfolio_backtest(
        start_date=bt_start_1y,
        end_date=bt_end,
        top_tickers=top_tickers,
        models_buy=models_buy,
        models_sell=models_sell,
        scalers=scalers,
        market_data=market_data,
        optimized_params_per_ticker=optimized_params_per_ticker,
        capital_per_stock=capital_per_stock,
        target_percentage=target_percentage,
        run_parallel=run_parallel,
        period_name="1-Year"
    )
    ai_1y_return = ((final_strategy_value_1y - INITIAL_BALANCE) / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0

    # --- Calculate Buy & Hold for 1-Year ---
    buy_hold_results_1y = []
    for ticker in processed_tickers_1y:
        df_bh = load_prices_robust(ticker, bt_start_1y, bt_end)
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            buy_hold_results_1y.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
        else:
            buy_hold_results_1y.append(capital_per_stock) # If no data, assume initial capital
    final_buy_hold_value_1y = sum(buy_hold_results_1y) + (len(top_tickers) - len(processed_tickers_1y)) * capital_per_stock


    # --- Training Models (for YTD Backtest) ---
    print("\n🔍 Step 3: Training AI models for YTD backtest...")
    ytd_start_date = datetime(bt_end.year, 1, 1, tzinfo=timezone.utc)
    train_end_ytd = ytd_start_date - timedelta(days=1)
    train_start_ytd = train_end_ytd - timedelta(days=TRAIN_LOOKBACK_DAYS)
    
    training_params_ytd = [(ticker, train_start_ytd, train_end_ytd, target_percentage, feature_set) for ticker in top_tickers]
    models_buy_ytd, models_sell_ytd, scalers_ytd = {}, {}, {}

    if run_parallel:
        print(f"🤖 Training YTD models in parallel for {len(top_tickers)} tickers using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            training_results_ytd = list(tqdm(pool.imap(train_worker, training_params_ytd), total=len(training_params_ytd), desc="Training YTD Models"))
    else:
        print(f"🤖 Training YTD models sequentially for {len(top_tickers)} tickers...")
        training_results_ytd = [train_worker(p) for p in tqdm(training_params_ytd, desc="Training YTD Models")]

    for res in training_results_ytd:
        if res and res.get('model_buy'):
            models_buy_ytd[res['ticker']] = res['model_buy']
            models_sell_ytd[res['ticker']] = res['model_sell']
            scalers_ytd[res['ticker']] = res['scaler']

    if not models_buy_ytd and USE_MODEL_GATE:
        print("⚠️ No models were trained for YTD backtest. Model-gating will be disabled for this run.\n")

    # --- Run YTD Backtest ---
    final_strategy_value_ytd, strategy_results_ytd, processed_tickers_ytd_local, performance_metrics_ytd = _run_portfolio_backtest(
        start_date=ytd_start_date,
        end_date=bt_end,
        top_tickers=top_tickers,
        models_buy=models_buy_ytd,
        models_sell=models_sell_ytd,
        scalers=scalers_ytd,
        market_data=market_data, # Use the same market data as 1-year backtest
        optimized_params_per_ticker=optimized_params_per_ticker,
        capital_per_stock=capital_per_stock,
        target_percentage=target_percentage,
        run_parallel=run_parallel,
        period_name="YTD"
    )
    ai_ytd_return = ((final_strategy_value_ytd - INITIAL_BALANCE) / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0

    # --- Calculate Buy & Hold for YTD ---
    buy_hold_results_ytd = []
    for ticker in processed_tickers_ytd_local: # Use processed_tickers_ytd_local here
        df_bh = load_prices_robust(ticker, ytd_start_date, bt_end)
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            buy_hold_results_ytd.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
        else:
            buy_hold_results_ytd.append(capital_per_stock) # If no data, assume initial capital
    final_buy_hold_value_ytd = sum(buy_hold_results_ytd) + (len(top_tickers) - len(processed_tickers_ytd_local)) * capital_per_stock

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    final_results = []
    for i, ticker in enumerate(processed_tickers_1y):
        # The performance_metrics_1y list contains the dictionaries returned by backtest_worker
        # Each dictionary has 'perf_data', 'individual_bh_return', and 'last_ai_action'
        backtest_result_for_ticker = next((res for res in performance_metrics_1y if res['ticker'] == ticker), None)
        
        if backtest_result_for_ticker:
            perf_data = backtest_result_for_ticker['perf_data']
            individual_bh_return = backtest_result_for_ticker['individual_bh_return']
            last_ai_action = backtest_result_for_ticker['last_ai_action']
        else:
            # Fallback if ticker not found in backtest_results (should not happen if processed_tickers_1y is accurate)
            perf_data = {'sharpe_ratio': 0.0}
            individual_bh_return = 0.0
            last_ai_action = "N/A"

        # Find the corresponding performance data (1Y and YTD from find_top_performers)
        perf_1y_benchmark, perf_ytd_benchmark = -np.inf, -np.inf
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y
                perf_ytd_benchmark = pytd
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': strategy_results_1y[i],
            'sharpe': perf_data['sharpe_ratio'],
            'one_year_perf': perf_1y_benchmark,
            'ytd_perf': perf_ytd_benchmark, # Use perf_ytd_benchmark here for consistency
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action
        })
    
    # Sort by 1Y performance for the final table
    sorted_final_results = sorted(final_results, key=lambda x: x['one_year_perf'], reverse=True)
    
    print_final_summary(sorted_final_results, models_buy, models_sell, scalers, optimized_params_per_ticker,
                        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
                        final_strategy_value_ytd, final_buy_hold_value_ytd, ai_ytd_return)
    
    return final_strategy_value_1y, final_buy_hold_value_1y, models_buy, models_sell, scalers, top_performers_data, strategy_results_1y, processed_tickers_1y, performance_metrics_1y, ai_1y_return, ai_ytd_return

def _run_portfolio_backtest(
    start_date: datetime,
    end_date: datetime,
    top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    market_data: Optional[pd.DataFrame],
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    capital_per_stock: float,
    target_percentage: float,
    run_parallel: bool,
    period_name: str
) -> Tuple[float, List[float], List[str], List[Dict]]:
    """
    Orchestrates the backtesting of a portfolio of tickers, handling model loading/saving
    and parallel execution.
    """
    import joblib
    _ensure_dir(Path("logs/models"))
    _ensure_dir(Path("logs"))

    backtest_params = []
    processed_tickers = []
    
    for ticker in top_tickers:
        model_buy = models_buy.get(ticker)
        model_sell = models_sell.get(ticker)
        scaler = scalers.get(ticker)

        # Load optimized parameters if available, otherwise use global defaults
        per_ticker_min_proba_buy = MIN_PROBA_BUY
        per_ticker_min_proba_sell = MIN_PROBA_SELL
        if optimized_params_per_ticker and ticker in optimized_params_per_ticker:
            if 'min_proba_buy' in optimized_params_per_ticker[ticker]:
                per_ticker_min_proba_buy = optimized_params_per_ticker[ticker]['min_proba_buy']
            if 'min_proba_sell' in optimized_params_per_ticker[ticker]:
                per_ticker_min_proba_sell = optimized_params_per_ticker[ticker]['min_proba_sell']

        # If models are not in memory, try to load them from disk
        if model_buy is None and scaler is None:
            try:
                model_buy = joblib.load(f"logs/models/{ticker}_model_buy.joblib")
                model_sell = joblib.load(f"logs/models/{ticker}_model_sell.joblib")
                scaler = joblib.load(f"logs/models/{ticker}_scaler.joblib")
                # Re-add to in-memory dicts for subsequent use
                models_buy[ticker] = model_buy
                models_sell[ticker] = model_sell
                scalers[ticker] = scaler
            except FileNotFoundError:
                # print(f"  ⚠️ No saved models found for {ticker}. Skipping backtest for this ticker.")
                continue
            except Exception as e:
                print(f"  ⚠️ Error loading models for {ticker}: {e}. Skipping backtest for this ticker.")
                continue
        
        # Ensure scaler has feature_names_in_ attribute for consistent use in RuleTradingEnv
        if scaler is not None and not hasattr(scaler, 'feature_names_in_'):
            # This is a fallback, ideally feature_names_in_ is set during training
            # For now, we'll use a default set, but this might need to be more robust
            scaler.feature_names_in_ = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper", 
                                        'Fin_Revenue', 'Fin_NetIncome', 'Fin_TotalAssets', 'Fin_TotalLiabilities', 'Fin_FreeCashFlow', 'Fin_EBITDA']

        # Only add to backtest_params if we have a model and scaler
        if model_buy and scaler:
            # Pass the feature_set from the scaler to the RuleTradingEnv
            feature_set_for_env = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
            backtest_params.append((ticker, start_date, end_date, capital_per_stock, model_buy, model_sell, scaler, market_data, feature_set_for_env, per_ticker_min_proba_buy, per_ticker_min_proba_sell))
            processed_tickers.append(ticker)
        else:
            print(f"  ⚠️ Skipping backtest for {ticker} due to missing model or scaler.")

    if not backtest_params:
        print(f"❌ No tickers with valid models/scalers to backtest for {period_name} period.")
        return INITIAL_BALANCE, [], [], []

    print(f"\n🔍 Step 4: Running {period_name} backtest for {len(processed_tickers)} tickers...")
    num_processes = max(1, cpu_count() - 2)

    if run_parallel:
        print(f"📈 Running {period_name} backtests in parallel using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            backtest_results = list(tqdm(pool.imap(backtest_worker, backtest_params), total=len(backtest_params), desc=f"Backtesting {period_name}"))
    else:
        print(f"📈 Running {period_name} backtests sequentially...")
        backtest_results = [backtest_worker(p) for p in tqdm(backtest_params, desc=f"Backtesting {period_name}")]

    # Filter out None results (e.g., from tickers with insufficient data)
    backtest_results = [res for res in backtest_results if res is not None]

    if not backtest_results:
        print(f"❌ No successful backtest results for {period_name} period.")
        return INITIAL_BALANCE, [], [], []

    final_strategy_value = sum(res['final_val'] for res in backtest_results) + (len(top_tickers) - len(processed_tickers)) * capital_per_stock
    strategy_results = [res['final_val'] for res in backtest_results]
    performance_metrics = [res for res in backtest_results] # Return full backtest_results for main to access individual_bh_return and last_ai_action

    print(f"\n--- {period_name} Backtest Summary ---")
    print(f"  Total Initial Capital: ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Strategy Value:  ${final_strategy_value:,.2f}")
    print(f"  Strategy Return:       {((final_strategy_value - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:.2f}%")
    print("-" * 30)

    return final_strategy_value, strategy_results, processed_tickers, performance_metrics

def print_final_summary(
    sorted_final_results: List[Dict],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    final_strategy_value_1y: float,
    final_buy_hold_value_1y: float,
    ai_1y_return: float,
    final_strategy_value_ytd: float,
    final_buy_hold_value_ytd: float,
    ai_ytd_return: float
):
    """Prints the final summary table and saves models."""
    import joblib
    
    print("\n\n====================================================================================================")
    print("                                 🚀 AI-Powered Momentum & Trend Strategy Results                               ")
    print("====================================================================================================")
    
    print(f"\nOverall Portfolio Performance (1-Year):")
    print(f"  Strategy Final Value: ${final_strategy_value_1y:,.2f} (Return: {ai_1y_return:.2f}%)")
    print(f"  Buy & Hold Final Value: ${final_buy_hold_value_1y:,.2f} (Return: {((final_buy_hold_value_1y - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:.2f}%)")

    print(f"\nOverall Portfolio Performance (YTD):")
    print(f"  Strategy Final Value: ${final_strategy_value_ytd:,.2f} (Return: {ai_ytd_return:.2f}%)")
    print(f"  Buy & Hold Final Value: ${final_buy_hold_value_ytd:,.2f} (Return: {((final_buy_hold_value_ytd - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:.2f}%)")

    print(f"\nML Model Thresholds:")
    print(f"  Minimum Buy Probability: {MIN_PROBA_BUY:.2f}")
    print(f"  Minimum Sell Probability: {MIN_PROBA_SELL:.2f}")

    print("\nIndividual Ticker Performance (1-Year Backtest):")
    print(f"{'Rank':<5} | {'Ticker':<10} | {'1Y Perf (%)':>12} | {'BH Perf (%)':>12} | {'Strategy Value':>18} | {'Sharpe Ratio':>14} | {'Min Buy Proba':>13} | {'Min Sell Proba':>14} | {'AI Action':<10}")
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    for i, res in enumerate(sorted_final_results, 1):
        ticker = res['ticker']
        strategy_value = res['performance']
        sharpe = res['sharpe']
        one_year_perf = res['one_year_perf']
        individual_bh_return = res['individual_bh_return'] # Get individual BH return
        last_ai_action = res['last_ai_action'] # Get last AI action

        # Get per-ticker thresholds or use global defaults
        min_proba_buy_ticker = MIN_PROBA_BUY
        min_proba_sell_ticker = MIN_PROBA_SELL
        if optimized_params_per_ticker and ticker in optimized_params_per_ticker:
            if 'min_proba_buy' in optimized_params_per_ticker[ticker]:
                min_proba_buy_ticker = optimized_params_per_ticker[ticker]['min_proba_buy']
            if 'min_proba_sell' in optimized_params_per_ticker[ticker]:
                min_proba_sell_ticker = optimized_params_per_ticker[ticker]['min_proba_sell']
        
        print(f"{i:<5} | {ticker:<10} | {one_year_perf:>12.2f} | {individual_bh_return:>12.2f} | {strategy_value:>18,.2f} | {sharpe:>14.2f} | {min_proba_buy_ticker:>13.2f} | {min_proba_sell_ticker:>14.2f} | {last_ai_action:<10}")

    print("==================================================================================================================================================================================================")
    
    print("\nOverall Recommendation:")
    if ai_1y_return > ((final_buy_hold_value_1y - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:
        print(f"The AI strategy outperformed a simple Buy & Hold strategy over the 1-Year period. Consider deploying this strategy.")
    else:
        print(f"The AI strategy did not outperform a simple Buy & Hold strategy over the 1-Year period. Further optimization may be needed.")
    
    if sorted_final_results:
        top_ticker = sorted_final_results[0]['ticker']
        top_perf = sorted_final_results[0]['one_year_perf']
        print(f"The top performing ticker in the backtest was {top_ticker} with a 1-Year benchmark performance of {top_perf:.2f}%.")

    print("\nSaving trained models and scalers...")
    for ticker, model_buy in models_buy.items():
        try:
            joblib.dump(model_buy, f"logs/models/{ticker}_model_buy.joblib")
            joblib.dump(models_sell[ticker], f"logs/models/{ticker}_model_sell.joblib")
            joblib.dump(scalers[ticker], f"logs/models/{ticker}_scaler.joblib")
            print(f"  ✅ Saved models for {ticker}")
        except Exception as e:
            print(f"  ⚠️ Could not save models for {ticker}: {e}")
            
    if optimized_params_per_ticker:
        try:
            with open(TOP_CACHE_PATH.parent / "optimized_per_ticker_params.json", 'w') as f:
                json.dump(optimized_params_per_ticker, f, indent=4)
            print(f"  ✅ Saved optimized parameters to {TOP_CACHE_PATH.parent.name}/{Path('optimized_per_ticker_params.json').name}")
        except Exception as e:
            print(f"  ⚠️ Could not save optimized parameters: {e}")

    print("\nAnalysis complete. Check 'logs/models/' for saved models and 'logs/optimized_per_ticker_params.json' for optimized parameters.")

if __name__ == "__main__":
    main()
