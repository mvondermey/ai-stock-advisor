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
from tqdm.contrib.concurrent import process_map # Re-import process_map for parallel progress bars

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

import joblib # New: Import joblib for model serialization

def _process_ticker_wrapper(args):
    """Wrapper function to unpack arguments for process_ticker."""
    return process_ticker(*args)

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
OPTIMIZED_PARAMS_PATH   = Path("logs/optimized_per_ticker_params.json") # New: Path for optimized params
MODEL_CACHE_DIR         = Path("logs/models") # New: Directory for cached models
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
N_TOP_TICKERS           = 50          # Set to 0 to disable the limit and run on all performers
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
MIN_PROBA_BUY           = 0.8       # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.8       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.01       # 1% target for buy/sell classification
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # re-enable market filter
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = False

# ============================
# Helpers
# ============================

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Directory ensured: {p}")
    except Exception as e:
        print(f"  ⚠️ Could not ensure directory {p}: {e}")


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

def _normalize_symbol(symbol: str, provider: str) -> str:
    """Normalizes ticker symbols for different data providers."""
    if provider.lower() == 'stooq':
        return symbol
    elif provider.lower() == 'yahoo':
        return symbol.replace('.', '-')
    return symbol

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
        print(f"⚠️ Could not fetch S&P 500 list ({e}). Using static fallback.")
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
            url_nasdaq = 'https://en.wikipedia.org/wiki/NASDAQ-100'
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
            print(f"✅ Fetched {len(sp500_tickers)} tickers from S&P 500.")
        except Exception as e:
            print(f"⚠️ Could not fetch S&P 500 list ({e}).")

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
    if df.empty:
        print(f"⚠️ No data fetched for {ticker} from {start.date()} to {end.date()}. Returning empty DataFrame.")
        return pd.DataFrame(), []

    # Ensure enough data for basic SMA calculations, otherwise warn but proceed
    if len(df) < FEAT_SMA_LONG + 10:
        print(f"  ℹ️ Potentially insufficient data for all features for {ticker} (only {len(df)} rows). Proceeding with available data.")

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

    # Stochastic Oscillator
    df['%K'] = ((df['Close'] - df['Low'].rolling(window=14).min()) / 
                (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Williams %R
    df['%R'] = ((df['High'].rolling(window=14).max() - df['Close']) / 
                (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * -100

    # Average Directional Index (ADX)
    # Calculate True Range (TR)
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift(1))
    low_prev_close = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # Calculate Directional Movement (DM)
    df['+DM'] = df['High'] - df['High'].shift(1)
    df['-DM'] = df['Low'].shift(1) - df['Low']

    df['+DM'] = df['+DM'].where( (df['+DM'] > 0) & (df['+DM'] > df['-DM']), 0)
    df['-DM'] = df['-DM'].where( (df['-DM'] > 0) & (df['-DM'] > df['+DM']), 0)

    # Calculate Smoothed True Range (ATR) and Directional Movement (ADX components)
    df['ATR_ADX'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['+DI'] = (df['+DM'].ewm(span=14, adjust=False).mean() / df['ATR_ADX']) * 100
    df['-DI'] = (df['-DM'].ewm(span=14, adjust=False).mean() / df['ATR_ADX']) * 100

    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()
    
    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]
    
    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill any remaining NaNs after all feature calculations to ensure columns are present
    df = df.fillna(0)

    df["Target"]     = df["Close"].shift(-1)

    # Classification label for BUY model: 5-day forward > +target_percentage
    fwd = df["Close"].shift(-CLASS_HORIZON)
    df["TargetClassBuy"] = ((fwd / df["Close"] - 1.0) > target_percentage).astype(float)

    # Classification label for SELL model: 5-day forward < -target_percentage
    df["TargetClassSell"] = ((fwd / df["Close"] - 1.0) < -target_percentage).astype(float)

    # New: Regression target for 1-year price change
    fwd_1y = df["Close"].shift(-252) # Approximately 252 trading days in a year
    df["Target1YChange"] = (fwd_1y / df["Close"] - 1.0) * 100 # Percentage change

    # Define core technical features
    core_tech_features = ["Close","Returns","SMA_F_S","SMA_F_L","Volatility", "RSI_feat", "MACD", "BB_upper", "%K", "%D", "%R", "ADX"]
    
    # Combine core technical features with dynamically found financial features
    all_potential_features = core_tech_features + financial_features
    
    # Filter to include only features actually present in the DataFrame
    available_features = [col for col in all_potential_features if col in df.columns]

    # Add target columns
    target_cols = ["Target", "TargetClassBuy", "TargetClassSell", "Target1YChange"]
    available_target_cols = [col for col in target_cols if col in df.columns]

    # Combine available features and target columns
    req_cols = available_features + available_target_cols
    
    # Create 'ready' DataFrame, ensuring all required columns are present and filled
    ready = df[req_cols].copy()
    
    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in available_features if col in ready.columns] # Use available_features as base
    
    # Drop rows with NaN values in the final feature set, but only if they are critical
    # For now, rely on fillna(0) to keep all rows. If models require no NaNs, this might need adjustment.
    # ready = ready.dropna() # Removed this line as fillna(0) should handle it.
    
    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell", "Target1YChange"]]

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
    # Ensure features are present (same logic as in classification models)
    if "Returns" not in df.columns and "Close" in df.columns:
        df["Returns"] = df["Close"].pct_change()
    if "SMA_F_S" not in df.columns and "Close" in df.columns:
        df["SMA_F_S"] = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    if "SMA_F_L" not in df.columns and "Close" in df.columns:
        df["SMA_F_L"] = df["Close"].rolling(FEAT_SMA_LONG).mean()
    if "Volatility" not in df.columns and "Returns" in df.columns:
        df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()
    
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

    financial_features_present = [col for col in df.columns if col.startswith('Fin_')]
    for col in financial_features_present:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # df is already the 'ready' DataFrame from fetch_training_data, so it's already cleaned and has features.
    # No need for redundant feature generation or another dropna() here.

    if target_col not in df.columns:
        print(f"⚠️ Target column '{target_col}' not found in DataFrame. Skipping model training.")
        return None

    if feature_set is None:
        print("⚠️ feature_set is None in train_and_evaluate_models. Skipping model training.")
        return None
    
    actual_features = [f for f in feature_set if f in df.columns]

    if not actual_features:
        print(f"⚠️ No valid features available for training on '{target_col}'. Skipping model training.")
        return None

    # Check if target_col is present and has enough diversity
    if df[target_col].nunique() < 2:
        print(f"⚠️ Not enough class diversity for training on '{target_col}'. Skipping model.")
        return None

    # Use the already cleaned and prepared DataFrame 'df' directly
    X_df = df[actual_features]
    y = df[target_col].values

    if len(X_df) < 50:  # Increased requirement for cross-validation
        print("⚠️ Not enough rows after feature prep to compare models (need ≥ 50). Skipping.")
        return None

    # Scale features for models that are sensitive to scale (like Logistic Regression and SVC)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X = pd.DataFrame(X_scaled, columns=actual_features, index=X_df.index)
    
    # Store feature names for consistent use during prediction
    scaler.feature_names_in_ = list(actual_features) 

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

def train_and_evaluate_regression_model(df: pd.DataFrame, target_col: str = "Target1YChange", feature_set: Optional[List[str]] = None):
    """Train and compare multiple regression models for a given target, returning the best one."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")
        LGBMRegressor = None

    df = df.copy()
    # Ensure features are present (same logic as in classification models)
    if "Returns" not in df.columns and "Close" in df.columns:
        df["Returns"] = df["Close"].pct_change()
    if "SMA_F_S" not in df.columns and "Close" in df.columns:
        df["SMA_F_S"] = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    if "SMA_F_L" not in df.columns and "Close" in df.columns:
        df["SMA_F_L"] = df["Close"].rolling(FEAT_SMA_LONG).mean()
    if "Volatility" not in df.columns and "Returns" in df.columns:
        df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()
    
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

    financial_features_present = [col for col in df.columns if col.startswith('Fin_')]
    for col in financial_features_present:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if target_col not in df.columns:
        print(f"⚠️ Target column '{target_col}' not found in DataFrame. Skipping regression model training.")
        return None

    if feature_set is None:
        print("⚠️ feature_set is None in train_and_evaluate_regression_model. Skipping regression model training.")
        return None
    
    actual_features = [f for f in feature_set if f in df.columns]

    if not actual_features:
        print(f"⚠️ No valid features available for training regression model on '{target_col}'. Skipping.")
        return None

    X_df = df[actual_features]
    y = df[target_col].values

    if len(X_df) < 50:
        print("⚠️ Not enough rows after feature prep to compare regression models (need ≥ 50). Skipping.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X = pd.DataFrame(X_scaled, columns=actual_features, index=X_df.index)
    scaler.feature_names_in_ = list(actual_features)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "SVM Regressor": SVR()
    }

    if LGBMRegressor:
        models["LightGBM Regressor"] = LGBMRegressor(random_state=SEED, verbosity=-1)

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    print("  🔬 Comparing regression model performance (R2 score via 5-fold cross-validation):")
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=1)
            results[name] = np.mean(scores)
            print(f"    - {name}: {results[name]:.4f} (std: {np.std(scores):.4f})")
        except Exception as e:
            print(f"    - {name}: Failed evaluation. Error: {e}")
            results[name] = -np.inf # Assign a very low score for failed models

    if not any(results.values()) or all(v == -np.inf for v in results.values()):
        print("  ⚠️ All regression models failed evaluation. No regression model will be used.")
        return None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best regression model: {best_model_name} with R2 = {results[best_model_name]:.4f}")

    best_model_instance = models[best_model_name]
    best_model_instance.fit(X, y)
    
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

        hl = None # Initialize to None
        h_pc = None # Initialize to None
        l_pc = None # Initialize to None
        tr = None # Initialize to None

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

        # Debugging: Print current state
        # print(f"  [{self.ticker}] Step {self.current_step}: Date={date}, Price={price:.2f}, Cash={self.cash:,.2f}, Shares={self.shares:.2f}, Portfolio={self.cash + self.shares * price:,.2f}")

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
        # Filter: Trend must be up (price above 200-day SMA)
        sma_200 = row.get('SMA_200')
        # trend_ok = price > sma_200 if sma_200 and not np.isnan(sma_200) else False # Not used, can remove

        # Trigger: Price above long SMA (no crossover)
        # sma_l = row.get('SMA_L') # Not used, can remove

        # --- Trend-Following Entry Signal ---
        # sma_s = row.get('SMA_S') # Not used, can remove
        # sma_l = row.get('SMA_L') # Not used, can remove
        # sma_200 = row.get('SMA_200') # Already fetched above

        # Condition: AI model is now the primary buy signal generator.
        if self.shares == 0 and self._allow_buy_by_model(self.current_step):
            self._buy(price, atr, date)
        
        # --- AI-driven Exit Signal ---
        if self.shares > 0 and self._allow_sell_by_model(self.current_step):
            self._sell(price, date)
            
        # Original Exit Logic (ATR-based) is kept as a fallback/stop-loss
        # --- This section has been disabled to rely solely on the AI sell model ---
        # if self.shares > 0:
        #     if self.highest_since_entry is None or price > self.highest_since_entry:
        #         self.highest_since_entry = price
        #     self.holding_bars += 1
            
        #     tp_level = self.entry_price * (1 + ATR_MULT_TP * (atr / price)) if (atr and price > 0) else self.entry_price * (1 + 0.12)
        #     tsl_level = None
        #     if self.highest_since_entry is not None and atr is not None:
        #         tsl_level = self.highest_since_entry - ATR_MULT_TRAIL * atr

        #     hit_tp = price >= tp_level
        #     hit_trail = (tsl_level is not None) and (price <= tsl_level)
            
        #     if hit_tp or hit_trail:
        #         self._sell(price, date)

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple]]:
        done = False
        while not done:
            done = self.step()
        if self.shares > 0:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log

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

# ============================
# Optimization & Backtesting
# ============================

def _process_ticker_wrapper(args):
    """Wrapper function to unpack arguments for process_ticker."""
    return process_ticker(*args)

def process_ticker(ticker: str, start_date: datetime, end_date: datetime,
                   market_data: Optional[pd.DataFrame] = None) -> Dict: # Changed return type to Dict
    """
    Processes a single ticker: fetches data, conditionally trains/loads models,
    performs backtesting, and returns performance metrics and recommendations.
    """
    _ensure_dir(MODEL_CACHE_DIR)
    model_buy_path = MODEL_CACHE_DIR / f"{ticker}_model_buy.joblib"
    model_sell_path = MODEL_CACHE_DIR / f"{ticker}_model_sell.joblib"
    scaler_path = MODEL_CACHE_DIR / f"{ticker}_scaler.joblib"
    optimized_params_path = OPTIMIZED_PARAMS_PATH

    model_buy, model_sell, scaler = None, None, None
    per_ticker_min_proba_buy, per_ticker_min_proba_sell = MIN_PROBA_BUY, MIN_PROBA_SELL
    feature_set = [] # Initialize feature_set here

    # Initialize result dictionary with default values
    result = {
        "ticker": ticker,
        "1y_performance": 0.0,
        "ytd_performance": 0.0,
        "strategy_final_value": INITIAL_BALANCE,
        "buy_hold_return": 0.0,
        "recommendation": "HOLD",
        "min_proba_buy": MIN_PROBA_BUY,  # Include default buy threshold
        "min_proba_sell": MIN_PROBA_SELL # Include default sell threshold
    }

    # --- Conditional Model Loading/Training ---
    if USE_MODEL_GATE:
        if model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
            print(f"  [{ticker}] Loading existing models and scaler...")
            try:
                model_buy = joblib.load(model_buy_path)
                model_sell = joblib.load(model_sell_path)
                scaler = joblib.load(scaler_path)
                feature_set = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else []
                # Load optimized thresholds if they exist
                if optimized_params_path.exists():
                    with open(optimized_params_path, 'r') as f:
                        optimized_params = json.load(f)
                    if ticker in optimized_params:
                        per_ticker_min_proba_buy = optimized_params[ticker].get('min_proba_buy', MIN_PROBA_BUY)
                        per_ticker_min_proba_sell = optimized_params[ticker].get('min_proba_sell', MIN_PROBA_SELL)
                        print(f"  [{ticker}] Loaded optimized thresholds: Buy={per_ticker_min_proba_buy:.2f}, Sell={per_ticker_min_proba_sell:.2f}")
                        result["min_proba_buy"] = per_ticker_min_proba_buy
                        result["min_proba_sell"] = per_ticker_min_proba_sell
            except Exception as e:
                print(f"  ⚠️ [{ticker}] Error loading models/scaler: {e}. Retraining.")
                model_buy, model_sell, scaler = None, None, None # Reset to force retraining
        
        if model_buy is None or model_sell is None or scaler is None:
            print(f"  [{ticker}] Training new models...")
            df_train, feature_set = fetch_training_data(ticker, start=start_date - timedelta(days=TRAIN_LOOKBACK_DAYS), end=end_date)
            if df_train.empty:
                print(f"  ⚠️ [{ticker}] Skipping model training due to insufficient training data.")
                model_buy, scaler_buy = None, None
                model_sell, scaler_sell = None, None
            else:
                model_buy_result = train_and_evaluate_models(df_train, target_col="TargetClassBuy", feature_set=feature_set)
                if model_buy_result is not None:
                    model_buy, scaler_buy = model_buy_result
                else:
                    model_buy, scaler_buy = None, None # Explicitly set to None if training failed

                model_sell_result = train_and_evaluate_models(df_train, target_col="TargetClassSell", feature_set=feature_set)
                if model_sell_result is not None:
                    model_sell, scaler_sell = model_sell_result
                else:
                    model_sell, scaler_sell = None, None # Explicitly set to None if training failed
                
                # Ensure scalers are consistent; ideally, they should be the same if feature_set is identical
                if scaler_buy and scaler_sell and all(f in scaler_buy.feature_names_in_ for f in scaler_sell.feature_names_in_) and all(f in scaler_sell.feature_names_in_ for f in scaler_buy.feature_names_in_):
                    scaler = scaler_buy # Use one consistent scaler
                elif scaler_buy:
                    scaler = scaler_buy
                elif scaler_sell:
                    scaler = scaler_sell
                
                if model_buy and model_sell and scaler:
                    try:
                        joblib.dump(model_buy, model_buy_path)
                        joblib.dump(model_sell, model_sell_path)
                        joblib.dump(scaler, scaler_path)
                        print(f"  [{ticker}] Models and scaler saved.")
                    except Exception as e:
                        print(f"  ⚠️ [{ticker}] Could not save models/scaler: {e}")
                else:
                    print(f"  ⚠️ [{ticker}] Model training failed or resulted in no valid models. Skipping ML gate.")
                    # USE_MODEL_GATE = False # Temporarily disable ML gate for this ticker if models fail

    # Update thresholds in result dictionary after potential training/loading
    result["min_proba_buy"] = per_ticker_min_proba_buy
    result["min_proba_sell"] = per_ticker_min_proba_sell

    # Determine if ML gate can be used for this ticker
    use_ml_gate_for_ticker = USE_MODEL_GATE and (model_buy is not None) and (model_sell is not None) and (scaler is not None) and (feature_set is not None)

    # --- Backtesting ---
    df_backtest = load_prices(ticker, start_date, end_date)
    if df_backtest.empty or len(df_backtest) < STRAT_SMA_LONG:
        print(f"  ⚠️ [{ticker}] Insufficient data for backtest. Skipping.")
        return result # Return default result if backtest data is insufficient

    # Calculate buy & hold performance for comparison
    buy_hold_initial = float(df_backtest["Close"].iloc[0])
    buy_hold_final = float(df_backtest["Close"].iloc[-1])
    buy_hold_return = ((buy_hold_final - buy_hold_initial) / buy_hold_initial) * 100 if buy_hold_initial > 0 else 0.0
    buy_hold_history = [INITIAL_BALANCE * (1 + (c - buy_hold_initial) / buy_hold_initial) for c in df_backtest["Close"]]

    env = RuleTradingEnv(df_backtest, INITIAL_BALANCE, TRANSACTION_COST,
                         model_buy=model_buy, model_sell=model_sell, scaler=scaler,
                         min_proba_buy=per_ticker_min_proba_buy, min_proba_sell=per_ticker_min_proba_sell,
                         use_gate=use_ml_gate_for_ticker, market_data=market_data, use_market_filter=USE_MARKET_FILTER,
                         feature_set=feature_set) # Pass feature_set to env
    
    final_portfolio_value, trade_log = env.run()
    
    # Calculate 1-year and YTD performance for the current ticker
    perf_1y = ((df_backtest['Close'].iloc[-1] - df_backtest['Close'].iloc[0]) / df_backtest['Close'].iloc[0]) * 100 if df_backtest['Close'].iloc[0] > 0 else 0.0
    
    ytd_start_price = df_backtest.loc[df_backtest.index.year == end_date.year, 'Close'].iloc[0] if not df_backtest.loc[df_backtest.index.year == end_date.year, 'Close'].empty else df_backtest['Close'].iloc[0]
    perf_ytd = ((df_backtest['Close'].iloc[-1] - ytd_start_price) / ytd_start_price) * 100 if ytd_start_price > 0 else 0.0

    # Update result dictionary
    result.update({
        "1y_performance": perf_1y,
        "ytd_performance": perf_ytd,
        "strategy_final_value": final_portfolio_value,
        "buy_hold_return": buy_hold_return
    })

    # Get today's recommendation
    if use_ml_gate_for_ticker: # Use the local flag here
        # To get the latest data for prediction, we need to ensure it has all features
        # We'll fetch a small window of data ending today to calculate features
        prediction_end_date = datetime.now(timezone.utc)
        prediction_start_date = prediction_end_date - timedelta(days=FEAT_SMA_LONG + 10) # Enough data for all features
        
        df_predict, _ = fetch_training_data(ticker, start=prediction_start_date, end=prediction_end_date)
        
        if df_predict.empty:
            print(f"  ⚠️ [{ticker}] Insufficient data for latest prediction. Skipping recommendation.")
        else:
            latest_data_for_prediction = df_predict.iloc[[-1]].copy() # Get the last row as a DataFrame
            
            # Ensure feature_set is available and consistent
            # The fetch_training_data function already ensures all required columns are present and filled with 0
            X_latest = latest_data_for_prediction[feature_set]
            
            # Check if scaler has feature_names_in_ and align columns
            if hasattr(scaler, 'feature_names_in_') and list(X_latest.columns) != list(scaler.feature_names_in_):
                # Reindex X_latest to match scaler's feature order, filling missing with 0
                X_latest = X_latest.reindex(columns=scaler.feature_names_in_, fill_value=0)

            X_scaled_latest = scaler.transform(X_latest)
            
            buy_proba = model_buy.predict_proba(X_scaled_latest)[0][1]
            sell_proba = model_sell.predict_proba(X_scaled_latest)[0][1]

            if buy_proba >= per_ticker_min_proba_buy:
                result["recommendation"] = "BUY"
            elif sell_proba >= per_ticker_min_proba_sell:
                result["recommendation"] = "SELL"

    return result

    # Calculate buy & hold performance for comparison
    buy_hold_initial = float(df_backtest["Close"].iloc[0])
    buy_hold_final = float(df_backtest["Close"].iloc[-1])
    buy_hold_return = ((buy_hold_final - buy_hold_initial) / buy_hold_initial) * 100 if buy_hold_initial > 0 else 0.0
    buy_hold_history = [INITIAL_BALANCE * (1 + (c - buy_hold_initial) / buy_hold_initial) for c in df_backtest["Close"]]

    env = RuleTradingEnv(df_backtest, INITIAL_BALANCE, TRANSACTION_COST,
                         model_buy=model_buy, model_sell=model_sell, scaler=scaler,
                         min_proba_buy=per_ticker_min_proba_buy, min_proba_sell=per_ticker_min_proba_sell,
                         use_gate=USE_MODEL_GATE, market_data=market_data, use_market_filter=USE_MARKET_FILTER,
                         feature_set=feature_set) # Pass feature_set to env
    
    final_portfolio_value, trade_log = env.run()
    
    # Calculate 1-year and YTD performance for the current ticker
    perf_1y = ((df_backtest['Close'].iloc[-1] - df_backtest['Close'].iloc[0]) / df_backtest['Close'].iloc[0]) * 100 if df_backtest['Close'].iloc[0] > 0 else 0.0
    
    ytd_start_price = df_backtest.loc[df_backtest.index.year == end_date.year, 'Close'].iloc[0] if not df_backtest.loc[df_backtest.index.year == end_date.year, 'Close'].empty else df_backtest['Close'].iloc[0]
    perf_ytd = ((df_backtest['Close'].iloc[-1] - ytd_start_price) / ytd_start_price) * 100 if ytd_start_price > 0 else 0.0

    # Update result dictionary
    result.update({
        "1y_performance": perf_1y,
        "ytd_performance": perf_ytd,
        "strategy_final_value": final_portfolio_value,
        "buy_hold_return": buy_hold_return
    })

    # Get today's recommendation
    if USE_MODEL_GATE and model_buy and model_sell and scaler and feature_set:
        # To get the latest data for prediction, we need to ensure it has all features
        # We'll fetch a small window of data ending today to calculate features
        prediction_end_date = datetime.now(timezone.utc)
        prediction_start_date = prediction_end_date - timedelta(days=FEAT_SMA_LONG + 10) # Enough data for all features
        
        df_predict, _ = fetch_training_data(ticker, start=prediction_start_date, end=prediction_end_date)
        
        if df_predict.empty:
            print(f"  ⚠️ [{ticker}] Insufficient data for latest prediction. Skipping recommendation.")
        else:
            latest_data_for_prediction = df_predict.iloc[[-1]].copy() # Get the last row as a DataFrame
            
            # Ensure feature_set is available and consistent
            # The fetch_training_data function already ensures all required columns are present and filled with 0
            X_latest = latest_data_for_prediction[feature_set]
            
            # Check if scaler has feature_names_in_ and align columns
            if hasattr(scaler, 'feature_names_in_') and list(X_latest.columns) != list(scaler.feature_names_in_):
                # Reindex X_latest to match scaler's feature order, filling missing with 0
                X_latest = X_latest.reindex(columns=scaler.feature_names_in_, fill_value=0)

            X_scaled_latest = scaler.transform(X_latest)
            
            buy_proba = model_buy.predict_proba(X_scaled_latest)[0][1]
            sell_proba = model_sell.predict_proba(X_scaled_latest)[0][1]

            if buy_proba >= per_ticker_min_proba_buy:
                result["recommendation"] = "BUY"
            elif sell_proba >= per_ticker_min_proba_sell:
                result["recommendation"] = "SELL"

    return result

def find_top_performers(n_top: int = N_TOP_TICKERS, fcf_min_threshold: float = 0.0, ebitda_min_threshold: float = 0.0):
    """
    Fetches tickers, processes them, and prints a table of recommendations.
    """
    global USE_MARKET_FILTER # Declare global here
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        print("❌ No tickers to process. Exiting.")
        return []

    # Apply N_TOP_TICKERS limit here
    if n_top > 0 and len(all_available_tickers) > n_top:
        import random
        random.seed(SEED)
        tickers_to_process = random.sample(all_available_tickers, n_top)
        print(f"✅ Selected top {n_top} tickers for analysis: {', '.join(tickers_to_process)}")
    else:
        tickers_to_process = all_available_tickers
        print(f"✅ Analyzing all {len(tickers_to_process)} available tickers.")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    
    # Fetch market data once if market filter is enabled
    market_data = None
    if USE_MARKET_FILTER:
        print(f"- Fetching market data for {MARKET_FILTER_TICKER}...")
        market_data = load_prices_robust(MARKET_FILTER_TICKER, start_date - timedelta(days=MARKET_FILTER_SMA), end_date)
        if market_data.empty:
            print(f"  ⚠️ Could not fetch market data for {MARKET_FILTER_TICKER}. Disabling market filter.")
            # Note: USE_MARKET_FILTER is a global variable, modifying it here will affect subsequent calls.
            # For a more robust solution, consider passing it as a parameter or making it a class member.
            USE_MARKET_FILTER = False
        else:
            market_data['SMA_L_MKT'] = market_data['Close'].rolling(window=MARKET_FILTER_SMA).mean()
            market_data = market_data.dropna()

    results = []
    # Use process_map for parallel processing with a progress bar
    # Pass market_data to each worker if needed
    func_args = [(ticker, start_date, end_date, market_data) for ticker in tickers_to_process]
    
    # Use process_map for a progress bar with multiprocessing
    processed_results = process_map(_process_ticker_wrapper, func_args, max_workers=cpu_count())

    for res in processed_results:
        if res:
            results.append(res)

    if not results:
        print("  ⚠️ No strong performers found after analysis.")
        return []

    # Sort results by 1-year performance
    sorted_results = sorted(results, key=lambda x: x['1y_performance'], reverse=True)
    
    print(f"\n\n🏆 Top Stock Recommendations ({len(sorted_results)} stocks) 🏆")
    print("-" * 110)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Recommendation':<15} | {'Buy Threshold':<15} | {'Sell Threshold':<15}")
    print("-" * 110)
    
    for i, res in enumerate(sorted_results, 1):
        print(f"{i:<5} | {res['ticker']:<10} | {res['1y_performance']:>9.2f}% | {res['ytd_performance']:>9.2f}% | {res['recommendation']:<15} | {res['min_proba_buy']:>13.2f} | {res['min_proba_sell']:>14.2f}")
    
    print("-" * 110)
    return sorted_results

if __name__ == "__main__":
    print("Starting AI Stock Advisor...")
    # Model training and optimization are performed conditionally per ticker within process_ticker.
    # N_TOP_TICKERS is set to 10 for a focused test.
    N_TOP_TICKERS = 10
    USE_PERFORMANCE_BENCHMARK = True
    find_top_performers(n_top=N_TOP_TICKERS)
    print("AI Stock Advisor finished.")
