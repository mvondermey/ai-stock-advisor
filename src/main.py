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
from multiprocessing import Pool, cpu_count, current_process
import joblib # Added for model saving/loading
import warnings # Added for warning suppression

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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

# Optional Alpaca provider
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient # Added for trading account access
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest # For submitting orders
    from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus # For order details
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    print("⚠️ Alpaca SDK not installed. Run: pip install alpaca-py. Alpaca data provider will be skipped.")
    ALPACA_AVAILABLE = False

# ============================
# Configuration / Hyperparams
# ============================

SEED                    = 42
np.random.seed(SEED)

# --- Provider & caching
DATA_PROVIDER           = 'twelvedata'    # 'stooq', 'yahoo', 'alpaca', or 'twelvedata'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
VALID_TICKERS_CACHE_PATH = Path("logs/valid_tickers.json")
CACHE_DAYS              = 7

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")

# TwelveData API credentials
TWELVEDATA_API_KEY      = os.environ.get("TWELVEDATA_API_KEY")

# --- Universe / selection
MARKET_SELECTION = {
    "ALPACA_STOCKS": True, # Fetch all tradable US equities from Alpaca
    "NASDAQ_ALL": False,
    "NASDAQ_100": False,
    "SP500": False,
    "DOW_JONES": False,
    "POPULAR_ETFS": False,
    "CRYPTO": False,
    "DAX": False,
    "MDAX": False,
    "SMI": False,
    "FTSE_MIB": False,
}
N_TOP_TICKERS           = 0         # Number of top performers to select (0 to disable limit)
BATCH_DOWNLOAD_SIZE     = 20000       # Reduced batch size for stability
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
NUM_PROCESSES           = max(1, cpu_count() - 5) # Use all but one CPU core for parallel processing

# --- Backtest & training windows
BACKTEST_DAYS           = 365        # 1 year for backtest
BACKTEST_DAYS_3MONTH    = 90         # 3 months for backtest
TRAIN_LOOKBACK_DAYS     = 360        # more data for model (e.g., 1 year)

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 20
STRAT_SMA_LONG          = 100
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 3.5
ATR_MULT_TP             = 0.0        # 0 disables hard TP; rely on trailing
RISK_PER_TRADE          = 0.9       # 1% of capital
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_BUY           = 0.4       # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.4       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.01       # 1% target for buy/sell classification
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # re-enable market filter
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering

# --- Misc
INITIAL_BALANCE         = 50_000.0
SAVE_PLOTS              = False
FORCE_OPTIMIZATION      = False      # Set to True to force re-optimization of ML thresholds

# ============================
# Helpers
# ============================

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _normalize_symbol(symbol: str, provider: str) -> str:
    """Normalizes a ticker symbol for the given data provider."""
    s_ticker = str(symbol).strip()
    if '$' in s_ticker:
        return "" # Or handle as an invalid ticker
    
    # For now, the main normalization is for Yahoo/Stooq US tickers
    if provider.lower() in ['yahoo', 'stooq']:
        if s_ticker.endswith(('.DE', '.MI', '.SW', '.PA', '.AS', '.HE', '.LS', '.BR', '.MC')):
            return s_ticker
        else:
            return s_ticker.replace('.', '-')
    # Alpaca expects symbols without suffixes like '.US'
    if provider.lower() == 'alpaca':
        return s_ticker.replace('.', '-').split('.US')[0]
        
    return s_ticker


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

def _fetch_from_alpaca(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV from Alpaca."""
    if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()
    
    try:
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df
        
        if df.empty:
            return pd.DataFrame()
        
        # Alpaca returns a MultiIndex DataFrame, we need to flatten it
        df = df.loc[ticker]
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        error_msg = str(e)
        if "subscription does not permit querying recent SIP data" in error_msg:
            # This is an expected condition on free Alpaca plans, not a critical error.
            print(f"  ℹ️ Alpaca (free tier) does not provide recent data for {ticker}. Attempting fallback provider.")
        else:
            print(f"  ⚠️ Could not fetch data from Alpaca for {ticker}: {e}")
        return pd.DataFrame()

def _fetch_from_twelvedata(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV from TwelveData."""
    if not TWELVEDATA_API_KEY:
        return pd.DataFrame()

    try:
        # TwelveData API endpoint for historical data
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval=1day&apikey={TWELVEDATA_API_KEY}&start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}&outputsize=5000"
        
        import requests
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()

        if "values" not in data or not data["values"]:
            print(f"  ℹ️ No data found for {ticker} from TwelveData.")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
        df.index.name = "Date"
        
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        
        # Ensure all relevant columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_index()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Error fetching data from TwelveData for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ⚠️ An unexpected error occurred while processing TwelveData for {ticker}: {e}")
        return pd.DataFrame()

# ============================
# Data access
# ============================

def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """A wrapper for load_prices that handles rate limiting with retries and other common API errors."""
    import time
    import random
    max_retries = 5
    base_wait_time = 5  # seconds, increased for more tolerance

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
    max_retries = 7 # Increased retries
    base_wait_time = 30  # seconds, increased for more tolerance

    for attempt in range(max_retries):
        try:
            # Use DOWNLOAD_THREADS for yfinance.download, or False to let yfinance manage
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True, threads=False, keepna=False)
            
            # Critical check: If the dataframe is empty or all values are NaN, it's a failed download.
            if data.empty or data.isnull().all().all():
                # This will be caught by the except block and trigger a retry
                raise ValueError("Batch download failed: DataFrame is empty or all-NaN.")
                
            return data
        except Exception as e:
            error_str = str(e).lower()
            # Catch common yfinance multi-ticker failure messages
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "failed download" in error_str or "batch download failed" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                print(f"  ⚠️ Batch download failed for {len(tickers)} tickers (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ An unexpected error occurred during batch download for {len(tickers)} tickers: {e}. Skipping batch.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to download batch data for {len(tickers)} tickers after {max_retries} retries.")
    return pd.DataFrame()

def _fetch_financial_data(ticker: str) -> pd.DataFrame:
    """Fetch key financial metrics from yfinance and prepare them for merging."""
    time.sleep(PAUSE_BETWEEN_YF_CALLS)
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

def _fetch_financial_data_from_alpaca(ticker: str) -> pd.DataFrame:
    """Placeholder for fetching financial metrics from Alpaca.
    Alpaca's SDK primarily provides market data (bars, quotes, trades) and does not
    directly offer a comprehensive set of fundamental financial statements (like income statements,
    balance sheets, cash flow statements) in the same way yfinance does.
    
    If Alpaca adds this functionality in the future, this function would be updated.
    For now, it returns an empty DataFrame, and the system will fall back to Yahoo Finance
    for fundamental data.
    """
    return pd.DataFrame()


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
        
        if provider == 'twelvedata':
            if not TWELVEDATA_API_KEY:
                print("⚠️ TwelveData is selected but API key is missing. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                twelvedata_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
                if not twelvedata_df.empty:
                    price_df = twelvedata_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ TwelveData fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                    except Exception as e:
                        print(f"  ⚠️ Yahoo Finance fallback also failed for {ticker}: {e}")

        elif provider == 'alpaca':
            if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                print("⚠️ Alpaca is selected but API keys are missing or SDK not available. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                alpaca_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
                if not alpaca_df.empty:
                    price_df = alpaca_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ Alpaca fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                    except Exception as e:
                        print(f"  ⚠️ Yahoo Finance fallback also failed for {ticker}: {e}")
        
        elif provider == 'stooq':
            stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
            if not stooq_df.empty:
                price_df = stooq_df.copy()
            elif USE_YAHOO_FALLBACK:
                print(f"  ℹ️ Stooq data for {ticker} empty. Falling back to Yahoo.")
                try:
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                    if downloaded_df is not None and not downloaded_df.empty:
                        price_df = downloaded_df.dropna()
                except Exception as e:
                    print(f"  ⚠️ Yahoo Finance fallback (after Stooq) also failed for {ticker}: {e}")
        
        if price_df.empty: # If previous provider failed or was yahoo
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    price_df = downloaded_df.dropna()
            except Exception as e:
                print(f"  ⚠️ Final Yahoo download attempt failed for {ticker}: {e}")
            if price_df.empty and pdr is not None and DATA_PROVIDER.lower() != 'stooq':
                print(f"  ℹ️ Yahoo data for {ticker} empty. Falling back to Stooq.")
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
        
        # Convert all relevant columns to numeric, coercing errors to NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        # Replace infinities with NaN
        price_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure 'Volume', 'High', 'Low', 'Open' columns exist, fill with 'Close' or 0 if missing
        if "Volume" not in price_df.columns:
            price_df["Volume"] = 0
        if "High" not in price_df.columns:
            price_df["High"] = price_df["Close"]
        if "Low" not in price_df.columns:
            price_df["Low"] = price_df["Close"]
        if "Open" not in price_df.columns:
            price_df["Open"] = price_df["Close"]
            
       # print(f"DEBUG: In load_prices for {ticker}, 'Volume' column exists: {'Volume' in price_df.columns}, 'High' exists: {'High' in price_df.columns}, 'Low' exists: {'Low' in price_df.columns}, 'Open' exists: {'Open' in price_df.columns}") # Debug print
        
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
        if DATA_PROVIDER.lower() == 'alpaca':
            financial_df = _fetch_financial_data_from_alpaca(ticker)
            if financial_df.empty: # If Alpaca financial data is empty, fall back to Yahoo
                financial_df = _fetch_financial_data(ticker) # This calls the original Yahoo-based function
        else:
            financial_df = _fetch_financial_data(ticker) # This calls the original Yahoo-based function
            
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
        url = "https://en.wikipedia.org/wiki/List_of_S%26P 500_companies"
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

    # --- Alpaca Stocks ---
    if MARKET_SELECTION.get("ALPACA_STOCKS"):
        if ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
                search_params = GetAssetsRequest(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE
                )
                assets = trading_client.get_all_assets(search_params)
                tradable_assets = [a for a in assets if a.tradable]
                alpaca_tickers = [asset.symbol for asset in tradable_assets]
                all_tickers.update(alpaca_tickers)
                print(f"✅ Fetched {len(alpaca_tickers)} tradable US equity tickers from Alpaca.")
            except Exception as e:
                print(f"⚠️ Could not fetch asset list from Alpaca ({e}).")
        else:
            print("⚠️ Alpaca stock selection is enabled, but SDK/API keys are not available.")

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
            ticker_col = 'Ticker'
            mib_tickers = [s if '.' in s else f"{s}.MI" for s in table_mib[ticker_col].tolist()]
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

    print(f"Total unique tickers found: {len(final_tickers)}")
    return sorted(list(final_tickers))


# ============================
# Feature prep & model
# ============================

def fetch_training_data(ticker: str, data: pd.DataFrame, target_percentage: float = 0.05) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ML features from a given DataFrame."""
    if data.empty or len(data) < FEAT_SMA_LONG + 10:
        print(f"  [DIAGNOSTIC] {ticker}: Skipping feature prep. Initial data has {len(data)} rows, required > {FEAT_SMA_LONG + 10}.")
        return pd.DataFrame(), []

    df = data.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    # The following checks are now handled in load_prices, so they are redundant here.
    # if "High" not in df.columns and "Close" in df.columns:
    #     df["High"] = df["Close"]
    # if "Low" not in df.columns and "Close" in df.columns:
    #     df["Low"] = df["Close"]
    # if "Open" not in df.columns and "Close" in df.columns:
    #     df["Open"] = df["Close"]
    # if "Volume" not in df.columns:
    #     df["Volume"] = 0
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
    # Ensure High/Low exist before calculation (redundant here, handled in load_prices)
    # if "High" not in df.columns: df["High"] = df["Close"]
    # if "Low" not in df.columns: df["Low"] = df["Close"]
    
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    # Handle division by zero for %K
    denominator_k = (high_14 - low_14)
    df['%K'] = np.where(denominator_k != 0, (df['Close'] - low_14) / denominator_k * 100, 0)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['%K'] = df['%K'].fillna(0) # Fill NaNs for Stochastic Oscillator
    df['%D'] = df['%D'].fillna(0) # Fill NaNs for Stochastic Oscillator

    # Average Directional Index (ADX)
    # Ensure High/Low exist before calculation (redundant here, handled in load_prices)
    # if "High" not in df.columns: df["High"] = df["Close"]
    # if "Low" not in df.columns: df["Low"] = df["Close"]

    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['+DM'] = df['+DM'].fillna(0) # Fill NaNs for DM

    # Calculate True Range (TR)
    high_low_diff = df['High'] - df['Low']
    high_prev_close_diff_abs = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close_diff_abs = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([high_low_diff, high_prev_close_diff_abs, low_prev_close_diff_abs], axis=1).max(axis=1)
    df['TR'] = df['TR'].fillna(0) # Fill NaNs for TR

    # Calculate Smoothed DM and TR
    alpha = 1/14
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM14'].fillna(0) # Fill NaNs for Smoothed DM
    df['-DM14'] = df['-DM14'].fillna(0) # Fill NaNs for Smoothed DM
    df['TR14'] = df['TR14'].fillna(0) # Fill NaNs for Smoothed TR

    # Calculate Directional Index (DX)
    denominator_dx = (df['+DM14'] + df['-DM14'])
    df['DX'] = np.where(denominator_dx != 0, (abs(df['+DM14'] - df['-DM14']) / denominator_dx) * 100, 0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    df['DX'] = df['DX'].fillna(0) # Fill NaNs for DX
    df['ADX'] = df['ADX'].fillna(0) # Fill NaNs for ADX
    
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

    # Dynamically build the list of features that are actually present in the DataFrame
    # This is the most critical part to ensure consistency
    
    # Define a base set of expected technical features
    expected_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", 
        "RSI_feat", "MACD", "BB_upper", "%K", "%D", "ADX"
    ]
    
    # Filter to only include technical features that are actually in df.columns
    present_technical_features = [col for col in expected_technical_features if col in df.columns]
    
    # Combine with financial features
    all_present_features = present_technical_features + financial_features
    
    # Also include target columns for the initial DataFrame selection before dropna
    target_cols = ["Target", "TargetClassBuy", "TargetClassSell"]
    cols_for_ready = all_present_features + target_cols
    
    # Filter cols_for_ready to ensure all are actually in df.columns (redundant but safe)
    cols_for_ready_final = [col for col in cols_for_ready if col in df.columns]

    ready = df[cols_for_ready_final].dropna()
    
    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in target_cols]

    print(f"   ↳ rows after features available: {len(ready)}")
    return ready, final_training_features

# Scikit-learn imports (fallback for CPU or if cuML not available)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning

# --- Globals for ML library status ---
_ml_libraries_initialized = False
CUDA_AVAILABLE = False
CUML_AVAILABLE = False
LGBMClassifier = None

def initialize_ml_libraries():
    """Initializes ML libraries and prints their status only once."""
    global _ml_libraries_initialized, CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier
    if _ml_libraries_initialized:
        return

    try:
        import torch
        if torch.cuda.is_available():
            CUDA_AVAILABLE = True
            print("✅ CUDA is available. GPU acceleration enabled.")
        else:
            print("⚠️ CUDA is not available. GPU acceleration will not be used.")
    except ImportError:
        print("⚠️ PyTorch not installed. Run: pip install torch. CUDA availability check skipped.")

    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier
        from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
        from cuml.preprocessing import StandardScaler as cuMLStandardScaler
        CUML_AVAILABLE = True
        print("✅ cuML found. GPU-accelerated models will be used if CUDA is available.")
    except ImportError:
        print("⚠️ cuML not installed. Run: pip install cuml. GPU-accelerated models will be skipped.")

    try:
        from lightgbm import LGBMClassifier as lgbm
        LGBMClassifier = lgbm
        if CUDA_AVAILABLE:
            print("ℹ️ LightGBM found. Will attempt to use GPU.")
        else:
            print("ℹ️ LightGBM found. Will use CPU (CUDA not available).")
    except ImportError:
        print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")
    
    _ml_libraries_initialized = True

def train_and_evaluate_models(df: pd.DataFrame, target_col: str = "TargetClassBuy", feature_set: Optional[List[str]] = None, ticker: str = "UNKNOWN"):
    """Train and compare multiple classifiers for a given target, returning the best one."""
    initialize_ml_libraries()
    d = df.copy() # Renamed to d to match the original error context
    
    if target_col not in d.columns:
        print(f"  [DIAGNOSTIC] {ticker}: Target column '{target_col}' not found. Skipping.")
        return None, None

    # The input 'df' (now 'd') is expected to already have all necessary features computed by fetch_training_data.
    # The feature_set parameter should accurately reflect the features present in 'd'.
    
    # Use the provided feature_set directly, as it's already filtered and ready
    if feature_set is None:
        # This fallback should ideally not be hit if train_worker passes feature_set correctly.
        # If it is hit, it means there's a problem in how feature_set is passed.
        # For robustness, we'll try to infer from 'd' if feature_set is None.
        print("⚠️ feature_set was None in train_and_evaluate_models. Inferring features from DataFrame columns.")
        final_feature_names = [col for col in d.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell"]]
        if not final_feature_names:
            print("⚠️ No features found in DataFrame after excluding target columns. Skipping model training.")
            return None, None
    else:
        # Ensure that all features in feature_set are actually present in 'd'
        final_feature_names = [f for f in feature_set if f in d.columns]
        if len(final_feature_names) != len(feature_set):
            missing_features = set(feature_set) - set(final_feature_names)
            print(f"⚠️ Missing features in DataFrame 'd' that were expected in feature_set: {missing_features}. Proceeding with available features.")
        if not final_feature_names:
            print("⚠️ No valid features to train with after filtering. Skipping model training.")
            return None, None

    # Ensure all required columns (features + target) are present in 'd'
    required_cols_for_training = final_feature_names + [target_col]
    if not all(col in d.columns for col in required_cols_for_training):
        missing = [col for col in required_cols_for_training if col not in d.columns]
        print(f"⚠️ Missing critical columns for model comparison (target: {target_col}, missing: {missing}). Skipping.")
        return None, None

    # The DataFrame 'd' already contains the necessary features and target from fetch_training_data.
    # We just need to ensure it has enough rows after any potential NaNs.
    d = d[required_cols_for_training].dropna() # Ensure only relevant columns are kept and NaNs are dropped
    
    if len(d) < 50:  # Increased requirement for cross-validation
        print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after feature prep ({len(d)} rows, need >= 50). Skipping.")
        return None, None # Return None for both model and scaler
    
    X_df = d[final_feature_names]
    y = d[target_col].values

    # --- More robust check for class balance and minimum samples for cross-validation ---
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        print(f"  [DIAGNOSTIC] {ticker}: Not enough class diversity for '{target_col}' (only 1 class found: {unique_classes}). Skipping.")
        return None, None
    
    n_splits = 5 # From StratifiedKFold
    if any(c < n_splits for c in counts):
        print(f"  [DIAGNOSTIC] {ticker}: Least populated class in '{target_col}' has {min(counts)} members (needs >= {n_splits}). Skipping.")
        return None, None

    # Scale features for models that are sensitive to scale (like Logistic Regression and SVM)
    if CUML_AVAILABLE:
        scaler = cuMLStandardScaler()
        # cuML expects cupy arrays or pandas dataframes, convert if X_df is numpy
        X_gpu = cuml.DataFrame(X_df) if not isinstance(X_df, cuml.DataFrame) else X_df
        X_scaled = scaler.fit_transform(X_gpu)
        X = pd.DataFrame(X_scaled.to_numpy(), columns=final_feature_names, index=X_df.index) # Convert back to pandas for GridSearchCV
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_df.index)
    
    # Store feature names for consistent use during prediction
    scaler.feature_names_in_ = list(final_feature_names) 

    # Define models and their parameter grids for GridSearchCV
    models_and_params = {}

    if CUML_AVAILABLE:
        models_and_params["cuML Logistic Regression"] = {
            "model": cuMLLogisticRegression(random_state=SEED, class_weight="balanced", solver='qn'), # cuML LogisticRegression uses 'qn' solver
            "params": {'C': [0.1, 1.0, 10.0]}
        }
        models_and_params["cuML Random Forest"] = {
            "model": cuMLRandomForestClassifier(random_state=SEED, class_weight="balanced"),
            "params": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
        }
        # SVM is not yet in cuML, so we keep the sklearn version as a fallback if cuML is available but SVM is desired.
        models_and_params["SVM"] = {
            "model": SVC(probability=True, random_state=SEED, class_weight="balanced"),
            "params": {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
        }
    else:
        models_and_params["Logistic Regression"] = {
            "model": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
            "params": {'C': [0.1, 1.0, 10.0]}
        }
        models_and_params["Random Forest"] = {
            "model": RandomForestClassifier(random_state=SEED, class_weight="balanced"),
            "params": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
        }
        models_and_params["SVM"] = {
            "model": SVC(probability=True, random_state=SEED, class_weight="balanced"),
            "params": {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
        }

    if LGBMClassifier:
        lgbm_model_params = {
            "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1),
            "params": {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2]}
        }
        if CUDA_AVAILABLE:
            lgbm_model_params["model"].set_params(device='gpu')
            models_and_params["LightGBM (GPU)"] = lgbm_model_params
        else:
            models_and_params["LightGBM (CPU)"] = lgbm_model_params

    results = {}
    best_model_overall = None
    best_auc_overall = -np.inf
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation with GridSearchCV):")
    for name, mp in models_and_params.items():
        model = mp["model"]
        params = mp["params"]
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Use GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
                grid_search.fit(X, y)
                
                best_score = grid_search.best_score_
                results[name] = best_score
                print(f"    - {name}: {best_score:.4f} (Best Params: {grid_search.best_params_})")

                if best_score > best_auc_overall:
                    best_auc_overall = best_score
                    best_model_overall = grid_search.best_estimator_ # Store the best estimator from GridSearchCV

        except Exception as e:
            print(f"    - {name}: Failed evaluation. Error: {e}")
            results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None, None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best model: {best_model_name} with AUC = {best_auc_overall:.4f}")

    # Return the best model found by GridSearchCV and the scaler
    return best_model_overall, scaler

def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, feature_set = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    model_sell_path = models_dir / f"{ticker}_model_sell.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"

    model_buy, model_sell, scaler = None, None, None

    # Load models if they exist and FORCE_OPTIMIZATION is False
    if not FORCE_OPTIMIZATION and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            print(f"  ✅ Loaded existing models for {ticker}.")
            return {
                'ticker': ticker,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler
            }
        except Exception as e:
            print(f"  ⚠️ Error loading models for {ticker}: {e}. Retraining.")

    print(f"  ⚙️ Training models for {ticker}...")
    # The full DataFrame for the training period is already passed in.
    df_train, actual_feature_set = fetch_training_data(ticker, df_train_period, target_percentage)

    if df_train.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data.")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    # Train BUY model
    model_buy, scaler_buy = train_and_evaluate_models(df_train, "TargetClassBuy", actual_feature_set, ticker=ticker)
    # Train SELL model (using the same scaler as buy for consistency, or a separate one if needed)
    model_sell, scaler_sell = train_and_evaluate_models(df_train, "TargetClassSell", actual_feature_set, ticker=ticker)

    # For simplicity, we'll use the scaler from the buy model for both if they are different.
    # In a more complex scenario, you might want to ensure feature_set consistency or use separate scalers.
    final_scaler = scaler_buy if scaler_buy else scaler_sell

    if model_buy and model_sell and final_scaler:
        try:
            joblib.dump(model_buy, model_buy_path)
            joblib.dump(model_sell, model_sell_path)
            joblib.dump(final_scaler, scaler_path)
            print(f"  ✅ Models and scaler saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving models for {ticker}: {e}")
            
        return {
            'ticker': ticker,
            'model_buy': model_buy,
            'model_sell': model_sell,
            'scaler': final_scaler,
            'status': 'trained',
            'reason': None
        }
    else:
        reason = "Insufficient training data" # Default reason
        if df_train.empty:
            reason = f"Insufficient training data (initial rows: {len(df_train_period)})"
        elif len(df_train) < 50:
            reason = f"Not enough rows after feature prep ({len(df_train)} rows, need >= 50)"
        
        print(f"  ❌ Failed to train models for {ticker}. Reason: {reason}")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'status': 'failed', 'reason': reason}

# ============================
# Rule-based backtester (ATR & ML gate)
# ============================

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Optional ML gate to allow buys."""
    def __init__(self, df: pd.DataFrame, ticker: str, initial_balance: float, transaction_cost: float, # Added ticker parameter
                 model_buy=None, model_sell=None, scaler=None, min_proba_buy: float = MIN_PROBA_BUY, min_proba_sell: float = MIN_PROBA_SELL, use_gate: bool = USE_MODEL_GATE,
                 market_data: Optional[pd.DataFrame] = None, use_market_filter: bool = USE_MARKET_FILTER, feature_set: Optional[List[str]] = None,
                 per_ticker_min_proba_buy: Optional[float] = None, per_ticker_min_proba_sell: Optional[float] = None):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.ticker = ticker # Assign ticker directly
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
        self.feature_set = feature_set if feature_set is not None else ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper", "%K", "%D", "ADX",
                                                                        'Fin_Revenue', 'Fin_NetIncome', 'Fin_TotalAssets', 'Fin_TotalLiabilities', 'Fin_FreeCashFlow', 'Fin_EBITDA']
        
        self.reset()
        self._prepare_data()

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
        # self.ticker is already set in __init__, no need to re-set here.
        self.last_ai_action: str = "HOLD" # New: Track last AI action
        self.last_buy_prob: float = 0.0
        self.last_sell_prob: float = 0.0
        
    def _prepare_data(self):
        # --- Data Cleaning ---
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if "Volume" not in self.df.columns: self.df["Volume"] = 0
        if "High" not in self.df.columns: self.df["High"] = self.df["Close"]
        if "Low" not in self.df.columns: self.df["Low"] = self.df["Close"]
        if "Open" not in self.df.columns: self.df["Open"] = self.df["Close"]
            
        self.df = self.df.dropna(subset=["Close"])
        if self.df.empty: return # Exit if no data left after cleaning
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.ffill().bfill()
        
        close = self.df["Close"]
        high = self.df["High"] if "High" in self.df.columns else None
        low  = self.df["Low"]  if "Low" in self.df.columns else None

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

        # Set current_step to the first index where ATR is not NaN
        first_valid_atr_idx = self.df['ATR'].first_valid_index()
        if first_valid_atr_idx is not None:
            self.current_step = self.df.index.get_loc(first_valid_atr_idx)
        else:
            pass # Removed warning print

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

        # Stochastic Oscillator
        low_14, high_14 = self.df['Low'].rolling(window=14).min(), self.df['High'].rolling(window=14).max()
        self.df['%K'] = (self.df['Close'] - low_14) / (high_14 - low_14) * 100
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()

        # Average Directional Index (ADX)
        self.df['up_move'] = self.df['High'] - self.df['High'].shift(1)
        self.df['down_move'] = self.df['Low'].shift(1) - self.df['Low']
        self.df['+DM'] = np.where((self.df['up_move'] > self.df['down_move']) & (self.df['up_move'] > 0), self.df['up_move'], 0)
        self.df['-DM'] = np.where((self.df['down_move'] > self.df['up_move']) & (self.df['down_move'] > 0), self.df['down_move'], 0)
        high_low_diff = self.df['High'] - self.df['Low']
        high_prev_close_diff_abs = (self.df['High'] - self.df['Close'].shift(1)).abs()
        low_prev_close_diff_abs = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        self.df['TR'] = pd.concat([high_low_diff, high_prev_close_diff_abs, low_prev_close_diff_abs], axis=1).max(axis=1)
        alpha = 1/14
        self.df['+DM14'] = self.df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['-DM14'] = self.df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['TR14'] = self.df['TR'].ewm(alpha=alpha, adjust=False).mean()
        self.df['DX'] = (abs(self.df['+DM14'] - self.df['-DM14']) / (self.df['+DM14'] + self.df['-DM14'])) * 100
        self.df['ADX'] = self.df['DX'].ewm(alpha=alpha, adjust=False).mean()
        # Fill NaNs for all ADX-related indicators after their calculations
        self.df['+DM'] = self.df['+DM'].fillna(0)
        self.df['-DM'] = self.df['-DM'].fillna(0) # Corrected: Removed tuple index
        self.df['TR'] = self.df['TR'].fillna(0)
        self.df['+DM14'] = self.df['+DM14'].fillna(0)
        self.df['-DM14'] = self.df['-DM14'].fillna(0)
        self.df['TR14'] = self.df['TR14'].fillna(0)
        self.df['DX'] = self.df['DX'].fillna(0)
        self.df['ADX'] = self.df['ADX'].fillna(0)
        # Fill NaNs for Stochastic Oscillator after its calculations
        self.df['%K'] = self.df['%K'].fillna(0)
        self.df['%D'] = self.df['%D'].fillna(0)

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
        self.last_buy_prob = self._get_model_prediction(i, self.model_buy)
        return self.last_buy_prob >= self.min_proba_buy

    def _allow_sell_by_model(self, i: int) -> bool: # Removed feature_set from signature
        self.last_sell_prob = self._get_model_prediction(i, self.model_sell)
        return self.last_sell_prob >= self.min_proba_sell

    def _position_size_from_atr(self, price: float, atr: float) -> int:
        if atr is None or np.isnan(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollars = min(self.initial_balance * RISK_PER_TRADE, 10000)  # Cap risk at $10,000
        per_share_risk = ATR_MULT_TRAIL * atr
        if per_share_risk <= 0:
            return 0
        qty = int(risk_dollars / per_share_risk)
        return max(qty, 0)

    def _buy(self, price: float, atr: Optional[float], date: str):
        if self.cash <= 0:
            return

        # 1. Determine initial quantity based on risk model
        qty = self._position_size_from_atr(price, atr if atr is not None else np.nan)
        if qty <= 0:
            return

        # 2. Adjust quantity based on backtester's available cash
        cost = price * qty * (1 + self.transaction_cost)
        if cost > self.cash:
            qty = int(self.cash / (price * (1 + self.transaction_cost)))
        
        if qty <= 0:
            return

        # 4. Finalize cost and fee, then update the backtester's state
        fee = price * qty * self.transaction_cost
        cost = price * qty + fee

        self.cash -= cost
        self.shares += qty
        self.entry_price = price
        self.entry_atr = atr if atr is not None and not np.isnan(atr) else None
        self.highest_since_entry = price
        self.holding_bars = 0
        self.trade_log.append((date, "BUY", price, qty, self.ticker, {"fee": fee}, fee))
        self.last_ai_action = "BUY"
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
        # Condition: AI model must approve
        ai_signal = self._allow_buy_by_model(self.current_step)

        if self.shares == 0 and ai_signal:
            self._buy(price, atr, date)
        
        # --- AI-driven Exit Signal ---
        elif self.shares > 0 and self._allow_sell_by_model(self.current_step): # Changed to elif to prioritize buy
            self._sell(price, date)
        else:
            self.last_ai_action = "HOLD"

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple], str, float, float]: # Added str for last_ai_action
        if self.df.empty:
            return self.initial_balance, [], "N/A", np.nan, np.nan
        done = False
        while not done:
            done = self.step()
        if self.shares > 0 and not self.df.empty:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log, self.last_ai_action, self.last_buy_prob, self.last_sell_prob # Return last_ai_action

# ============================
# Analytics
# ============================

def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, \
        market_data, feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data = params
    
    # Initial log to confirm the worker has started for a ticker
    with open("logs/worker_debug.log", "a") as f:
        f.write(f"Worker started for ticker: {ticker}\n")

    if df_backtest.empty:
        print(f"  ⚠️ Skipping backtest for {ticker}: DataFrame is empty.")
        return None
        
    try:
        env = RuleTradingEnv(
            df=df_backtest.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model_buy=model_buy,
            model_sell=model_sell,
            scaler=scaler,
            use_gate=USE_MODEL_GATE,
            market_data=market_data,
            use_market_filter=USE_MARKET_FILTER,
            feature_set=feature_set,
            per_ticker_min_proba_buy=min_proba_buy,
            per_ticker_min_proba_sell=min_proba_sell
        )
        final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob = env.run()

        # Calculate individual Buy & Hold for the same period
        start_price_bh = float(df_backtest["Close"].iloc[0])
        end_price_bh = float(df_backtest["Close"].iloc[-1])
        individual_bh_return = ((end_price_bh - start_price_bh) / start_price_bh) * 100 if start_price_bh > 0 else 0.0
        
        # Analyze performance for this ticker
        perf_data = analyze_performance(trade_log, env.portfolio_history, df_backtest["Close"].tolist(), ticker)

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob
        }
    finally:
        # This block will execute whether an exception occurred or not.
        with open("logs/worker_debug.log", "a") as f:
            final_val_to_log = 'Error' if 'final_val' not in locals() else final_val
            f.write(f"Worker finished for ticker: {ticker}. Final Value: {final_val_to_log}\n")

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
    strat_returns = pd.Series(strategy_history).pct_change(fill_method=None).dropna()
    bh_returns = pd.Series(buy_hold_history).pct_change(fill_method=None).dropna()

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

def _calculate_performance_worker(params: Tuple[str, pd.DataFrame]) -> Optional[Tuple[str, float, pd.DataFrame]]:
    """Worker to calculate 1Y performance from a given DataFrame."""
    ticker, df_1y = params
    try:
        # The data is already passed in, no need to fetch.
        # The dataframe might have NaN rows for dates where the ticker didn't trade.
        df_1y = df_1y.dropna(subset=['Close'])
        if not df_1y.empty and len(df_1y) > 200:  # Some basic data quality check
            start_price = df_1y['Close'].iloc[0]
            end_price = df_1y['Close'].iloc[-1]
            if start_price > 0.01:  # Ensure start_price is a meaningful positive number
                perf_1y = ((end_price - start_price) / start_price) * 100
                if np.isfinite(perf_1y):  # Check for nan or inf
                    return (ticker, perf_1y, df_1y)
    except Exception as e:
        # Suppress verbose errors for single ticker failures in a large batch
        # print(f"  ℹ️ Could not process {ticker} for performance calculation: {e}")
        pass
    return None


def find_top_performers(
    all_available_tickers: List[str],
    all_tickers_data: pd.DataFrame,
    return_tickers: bool = False,
    n_top: int = N_TOP_TICKERS,
    fcf_min_threshold: float = 0.0,
    ebitda_min_threshold: float = 0.0
):
    """
    Screens pre-fetched data for the top N performers and returns a list of (ticker, performance) tuples.
    """
    if all_tickers_data.empty:
        print("❌ No ticker data provided to find_top_performers. Exiting.")
        return []

    end_date = all_tickers_data.index.max()
    start_date = end_date - timedelta(days=365)
    ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    # --- Step 1: Calculate Benchmark Performances (if enabled) ---
    final_benchmark_perf = -np.inf
    ytd_benchmark_perf = -np.inf
    if USE_PERFORMANCE_BENCHMARK:
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
        
        if not benchmark_perfs:
            print("❌ Could not calculate any benchmark performance. Cannot proceed.")
            return []
            
        final_benchmark_perf = max(benchmark_perfs.values())
        print(f"  📈 Using final 1-Year performance benchmark of {final_benchmark_perf:.2f}%")

        print("- Calculating YTD Performance Benchmarks...")
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
        
        if not ytd_benchmark_perfs:
            print("❌ Could not calculate any YTD benchmark performance. Cannot proceed.")
            return []
        ytd_benchmark_perf = max(ytd_benchmark_perfs.values())
        print(f"  📈 Using YTD performance benchmark of {ytd_benchmark_perf:.2f}%")
    else:
        print("ℹ️ Performance benchmark is disabled. All tickers will be considered.")

    # --- Step 2: Calculate 1-Year Performance from pre-fetched data ---
    print("🔍 Calculating 1-Year performance from pre-fetched data...")
    # Slice the main DataFrame for the 1-year performance calculation period
    all_data = all_tickers_data.loc[start_date:end_date]
    print("...performance calculation complete.")

    all_tickers_performance_with_df = []

    # Create a list of parameters for the worker function
    params = []
    # yfinance download with multiple tickers returns a multi-index column.
    # Tickers that failed to download will be missing from the columns.
    valid_tickers = all_data.columns.get_level_values(1).unique()

    for ticker in valid_tickers:
        try:
            # Select columns for the current ticker
            ticker_data = all_data.loc[:, (slice(None), ticker)]
            # Drop the ticker level from the column index
            ticker_data.columns = ticker_data.columns.droplevel(1)
            # The data is already adjusted, so we can use 'Close'
            params.append((ticker, ticker_data.copy()))
        except KeyError:
            # This handles cases where a ticker might be in the list but not in the downloaded data
            pass

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(_calculate_performance_worker, params), total=len(params), desc="Calculating 1Y Performance"))
        for res in results:
            if res:
                all_tickers_performance_with_df.append(res)

    if not all_tickers_performance_with_df:
        print("❌ No tickers with valid 1-Year performance found. Aborting.")
        return []

    # Sort all tickers by 1-Year performance in descending order
    sorted_all_tickers_performance_with_df = sorted(all_tickers_performance_with_df, key=lambda item: item[1], reverse=True)
    
    # Filter out extreme outliers (e.g., >1000% gain)
    sorted_all_tickers_performance_with_df = [item for item in sorted_all_tickers_performance_with_df if item[1] < 1000]

    # Apply n_top limit AFTER sorting
    if n_top > 0:
        final_performers_for_selection = sorted_all_tickers_performance_with_df[:n_top]
        print(f"\n✅ Selected top {len(final_performers_for_selection)} tickers based on 1-Year performance.")
    else:
        final_performers_for_selection = sorted_all_tickers_performance_with_df
        print(f"\n✅ Analyzing all {len(final_performers_for_selection)} tickers (N_TOP_TICKERS is {n_top}).")

    # --- Step 3: Apply Performance Benchmarks (if enabled) and YTD performance in parallel ---
    print(f"🔍 Applying performance benchmarks and fetching YTD for selected tickers in parallel...")
    
    finalize_params = [
        (ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, USE_PERFORMANCE_BENCHMARK)
        for ticker, perf_1y, df_1y in final_performers_for_selection
    ]
    performance_data = []

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(_finalize_single_ticker_performance, finalize_params), total=len(finalize_params), desc="Finalizing Top Performers"))
        for res in results:
            if res: # Only add if the ticker passed the benchmarks or if benchmarks are disabled
                performance_data.append(res)

    if USE_PERFORMANCE_BENCHMARK:
        print(f"\n✅ Found {len(performance_data)} stocks that passed the performance benchmarks.")
    else:
        print(f"\n✅ Found {len(performance_data)} stocks for analysis (performance benchmark disabled).")
        
    if not performance_data:
        return []

    final_performers = performance_data # This now contains (ticker, perf_1y, perf_ytd) tuples

    # --- Step 3: Fundamental Screen (Optional Free Cash Flow & EBITDA for the last fiscal year) ---
    if fcf_min_threshold is not None or ebitda_min_threshold is not None:
        print(f"  🔍 Screening {len(final_performers)} strong performers for fundamental metrics in parallel...")
        
        fundamental_screen_params = [
            (ticker, perf_1y, perf_ytd, fcf_min_threshold, ebitda_min_threshold)
            for ticker, perf_1y, perf_ytd in final_performers
        ]
        screened_performers = []

        with Pool(processes=NUM_PROCESSES) as pool:
            results = list(tqdm(pool.imap(_apply_fundamental_screen_worker, fundamental_screen_params), total=len(fundamental_screen_params), desc="Applying fundamental screens"))
            for res in results:
                if res: # Only add if the ticker passed the fundamental screen
                    screened_performers.append(res)

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

def _apply_fundamental_screen_worker(params: Tuple[str, float, float, float, float]) -> Optional[Tuple[str, float, float]]:
    """Worker function to perform fundamental screening for a single ticker."""
    ticker, perf_1y, perf_ytd, fcf_min_threshold, ebitda_min_threshold = params
    
    try:
        time.sleep(PAUSE_BETWEEN_YF_CALLS)
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
                for key in latest_financials.index:
                    if key in latest_financials.index:
                        ebitda = latest_financials[key]
                        break
                if ebitda is not None and ebitda <= ebitda_min_threshold:
                    ebitda_ok = False

        if fcf_ok and ebitda_ok:
            return (ticker, perf_1y, perf_ytd)
        else:
            return None # Does not pass fundamental screen

    except Exception as e:
        print(f"  ⚠️ Error fetching financials for {ticker}: {e}. Skipping fundamental screen.")
        return (ticker, perf_1y, perf_ytd) # Let it pass if there's an error fetching financials

def _finalize_single_ticker_performance(params: Tuple[str, float, pd.DataFrame, datetime, datetime, float, float, bool]) -> Optional[Tuple[str, float, float]]:
    """Worker function to derive YTD data from 1Y DataFrame, apply benchmarks, and perform fundamental screening for a single ticker."""
    ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, use_performance_benchmark = params
    
    perf_ytd = np.nan
    try:
        # Derive YTD data from the already fetched df_1y
        df_ytd = df_1y.loc[(df_1y.index >= _to_utc(ytd_start_date)) & (df_1y.index <= _to_utc(end_date))].copy()
        
        if not df_ytd.empty:
            start_price = df_ytd['Close'].iloc[0]
            end_price = df_ytd['Close'].iloc[-1]
            if start_price > 0:
                perf_ytd = ((end_price - start_price) / start_price) * 100
            else:
                perf_ytd = np.nan
        
        # Apply benchmark filter if enabled
        if use_performance_benchmark:
            if perf_1y > final_benchmark_perf and perf_ytd > ytd_benchmark_perf:
                return (ticker, perf_1y, perf_ytd)
            else:
                return None # Does not pass benchmark
        else:
            # If benchmark is disabled, just return the ticker with its 1Y and YTD performance
            return (ticker, perf_1y, perf_ytd)
    except Exception as e:
        print(f"  ⚠️ Error deriving YTD or applying benchmark for {ticker}: {e}. Skipping.")
        return None

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
    """
    Optimizes ML thresholds (min_proba_buy, min_proba_sell, target_percentage)
    for each ticker in the portfolio based on Sharpe Ratio.
    """
    print("\n🔬 Optimizing ML thresholds for each ticker...")
    optimized_params_per_ticker = {}
    
    optimization_params = []
    for ticker in top_tickers:
        if ticker not in models_buy or ticker not in models_sell or ticker not in scalers:
            print(f"  ⚠️ Skipping optimization for {ticker}: Models or scaler not available.")
            continue
        
        # Pass necessary data for optimization
        optimization_params.append((
            ticker, train_start, train_end, target_percentage, feature_set,
            models_buy[ticker], models_sell[ticker], scalers[ticker],
            market_data, capital_per_stock
        ))

    if run_parallel:
        print(f"  Running optimization in parallel for {len(optimization_params)} tickers using {NUM_PROCESSES} processes...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results = list(tqdm(pool.imap(optimize_single_ticker_worker, optimization_params), total=len(optimization_params), desc="Optimizing Thresholds"))
    else:
        print(f"  Running optimization sequentially for {len(optimization_params)} tickers...")
        results = [optimize_single_ticker_worker(p) for p in tqdm(optimization_params, desc="Optimizing Thresholds")]

    for res in results:
        if res and res['ticker']:
            optimized_params_per_ticker[res['ticker']] = {
                'min_proba_buy': res['min_proba_buy'],
                'min_proba_sell': res['min_proba_sell'],
                'target_percentage': res['target_percentage']
            }
            print(f"  ✅ {res['ticker']} optimized: Buy={res['min_proba_buy']:.2f}, Sell={res['min_proba_sell']:.2f}, Target%={res['target_percentage']:.2%}")
    
    return optimized_params_per_ticker

def optimize_single_ticker_worker(params: Tuple) -> Dict:
    """Worker function to optimize thresholds for a single ticker."""
    ticker, train_start, train_end, default_target_percentage, feature_set, \
        model_buy, model_sell, scaler, market_data, capital_per_stock = params

    best_sharpe = -np.inf
    best_min_proba_buy = MIN_PROBA_BUY
    best_min_proba_sell = MIN_PROBA_SELL
    best_target_percentage = default_target_percentage

    # Define ranges for optimization
    min_proba_buy_range = np.arange(0.2, 0.9, 0.1) # Example range
    min_proba_sell_range = np.arange(0.2, 0.9, 0.1) # Example range

    # Load data for backtesting during optimization
    df_backtest_opt = load_prices(ticker, train_start, train_end)
    if df_backtest_opt.empty:
        return {'ticker': ticker, 'min_proba_buy': MIN_PROBA_BUY, 'min_proba_sell': MIN_PROBA_SELL, 'target_percentage': default_target_percentage}

    for p_buy in min_proba_buy_range:
        for p_sell in min_proba_sell_range:
            env = RuleTradingEnv(
                df=df_backtest_opt.copy(),
                ticker=ticker,
                initial_balance=capital_per_stock,
                transaction_cost=TRANSACTION_COST,
                model_buy=model_buy,
                model_sell=model_sell,
                scaler=scaler,
                per_ticker_min_proba_buy=p_buy,
                per_ticker_min_proba_sell=p_sell,
                use_gate=USE_MODEL_GATE,
                market_data=market_data,
                use_market_filter=USE_MARKET_FILTER,
                feature_set=feature_set
            )
            final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob = env.run()
            
            # Calculate Sharpe Ratio for this combination
            strategy_history = env.portfolio_history
            if len(strategy_history) > 1:
                strat_returns = pd.Series(strategy_history).pct_change(fill_method=None).dropna()
                if strat_returns.std() > 0:
                    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252)
                else:
                    sharpe = 0.0 # Avoid division by zero
            else:
                sharpe = 0.0

            # --- Diagnostic Logging ---
            # This will show the Sharpe ratio for each combination of thresholds tested.
            # print(f"  [Opti] {ticker}: Buy={p_buy:.2f}, Sell={p_sell:.2f} -> Sharpe={sharpe:.4f}")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_min_proba_buy = p_buy
                best_min_proba_sell = p_sell
    
    # The target_percentage is not optimized here because the models are pre-trained.
    # We return the default_target_percentage that the models were trained with.
    best_target_percentage = default_target_percentage
    
    return {
        'ticker': ticker,
        'min_proba_buy': best_min_proba_buy,
        'min_proba_sell': best_min_proba_sell,
        'target_percentage': best_target_percentage,
        'best_sharpe': best_sharpe
    }

def _run_portfolio_backtest(
    all_tickers_data: pd.DataFrame,
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
    period_name: str,
    top_performers_data: List[Tuple] # Added top_performers_data
) -> Tuple[float, List[float], List[str], List[Dict]]:
    """Helper function to run portfolio backtest for a given period."""
    num_processes = NUM_PROCESSES

    backtest_params = []
    for ticker in top_tickers:
        min_proba_buy_ticker = MIN_PROBA_BUY
        min_proba_sell_ticker = MIN_PROBA_SELL
        target_percentage_ticker = target_percentage # Default to global if not optimized
        
        if optimized_params_per_ticker and ticker in optimized_params_per_ticker:
            min_proba_buy_ticker = optimized_params_per_ticker[ticker]['min_proba_buy']
            min_proba_sell_ticker = optimized_params_per_ticker[ticker]['min_proba_sell']
            target_percentage_ticker = optimized_params_per_ticker[ticker]['target_percentage']

        # Ensure feature_set is passed to backtest_worker
        feature_set_for_worker = scalers.get(ticker).feature_names_in_ if scalers.get(ticker) and hasattr(scalers.get(ticker), 'feature_names_in_') else None
        
        # Slice the main DataFrame for the backtest period for this specific ticker
        try:
            ticker_backtest_data = all_tickers_data.loc[start_date:end_date, (slice(None), ticker)]
            ticker_backtest_data.columns = ticker_backtest_data.columns.droplevel(1)
            if ticker_backtest_data.empty:
                print(f"  ⚠️ Sliced backtest data for {ticker} for period {period_name} is empty. Skipping.")
                continue
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice backtest data for {ticker} for period {period_name}. Skipping.")
            continue

        backtest_params.append((
            ticker, ticker_backtest_data.copy(), capital_per_stock,
            models_buy.get(ticker), models_sell.get(ticker), scalers.get(ticker),
            market_data, feature_set_for_worker, min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data # Pass top_performers_data
        ))

    portfolio_values = []
    processed_tickers = []
    performance_metrics = []
    
    total_tickers_to_process = len(top_tickers)
    processed_count = 0

    if run_parallel:
        print(f"📈 Running {period_name} backtest in parallel for {total_tickers_to_process} tickers using {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            results = []
            for res in tqdm(pool.imap(backtest_worker, backtest_params), total=total_tickers_to_process, desc=f"Backtesting {period_name}"):
                if res:
                    print(f"  [DEBUG] Ticker: {res['ticker']}, Final Value: {res['final_val']}")
                    portfolio_values.append(res['final_val'])
                    processed_tickers.append(res['ticker'])
                    performance_metrics.append(res)
                    
                    # Find the corresponding performance data (1Y and YTD from top_performers_data)
                    perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                    for t, p1y, pytd in top_performers_data:
                        if t == res['ticker']:
                            perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                            perf_ytd_benchmark = pytd if np.isfinite(pytd) else np.nan
                            break
                    
                    # Print individual stock performance immediately
                    print(f"\n📈 Individual Stock Performance for {res['ticker']} ({period_name}):")
                    print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                    print(f"  - YTD Performance: {perf_ytd_benchmark:.2f}%" if pd.notna(perf_ytd_benchmark) else "  - YTD Performance: N/A")
                    print(f"  - AI Sharpe Ratio: {res['perf_data']['sharpe_ratio']:.2f}")
                    print(f"  - Last AI Action: {res['last_ai_action']}")
                    print(f"  - Optimized Buy Threshold: {optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY):.2f}")
                    print(f"  - Optimized Sell Threshold: {optimized_params_per_ticker.get(res['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL):.2f}")
                    print(f"  - Optimized Target Percentage: {optimized_params_per_ticker.get(res['ticker'], {}).get('target_percentage', TARGET_PERCENTAGE):.2%}")
                    print("-" * 40)
                processed_count += 1
    else:
        print(f"📈 Running {period_name} backtest sequentially for {total_tickers_to_process} tickers...")
        results = []
        for res in tqdm(backtest_params, desc=f"Backtesting {period_name}"):
            worker_result = backtest_worker(res)
            if worker_result:
                print(f"  [DEBUG] Ticker: {worker_result['ticker']}, Final Value: {worker_result['final_val']}")
                portfolio_values.append(worker_result['final_val'])
                processed_tickers.append(worker_result['ticker'])
                performance_metrics.append(worker_result)
                
                # Find the corresponding performance data (1Y and YTD from top_performers_data)
                perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                for t, p1y, pytd in top_performers_data:
                    if t == worker_result['ticker']:
                        perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                        perf_ytd_benchmark = pytd if np.isfinite(pytd) else np.nan
                        break
                
                # Print individual stock performance immediately
                print(f"\n📈 Individual Stock Performance for {worker_result['ticker']} ({period_name}):")
                print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                print(f"  - YTD Performance: {perf_ytd_benchmark:.2f}%" if pd.notna(perf_ytd_benchmark) else "  - YTD Performance: N/A")
                print(f"  - AI Sharpe Ratio: {worker_result['perf_data']['sharpe_ratio']:.2f}")
                print(f"  - Last AI Action: {worker_result['last_ai_action']}")
                print(f"  - Optimized Buy Threshold: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_buy', MIN_PROBA_BUY):.2f}")
                print(f"  - Optimized Sell Threshold: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('min_proba_sell', MIN_PROBA_SELL):.2f}")
                print(f"  - Optimized Target Percentage: {optimized_params_per_ticker.get(worker_result['ticker'], {}).get('target_percentage', TARGET_PERCENTAGE):.2%}")
                print("-" * 40)
            processed_count += 1

    # Filter out any None values from portfolio_values before summing
    valid_portfolio_values = [v for v in portfolio_values if v is not None and np.isfinite(v)]
    
    final_portfolio_value = sum(valid_portfolio_values) + (total_tickers_to_process - len(processed_tickers)) * capital_per_stock
    print(f"✅ {period_name} Backtest complete. Final portfolio value: ${final_portfolio_value:,.2f}\n")
    return final_portfolio_value, portfolio_values, processed_tickers, performance_metrics

def print_final_summary(
    sorted_final_results: List[Dict],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Dict[str, Dict[str, float]],
    final_strategy_value_1y: float,
    final_buy_hold_value_1y: float,
    ai_1y_return: float,
    final_strategy_value_ytd: float,
    final_buy_hold_value_ytd: float,
    ai_ytd_return: float,
    final_strategy_value_3month: float,
    final_buy_hold_value_3month: float,
    ai_3month_return: float,
    initial_balance_used: float, # Added parameter
    num_tickers_analyzed: int
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}") # Use the passed initial_balance_used
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Buy & Hold Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  YTD Strategy Value: ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)")
    print(f"  YTD Buy & Hold Value: ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  3-Month Strategy Value: ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)")
    print(f"  3-Month Buy & Hold Value: ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (Sorted by 1-Year Performance):")
    print("-" * 160)
    print(f"{'Ticker':<10} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10}")
    print("-" * 160)
    for res in sorted_final_results:
        # --- Safely get ticker and parameters ---
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)

        # --- Safely format performance numbers ---
        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        
        print(f"{ticker:<10} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%}")
    print("-" * 160)

    print("\n🤖 ML Model Status:")
    for ticker in sorted_final_results:
        t = ticker['ticker']
        buy_model_status = "✅ Trained" if models_buy.get(t) else "❌ Not Trained"
        sell_model_status = "✅ Trained" if models_sell.get(t) else "❌ Not Trained"
        print(f"  - {t}: Buy Model: {buy_model_status}, Sell Model: {sell_model_status}")
    print("="*80)

    print("\n💡 Next Steps:")
    print("  - Review individual ticker performance and trade logs for deeper insights.")
    print("  - Experiment with different `MARKET_SELECTION` options and `N_TOP_TICKERS`.")
    print("  - Adjust `TARGET_PERCENTAGE` and `RISK_PER_TRADE` for different risk appetites.")
    print("  - Consider enabling `USE_MARKET_FILTER` and `USE_PERFORMANCE_BENCHMARK` for additional filtering.")
    print("  - Explore advanced ML models or feature engineering for further improvements.")
    print("="*80)


# ============================
# Main
# ============================

def main(
    fcf_threshold: float = 0.0,
    ebitda_threshold: float = 0.0,
    min_proba_buy: float = MIN_PROBA_BUY,
    min_proba_sell: float = MIN_PROBA_SELL,
    target_percentage: float = TARGET_PERCENTAGE, # This will now be the initial/default target_percentage for optimization
    top_performers_data=None,
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True,
    single_ticker: Optional[str] = None,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None,
    force_optimization: bool = FORCE_OPTIMIZATION # Add force_optimization parameter
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[Dict]]:
    
    end_date = datetime.now(timezone.utc)
    bt_end = end_date
    
    alpaca_trading_client = None
    current_initial_balance = INITIAL_BALANCE
    print(f"Using initial balance: ${current_initial_balance:,.2f}")

    # --- Handle single ticker case for initial performance calculation ---
    if single_ticker:
        print(f"🔍 Running analysis for single ticker: {single_ticker}")
        start_date_1y = end_date - timedelta(days=365)
        ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)
        
        df_1y = load_prices_robust(single_ticker, start_date_1y, end_date)
        perf_1y = np.nan
        if df_1y is not None and not df_1y.empty:
            start_price = df_1y['Close'].iloc[0]
            end_price = df_1y['Close'].iloc[-1]
            if start_price > 0:
                perf_1y = ((end_price - start_price) / start_price) * 100
            else:
                perf_1y = np.nan

        df_ytd = load_prices_robust(single_ticker, ytd_start_date, end_date)
        perf_ytd = np.nan
        if df_ytd is not None and not df_ytd.empty:
            start_price = df_ytd['Close'].iloc[0]
            end_price = df_ytd['Close'].iloc[-1]
            if start_price > 0:
                perf_ytd = ((end_price - start_price) / start_price) * 100
            else:
                perf_ytd = np.nan
        
        top_performers_data = [(single_ticker, perf_1y, perf_ytd)]
    
    # --- Step 1: Get all tickers and perform a single, comprehensive data download ---
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        print("❌ No tickers found from market selection. Aborting.")
        return (None,) * 15

    # Determine the absolute earliest date needed for any calculation
    train_start_1y = end_date - timedelta(days=BACKTEST_DAYS + TRAIN_LOOKBACK_DAYS + 1)
    earliest_date_needed = train_start_1y

    print(f"🚀 Step 1: Batch downloading data for {len(all_available_tickers)} tickers from {earliest_date_needed.date()} to {end_date.date()}...")
    
    all_tickers_data_list = []
    for i in range(0, len(all_available_tickers), BATCH_DOWNLOAD_SIZE):
        batch = all_available_tickers[i:i + BATCH_DOWNLOAD_SIZE]
        print(f"  - Downloading batch {i//BATCH_DOWNLOAD_SIZE + 1}/{(len(all_available_tickers) + BATCH_DOWNLOAD_SIZE - 1)//BATCH_DOWNLOAD_SIZE} ({len(batch)} tickers)...")
        batch_data = _download_batch_robust(batch, start=earliest_date_needed, end=end_date)
        if not batch_data.empty:
            all_tickers_data_list.append(batch_data)
        if i + BATCH_DOWNLOAD_SIZE < len(all_available_tickers):
            print(f"  - Pausing for {PAUSE_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    if not all_tickers_data_list:
        print("❌ Comprehensive batch download failed. Aborting.")
        return (None,) * 15

    all_tickers_data = pd.concat(all_tickers_data_list, axis=1)

    if all_tickers_data.empty:
        print("❌ Comprehensive batch download failed. Aborting.")
        return (None,) * 15
    
    # Ensure index is timezone-aware
    if all_tickers_data.index.tzinfo is None:
        all_tickers_data.index = all_tickers_data.index.tz_localize('UTC')
    else:
        all_tickers_data.index = all_tickers_data.index.tz_convert('UTC')
    print("✅ Comprehensive data download complete.")

    # --- Identify top performers if not provided ---
    if top_performers_data is None:
        title = "🚀 AI-Powered Momentum & Trend Strategy"
        # ... (rest of the title and filter logic remains the same)
        print(title + "\n" + "="*50 + "\n")

        print("🔍 Step 2: Identifying stocks outperforming market benchmarks...")
        
        # Get top performers from market selection using the pre-fetched data
        market_selected_performers = find_top_performers(
            all_available_tickers=all_available_tickers,
            all_tickers_data=all_tickers_data,
            return_tickers=True,
            n_top=N_TOP_TICKERS,
            fcf_min_threshold=fcf_threshold,
            ebitda_min_threshold=ebitda_threshold
        )
        

        top_performers_data = market_selected_performers
                
    if not top_performers_data:
        print("❌ Could not identify top tickers. Aborting backtest.")
        return (None,) * 15
    
    top_tickers = [ticker for ticker, _, _ in top_performers_data]
    print(f"\n✅ Identified {len(top_tickers)} stocks for backtesting: {', '.join(top_tickers)}\n")

    # Log skipped tickers
    skipped_tickers = set(all_available_tickers) - set(top_tickers)
    if skipped_tickers:
        with open("logs/skipped_tickers.log", "w") as f:
            f.write("Tickers skipped during performance analysis:\n")
            for ticker in sorted(list(skipped_tickers)):
                f.write(f"{ticker}\n")

    # --- Training Models (for 1-Year Backtest) ---
    print("🔍 Step 3: Training AI models for 1-Year backtest...")
    bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)
    train_end_1y = bt_start_1y - timedelta(days=1)
    train_start_1y_calc = train_end_1y - timedelta(days=TRAIN_LOOKBACK_DAYS)
    
    training_params_1y = []
    for ticker in top_tickers:
        try:
            # Slice the main DataFrame for the training period
            ticker_train_data = all_tickers_data.loc[train_start_1y_calc:train_end_1y, (slice(None), ticker)]
            ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
            training_params_1y.append((ticker, ticker_train_data.copy(), target_percentage, feature_set))
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice training data for {ticker} for 1-Year period. Skipping.")
            continue
    models_buy, models_sell, scalers = {}, {}, {}
    failed_training_tickers_1y = {} # New: Store failed tickers and their reasons

    if run_parallel:
        print(f"🤖 Training 1-Year models in parallel for {len(top_tickers)} tickers using {NUM_PROCESSES} processes...")
        with Pool(processes=NUM_PROCESSES) as pool:
            training_results_1y = list(tqdm(pool.imap(train_worker, training_params_1y), total=len(training_params_1y), desc="Training 1-Year Models"))
    else:
        print(f"🤖 Training 1-Year models sequentially for {len(top_tickers)} tickers...")
        training_results_1y = [train_worker(p) for p in tqdm(training_params_1y, desc="Training 1-Year Models")]

    for res in training_results_1y:
        if res and res.get('status') == 'trained':
            models_buy[res['ticker']] = res['model_buy']
            models_sell[res['ticker']] = res['model_sell']
            scalers[res['ticker']] = res['scaler']
        elif res and res.get('status') == 'failed':
            failed_training_tickers_1y[res['ticker']] = res['reason']

    if not models_buy and USE_MODEL_GATE:
        print("⚠️ No models were trained for 1-Year backtest. Model-gating will be disabled for this run.\n")
    
    # Filter out failed tickers from top_tickers for subsequent steps
    top_tickers_1y_filtered = [t for t in top_tickers if t not in failed_training_tickers_1y]
    print(f"  ℹ️ {len(failed_training_tickers_1y)} tickers failed 1-Year model training and will be skipped: {', '.join(failed_training_tickers_1y.keys())}")
    
    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1y_filtered = [item for item in top_performers_data if item[0] in top_tickers_1y_filtered]
    
    # Update capital_per_stock based on filtered tickers
    capital_per_stock_1y = current_initial_balance / max(len(top_tickers_1y_filtered), 1)
    
    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_1y_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1y_filtered}
    else:
        optimized_params_per_ticker_1y_filtered = {}
    
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

    # capital_per_stock = current_initial_balance / max(len(top_tickers), 1) # Original line, now replaced by capital_per_stock_1y
    
    # --- OPTIMIZE THRESHOLDS ---
    # Ensure logs directory exists for optimized parameters
    _ensure_dir(TOP_CACHE_PATH.parent)
    optimized_params_file = TOP_CACHE_PATH.parent / "optimized_per_ticker_params.json"
    
    # If force_optimization is True and the file exists, delete it to force re-optimization
    if force_optimization and optimized_params_file.exists():
        try:
            os.remove(optimized_params_file)
            print(f"🗑️ Deleted existing optimized parameters file: {optimized_params_file} to force re-optimization.")
        except Exception as e:
            print(f"⚠️ Could not delete optimized parameters file: {e}")

    optimized_params_per_ticker = None
    if force_optimization or not optimized_params_file.exists():
        print("\n🔄 Step 2.5: Optimizing ML thresholds for each ticker (forced or no existing file)...")
        optimized_params_per_ticker = optimize_thresholds_for_portfolio(
            top_tickers=top_tickers_1y_filtered, # Use filtered tickers for optimization
            train_start=train_start_1y, # Use training data for optimization
            train_end=train_end_1y,
            target_percentage=target_percentage,
            feature_set=feature_set,
            models_buy=models_buy,
            models_sell=models_sell,
            scalers=scalers,
            market_data=market_data,
            capital_per_stock=capital_per_stock_1y, # Use filtered capital per stock
            run_parallel=run_parallel
        )
        if optimized_params_per_ticker:
            try:
                with open(optimized_params_file, 'w') as f:
                    json.dump(optimized_params_per_ticker, f, indent=4)
                print(f"✅ Optimized parameters saved to {optimized_params_file}")
            except Exception as e:
                print(f"⚠️ Could not save optimized parameters to file: {e}")
    else:
        try:
            with open(optimized_params_file, 'r') as f:
                loaded_optimized_params = json.load(f)
                # Filter loaded params to only include successfully trained tickers
                optimized_params_per_ticker = {k: v for k, v in loaded_optimized_params.items() if k in top_tickers_1y_filtered}
            print(f"\n✅ Loaded optimized parameters from {optimized_params_file} (set 'force_optimization=True' in main() call to re-run)")
        except Exception as e:
            print(f"⚠️ Could not load optimized parameters from file: {e}. Re-running optimization.")
            print(f"\n🔄 Step 2.5: Optimizing ML thresholds for each ticker (re-running due to load error)...")
            optimized_params_per_ticker = optimize_thresholds_for_portfolio(
                top_tickers=top_tickers_1y_filtered, # Use filtered tickers for optimization
                train_start=train_start_1y, # Use training data for optimization
                train_end=train_end_1y,
                target_percentage=target_percentage,
                feature_set=feature_set,
                models_buy=models_buy,
                models_sell=models_sell,
                scalers=scalers,
                market_data=market_data,
                capital_per_stock=capital_per_stock_1y, # Use filtered capital per stock
                run_parallel=run_parallel
            )
            if optimized_params_per_ticker:
                try:
                    with open(optimized_params_file, 'w') as f:
                        json.dump(optimized_params_per_ticker, f, indent=4)
                    print(f"✅ Optimized parameters saved to {optimized_params_file}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized parameters to file: {e}")

    # --- Run 1-Year Backtest ---
    print("\n🔍 Step 4: Running 1-Year Backtest...")
    final_strategy_value_1y, strategy_results_1y, processed_tickers_1y, performance_metrics_1y = _run_portfolio_backtest(
        all_tickers_data=all_tickers_data,
        start_date=bt_start_1y,
        end_date=bt_end,
        top_tickers=top_tickers_1y_filtered, # Use filtered tickers for backtest
        models_buy=models_buy,
        models_sell=models_sell,
        scalers=scalers,
        market_data=market_data,
        optimized_params_per_ticker=optimized_params_per_ticker,
        capital_per_stock=capital_per_stock_1y, # Use filtered capital per stock
        # Pass the global target_percentage here, as the individual backtest_worker will use the optimized one
        target_percentage=target_percentage, 
        run_parallel=run_parallel,
        period_name="1-Year",
        top_performers_data=top_performers_data_1y_filtered # Pass filtered top_performers_data
    )
    ai_1y_return = ((final_strategy_value_1y - current_initial_balance) / abs(current_initial_balance)) * 100 if current_initial_balance != 0 else 0

    # --- Calculate Buy & Hold for 1-Year ---
    print("\n📊 Calculating Buy & Hold performance for 1-Year period...")
    buy_hold_results_1y = []
    for ticker in processed_tickers_1y:
        df_bh = load_prices_robust(ticker, bt_start_1y, bt_end)
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock_1y / start_price) if start_price > 0 else 0 # Use filtered capital per stock
            cash_bh = capital_per_stock_1y - shares_bh * start_price # Use filtered capital per stock
            buy_hold_results_1y.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
        else:
            buy_hold_results_1y.append(capital_per_stock_1y) # If no data, assume initial capital
    final_buy_hold_value_1y = sum(buy_hold_results_1y) + (len(top_tickers_1y_filtered) - len(processed_tickers_1y)) * capital_per_stock_1y
    print("✅ 1-Year Buy & Hold calculation complete.")


    # --- Training Models (for YTD Backtest) ---
    print("\n🔍 Step 5: Training AI models for YTD backtest...")
    ytd_start_date = datetime(bt_end.year, 1, 1, tzinfo=timezone.utc)
    train_end_ytd = ytd_start_date - timedelta(days=1)
    train_start_ytd = train_end_ytd - timedelta(days=TRAIN_LOOKBACK_DAYS)
    
    training_params_ytd = []
    for ticker in top_tickers_1y_filtered: # Use filtered tickers for YTD training
        try:
            # Slice the main DataFrame for the training period
            ticker_train_data = all_tickers_data.loc[train_start_ytd:train_end_ytd, (slice(None), ticker)]
            ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
            training_params_ytd.append((ticker, ticker_train_data.copy(), target_percentage, feature_set))
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice training data for {ticker} for YTD period. Skipping.")
            continue
    models_buy_ytd, models_sell_ytd, scalers_ytd = {}, {}, {}
    failed_training_tickers_ytd = {} # New: Store failed tickers and their reasons

    if run_parallel:
        print(f"🤖 Training YTD models in parallel for {len(top_tickers_1y_filtered)} tickers using {NUM_PROCESSES} processes...")
        with Pool(processes=NUM_PROCESSES) as pool:
            training_results_ytd = list(tqdm(pool.imap(train_worker, training_params_ytd), total=len(training_params_ytd), desc="Training YTD Models"))
    else:
        print(f"🤖 Training YTD models sequentially for {len(top_tickers_1y_filtered)} tickers...")
        training_results_ytd = [train_worker(p) for p in tqdm(training_params_ytd, desc="Training YTD Models")]

    for res in training_results_ytd:
        if res and res.get('status') == 'trained':
            models_buy_ytd[res['ticker']] = res['model_buy']
            models_sell_ytd[res['ticker']] = res['model_sell']
            scalers_ytd[res['ticker']] = res['scaler']
        elif res and res.get('status') == 'failed':
            failed_training_tickers_ytd[res['ticker']] = res['reason']

    if not models_buy_ytd and USE_MODEL_GATE:
        print("⚠️ No models were trained for YTD backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_1y_filtered for subsequent steps
    top_tickers_ytd_filtered = [t for t in top_tickers_1y_filtered if t not in failed_training_tickers_ytd]
    print(f"  ℹ️ {len(failed_training_tickers_ytd)} tickers failed YTD model training and will be skipped: {', '.join(failed_training_tickers_ytd.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_ytd_filtered = [item for item in top_performers_data_1y_filtered if item[0] in top_tickers_ytd_filtered]

    # Update capital_per_stock based on filtered tickers
    capital_per_stock_ytd = current_initial_balance / max(len(top_tickers_ytd_filtered), 1)

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_ytd_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_ytd_filtered}
    else:
        optimized_params_per_ticker_ytd_filtered = {}

    # --- Run YTD Backtest ---
    print("\n🔍 Step 6: Running YTD Backtest...")
    final_strategy_value_ytd, strategy_results_ytd, processed_tickers_ytd_local, performance_metrics_ytd = _run_portfolio_backtest(
        all_tickers_data=all_tickers_data,
        start_date=ytd_start_date,
        end_date=bt_end,
        top_tickers=top_tickers_ytd_filtered, # Use filtered tickers for backtest
        models_buy=models_buy_ytd,
        models_sell=models_sell_ytd,
        scalers=scalers_ytd,
        market_data=market_data, # Use the same market data as 1-year backtest
        optimized_params_per_ticker=optimized_params_per_ticker_ytd_filtered,
        capital_per_stock=capital_per_stock_ytd, # Use filtered capital per stock
        target_percentage=target_percentage,
        run_parallel=run_parallel,
        period_name="YTD",
        top_performers_data=top_performers_data_ytd_filtered # Pass filtered top_performers_data
    )
    ai_ytd_return = ((final_strategy_value_ytd - current_initial_balance) / abs(current_initial_balance)) * 100 if current_initial_balance != 0 else 0

    # --- Calculate Buy & Hold for YTD ---
    print("\n📊 Calculating Buy & Hold performance for YTD period...")
    buy_hold_results_ytd = []
    for ticker in processed_tickers_ytd_local: # Use processed_tickers_ytd_local here
        df_bh = load_prices_robust(ticker, ytd_start_date, bt_end)
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock_ytd / start_price) if start_price > 0 else 0 # Use filtered capital per stock
            cash_bh = capital_per_stock_ytd - shares_bh * start_price # Use filtered capital per stock
            buy_hold_results_ytd.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
        else:
            buy_hold_results_ytd.append(capital_per_stock_ytd) # If no data, assume initial capital
    final_buy_hold_value_ytd = sum(buy_hold_results_ytd) + (len(top_tickers_ytd_filtered) - len(processed_tickers_ytd_local)) * capital_per_stock_ytd
    print("✅ YTD Buy & Hold calculation complete.")

    # --- Training Models (for 3-Month Backtest) ---
    print("\n🔍 Step 7: Training AI models for 3-Month backtest...")
    bt_start_3month = bt_end - timedelta(days=BACKTEST_DAYS_3MONTH)
    train_end_3month = bt_start_3month - timedelta(days=1)
    train_start_3month = train_end_3month - timedelta(days=TRAIN_LOOKBACK_DAYS)

    training_params_3month = []
    for ticker in top_tickers_ytd_filtered: # Use filtered tickers for 3-Month training
        try:
            # Slice the main DataFrame for the training period
            ticker_train_data = all_tickers_data.loc[train_start_3month:train_end_3month, (slice(None), ticker)]
            ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
            training_params_3month.append((ticker, ticker_train_data.copy(), target_percentage, feature_set))
        except (KeyError, IndexError):
            print(f"  ⚠️ Could not slice training data for {ticker} for 3-Month period. Skipping.")
            continue
    models_buy_3month, models_sell_3month, scalers_3month = {}, {}, {}
    failed_training_tickers_3month = {} # New: Store failed tickers and their reasons

    if run_parallel:
        print(f"🤖 Training 3-Month models in parallel for {len(top_tickers_ytd_filtered)} tickers using {NUM_PROCESSES} processes...")
        with Pool(processes=NUM_PROCESSES) as pool:
            training_results_3month = list(tqdm(pool.imap(train_worker, training_params_3month), total=len(training_params_3month), desc="Training 3-Month Models"))
    else:
        print(f"🤖 Training 3-Month models sequentially for {len(top_tickers_ytd_filtered)} tickers...")
        training_results_3month = [train_worker(p) for p in tqdm(training_params_3month, desc="Training 3-Month Models")]

    for res in training_results_3month:
        if res and res.get('status') == 'trained':
            models_buy_3month[res['ticker']] = res['model_buy']
            models_sell_3month[res['ticker']] = res['model_sell']
            scalers_3month[res['ticker']] = res['scaler']
        elif res and res.get('status') == 'failed':
            failed_training_tickers_3month[res['ticker']] = res['reason']

    if not models_buy_3month and USE_MODEL_GATE:
        print("⚠️ No models were trained for 3-Month backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_ytd_filtered for subsequent steps
    top_tickers_3month_filtered = [t for t in top_tickers_ytd_filtered if t not in failed_training_tickers_3month]
    print(f"  ℹ️ {len(failed_training_tickers_3month)} tickers failed 3-Month model training and will be skipped: {', '.join(failed_training_tickers_3month.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_3month_filtered = [item for item in top_performers_data_ytd_filtered if item[0] in top_tickers_3month_filtered]

    # Update capital_per_stock based on filtered tickers
    capital_per_stock_3month = current_initial_balance / max(len(top_tickers_3month_filtered), 1)

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_3month_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_3month_filtered}
    else:
        optimized_params_per_ticker_3month_filtered = {}

    # --- Run 3-Month Backtest ---
    print("\n🔍 Step 8: Running 3-Month Backtest...")
    final_strategy_value_3month, strategy_results_3month, processed_tickers_3month_local, performance_metrics_3month = _run_portfolio_backtest(
        all_tickers_data=all_tickers_data,
        start_date=bt_start_3month,
        end_date=bt_end,
        top_tickers=top_tickers_3month_filtered, # Use filtered tickers for backtest
        models_buy=models_buy_3month,
        models_sell=models_sell_3month,
        scalers=scalers_3month,
        market_data=market_data,
        optimized_params_per_ticker=optimized_params_per_ticker_3month_filtered,
        capital_per_stock=capital_per_stock_3month, # Use filtered capital per stock
        target_percentage=target_percentage,
        run_parallel=run_parallel,
        period_name="3-Month",
        top_performers_data=top_performers_data_3month_filtered # Pass filtered top_performers_data
    )
    ai_3month_return = ((final_strategy_value_3month - current_initial_balance) / abs(current_initial_balance)) * 100 if current_initial_balance != 0 else 0

    # --- Calculate Buy & Hold for 3-Month ---
    print("\n📊 Calculating Buy & Hold performance for 3-Month period...")
    buy_hold_results_3month = []
    for ticker in processed_tickers_3month_local:
        df_bh = load_prices_robust(ticker, bt_start_3month, bt_end)
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock_3month / start_price) if start_price > 0 else 0 # Use filtered capital per stock
            cash_bh = capital_per_stock_3month - shares_bh * start_price # Use filtered capital per stock
            buy_hold_results_3month.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
        else:
            buy_hold_results_3month.append(capital_per_stock_3month)
    final_buy_hold_value_3month = sum(buy_hold_results_3month) + (len(top_tickers_3month_filtered) - len(processed_tickers_3month_local)) * capital_per_stock_3month
    print("✅ 3-Month Buy & Hold calculation complete.")

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    print("\n📝 Preparing final summary data...")
    final_results = []
    
    # Combine all failed tickers from all periods
    all_failed_tickers = {}
    all_failed_tickers.update(failed_training_tickers_1y)
    all_failed_tickers.update(failed_training_tickers_ytd)
    all_failed_tickers.update(failed_training_tickers_3month)

    # Add successfully processed tickers
    for i, ticker in enumerate(processed_tickers_1y):
        backtest_result_for_ticker = next((res for res in performance_metrics_1y if res['ticker'] == ticker), None)
        
        if backtest_result_for_ticker:
            perf_data = backtest_result_for_ticker['perf_data']
            individual_bh_return = backtest_result_for_ticker['individual_bh_return']
            last_ai_action = backtest_result_for_ticker['last_ai_action']
            buy_prob = backtest_result_for_ticker['buy_prob']
            sell_prob = backtest_result_for_ticker['sell_prob']
        else:
            perf_data = {'sharpe_ratio': 0.0}
            individual_bh_return = 0.0
            last_ai_action = "N/A"
            buy_prob = 0.0
            sell_prob = 0.0

        perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                perf_ytd_benchmark = pytd if np.isfinite(pytd) else np.nan
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': strategy_results_1y[i],
            'sharpe': perf_data['sharpe_ratio'],
            'one_year_perf': perf_1y_benchmark,
            'ytd_perf': perf_ytd_benchmark,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'status': 'trained',
            'reason': None
        })
    
    # Add failed tickers to the final results
    for ticker, reason in all_failed_tickers.items():
        final_results.append({
            'ticker': ticker,
            'performance': current_initial_balance / max(len(top_tickers), 1), # Assign initial capital for failed tickers
            'sharpe': np.nan,
            'one_year_perf': np.nan,
            'ytd_perf': np.nan,
            'individual_bh_return': np.nan,
            'last_ai_action': "FAILED",
            'buy_prob': np.nan,
            'sell_prob': np.nan,
            'status': 'failed',
            'reason': reason
        })

    # Sort by 1Y performance for the final table, handling potential NaN values
    sorted_final_results = sorted(final_results, key=lambda x: x.get('one_year_perf', -np.inf) if pd.notna(x.get('one_year_perf')) else -np.inf, reverse=True)
    
    print_final_summary(sorted_final_results, models_buy, models_sell, scalers, optimized_params_per_ticker,
                        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
                        final_strategy_value_ytd, final_buy_hold_value_ytd, ai_ytd_return,
                        final_strategy_value_3month, final_buy_hold_value_3month, ai_3month_return,
                        initial_balance_used=current_initial_balance,
                        num_tickers_analyzed=len(top_tickers))
    print("\n✅ Final summary prepared and printed.")

    # --- Save recommendations to file ---
    recommendations_path = Path("logs/recommendations.json")
    with open(recommendations_path, 'w') as f:
        json.dump(sorted_final_results, f, indent=4)
    print(f"\n✅ Recommendations saved to {recommendations_path}")

    
    return final_strategy_value_1y, final_buy_hold_value_1y, models_buy, models_sell, scalers, top_performers_data, strategy_results_1y, processed_tickers_1y, performance_metrics_1y, ai_1y_return, ai_ytd_return, final_strategy_value_3month, final_buy_hold_value_3month, ai_3month_return, optimized_params_per_ticker

if __name__ == "__main__":
    # Run main.py with optimization disabled for faster subsequent runs
    main(
        fcf_threshold=0.0, ebitda_threshold=0.0, run_parallel=True, single_ticker=None, force_optimization=False, top_performers_data=None
    )
