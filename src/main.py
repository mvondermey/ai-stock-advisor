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
print("DEBUG: Script execution initiated.") # Moved for debugging silent exits

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

# TwelveData SDK client
try:
    from twelvedata import TDClient
    TWELVEDATA_SDK_AVAILABLE = True
except ImportError:
    print("⚠️ TwelveData SDK client not found. TwelveData data provider will be skipped.")
    TWELVEDATA_SDK_AVAILABLE = False
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
    "ALPACA_STOCKS": False, # Fetch all tradable US equities from Alpaca
    "NASDAQ_ALL": False,
    "NASDAQ_100": True,
    "SP500": False,
    "DOW_JONES": False,
    "POPULAR_ETFS": False,
    "CRYPTO": False,
    "DAX": False,
    "MDAX": False,
    "SMI": False,
    "FTSE_MIB": False,
}
N_TOP_TICKERS           = 2        # Number of top performers to select (0 to disable limit)
BATCH_DOWNLOAD_SIZE     = 20000       # Reduced batch size for stability
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
NUM_PROCESSES           = max(1, cpu_count() - 5) # Use all but one CPU core for parallel processing

# --- Backtest & training windows
BACKTEST_DAYS           = 365        # 1 year for backtest
BACKTEST_DAYS_3MONTH    = 90         # 3 months for backtest
BACKTEST_DAYS_1MONTH    = 30         # 1 month for backtest
TRAIN_LOOKBACK_DAYS     = 360        # more data for model (e.g., 1 year)

# --- Backtest Period Enable/Disable Flags ---
ENABLE_1YEAR_BACKTEST   = True
ENABLE_YTD_BACKTEST     = True
ENABLE_3MONTH_BACKTEST  = True
ENABLE_1MONTH_BACKTEST  = True

# --- Training Period Enable/Disable Flags ---
ENABLE_1YEAR_TRAINING   = True
ENABLE_YTD_TRAINING     = True
ENABLE_3MONTH_TRAINING  = True
ENABLE_1MONTH_TRAINING  = True

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 10
STRAT_SMA_LONG          = 50
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 2.0
ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing
INVESTMENT_PER_STOCK    = 15000.0    # Fixed amount to invest per stock
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_BUY           = 0.2       # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.2       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.01       # 1% target for buy/sell classification
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # market filter removed as per user request
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering

# --- ML Model Selection Flags ---
USE_LOGISTIC_REGRESSION = False
USE_SVM                 = False
USE_MLP_CLASSIFIER      = False
USE_LIGHTGBM            = False # Enable LightGBM - GOOD
#GOOD
USE_XGBOOST             = False # Disable XGBoost
USE_LSTM                = False
#Not so GOOD
USE_GRU                 = True # Enable GRU - BEST
#BEST
USE_RANDOM_FOREST       = False # Disable RandomForest
#WORST

# --- Simple Rule-Based Strategy specific hyperparameters
USE_SIMPLE_RULE_STRATEGY = False
SIMPLE_RULE_TRAILING_STOP_PERCENT = 0.10 # 10% trailing stop
SIMPLE_RULE_TAKE_PROFIT_PERCENT = 0.10   # 10% take profit

# --- Deep Learning specific hyperparameters
SEQUENCE_LENGTH         = 20         # Number of past days to consider for LSTM/GRU
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.2
LSTM_EPOCHS             = 50
LSTM_BATCH_SIZE         = 64
LSTM_LEARNING_RATE      = 0.001

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = False
FORCE_TRAINING          = True      # Set to True to force re-training of ML models
FORCE_THRESHOLDS_OPTIMIZATION = False # Set to True to force re-optimization of ML thresholds
FORCE_PERCENTAGE_OPTIMIZATION = False # Set to True to force re-optimization of TARGET_PERCENTAGE

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
    """Fetch OHLCV from TwelveData using the SDK."""
    if not TWELVEDATA_SDK_AVAILABLE or not TWELVEDATA_API_KEY:
        return pd.DataFrame()

    try:
        tdc = TDClient(apikey=TWELVEDATA_API_KEY)
        
        # Construct the time series API call
        ts = tdc.time_series(
            symbol=ticker,
            interval="1day",
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
            outputsize=5000 # Max outputsize for historical data
        ).as_pandas() # Get data as pandas DataFrame

        if ts.empty:
            print(f"  ℹ️ No data found for {ticker} from TwelveData SDK.")
            return pd.DataFrame()

        df = ts.copy()
        
        # TwelveData returns 'datetime' as index, ensure it's UTC and named 'Date'
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "Date"
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        
        # Ensure all relevant columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_index()
    except Exception as e:
        print(f"  ⚠️ An error occurred while fetching data from TwelveData SDK for {ticker}: {e}")
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
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "batch download failed" in error_str:
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
        df_financial[col] = pd.to_numeric(df[col], errors='coerce')

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
            print(f"✅ Fetched {len(dow_tickers)} tickers from Dow Jones. ")
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
            for table in tables_mib:
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
    print(f"  [DIAGNOSTIC] {ticker}: fetch_training_data - Initial data rows: {len(data)}")
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
    # ATR (Average True Range)
    high = df["High"] if "High" in df.columns else None
    low  = df["Low"]  if "Low" in df.columns else None
    prev_close = df["Close"].shift(1)
    if high is not None and low is not None:
        hl = (high - low).abs()
        h_pc = (high - prev_close).abs()
        l_pc = (low  - prev_close).abs()
        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    else:
        # Fallback for ATR if High/Low are not available (though they should be after load_prices)
        ret = df["Close"].pct_change(fill_method=None)
        df["ATR"] = (ret.rolling(ATR_PERIOD).std() * df["Close"]).rolling(2).mean()
    df["ATR"] = df["ATR"].fillna(0) # Fill any NaNs from initial ATR calculation

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
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    denominator_k = (high_14 - low_14)
    df['%K'] = np.where(denominator_k != 0, (df['Close'] - low_14) / denominator_k * 100, 0)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%D'].fillna(0)

    # Average Directional Index (ADX)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['+DM'] = df['+DM'].fillna(0)

    # Calculate True Range (TR)
    high_low_diff = df['High'] - df['Low']
    high_prev_close_diff_abs = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close_diff_abs = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([high_low_diff, high_prev_close_diff_abs, low_prev_close_diff_abs], axis=1).max(axis=1)
    df['TR'] = df['TR'].fillna(0)

    # Calculate Smoothed DM and TR
    alpha = 1/14
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM14'].fillna(0)
    df['-DM14'] = df['-DM14'].fillna(0)
    df['TR14'] = df['TR14'].fillna(0)

    # Calculate Directional Index (DX)
    denominator_dx = (df['+DM14'] + df['-DM14'])
    df['DX'] = np.where(denominator_dx != 0, (abs(df['+DM14'] - df['-DM14']) / denominator_dx) * 100, 0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean() # Missing line added
    df['ADX'] = df['ADX'].fillna(0)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['CMF'] = df['CMF'].fillna(0)

    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # Keltner Channels
    df['KC_TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['KC_ATR'] = df['KC_TR'].rolling(window=10).mean()
    df['KC_Middle'] = df['Close'].rolling(window=20).mean()
    df['KC_Upper'] = df['KC_Middle'] + (df['KC_ATR'] * 2)
    df['KC_Lower'] = df['KC_Middle'] - (df['KC_ATR'] * 2)

    # Donchian Channels
    df['DC_Upper'] = df['High'].rolling(window=20).max()
    df['DC_Lower'] = df['Low'].rolling(window=20).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2

    # Parabolic SAR (PSAR)
    # Initialize PSAR
    psar = df['Close'].copy()
    af = 0.02 # Acceleration Factor
    max_af = 0.2 # Maximum Acceleration Factor

    # Initial trend and extreme point
    # Assume initial uptrend if Close > Open, downtrend otherwise
    uptrend = True if df['Close'].iloc[0] > df['Open'].iloc[0] else False
    ep = df['High'].iloc[0] if uptrend else df['Low'].iloc[0]
    sar = df['Low'].iloc[0] if uptrend else df['High'].iloc[0]
    
    # Iterate to calculate PSAR
    for i in range(1, len(df)):
        if uptrend:
            sar = sar + af * (ep - sar)
            if df['Low'].iloc[i] < sar: # Trend reversal
                uptrend = False
                sar = ep
                ep = df['Low'].iloc[i]
                af = 0.02
            else:
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(max_af, af + 0.02)
        else: # Downtrend
            sar = sar + af * (ep - sar)
            if df['High'].iloc[i] > sar: # Trend reversal
                uptrend = True
                sar = ep
                ep = df['High'].iloc[i]
                af = 0.02
            else:
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(max_af, af + 0.02)
            psar.iloc[i] = sar
    df['PSAR'] = psar

    # Accumulation/Distribution Line (ADL)
    # Money Flow Volume (MFV)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['ADL'] = mf_volume.cumsum()
    df['ADL'] = df['ADL'].fillna(0) # Fill initial NaNs with 0

    # Commodity Channel Index (CCI)
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
    df['CCI'] = df['CCI'].fillna(0) # Fill initial NaNs with 0

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=FEAT_VOL_WINDOW).sum() / df['Volume'].rolling(window=FEAT_VOL_WINDOW).sum()
    df['VWAP'] = df['VWAP'].fillna(df['Close']) # Fill initial NaNs with Close price

    # ATR Percentage
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['ATR_Pct'] = df['ATR_Pct'].fillna(0)

    # Chaikin Oscillator
    adl_fast = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    adl_slow = adl_fast.ewm(span=10, adjust=False).mean()
    adl_fast = adl_fast.ewm(span=3, adjust=False).mean()
    df['Chaikin_Oscillator'] = adl_fast - adl_slow
    df['Chaikin_Oscillator'] = df['Chaikin_Oscillator'].fillna(0)

    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfi_ratio = positive_mf.rolling(window=14).sum() / negative_mf.rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['MFI'] = df['MFI'].fillna(0)

    # OBV Moving Average
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()
    df['OBV_SMA'] = df['OBV_SMA'].fillna(0)

    # Historical Volatility (e.g., 20-day rolling standard deviation of log returns)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Historical_Volatility'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252) # Annualized
    df['Historical_Volatility'] = df['Historical_Volatility'].fillna(0)

    # Market Momentum (using SPY) - Requires fetching SPY data
    # This feature will be handled differently, as it requires external data.
    # For now, we'll add a placeholder and assume it's handled in the main loop or a separate function.
    # For the purpose of feature generation, we'll assume it's a column that will be merged later.
    # df['Market_Momentum_SPY'] = 0 # Placeholder

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
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
        "Market_Momentum_SPY" # Added new feature
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

    print(f"   ↳ {ticker}: rows after features available: {len(ready)}")
    return ready, final_training_features

# Scikit-learn imports (fallback for CPU or if cuML not available)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier # Added for Neural Network model
from sklearn.preprocessing import MinMaxScaler # Added for scaling data for neural networks

# Added for XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ xgboost not installed. Run: pip install xgboost. It will be skipped.")
    XGBOOST_AVAILABLE = False

# Added for PyTorch and LSTM/GRU models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not installed. Run: pip install torch. Deep learning models will be skipped.")
    PYTORCH_AVAILABLE = False

# --- Globals for ML library status ---
_ml_libraries_initialized = False
CUDA_AVAILABLE = False
CUML_AVAILABLE = False
LGBMClassifier = None
XGBClassifier = None # Added for XGBoost
cuMLRandomForestClassifier = None
cuMLLogisticRegression = None
cuMLStandardScaler = None
models_and_params: Dict = {} # Declare as global and initialize

# Define LSTM/GRU model architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Get output from the last time step
        out = self.sigmoid(out)
        return out

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # Get output from the last time step
        out = self.sigmoid(out)
        return out

def initialize_ml_libraries():
    """Initializes ML libraries and prints their status only once."""
    global _ml_libraries_initialized, CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier, XGBClassifier, models_and_params, \
