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
_script_initialized = False
if not _script_initialized:
    print("DEBUG: Script execution initiated.")
    _script_initialized = True

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

# Import ML model related functions and classes from ml_models.py
from ml_models import initialize_ml_libraries, LSTMClassifier, GRUClassifier, PYTORCH_AVAILABLE, CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier, XGBClassifier, models_and_params, cuMLRandomForestClassifier, cuMLLogisticRegression, cuMLStandardScaler, SHAP_AVAILABLE

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
import random
random.seed(SEED)

# --- Provider & caching
DATA_PROVIDER           = 'alpaca'    # 'stooq', 'yahoo', 'alpaca', or 'twelvedata'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
VALID_TICKERS_CACHE_PATH = Path("logs/valid_tickers.json")
CACHE_DAYS              = 7

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")

# TwelveData API credentials
TWELVEDATA_API_KEY      = "aed912386d7c47939ebc28a86a96a021"

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
BACKTEST_DAYS_1MONTH    = 32         # 1 month for backtest
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
MIN_PROBA_BUY           = 0.20      # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.20       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.008       # 0.8% target for buy/sell classification
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # market filter removed as per user request
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering

# --- ML Model Selection Flags ---
USE_LOGISTIC_REGRESSION = False
USE_SVM                 = False
USE_MLP_CLASSIFIER      = False
USE_LIGHTGBM            = True # Enable LightGBM - GOOD
#GOOD
USE_XGBOOST             = True # Enable XGBoost
USE_LSTM                = False
#Not so GOOD
USE_GRU                 = True # Enable GRU - BEST
#BEST
USE_RANDOM_FOREST       = False # Enable RandomForest
#WORST

# --- Simple Rule-Based Strategy specific hyperparameters
USE_SIMPLE_RULE_STRATEGY = False
SIMPLE_RULE_TRAILING_STOP_PERCENT = 0.10 # 10% trailing stop
SIMPLE_RULE_TAKE_PROFIT_PERCENT = 0.10   # 10% take profit

# --- Deep Learning specific hyperparameters
SEQUENCE_LENGTH         = 32         # Number of past days to consider for LSTM/GRU
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.2
LSTM_EPOCHS             = 50
LSTM_BATCH_SIZE         = 64
LSTM_LEARNING_RATE      = 0.001

# --- GRU Hyperparameter Search Ranges ---
GRU_HIDDEN_SIZE_OPTIONS = [16, 32, 64, 128, 256]
GRU_NUM_LAYERS_OPTIONS  = [1, 2, 3, 4]
GRU_DROPOUT_OPTIONS     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
GRU_LEARNING_RATE_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01]
GRU_BATCH_SIZE_OPTIONS  = [16, 32, 64, 128, 256]
GRU_EPOCHS_OPTIONS      = [10, 30, 50, 70, 100]
GRU_CLASS_HORIZON_OPTIONS = [1, 2, 3, 4, 5, 7, 10, 15, 20] # New: Options for class_horizon
GRU_TARGET_PERCENTAGE_OPTIONS = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05] # New: Options for target_percentage
ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = True # Set to True to enable GRU hyperparameter search

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = True
FORCE_TRAINING          = True      # Set to True to force re-training of ML models
CONTINUE_TRAINING_FROM_EXISTING = False # Set to True to load existing models and continue training
FORCE_THRESHOLDS_OPTIMIZATION = True # Set to True to force re-optimization of ML thresholds
FORCE_PERCENTAGE_OPTIMIZATION = True # Set to True to force re-optimization of TARGET_PERCENTAGE


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

def _fetch_from_twelvedata(ticker: str, start: datetime, end: datetime, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV from TwelveData using the SDK."""
    key_to_use = api_key if api_key else TWELVEDATA_API_KEY
    if not TWELVEDATA_SDK_AVAILABLE or not key_to_use:
        return pd.DataFrame()

    try:
        tdc = TDClient(apikey=key_to_use)
        
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
        df_financial[col] = pd.to_numeric(df_financial[col], errors='coerce') # Corrected to df_financial[col]

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
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")

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
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")
        
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
                    else:
                        print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                except Exception as e:
                    print(f"  ❌ Yahoo Finance fallback (after Stooq) failed for {ticker}: {e}")
        
        if price_df.empty: # If previous provider failed or was yahoo
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    price_df = downloaded_df.dropna()
                else:
                    print(f"  ⚠️ Final Yahoo download attempt returned empty data for {ticker}.")
            except Exception as e:
                print(f"  ❌ Final Yahoo download attempt failed for {ticker}: {e}")
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

    # --- Add placeholder for sentiment data (for demonstration) ---
    # In a real scenario, this would be fetched from a sentiment API
    if 'Sentiment_Score' not in final_df.columns:
        final_df['Sentiment_Score'] = np.random.uniform(-1, 1, len(final_df)) # Placeholder: random sentiment
        final_df['Sentiment_Score'] = final_df['Sentiment_Score'].rolling(window=5).mean().fillna(0) # Smooth it a bit

    # Return the specifically requested slice
    return final_df.loc[(final_df.index >= _to_utc(start)) & (final_df.index <= _to_utc(end))].copy()

def _fetch_intermarket_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetches intermarket data (e.g., bond yields, commodities, currencies)."""
    intermarket_tickers = {
        '^VIX': 'VIX_Index',  # CBOE Volatility Index
        'DX-Y.NYB': 'DXY_Index', # U.S. Dollar Index
        'GC=F': 'Gold_Futures', # Gold Futures
        'CL=F': 'Oil_Futures',  # Crude Oil Futures
        '^TNX': 'US10Y_Yield',  # 10-Year Treasury Yield
        'USO': 'Oil_Price',    # United States Oil Fund ETF
        'GLD': 'Gold_Price',   # SPDR Gold Shares ETF
    }
    
    all_intermarket_dfs = []
    for ticker, name in intermarket_tickers.items():
        try:
            # Use load_prices_robust to respect DATA_PROVIDER and handle caching/fallbacks
            df = load_prices_robust(ticker, start, end)
            if not df.empty:
                # load_prices_robust returns a DataFrame with 'Close' column
                single_ticker_df = df[['Close']].rename(columns={'Close': name})
                all_intermarket_dfs.append(single_ticker_df)
        except Exception as e:
            print(f"  ⚠️ Could not fetch intermarket data for {ticker} ({name}): {e}")
            
    if not all_intermarket_dfs:
        return pd.DataFrame()

    # Concatenate all individual DataFrames
    intermarket_df = pd.concat(all_intermarket_dfs, axis=1)
    intermarket_df.index = pd.to_datetime(intermarket_df.index, utc=True)
    intermarket_df.index.name = "Date"
    
    # Calculate returns for intermarket features
    for col in intermarket_df.columns:
        intermarket_df[f"{col}_Returns"] = intermarket_df[col].pct_change(fill_method=None).fillna(0)
        
    return intermarket_df.ffill().bfill().fillna(0)

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

def fetch_training_data(ticker: str, data: pd.DataFrame, target_percentage: float = 0.05, class_horizon: int = CLASS_HORIZON) -> Tuple[pd.DataFrame, List[str]]:
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
    
    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []
    
    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []

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
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    df['ROC_60'] = df['Close'].pct_change(periods=60) * 100

    # Chande Momentum Oscillator (CMO)
    # CMO = (Sum(Up Moves) - Sum(Down Moves)) / (Sum(Up Moves) + Sum(Down Moves)) * 100
    cmo_period = 14
    df['cmo_diff'] = df['Close'].diff()
    df['cmo_up'] = df['cmo_diff'].apply(lambda x: x if x > 0 else 0)
    df['cmo_down'] = df['cmo_diff'].apply(lambda x: abs(x) if x < 0 else 0)
    df['cmo_sum_up'] = df['cmo_up'].rolling(window=cmo_period).sum()
    df['cmo_sum_down'] = df['cmo_down'].rolling(window=cmo_period).sum()
    df['CMO'] = ((df['cmo_sum_up'] - df['cmo_sum_down']) / (df['cmo_sum_up'] + df['cmo_sum_down'])) * 100
    df['CMO'] = df['CMO'].fillna(0)

    # Kaufman's Adaptive Moving Average (KAMA)
    kama_period = 10
    fast_ema_const = 2 / (2 + 1)
    slow_ema_const = 2 / (30 + 1)
    df['kama_change'] = abs(df['Close'] - df['Close'].shift(kama_period))
    df['kama_volatility'] = df['Close'].diff().abs().rolling(window=kama_period).sum()
    df['kama_er'] = df['kama_change'] / df['kama_volatility']
    df['kama_er'] = df['kama_er'].fillna(0)
    df['kama_sc'] = (df['kama_er'] * (fast_ema_const - slow_ema_const) + slow_ema_const)**2
    df['KAMA'] = np.nan
    df.iloc[kama_period-1, df.columns.get_loc('KAMA')] = df['Close'].iloc[kama_period-1] # Initialize first KAMA value
    for i in range(kama_period, len(df)):
        df.iloc[i, df.columns.get_loc('KAMA')] = df.iloc[i-1, df.columns.get_loc('KAMA')] + df.iloc[i, df.columns.get_loc('kama_sc')] * (df.iloc[i, df.columns.get_loc('Close')] - df.iloc[i-1, df.columns.get_loc('KAMA')])
    df['KAMA'] = df['KAMA'].ffill().bfill().fillna(df['Close']) # Fill initial NaNs

    # Elder's Force Index (EFI)
    efi_period = 13
    df['EFI'] = (df['Close'].diff() * df['Volume']).ewm(span=efi_period, adjust=False).mean()
    df['EFI'] = df['EFI'].fillna(0)

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
            if df.iloc[i, df.columns.get_loc('Low')] < sar: # Trend reversal
                uptrend = False
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('Low')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('High')] > ep:
                    ep = df.iloc[i, df.columns.get_loc('High')]
                    af = min(max_af, af + 0.02)
        else: # Downtrend
            sar = sar + af * (ep - sar)
            if df.iloc[i, df.columns.get_loc('High')] > sar: # Trend reversal
                uptrend = True
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('High')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('Low')] < ep:
                    ep = df.iloc[i, df.columns.get_loc('Low')]
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

    # Classification label for BUY model: class_horizon-day forward > +target_percentage
    fwd = df["Close"].shift(-class_horizon)
    df["TargetClassBuy"] = ((fwd / df["Close"] - 1.0) > target_percentage).astype(float)

    # Classification label for SELL model: class_horizon-day forward < -target_percentage
    df["TargetClassSell"] = ((fwd / df["Close"] - 1.0) < -target_percentage).astype(float)

    # Dynamically build the list of features that are actually present in the DataFrame
    # This is the most critical part to ensure consistency
    
    # Define a base set of expected technical features
    expected_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", 
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "ROC_20", "ROC_60", "CMO", "KAMA", "EFI", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
        "Market_Momentum_SPY",
        "Sentiment_Score",
        "VIX_Index_Returns", "DXY_Index_Returns", "Gold_Futures_Returns", "Oil_Futures_Returns", "US10Y_Yield_Returns",
        "Oil_Price_Returns", "Gold_Price_Returns"
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV # Added RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier # Added for Neural Network model
from sklearn.preprocessing import MinMaxScaler # Added for scaling data for neural networks
from scipy.stats import uniform, randint # Added for RandomizedSearchCV

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

# Added for SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Run: pip install shap. SHAP analysis will be skipped.")
    SHAP_AVAILABLE = False

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
# These classes must be defined only if PyTorch is available,
# otherwise, 'nn' will not be defined and cause a NameError.



def analyze_shap_for_gru(model: GRUClassifier, scaler: MinMaxScaler, X_df: pd.DataFrame, feature_names: List[str], ticker: str, target_col: str):
    """
    Calculates and visualizes SHAP values for a GRU model.
    """
    if not SHAP_AVAILABLE:
        print(f"  [{ticker}] SHAP is not available. Skipping SHAP analysis.")
        return

    if isinstance(model, GRUClassifier):
        print(f"  [{ticker}] SHAP KernelExplainer is not directly compatible with GRU models due to sequential input. Skipping SHAP analysis for GRU.")
        return

    print(f"  [{ticker}] Calculating SHAP values for GRU model ({target_col})...")
    print(f"DEBUG: analyze_shap_for_gru - X_df shape: {X_df.shape}")
    
    try:
        # Define a wrapper prediction function for KernelExplainer
        # This function will capture 'model', 'scaler', 'SEQUENCE_LENGTH' from the outer scope.
        def gru_predict_proba_wrapper_for_kernel(X_unsequenced_np):
            # X_unsequenced_np is a 2D numpy array (num_samples, num_features)
            
            # Scale the data using the fitted scaler
            X_scaled_np = scaler.transform(X_unsequenced_np)
            
            # Create sequences from the scaled data
            X_sequences_for_pred = []
            # Handle cases where X_scaled_np might be too short for a full sequence
            if len(X_scaled_np) < SEQUENCE_LENGTH:
                # Return neutral probability for insufficient sequences
                return np.full(len(X_unsequenced_np), 0.5)

            for i in range(len(X_scaled_np) - SEQUENCE_LENGTH + 1):
                X_sequences_for_pred.append(X_scaled_np[i:i + SEQUENCE_LENGTH])
            
            if not X_sequences_for_pred:
                return np.full(len(X_unsequenced_np), 0.5) # Should not happen if len(X_scaled_np) >= SEQUENCE_LENGTH

            X_sequences_tensor = torch.tensor(np.array(X_sequences_for_pred), dtype=torch.float32)
            
            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            model.to(device)
            X_sequences_tensor = X_sequences_tensor.to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(X_sequences_tensor)
                # Apply sigmoid to get probabilities for KernelExplainer
                return torch.sigmoid(outputs).cpu().numpy().flatten()

        # Re-generate X_sequences from the full training data (X_df)
        # This ensures consistency with how the model was trained.
        # Use the scaler that was fitted during model training
        X_scaled_dl_full = scaler.transform(X_df)
        
        X_sequences_full = []
        for i in range(len(X_scaled_dl_full) - SEQUENCE_LENGTH):
            X_sequences_full.append(X_scaled_dl_full[i:i + SEQUENCE_LENGTH])
        
        if not X_sequences_full:
            print(f"  [{ticker}] Not enough data to create sequences for SHAP. Skipping.")
            return
        
        X_sequences_full_np = np.array(X_sequences_full) # Shape: (num_total_sequences, SEQUENCE_LENGTH, num_features)
        print(f"DEBUG: analyze_shap_for_gru - X_sequences_full_np shape: {X_sequences_full_np.shape}")

        # Sample background data for KernelExplainer (unsequenced, unscaled)
        # KernelExplainer expects background data in the same format as the prediction function input.
        # So, we need to sample from the original X_df (unsequenced, unscaled).
        num_background_samples = min(50, len(X_df)) # Smaller background for performance
        background_data_for_kernel = X_df.sample(num_background_samples, random_state=SEED).values # NumPy array (num_samples, num_features)
        print(f"DEBUG: analyze_shap_for_gru - background_data_for_kernel shape: {background_data_for_kernel.shape}")

        # Sample data to explain (unsequenced, unscaled)
        num_explain_samples = min(20, len(X_df)) # Very small sample for performance
        explain_data_for_kernel = X_df.sample(num_explain_samples, random_state=SEED).values # NumPy array (num_samples, num_features)
        print(f"DEBUG: analyze_shap_for_gru - explain_data_for_kernel shape: {explain_data_for_kernel.shape}")

        if explain_data_for_kernel.shape[0] == 0:
            print(f"  [{ticker}] Explain data for KernelExplainer is empty. Skipping SHAP calculation.")
            return

        # Create a KernelExplainer
        explainer = shap.KernelExplainer(
            gru_predict_proba_wrapper_for_kernel, # Pass the nested function directly
            background_data_for_kernel
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(explain_data_for_kernel)
        
        print(f"DEBUG: SHAP values raw output: {shap_values}")
        print(f"DEBUG: SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"DEBUG: SHAP values list length: {len(shap_values)}")
            if len(shap_values) > 0 and isinstance(shap_values[0], np.ndarray):
                print(f"DEBUG: SHAP values[0] shape: {shap_values[0].shape}")
            if len(shap_values) > 1 and isinstance(shap_values[1], np.ndarray):
                print(f"DEBUG: SHAP values[1] shape: {shap_values[1].shape}")
        elif isinstance(shap_values, np.ndarray):
            print(f"DEBUG: SHAP values array shape: {shap_values.shape}")

        shap_values_for_plot = None
        if isinstance(shap_values, list):
            if len(shap_values) == 2: # For binary classification, take the positive class (index 1)
                shap_values_for_plot = shap_values[1]
            elif len(shap_values) == 1: # If only one output, take that output
                shap_values_for_plot = shap_values[0]
            else:
                print(f"  [{ticker}] SHAP values list has unexpected length ({len(shap_values)}). Skipping plot.")
                return
        elif isinstance(shap_values, np.ndarray):
            shap_values_for_plot = shap_values
        else:
            print(f"  [{ticker}] SHAP values are not a list or numpy array. Type: {type(shap_values)}. Skipping plot.")
            return

        if shap_values_for_plot is None or shap_values_for_plot.size == 0:
            print(f"  [{ticker}] SHAP values for plotting are empty or None. Skipping plot.")
            return

        # For KernelExplainer, shap_values_for_plot is already (num_samples, num_features)
        # and explain_data_for_kernel is also (num_samples, num_features).
        # No need to average over sequence length.
        
        # Generate SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, explain_data_for_kernel, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for {ticker} GRU ({target_col})")
        plt.tight_layout()
        
        shap_plot_path = Path(f"logs/shap_plots/{ticker}_GRU_SHAP_{target_col}.png")
        _ensure_dir(shap_plot_path.parent)
        print(f"DEBUG: Attempting to save SHAP plot to {shap_plot_path}")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"  [{ticker}] SHAP summary plot saved to {shap_plot_path}")

    except Exception as e:
        print(f"  [{ticker}] Error during SHAP analysis for GRU model ({target_col}): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

def analyze_shap_for_tree_model(model, X_df: pd.DataFrame, feature_names: List[str], ticker: str, target_col: str):
    """
    Calculates and visualizes SHAP values for tree-based models (XGBoost, RandomForest).
    """
    if not SHAP_AVAILABLE:
        print(f"  [{ticker}] SHAP is not available. Skipping SHAP analysis.")
        return

    print(f"  [{ticker}] Calculating SHAP values for tree model ({target_col})...")
    
    try:
        # For tree-based models, TreeExplainer is more efficient
        explainer = shap.TreeExplainer(model)
        
        # Sample data to explain (unscaled, original features)
        num_explain_samples = min(100, len(X_df)) # Use a larger sample for tree models
        explain_data = X_df.sample(num_explain_samples, random_state=SEED)

        shap_values = explainer.shap_values(explain_data)
        
        print(f"DEBUG: SHAP values raw output: {shap_values}")
        print(f"DEBUG: SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"DEBUG: SHAP values list length: {len(shap_values)}")
            if len(shap_values) > 0 and isinstance(shap_values[0], np.ndarray):
                print(f"DEBUG: SHAP values[0] shape: {shap_values[0].shape}")
            if len(shap_values) > 1 and isinstance(shap_values[1], np.ndarray):
                print(f"DEBUG: SHAP values[1] shape: {shap_values[1].shape}")
        elif isinstance(shap_values, np.ndarray):
            print(f"DEBUG: SHAP values array shape: {shap_values.shape}")

        shap_values_for_plot = None
        if isinstance(shap_values, list):
            if len(shap_values) == 2: # For binary classification, take the positive class (index 1)
                shap_values_for_plot = shap_values[1]
            elif len(shap_values) == 1: # If only one output, take that output
                shap_values_for_plot = shap_values[0]
            else:
                print(f"  [{ticker}] SHAP values list has unexpected length ({len(shap_values)}). Skipping plot.")
                return
        elif isinstance(shap_values, np.ndarray):
            shap_values_for_plot = shap_values
        else:
            print(f"  [{ticker}] SHAP values are not a list or numpy array. Type: {type(shap_values)}. Skipping plot.")
            return

        if shap_values_for_plot is None or shap_values_for_plot.size == 0:
            print(f"  [{ticker}] SHAP values for plotting are empty or None. Skipping plot.")
            return

        # Generate SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, explain_data, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for {ticker} Tree Model ({target_col})")
        plt.tight_layout()
        
        shap_plot_path = Path(f"logs/shap_plots/{ticker}_TREE_SHAP_{target_col}.png")
        _ensure_dir(shap_plot_path.parent)
        print(f"DEBUG: Attempting to save SHAP plot to {shap_plot_path}")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"  [{ticker}] SHAP summary plot saved to {shap_plot_path}")

    except Exception as e:
        print(f"  [{ticker}] Error during SHAP analysis for tree model ({target_col}): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging


def train_and_evaluate_models(
    df: pd.DataFrame, # This is df_train_initial for non-GRU, or raw_df_for_feature_gen for GRU
    target_col: str = "TargetClassBuy",
    feature_set: Optional[List[str]] = None, # This is actual_feature_set_initial for non-GRU, or None for GRU
    ticker: str = "UNKNOWN",
    initial_model=None,
    loaded_gru_hyperparams: Optional[Dict] = None,
    models_and_params_global: Optional[Dict] = None,
    perform_gru_hp_optimization: bool = True,
    default_target_percentage: float = TARGET_PERCENTAGE, # New parameter
    default_class_horizon: int = CLASS_HORIZON # New parameter
):
    """Train and compare multiple classifiers for a given target, returning the best one."""
    models_and_params = models_and_params_global if models_and_params_global is not None else initialize_ml_libraries()
    
    # Initialize X, y, and scaler for non-DL models
    X_non_dl = pd.DataFrame()
    y_non_dl = np.array([])
    scaler_non_dl = None
    final_feature_names_non_dl = []

    # If GRU is not enabled or GRU optimization is disabled, prepare data once for all non-DL models.
    # In this case, `df` is `df_train_initial` and `feature_set` is `actual_feature_set_initial`.
    if not (USE_GRU and perform_gru_hp_optimization):
        d = df.copy() # This is df_train_initial
        if d.empty:
            print(f"  [DIAGNOSTIC] {ticker}: Input DataFrame for non-DL models is empty. Skipping non-DL models.")
        else:
            # Use the provided feature_set for non-DL models
            if feature_set is None:
                print("⚠️ feature_set was None for non-DL models. Inferring features from DataFrame columns.")
                final_feature_names_non_dl = [col for col in d.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell"]]
            else:
                final_feature_names_non_dl = [f for f in feature_set if f in d.columns]
            
            required_cols_for_training_non_dl = final_feature_names_non_dl + [target_col]
            if not all(col in d.columns for col in required_cols_for_training_non_dl):
                missing = [col for col in required_cols_for_training_non_dl if col not in d.columns]
                print(f"⚠️ Missing critical columns for non-DL model training (target: {target_col}, missing: {missing}). Skipping non-DL models.")
            else:
                d = d[required_cols_for_training_non_dl].dropna()
                if len(d) < 50:
                    print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after feature prep for non-DL models ({len(d)} rows, need >= 50). Skipping non-DL models.")
                else:
                    X_df_non_dl = d[final_feature_names_non_dl]
                    y_non_dl = d[target_col].values

                    # Scale features for non-DL models
                    if CUML_AVAILABLE and cuMLStandardScaler:
                        try:
                            scaler_non_dl = cuMLStandardScaler()
                            X_gpu_np = X_df_non_dl.values
                            X_scaled_non_dl = scaler_non_dl.fit_transform(X_gpu_np)
                            X_non_dl = pd.DataFrame(X_scaled_non_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                        except Exception as e:
                            print(f"⚠️ Error using cuML StandardScaler for non-DL models: {e}. Falling back to sklearn.StandardScaler.")
                            scaler_non_dl = StandardScaler()
                            X_scaled_non_dl = scaler_non_dl.fit_transform(X_df_non_dl)
                            X_non_dl = pd.DataFrame(X_scaled_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                    else:
                        scaler_non_dl = StandardScaler()
                        X_scaled_non_dl = scaler_non_dl.fit_transform(X_df_non_dl)
                        X_non_dl = pd.DataFrame(X_scaled_non_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                    scaler_non_dl.feature_names_in_ = list(final_feature_names_non_dl)
    
    # Define models and their parameter grids for GridSearchCV
    models_and_params_local = {} 

    # Add non-DL models to models_and_params_local, using X_non_dl, y_non_dl, scaler_non_dl
    if not X_non_dl.empty and len(y_non_dl) > 0:
        if CUML_AVAILABLE and USE_LOGISTIC_REGRESSION:
            models_and_params_local["cuML Logistic Regression"] = {
                "model": cuMLLogisticRegression(class_weight="balanced", solver='qn'),
                "params": {'C': [0.1, 1.0, 10.0]}
            }
        if CUML_AVAILABLE and USE_RANDOM_FOREST:
            models_and_params_local["cuML Random Forest"] = {
                "model": cuMLRandomForestClassifier(random_state=SEED),
                "params": {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, None]}
            }
        
        if USE_LOGISTIC_REGRESSION:
            models_and_params_local["Logistic Regression"] = {
                "model": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
                "params": {'C': [0.1, 1.0, 10.0, 100.0]}
            }
        if USE_RANDOM_FOREST:
            models_and_params_local["Random Forest"] = {
                "model": RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                "params": {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, None]}
            }
        if USE_SVM:
            models_and_params_local["SVM"] = {
                "model": SVC(probability=True, random_state=SEED, class_weight="balanced"),
                "params": {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['rbf', 'linear']}
            }
        if USE_MLP_CLASSIFIER:
            models_and_params_local["MLPClassifier"] = {
                "model": MLPClassifier(random_state=SEED, max_iter=500, early_stopping=True),
                "params": {'hidden_layer_sizes': [(100,), (100, 50), (50, 25)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]}
            }

        if LGBMClassifier and USE_LIGHTGBM:
            lgbm_model_params = {
                "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='cpu'), # Always set to CPU
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
            }
            models_and_params_local["LightGBM (CPU)"] = lgbm_model_params
            print("ℹ️ LightGBM found. Will use CPU.") # Update message

        if XGBOOST_AVAILABLE and XGBClassifier and USE_XGBOOST:
            xgb_device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            xgb_tree_method = 'gpu_hist' if CUDA_AVAILABLE else 'hist'
            xgb_predictor = 'gpu_predictor' if CUDA_AVAILABLE else 'cpu_predictor'
            xgb_model_params = {
                "model": XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=1, tree_method=xgb_tree_method, predictor=xgb_predictor),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            }
            models_and_params_local[f"XGBoost ({xgb_device.upper()})"] = xgb_model_params
            print(f"ℹ️ XGBoost found. Will use {xgb_device.upper()}.")


    # --- Deep Learning Models (LSTM/GRU) ---
    if PYTORCH_AVAILABLE and (USE_LSTM or USE_GRU):
        # `df` here is `raw_df_for_feature_gen`
        raw_df_dl = df.copy()

        if raw_df_dl.empty:
            print(f"  [DIAGNOSTIC] {ticker}: Input DataFrame for DL models is empty. Skipping DL models.")
        else:
            # Determine device
            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            
            input_size = 0 # Will be set after feature generation

            if USE_GRU:
                if perform_gru_hp_optimization and ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION:
                    print(f"    - Starting GRU hyperparameter randomized search for {ticker} ({target_col})...")
                    best_gru_auc = -np.inf
                    best_gru_model = None
                    best_gru_scaler = None # Scaler will be fitted per trial
                    best_gru_hyperparams = {}
                    
                    # Define parameter distributions for RandomizedSearchCV
                    gru_param_distributions = {
                        "hidden_size": GRU_HIDDEN_SIZE_OPTIONS, # Use predefined options
                        "num_layers": GRU_NUM_LAYERS_OPTIONS,
                        "dropout_rate": GRU_DROPOUT_OPTIONS,
                        "learning_rate": GRU_LEARNING_RATE_OPTIONS,
                        "batch_size": GRU_BATCH_SIZE_OPTIONS,
                        "epochs": GRU_EPOCHS_OPTIONS,
                        "class_horizon": GRU_CLASS_HORIZON_OPTIONS, # Include class_horizon
                        "target_percentage": GRU_TARGET_PERCENTAGE_OPTIONS # Include target_percentage
                    }
                    
                    n_trials = 20 # Number of random combinations to test

                    print(f"      GRU Hyperparameter Randomized Search for {ticker} ({target_col}) with {n_trials} trials:")

                    for trial in range(n_trials):
                        # Sample hyperparameters for the current trial
                        temp_hyperparams = {
                            "hidden_size": random.choice(gru_param_distributions["hidden_size"]),
                            "num_layers": random.choice(gru_param_distributions["num_layers"]),
                            "dropout_rate": random.choice(gru_param_distributions["dropout_rate"]),
                            "learning_rate": random.choice(gru_param_distributions["learning_rate"]),
                            "batch_size": random.choice(gru_param_distributions["batch_size"]),
                            "epochs": random.choice(gru_param_distributions["epochs"]),
                            "class_horizon": random.choice(gru_param_distributions["class_horizon"]), # Sample class_horizon
                            "target_percentage": random.choice(gru_param_distributions["target_percentage"]) # Sample target_percentage
                        }
                        
                        # Adjust dropout_rate if num_layers is 1 to avoid UserWarning
                        current_dropout_rate = temp_hyperparams["dropout_rate"] if temp_hyperparams["num_layers"] > 1 else 0.0
                        temp_hyperparams["dropout_rate"] = current_dropout_rate

                        print(f"          Testing GRU (Trial {trial + 1}/{n_trials}) with: HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']}, CH={temp_hyperparams['class_horizon']}, TP={temp_hyperparams['target_percentage']:.4f}")

                        # --- Re-generate features and labels for this trial's class_horizon and target_percentage ---
                        df_train_trial, actual_feature_set_trial = fetch_training_data(ticker, raw_df_dl.copy(), temp_hyperparams["target_percentage"], temp_hyperparams["class_horizon"])
                        
                        if df_train_trial.empty:
                            print(f"          [DIAGNOSTIC] {ticker}: Insufficient training data for GRU trial. Skipping.")
                            continue
                        
                        X_df_trial = df_train_trial[actual_feature_set_trial]
                        y_trial = df_train_trial[target_col].values

                        if len(X_df_trial) < SEQUENCE_LENGTH + 1:
                            print(f"          [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in GRU trial (need > {SEQUENCE_LENGTH} rows). Skipping.")
                            continue

                        # Scale data for DL models (MinMaxScaler)
                        dl_scaler_trial = MinMaxScaler(feature_range=(0, 1))
                        X_scaled_dl_trial = dl_scaler_trial.fit_transform(X_df_trial)
                        
                        # Create sequences
                        X_sequences_trial = []
                        y_sequences_trial = []
                        for i in range(len(X_scaled_dl_trial) - SEQUENCE_LENGTH):
                            X_sequences_trial.append(X_scaled_dl_trial[i:i + SEQUENCE_LENGTH])
                            y_sequences_trial.append(y_trial[i + SEQUENCE_LENGTH])
                        
                        if not X_sequences_trial:
                            print(f"          [DIAGNOSTIC] {ticker}: Not enough data to create sequences for GRU trial. Skipping.")
                            continue
                        
                        X_sequences_trial = torch.tensor(np.array(X_sequences_trial), dtype=torch.float32)
                        y_sequences_trial = torch.tensor(np.array(y_sequences_trial), dtype=torch.float32).unsqueeze(1)

                        input_size = X_sequences_trial.shape[2] # Number of features
                        
                        # Calculate pos_weight for BCEWithLogitsLoss
                        neg_count_trial = (y_sequences_trial == 0).sum()
                        pos_count_trial = (y_sequences_trial == 1).sum()
                        if pos_count_trial > 0 and neg_count_trial > 0:
                            pos_weight_trial = torch.tensor([neg_count_trial / pos_count_trial], device=device, dtype=torch.float32)
                            criterion_trial = nn.BCEWithLogitsLoss(pos_weight=pos_weight_trial)
                        else:
                            criterion_trial = nn.BCEWithLogitsLoss()

                        gru_model = GRUClassifier(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                        optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
                        
                        current_dataloader = DataLoader(TensorDataset(X_sequences_trial, y_sequences_trial), batch_size=temp_hyperparams["batch_size"], shuffle=True)

                        for epoch in range(temp_hyperparams["epochs"]):
                            for batch_X, batch_y in current_dataloader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                optimizer_gru.zero_grad()
                                outputs = gru_model(batch_X)
                                loss = criterion_trial(outputs, batch_y)
                                loss.backward()
                                optimizer_gru.step()
                        
                        # Evaluate GRU
                        gru_model.eval()
                        with torch.no_grad():
                            all_outputs = []
                            for batch_X, _ in current_dataloader:
                                batch_X = batch_X.to(device)
                                outputs = gru_model(batch_X)
                                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                            y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                        try:
                            from sklearn.metrics import roc_auc_score
                            auc_gru = roc_auc_score(y_sequences_trial.cpu().numpy(), y_pred_proba_gru)
                            print(f"            GRU AUC: {auc_gru:.4f}")

                            if auc_gru > best_gru_auc:
                                best_gru_auc = auc_gru
                                best_gru_model = gru_model
                                best_gru_scaler = dl_scaler_trial # Store the scaler for this best model
                                best_gru_hyperparams = temp_hyperparams.copy()
                        except ValueError:
                            print(f"            GRU AUC: Not enough samples with positive class for AUC calculation.")
                                
                    if best_gru_model:
                        models_and_params_local["GRU"] = {"model": best_gru_model, "scaler": best_gru_scaler, "auc": best_gru_auc, "hyperparams": best_gru_hyperparams}
                        print(f"      Best GRU found for {ticker} ({target_col}) with AUC: {best_gru_auc:.4f}, Hyperparams: {best_gru_hyperparams}")
                        if SAVE_PLOTS and SHAP_AVAILABLE:
                            # Need to pass X_df_trial (unscaled, unsequenced) for SHAP background
                            analyze_shap_for_gru(best_gru_model, best_gru_scaler, X_df_trial, actual_feature_set_trial, ticker, target_col)
                    else:
                        print(f"      No valid GRU model found after hyperparameter search for {ticker} ({target_col}).")
                        models_and_params_local["GRU"] = {"model": None, "scaler": None, "auc": 0.0}

                else: # ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION is False, use fixed or loaded hyperparameters
                    # This block needs to use the `default_target_percentage` and `default_class_horizon`
                    # or `loaded_gru_hyperparams` to generate features once.
                    
                    if loaded_gru_hyperparams:
                        print(f"    - Training GRU for {ticker} ({target_col}) with loaded hyperparameters...")
                        hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE)
                        num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS)
                        dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT)
                        learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE)
                        batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE)
                        epochs = loaded_gru_hyperparams.get("epochs", LSTM_EPOCHS)
                        class_horizon_fixed = loaded_gru_hyperparams.get("class_horizon", default_class_horizon)
                        target_percentage_fixed = loaded_gru_hyperparams.get("target_percentage", default_target_percentage)
                        print(f"      Loaded GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, CH={class_horizon_fixed}, TP={target_percentage_fixed:.4f}")
                    else:
                        print(f"    - Training GRU for {ticker} ({target_col}) with default fixed hyperparameters...")
                        hidden_size = LSTM_HIDDEN_SIZE
                        num_layers = LSTM_NUM_LAYERS
                        dropout_rate = LSTM_DROPOUT
                        learning_rate = LSTM_LEARNING_RATE
                        batch_size = LSTM_BATCH_SIZE
                        epochs = LSTM_EPOCHS
                        class_horizon_fixed = default_class_horizon
                        target_percentage_fixed = default_target_percentage
                        print(f"      Default GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, CH={class_horizon_fixed}, TP={target_percentage_fixed:.4f}")

                    # --- Generate features and labels once for fixed/loaded parameters ---
                    df_train_fixed_gru, actual_feature_set_fixed_gru = fetch_training_data(ticker, raw_df_dl.copy(), target_percentage_fixed, class_horizon_fixed)
                    
                    if df_train_fixed_gru.empty:
                        print(f"    [DIAGNOSTIC] {ticker}: Insufficient training data for fixed GRU. Skipping.")
                    else:
                        X_df_fixed_gru = df_train_fixed_gru[actual_feature_set_fixed_gru]
                        y_fixed_gru = df_train_fixed_gru[target_col].values

                        if len(X_df_fixed_gru) < SEQUENCE_LENGTH + 1:
                            print(f"    [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in fixed GRU (need > {SEQUENCE_LENGTH} rows). Skipping.")
                        else:
                            dl_scaler_fixed_gru = MinMaxScaler(feature_range=(0, 1))
                            X_scaled_dl_fixed_gru = dl_scaler_fixed_gru.fit_transform(X_df_fixed_gru)
                            
                            X_sequences_fixed_gru = []
                            y_sequences_fixed_gru = []
                            for i in range(len(X_scaled_dl_fixed_gru) - SEQUENCE_LENGTH):
                                X_sequences_fixed_gru.append(X_scaled_dl_fixed_gru[i:i + SEQUENCE_LENGTH])
                                y_sequences_fixed_gru.append(y_fixed_gru[i + SEQUENCE_LENGTH])
                            
                            if not X_sequences_fixed_gru:
                                print(f"    [DIAGNOSTIC] {ticker}: Not enough data to create sequences for fixed GRU. Skipping.")
                            else:
                                X_sequences_fixed_gru = torch.tensor(np.array(X_sequences_fixed_gru), dtype=torch.float32)
                                y_sequences_fixed_gru = torch.tensor(np.array(y_sequences_fixed_gru), dtype=torch.float32).unsqueeze(1)

                                input_size = X_sequences_fixed_gru.shape[2]
                                
                                neg_count_fixed_gru = (y_sequences_fixed_gru == 0).sum()
                                pos_count_fixed_gru = (y_sequences_fixed_gru == 1).sum()
                                if pos_count_fixed_gru > 0 and neg_count_fixed_gru > 0:
                                    pos_weight_fixed_gru = torch.tensor([neg_count_fixed_gru / pos_count_fixed_gru], device=device, dtype=torch.float32)
                                    criterion_fixed_gru = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fixed_gru)
                                else:
                                    criterion_fixed_gru = nn.BCEWithLogitsLoss()

                                gru_model = GRUClassifier(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                                if initial_model and isinstance(initial_model, GRUClassifier):
                                    try:
                                        gru_model.load_state_dict(initial_model.state_dict())
                                        print(f"    - Loaded existing GRU model state for {ticker} to continue training.")
                                    except Exception as e:
                                        print(f"    - Error loading GRU model state for {ticker}: {e}. Training from scratch.")
                                
                                optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)
                                
                                current_dataloader = DataLoader(TensorDataset(X_sequences_fixed_gru, y_sequences_fixed_gru), batch_size=batch_size, shuffle=True)

                                for epoch in range(epochs):
                                    for batch_X, batch_y in current_dataloader:
                                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                        optimizer_gru.zero_grad()
                                        outputs = gru_model(batch_X)
                                        loss = criterion_fixed_gru(outputs, batch_y)
                                        loss.backward()
                                        optimizer_gru.step()
                                
                                gru_model.eval()
                                with torch.no_grad():
                                    all_outputs = []
                                    for batch_X, _ in current_dataloader:
                                        batch_X = batch_X.to(device)
                                        outputs = gru_model(batch_X)
                                        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                                    y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                                try:
                                    from sklearn.metrics import roc_auc_score
                                    auc_gru = roc_auc_score(y_sequences_fixed_gru.cpu().numpy(), y_pred_proba_gru)
                                    current_gru_hyperparams = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout_rate": dropout_rate, "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs, "class_horizon": class_horizon_fixed, "target_percentage": target_percentage_fixed}
                                    models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler_fixed_gru, "auc": auc_gru, "hyperparams": current_gru_hyperparams}
                                    print(f"      GRU AUC (fixed/loaded params): {auc_gru:.4f}")
                                    if SAVE_PLOTS and SHAP_AVAILABLE:
                                        analyze_shap_for_gru(gru_model, dl_scaler_fixed_gru, X_df_fixed_gru, actual_feature_set_fixed_gru, ticker, target_col)
                                except ValueError:
                                    print(f"      GRU AUC (fixed/loaded params): Not enough samples with positive class for AUC calculation.")
                                    models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler_fixed_gru, "auc": 0.0}
            if USE_LSTM:
                # LSTM will use the same logic as fixed GRU, but with LSTM_HIDDEN_SIZE, etc.
                # For simplicity, I'll assume LSTM does not get hyperparameter optimization for class_horizon/target_percentage
                # and uses the default_target_percentage and default_class_horizon.
                # This is a simplification to keep the task focused on GRU.
                print(f"    - Training LSTM for {ticker}...")
                
                df_train_fixed_lstm, actual_feature_set_fixed_lstm = fetch_training_data(ticker, raw_df_dl.copy(), default_target_percentage, default_class_horizon)
                
                if df_train_fixed_lstm.empty:
                    print(f"    [DIAGNOSTIC] {ticker}: Insufficient training data for LSTM. Skipping.")
                else:
                    X_df_fixed_lstm = df_train_fixed_lstm[actual_feature_set_fixed_lstm]
                    y_fixed_lstm = df_train_fixed_lstm[target_col].values

                    if len(X_df_fixed_lstm) < SEQUENCE_LENGTH + 1:
                        print(f"    [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in LSTM (need > {SEQUENCE_LENGTH} rows). Skipping.")
                    else:
                        dl_scaler_fixed_lstm = MinMaxScaler(feature_range=(0, 1))
                        X_scaled_dl_fixed_lstm = dl_scaler_fixed_lstm.fit_transform(X_df_fixed_lstm)
                        
                        X_sequences_fixed_lstm = []
                        y_sequences_fixed_lstm = []
                        for i in range(len(X_scaled_dl_fixed_lstm) - SEQUENCE_LENGTH):
                            X_sequences_fixed_lstm.append(X_scaled_dl_fixed_lstm[i:i + SEQUENCE_LENGTH])
                            y_sequences_fixed_lstm.append(y_fixed_lstm[i + SEQUENCE_LENGTH])
                        
                        if not X_sequences_fixed_lstm:
                            print(f"    [DIAGNOSTIC] {ticker}: Not enough data to create sequences for LSTM. Skipping.")
                        else:
                            X_sequences_fixed_lstm = torch.tensor(np.array(X_sequences_fixed_lstm), dtype=torch.float32)
                            y_sequences_fixed_lstm = torch.tensor(np.array(y_sequences_fixed_lstm), dtype=torch.float32).unsqueeze(1)

                            input_size = X_sequences_fixed_lstm.shape[2]
                            
                            neg_count_fixed_lstm = (y_sequences_fixed_lstm == 0).sum()
                            pos_count_fixed_lstm = (y_sequences_fixed_lstm == 1).sum()
                            if pos_count_fixed_lstm > 0 and neg_count_fixed_lstm > 0:
                                pos_weight_fixed_lstm = torch.tensor([neg_count_fixed_lstm / pos_count_fixed_lstm], device=device, dtype=torch.float32)
                                criterion_fixed_lstm = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fixed_lstm)
                            else:
                                criterion_fixed_lstm = nn.BCEWithLogitsLoss()

                            lstm_model = LSTMClassifier(input_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 1, LSTM_DROPOUT).to(device)
                            if initial_model and isinstance(initial_model, LSTMClassifier):
                                try:
                                    lstm_model.load_state_dict(initial_model.state_dict())
                                    print(f"    - Loaded existing LSTM model state for {ticker} to continue training.")
                                except Exception as e:
                                    print(f"    - Error loading LSTM model state for {ticker}: {e}. Training from scratch.")
                            
                            optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LSTM_LEARNING_RATE)
                            
                            current_dataloader = DataLoader(TensorDataset(X_sequences_fixed_lstm, y_sequences_fixed_lstm), batch_size=LSTM_BATCH_SIZE, shuffle=True)

                            for epoch in range(LSTM_EPOCHS):
                                for batch_X, batch_y in current_dataloader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer_lstm.zero_grad()
                                    loss = criterion_fixed_lstm(outputs, batch_y)
                                    loss.backward()
                                    optimizer_lstm.step()
                            
                            lstm_model.eval()
                            with torch.no_grad():
                                all_outputs = []
                                for batch_X, _ in current_dataloader:
                                    batch_X = batch_X.to(device)
                                    outputs = lstm_model(batch_X)
                                    all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                                y_pred_proba_lstm = np.concatenate(all_outputs).flatten()

                            try:
                                from sklearn.metrics import roc_auc_score
                                auc_lstm = roc_auc_score(y_sequences_fixed_lstm.cpu().numpy(), y_pred_proba_lstm)
                                models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler_fixed_lstm, "auc": auc_lstm}
                                print(f"      LSTM AUC: {auc_lstm:.4f}")
                            except ValueError:
                                print(f"      LSTM AUC: Not enough samples with positive class for AUC calculation.")
                                models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler_fixed_lstm, "auc": 0.0}

    best_model_overall = None
    best_auc_overall = -np.inf
    best_hyperparams_overall: Optional[Dict] = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # Use 5 splits for consistency
    results = {}

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation with GridSearchCV):")
    for name, mp in models_and_params_local.items():
        if name in ["LSTM", "GRU"]:
            current_auc = mp["auc"]
            results[name] = current_auc
            print(f"    - {name}: {current_auc:.4f}")
            if current_auc > best_auc_overall:
                best_auc_overall = current_auc
                best_model_overall = mp["model"]
                scaler_non_dl = mp["scaler"] # Use the DL scaler for DL models
                if name == "GRU":
                    best_hyperparams_overall = mp.get("hyperparams")
        else:
            # For non-DL models, use X_non_dl and y_non_dl
            if X_non_dl.empty or len(y_non_dl) == 0:
                print(f"    - {name}: Skipping evaluation due to empty non-DL training data.")
                results[name] = 0.0
                continue

            model = mp["model"]
            params = mp["params"]
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
                    
                    grid_search = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
                    grid_search.fit(X_non_dl, y_non_dl) # Use X_non_dl and y_non_dl
                    
                    best_score = grid_search.best_score_
                    results[name] = best_score
                    print(f"    - {name}: {best_score:.4f} (Best Params: {grid_search.best_params_})")

                    if best_score > best_auc_overall:
                        best_auc_overall = best_score
                        best_model_overall = grid_search.best_estimator_
                        best_hyperparams_overall = None

            except Exception as e:
                print(f"    - {name}: Failed evaluation. Error: {e}")
                results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None, None, None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best model: {best_model_name} with AUC = {best_auc_overall:.4f}")

    if best_model_name in ["LSTM", "GRU"]:
        return models_and_params_local[best_model_name]["model"], models_and_params_local[best_model_name]["scaler"], best_hyperparams_overall
    else:
        print(f"  [DIAGNOSTIC] {ticker}: train_and_evaluate_models - Returning best model: {best_model_name}, AUC: {best_auc_overall:.4f}")
        if SAVE_PLOTS and SHAP_AVAILABLE and isinstance(best_model_overall, (RandomForestClassifier, XGBClassifier)):
            analyze_shap_for_tree_model(best_model_overall, X_df_non_dl, final_feature_names_non_dl, ticker, target_col)
        return best_model_overall, scaler_non_dl, best_hyperparams_overall

def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    model_sell_path = models_dir / f"{ticker}_model_sell.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_buy_path = models_dir / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
    gru_hyperparams_sell_path = models_dir / f"{ticker}_TargetClassSell_gru_optimized_params.json"

    model_buy, model_sell, scaler = None, None, None
    
    # Flag to indicate if we successfully loaded a model to continue training
    loaded_for_retraining = False

    # Attempt to load models and GRU hyperparams if CONTINUE_TRAINING_FROM_EXISTING is True
    if CONTINUE_TRAINING_FROM_EXISTING and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

    # If FORCE_TRAINING is False and we didn't load for retraining, then we just load and skip training
    if not FORCE_TRAINING and not loaded_for_retraining and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} (FORCE_TRAINING is False).")
            # Before returning, ensure PyTorch models are on CPU if they are deep learning models
            if PYTORCH_AVAILABLE:
                if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                    model_buy = model_buy.cpu()
                if isinstance(model_sell, (LSTMClassifier, GRUClassifier)):
                    model_sell = model_sell.cpu()
            return {
                'ticker': ticker,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler,
                'gru_hyperparams_buy': loaded_gru_hyperparams_buy,
                'gru_hyperparams_sell': loaded_gru_hyperparams_sell,
                'status': 'loaded',
                'reason': None
            }
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker}: {e}. Training from scratch.")
            # Fall through to training from scratch if loading fails

    print(f"  ⚙️ Training models for {ticker} (FORCE_TRAINING is {FORCE_TRAINING}, CONTINUE_TRAINING_FROM_EXISTING is {CONTINUE_TRAINING_FROM_EXISTING})...")
    print(f"  [DEBUG] {current_process().name} - {ticker}: Initiating feature extraction for training.")
    
    df_train, actual_feature_set = fetch_training_data(ticker, df_train_period, target_percentage, class_horizon)

    if df_train.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data.")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for BUY target.")
    # Train BUY model, passing the potentially loaded model and GRU hyperparams
    # Pass the global models_and_params to avoid re-initialization in worker processes
    global_models_and_params = initialize_ml_libraries() # Ensure it's initialized in the worker process too
    model_buy, scaler_buy, gru_hyperparams_buy = train_and_evaluate_models(
        df_train, "TargetClassBuy", actual_feature_set, ticker=ticker,
        initial_model=model_buy if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_buy,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )
    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for SELL target.")
    # Train SELL model, passing the potentially loaded model and GRU hyperparams
    model_sell, scaler_sell, gru_hyperparams_sell = train_and_evaluate_models(
        df_train, "TargetClassSell", actual_feature_set, ticker=ticker,
        initial_model=model_sell if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_sell,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=target_percentage, # Pass current target_percentage
        default_class_horizon=class_horizon # Pass current class_horizon
    )

    # For simplicity, we'll use the scaler from the buy model for both if they are different.
    # In a more complex scenario, you might want to ensure feature_set consistency or use separate scalers.
    final_scaler = scaler_buy if scaler_buy else scaler_sell

    if model_buy and model_sell and final_scaler:
        try:
            joblib.dump(model_buy, model_buy_path)
            joblib.dump(model_sell, model_sell_path)
            joblib.dump(final_scaler, scaler_path)
            
            if gru_hyperparams_buy:
                with open(gru_hyperparams_buy_path, 'w') as f:
                    json.dump(gru_hyperparams_buy, f, indent=4)
            if gru_hyperparams_sell:
                with open(gru_hyperparams_sell_path, 'w') as f:
                    json.dump(gru_hyperparams_sell, f, indent=4)

            print(f"  ✅ Models, scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving models or GRU hyperparams for {ticker}: {e}")
            
        # Before returning, ensure PyTorch models are on CPU if they are deep learning models
        if PYTORCH_AVAILABLE:
            if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                model_buy = model_buy.cpu()
            if isinstance(model_sell, (LSTMClassifier, GRUClassifier)):
                model_sell = model_sell.cpu()

        return {
            'ticker': ticker,
            'model_buy': model_buy,
            'model_sell': model_sell,
            'scaler': final_scaler,
            'gru_hyperparams_buy': gru_hyperparams_buy,
            'gru_hyperparams_sell': gru_hyperparams_sell,
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
                 feature_set: Optional[List[str]] = None,
                 per_ticker_min_proba_buy: Optional[float] = None, per_ticker_min_proba_sell: Optional[float] = None,
                 use_simple_rule_strategy: bool = USE_SIMPLE_RULE_STRATEGY, # New parameter
                 simple_rule_trailing_stop_percent: float = SIMPLE_RULE_TRAILING_STOP_PERCENT, # New parameter
                 simple_rule_take_profit_percent: float = SIMPLE_RULE_TAKE_PROFIT_PERCENT): # New parameter
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
        self.use_simple_rule_strategy = use_simple_rule_strategy # Assign new parameter
        self.simple_rule_trailing_stop_percent = simple_rule_trailing_stop_percent # Assign new parameter
        self.simple_rule_take_profit_percent = simple_rule_take_profit_percent # Assign new parameter
        # Dynamically determine the full feature set including financial features
        # This will be passed from the training worker
        self.feature_set = feature_set if feature_set is not None else [
            "Close", "Volume", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "ATR",
            "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
            "OBV", "CMF", "ROC", "ROC_20", "ROC_60", "CMO", "KAMA", "EFI", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
            "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
            'Fin_Revenue', 'Fin_NetIncome', 'Fin_TotalAssets', 'Fin_TotalLiabilities', 'Fin_FreeCashFlow', 'Fin_EBITDA',
            'Market_Momentum_SPY'
        ]
        
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
        self.trailing_stop_price: Optional[float] = None # New: For simple rule strategy
        self.take_profit_price: Optional[float] = None # New: For simple rule strategy
        
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
        if self.df.empty:
            print(f"  [DIAGNOSTIC] {self.ticker}: DataFrame became empty after dropping NaNs in 'Close' during _prepare_data. Skipping further prep.")
            return # Exit if no data left after cleaning
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
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean() # Added MACD_signal
        
        # Bollinger Bands for features
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.df['BB_upper'] = bb_mid + (bb_std * 2)
        self.df['BB_lower'] = bb_mid - (bb_std * 2) # Added BB_lower

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
        self.df['TR'] = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        alpha = 1/14
        self.df['+DM14'] = self.df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['-DM14'] = self.df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['TR14'] = self.df['TR'].ewm(alpha=alpha, adjust=False).mean()
        self.df['DX'] = (abs(self.df['+DM14'] - self.df['-DM14']) / (self.df['+DM14'] + self.df['-DM14'])) * 100
        self.df['DX'] = self.df['DX'].fillna(0) # Fill NaNs for DX immediately after calculation
        self.df['ADX'] = self.df['DX'].ewm(alpha=alpha, adjust=False).mean()
        self.df['ADX'] = self.df['ADX'].fillna(0) # Fill NaNs for ADX immediately after calculation
        # Fill NaNs for all other ADX-related indicators after their calculations
        self.df['+DM'] = self.df['+DM'].fillna(0)
        self.df['-DM'] = self.df['-DM'].fillna(0)
        self.df['TR'] = self.df['TR'].fillna(0)
        self.df['+DM14'] = self.df['+DM14'].fillna(0)
        self.df['-DM14'] = self.df['-DM14'].fillna(0)
        self.df['TR14'] = self.df['TR14'].fillna(0)
        # Fill NaNs for Stochastic Oscillator after its calculations
        self.df['%K'] = self.df['%K'].fillna(0)
        self.df['%D'] = self.df['%D'].fillna(0)

        # On-Balance Volume (OBV)
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        # Chaikin Money Flow (CMF)
        mfv = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low']) * self.df['Volume']
        self.df['CMF'] = mfv.rolling(window=20).sum() / self.df['Volume'].rolling(window=20).sum()
        self.df['CMF'] = self.df['CMF'].fillna(0)

        # Rate of Change (ROC)
        self.df['ROC'] = self.df['Close'].pct_change(periods=12) * 100
        self.df['ROC_20'] = self.df['Close'].pct_change(periods=20) * 100
        self.df['ROC_60'] = self.df['Close'].pct_change(periods=60) * 100

        # Chande Momentum Oscillator (CMO)
        cmo_period = 14
        self.df['cmo_diff'] = self.df['Close'].diff()
        self.df['cmo_up'] = self.df['cmo_diff'].apply(lambda x: x if x > 0 else 0)
        self.df['cmo_down'] = self.df['cmo_diff'].apply(lambda x: abs(x) if x < 0 else 0)
        self.df['cmo_sum_up'] = self.df['cmo_up'].rolling(window=cmo_period).sum()
        self.df['cmo_sum_down'] = self.df['cmo_down'].rolling(window=cmo_period).sum()
        self.df['CMO'] = ((self.df['cmo_sum_up'] - self.df['cmo_sum_down']) / (self.df['cmo_sum_up'] + self.df['cmo_sum_down'])) * 100
        self.df['CMO'] = self.df['CMO'].fillna(0)

        # Kaufman's Adaptive Moving Average (KAMA)
        kama_period = 10
        fast_ema_const = 2 / (2 + 1)
        slow_ema_const = 2 / (30 + 1)
        self.df['kama_change'] = abs(self.df['Close'] - self.df['Close'].shift(kama_period))
        self.df['kama_volatility'] = self.df['Close'].diff().abs().rolling(window=kama_period).sum()
        self.df['kama_er'] = self.df['kama_change'] / self.df['kama_volatility']
        self.df['kama_er'] = self.df['kama_er'].fillna(0)
        self.df['kama_sc'] = (self.df['kama_er'] * (fast_ema_const - slow_ema_const) + slow_ema_const)**2
        self.df['KAMA'] = np.nan
        self.df.iloc[kama_period-1, self.df.columns.get_loc('KAMA')] = self.df['Close'].iloc[kama_period-1] # Initialize first KAMA value
        for i in range(kama_period, len(self.df)):
            self.df.iloc[i, self.df.columns.get_loc('KAMA')] = self.df.iloc[i-1, self.df.columns.get_loc('KAMA')] + self.df.iloc[i, self.df.columns.get_loc('kama_sc')] * (self.df.iloc[i, self.df.columns.get_loc('Close')] - self.df.iloc[i-1, self.df.columns.get_loc('KAMA')])
        self.df['KAMA'] = self.df['KAMA'].ffill().bfill().fillna(self.df['Close']) # Fill initial NaNs

        # Elder's Force Index (EFI)
        efi_period = 13
        self.df['EFI'] = (self.df['Close'].diff() * self.df['Volume']).ewm(span=efi_period, adjust=False).mean()
        self.df['EFI'] = self.df['EFI'].fillna(0)

        # Keltner Channels
        self.df['KC_TR'] = pd.concat([self.df['High'] - self.df['Low'], (self.df['High'] - self.df['Close'].shift(1)).abs(), (self.df['Low'] - self.df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        self.df['KC_ATR'] = self.df['KC_TR'].rolling(window=10).mean()
        self.df['KC_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['KC_Upper'] = self.df['KC_Middle'] + (self.df['KC_ATR'] * 2)
        self.df['KC_Lower'] = self.df['KC_Middle'] - (self.df['KC_ATR'] * 2)

        # Donchian Channels
        self.df['DC_Upper'] = self.df['High'].rolling(window=20).max()
        self.df['DC_Lower'] = self.df['Low'].rolling(window=20).min()
        self.df['DC_Middle'] = (self.df['DC_Upper'] + self.df['DC_Lower']) / 2

        # Parabolic SAR (PSAR) - Re-implementing for RuleTradingEnv
        psar = self.df['Close'].copy()
        af = 0.02 # Acceleration Factor
        max_af = 0.2 # Maximum Acceleration Factor

        # Initial trend and extreme point
        # Assume initial uptrend if Close > Open, downtrend otherwise
        uptrend = True if self.df['Close'].iloc[0] > self.df['Open'].iloc[0] else False
        ep = self.df['High'].iloc[0] if uptrend else self.df['Low'].iloc[0]
        sar = self.df['Low'].iloc[0] if uptrend else self.df['High'].iloc[0]
        
        # Iterate to calculate PSAR
        for i in range(1, len(self.df)):
            if uptrend:
                sar = sar + af * (ep - sar)
                if self.df.iloc[i, self.df.columns.get_loc('Low')] < sar: # Trend reversal
                    uptrend = False
                    sar = ep
                    ep = self.df.iloc[i, self.df.columns.get_loc('Low')]
                    af = 0.02
                else:
                    if self.df.iloc[i, self.df.columns.get_loc('High')] > ep:
                        ep = self.df.iloc[i, self.df.columns.get_loc('High')]
                        af = min(max_af, af + 0.02)
            else: # Downtrend
                sar = sar + af * (ep - sar)
                if self.df.iloc[i, self.df.columns.get_loc('High')] > sar: # Trend reversal
                    uptrend = True
                    sar = ep
                    ep = self.df.iloc[i, self.df.columns.get_loc('High')]
                    af = 0.02
                else:
                    if self.df.iloc[i, self.df.columns.get_loc('Low')] < ep:
                        ep = self.df.iloc[i, self.df.columns.get_loc('Low')]
                        af = min(max_af, af + 0.02)
            psar.iloc[i] = sar
        self.df['PSAR'] = psar

        # Accumulation/Distribution Line (ADL) - Re-implementing for RuleTradingEnv
        mf_multiplier = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        mf_volume = mf_multiplier * self.df['Volume']
        self.df['ADL'] = mf_volume.cumsum()
        self.df['ADL'] = self.df['ADL'].fillna(0) # Fill initial NaNs with 0

        # Commodity Channel Index (CCI) - Re-implementing for RuleTradingEnv
        TP = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
        self.df['CCI'] = self.df['CCI'].fillna(0) # Fill initial NaNs with 0

        # Volume Weighted Average Price (VWAP)
        self.df['VWAP'] = (self.df['Close'] * self.df['Volume']).rolling(window=FEAT_VOL_WINDOW).sum() / self.df['Volume'].rolling(window=FEAT_VOL_WINDOW).sum()
        self.df['VWAP'] = self.df['VWAP'].fillna(self.df['Close']) # Fill initial NaNs with Close price

        # ATR Percentage
        # Ensure ATR is calculated before ATR_Pct
        if "ATR" not in self.df.columns:
            high = self.df["High"] if "High" in self.df.columns else None
            low  = self.df["Low"]  if "Low" in self.df.columns else None
            prev_close = self.df["Close"].shift(1)
            if high is not None and low is not None:
                hl = (high - low).abs()
                h_pc = (high - prev_close).abs()
                l_pc = (low  - prev_close).abs()
                tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
                self.df["ATR"] = tr.rolling(ATR_PERIOD).mean()
            else:
                ret = self.df["Close"].pct_change(fill_method=None)
                self.df["ATR"] = (ret.rolling(ATR_PERIOD).std() * self.df["Close"]).rolling(2).mean()
            self.df["ATR"] = self.df["ATR"].fillna(0)

        self.df['ATR_Pct'] = (self.df['ATR'] / self.df['Close']) * 100
        self.df['ATR_Pct'] = self.df['ATR_Pct'].fillna(0)

        # Chaikin Oscillator
        mf_multiplier_co = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        adl_fast = (mf_multiplier_co * self.df['Volume']).ewm(span=3, adjust=False).mean()
        adl_slow = (mf_multiplier_co * self.df['Volume']).ewm(span=10, adjust=False).mean()
        self.df['Chaikin_Oscillator'] = adl_fast - adl_slow
        self.df['Chaikin_Oscillator'] = self.df['Chaikin_Oscillator'].fillna(0)

        # Money Flow Index (MFI)
        typical_price_mfi = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow_mfi = typical_price_mfi * self.df['Volume']
        positive_mf_mfi = money_flow_mfi.where(typical_price_mfi > typical_price_mfi.shift(1), 0)
        negative_mf_mfi = money_flow_mfi.where(typical_price_mfi < typical_price_mfi.shift(1), 0)
        mfi_ratio = positive_mf_mfi.rolling(window=14).sum() / negative_mf_mfi.rolling(window=14).sum()
        self.df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        self.df['MFI'] = self.df['MFI'].fillna(0)

        # OBV Moving Average
        self.df['OBV_SMA'] = self.df['OBV'].rolling(window=10).mean()
        self.df['OBV_SMA'] = self.df['OBV_SMA'].fillna(0)

        # Historical Volatility (e.g., 20-day rolling standard deviation of log returns)
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Historical_Volatility'] = self.df['Log_Returns'].rolling(window=20).std() * np.sqrt(252) # Annualized
        self.df['Historical_Volatility'] = self.df['Historical_Volatility'].fillna(0)

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
            # Handle PyTorch models (LSTM/GRU) separately
            if PYTORCH_AVAILABLE and isinstance(model, (LSTMClassifier, GRUClassifier)):
                # For PyTorch models, we need to scale the data and create sequences
                # The scaler for DL models is MinMaxScaler, which was fitted on unsequenced data
                # We need to get the last SEQUENCE_LENGTH rows for prediction
                start_idx = max(0, i - SEQUENCE_LENGTH + 1)
                end_idx = i + 1
                
                # Ensure we have enough data for the sequence
                if end_idx < SEQUENCE_LENGTH: # Not enough data for a full sequence yet
                    # Pad with zeros or handle as insufficient data
                    # For now, return 0.0 (no strong signal)
                    return 0.0
                
                # Get the relevant historical data for sequencing
                historical_data_for_seq = self.df.loc[start_idx:end_idx-1, model_feature_names].copy()
                
                # Ensure all columns are numeric and fill any NaNs
                for col in historical_data_for_seq.columns:
                    historical_data_for_seq[col] = pd.to_numeric(historical_data_for_seq[col], errors='coerce').fillna(0.0)

                # Scale the sequence data
                X_scaled_seq = self.scaler.transform(historical_data_for_seq)
                
                # Convert to tensor and add batch dimension
                X_tensor = torch.tensor(X_scaled_seq, dtype=torch.float32).unsqueeze(0)
                
                # Move to appropriate device
                device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
                X_tensor = X_tensor.to(device)

                model.eval() # Set model to evaluation mode
                with torch.no_grad():
                    output = model(X_tensor)
                    return float(torch.sigmoid(output).cpu().numpy()[0][0]) # Apply sigmoid to get probability
            else:
                # For traditional ML models, use the existing scaling and prediction logic
                X_scaled_np = self.scaler.transform(X_df)
                X = pd.DataFrame(X_scaled_np, columns=model_feature_names)
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
        # Use a fixed investment amount per stock
        investment_amount = INVESTMENT_PER_STOCK
        
        # Calculate quantity based on the fixed investment amount
        qty = int(investment_amount / price)
        
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

        # --- Entry Signal ---
        # Condition: AI model must approve (if not using simple rule strategy)
        ai_signal = False
        if not self.use_simple_rule_strategy:
            ai_signal = self._allow_buy_by_model(self.current_step)
        
        # Simple rule-based entry: SMA crossover (if not using AI gate)
        simple_rule_entry_signal = False
        if self.use_simple_rule_strategy:
            sma_short = self.df.loc[self.current_step, "SMA_F_S"]
            sma_long = self.df.loc[self.current_step, "SMA_F_L"]
            prev_sma_short = self.df.loc[self.current_step - 1, "SMA_F_S"]
            prev_sma_long = self.df.loc[self.current_step - 1, "SMA_F_L"]
            if prev_sma_short <= prev_sma_long and sma_short > sma_long:
                simple_rule_entry_signal = True

        if self.shares == 0 and (ai_signal or simple_rule_entry_signal):
            # print(f"  [{self.ticker}] DEBUG: Attempting BUY. Buy Prob: {self.last_buy_prob:.2f}, Threshold: {self.min_proba_buy:.2f}") # Commented out
            self._buy(price, atr, date)
        
        # --- Exit Signals ---
        # 1. AI-driven Exit Signal (if not using simple rule strategy)
        ai_exit_signal = False
        if not self.use_simple_rule_strategy:
            ai_exit_signal = self._allow_sell_by_model(self.current_step)

        # 2. Simple Rule-based Exit (Trailing Stop / Take Profit)
        simple_rule_exit_signal = False
        if self.shares > 0 and self.use_simple_rule_strategy:
            # Update highest price since entry
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            
            # Calculate trailing stop and take profit levels
            if self.entry_price is not None:
                self.trailing_stop_price = self.highest_since_entry * (1 - self.simple_rule_trailing_stop_percent)
                self.take_profit_price = self.entry_price * (1 + self.simple_rule_take_profit_percent)
            
            # Check for trailing stop or take profit
            if self.trailing_stop_price is not None and price <= self.trailing_stop_price:
                simple_rule_exit_signal = True
                self.last_ai_action = "SELL (Trailing Stop)"
            elif self.take_profit_price is not None and price >= self.take_profit_price:
                simple_rule_exit_signal = True
                self.last_ai_action = "SELL (Take Profit)"

        if self.shares > 0 and (ai_exit_signal or simple_rule_exit_signal):
            # print(f"  [{self.ticker}] DEBUG: Attempting SELL. Sell Prob: {self.last_sell_prob:.2f}, Threshold: {self.min_proba_sell:.2f}") # Commented out
            self._sell(price, date)
        else:
            self.last_ai_action = "HOLD"
            # print(f"  [{self.ticker}] DEBUG: HOLD. Buy Prob: {self.last_buy_prob:.2f}, Sell Prob: {self.last_sell_prob:.2f}") # Commented out

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple], str, float, float, float]: # Added float for shares_before_liquidation
        if self.df.empty:
            return self.initial_balance, [], "N/A", np.nan, np.nan, 0.0
        done = False
        while not done:
            done = self.step()
        
        shares_before_liquidation = self.shares # Capture shares before final liquidation
        
        if self.shares > 0 and not self.df.empty:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log, self.last_ai_action, self.last_buy_prob, self.last_sell_prob, shares_before_liquidation # Return shares_before_liquidation
# ============================
# Analytics
# ============================

def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy = params # Added use_simple_rule_strategy
    
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
            feature_set=feature_set,
            per_ticker_min_proba_buy=min_proba_buy,
            per_ticker_min_proba_sell=min_proba_sell,
            use_simple_rule_strategy=use_simple_rule_strategy # Pass new parameter
        )
        final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation = env.run()

        # Calculate individual Buy & Hold for the same period
        start_price_bh = float(df_backtest["Close"].iloc[0])
        end_price_bh = float(df_backtest["Close"].iloc[-1])
        individual_bh_return = ((end_price_bh - start_price_bh) / start_price_bh) * 100 if start_price_bh > 0 else 0.0
        
        # Analyze performance for this ticker
        perf_data = analyze_performance(trade_log, env.portfolio_history, df_backtest["Close"].tolist(), ticker)

        # Calculate Buy & Hold history for this ticker
        bh_history_for_ticker = []
        if not df_backtest.empty:
            start_price = float(df_backtest["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            for price_day in df_backtest["Close"].tolist():
                bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
        else:
            bh_history_for_ticker.append(capital_per_stock) # If no data, assume initial capital

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob,
            'shares_before_liquidation': shares_before_liquidation, # Return the shares before liquidation
            'buy_hold_history': bh_history_for_ticker # Return the buy_hold_history
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

def calculate_buy_hold_performance_metrics(
    buy_hold_history: List[float],
    ticker: str
) -> Dict[str, float]:
    """Calculates key performance metrics for Buy & Hold strategy."""
    bh_returns = pd.Series(buy_hold_history).pct_change(fill_method=None).dropna()

    # Sharpe Ratio (annualized, assuming 252 trading days)
    sharpe_bh = (bh_returns.mean() / bh_returns.std()) * np.sqrt(252) if bh_returns.std() > 0 else 0

    # Max Drawdown
    bh_series = pd.Series(buy_hold_history)
    bh_cummax = bh_series.cummax()
    bh_drawdown = ((bh_series - bh_cummax) / bh_cummax).min()

    return {
        "trades": 0, "win_rate": 0.0, "total_pnl": 0.0, # No trades for buy & hold
        "sharpe_ratio": sharpe_bh, "max_drawdown": bh_drawdown
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
        if not df_1y.empty:  # Some basic data quality check
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
                if fcf is not None:
                    print(f"  [DEBUG] {ticker}: FCF = {fcf}, Threshold = {fcf_min_threshold}")
                    if fcf <= fcf_min_threshold:
                        fcf_ok = False
                else:
                    print(f"  [DEBUG] {ticker}: FCF data not found.")
                    fcf_ok = False # If FCF data is missing, consider it a failure
        
        # EBITDA Check
        ebitda_ok = True
        if ebitda_min_threshold is not None:
            financials = yf_ticker.financials
            if not financials.empty:
                latest_financials = financials.iloc[:, 0]
                ebitda_keys = ['EBITDA', 'ebitda']
                ebitda = None
                for key in ebitda_keys: # Corrected loop to iterate over ebitda_keys
                    if key in latest_financials.index:
                        ebitda = latest_financials[key]
                        break
                if ebitda is not None:
                    print(f"  [DEBUG] {ticker}: EBITDA = {ebitda}, Threshold = {ebitda_min_threshold}")
                    if ebitda <= ebitda_min_threshold:
                        ebitda_ok = False
                else:
                    print(f"  [DEBUG] {ticker}: EBITDA data not found.")
                    ebitda_ok = False # If EBITDA data is missing, consider it a failure
            else:
                print(f"  [DEBUG] {ticker}: Financials (EBITDA) dataframe is empty.")
                ebitda_ok = False # If financials dataframe is empty, consider it a failure

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
    default_target_percentage: float, # Renamed to avoid confusion with per-ticker target
    default_class_horizon: int, # New parameter
    feature_set: Optional[List[str]],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    capital_per_stock: float,
    run_parallel: bool,
    force_percentage_optimization: bool, # New parameter
    force_thresholds_optimization: bool, # Add this parameter
    current_optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None
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
        
        current_buy_proba = MIN_PROBA_BUY
        current_sell_proba = MIN_PROBA_SELL
        current_target_perc = default_target_percentage
        current_class_horizon = default_class_horizon

        if current_optimized_params_per_ticker and ticker in current_optimized_params_per_ticker:
            current_buy_proba = current_optimized_params_per_ticker[ticker].get('min_proba_buy', MIN_PROBA_BUY)
            current_sell_proba = current_optimized_params_per_ticker[ticker].get('min_proba_sell', MIN_PROBA_SELL)
            current_target_perc = current_optimized_params_per_ticker[ticker].get('target_percentage', default_target_percentage)
            current_class_horizon = current_optimized_params_per_ticker[ticker].get('class_horizon', default_class_horizon)

        optimization_params.append((
            ticker, train_start, train_end, current_target_perc, current_class_horizon,
            feature_set, models_buy[ticker], models_sell[ticker], scalers[ticker],
            capital_per_stock,
            current_buy_proba, current_sell_proba,
            force_percentage_optimization,
            force_thresholds_optimization # Pass this new parameter
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
                'target_percentage': res['target_percentage'],
                'class_horizon': res['class_horizon'], # Store optimized class_horizon
                'optimization_status': res['optimization_status']
            }
            print(f"  ✅ {res['ticker']} optimized: Buy={res['min_proba_buy']:.2f}, Sell={res['min_proba_sell']:.2f}, Target%={res['target_percentage']:.2%}, Class Horiz={res['class_horizon']}, Status: {res['optimization_status']}")
    
    return optimized_params_per_ticker

def optimize_single_ticker_worker(params: Tuple) -> Dict:
    """Worker function to optimize thresholds and target percentage for a single ticker."""
    ticker, train_start, train_end, initial_target_percentage, initial_class_horizon, feature_set, \
        model_buy, model_sell, scaler, capital_per_stock, \
        current_min_proba_buy, current_min_proba_sell, force_percentage_optimization, \
        force_thresholds_optimization = params # Add force_thresholds_optimization here
    
    print(f"  [DEBUG] {current_process().name} - Optimizing for ticker: {ticker}")

    best_sharpe = -np.inf # Initialize best_sharpe
    best_min_proba_buy = current_min_proba_buy
    best_min_proba_sell = current_min_proba_sell
    best_target_percentage = initial_target_percentage
    best_class_horizon = initial_class_horizon

    step_proba = 0.05
    min_proba_buy_range = sorted(list(set([round(x, 2) for x in [max(0.0, current_min_proba_buy - step_proba), current_min_proba_buy, current_min_proba_buy + step_proba] if 0.0 <= x <= 1.0])))
    min_proba_sell_range = sorted(list(set([round(x, 2) for x in [max(0.0, current_min_proba_sell - step_proba), current_min_proba_sell, current_min_proba_sell + step_proba] if 0.0 <= x <= 1.0])))

    # Determine target_percentage_range
    if force_percentage_optimization:
        # Use the global options for a broader search if forced
        target_percentage_range = GRU_TARGET_PERCENTAGE_OPTIONS
    else:
        # Otherwise, use a focused range around the current best
        target_percentage_range = sorted(list(set([
            max(0.001, round(initial_target_percentage - 0.001, 4)),
            initial_target_percentage,
            round(initial_target_percentage + 0.001, 4)
        ])))
    
    # Determine class_horizon_range
    if force_thresholds_optimization: # Assuming force_thresholds_optimization implies a broader search for class_horizon too
        # Use the global options for a broader search if forced
        class_horizon_range = GRU_CLASS_HORIZON_OPTIONS
    else:
        # Otherwise, use a focused range around the current best
        class_horizon_range = sorted(list(set([
            max(1, initial_class_horizon - 1),
            initial_class_horizon,
            initial_class_horizon + 1
        ])))
    
    print(f"  [DEBUG] {current_process().name} - {ticker}: Loading prices for optimization...")
    df_backtest_opt = load_prices(ticker, train_start, train_end)
    if df_backtest_opt.empty:
        print(f"  [DEBUG] {current_process().name} - {ticker}: No data for optimization. Returning default.")
        return {'ticker': ticker, 'min_proba_buy': current_min_proba_buy, 'min_proba_sell': current_min_proba_sell, 'target_percentage': initial_target_percentage, 'class_horizon': initial_class_horizon, 'optimization_status': "Failed (no data)"}
    print(f"  [DEBUG] {current_process().name} - {ticker}: Prices loaded. Starting optimization loops.")

    models_cache = {}

    for p_target in target_percentage_range:
        for c_horizon in class_horizon_range:
            cache_key = (p_target, c_horizon)
            if cache_key not in models_cache:
                print(f"  [DEBUG] {current_process().name} - {ticker}: Re-fetching training data and re-training models for target_percentage={p_target:.4f}, class_horizon={c_horizon}")
                df_train, actual_feature_set = fetch_training_data(ticker, df_backtest_opt.copy(), p_target, c_horizon)
                if df_train.empty:
                    print(f"  [DEBUG] {current_process().name} - {ticker}: Insufficient training data for target_percentage={p_target:.4f}, class_horizon={c_horizon}. Skipping.")
                    continue
                
                global_models_and_params = initialize_ml_libraries()
                model_buy_for_opt, scaler_buy_for_opt, _ = train_and_evaluate_models(
                    df_train, "TargetClassBuy", actual_feature_set, ticker=ticker,
                    models_and_params_global=global_models_and_params,
                    perform_gru_hp_optimization=False, # Disable internal GRU HP opt
                    default_target_percentage=p_target, # Pass current p_target from outer loop
                    default_class_horizon=c_horizon # Pass current c_horizon from outer loop
                )
                model_sell_for_opt, scaler_sell_for_opt, _ = train_and_evaluate_models(
                    df_train, "TargetClassSell", actual_feature_set, ticker=ticker,
                    models_and_params_global=global_models_and_params,
                    perform_gru_hp_optimization=False, # Disable internal GRU HP opt
                    default_target_percentage=p_target, # Pass current p_target from outer loop
                    default_class_horizon=c_horizon # Pass current c_horizon from outer loop
                )
                
                if model_buy_for_opt and model_sell_for_opt and scaler_buy_for_opt:
                    models_cache[cache_key] = (model_buy_for_opt, model_sell_for_opt, scaler_buy_for_opt)
                else:
                    print(f"  [DEBUG] {current_process().name} - {ticker}: Failed to train models for target_percentage={p_target:.4f}, class_horizon={c_horizon}. Skipping.")
                    continue
            
            current_model_buy, current_model_sell, current_scaler = models_cache[cache_key]

            for p_buy in min_proba_buy_range:
                for p_sell in min_proba_sell_range:
                    print(f"  [DEBUG] {current_process().name} - {ticker}: Testing p_buy={p_buy:.2f}, p_sell={p_sell:.2f}, p_target={p_target:.4f}, c_horizon={c_horizon}")
                    
                    env = RuleTradingEnv(
                        df=df_backtest_opt.copy(),
                        ticker=ticker,
                        initial_balance=capital_per_stock,
                        transaction_cost=TRANSACTION_COST,
                        model_buy=current_model_buy,
                        model_sell=current_model_sell,
                        scaler=current_scaler,
                        per_ticker_min_proba_buy=p_buy,
                        per_ticker_min_proba_sell=p_sell,
                        use_gate=USE_MODEL_GATE,
                        feature_set=feature_set,
                        use_simple_rule_strategy=False
                    )
                    print(f"  [DEBUG] {current_process().name} - {ticker}: RuleTradingEnv initialized. Running env.run()...")
                    final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, _ = env.run()
                    print(f"  [DEBUG] {current_process().name} - {ticker}: env.run() completed. Getting final value.")
                    
                    # Calculate Sharpe Ratio for the current backtest run
                    strategy_history = pd.Series(env.portfolio_history)
                    strat_returns = strategy_history.pct_change(fill_method=None).dropna()
                    current_sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() > 0 else 0

                    print(f"  [DEBUG] {current_process().name} - [Opti] {ticker}: Buy={p_buy:.2f}, Sell={p_sell:.2f}, Target%={p_target:.4f}, CH={c_horizon} -> Sharpe={current_sharpe:.2f}")
                    
                    if current_sharpe > best_sharpe:
                        best_sharpe = current_sharpe
                        best_min_proba_buy = p_buy
                        best_min_proba_sell = p_sell
                        best_target_percentage = p_target
                        best_class_horizon = c_horizon
    
    optimization_status = "No Change"
    if not np.isclose(best_min_proba_buy, current_min_proba_buy) or \
       not np.isclose(best_min_proba_sell, current_min_proba_sell) or \
       not np.isclose(best_target_percentage, initial_target_percentage) or \
       not np.isclose(best_class_horizon, initial_class_horizon):
        optimization_status = "Optimized"

    print(f"  [DEBUG] {current_process().name} - {ticker}: Optimization complete. Best Sharpe={best_sharpe:.2f}, Status: {optimization_status}")
    return {
        'ticker': ticker,
        'min_proba_buy': best_min_proba_buy,
        'min_proba_sell': best_min_proba_sell,
        'target_percentage': best_target_percentage,
        'class_horizon': best_class_horizon,
        'best_sharpe': best_sharpe, # Changed from best_revenue to best_sharpe
        'optimization_status': optimization_status
    }

def _run_portfolio_backtest(
    all_tickers_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    capital_per_stock: float,
    target_percentage: float,
    run_parallel: bool,
    period_name: str,
    top_performers_data: List[Tuple], # Added top_performers_data
    use_simple_rule_strategy: bool = False # New parameter for simple rule strategy
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]: # Added Dict[str, List[float]] for buy_hold_histories_per_ticker
    """Helper function to run portfolio backtest for a given period."""
    num_processes = NUM_PROCESSES

    backtest_params = []
    for ticker in top_tickers:
        # Use optimized parameters if available, otherwise fall back to global defaults
        min_proba_buy_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_buy', MIN_PROBA_BUY)
        min_proba_sell_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_sell', MIN_PROBA_SELL)
        target_percentage_ticker = optimized_params_per_ticker.get(ticker, {}).get('target_percentage', target_percentage)

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
            feature_set_for_worker, min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data, use_simple_rule_strategy # Pass new parameter
        ))

    portfolio_values = []
    processed_tickers = []
    performance_metrics = []
    buy_hold_histories_per_ticker: Dict[str, List[float]] = {} # New: Store buy_hold_histories
    
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
                    buy_hold_histories_per_ticker[res['ticker']] = res.get('buy_hold_history', []) # Store history
                    
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
                buy_hold_histories_per_ticker[worker_result['ticker']] = worker_result.get('buy_hold_history', []) # Store history
                
                # Find the corresponding performance data (1Y and YTD from top_performers_data)
                perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                for t, p1y, pytd in top_performers_data:
                    if t == worker_result['ticker']:
                        perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                        ytd_perf_benchmark = pytd if np.isfinite(pytd) else np.nan
                        break
                
                # Print individual stock performance immediately
                print(f"\n📈 Individual Stock Performance for {worker_result['ticker']} ({period_name}):")
                print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                print(f"  - YTD Performance: {ytd_perf_benchmark:.2f}%" if pd.notna(ytd_perf_benchmark) else "  - YTD Performance: N/A")
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
    return final_portfolio_value, portfolio_values, processed_tickers, performance_metrics, buy_hold_histories_per_ticker

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
    num_tickers_analyzed: int,
    final_strategy_value_1month: float, # Added parameter
    ai_1month_return: float, # Added parameter
    final_buy_hold_value_1month: float, # Added parameter
    final_simple_rule_value_1y: float, # New parameter
    simple_rule_1y_return: float, # New parameter
    final_simple_rule_value_ytd: float, # New parameter
    simple_rule_ytd_return: float, # New parameter
    final_simple_rule_value_3month: float, # New parameter
    simple_rule_3month_return: float, # New parameter
    final_simple_rule_value_1month: float, # New parameter
    simple_rule_1month_return: float, # New parameter
    performance_metrics_simple_rule_1y: List[Dict], # New parameter for simple rule performance
    performance_metrics_buy_hold_1y: List[Dict], # New parameter for Buy & Hold performance
    top_performers_data: List[Tuple] # Add top_performers_data here
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}") # Use the passed initial_balance_used
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)") # New
    print(f"  1-Year Buy & Hold Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  YTD AI Strategy Value: ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)")
    print(f"  YTD Simple Rule Value: ${final_simple_rule_value_ytd:,.2f} ({simple_rule_ytd_return:+.2f}%)") # New
    print(f"  YTD Buy & Hold Value: ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  3-Month AI Strategy Value: ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)")
    print(f"  3-Month Simple Rule Value: ${final_simple_rule_value_3month:,.2f} ({simple_rule_3month_return:+.2f}%)") # New
    print(f"  3-Month Buy & Hold Value: ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  1-Month AI Strategy Value: ${final_strategy_value_1month:,.2f} ({ai_1month_return:+.2f}%)")
    print(f"  1-Month Simple Rule Value: ${final_simple_rule_value_1month:,.2f} ({simple_rule_1month_return:+.2f}%)") # New
    print(f"  1-Month Buy & Hold Value: ${final_buy_hold_value_1month:,.2f} ({((final_buy_hold_value_1month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 290)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Shares Before Liquidation':>25}")
    print("-" * 290)
    for res in sorted_final_results:
        # --- Safely get ticker and parameters ---
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', CLASS_HORIZON) # Get class_horizon
        opt_status = optimized_params.get('optimization_status', 'N/A') # Get optimization status

        # --- Calculate allocated capital and strategy gain ---
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('performance', 0.0) - allocated_capital

        # --- Safely format performance numbers ---
        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {shares_before_liquidation_str}")
    print("-" * 290)

    # --- Simple Rule Strategy Individual Ticker Performance ---
    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    # Sort simple rule results by 1Y performance for the table
    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital
        
        # Find the corresponding 1Y and YTD performance from top_performers_data
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data: # Assuming top_performers_data is available in this scope
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD')) # Renamed from last_ai_action to last_action for clarity
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 136)

    # --- Buy & Hold Strategy Individual Ticker Performance ---
    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    # Sort Buy & Hold results by 1Y performance for the table
    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
        
        # Find the corresponding 1Y and YTD performance from top_performers_data
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 136)

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
    class_horizon: int = CLASS_HORIZON, # New parameter for initial/default class_horizon for optimization
    force_thresholds_optimization: bool = FORCE_THRESHOLDS_OPTIMIZATION, # New parameter
    force_percentage_optimization: bool = FORCE_PERCENTAGE_OPTIMIZATION, # New parameter
    top_performers_data=None,
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True,
    single_ticker: Optional[str] = None,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None,
    use_simple_rule_strategy: bool = USE_SIMPLE_RULE_STRATEGY # New parameter for simple rule strategy
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[Dict]]:
    
    # Set the start method for multiprocessing to 'spawn'
    # This is crucial for CUDA compatibility with multiprocessing
    try:
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method set to 'spawn' for CUDA compatibility.")
    except RuntimeError as e:
        print(f"⚠️ Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA and multiprocessing.")

    end_date = datetime.now(timezone.utc)
    bt_end = end_date
    
    alpaca_trading_client = None

    # Initialize ML libraries to determine CUDA availability
    initialize_ml_libraries()
    
    # Disable parallel processing if deep learning models are used with CUDA
    if PYTORCH_AVAILABLE and CUDA_AVAILABLE and (USE_LSTM or USE_GRU):
        print("⚠️ CUDA is available and deep learning models are enabled.")
        run_parallel = True
    
    # Initialize initial_balance_used here with a default value
    initial_balance_used = INITIAL_BALANCE 
    print(f"Using initial balance: ${initial_balance_used:,.2f}")

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
                perf_ytd = ((end_date - start_price) / start_price) * 100
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

    # --- Fetch SPY data for Market Momentum feature ---
    print("🔍 Fetching SPY data for Market Momentum feature...")
    spy_df = load_prices_robust('SPY', earliest_date_needed, end_date)
    if not spy_df.empty:
        spy_df['SPY_Returns'] = spy_df['Close'].pct_change()
        spy_df['Market_Momentum_SPY'] = spy_df['SPY_Returns'].rolling(window=FEAT_VOL_WINDOW).mean()
        spy_df = spy_df[['Market_Momentum_SPY']]
        spy_df.columns = pd.MultiIndex.from_product([spy_df.columns, ['SPY']])
        
        # Merge SPY data into all_tickers_data
        all_tickers_data = all_tickers_data.merge(spy_df, left_index=True, right_index=True, how='left')
        # Forward fill and then back fill any NaNs introduced by the merge
        all_tickers_data['Market_Momentum_SPY', 'SPY'] = all_tickers_data['Market_Momentum_SPY', 'SPY'].ffill().bfill().fillna(0)
        print("✅ SPY Market Momentum data fetched and merged.")
    else:
        print("⚠️ Could not fetch SPY data. Market Momentum feature will be 0.")
        # Add a zero-filled column if SPY data couldn't be fetched
        all_tickers_data['Market_Momentum_SPY', 'SPY'] = 0.0

    # --- Fetch and merge intermarket data ---
    # --- Fetch and merge intermarket data ---
    print("🔍 Fetching intermarket data...")
    intermarket_df = _fetch_intermarket_data(earliest_date_needed, end_date)
    if not intermarket_df.empty:
        # Rename columns to include 'Intermarket' level for MultiIndex
        intermarket_df.columns = pd.MultiIndex.from_product([intermarket_df.columns, ['Intermarket']])
        all_tickers_data = all_tickers_data.merge(intermarket_df, left_index=True, right_index=True, how='left')
        # Forward fill and then back fill any NaNs introduced by the merge
        for col in intermarket_df.columns:
            all_tickers_data[col] = all_tickers_data[col].ffill().bfill().fillna(0)
        print("✅ Intermarket data fetched and merged.")
    else:
        print("⚠️ Could not fetch intermarket data. Intermarket features will be 0.")
        # Add zero-filled columns for intermarket features to ensure feature set consistency
        for col_name in ['Bond_Yield_Returns', 'Oil_Price_Returns', 'Gold_Price_Returns']: # DXY_Index_Returns removed from _fetch_intermarket_data
            if (col_name, 'Intermarket') not in all_tickers_data.columns:
                all_tickers_data[col_name, 'Intermarket'] = 0.0
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
        _ensure_dir(Path("logs")) # Ensure logs directory exists
        with open("logs/skipped_tickers.log", "w") as f:
            f.write("Tickers skipped during performance analysis:\n")
            for ticker in sorted(list(skipped_tickers)):
                f.write(f"{ticker}\n")

    # --- Training Models (for 1-Year Backtest) ---
    models_buy, models_sell, scalers = {}, {}, {}
    gru_hyperparams_buy_dict, gru_hyperparams_sell_dict = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_1y = {} # New: Store failed tickers and their reasons
    if ENABLE_1YEAR_TRAINING:
        print("🔍 Step 3: Training AI models for 1-Year backtest...")
        bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)
        train_end_1y = bt_start_1y - timedelta(days=1)
        train_start_1y_calc = train_end_1y - timedelta(days=TRAIN_LOOKBACK_DAYS)
        
        training_params_1y = []
        for ticker in top_tickers:
            # Load existing GRU hyperparams for this ticker if available
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_1y_calc:train_end_1y, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_1y.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 1-Year period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 1-Year models in parallel for {len(top_tickers)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_1y = list(tqdm(pool.imap(train_worker, training_params_1y), total=len(training_params_1y), desc="Training 1-Year Models"))
        else:
            print(f"🤖 Training 1-Year models sequentially for {len(top_tickers)} tickers...")
            training_results_1y = [train_worker(p) for p in tqdm(training_params_1y, desc="Training 1-Year Models")]

        for res in training_results_1y:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy[res['ticker']] = res['model_buy']
                models_sell[res['ticker']] = res['model_sell']
                scalers[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_1y[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 1-Year training loop, models_buy has {len(models_buy)} entries.")

        if not models_buy and USE_MODEL_GATE:
            print("⚠️ No models were trained for 1-Year backtest. Model-gating will be disabled for this run.\n")
    
    # Filter out failed tickers from top_tickers for subsequent steps
    top_tickers_1y_filtered = [t for t in top_tickers if t not in failed_training_tickers_1y]
    print(f"  ℹ️ {len(failed_training_tickers_1y)} tickers failed 1-Year model training and will be skipped: {', '.join(failed_training_tickers_1y.keys())}")
    
    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1y_filtered = [item for item in top_performers_data if item[0] in top_tickers_1y_filtered]
    
    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1y = INVESTMENT_PER_STOCK
    
    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_1y_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1y_filtered}
    else:
        optimized_params_per_ticker_1y_filtered = {}
    
    
    # --- OPTIMIZE THRESHOLDS ---
    # Ensure logs directory exists for optimized parameters
    _ensure_dir(TOP_CACHE_PATH.parent)
    optimized_params_file = TOP_CACHE_PATH.parent / "optimized_per_ticker_params.json"
    
    # If force_thresholds_optimization is True and the file exists, delete it to force re-optimization
    if force_thresholds_optimization and optimized_params_file.exists():
        try:
            os.remove(optimized_params_file)
            print(f"🗑️ Deleted existing optimized parameters file: {optimized_params_file} to force re-optimization.")
        except Exception as e:
            print(f"⚠️ Could not delete optimized parameters file: {e}")

    optimized_params_per_ticker = {}
    loaded_optimized_params = {}

    # Try to load existing optimized parameters if not forcing re-optimization
    if optimized_params_file.exists():
        try:
            with open(optimized_params_file, 'r') as f:
                loaded_optimized_params = json.load(f)
            print(f"\n✅ Loaded existing optimized parameters from {optimized_params_file}.")
        except Exception as e:
            print(f"⚠️ Could not load optimized parameters from file: {e}. Starting with default thresholds.")

    # Determine if optimization needs to run at all
    should_run_optimization = force_thresholds_optimization or force_percentage_optimization

    if should_run_optimization:
        print("\n🔄 Step 2.5: Optimizing ML parameters for each ticker...")
        optimized_params_per_ticker = optimize_thresholds_for_portfolio(
            top_tickers=top_tickers_1y_filtered,
            train_start=train_start_1y,
            train_end=train_end_1y,
            default_target_percentage=target_percentage,
            default_class_horizon=class_horizon, # Pass class_horizon here
            feature_set=feature_set,
            models_buy=models_buy,
            models_sell=models_sell,
            scalers=scalers,
            capital_per_stock=capital_per_stock_1y,
            run_parallel=run_parallel,
            force_percentage_optimization=force_percentage_optimization,
            force_thresholds_optimization=force_thresholds_optimization, # Pass this parameter
            current_optimized_params_per_ticker=loaded_optimized_params
        )
        if optimized_params_per_ticker:
            try:
                with open(optimized_params_file, 'w') as f:
                    json.dump(optimized_params_per_ticker, f, indent=4)
                print(f"✅ Optimized parameters saved to {optimized_params_file}")
            except Exception as e:
                print(f"⚠️ Could not save optimized parameters to file: {e}")
    else:
        # If no optimization is forced, load existing or use defaults
        optimized_params_per_ticker = {}
        for ticker in top_tickers_1y_filtered:
            if ticker in loaded_optimized_params:
                optimized_params_per_ticker[ticker] = loaded_optimized_params[ticker]
                optimized_params_per_ticker[ticker]['optimization_status'] = "Loaded"
            else:
                optimized_params_per_ticker[ticker] = {
                    'min_proba_buy': MIN_PROBA_BUY,
                    'min_proba_sell': MIN_PROBA_SELL,
                    'target_percentage': target_percentage,
                    'optimization_status': "Not Optimized (using defaults)"
                }
        print(f"\n✅ Using loaded or default parameters (set 'force_thresholds_optimization=True' or 'force_percentage_optimization=True' in main() call to re-run optimization).")
        if not optimized_params_per_ticker:
            print("\nℹ️ No optimized parameters found for current tickers. Using default thresholds.")


    # Initialize all backtest related variables to default values
    final_strategy_value_1y = initial_balance_used
    strategy_results_1y = []
    processed_tickers_1y = []
    performance_metrics_1y = []
    ai_1y_return = 0.0
    final_simple_rule_value_1y = initial_balance_used
    simple_rule_results_1y = []
    processed_simple_rule_tickers_1y = []
    performance_metrics_simple_rule_1y = []
    simple_rule_1y_return = 0.0
    final_buy_hold_value_1y = initial_balance_used
    buy_hold_results_1y = []
    performance_metrics_buy_hold_1y_actual = [] # Initialize here

    final_strategy_value_ytd = initial_balance_used
    strategy_results_ytd = []
    processed_tickers_ytd_local = []
    performance_metrics_ytd = []
    ai_ytd_return = 0.0
    final_simple_rule_value_ytd = initial_balance_used
    simple_rule_results_ytd = []
    processed_simple_rule_tickers_ytd = []
    performance_metrics_simple_rule_ytd = []
    simple_rule_ytd_return = 0.0
    final_buy_hold_value_ytd = initial_balance_used
    buy_hold_results_ytd = []
    performance_metrics_buy_hold_ytd_actual = [] # Initialize here

    final_strategy_value_3month = initial_balance_used
    strategy_results_3month = []
    processed_tickers_3month_local = []
    performance_metrics_3month = []
    ai_3month_return = 0.0
    final_simple_rule_value_3month = initial_balance_used
    simple_rule_results_3month = []
    processed_simple_rule_tickers_3month = []
    performance_metrics_simple_rule_3month = []
    simple_rule_3month_return = 0.0
    final_buy_hold_value_3month = initial_balance_used
    buy_hold_results_3month = []
    performance_metrics_buy_hold_3month_actual = [] # Initialize here

    final_strategy_value_1month = initial_balance_used
    strategy_results_1month = []
    processed_tickers_1month_local = []
    performance_metrics_1month = []
    ai_1month_return = 0.0
    final_simple_rule_value_1month = initial_balance_used
    simple_rule_results_1month = []
    processed_simple_rule_tickers_1month = []
    performance_metrics_simple_rule_1month = []
    simple_rule_1month_return = 0.0
    final_buy_hold_value_1month = initial_balance_used
    buy_hold_results_1month = []
    performance_metrics_buy_hold_1month_actual = [] # Initialize here
    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here
    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here

    # --- Run 1-Year Backtest ---
    if ENABLE_1YEAR_BACKTEST:
        print("\n🔍 Step 4: Running 1-Year Backtest...")
        # --- Run 1-Year Backtest (AI Strategy) ---
        print("\n🔍 Step 4: Running 1-Year Backtest (AI Strategy)...")
        final_strategy_value_1y, strategy_results_1y, processed_tickers_1y, performance_metrics_1y, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1y,
            end_date=bt_end,
            top_tickers=top_tickers_1y_filtered, # Use filtered tickers for backtest
            models_buy=models_buy,
            models_sell=models_sell,
            scalers=scalers,
            optimized_params_per_ticker=optimized_params_per_ticker,
            capital_per_stock=capital_per_stock_1y, # Use fixed capital per stock
            # Pass the global target_percentage here, as the individual backtest_worker will use the optimized one
            target_percentage=target_percentage, 
            run_parallel=run_parallel,
            period_name="1-Year (AI)",
            top_performers_data=top_performers_data_1y_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_1y_return = ((final_strategy_value_1y - (capital_per_stock_1y * len(top_tickers_1y_filtered))) / abs(capital_per_stock_1y * len(top_tickers_1y_filtered))) * 100 if (capital_per_stock_1y * len(top_tickers_1y_filtered)) != 0 else 0

        # --- Run 1-Year Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 1-Year Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_1y, simple_rule_results_1y, processed_simple_rule_tickers_1y, performance_metrics_simple_rule_1y, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1y,
            end_date=bt_end,
            top_tickers=top_tickers_1y_filtered, # Use filtered tickers for backtest
            models_buy={}, # No ML models for simple rule strategy
            models_sell={}, # No ML models for simple rule strategy
            scalers={}, # No scalers for simple rule strategy
            optimized_params_per_ticker={}, # No optimized params for simple rule strategy
            capital_per_stock=capital_per_stock_1y,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="1-Year (Simple Rule)",
            top_performers_data=top_performers_data_1y_filtered,
            use_simple_rule_strategy=True # Explicitly set to True for simple rule strategy
        )
        simple_rule_1y_return = ((final_simple_rule_value_1y - (capital_per_stock_1y * len(top_tickers_1y_filtered))) / abs(capital_per_stock_1y * len(top_tickers_1y_filtered))) * 100 if (capital_per_stock_1y * len(top_tickers_1y_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 1-Year ---
        print("\n📊 Calculating Buy & Hold performance for 1-Year period...")
        buy_hold_results_1y = []
        performance_metrics_buy_hold_1y_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_1y_filtered: # Iterate over filtered tickers
            df_bh = load_prices_robust(ticker, bt_start_1y, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_1y / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_1y - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_1y
                buy_hold_results_1y.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_1y_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_1y) / abs(capital_per_stock_1y)) * 100 if capital_per_stock_1y != 0 else 0.0,
                    'final_shares': shares_bh # Add this line
                })
            else:
                buy_hold_results_1y.append(capital_per_stock_1y)
                performance_metrics_buy_hold_1y_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_1y,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0,
                    'final_shares': 0.0 # Add this line
                })
        final_buy_hold_value_1y = sum(buy_hold_results_1y) + (len(top_tickers_1y_filtered) - len(buy_hold_results_1y)) * capital_per_stock_1y
        print("✅ 1-Year Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 1-Year Backtest is disabled by ENABLE_1YEAR_BACKTEST flag.")


    # --- Training Models (for YTD Backtest) ---
    models_buy_ytd, models_sell_ytd, scalers_ytd = {}, {}, {}
    gru_hyperparams_buy_dict_ytd, gru_hyperparams_sell_dict_ytd = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_ytd = {} # New: Store failed tickers and their reasons
    if ENABLE_YTD_TRAINING:
        print("\n🔍 Step 5: Training AI models for YTD backtest...")
        ytd_start_date = datetime(bt_end.year, 1, 1, tzinfo=timezone.utc)
        train_end_ytd = ytd_start_date - timedelta(days=1)
        train_start_ytd = train_end_ytd - timedelta(days=TRAIN_LOOKBACK_DAYS)
        
        training_params_ytd = []
        for ticker in top_tickers_1y_filtered: # Use filtered tickers for YTD training
            # Load existing GRU hyperparams for this ticker if available
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_ytd:train_end_ytd, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_ytd.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for YTD period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training YTD models in parallel for {len(top_tickers_1y_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_ytd = list(tqdm(pool.imap(train_worker, training_params_ytd), total=len(training_params_ytd), desc="Training YTD Models"))
        else:
            print(f"🤖 Training YTD models sequentially for {len(top_tickers_1y_filtered)} tickers...")
            training_results_ytd = [train_worker(p) for p in tqdm(training_params_ytd, desc="Training YTD Models")]

        for res in training_results_ytd:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_ytd[res['ticker']] = res['model_buy']
                models_sell_ytd[res['ticker']] = res['model_sell']
                scalers_ytd[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_ytd[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_ytd[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_ytd[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After YTD training loop, models_buy_ytd has {len(models_buy_ytd)} entries.")

        if not models_buy_ytd and USE_MODEL_GATE:
            print("⚠️ No models were trained for YTD backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_1y_filtered for subsequent steps
    top_tickers_ytd_filtered = [t for t in top_tickers_1y_filtered if t not in failed_training_tickers_ytd]
    print(f"  ℹ️ {len(failed_training_tickers_ytd)} tickers failed YTD model training and will be skipped: {', '.join(failed_training_tickers_ytd.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_ytd_filtered = [item for item in top_performers_data_1y_filtered if item[0] in top_tickers_ytd_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_ytd = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_ytd_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_ytd_filtered}
    else:
        optimized_params_per_ticker_ytd_filtered = {}

    # --- Run YTD Backtest ---
    if ENABLE_YTD_BACKTEST:
        print("\n🔍 Step 6: Running YTD Backtest...")
        # --- Run YTD Backtest (AI Strategy) ---
        print("\n🔍 Step 6: Running YTD Backtest (AI Strategy)...")
        final_strategy_value_ytd, strategy_results_ytd, processed_tickers_ytd_local, performance_metrics_ytd, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=ytd_start_date,
            end_date=bt_end,
            top_tickers=top_tickers_ytd_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_ytd,
            models_sell=models_sell_ytd,
            scalers=scalers_ytd,
            optimized_params_per_ticker=optimized_params_per_ticker_ytd_filtered,
            capital_per_stock=capital_per_stock_ytd, # Use fixed capital per stock
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="YTD (AI)",
            top_performers_data=top_performers_data_ytd_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_ytd_return = ((final_strategy_value_ytd - (capital_per_stock_ytd * len(top_tickers_ytd_filtered))) / abs(capital_per_stock_ytd * len(top_tickers_ytd_filtered))) * 100 if (capital_per_stock_ytd * len(top_tickers_ytd_filtered)) != 0 else 0

        # --- Run YTD Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running YTD Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_ytd, simple_rule_results_ytd, processed_simple_rule_tickers_ytd, performance_metrics_simple_rule_ytd, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=ytd_start_date,
            end_date=bt_end,
            top_tickers=top_tickers_ytd_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_ytd,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="YTD (Simple Rule)",
            top_performers_data=top_performers_data_ytd_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_ytd_return = ((final_simple_rule_value_ytd - (capital_per_stock_ytd * len(top_tickers_ytd_filtered))) / abs(capital_per_stock_ytd * len(top_tickers_ytd_filtered))) * 100 if (capital_per_stock_ytd * len(top_tickers_ytd_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for YTD ---
        print("\n📊 Calculating Buy & Hold performance for YTD period...")
        buy_hold_results_ytd = []
        performance_metrics_buy_hold_ytd_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_ytd_filtered: # Iterate over filtered tickers
            df_bh = load_prices_robust(ticker, ytd_start_date, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_ytd / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_ytd - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_ytd
                buy_hold_results_ytd.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_ytd_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_ytd) / abs(capital_per_stock_ytd)) * 100 if capital_per_stock_ytd != 0 else 0.0
                })
            else:
                buy_hold_results_ytd.append(capital_per_stock_ytd)
                performance_metrics_buy_hold_ytd_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_ytd,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_ytd = sum(buy_hold_results_ytd) + (len(top_tickers_ytd_filtered) - len(buy_hold_results_ytd)) * capital_per_stock_ytd
        print("✅ YTD Buy & Hold calculation complete.")
    else:
        print("\nℹ️ YTD Backtest is disabled by ENABLE_YTD_BACKTEST flag.")

    # --- Training Models (for 3-Month Backtest) ---
    models_buy_3month, models_sell_3month, scalers_3month = {}, {}, {}
    gru_hyperparams_buy_dict_3month, gru_hyperparams_sell_dict_3month = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_3month = {} # New: Store failed tickers and their reasons
    if ENABLE_3MONTH_TRAINING:
        print("\n🔍 Step 7: Training AI models for 3-Month backtest...")
        bt_start_3month = bt_end - timedelta(days=BACKTEST_DAYS_3MONTH)
        train_end_3month = bt_start_3month - timedelta(days=1)
        train_start_3month = train_end_3month - timedelta(days=TRAIN_LOOKBACK_DAYS)

        training_params_3month = []
        for ticker in top_tickers_ytd_filtered: # Use filtered tickers for 3-Month training
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_3month:train_end_3month, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_3month.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 3-Month period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 3-Month models in parallel for {len(top_tickers_ytd_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_3month = list(tqdm(pool.imap(train_worker, training_params_3month), total=len(training_params_3month), desc="Training 3-Month Models"))
        else:
            print(f"🤖 Training 3-Month models sequentially for {len(top_tickers_ytd_filtered)} tickers...")
            training_results_3month = [train_worker(p) for p in tqdm(training_params_3month, desc="Training 3-Month Models")]

        for res in training_results_3month:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_3month[res['ticker']] = res['model_buy']
                models_sell_3month[res['ticker']] = res['model_sell']
                scalers_3month[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_3month[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_3month[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_3month[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 3-Month training loop, models_buy_3month has {len(models_buy_3month)} entries.")

        if not models_buy_3month and USE_MODEL_GATE:
            print("⚠️ No models were trained for 3-Month backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_ytd_filtered for subsequent steps
    top_tickers_3month_filtered = [t for t in top_tickers_ytd_filtered if t not in failed_training_tickers_3month]
    print(f"  ℹ️ {len(failed_training_tickers_3month)} tickers failed 3-Month model training and will be skipped: {', '.join(failed_training_tickers_3month.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_3month_filtered = [item for item in top_performers_data_ytd_filtered if item[0] in top_tickers_3month_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_3month = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_3month_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_3month_filtered}
    else:
        optimized_params_per_ticker_3month_filtered = {}

    # --- Run 3-Month Backtest ---
    if ENABLE_3MONTH_BACKTEST:
        print("\n🔍 Step 8: Running 3-Month Backtest...")
        # --- Run 3-Month Backtest (AI Strategy) ---
        print("\n🔍 Step 8: Running 3-Month Backtest (AI Strategy)...")
        final_strategy_value_3month, strategy_results_3month, processed_tickers_3month_local, performance_metrics_3month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_3month,
            end_date=bt_end,
            top_tickers=top_tickers_3month_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_3month,
            models_sell=models_sell_3month,
            scalers=scalers_3month,
            optimized_params_per_ticker=optimized_params_per_ticker_3month_filtered,
            capital_per_stock=capital_per_stock_3month, # Use fixed capital per stock
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="3-Month (AI)",
            top_performers_data=top_performers_data_3month_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_3month_return = ((final_strategy_value_3month - (capital_per_stock_3month * len(top_tickers_3month_filtered))) / abs(capital_per_stock_3month * len(top_tickers_3month_filtered))) * 100 if (capital_per_stock_3month * len(top_tickers_3month_filtered)) != 0 else 0

        # --- Run 3-Month Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 3-Month Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_3month, simple_rule_results_3month, processed_simple_rule_tickers_3month, performance_metrics_simple_rule_3month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_3month,
            end_date=bt_end,
            top_tickers=top_tickers_3month_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_3month,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="3-Month (Simple Rule)",
            top_performers_data=top_performers_data_3month_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_3month_return = ((final_simple_rule_value_3month - (capital_per_stock_3month * len(top_tickers_3month_filtered))) / abs(capital_per_stock_3month * len(top_tickers_3month_filtered))) * 100 if (capital_per_stock_3month * len(top_tickers_3month_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 3-Month ---
        print("\n📊 Calculating Buy & Hold performance for 3-Month period...")
        buy_hold_results_3month = []
        performance_metrics_buy_hold_3month_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_3month_filtered:
            df_bh = load_prices_robust(ticker, bt_start_3month, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_3month / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_3month - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_3month
                buy_hold_results_3month.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_3month_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_3month) / abs(capital_per_stock_3month)) * 100 if capital_per_stock_3month != 0 else 0.0
                })
            else:
                buy_hold_results_3month.append(capital_per_stock_3month)
                performance_metrics_buy_hold_3month_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_3month,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_3month = sum(buy_hold_results_3month) + (len(top_tickers_3month_filtered) - len(buy_hold_results_3month)) * capital_per_stock_3month
        print("✅ 3-Month Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 3-Month Backtest is disabled by ENABLE_3MONTH_BACKTEST flag.")

    # --- Training Models (for 1-Month Backtest) ---
    models_buy_1month, models_sell_1month, scalers_1month = {}, {}, {}
    failed_training_tickers_1month = {} # New: Store failed tickers and their reasons
    if ENABLE_1MONTH_TRAINING:
        print("\n🔍 Step 9: Training AI models for 1-Month backtest...")
        bt_start_1month = bt_end - timedelta(days=BACKTEST_DAYS_1MONTH)
        train_end_1month = bt_start_1month - timedelta(days=1)
        train_start_1month = train_end_1month - timedelta(days=TRAIN_LOOKBACK_DAYS)

        training_params_1month = []
        for ticker in top_tickers_3month_filtered: # Use filtered tickers for 1-Month training
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_1month:train_end_1month, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_1month.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 1-Month period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 1-Month models in parallel for {len(top_tickers_3month_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_1month = list(tqdm(pool.imap(train_worker, training_params_1month), total=len(training_params_1month), desc="Training 1-Month Models"))
        else:
            print(f"🤖 Training 1-Month models sequentially for {len(top_tickers_3month_filtered)} tickers...")
            training_results_1month = [train_worker(p) for p in tqdm(training_params_1month, desc="Training 1-Month Models")]

        for res in training_results_1month:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_1month[res['ticker']] = res['model_buy']
                models_sell_1month[res['ticker']] = res['model_sell']
                scalers_1month[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_1month[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_1month[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_1month[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 1-Month training loop, models_buy_1month has {len(models_buy_1month)} entries.")

        if not models_buy_1month and USE_MODEL_GATE:
            print("⚠️ No models were trained for 1-Month backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_3month_filtered for subsequent steps
    top_tickers_1month_filtered = [t for t in top_tickers_3month_filtered if t not in failed_training_tickers_1month]
    print(f"  ℹ️ {len(failed_training_tickers_1month)} tickers failed 1-Month model training and will be skipped: {', '.join(failed_training_tickers_1month.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1month_filtered = [item for item in top_performers_data_3month_filtered if item[0] in top_tickers_1month_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1month = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_1month_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1month_filtered}
    else:
        optimized_params_per_ticker_1month_filtered = {}

    # --- Run 1-Month Backtest ---
    if ENABLE_1MONTH_BACKTEST:
        print("\n🔍 Step 10: Running 1-Month Backtest...")
        # --- Run 1-Month Backtest (AI Strategy) ---
        print("\n🔍 Step 10: Running 1-Month Backtest (AI Strategy)...")
        final_strategy_value_1month, strategy_results_1month, processed_tickers_1month_local, performance_metrics_1month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1month,
            end_date=bt_end,
            top_tickers=top_tickers_1month_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_1month,
            models_sell=models_sell_1month,
            scalers=scalers_1month,
            optimized_params_per_ticker=optimized_params_per_ticker_1month_filtered,
            capital_per_stock=capital_per_stock_1month, # Use fixed capital per stock
            target_percentage=target_percentage, 
            run_parallel=run_parallel,
            period_name="1-Month (AI)",
            top_performers_data=top_performers_data_1month_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_1month_return = ((final_strategy_value_1month - (capital_per_stock_1month * len(top_tickers_1month_filtered))) / abs(capital_per_stock_1month * len(top_tickers_1month_filtered))) * 100 if (capital_per_stock_1month * len(top_tickers_1month_filtered)) != 0 else 0

        # --- Run 1-Month Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 1-Month Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_1month, simple_rule_results_1month, processed_simple_rule_tickers_1month, performance_metrics_simple_rule_1month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1month,
            end_date=bt_end,
            top_tickers=top_tickers_1month_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_1month,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="1-Month (Simple Rule)",
            top_performers_data=top_performers_data_1month_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_1month_return = ((final_simple_rule_value_1month - (capital_per_stock_1month * len(top_tickers_1month_filtered))) / abs(capital_per_stock_1month * len(top_tickers_1month_filtered))) * 100 if (capital_per_stock_1month * len(top_tickers_1month_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 1-Month ---
        print("\n📊 Calculating Buy & Hold performance for 1-Month period...")
        buy_hold_results_1month = []
        performance_metrics_buy_hold_1month_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_1month_filtered:
            df_bh = load_prices_robust(ticker, bt_start_1month, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_1month / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_1month - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_1month
                buy_hold_results_1month.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_1month_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_1month) / abs(capital_per_stock_1month)) * 100 if capital_per_stock_1month != 0 else 0.0
                })
            else:
                buy_hold_results_1month.append(capital_per_stock_1month)
                performance_metrics_buy_hold_1month_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_1month,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_1month = sum(buy_hold_results_1month) + (len(top_tickers_1month_filtered) - len(buy_hold_results_1month)) * capital_per_stock_1month
        print("✅ 1-Month Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 1-Month Backtest is disabled by ENABLE_1MONTH_BACKTEST flag.")

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    print("\n📝 Preparing final summary data...")
    final_results = []
    
    # Combine all failed tickers from all periods
    all_failed_tickers = {}
    all_failed_tickers.update(failed_training_tickers_1y)
    all_failed_tickers.update(failed_training_tickers_ytd)
    all_failed_tickers.update(failed_training_tickers_3month)
    all_failed_tickers.update(failed_training_tickers_1month) # Add 1-month failed tickers

    # Add successfully processed tickers
    for i, ticker in enumerate(processed_tickers_1y):
        backtest_result_for_ticker = next((res for res in performance_metrics_1y if res['ticker'] == ticker), None)
        
        if backtest_result_for_ticker:
            perf_data = backtest_result_for_ticker['perf_data']
            individual_bh_return = backtest_result_for_ticker['individual_bh_return']
            last_ai_action = backtest_result_for_ticker['last_ai_action']
            buy_prob = backtest_result_for_ticker['buy_prob']
            sell_prob = backtest_result_for_ticker['sell_prob']
            final_shares = backtest_result_for_ticker['shares_before_liquidation'] # This is where final_shares is retrieved
        else:
            perf_data = {'sharpe_ratio': 0.0}
            individual_bh_return = 0.0
            last_ai_action = "N/A"
            buy_prob = 0.0
            sell_prob = 0.0
            final_shares = 0.0 # Set to 0.0 for tickers that didn't have a backtest result

        perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                ytd_perf_benchmark = pytd if np.isfinite(pytd) else np.nan
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': strategy_results_1y[i],
            'sharpe': perf_data['sharpe_ratio'],
            'one_year_perf': perf_1y_benchmark,
            'ytd_perf': ytd_perf_benchmark,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'final_shares': final_shares, # Add final_shares here
            'status': 'trained',
            'reason': None
        })
    
    # Add failed tickers to the final results
    for ticker, reason in all_failed_tickers.items():
        final_results.append({
            'ticker': ticker,
            'performance': INVESTMENT_PER_STOCK, # Assign initial capital for failed tickers
            'sharpe': np.nan,
            'one_year_perf': np.nan,
            'ytd_perf': np.nan,
            'individual_bh_return': np.nan,
            'last_ai_action': "FAILED",
            'buy_prob': np.nan,
            'sell_prob': np.nan,
            'final_shares': 0.0, # Set to 0.0 for failed tickers
            'status': 'failed',
            'reason': reason
        })

    # Sort by 1Y performance for the final table, handling potential NaN values
    sorted_final_results = sorted(final_results, key=lambda x: x.get('one_year_perf', -np.inf) if pd.notna(x.get('one_year_perf')) else -np.inf, reverse=True)
    
    print_final_summary(
        sorted_final_results, models_buy, models_sell, scalers, optimized_params_per_ticker,
        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
        final_strategy_value_ytd, final_buy_hold_value_ytd, ai_ytd_return,
        final_strategy_value_3month, final_buy_hold_value_3month, ai_3month_return,
        (INVESTMENT_PER_STOCK * len(top_tickers)),
        len(top_tickers),
        final_strategy_value_1month,
        ai_1month_return,
        final_buy_hold_value_1month,
        final_simple_rule_value_1y,
        simple_rule_1y_return,
        final_simple_rule_value_ytd,
        simple_rule_ytd_return,
        final_simple_rule_value_3month,
        simple_rule_3month_return,
        final_simple_rule_value_1month,
        simple_rule_1month_return,
        performance_metrics_simple_rule_1y,
        performance_metrics_buy_hold_1y_actual, # Pass performance_metrics_buy_hold_1y_actual for Buy & Hold
        top_performers_data
    )
    print("\n✅ Final summary prepared and printed.")

    # --- Select and save best performing models for live trading ---
    # Determine which period had the highest portfolio return
    performance_values = {
        "1-Year": final_strategy_value_1y,
        "YTD": final_strategy_value_ytd,
        "3-Month": final_strategy_value_3month,
        "1-Month": final_strategy_value_1month # Include 1-Month performance
    }
    
    best_period_name = max(performance_values, key=performance_values.get)
    
    # Get the models and scalers corresponding to the best period
    if best_period_name == "1-Year":
        best_models_buy_dict = models_buy
        best_models_sell_dict = models_sell
        best_scalers_dict = scalers
    elif best_period_name == "YTD":
        best_models_buy_dict = models_buy_ytd
        best_models_sell_dict = models_sell_ytd
        best_scalers_dict = scalers_ytd
    elif best_period_name == "3-Month":
        best_models_buy_dict = models_buy_3month
        best_models_sell_dict = models_sell_3month
        best_scalers_dict = scalers_3month
    else: # "1-Month"
        best_models_buy_dict = models_buy_1month
        best_models_sell_dict = models_sell_1month
        best_scalers_dict = scalers_1month

    # Save the best models and scalers for each ticker to the paths used by live_trading.py
    models_dir = Path("logs/models")
    _ensure_dir(models_dir) # Ensure the directory exists

    print(f"\n🏆 Saving best performing models for live trading from {best_period_name} period...")

    for ticker in best_models_buy_dict.keys():
        try:
            joblib.dump(best_models_buy_dict[ticker], models_dir / f"{ticker}_model_buy.joblib")
            joblib.dump(best_models_sell_dict[ticker], models_dir / f"{ticker}_model_sell.joblib")
            joblib.dump(best_scalers_dict[ticker], models_dir / f"{ticker}_scaler.joblib")
            print(f"  ✅ Saved models for {ticker} from {best_period_name} period.")
        except Exception as e:
            print(f"  ⚠️ Error saving models for {ticker} from {best_period_name} period: {e}")

    best_class_horizon = initial_class_horizon

    # Define parameter distributions for RandomizedSearchCV for thresholds
    # Use uniform distribution for probabilities (0.0 to 1.0)
    min_proba_buy_dist = uniform(loc=0.0, scale=1.0)
    min_proba_sell_dist = uniform(loc=0.0, scale=1.0)

    # Determine target_percentage_range
    if force_percentage_optimization:
        target_percentage_range = GRU_TARGET_PERCENTAGE_OPTIONS
    else:
        target_percentage_range = sorted(list(set([
            max(0.001, round(initial_target_percentage - 0.001, 4)),
            initial_target_percentage,
            round(initial_target_percentage + 0.001, 4)
        ])))
    
    # Determine class_horizon_range
    if force_thresholds_optimization: # Assuming force_thresholds_optimization implies a broader search for class_horizon too
        class_horizon_range = GRU_CLASS_HORIZON_OPTIONS
    else:
        class_horizon_range = sorted(list(set([
            max(1, initial_class_horizon - 1),
            initial_class_horizon,
            initial_class_horizon + 1
        ])))
    
    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Loading prices for optimization...\n")
    df_backtest_opt = load_prices(ticker, train_start, train_end)
    if df_backtest_opt.empty:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: No data for optimization. Returning default.\n")
        return {'ticker': ticker, 'min_proba_buy': current_min_proba_buy, 'min_proba_sell': current_min_proba_sell, 'target_percentage': initial_target_percentage, 'class_horizon': initial_class_horizon, 'optimization_status': "Failed (no data)"}
    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Prices loaded. Starting optimization loops.\n")

    models_cache = {}
    n_threshold_trials = 10 # Number of random trials for thresholds per (p_target, c_horizon) combination

    for p_target in target_percentage_range:
        for c_horizon in class_horizon_range:
            cache_key = (p_target, c_horizon)
            if cache_key not in models_cache:
                sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Re-fetching training data and re-training models for target_percentage={p_target:.4f}, class_horizon={c_horizon}\n")
                df_train, actual_feature_set = fetch_training_data(ticker, df_backtest_opt.copy(), p_target, c_horizon)
                if df_train.empty:
                    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Insufficient training data for target_percentage={p_target:.4f}, class_horizon={c_horizon}. Skipping.\n")
                    continue
                
                global_models_and_params = initialize_ml_libraries()
                model_buy_for_opt, scaler_buy_for_opt, _ = train_and_evaluate_models(
                    df_train, "TargetClassBuy", actual_feature_set, ticker=ticker,
                    models_and_params_global=global_models_and_params,
                    perform_gru_hp_optimization=False, # Disable internal GRU HP opt
                    default_target_percentage=p_target, # Pass current p_target from outer loop
                    default_class_horizon=c_horizon # Pass current c_horizon from outer loop
                )
                model_sell_for_opt, scaler_sell_for_opt, _ = train_and_evaluate_models(
                    df_train, "TargetClassSell", actual_feature_set, ticker=ticker,
                    models_and_params_global=global_models_and_params,
                    perform_gru_hp_optimization=False, # Disable internal GRU HP opt
                    default_target_percentage=p_target, # Pass current p_target from outer loop
                    default_class_horizon=c_horizon # Pass current c_horizon from outer loop
                )
                
                if model_buy_for_opt and model_sell_for_opt and scaler_buy_for_opt:
                    models_cache[cache_key] = (model_buy_for_opt, model_sell_for_opt, scaler_buy_for_opt)
                else:
                    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Failed to train models for target_percentage={p_target:.4f}, class_horizon={c_horizon}. Skipping.\n")
                    continue
            
            current_model_buy, current_model_sell, current_scaler = models_cache[cache_key]

            for trial_idx in range(n_threshold_trials):
                # Randomly sample min_proba_buy and min_proba_sell
                p_buy = round(min_proba_buy_dist.rvs(random_state=SEED + trial_idx), 2)
                p_sell = round(min_proba_sell_dist.rvs(random_state=SEED + trial_idx + n_threshold_trials), 2)

                sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Testing p_buy={p_buy:.2f}, p_sell={p_sell:.2f}, p_target={p_target:.4f}, c_horizon={c_horizon} (Trial {trial_idx+1}/{n_threshold_trials})\n")
                    
                env = RuleTradingEnv(
                    df=df_backtest_opt.copy(),
                    ticker=ticker,
                    initial_balance=capital_per_stock,
                    transaction_cost=TRANSACTION_COST,
                    model_buy=current_model_buy,
                    model_sell=current_model_sell,
                    scaler=current_scaler,
                    per_ticker_min_proba_buy=p_buy,
                    per_ticker_min_proba_sell=p_sell,
                    use_gate=USE_MODEL_GATE,
                    feature_set=feature_set,
                    use_simple_rule_strategy=False
                )
                sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: RuleTradingEnv initialized. Running env.run()...\n")
                final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, _ = env.run()
                sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: env.run() completed. Getting final value.\n")
                
                current_revenue = final_val

                sys.stderr.write(f"  [DEBUG] {current_process().name} - [Opti] {ticker}: Buy={p_buy:.2f}, Sell={p_sell:.2f}, Target%={p_target:.4f}, CH={c_horizon} -> Revenue=${current_revenue:,.2f}\n")
                
                if current_revenue > best_revenue:
                    best_revenue = current_revenue
                    best_min_proba_buy = p_buy
                    best_min_proba_sell = p_sell
                    best_target_percentage = p_target
                    best_class_horizon = c_horizon
    
    optimization_status = "No Change"
    if not np.isclose(best_min_proba_buy, current_min_proba_buy) or \
       not np.isclose(best_min_proba_sell, current_min_proba_sell) or \
       not np.isclose(best_target_percentage, initial_target_percentage) or \
       not np.isclose(best_class_horizon, initial_class_horizon):
        optimization_status = "Optimized"

    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Optimization complete. Best Revenue=${best_revenue:,.2f}, Status: {optimization_status}\n")
    return {
        'ticker': ticker,
        'min_proba_buy': best_min_proba_buy,
        'min_proba_sell': best_min_proba_sell,
        'target_percentage': best_target_percentage,
        'class_horizon': best_class_horizon,
        'best_revenue': best_revenue, # Changed from best_sharpe to best_revenue
        'optimization_status': optimization_status
    }

def _run_portfolio_backtest(
    all_tickers_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    top_tickers: List[str],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]],
    capital_per_stock: float,
    target_percentage: float,
    run_parallel: bool,
    period_name: str,
    top_performers_data: List[Tuple], # Added top_performers_data
    use_simple_rule_strategy: bool = False # New parameter for simple rule strategy
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]]]: # Added Dict[str, List[float]] for buy_hold_histories_per_ticker
    """Helper function to run portfolio backtest for a given period."""
    num_processes = NUM_PROCESSES

    backtest_params = []
    for ticker in top_tickers:
        # Use optimized parameters if available, otherwise fall back to global defaults
        min_proba_buy_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_buy', MIN_PROBA_BUY)
        min_proba_sell_ticker = optimized_params_per_ticker.get(ticker, {}).get('min_proba_sell', MIN_PROBA_SELL)
        target_percentage_ticker = optimized_params_per_ticker.get(ticker, {}).get('target_percentage', target_percentage)

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
            feature_set_for_worker, min_proba_buy_ticker, min_proba_sell_ticker, target_percentage_ticker,
            top_performers_data, use_simple_rule_strategy # Pass new parameter
        ))

    portfolio_values = []
    processed_tickers = []
    performance_metrics = []
    buy_hold_histories_per_ticker: Dict[str, List[float]] = {} # New: Store buy_hold_histories
    
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
                    buy_hold_histories_per_ticker[res['ticker']] = res.get('buy_hold_history', []) # Store history
                    
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
                buy_hold_histories_per_ticker[worker_result['ticker']] = worker_result.get('buy_hold_history', []) # Store history
                
                # Find the corresponding performance data (1Y and YTD from top_performers_data)
                perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
                for t, p1y, pytd in top_performers_data:
                    if t == worker_result['ticker']:
                        perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                        ytd_perf_benchmark = pytd if np.isfinite(pytd) else np.nan
                        break
                
                # Print individual stock performance immediately
                print(f"\n📈 Individual Stock Performance for {worker_result['ticker']} ({period_name}):")
                print(f"  - 1-Year Performance: {perf_1y_benchmark:.2f}%" if pd.notna(perf_1y_benchmark) else "  - 1-Year Performance: N/A")
                print(f"  - YTD Performance: {ytd_perf_benchmark:.2f}%" if pd.notna(ytd_perf_benchmark) else "  - YTD Performance: N/A")
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
    return final_portfolio_value, portfolio_values, processed_tickers, performance_metrics, buy_hold_histories_per_ticker

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
    num_tickers_analyzed: int,
    final_strategy_value_1month: float, # Added parameter
    ai_1month_return: float, # Added parameter
    final_buy_hold_value_1month: float, # Added parameter
    final_simple_rule_value_1y: float, # New parameter
    simple_rule_1y_return: float, # New parameter
    final_simple_rule_value_ytd: float, # New parameter
    simple_rule_ytd_return: float, # New parameter
    final_simple_rule_value_3month: float, # New parameter
    simple_rule_3month_return: float, # New parameter
    final_simple_rule_value_1month: float, # New parameter
    simple_rule_1month_return: float, # New parameter
    performance_metrics_simple_rule_1y: List[Dict], # New parameter for simple rule performance
    performance_metrics_buy_hold_1y: List[Dict], # New parameter for Buy & Hold performance
    top_performers_data: List[Tuple] # Add top_performers_data here
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}") # Use the passed initial_balance_used
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)") # New
    print(f"  1-Year Buy & Hold Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  YTD AI Strategy Value: ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)")
    print(f"  YTD Simple Rule Value: ${final_simple_rule_value_ytd:,.2f} ({simple_rule_ytd_return:+.2f}%)") # New
    print(f"  YTD Buy & Hold Value: ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  3-Month AI Strategy Value: ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)")
    print(f"  3-Month Simple Rule Value: ${final_simple_rule_value_3month:,.2f} ({simple_rule_3month_return:+.2f}%)") # New
    print(f"  3-Month Buy & Hold Value: ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  1-Month AI Strategy Value: ${final_strategy_value_1month:,.2f} ({ai_1month_return:+.2f}%)")
    print(f"  1-Month Simple Rule Value: ${final_simple_rule_value_1month:,.2f} ({simple_rule_1month_return:+.2f}%)") # New
    print(f"  1-Month Buy & Hold Value: ${final_buy_hold_value_1month:,.2f} ({((final_buy_hold_value_1month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 290)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Shares Before Liquidation':>25}")
    print("-" * 290)
    for res in sorted_final_results:
        # --- Safely get ticker and parameters ---
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', CLASS_HORIZON) # Get class_horizon
        opt_status = optimized_params.get('optimization_status', 'N/A') # Get optimization status

        # --- Calculate allocated capital and strategy gain ---
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('performance', 0.0) - allocated_capital

        # --- Safely format performance numbers ---
        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {shares_before_liquidation_str}")
    print("-" * 290)

    # --- Simple Rule Strategy Individual Ticker Performance ---
    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    # Sort simple rule results by 1Y performance for the table
    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital
        
        # Find the corresponding 1Y and YTD performance from top_performers_data
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data: # Assuming top_performers_data is available in this scope
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD')) # Renamed from last_ai_action to last_action for clarity
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 136)

    # --- Buy & Hold Strategy Individual Ticker Performance ---
    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    # Sort Buy & Hold results by 1Y performance for the table
    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
        
        # Find the corresponding 1Y and YTD performance from top_performers_data
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}" # New: Shares Before Liquidation

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 136)

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
    class_horizon: int = CLASS_HORIZON, # New parameter for initial/default class_horizon for optimization
    force_thresholds_optimization: bool = FORCE_THRESHOLDS_OPTIMIZATION, # New parameter
    force_percentage_optimization: bool = FORCE_PERCENTAGE_OPTIMIZATION, # New parameter
    top_performers_data=None,
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True,
    single_ticker: Optional[str] = None,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None,
    use_simple_rule_strategy: bool = USE_SIMPLE_RULE_STRATEGY # New parameter for simple rule strategy
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[Dict]]:
    
    # Set the start method for multiprocessing to 'spawn'
    # This is crucial for CUDA compatibility with multiprocessing
    try:
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method set to 'spawn' for CUDA compatibility.")
    except RuntimeError as e:
        print(f"⚠️ Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA and multiprocessing.")

    end_date = datetime.now(timezone.utc)
    bt_end = end_date
    
    alpaca_trading_client = None

    # Initialize ML libraries to determine CUDA availability
    initialize_ml_libraries()
    
    # Disable parallel processing if deep learning models are used with CUDA
    if PYTORCH_AVAILABLE and CUDA_AVAILABLE and (USE_LSTM or USE_GRU):
        print("⚠️ CUDA is available and deep learning models are enabled.")
        run_parallel = True
    
    # Initialize initial_balance_used here with a default value
    initial_balance_used = INITIAL_BALANCE 
    print(f"Using initial balance: ${initial_balance_used:,.2f}")

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
                perf_ytd = ((end_date - start_price) / start_price) * 100
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

    # --- Fetch SPY data for Market Momentum feature ---
    print("🔍 Fetching SPY data for Market Momentum feature...")
    spy_df = load_prices_robust('SPY', earliest_date_needed, end_date)
    if not spy_df.empty:
        spy_df['SPY_Returns'] = spy_df['Close'].pct_change()
        spy_df['Market_Momentum_SPY'] = spy_df['SPY_Returns'].rolling(window=FEAT_VOL_WINDOW).mean()
        spy_df = spy_df[['Market_Momentum_SPY']]
        spy_df.columns = pd.MultiIndex.from_product([spy_df.columns, ['SPY']])
        
        # Merge SPY data into all_tickers_data
        all_tickers_data = all_tickers_data.merge(spy_df, left_index=True, right_index=True, how='left')
        # Forward fill and then back fill any NaNs introduced by the merge
        all_tickers_data['Market_Momentum_SPY', 'SPY'] = all_tickers_data['Market_Momentum_SPY', 'SPY'].ffill().bfill().fillna(0)
        print("✅ SPY Market Momentum data fetched and merged.")
    else:
        print("⚠️ Could not fetch SPY data. Market Momentum feature will be 0.")
        # Add a zero-filled column if SPY data couldn't be fetched
        all_tickers_data['Market_Momentum_SPY', 'SPY'] = 0.0

    # --- Fetch and merge intermarket data ---
    # --- Fetch and merge intermarket data ---
    print("🔍 Fetching intermarket data...")
    intermarket_df = _fetch_intermarket_data(earliest_date_needed, end_date)
    if not intermarket_df.empty:
        # Rename columns to include 'Intermarket' level for MultiIndex
        intermarket_df.columns = pd.MultiIndex.from_product([intermarket_df.columns, ['Intermarket']])
        all_tickers_data = all_tickers_data.merge(intermarket_df, left_index=True, right_index=True, how='left')
        # Forward fill and then back fill any NaNs introduced by the merge
        for col in intermarket_df.columns:
            all_tickers_data[col] = all_tickers_data[col].ffill().bfill().fillna(0)
        print("✅ Intermarket data fetched and merged.")
    else:
        print("⚠️ Could not fetch intermarket data. Intermarket features will be 0.")
        # Add zero-filled columns for intermarket features to ensure feature set consistency
        for col_name in ['Bond_Yield_Returns', 'Oil_Price_Returns', 'Gold_Price_Returns']: # DXY_Index_Returns removed from _fetch_intermarket_data
            if (col_name, 'Intermarket') not in all_tickers_data.columns:
                all_tickers_data[col_name, 'Intermarket'] = 0.0
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
        _ensure_dir(Path("logs")) # Ensure logs directory exists
        with open("logs/skipped_tickers.log", "w") as f:
            f.write("Tickers skipped during performance analysis:\n")
            for ticker in sorted(list(skipped_tickers)):
                f.write(f"{ticker}\n")

    # --- Training Models (for 1-Year Backtest) ---
    models_buy, models_sell, scalers = {}, {}, {}
    gru_hyperparams_buy_dict, gru_hyperparams_sell_dict = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_1y = {} # New: Store failed tickers and their reasons
    if ENABLE_1YEAR_TRAINING:
        print("🔍 Step 3: Training AI models for 1-Year backtest...")
        bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)
        train_end_1y = bt_start_1y - timedelta(days=1)
        train_start_1y_calc = train_end_1y - timedelta(days=TRAIN_LOOKBACK_DAYS)
        
        training_params_1y = []
        for ticker in top_tickers:
            # Load existing GRU hyperparams for this ticker if available
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_1y_calc:train_end_1y, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_1y.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 1-Year period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 1-Year models in parallel for {len(top_tickers)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_1y = list(tqdm(pool.imap(train_worker, training_params_1y), total=len(training_params_1y), desc="Training 1-Year Models"))
        else:
            print(f"🤖 Training 1-Year models sequentially for {len(top_tickers)} tickers...")
            training_results_1y = [train_worker(p) for p in tqdm(training_params_1y, desc="Training 1-Year Models")]

        for res in training_results_1y:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy[res['ticker']] = res['model_buy']
                models_sell[res['ticker']] = res['model_sell']
                scalers[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_1y[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 1-Year training loop, models_buy has {len(models_buy)} entries.")

        if not models_buy and USE_MODEL_GATE:
            print("⚠️ No models were trained for 1-Year backtest. Model-gating will be disabled for this run.\n")
    
    # Filter out failed tickers from top_tickers for subsequent steps
    top_tickers_1y_filtered = [t for t in top_tickers if t not in failed_training_tickers_1y]
    print(f"  ℹ️ {len(failed_training_tickers_1y)} tickers failed 1-Year model training and will be skipped: {', '.join(failed_training_tickers_1y.keys())}")
    
    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1y_filtered = [item for item in top_performers_data if item[0] in top_tickers_1y_filtered]
    
    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1y = INVESTMENT_PER_STOCK
    
    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_1y_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1y_filtered}
    else:
        optimized_params_per_ticker_1y_filtered = {}
    
    
    # --- OPTIMIZE THRESHOLDS ---
    # Ensure logs directory exists for optimized parameters
    _ensure_dir(TOP_CACHE_PATH.parent)
    optimized_params_file = TOP_CACHE_PATH.parent / "optimized_per_ticker_params.json"
    
    # If force_thresholds_optimization is True and the file exists, delete it to force re-optimization
    if force_thresholds_optimization and optimized_params_file.exists():
        try:
            os.remove(optimized_params_file)
            print(f"🗑️ Deleted existing optimized parameters file: {optimized_params_file} to force re-optimization.")
        except Exception as e:
            print(f"⚠️ Could not delete optimized parameters file: {e}")

    optimized_params_per_ticker = {}
    loaded_optimized_params = {}

    # Try to load existing optimized parameters if not forcing re-optimization
    if optimized_params_file.exists():
        try:
            with open(optimized_params_file, 'r') as f:
                loaded_optimized_params = json.load(f)
            print(f"\n✅ Loaded existing optimized parameters from {optimized_params_file}.")
        except Exception as e:
            print(f"⚠️ Could not load optimized parameters from file: {e}. Starting with default thresholds.")

    # Determine if optimization needs to run at all
    should_run_optimization = force_thresholds_optimization or force_percentage_optimization

    if should_run_optimization:
        print("\n🔄 Step 2.5: Optimizing ML parameters for each ticker...")
        optimized_params_per_ticker = optimize_thresholds_for_portfolio(
            top_tickers=top_tickers_1y_filtered,
            train_start=train_start_1y,
            train_end=train_end_1y,
            default_target_percentage=target_percentage,
            default_class_horizon=class_horizon, # Pass class_horizon here
            feature_set=feature_set,
            models_buy=models_buy,
            models_sell=models_sell,
            scalers=scalers,
            capital_per_stock=capital_per_stock_1y,
            run_parallel=run_parallel,
            force_percentage_optimization=force_percentage_optimization,
            force_thresholds_optimization=force_thresholds_optimization, # Pass this parameter
            current_optimized_params_per_ticker=loaded_optimized_params
        )
        if optimized_params_per_ticker:
            try:
                with open(optimized_params_file, 'w') as f:
                    json.dump(optimized_params_per_ticker, f, indent=4)
                print(f"✅ Optimized parameters saved to {optimized_params_file}")
            except Exception as e:
                print(f"⚠️ Could not save optimized parameters to file: {e}")
    else:
        # If no optimization is forced, load existing or use defaults
        optimized_params_per_ticker = {}
        for ticker in top_tickers_1y_filtered:
            if ticker in loaded_optimized_params:
                optimized_params_per_ticker[ticker] = loaded_optimized_params[ticker]
                optimized_params_per_ticker[ticker]['optimization_status'] = "Loaded"
            else:
                optimized_params_per_ticker[ticker] = {
                    'min_proba_buy': MIN_PROBA_BUY,
                    'min_proba_sell': MIN_PROBA_SELL,
                    'target_percentage': target_percentage,
                    'optimization_status': "Not Optimized (using defaults)"
                }
        print(f"\n✅ Using loaded or default parameters (set 'force_thresholds_optimization=True' or 'force_percentage_optimization=True' in main() call to re-run optimization).")
        if not optimized_params_per_ticker:
            print("\nℹ️ No optimized parameters found for current tickers. Using default thresholds.")


    # Initialize all backtest related variables to default values
    final_strategy_value_1y = initial_balance_used
    strategy_results_1y = []
    processed_tickers_1y = []
    performance_metrics_1y = []
    ai_1y_return = 0.0
    final_simple_rule_value_1y = initial_balance_used
    simple_rule_results_1y = []
    processed_simple_rule_tickers_1y = []
    performance_metrics_simple_rule_1y = []
    simple_rule_1y_return = 0.0
    final_buy_hold_value_1y = initial_balance_used
    buy_hold_results_1y = []
    performance_metrics_buy_hold_1y_actual = [] # Initialize here

    final_strategy_value_ytd = initial_balance_used
    strategy_results_ytd = []
    processed_tickers_ytd_local = []
    performance_metrics_ytd = []
    ai_ytd_return = 0.0
    final_simple_rule_value_ytd = initial_balance_used
    simple_rule_results_ytd = []
    processed_simple_rule_tickers_ytd = []
    performance_metrics_simple_rule_ytd = []
    simple_rule_ytd_return = 0.0
    final_buy_hold_value_ytd = initial_balance_used
    buy_hold_results_ytd = []
    performance_metrics_buy_hold_ytd_actual = [] # Initialize here

    final_strategy_value_3month = initial_balance_used
    strategy_results_3month = []
    processed_tickers_3month_local = []
    performance_metrics_3month = []
    ai_3month_return = 0.0
    final_simple_rule_value_3month = initial_balance_used
    simple_rule_results_3month = []
    processed_simple_rule_tickers_3month = []
    performance_metrics_simple_rule_3month = []
    simple_rule_3month_return = 0.0
    final_buy_hold_value_3month = initial_balance_used
    buy_hold_results_3month = []
    performance_metrics_buy_hold_3month_actual = [] # Initialize here

    final_strategy_value_1month = initial_balance_used
    strategy_results_1month = []
    processed_tickers_1month_local = []
    performance_metrics_1month = []
    ai_1month_return = 0.0
    final_simple_rule_value_1month = initial_balance_used
    simple_rule_results_1month = []
    processed_simple_rule_tickers_1month = []
    performance_metrics_simple_rule_1month = []
    simple_rule_1month_return = 0.0
    final_buy_hold_value_1month = initial_balance_used
    buy_hold_results_1month = []
    performance_metrics_buy_hold_1month_actual = [] # Initialize here
    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here
    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here

    # --- Run 1-Year Backtest ---
    if ENABLE_1YEAR_BACKTEST:
        print("\n🔍 Step 4: Running 1-Year Backtest...")
        # --- Run 1-Year Backtest (AI Strategy) ---
        print("\n🔍 Step 4: Running 1-Year Backtest (AI Strategy)...")
        final_strategy_value_1y, strategy_results_1y, processed_tickers_1y, performance_metrics_1y, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1y,
            end_date=bt_end,
            top_tickers=top_tickers_1y_filtered, # Use filtered tickers for backtest
            models_buy=models_buy,
            models_sell=models_sell,
            scalers=scalers,
            optimized_params_per_ticker=optimized_params_per_ticker,
            capital_per_stock=capital_per_stock_1y, # Use fixed capital per stock
            # Pass the global target_percentage here, as the individual backtest_worker will use the optimized one
            target_percentage=target_percentage, 
            run_parallel=run_parallel,
            period_name="1-Year (AI)",
            top_performers_data=top_performers_data_1y_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_1y_return = ((final_strategy_value_1y - (capital_per_stock_1y * len(top_tickers_1y_filtered))) / abs(capital_per_stock_1y * len(top_tickers_1y_filtered))) * 100 if (capital_per_stock_1y * len(top_tickers_1y_filtered)) != 0 else 0

        # --- Run 1-Year Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 1-Year Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_1y, simple_rule_results_1y, processed_simple_rule_tickers_1y, performance_metrics_simple_rule_1y, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1y,
            end_date=bt_end,
            top_tickers=top_tickers_1y_filtered, # Use filtered tickers for backtest
            models_buy={}, # No ML models for simple rule strategy
            models_sell={}, # No ML models for simple rule strategy
            scalers={}, # No scalers for simple rule strategy
            optimized_params_per_ticker={}, # No optimized params for simple rule strategy
            capital_per_stock=capital_per_stock_1y,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="1-Year (Simple Rule)",
            top_performers_data=top_performers_data_1y_filtered,
            use_simple_rule_strategy=True # Explicitly set to True for simple rule strategy
        )
        simple_rule_1y_return = ((final_simple_rule_value_1y - (capital_per_stock_1y * len(top_tickers_1y_filtered))) / abs(capital_per_stock_1y * len(top_tickers_1y_filtered))) * 100 if (capital_per_stock_1y * len(top_tickers_1y_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 1-Year ---
        print("\n📊 Calculating Buy & Hold performance for 1-Year period...")
        buy_hold_results_1y = []
        performance_metrics_buy_hold_1y_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_1y_filtered: # Iterate over filtered tickers
            df_bh = load_prices_robust(ticker, bt_start_1y, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_1y / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_1y - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_1y
                buy_hold_results_1y.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_1y_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_1y) / abs(capital_per_stock_1y)) * 100 if capital_per_stock_1y != 0 else 0.0,
                    'final_shares': shares_bh # Add this line
                })
            else:
                buy_hold_results_1y.append(capital_per_stock_1y)
                performance_metrics_buy_hold_1y_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_1y,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0,
                    'final_shares': 0.0 # Add this line
                })
        final_buy_hold_value_1y = sum(buy_hold_results_1y) + (len(top_tickers_1y_filtered) - len(buy_hold_results_1y)) * capital_per_stock_1y
        print("✅ 1-Year Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 1-Year Backtest is disabled by ENABLE_1YEAR_BACKTEST flag.")


    # --- Training Models (for YTD Backtest) ---
    models_buy_ytd, models_sell_ytd, scalers_ytd = {}, {}, {}
    gru_hyperparams_buy_dict_ytd, gru_hyperparams_sell_dict_ytd = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_ytd = {} # New: Store failed tickers and their reasons
    if ENABLE_YTD_TRAINING:
        print("\n🔍 Step 5: Training AI models for YTD backtest...")
        ytd_start_date = datetime(bt_end.year, 1, 1, tzinfo=timezone.utc)
        train_end_ytd = ytd_start_date - timedelta(days=1)
        train_start_ytd = train_end_ytd - timedelta(days=TRAIN_LOOKBACK_DAYS)
        
        training_params_ytd = []
        for ticker in top_tickers_1y_filtered: # Use filtered tickers for YTD training
            # Load existing GRU hyperparams for this ticker if available
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_ytd:train_end_ytd, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_ytd.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for YTD period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training YTD models in parallel for {len(top_tickers_1y_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_ytd = list(tqdm(pool.imap(train_worker, training_params_ytd), total=len(training_params_ytd), desc="Training YTD Models"))
        else:
            print(f"🤖 Training YTD models sequentially for {len(top_tickers_1y_filtered)} tickers...")
            training_results_ytd = [train_worker(p) for p in tqdm(training_params_ytd, desc="Training YTD Models")]

        for res in training_results_ytd:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_ytd[res['ticker']] = res['model_buy']
                models_sell_ytd[res['ticker']] = res['model_sell']
                scalers_ytd[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_ytd[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_ytd[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_ytd[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After YTD training loop, models_buy_ytd has {len(models_buy_ytd)} entries.")

        if not models_buy_ytd and USE_MODEL_GATE:
            print("⚠️ No models were trained for YTD backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_1y_filtered for subsequent steps
    top_tickers_ytd_filtered = [t for t in top_tickers_1y_filtered if t not in failed_training_tickers_ytd]
    print(f"  ℹ️ {len(failed_training_tickers_ytd)} tickers failed YTD model training and will be skipped: {', '.join(failed_training_tickers_ytd.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_ytd_filtered = [item for item in top_performers_data_1y_filtered if item[0] in top_tickers_ytd_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_ytd = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_ytd_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_ytd_filtered}
    else:
        optimized_params_per_ticker_ytd_filtered = {}

    # --- Run YTD Backtest ---
    if ENABLE_YTD_BACKTEST:
        print("\n🔍 Step 6: Running YTD Backtest...")
        # --- Run YTD Backtest (AI Strategy) ---
        print("\n🔍 Step 6: Running YTD Backtest (AI Strategy)...")
        final_strategy_value_ytd, strategy_results_ytd, processed_tickers_ytd_local, performance_metrics_ytd, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=ytd_start_date,
            end_date=bt_end,
            top_tickers=top_tickers_ytd_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_ytd,
            models_sell=models_sell_ytd,
            scalers=scalers_ytd,
            optimized_params_per_ticker=optimized_params_per_ticker_ytd_filtered,
            capital_per_stock=capital_per_stock_ytd, # Use fixed capital per stock
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="YTD (AI)",
            top_performers_data=top_performers_data_ytd_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_ytd_return = ((final_strategy_value_ytd - (capital_per_stock_ytd * len(top_tickers_ytd_filtered))) / abs(capital_per_stock_ytd * len(top_tickers_ytd_filtered))) * 100 if (capital_per_stock_ytd * len(top_tickers_ytd_filtered)) != 0 else 0

        # --- Run YTD Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running YTD Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_ytd, simple_rule_results_ytd, processed_simple_rule_tickers_ytd, performance_metrics_simple_rule_ytd, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=ytd_start_date,
            end_date=bt_end,
            top_tickers=top_tickers_ytd_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_ytd,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="YTD (Simple Rule)",
            top_performers_data=top_performers_data_ytd_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_ytd_return = ((final_simple_rule_value_ytd - (capital_per_stock_ytd * len(top_tickers_ytd_filtered))) / abs(capital_per_stock_ytd * len(top_tickers_ytd_filtered))) * 100 if (capital_per_stock_ytd * len(top_tickers_ytd_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for YTD ---
        print("\n📊 Calculating Buy & Hold performance for YTD period...")
        buy_hold_results_ytd = []
        performance_metrics_buy_hold_ytd_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_ytd_filtered: # Iterate over filtered tickers
            df_bh = load_prices_robust(ticker, ytd_start_date, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_ytd / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_ytd - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_ytd
                buy_hold_results_ytd.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_ytd_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_ytd) / abs(capital_per_stock_ytd)) * 100 if capital_per_stock_ytd != 0 else 0.0
                })
            else:
                buy_hold_results_ytd.append(capital_per_stock_ytd)
                performance_metrics_buy_hold_ytd_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_ytd,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_ytd = sum(buy_hold_results_ytd) + (len(top_tickers_ytd_filtered) - len(buy_hold_results_ytd)) * capital_per_stock_ytd
        print("✅ YTD Buy & Hold calculation complete.")
    else:
        print("\nℹ️ YTD Backtest is disabled by ENABLE_YTD_BACKTEST flag.")

    # --- Training Models (for 3-Month Backtest) ---
    models_buy_3month, models_sell_3month, scalers_3month = {}, {}, {}
    gru_hyperparams_buy_dict_3month, gru_hyperparams_sell_dict_3month = {}, {} # New: To store GRU hyperparams
    failed_training_tickers_3month = {} # New: Store failed tickers and their reasons
    if ENABLE_3MONTH_TRAINING:
        print("\n🔍 Step 7: Training AI models for 3-Month backtest...")
        bt_start_3month = bt_end - timedelta(days=BACKTEST_DAYS_3MONTH)
        train_end_3month = bt_start_3month - timedelta(days=1)
        train_start_3month = train_end_3month - timedelta(days=TRAIN_LOOKBACK_DAYS)

        training_params_3month = []
        for ticker in top_tickers_ytd_filtered: # Use filtered tickers for 3-Month training
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_3month:train_end_3month, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_3month.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 3-Month period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 3-Month models in parallel for {len(top_tickers_ytd_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_3month = list(tqdm(pool.imap(train_worker, training_params_3month), total=len(training_params_3month), desc="Training 3-Month Models"))
        else:
            print(f"🤖 Training 3-Month models sequentially for {len(top_tickers_ytd_filtered)} tickers...")
            training_results_3month = [train_worker(p) for p in tqdm(training_params_3month, desc="Training 3-Month Models")]

        for res in training_results_3month:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_3month[res['ticker']] = res['model_buy']
                models_sell_3month[res['ticker']] = res['model_sell']
                scalers_3month[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_3month[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_3month[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_3month[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 3-Month training loop, models_buy_3month has {len(models_buy_3month)} entries.")

        if not models_buy_3month and USE_MODEL_GATE:
            print("⚠️ No models were trained for 3-Month backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_ytd_filtered for subsequent steps
    top_tickers_3month_filtered = [t for t in top_tickers_ytd_filtered if t not in failed_training_tickers_3month]
    print(f"  ℹ️ {len(failed_training_tickers_3month)} tickers failed 3-Month model training and will be skipped: {', '.join(failed_training_tickers_3month.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_3month_filtered = [item for item in top_performers_data_ytd_filtered if item[0] in top_tickers_3month_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_3month = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_3month_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_3month_filtered}
    else:
        optimized_params_per_ticker_3month_filtered = {}

    # --- Run 3-Month Backtest ---
    if ENABLE_3MONTH_BACKTEST:
        print("\n🔍 Step 8: Running 3-Month Backtest...")
        # --- Run 3-Month Backtest (AI Strategy) ---
        print("\n🔍 Step 8: Running 3-Month Backtest (AI Strategy)...")
        final_strategy_value_3month, strategy_results_3month, processed_tickers_3month_local, performance_metrics_3month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_3month,
            end_date=bt_end,
            top_tickers=top_tickers_3month_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_3month,
            models_sell=models_sell_3month,
            scalers=scalers_3month,
            optimized_params_per_ticker=optimized_params_per_ticker_3month_filtered,
            capital_per_stock=capital_per_stock_3month, # Use fixed capital per stock
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="3-Month (AI)",
            top_performers_data=top_performers_data_3month_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_3month_return = ((final_strategy_value_3month - (capital_per_stock_3month * len(top_tickers_3month_filtered))) / abs(capital_per_stock_3month * len(top_tickers_3month_filtered))) * 100 if (capital_per_stock_3month * len(top_tickers_3month_filtered)) != 0 else 0

        # --- Run 3-Month Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 3-Month Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_3month, simple_rule_results_3month, processed_simple_rule_tickers_3month, performance_metrics_simple_rule_3month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_3month,
            end_date=bt_end,
            top_tickers=top_tickers_3month_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_3month,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="3-Month (Simple Rule)",
            top_performers_data=top_performers_data_3month_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_3month_return = ((final_simple_rule_value_3month - (capital_per_stock_3month * len(top_tickers_3month_filtered))) / abs(capital_per_stock_3month * len(top_tickers_3month_filtered))) * 100 if (capital_per_stock_3month * len(top_tickers_3month_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 3-Month ---
        print("\n📊 Calculating Buy & Hold performance for 3-Month period...")
        buy_hold_results_3month = []
        performance_metrics_buy_hold_3month_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_3month_filtered:
            df_bh = load_prices_robust(ticker, bt_start_3month, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_3month / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_3month - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_3month
                buy_hold_results_3month.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_3month_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_3month) / abs(capital_per_stock_3month)) * 100 if capital_per_stock_3month != 0 else 0.0
                })
            else:
                buy_hold_results_3month.append(capital_per_stock_3month)
                performance_metrics_buy_hold_3month_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_3month,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_3month = sum(buy_hold_results_3month) + (len(top_tickers_3month_filtered) - len(buy_hold_results_3month)) * capital_per_stock_3month
        print("✅ 3-Month Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 3-Month Backtest is disabled by ENABLE_3MONTH_BACKTEST flag.")

    # --- Training Models (for 1-Month Backtest) ---
    models_buy_1month, models_sell_1month, scalers_1month = {}, {}, {}
    failed_training_tickers_1month = {} # New: Store failed tickers and their reasons
    if ENABLE_1MONTH_TRAINING:
        print("\n🔍 Step 9: Training AI models for 1-Month backtest...")
        bt_start_1month = bt_end - timedelta(days=BACKTEST_DAYS_1MONTH)
        train_end_1month = bt_start_1month - timedelta(days=1)
        train_start_1month = train_end_1month - timedelta(days=TRAIN_LOOKBACK_DAYS)

        training_params_1month = []
        for ticker in top_tickers_3month_filtered: # Use filtered tickers for 1-Month training
            loaded_gru_hyperparams_buy = None
            loaded_gru_hyperparams_sell = None
            gru_hyperparams_buy_path = Path("logs/models") / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
            gru_hyperparams_sell_path = Path("logs/models") / f"{ticker}_TargetClassSell_gru_optimized_params.json"

            if gru_hyperparams_buy_path.exists():
                try:
                    with open(gru_hyperparams_buy_path, 'r') as f:
                        loaded_gru_hyperparams_buy = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            try:
                # Slice the main DataFrame for the training period
                ticker_train_data = all_tickers_data.loc[train_start_1month:train_end_1month, (slice(None), ticker)]
                ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                training_params_1month.append((ticker, ticker_train_data.copy(), target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell))
            except (KeyError, IndexError):
                print(f"  ⚠️ Could not slice training data for {ticker} for 1-Month period. Skipping.")
                continue
        
        if run_parallel:
            print(f"🤖 Training 1-Month models in parallel for {len(top_tickers_3month_filtered)} tickers using {NUM_PROCESSES} processes...")
            with Pool(processes=NUM_PROCESSES) as pool:
                training_results_1month = list(tqdm(pool.imap(train_worker, training_params_1month), total=len(training_params_1month), desc="Training 1-Month Models"))
        else:
            print(f"🤖 Training 1-Month models sequentially for {len(top_tickers_3month_filtered)} tickers...")
            training_results_1month = [train_worker(p) for p in tqdm(training_params_1month, desc="Training 1-Month Models")]

        for res in training_results_1month:
            if res and (res.get('status') == 'trained' or res.get('status') == 'loaded'): # Check for both 'trained' and 'loaded'
                models_buy_1month[res['ticker']] = res['model_buy']
                models_sell_1month[res['ticker']] = res['model_sell']
                scalers_1month[res['ticker']] = res['scaler']
                if res.get('gru_hyperparams_buy'):
                    gru_hyperparams_buy_dict_1month[res['ticker']] = res['gru_hyperparams_buy']
                if res.get('gru_hyperparams_sell'):
                    gru_hyperparams_sell_dict_1month[res['ticker']] = res['gru_hyperparams_sell']
            elif res and res.get('status') == 'failed':
                failed_training_tickers_1month[res['ticker']] = res['reason']
        print(f"  [DIAGNOSTIC] After 1-Month training loop, models_buy_1month has {len(models_buy_1month)} entries.")

        if not models_buy_1month and USE_MODEL_GATE:
            print("⚠️ No models were trained for 1-Month backtest. Model-gating will be disabled for this run.\n")

    # Filter out failed tickers from top_tickers_3month_filtered for subsequent steps
    top_tickers_1month_filtered = [t for t in top_tickers_3month_filtered if t not in failed_training_tickers_1month]
    print(f"  ℹ️ {len(failed_training_tickers_1month)} tickers failed 1-Month model training and will be skipped: {', '.join(failed_training_tickers_1month.keys())}")

    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1month_filtered = [item for item in top_performers_data_3month_filtered if item[0] in top_tickers_1month_filtered]

    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1month = INVESTMENT_PER_STOCK

    # Update optimized_params_per_ticker to only include successfully trained tickers
    if optimized_params_per_ticker:
        optimized_params_per_ticker_1month_filtered = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1month_filtered}
    else:
        optimized_params_per_ticker_1month_filtered = {}

    # --- Run 1-Month Backtest ---
    if ENABLE_1MONTH_BACKTEST:
        print("\n🔍 Step 10: Running 1-Month Backtest...")
        # --- Run 1-Month Backtest (AI Strategy) ---
        print("\n🔍 Step 10: Running 1-Month Backtest (AI Strategy)...")
        final_strategy_value_1month, strategy_results_1month, processed_tickers_1month_local, performance_metrics_1month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1month,
            end_date=bt_end,
            top_tickers=top_tickers_1month_filtered, # Use filtered tickers for backtest
            models_buy=models_buy_1month,
            models_sell=models_sell_1month,
            scalers=scalers_1month,
            optimized_params_per_ticker=optimized_params_per_ticker_1month_filtered,
            capital_per_stock=capital_per_stock_1month, # Use fixed capital per stock
            target_percentage=target_percentage, 
            run_parallel=run_parallel,
            period_name="1-Month (AI)",
            top_performers_data=top_performers_data_1month_filtered, # Pass filtered top_performers_data
            use_simple_rule_strategy=False # Explicitly set to False for AI strategy
        )
        ai_1month_return = ((final_strategy_value_1month - (capital_per_stock_1month * len(top_tickers_1month_filtered))) / abs(capital_per_stock_1month * len(top_tickers_1month_filtered))) * 100 if (capital_per_stock_1month * len(top_tickers_1month_filtered)) != 0 else 0

        # --- Run 1-Month Backtest (Simple Rule Strategy) ---
        print("\n🔍 Running 1-Month Backtest (Simple Rule Strategy)...")
        final_simple_rule_value_1month, simple_rule_results_1month, processed_simple_rule_tickers_1month, performance_metrics_simple_rule_1month, _ = _run_portfolio_backtest(
            all_tickers_data=all_tickers_data,
            start_date=bt_start_1month,
            end_date=bt_end,
            top_tickers=top_tickers_1month_filtered,
            models_buy={},
            models_sell={},
            scalers={},
            optimized_params_per_ticker={},
            capital_per_stock=capital_per_stock_1month,
            target_percentage=target_percentage,
            run_parallel=run_parallel,
            period_name="1-Month (Simple Rule)",
            top_performers_data=top_performers_data_1month_filtered,
            use_simple_rule_strategy=True
        )
        simple_rule_1month_return = ((final_simple_rule_value_1month - (capital_per_stock_1month * len(top_tickers_1month_filtered))) / abs(capital_per_stock_1month * len(top_tickers_1month_filtered))) * 100 if (capital_per_stock_1month * len(top_tickers_1month_filtered)) != 0 else 0

        # --- Calculate Buy & Hold for 1-Month ---
        print("\n📊 Calculating Buy & Hold performance for 1-Month period...")
        buy_hold_results_1month = []
        performance_metrics_buy_hold_1month_actual = [] # New list for actual BH performance metrics
        for ticker in top_tickers_1month_filtered:
            df_bh = load_prices_robust(ticker, bt_start_1month, bt_end)
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(capital_per_stock_1month / start_price) if start_price > 0 else 0
                cash_bh = capital_per_stock_1month - shares_bh * start_price
                
                bh_history_for_ticker = []
                for price_day in df_bh["Close"].tolist():
                    bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
                
                final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_1month
                buy_hold_results_1month.append(final_bh_val_ticker)
                
                # Calculate and store actual Buy & Hold performance metrics
                perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
                performance_metrics_buy_hold_1month_actual.append({
                    'ticker': ticker,
                    'final_val': final_bh_val_ticker,
                    'perf_data': perf_data_bh,
                    'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_1month) / abs(capital_per_stock_1month)) * 100 if capital_per_stock_1month != 0 else 0.0
                })
            else:
                buy_hold_results_1month.append(capital_per_stock_1month)
                performance_metrics_buy_hold_1month_actual.append({
                    'ticker': ticker,
                    'final_val': capital_per_stock_1month,
                    'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                    'individual_bh_return': 0.0
                })
        final_buy_hold_value_1month = sum(buy_hold_results_1month) + (len(top_tickers_1month_filtered) - len(buy_hold_results_1month)) * capital_per_stock_1month
        print("✅ 1-Month Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 1-Month Backtest is disabled by ENABLE_1MONTH_BACKTEST flag.")

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    print("\n📝 Preparing final summary data...")
    final_results = []
    
    # Combine all failed tickers from all periods
    all_failed_tickers = {}
    all_failed_tickers.update(failed_training_tickers_1y)
    all_failed_tickers.update(failed_training_tickers_ytd)
    all_failed_tickers.update(failed_training_tickers_3month)
    all_failed_tickers.update(failed_training_tickers_1month) # Add 1-month failed tickers

    # Add successfully processed tickers
    for i, ticker in enumerate(processed_tickers_1y):
        backtest_result_for_ticker = next((res for res in performance_metrics_1y if res['ticker'] == ticker), None)
        
        if backtest_result_for_ticker:
            perf_data = backtest_result_for_ticker['perf_data']
            individual_bh_return = backtest_result_for_ticker['individual_bh_return']
            last_ai_action = backtest_result_for_ticker['last_ai_action']
            buy_prob = backtest_result_for_ticker['buy_prob']
            sell_prob = backtest_result_for_ticker['sell_prob']
            final_shares = backtest_result_for_ticker['shares_before_liquidation'] # This is where final_shares is retrieved
        else:
            perf_data = {'sharpe_ratio': 0.0}
            individual_bh_return = 0.0
            last_ai_action = "N/A"
            buy_prob = 0.0
            sell_prob = 0.0
            final_shares = 0.0 # Set to 0.0 for tickers that didn't have a backtest result

        perf_1y_benchmark, perf_ytd_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                ytd_perf_benchmark = pytd if np.isfinite(pytd) else np.nan
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': strategy_results_1y[i],
            'sharpe': perf_data['sharpe_ratio'],
            'one_year_perf': perf_1y_benchmark,
            'ytd_perf': ytd_perf_benchmark,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'final_shares': final_shares, # Add final_shares here
            'status': 'trained',
            'reason': None
        })
    
    # Add failed tickers to the final results
    for ticker, reason in all_failed_tickers.items():
        final_results.append({
            'ticker': ticker,
            'performance': INVESTMENT_PER_STOCK, # Assign initial capital for failed tickers
            'sharpe': np.nan,
            'one_year_perf': np.nan,
            'ytd_perf': np.nan,
            'individual_bh_return': np.nan,
            'last_ai_action': "FAILED",
            'buy_prob': np.nan,
            'sell_prob': np.nan,
            'final_shares': 0.0, # Set to 0.0 for failed tickers
            'status': 'failed',
            'reason': reason
        })

    # Sort by 1Y performance for the final table, handling potential NaN values
    sorted_final_results = sorted(final_results, key=lambda x: x.get('one_year_perf', -np.inf) if pd.notna(x.get('one_year_perf')) else -np.inf, reverse=True)
    
    print_final_summary(
        sorted_final_results, models_buy, models_sell, scalers, optimized_params_per_ticker,
        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
        final_strategy_value_ytd, final_buy_hold_value_ytd, ai_ytd_return,
        final_strategy_value_3month, final_buy_hold_value_3month, ai_3month_return,
        (INVESTMENT_PER_STOCK * len(top_tickers)),
        len(top_tickers),
        final_strategy_value_1month,
        ai_1month_return,
        final_buy_hold_value_1month,
        final_simple_rule_value_1y,
        simple_rule_1y_return,
        final_simple_rule_value_ytd,
        simple_rule_ytd_return,
        final_simple_rule_value_3month,
        simple_rule_3month_return,
        final_simple_rule_value_1month,
        simple_rule_1month_return,
        performance_metrics_simple_rule_1y,
        performance_metrics_buy_hold_1y_actual, # Pass performance_metrics_buy_hold_1y_actual for Buy & Hold
        top_performers_data
    )
    print("\n✅ Final summary prepared and printed.")

    # --- Select and save best performing models for live trading ---
    # Determine which period had the highest portfolio return
    performance_values = {
        "1-Year": final_strategy_value_1y,
        "YTD": final_strategy_value_ytd,
        "3-Month": final_strategy_value_3month,
        "1-Month": final_strategy_value_1month # Include 1-Month performance
    }
    
    best_period_name = max(performance_values, key=performance_values.get)
    
    # Get the models and scalers corresponding to the best period
    if best_period_name == "1-Year":
        best_models_buy_dict = models_buy
        best_models_sell_dict = models_sell
        best_scalers_dict = scalers
    elif best_period_name == "YTD":
        best_models_buy_dict = models_buy_ytd
        best_models_sell_dict = models_sell_ytd
        best_scalers_dict = scalers_ytd
    elif best_period_name == "3-Month":
        best_models_buy_dict = models_buy_3month
        best_models_sell_dict = models_sell_3month
        best_scalers_dict = scalers_3month
    else: # "1-Month"
        best_models_buy_dict = models_buy_1month
        best_models_sell_dict = models_sell_1month
        best_scalers_dict = scalers_1month

    # Save the best models and scalers for each ticker to the paths used by live_trading.py
    models_dir = Path("logs/models")
    _ensure_dir(models_dir) # Ensure the directory exists

    print(f"\n🏆 Saving best performing models for live trading from {best_period_name} period...")

    for ticker in best_models_buy_dict.keys():
        try:
            joblib.dump(best_models_buy_dict[ticker], models_dir / f"{ticker}_model_buy.joblib")
            joblib.dump(best_models_sell_dict[ticker], models_dir / f"{ticker}_model_sell.joblib")
            joblib.dump(best_scalers_dict[ticker], models_dir / f"{ticker}_scaler.joblib")
            print(f"  ✅ Saved models for {ticker} from {best_period_name} period.")
        except Exception as e:
            print(f"  ⚠️ Error saving models for {ticker} from {best_period_name} period: {e}")

    return (
        final_strategy_value_1y, final_buy_hold_value_1y, models_buy, models_sell, scalers,
        strategy_results_1y, processed_tickers_1y, performance_metrics_1y, top_performers_data,
        final_strategy_value_ytd, final_buy_hold_value_ytd, final_strategy_value_3month,
        final_buy_hold_value_3month, final_strategy_value_1month, optimized_params_per_ticker
    )

if __name__ == "__main__":
    main()
