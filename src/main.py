from __future__ import annotations

# -*- coding: utf-8 -*-

# Toggle: use alpha-optimized probability threshold for buys
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

# Import config FIRST before any other local imports to avoid circular dependencies
import sys
import argparse
from pathlib import Path

# --- Add project root and src directory to sys.path ---
project_root = Path(__file__).resolve().parent.parent
src_dir = Path(__file__).resolve().parent
# Insert src directory first so local modules take precedence over system packages
sys.path.insert(0, str(src_dir))
sys.path.insert(1, str(project_root))

from config import (
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE,
    INITIAL_BALANCE, INVESTMENT_PER_STOCK, TRANSACTION_COST,
    BACKTEST_DAYS,
    TOP_CACHE_PATH, N_TOP_TICKERS, NUM_PROCESSES, PARALLEL_THRESHOLD, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES, PAUSE_BETWEEN_YF_CALLS,
    PORTFOLIO_SIZE,
        ENABLE_1YEAR_BACKTEST,
    ENABLE_MEAN_REVERSION,
    ENABLE_QUALITY_MOM,
    ENABLE_MOMENTUM_AI_HYBRID,
    ENABLE_STATIC_BH,
    OPTIMIZE_REBALANCE_HORIZON,
    REBALANCE_HORIZON_MIN,
    REBALANCE_HORIZON_MAX,
    FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD,
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    GRU_HIDDEN_SIZE_OPTIONS, GRU_NUM_LAYERS_OPTIONS, GRU_DROPOUT_OPTIONS,
    GRU_LEARNING_RATE_OPTIONS, GRU_BATCH_SIZE_OPTIONS, GRU_EPOCHS_OPTIONS,
    USE_GRU, USE_LSTM, USE_LOGISTIC_REGRESSION, USE_RANDOM_FOREST,
    USE_SVM, USE_MLP_CLASSIFIER, USE_LIGHTGBM, USE_XGBOOST,
    FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    USE_PERFORMANCE_BENCHMARK, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    SEED, SAVE_PLOTS, MARKET_SELECTION, DATA_INTERVAL,
    SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_EPOCHS,
    ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION,
    PERIOD_HORIZONS, POSITION_SCALING_BY_CONFIDENCE,
     RUN_BACKTEST_UNTIL_TODAY,
    RETRAIN_FREQUENCY_DAYS, ENABLE_WALK_FORWARD_RETRAINING
)

# Global variables for parallel processing (will be set in main)
_grouped_data = None

def validate_detailed_ticker(ticker):
    """Validate a single ticker's data quality (for parallel processing)"""
    global _grouped_data
    try:
        if _grouped_data and ticker in _grouped_data.groups:
            ticker_data = _grouped_data.get_group(ticker).copy()
            if not ticker_data.empty:
                ticker_data = ticker_data.set_index('date')
            summary = get_data_summary(ticker_data, ticker)
        else:
            # Ticker has no data
            summary = {
                'ticker': ticker,
                'rows': 0,
                'status': 'EMPTY',
                'message': 'No data available'
            }
        return summary
    except Exception as e:
        # Skip problematic tickers
        return {
            'ticker': ticker,
            'rows': 0,
            'status': 'ERROR',
            'message': f'Error: {str(e)[:30]}'
        }

def validate_single_ticker(ticker, ticker_counts):
    """Validate a single ticker's row count (for parallel processing)"""
    row_count = ticker_counts.get(ticker, 0)
    if row_count >= 175:  # 70% of 250 days minimum
        status = 'OK'
        message = ''
    elif row_count > 0:
        status = 'INSUFFICIENT'
        message = f"Only {row_count} rows"
    else:
        status = 'EMPTY'
        message = 'No data'
    
    return {
        'ticker': ticker,
        'rows': row_count,
        'status': status,
        'message': message
    }

def _gpu_diag():
    """Run GPU diagnostics only for enabled models"""
    # Only check PyTorch if LSTM or GRU are enabled
    if USE_LSTM or USE_GRU:
        try:
            import torch
            import multiprocessing as _mp
            if _mp.current_process().name == 'MainProcess':
                print(f"[GPU] torch.cuda.is_available(): {torch.cuda.is_available()}")
        except Exception as e:
            print("[GPU] torch check failed:", e)
    
    # Only check XGBoost if it's enabled
    if USE_XGBOOST:
        try:
            import xgboost as xgb
            import multiprocessing as _mp2
            if _mp2.current_process().name == 'MainProcess':
                print("[GPU] XGBoost version:", getattr(xgb, "__version__", "?"))
        except Exception as e:
            print("[GPU] XGBoost check failed:", e)
    
    # Only check LightGBM if it's enabled
    if USE_LIGHTGBM:
        try:
            import lightgbm as lgb
            import multiprocessing as _mp3
            if _mp3.current_process().name == 'MainProcess':
                print("[GPU] LightGBM version:", getattr(lgb, "__version__", "?"))
        except Exception as e:
            print("[GPU] LightGBM check failed:", e)

# Run GPU diagnostics
_gpu_diag()

def setup_logging(verbose: bool = False) -> None:
    """Central logging config; safe for multiprocessing (basic)."""
    import logging, os, sys
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s")
    root = logging.getLogger()
    # Avoid duplicate handlers on re-init
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)

_script_initialized = False
if not _script_initialized and __name__ == '__main__':
    print("DEBUG: Script execution initiated.")
    _script_initialized = True

import os
# from portfolio_rebalancing import run_portfolio_rebalancing_backtest  # Module deleted
# from rule_based_strategy import run_rule_based_portfolio_strategy  # Module deleted
from summary_phase import print_final_summary
from notifications import send_training_notification, send_backtesting_notification, send_error_notification
from backtesting_phase import _run_portfolio_backtest_walk_forward
from data_validation import get_data_summary, print_data_diagnostics, InsufficientDataError
from selection_backtester import run_selection_strategy_comparison, print_strategy_stock_overlap
from rebalance_optimizer import optimize_rebalance_horizons
import json
import time
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Import torch if available
if PYTORCH_AVAILABLE:
    import torch
# import gymnasium as gym  # Not used, causing import error
import codecs
import random
import requests  # Added for internet time fetching
from io import StringIO
from multiprocessing import Pool, cpu_count, current_process
import joblib # Added for model saving/loading
import warnings # Added for warning suppression
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # Fallback: no progress bar
from data_utils import (
    load_prices, fetch_training_data, load_prices_robust, _ensure_dir, 
    _download_batch_robust, _fetch_financial_data, _fetch_financial_data_from_alpaca,
    _fetch_from_stooq, _fetch_from_alpaca, _fetch_from_twelvedata, _fetch_intermarket_data
)
from utils import _normalize_symbol, _to_utc
from ticker_selection import get_all_tickers, find_top_performers
from summary_phase import (
    print_final_summary, print_prediction_vs_actual_comparison,
    print_horizon_validation_summary, print_training_phase_summary,
    print_portfolio_comparison_summary
)
from analytics import calculate_buy_hold_performance_metrics

# Import ML model related functions and classes from ml_models.py
from ml_models import initialize_ml_libraries, CUML_AVAILABLE, LGBMClassifier, XGBClassifier, models_and_params, cuMLRandomForestClassifier, cuMLLogisticRegression, cuMLStandardScaler, SHAP_AVAILABLE

# Conditionally import LSTM/GRU classes if PyTorch is available
try:
    from ml_models import LSTMClassifier, GRUClassifier, GRURegressor
except ImportError:
    LSTMClassifier = None
    GRUClassifier = None
    GRURegressor = None

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
except ImportError:
    print("⚠️ Alpaca SDK not installed. Run: pip install alpaca-py. Alpaca data provider will be skipped.")

# TwelveData SDK client
try:
    from twelvedata import TDClient
except ImportError:
    print("⚠️ TwelveData SDK client not found. TwelveData data provider will be skipped.")
# ============================
# Configuration / Hyperparams
# ============================

# --- Provider & caching
# DATA_PROVIDER           = 'alpaca'    # 'stooq', 'yahoo', 'alpaca', or 'twelvedata' # Moved to config.py
# USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin # Moved to config.py
# DATA_CACHE_DIR          = Path("data_cache") # Moved to config.py
# TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json") # Moved to config.py
# VALID_TICKERS_CACHE_PATH = Path("logs/valid_tickers.json") # Moved to config.py
# CACHE_DAYS              = 7 # Moved to config.py

# Alpaca API credentials (set as environment variables for security)
# ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY") # Moved to config.py
# ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY") # Moved to config.py

# TwelveData API credentials
# TWELVEDATA_API_KEY      = os.environ.get("TWELVEDATA_API_KEY", "YOUR_DEFAULT_KEY_OR_EMPTY_STRING") # Load from environment variable # Moved to config.py

# --- Universe / selection
# MARKET_SELECTION = { # Moved to config.py
#     "ALPACA_STOCKS": False, # Fetch all tradable US equities from Alpaca
#     "NASDAQ_ALL": False,
#     "NASDAQ_100": True,
#     "SP500": False,
#     "DOW_JONES": False,
#     "POPULAR_ETFS": False,
#     "CRYPTO": False,
#     "DAX": False,
#     "MDAX": False,
#     "SMI": False,
#     "FTSE_MIB": False,
# }
# N_TOP_TICKERS           = 2        # Number of top performers to select (0 to disable limit) # Moved to config.py
# BATCH_DOWNLOAD_SIZE     = 20000       # Reduced batch size for stability # Moved to config.py
# PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability # Moved to config.py
# PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals # Moved to config.py

# --- Parallel Processing
# NUM_PROCESSES           = max(1, cpu_count() - 5) # Use all but one CPU core for parallel processing # Moved to config.py

# --- Backtest windows (see config.py for BACKTEST_DAYS)

# --- Backtest Period Enable/Disable Flags ---
# ENABLE_1YEAR_BACKTEST   = True # Moved to config.py
# ENABLE_YTD_BACKTEST     = True # Moved to config.py
# ENABLE_3MONTH_BACKTEST  = True # Moved to config.py
# ENABLE_1MONTH_BACKTEST  = True # Moved to config.py

# --- Training Period Enable/Disable Flags ---
# Initial training removed - models are now trained during walk-forward backtest

# --- Strategy (separate from feature windows)
# STRAT_SMA_SHORT         = 10 # Moved to config.py
# STRAT_SMA_LONG          = 50 # Moved to config.py
# ATR_PERIOD              = 14 # Moved to config.py
# ATR_MULT_TRAIL          = 2.0 # Moved to config.py
# ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing # Moved to config.py
# INVESTMENT_PER_STOCK    = 15000.0    # Fixed amount to invest per stock # Moved to config.py
# TRANSACTION_COST        = 0.001      # 0.1% # Moved to config.py

# --- Feature windows (for ML only)
# FEAT_SMA_SHORT          = 5 # Moved to config.py
# FEAT_SMA_LONG           = 20 # Moved to config.py
# FEAT_VOL_WINDOW         = 10 # Moved to config.py
# CLASS_HORIZON           = 5          # days ahead for classification target # Moved to config.py
# MIN_PROBA_BUY           = 0.20      # ML gate threshold for buy model # Moved to config.py
# MIN_PROBA_SELL          = 0.20       # ML gate threshold for sell model # Moved to config.py
# TARGET_PERCENTAGE       = 0.008       # 0.8% target for buy/sell classification # Moved to config.py
# USE_MODEL_GATE          = True       # ENABLE ML gate # Moved to config.py
# USE_MARKET_FILTER       = False      # market filter removed as per user request # Moved to config.py
# MARKET_FILTER_TICKER    = 'SPY' # Moved to config.py
# MARKET_FILTER_SMA       = 200 # Moved to config.py
# USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering # Moved to config.py

# --- ML Model Selection Flags ---
# USE_LOGISTIC_REGRESSION = False # Moved to config.py
# USE_SVM                 = False # Moved to config.py
# USE_MLP_CLASSIFIER      = False # Moved to config.py
# USE_LIGHTGBM            = False # Enable LightGBM - GOOD # Moved to config.py
# #GOOD
# USE_XGBOOST             = False # Enable XGBoost # Moved to config.py
# USE_LSTM                = False # Moved to config.py
# #Not so GOOD
# USE_GRU                 = True # Enable GRU - BEST # Moved to config.py
# #BEST
# USE_RANDOM_FOREST       = False # Enable RandomForest # Moved to config.py
# #WORST


# --- Deep Learning specific hyperparameters
# SEQUENCE_LENGTH         = 32         # Number of past days to consider for LSTM/GRU # Moved to config.py
# LSTM_HIDDEN_SIZE        = 64 # Moved to config.py
# LSTM_NUM_LAYERS         = 2 # Moved to config.py
# LSTM_DROPOUT            = 0.2 # Moved to config.py
# LSTM_EPOCHS             = 50 # Moved to config.py
# LSTM_BATCH_SIZE         = 64 # Moved to config.py
# LSTM_LEARNING_RATE      = 0.001 # Moved to config.py

# --- GRU Hyperparameter Search Ranges ---
# GRU_HIDDEN_SIZE_OPTIONS = [16, 32, 64, 128, 256] # Moved to config.py
# GRU_NUM_LAYERS_OPTIONS  = [1, 2, 3, 4] # Moved to config.py
# GRU_DROPOUT_OPTIONS     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # Moved to config.py
# GRU_LEARNING_RATE_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01] # Moved to config.py
# GRU_BATCH_SIZE_OPTIONS  = [16, 32, 64, 128, 256] # Moved to config.py
# GRU_EPOCHS_OPTIONS      = [10, 30, 50, 70, 100] # Moved to config.py
# GRU_CLASS_HORIZON_OPTIONS = [1, 2, 3, 4, 5, 7, 10, 15, 20] # New: Options for class_horizon # Moved to config.py
# GRU_TARGET_PERCENTAGE_OPTIONS = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05] # New: Options for target_percentage # Moved to config.py
# ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = True # Set to True to enable GRU hyperparameter search # Moved to config.py

# --- Misc
# INITIAL_BALANCE         = 100_000.0 # Moved to config.py
# SAVE_PLOTS              = True # Moved to config.py
# FORCE_TRAINING          = True      # Set to True to force re-training of ML models # Moved to config.py
# CONTINUE_TRAINING_FROM_EXISTING = False # Set to True to load existing models and continue training # Moved to config.py
# FORCE_THRESHOLDS_OPTIMIZATION = True # Set to True to force re-optimization of ML thresholds # Moved to config.py
# FORCE_PERCENTAGE_OPTIMIZATION = True # Set to True to force re-optimization of TARGET_PERCENTAGE # Moved to config.py

# ============================
# Helpers (Moved to specialized modules - data_fetcher.py, data_utils.py, etc.)
# ============================
# All helper functions have been moved to their respective modules and are imported above.

# ============================
# Feature prep & model (Moved to data_utils.py and ml_models.py)
# ============================
# These functions are imported from their respective modules.

# ============================
# All duplicate functions removed and moved to specialized modules:
# ============================
# - _calculate_technical_indicators (now in data_utils.py)
# - train_and_evaluate_models (now in ml_models.py)
# - alpha_opt_threshold_for_df (now in alpha_integration.py)
# - backtest_worker (now in backtesting.py)
# - analyze_performance (now in backtesting.py)
# - find_top_performers (now in ticker_selection.py)
# - All data fetching functions (now in data_fetcher.py)
# - All ticker selection functions (now in ticker_selection.py)

def get_internet_time():
    """
    Get accurate UTC time from reliable internet sources for consistent backtesting.
    Uses simple, fast APIs with quick fallback to local time.
    """
    import requests
    from datetime import datetime, timezone

    # Single, reliable time source (Google's public NTP service via HTTP)
    # This is more reliable than the previous APIs
    try:
        # Use httpbin.org which provides current time in a simple format
        response = requests.get("https://httpbin.org/get", timeout=5)
        response.raise_for_status()
        # httpbin returns current time, but we can also use a timestamp approach
        # For simplicity, if we get a response, assume internet is working and use local time
        # This avoids complex parsing while still verifying internet connectivity

        local_time = datetime.now(timezone.utc)
        print(f"[INTERNET] Connection verified, using local time with internet sync: {local_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return local_time

    except Exception as e:
        print(f"[OFFLINE] Internet time check failed ({str(e)[:40]}...), using local system time")

    # Always return local time - the important thing is consistency, not perfect accuracy
    local_time = datetime.now(timezone.utc)
    print(f"[LOCAL] Using system time: {local_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    return local_time

def main(
    fcf_threshold: float = None,
    ebitda_threshold: float = None,
    class_horizon: int = PERIOD_HORIZONS.get("1-Year", 10),  # 10 calendar days from config
    top_performers_data=None,
    feature_set: Optional[List[str]] = None,
    run_parallel: bool = True,
    single_ticker: Optional[str] = None,
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[Dict]]:
    
    # Set the start method for multiprocessing to 'spawn'
    # This is crucial for CUDA compatibility with multiprocessing
    try:
        if PYTORCH_AVAILABLE:
            import torch
            from config import PYTORCH_USE_GPU
            if not PYTORCH_USE_GPU:
                print("🖥️  PYTORCH_USE_GPU=False - PyTorch models will run on CPU (allows higher parallelism)")
                import multiprocessing
                multiprocessing.set_start_method('spawn', force=True)
            elif torch.cuda.is_available():
                print("🎮 PYTORCH_USE_GPU=True - PyTorch models will use CUDA")
                import multiprocessing
                multiprocessing.set_start_method('spawn', force=True)
                print("✅ Multiprocessing start method set to 'spawn' for CUDA compatibility.")
            else:
                print("🖥️  No GPU detected - PyTorch models will run on CPU")
    except RuntimeError as e:
        print(f"⚠️ Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA and multiprocessing.")

    # Get accurate time from internet source for consistent backtesting
    end_date = get_internet_time()
    
    # IMPORTANT: Use last trading day, not today's date (to avoid holiday issues)
    from data_utils import _get_last_trading_day
    last_trading_day = _get_last_trading_day()
    bt_end = datetime.combine(last_trading_day, datetime.min.time(), tzinfo=timezone.utc)
    
    # Update end_date to last trading day for consistency
    end_date = bt_end
    
    print(f"📅 Today: {get_internet_time().date()}, Last Trading Day: {last_trading_day}")
    print(f"📅 Using last trading day for backtesting: {bt_end.date()}")
    
    # Store TODAY's date for data fetching (always fetch up to current date)
    today_for_data_fetch = end_date
    
        
    # Track timing for notifications
    script_start_time = time.time()
    training_start_time = None
    backtest_start_time = None
    
    alpaca_trading_client = None

    # Initialize ML libraries to determine CUDA availability
    initialize_ml_libraries()
    
    # Note: multiprocessing with CUDA-enabled DL models uses 'spawn' start method for stability
    if PYTORCH_AVAILABLE and CUDA_AVAILABLE and (USE_LSTM or USE_GRU):
        print("✅ CUDA + DL enabled: multiprocessing ON with 'spawn' start method for stability.")
        run_parallel = True
    
    # Initialize initial_balance_used here with a default value
    initial_balance_used = INITIAL_BALANCE 
    print(f"Using initial balance: ${initial_balance_used:,.2f}")

    # Initialize filtered ticker lists to avoid UnboundLocalError
    top_tickers_1y_filtered = []

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

        # YTD performance calculation removed since YTD support was removed
        top_performers_data = [(single_ticker, perf_1y)]
    
    # --- Step 1: Get all tickers and perform a single, comprehensive data download ---
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        print(" No tickers found from market selection. Aborting.")
        return (None,) * 15

    # Use shared data loading function (same for backtesting and live trading)
    print(f"🚀 Step 1: Loading market data...")
    from data_utils import load_all_market_data
    all_tickers_data = load_all_market_data(all_available_tickers, end_date=today_for_data_fetch)
    
    if all_tickers_data.empty:
        print("❌ Data loading failed. Aborting.")
        return (None,) * 15
    
    # Data is in long format with date and ticker columns
    print(f"   📊 Data shape: {all_tickers_data.shape}")
    print(f"   📊 Columns: {list(all_tickers_data.columns[:5])}")
    
    # Expand all_available_tickers to include any extra tickers found in downloaded data
    if 'ticker' in all_tickers_data.columns:
        tickers_in_data = set(all_tickers_data['ticker'].unique())
        extra_tickers = tickers_in_data - set(all_available_tickers)
        if extra_tickers:
            print(f"   ℹ️ Found {len(extra_tickers)} extra tickers in data not in original list: {sorted(extra_tickers)[:10]}{'...' if len(extra_tickers) > 10 else ''}")
            all_available_tickers = sorted(set(all_available_tickers) | extra_tickers)
            print(f"   ℹ️ Expanded ticker universe to {len(all_available_tickers)} tickers")
    else:
        print("   ⚠️ No 'ticker' column found in data")
        return (None,) * 15
    
    # Filter out delisted stocks (no recent data within last 30 days)
    print(" Filtering out delisted stocks (no recent data)...")
    cutoff_date = today_for_data_fetch - timedelta(days=30)
    
    # Find tickers with data in the last 30 days
    recent_data = all_tickers_data[all_tickers_data['date'] >= cutoff_date]
    active_tickers = recent_data['ticker'].unique().tolist()
    
    # Remove tickers without recent data
    all_tickers_before = all_tickers_data['ticker'].nunique()
    all_tickers_data = all_tickers_data[all_tickers_data['ticker'].isin(active_tickers)]
    all_tickers_after = all_tickers_data['ticker'].nunique()
    
    delisted_count = all_tickers_before - all_tickers_after
    if delisted_count > 0:
        print(f"   ✅ Filtered out {delisted_count} delisted/stale tickers (no data in last 30 days)")
        print(f"   ✅ Remaining: {all_tickers_after} active tickers")
    else:
        print(f"   ✅ All {all_tickers_after} tickers have recent data")
    
    # Ensure 'date' column is timezone-aware
    if 'date' in all_tickers_data.columns:
        if all_tickers_data['date'].dtype == 'object' or not hasattr(all_tickers_data['date'].iloc[0], 'tzinfo'):
            all_tickers_data['date'] = pd.to_datetime(all_tickers_data['date'], utc=True)
        elif all_tickers_data['date'].dt.tz is None:
            all_tickers_data['date'] = all_tickers_data['date'].dt.tz_localize('UTC')
        else:
            all_tickers_data['date'] = all_tickers_data['date'].dt.tz_convert('UTC')
    
    print("✅ Comprehensive data download complete.")

    # Cap bt_end to the latest available data to avoid future-dated slices
    if 'date' in all_tickers_data.columns:
        last_available = pd.to_datetime(all_tickers_data['date'].max())
    else:
        last_available = all_tickers_data.index.max()
    
    if last_available < bt_end:
        print(f"ℹ️ Capping backtest end date from {bt_end.date()} to last available data {last_available.date()}")
        end_date = last_available
        bt_end = last_available
    
    # Calculate initial backtest dates (needed for all conditional blocks)
    # Note: BACKTEST_DAYS only affects the backtest period, NOT data fetching range
    bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)  # When stocks will be bought
    
    # Import RETRAIN_FREQUENCY_DAYS to ensure it's available
    from config import RETRAIN_FREQUENCY_DAYS
    
    # Control backtest end date based on config
    if RUN_BACKTEST_UNTIL_TODAY:
        # Run backtest until today (ignoring prediction horizon constraint)
        print(f"📅 Data available until: {bt_end.date()}")
        print(f"📅 Running backtest until today: {bt_end.date()} (prediction horizon constraint disabled)")
        # Keep bt_end as is (today's date or last available data)
        training_end_date = bt_end
    else:
        # ✅ FIX: Subtract prediction horizon from backtest end to ensure future data availability
        # If prediction horizon is 10 days and last data is Jan 23, backtest should end Jan 13
        # This ensures we have enough future data to validate all predictions made during backtest
        prediction_horizon = PERIOD_HORIZONS.get("1-Year", 10)  # 10 calendar days from config
        bt_end_with_horizon = bt_end - timedelta(days=prediction_horizon)
        
        print(f"📅 Data available until: {bt_end.date()}")
        print(f"📅 Prediction horizon: {prediction_horizon} days")
        print(f"📅 Backtest will end: {bt_end_with_horizon.date()} (ensuring {prediction_horizon} days of future data for validation)")
        
        bt_end = bt_end_with_horizon
        end_date = bt_end_with_horizon
        training_end_date = bt_end_with_horizon

    # --- Fetch SPY data for Market Momentum feature ---
    # Use shared function to calculate start date
    from config import get_data_lookback_days
    data_start_date = today_for_data_fetch - timedelta(days=get_data_lookback_days())
    
    print("🔍 Fetching SPY data for Market Momentum feature...")
    spy_df = load_prices_robust('SPY', data_start_date, today_for_data_fetch)
    if not spy_df.empty:
        spy_df['SPY_Returns'] = spy_df['Close'].pct_change()
        spy_df['Market_Momentum_SPY'] = spy_df['SPY_Returns'].rolling(window=FEAT_VOL_WINDOW).mean()
        spy_df = spy_df[['Market_Momentum_SPY']].reset_index()
        spy_df.columns = ['date', 'Market_Momentum_SPY']
        
        # ✅ FIX 2: Merge SPY data on 'date' column (long format)
        if 'date' in all_tickers_data.columns:
            all_tickers_data = all_tickers_data.merge(spy_df, on='date', how='left')
            # Forward fill and then back fill any NaNs introduced by the merge
            all_tickers_data['Market_Momentum_SPY'] = all_tickers_data['Market_Momentum_SPY'].ffill().bfill().fillna(0)
            print("✅ SPY Market Momentum data fetched and merged.")
        else:
            print("⚠️ 'date' column not found in all_tickers_data. Skipping SPY merge.")
    else:
        print("⚠️ Could not fetch SPY data. Market Momentum feature will be 0.")
        # Add a zero-filled column if SPY data couldn't be fetched
        all_tickers_data['Market_Momentum_SPY'] = 0.0

    # --- Fetch and merge intermarket data ---
    print("🔍 Fetching intermarket data...")
    intermarket_df = _fetch_intermarket_data(data_start_date, today_for_data_fetch)
    if not intermarket_df.empty:
        # ✅ FIX 3: Ensure intermarket_df index is timezone-aware before merge
        if intermarket_df.index.tzinfo is None:
            intermarket_df.index = intermarket_df.index.tz_localize('UTC')
        else:
            intermarket_df.index = intermarket_df.index.tz_convert('UTC')
        
        intermarket_df = intermarket_df.reset_index()
        intermarket_df.columns = ['date'] + [f'Intermarket_{col}' for col in intermarket_df.columns[1:]]
        
        # Merge intermarket data on 'date' column (long format)
        if 'date' in all_tickers_data.columns:
            all_tickers_data = all_tickers_data.merge(intermarket_df, on='date', how='left')
            # Forward fill and then back fill any NaNs introduced by the merge
            for col in intermarket_df.columns[1:]:  # Skip 'date' column
                all_tickers_data[col] = all_tickers_data[col].ffill().bfill().fillna(0)
            print("✅ Intermarket data fetched and merged.")
        else:
            print("⚠️ 'date' column not found in all_tickers_data. Skipping intermarket merge.")
    else:
        print("⚠️ Could not fetch intermarket data. Intermarket features will be 0.")
        # Add zero-filled columns for intermarket features to ensure feature set consistency
        for col_name in ['VIX_Index_Returns', 'DXY_Index_Returns', 'Gold_Futures_Returns', 'Oil_Futures_Returns', 'US10Y_Yield_Returns', 'Oil_Price_Returns', 'Gold_Price_Returns']:
            feature_col = f'Intermarket_{col_name}'
            if feature_col not in all_tickers_data.columns:
                all_tickers_data[feature_col] = 0.0
    # ✅ FIX 6: Add data validation before proceeding
    print("\n🔍 Validating data structure...")
    if 'ticker' not in all_tickers_data.columns:
        print("❌ ERROR: 'ticker' column not found in all_tickers_data after conversion!")
        print(f"   Available columns: {list(all_tickers_data.columns)}")
        return (None,) * 15
    
    if 'date' not in all_tickers_data.columns:
        print("❌ ERROR: 'date' column not found in all_tickers_data after conversion!")
        print(f"   Available columns: {list(all_tickers_data.columns)}")
        return (None,) * 15
    
    # Check for required OHLCV columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in all_tickers_data.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(all_tickers_data.columns)}")
        return (None,) * 15
    
    print(f"✅ Data validation passed:")
    print(f"   - Shape: {all_tickers_data.shape}")
    print(f"   - Tickers: {len(all_tickers_data['ticker'].unique())}")
    print(f"   - Date range: {all_tickers_data['date'].min()} to {all_tickers_data['date'].max()}")
    print(f"   - Columns: {list(all_tickers_data.columns)}")
    
    # ✅ OPTIMIZED: Generate data quality diagnostics efficiently
    print("\n🔍 Analyzing data quality for each ticker...")
    
    # For large ticker lists (>1000), use fast aggregation without detailed checks
    if len(all_available_tickers) > 1000:
        print(f"   📊 Fast validation for {len(all_available_tickers)} tickers...")
        
        # Quick aggregation: count rows per ticker
        ticker_counts = all_tickers_data.groupby('ticker').size()
        
        # Use parallel processing for large ticker sets
        if len(all_available_tickers) > PARALLEL_THRESHOLD:
            print(f"   Using parallel validation for {len(all_available_tickers)} tickers...")
            from multiprocessing import Pool
            import functools
            
            # Add progress bar
            progress_bar = tqdm(total=len(all_available_tickers), desc="   [INFO] Validating tickers", ncols=100)
            
            # Partial function to pass ticker_counts
            validate_func = functools.partial(validate_single_ticker, ticker_counts=ticker_counts)
            
            num_workers = min(NUM_PROCESSES, len(all_available_tickers))
            with Pool(processes=num_workers) as pool:
                if progress_bar:
                    data_summaries = []
                    for result in pool.imap_unordered(validate_func, all_available_tickers):
                        data_summaries.append(result)
                        progress_bar.update(1)
                    progress_bar.close()
                else:
                    data_summaries = list(pool.imap(validate_func, all_available_tickers))
        else:
            # Sequential for small lists
            data_summaries = [validate_single_ticker(ticker, ticker_counts) 
                             for ticker in tqdm(all_available_tickers, desc="   [INFO] Quick validation", ncols=100)]
        
        # Print summary stats
        ok_count = sum(1 for s in data_summaries if s['status'] == 'OK')
        insufficient_count = sum(1 for s in data_summaries if s['status'] == 'INSUFFICIENT')
        empty = sum(1 for s in data_summaries if s['status'] == 'EMPTY')
        
        print(f"\n   ✅ Validation complete: {ok_count} OK, {insufficient_count} insufficient, {empty} empty")
        print(f"   📊 Overall data: {all_tickers_data.shape[0]} rows, {len(all_available_tickers)} tickers\n")
        
        # Show problematic tickers (INSUFFICIENT or EMPTY)
        problematic = [s for s in data_summaries if s['status'] in ['INSUFFICIENT', 'EMPTY', 'ERROR']]
        if problematic:
            print(f"   ⚠️  Tickers with issues ({len(problematic)} total):")
            print("   " + "=" * 90)
            print(f"   {'Ticker':<10} {'Rows':<8} {'Status':<15} {'Message':<40}")
            print("   " + "-" * 90)
            
            # Show first 50 problematic tickers
            for i, s in enumerate(problematic[:50]):
                print(f"   {s['ticker']:<10} {s['rows']:<8} {s['status']:<15} {s.get('message', ''):<40}")
            
            if len(problematic) > 50:
                print(f"   ... and {len(problematic) - 50} more problematic tickers")
            print("   " + "=" * 90)
            print()
        
        # Warn if many tickers have insufficient data
        if insufficient_count > len(data_summaries) * 0.3:
            print(f"   ⚠️  WARNING: {insufficient_count}/{len(data_summaries)} tickers have insufficient data!")
            print(f"   💡 Consider using a longer data period or filtering these tickers\n")
    else:
        # For smaller lists, use optimized groupby approach
        data_summaries = []
        
        # Optimize: Group by ticker once (much faster than filtering repeatedly)
        global _grouped_data
        _grouped_data = all_tickers_data.groupby('ticker')
        
        # Use parallel processing for large ticker sets
        if len(all_available_tickers) > PARALLEL_THRESHOLD:
            print(f"   Using parallel detailed validation for {len(all_available_tickers)} tickers...")
            from concurrent.futures import ThreadPoolExecutor
            
            num_workers = min(NUM_PROCESSES, len(all_available_tickers))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                data_summaries = list(tqdm(
                    executor.map(validate_detailed_ticker, all_available_tickers),
                    total=len(all_available_tickers),
                    desc="Validating data quality",
                    ncols=100
                ))
        else:
            # Sequential for small lists
            data_summaries = []
            for ticker in tqdm(all_available_tickers, desc="Validating data quality"):
                data_summaries.append(validate_detailed_ticker(ticker))
        
        # Clean up global variable
        _grouped_data = None
        
        # Print diagnostics table (includes warnings)
        print_data_diagnostics(data_summaries)
    
    # --- Calculate backtest dates first (needed for ticker selection) ---
    # bt_end and last_available are already calculated above (lines 634-643)
    # No need to recalculate - use the existing values
    
    # --- Identify top performers if not provided ---
    if top_performers_data is None:
        title = "🚀 AI-Powered Momentum & Trend Strategy"
        # ... (rest of the title and filter logic remains the same)
        print(title + "\n" + "="*50 + "\n")

        print("🔍 Step 2: Identifying stocks outperforming market benchmarks...")
        print(f"  📅 Using latest available performance data for ticker selection")

        # Get top performers from market selection using the pre-fetched data
        # Use latest available data to select best current performers
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

    top_tickers = [ticker for ticker, _ in top_performers_data]
    print(f"\n✅ Identified {len(top_tickers)} stocks for backtesting: {', '.join(top_tickers)}\n")

    # --- Run Selection Strategy Comparison ---
    # Compare multiple selection strategies to see which would have picked the best performers
    print("\n" + "="*80)
    print("📊 Running Selection Strategy Comparison...")
    print("   This compares different stock selection criteria to find the best approach.")
    print("="*80)
    
    # Convert all_tickers_data to dict format for selection backtester
    ticker_data_dict = {}
    if isinstance(all_tickers_data, pd.DataFrame):
        if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
            # Long format - convert to dict
            for ticker in all_available_tickers:
                ticker_df = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                if not ticker_df.empty:
                    ticker_df = ticker_df.set_index('date')
                    ticker_data_dict[ticker] = ticker_df
        else:
            # Should not happen with new long format data
            print(f"   [WARN] Unexpected data format in main.py")
            print(f"   Available columns: {list(all_tickers_data.columns[:5])}")
    elif isinstance(all_tickers_data, dict):
        ticker_data_dict = all_tickers_data
    
    # Run the comparison
    # Use 20 tickers per strategy for meaningful comparison between strategies
    selection_comparison_results = run_selection_strategy_comparison(
        all_tickers_data=ticker_data_dict,
        all_available_tickers=all_available_tickers,
        selection_date=bt_start_1y,
        evaluation_date=bt_end,
        n_top=20,
        benchmark_ticker='SPY'
    )
    
    # Print stock overlap analysis
    if selection_comparison_results and 'strategy_results' in selection_comparison_results:
        print_strategy_stock_overlap(selection_comparison_results['strategy_results'])

    # Log skipped tickers
    skipped_tickers = set(all_available_tickers) - set(top_tickers)
    if skipped_tickers:
        _ensure_dir(Path("logs")) # Ensure logs directory exists
        with open("logs/skipped_tickers.log", "w") as f:
            f.write("Tickers skipped during performance analysis:\n")
            for ticker in sorted(list(skipped_tickers)):
                f.write(f"{ticker}\n")

    # --- Define training date variables (needed for backtest even when training disabled) ---
    # Train on data before backtest starts (clean temporal separation)
    train_end_1y = bt_start_1y - timedelta(days=1)

    # ✅ FIX: Calculate actual period name based on backtest length
    if BACKTEST_DAYS >= 250:  # ~1 year
        actual_period_name = "1-Year"
    elif BACKTEST_DAYS >= 125:  # ~6 months
        actual_period_name = "6-Month"
    elif BACKTEST_DAYS >= 90:  # ~4-5 months
        actual_period_name = "3-Month"
    elif BACKTEST_DAYS >= 60:  # ~3 months
        actual_period_name = "2-Month"
    elif BACKTEST_DAYS >= 30:  # ~1-2 months
        actual_period_name = "1-Month"
    else:
        actual_period_name = f"{BACKTEST_DAYS}-Day"

    # --- Training Models (for 1-Year Backtest) ---
    
    # Initialize variables that are used outside the if-block
    failed_training_tickers_1y = {}
    
    # --- Clean up old models before fresh training ---
    if FORCE_TRAINING:
        print(f"\n🧹 Step 2: Cleaning up old models before fresh training...")
        models_dir = Path("logs/models")
        if models_dir.exists():
            # Count old models
            old_model_files = list(models_dir.glob("*.joblib"))
            old_scaler_files = list(models_dir.glob("*_scaler.joblib"))
            old_y_scaler_files = list(models_dir.glob("*_y_scaler.joblib"))
            old_json_files = list(models_dir.glob("*.json"))
            total_old_files = len(old_model_files) + len(old_scaler_files) + len(old_y_scaler_files) + len(old_json_files)
            
            if total_old_files > 0:
                print(f"   🗑️  Found {total_old_files} old model files:")
                print(f"      - {len(old_model_files)} model files")
                print(f"      - {len(old_scaler_files)} scaler files")
                print(f"      - {len(old_y_scaler_files)} y-scaler files")
                print(f"      - {len(old_json_files)} hyperparameter files")
                
                # Remove all old model files
                import shutil
                shutil.rmtree(models_dir)
                models_dir.mkdir(exist_ok=True)
                print(f"   ✅ Deleted all old models. Fresh training will start.")
            else:
                print(f"   ℹ️  No old models found. Ready for fresh training.")
        else:
            print(f"   ℹ️  Models directory doesn't exist. Will create during training.")
    
    # --- Step 3: Load existing models or prepare for training ---
    print(f"\n🔄 Loading existing models...")
    training_results = []
    
    # Try to load existing models
    from prediction import load_models_for_tickers
    
    # Get all available tickers with models
    import os
    
    models_dir = Path("logs/models")
    available_model_files = []
    
    if models_dir.exists():
        for file in models_dir.glob("*_TargetReturn_model.joblib"):
            ticker = file.stem.replace("_TargetReturn_model", "")
            available_model_files.append(ticker)
    
    print(f"   📁 Found {len(available_model_files)} existing models")
    
    if available_model_files:
        print(f"   🔄 Loading models for {len(available_model_files)} tickers...")
        try:
            models, scalers, y_scalers = load_models_for_tickers(available_model_files)
            print(f"   ✅ Loaded {len(models)} models, {len(scalers)} scalers, {len(y_scalers)} y_scalers")
        except Exception as e:
            print(f"   ⚠️ Error loading models: {e}")
            models, scalers, y_scalers = {}, {}, {}
    else:
        print(f"   ⚠️ No existing models found - will train during walk-forward")
        models, scalers, y_scalers = {}, {}, {}
    
    # AI Strategy removed - always disabled
    from config import ENABLE_WALK_FORWARD_RETRAINING
    ai_strategy_available = False  # AI Strategy removed
    failed_training_tickers_1y = {}
    top_tickers_ai_filtered = top_tickers  # All tickers available for AI when trained
    
    # Send training completion notification
    if training_start_time:
        training_time_minutes = (time.time() - training_start_time) / 60
        trained_count = len(models)
        total_count = len(top_tickers) * 5  # Approximate (5 models per ticker)
        
        send_training_notification(
            models_trained=trained_count,
            total_models=total_count,
            training_time_minutes=training_time_minutes,
            failed_models=failed_training_tickers_1y if 'failed_training_tickers_1y' in locals() else None
        )
        
    # All tickers available for both AI and non-AI strategies
    # Models will be trained during walk-forward backtest
    top_tickers_ai_filtered = top_tickers
    top_tickers_1y_filtered = top_tickers
    
    print(f"  ✅ {len(top_tickers)} tickers available for all strategies: {', '.join(top_tickers[:10])}{'...' if len(top_tickers) > 10 else ''}", flush=True)

    # --- Skip initial AI training - will be trained during walk-forward backtest ---
    print(f"\n⏭️ Skipping initial AI training - will be trained during walk-forward backtest")

    # 🧠 Initialize dictionaries for model training data before threshold optimization
    X_train_dict, y_train_dict, X_test_dict, y_test_dict = {}, {}, {}, {}
    prices_dict, signals_dict = {}, {}
    
    # Update top_performers_data to reflect only successfully trained tickers (for AI strategies only)
    # Non-AI strategies use the full top_performers_data
    top_performers_data_1y_filtered = [item for item in top_performers_data if item[0] in top_tickers_1y_filtered]
    
    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1y = INVESTMENT_PER_STOCK
    # Ensure logs directory exists for optimized parameters
    _ensure_dir(TOP_CACHE_PATH.parent)
    # Initialize optimized_params_per_ticker with default values (no optimization)
    optimized_params_per_ticker = {}
    for ticker in top_tickers_1y_filtered:
        optimized_params_per_ticker[ticker] = {
            'optimization_status': "Using defaults (no optimization)"
        }

    # Initialize all_tested_combinations (empty since no optimization)
    all_tested_combinations = {}

    print(f"\n✅ Using default parameters for all tickers (threshold optimization removed).")

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
    final_buy_hold_value_3m = initial_balance_used
    buy_hold_results_1y = []
    performance_metrics_buy_hold_1y_actual = [] # Initialize here

    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here




    # ========================================================================
    # PHASE 2: OPTIMIZE THRESHOLDS FOR ALL PERIODS
    # ========================================================================
    
    # Initialize optimized_params dictionaries (only 1-year period supported)
    # ========================================================================
    # PHASE 3: RUN ALL BACKTESTS
    # ========================================================================
    
    # Optional: Run rebalance optimization
    try:
        rebalance_optimization_results = optimize_rebalance_horizons(
            all_tickers_data=all_tickers_data,
            backtest_start=bt_start_1y,
            backtest_end=bt_end,
            initial_capital=capital_per_stock_1y * 3,  # Same as n_top_rebal
            portfolio_size=3,
            strategy_types=['1Y', '6M', '3M', '1M']
        )
    except Exception as e:
        print(f"   ⚠️ Rebalance optimization failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # DEBUG: Check what's in models dictionaries
    print(f"\n[DEBUG MAIN] 1-Year models keys: {list(models.keys())}")
    print(f"[DEBUG MAIN] 1-Year models values types: {[type(v).__name__ if v else 'None' for v in models.values()]}")

    # AI Strategy removed - running comparison strategies only
    print(f"\n🔍 Step 8: Running {actual_period_name} Backtest (comparison strategies only)...")
    
    n_top_rebal = 3
    initial_capital_1y = capital_per_stock_1y * n_top_rebal
    
    # Use walk-forward backtest with periodic retraining and rebalancing
    try:
        result = _run_portfolio_backtest_walk_forward(
            all_tickers_data=all_tickers_data,
            backtest_start_date=bt_start_1y,
            backtest_end_date=bt_end,
            initial_top_tickers=top_tickers,
            initial_models=models,
            initial_scalers=scalers,
            initial_y_scalers=y_scalers,
            capital_per_stock=capital_per_stock_1y,
            period_name=actual_period_name,
            top_performers_data=top_performers_data,
            horizon_days=PERIOD_HORIZONS.get("1-Year", 10),  # 10 calendar days from config
            enable_ai_strategy=False  # AI Strategy removed
        )
        
        if result is None:
            print("\n" + "="*100)
            print("❌ CRITICAL ERROR: _run_portfolio_backtest_walk_forward returned None!")
            print("="*100)
            raise ValueError("Backtest function returned None instead of expected dict")
        
        # Results are now a clean dict - no more fragile 149-value tuple unpacking
        r = result  # results dict from backtesting.py
        s = r['strategies']  # shorthand for strategy results
        
        # Extract general values
        final_strategy_value_1y = r['general']['final_strategy_value']
        portfolio_values_1y = r['general']['portfolio_values_history']
        processed_tickers_1y = r['general']['processed_tickers']
        performance_metrics_1y = r['general']['performance_metrics']
        buy_hold_histories_1y = r['general']['buy_hold_histories']
        day_count_1y = r['general']['day_count']
        
        # Extract strategy portfolio values (used for print_final_summary and debug)
        bh_portfolio_value_1y = s['static_bh_1y']['value']
        bh_3m_portfolio_value_1y = s['static_bh_3m']['value']
        bh_6m_portfolio_value_1y = s['static_bh_6m']['value']
        bh_1m_portfolio_value_1y = s['static_bh_1m']['value']
        
    except Exception as e:
        print("\n" + "="*100)
        print("EXCEPTION CAUGHT IN MAIN.PY:")
        print("="*100)
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("="*100 + "\n")
        raise
    
    # AI Strategy removed - compute return using portfolio initial capital for consistency
    portfolio_initial_capital_for_ret = INVESTMENT_PER_STOCK * PORTFOLIO_SIZE
    ai_1y_return = ((final_strategy_value_1y - portfolio_initial_capital_for_ret) / portfolio_initial_capital_for_ret) * 100 if portfolio_initial_capital_for_ret != 0 else 0.0
    prediction_stats_1y = {}  # Not provided by this function
    strategy_results_1y = []  # Per-ticker results in performance_metrics_1y
    
    # 🔍 DEBUG: Check backtest results
    print(f"\n[DEBUG] 1-Year Backtest Results:")
    print(f"  - top_tickers_1y_filtered: {top_tickers_1y_filtered}")
    print(f"  - final_strategy_value_1y: ${final_strategy_value_1y:,.2f}")
    print(f"  - processed_tickers_1y: {processed_tickers_1y}")
    print(f"  - strategy_results_1y count: {len(strategy_results_1y)}")
    print(f"  - ai_1y_return: {ai_1y_return:.2f}%\n")

    # --- Rule-Based Strategy for 1-Year ---
    # Rule-based strategy disabled (removed during refactoring)
    rule_results_1y = {}  # Placeholder - rule-based strategy disabled
    final_rule_value_1y = rule_results_1y.get('final_value', initial_capital_1y)
    rule_1y_return = rule_results_1y.get('total_return', 0) * 100
    
    # Simple Rule Strategy removed - using AI strategy only
    final_simple_rule_value_1y = None
    simple_rule_results_1y = []
    performance_metrics_simple_rule_1y = []
    simple_rule_1y_return = None

    # --- Calculate Buy & Hold ---
    print(f"\n📊 Calculating Buy & Hold performance for {actual_period_name} period...")
    print(f"   Processing {len(top_tickers)} tickers (using cached data)...")
    buy_hold_results_1y = []
    performance_metrics_buy_hold_1y_actual = []
    for idx, ticker in enumerate(top_tickers):
        if (idx + 1) % 50 == 0:
            print(f"   [{idx+1}/{len(top_tickers)}] Processed...")
        # Use cached data instead of re-fetching
        # ✅ FIX: Handle both dict and DataFrame (long format) data structures
        if isinstance(all_tickers_data, dict):
            if ticker in all_tickers_data and not all_tickers_data[ticker].empty:
                df_full = all_tickers_data[ticker]
                df_bh = df_full[(df_full.index >= bt_start_1y) & (df_full.index <= bt_end)].copy()
            else:
                df_bh = pd.DataFrame()
        else:
            # DataFrame in long format - filter by ticker
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if not ticker_data.empty:
                ticker_data = ticker_data.set_index('date')
                df_bh = ticker_data[(ticker_data.index >= bt_start_1y) & (ticker_data.index <= bt_end)].copy()
            else:
                df_bh = pd.DataFrame()
        if not df_bh.empty:
            start_price = float(df_bh["Close"].iloc[0])
            shares_bh = int(capital_per_stock_1y / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock_1y - shares_bh * start_price
            
            bh_history_for_ticker = []
            for price_day in df_bh["Close"].tolist():
                bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
            
            final_bh_val_ticker = bh_history_for_ticker[-1] if bh_history_for_ticker else capital_per_stock_1y
            buy_hold_results_1y.append(final_bh_val_ticker)
            
            perf_data_bh = calculate_buy_hold_performance_metrics(bh_history_for_ticker, ticker)
            performance_metrics_buy_hold_1y_actual.append({
                'ticker': ticker,
                'final_val': final_bh_val_ticker,
                'perf_data': perf_data_bh,
                'individual_bh_return': ((final_bh_val_ticker - capital_per_stock_1y) / abs(capital_per_stock_1y)) * 100 if capital_per_stock_1y != 0 else 0.0,
                'final_shares': shares_bh
            })
        else:
            buy_hold_results_1y.append(capital_per_stock_1y)
            performance_metrics_buy_hold_1y_actual.append({
                'ticker': ticker,
                'final_val': capital_per_stock_1y,
                'perf_data': {'sharpe_ratio': np.nan, 'max_drawdown': np.nan},
                'individual_bh_return': 0.0,
                'final_shares': 0.0
            })
    
    # Build prediction vs B&H rows for all candidate tickers (whether AI held them or not)
    bh_returns_lookup = {d['ticker']: d.get('individual_bh_return') for d in performance_metrics_buy_hold_1y_actual}
    prediction_vs_bh_1y = []
    for ticker in top_tickers_1y_filtered:
        stats = prediction_stats_1y.get(ticker, {}) if 'prediction_stats_1y' in locals() else {}
        prediction_vs_bh_1y.append({
            'ticker': ticker,
            'pred_mean_pct': stats.get('pred_mean_pct'),
            'pred_min_pct': stats.get('pred_min_pct'),
            'pred_max_pct': stats.get('pred_max_pct'),
            'bh_horizon_return_pct': stats.get('bh_horizon_return_pct'),
            'individual_bh_return': bh_returns_lookup.get(ticker)
        })
    # BH portfolio value is now calculated in the walk-forward backtest
    # It invests only in the top 3 highest performing stocks
    final_buy_hold_value_1y = bh_portfolio_value_1y
    final_buy_hold_value_3m = bh_3m_portfolio_value_1y
    print(f"✅ BH Portfolio (Top 3 Performers): ${final_buy_hold_value_1y:,.0f}")
    print(f"✅ {actual_period_name} Buy & Hold calculation complete.")




    # ========================================================================
    # PHASE 4: FINAL SUMMARY
    # ========================================================================

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    print("\n📝 Preparing final summary data...")
    final_results = []
    
    # Combine all failed tickers from all periods
    all_failed_tickers = {}

    # Initialize YTD/3month/1month failed tickers if not defined (when those periods are disabled)
    failed_training_tickers_ytd = failed_training_tickers_ytd if 'failed_training_tickers_ytd' in locals() else {}
    failed_training_tickers_3month = failed_training_tickers_3month if 'failed_training_tickers_3month' in locals() else {}
    failed_training_tickers_1month = failed_training_tickers_1month if 'failed_training_tickers_1month' in locals() else {}

    all_failed_tickers.update(failed_training_tickers_1y)
    all_failed_tickers.update(failed_training_tickers_ytd)
    all_failed_tickers.update(failed_training_tickers_3month)
    all_failed_tickers.update(failed_training_tickers_1month) # Add 1-month failed tickers

    # Distribute the portfolio-level final value across tickers for display (no per-ticker result in rebalancing mode)
    per_ticker_portfolio_value_1y = (final_strategy_value_1y / len(processed_tickers_1y)) if processed_tickers_1y else INVESTMENT_PER_STOCK

    # Add successfully processed tickers (✅ Updated for new tracking system)
    for ticker in processed_tickers_1y:
        backtest_result_for_ticker = next((res for res in performance_metrics_1y if res['ticker'] == ticker), None)
        
        if backtest_result_for_ticker:
            # ✅ NEW: Use new performance tracking structure
            strategy_gain = backtest_result_for_ticker.get('strategy_gain', 0.0)
            days_held = backtest_result_for_ticker.get('days_held', 0)
            max_shares = backtest_result_for_ticker.get('max_shares', 0.0)
            return_pct = backtest_result_for_ticker.get('return_pct', 0.0)
            total_invested = backtest_result_for_ticker.get('total_invested', INVESTMENT_PER_STOCK)
        else:
            # Fallback if no tracking data
            strategy_gain = 0.0
            days_held = 0
            max_shares = 0.0
            return_pct = 0.0
            total_invested = INVESTMENT_PER_STOCK

        # Get benchmark performance
        perf_1y_benchmark = np.nan
        ytd_perf_benchmark = np.nan
        for t, p1y in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': total_invested + strategy_gain,  # Final value
            'strategy_gain': strategy_gain,  # ✅ NEW: Actual gain
            'sharpe': 0.0,  # Not calculated in walk-forward
            'one_year_perf': perf_1y_benchmark,
            'ytd_perf': ytd_perf_benchmark,
            'individual_bh_return': return_pct,
            'last_ai_action': f"Held {days_held}d",  # Show days held
            'buy_prob': 0.0,  # Not used in walk-forward
            'sell_prob': 0.0,  # Not used in walk-forward
            'final_shares': max_shares,  # ✅ NEW: Show max shares held
            'days_held': days_held,  # ✅ NEW
            'total_invested': total_invested,  # ✅ NEW
            'status': 'trained',
            'reason': None
        })
    
    # Add failed tickers to the final results (only if they didn't succeed in 1Y)
    # Don't add tickers that already appear in processed_tickers_1y
    for ticker, reason in all_failed_tickers.items():
        if ticker not in processed_tickers_1y:  # ✅ FIX: Skip if ticker succeeded in 1Y
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
    
    # 🔍 DEBUG: Check values before summary
    print(f"\n[DEBUG] Before print_final_summary:")
    print(f"  - final_strategy_value_1y: ${final_strategy_value_1y:,.2f}")
    print(f"  - final_buy_hold_value_1y: ${final_buy_hold_value_1y:,.2f}")
    print(f"  - performance_metrics_buy_hold_1y_actual type: {type(performance_metrics_buy_hold_1y_actual)}")
    print(f"  - performance_metrics_buy_hold_1y_actual length: {len(performance_metrics_buy_hold_1y_actual) if isinstance(performance_metrics_buy_hold_1y_actual, list) else 'N/A'}")
    print(f"  - top_tickers count: {len(top_tickers)}")
    print(f"  - processed_tickers_1y count: {len(processed_tickers_1y)}")
    print(f"  - top_tickers sample: {top_tickers[:5]}")
    print(f"  - processed_tickers_1y sample: {processed_tickers_1y[:5] if processed_tickers_1y else 'None'}")

    # ✅ FIX: Use the portfolio initial capital (10 stocks), not the AI comparison capital (3 stocks)
    portfolio_initial_capital = INVESTMENT_PER_STOCK * PORTFOLIO_SIZE
    actual_initial_capital_1y = portfolio_initial_capital
    # ✅ FIX: Use ALL top tickers for the summary, not just AI-processed ones
    actual_tickers_analyzed = len(top_tickers)
    
    # ✅ Calculate actual backtest days for annualization (use actual day count from backtest)
    actual_backtest_days = (bt_end - bt_start_1y).days
    print(f"   📅 Actual backtest days: {actual_backtest_days} (config BACKTEST_DAYS={BACKTEST_DAYS})")
    
    # Helper: get return % for a strategy from the results dict
    def _ret(name):
        v = s[name]['value']
        return ((v - portfolio_initial_capital) / abs(portfolio_initial_capital)) * 100 if portfolio_initial_capital != 0 else 0.0
    
    print_final_summary(
        sorted_final_results, models, scalers, optimized_params_per_ticker,
        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
        actual_initial_capital_1y,
        actual_tickers_analyzed,
        performance_metrics_buy_hold_1y_actual,
        top_performers_data,
        final_buy_hold_value_3m=final_buy_hold_value_3m,
        final_static_bh_1m_value_1y=s['static_bh_1m']['value'],
        static_bh_1m_1y_return=_ret('static_bh_1m'),
        final_static_bh_6m_value_1y=s['static_bh_6m']['value'],
        static_bh_6m_1y_return=_ret('static_bh_6m'),
        final_dynamic_bh_value_1y=s['dynamic_bh_1y']['value'],
        dynamic_bh_1y_return=_ret('dynamic_bh_1y'),
        final_dynamic_bh_3m_value_1y=s['dynamic_bh_3m']['value'],
        dynamic_bh_3m_1y_return=_ret('dynamic_bh_3m'),
        final_dynamic_bh_6m_value_1y=s['dynamic_bh_6m']['value'],
        dynamic_bh_6m_1y_return=_ret('dynamic_bh_6m'),
        final_dynamic_bh_1m_value_1y=s['dynamic_bh_1m']['value'],
        dynamic_bh_1m_1y_return=_ret('dynamic_bh_1m'),
        final_risk_adj_mom_value_1y=s['risk_adj_mom']['value'],
        risk_adj_mom_1y_return=_ret('risk_adj_mom'),
        final_multitask_value_1y=s['multitask']['value'],
        multitask_1y_return=_ret('multitask'),
        final_mean_reversion_value_1y=s['mean_reversion']['value'],
        mean_reversion_1y_return=_ret('mean_reversion'),
        final_quality_momentum_value_1y=s['quality_momentum']['value'],
        quality_momentum_1y_return=_ret('quality_momentum'),
        final_volatility_adj_mom_value_1y=s['volatility_adj_mom']['value'],
        volatility_adj_mom_1y_return=_ret('volatility_adj_mom'),
        final_momentum_ai_hybrid_value_1y=s['momentum_ai_hybrid']['value'],
        momentum_ai_hybrid_1y_return=_ret('momentum_ai_hybrid'),
        final_dynamic_bh_1y_vol_filter_value_1y=s['dynamic_bh_1y_vol_filter']['value'],
        dynamic_bh_1y_vol_filter_1y_return=_ret('dynamic_bh_1y_vol_filter'),
        final_dynamic_bh_1y_trailing_stop_value_1y=s['dynamic_bh_1y_trailing_stop']['value'],
        dynamic_bh_1y_trailing_stop_1y_return=_ret('dynamic_bh_1y_trailing_stop'),
        final_sector_rotation_value_1y=s['sector_rotation']['value'],
        sector_rotation_1y_return=_ret('sector_rotation'),
        final_ratio_3m_1y_value_1y=s['ratio_3m_1y']['value'],
        ratio_3m_1y_1y_return=_ret('ratio_3m_1y'),
        final_ratio_1y_3m_value_1y=s['ratio_1y_3m']['value'],
        ratio_1y_3m_1y_return=_ret('ratio_1y_3m'),
        final_momentum_volatility_hybrid_value_1y=s['momentum_volatility_hybrid']['value'],
        momentum_volatility_hybrid_1y_return=_ret('momentum_volatility_hybrid'),
        final_momentum_volatility_hybrid_6m_value_1y=s['momentum_volatility_hybrid_6m']['value'],
        momentum_volatility_hybrid_6m_1y_return=_ret('momentum_volatility_hybrid_6m'),
        final_momentum_volatility_hybrid_1y_value_1y=s['momentum_volatility_hybrid_1y']['value'],
        momentum_volatility_hybrid_1y_1y_return=_ret('momentum_volatility_hybrid_1y'),
        final_momentum_volatility_hybrid_1y3m_value_1y=s['momentum_volatility_hybrid_1y3m']['value'],
        momentum_volatility_hybrid_1y3m_1y_return=_ret('momentum_volatility_hybrid_1y3m'),
        final_price_acceleration_value_1y=s['price_acceleration']['value'],
        price_acceleration_1y_return=_ret('price_acceleration'),
        final_turnaround_value_1y=s['turnaround']['value'],
        turnaround_1y_return=_ret('turnaround'),
        # Transaction costs - all from dict
        ai_transaction_costs=s['ai_strategy']['costs'],
        static_bh_transaction_costs=s['static_bh_1y']['costs'],
        static_bh_6m_transaction_costs=s['static_bh_6m']['costs'],
        static_bh_3m_transaction_costs=s['static_bh_3m']['costs'],
        static_bh_1m_transaction_costs=s['static_bh_1m']['costs'],
        dynamic_bh_1y_transaction_costs=s['dynamic_bh_1y']['costs'],
        dynamic_bh_6m_transaction_costs=s['dynamic_bh_6m']['costs'],
        dynamic_bh_1y_vol_filter_transaction_costs=s['dynamic_bh_1y_vol_filter']['costs'],
        dynamic_bh_1y_trailing_stop_transaction_costs=s['dynamic_bh_1y_trailing_stop']['costs'],
        sector_rotation_transaction_costs=s['sector_rotation']['costs'],
        ratio_3m_1y_transaction_costs=s['ratio_3m_1y']['costs'],
        ratio_1y_3m_transaction_costs=s['ratio_1y_3m']['costs'],
        momentum_volatility_hybrid_transaction_costs=s['momentum_volatility_hybrid']['costs'],
        momentum_volatility_hybrid_6m_transaction_costs=s['momentum_volatility_hybrid_6m']['costs'],
        momentum_volatility_hybrid_1y_transaction_costs=s['momentum_volatility_hybrid_1y']['costs'],
        momentum_volatility_hybrid_1y3m_transaction_costs=s['momentum_volatility_hybrid_1y3m']['costs'],
        price_acceleration_transaction_costs=s['price_acceleration']['costs'],
        turnaround_transaction_costs=s['turnaround']['costs'],
        dynamic_bh_3m_transaction_costs=s['dynamic_bh_3m']['costs'],
        dynamic_bh_1m_transaction_costs=s['dynamic_bh_1m']['costs'],
        risk_adj_mom_transaction_costs=s['risk_adj_mom']['costs'],
        mean_reversion_transaction_costs=s['mean_reversion']['costs'],
        quality_momentum_transaction_costs=s['quality_momentum']['costs'],
        momentum_ai_hybrid_transaction_costs=s['momentum_ai_hybrid']['costs'],
        volatility_adj_mom_transaction_costs=s['volatility_adj_mom']['costs'],
        period_name=actual_period_name,
        backtest_days=actual_backtest_days,
        strategy_results_ytd=None,
        strategy_results_3month=None,
        strategy_results_1month=None,
        performance_metrics_buy_hold_ytd=None,
        performance_metrics_buy_hold_3month=None,
        performance_metrics_buy_hold_1month=None,
        prediction_vs_bh_1y=None,
        prediction_vs_bh_ytd=None,
        prediction_vs_bh_3month=None,
        prediction_vs_bh_1month=None,
        final_rule_value_1y=None,
        rule_1y_return=None,
        # Cash deployed - all from dict
        ai_cash_deployed=s['ai_strategy']['cash_deployed'],
        static_bh_cash_deployed=s['static_bh_1y']['cash_deployed'],
        static_bh_6m_cash_deployed=s['static_bh_6m']['cash_deployed'],
        static_bh_3m_cash_deployed=s['static_bh_3m']['cash_deployed'],
        static_bh_1m_cash_deployed=s['static_bh_1m']['cash_deployed'],
        dynamic_bh_1y_cash_deployed=s['dynamic_bh_1y']['cash_deployed'],
        dynamic_bh_6m_cash_deployed=s['dynamic_bh_6m']['cash_deployed'],
        dynamic_bh_3m_cash_deployed=s['dynamic_bh_3m']['cash_deployed'],
        dynamic_bh_1m_cash_deployed=s['dynamic_bh_1m']['cash_deployed'],
        risk_adj_mom_cash_deployed=s['risk_adj_mom']['cash_deployed'],
        mean_reversion_cash_deployed=s['mean_reversion']['cash_deployed'],
        quality_momentum_cash_deployed=s['quality_momentum']['cash_deployed'],
        volatility_adj_mom_cash_deployed=s['volatility_adj_mom']['cash_deployed'],
        momentum_ai_hybrid_cash_deployed=s['momentum_ai_hybrid']['cash_deployed'],
        dynamic_bh_1y_vol_filter_cash_deployed=s['dynamic_bh_1y_vol_filter']['cash_deployed'],
        dynamic_bh_1y_trailing_stop_cash_deployed=s['dynamic_bh_1y_trailing_stop']['cash_deployed'],
        multitask_cash_deployed=s['multitask']['cash_deployed'],
        sector_rotation_cash_deployed=s['sector_rotation']['cash_deployed'],
        ratio_3m_1y_cash_deployed=s['ratio_3m_1y']['cash_deployed'],
        ratio_1y_3m_cash_deployed=s['ratio_1y_3m']['cash_deployed'],
        momentum_volatility_hybrid_cash_deployed=s['momentum_volatility_hybrid']['cash_deployed'],
        momentum_volatility_hybrid_6m_cash_deployed=s['momentum_volatility_hybrid_6m']['cash_deployed'],
        momentum_volatility_hybrid_1y_cash_deployed=s['momentum_volatility_hybrid_1y']['cash_deployed'],
        momentum_volatility_hybrid_1y3m_cash_deployed=s['momentum_volatility_hybrid_1y3m']['cash_deployed'],
        price_acceleration_cash_deployed=s['price_acceleration']['cash_deployed'],
        turnaround_cash_deployed=s['turnaround']['cash_deployed'],
        # Ensemble strategy values
        final_adaptive_ensemble_value_1y=s['adaptive_ensemble']['value'],
        adaptive_ensemble_1y_return=_ret('adaptive_ensemble'),
        final_volatility_ensemble_value_1y=s['volatility_ensemble']['value'],
        volatility_ensemble_1y_return=_ret('volatility_ensemble'),
        final_ai_volatility_ensemble_value_1y=s['ai_volatility_ensemble']['value'],
        ai_volatility_ensemble_1y_return=_ret('ai_volatility_ensemble'),
        final_multi_tf_ensemble_value_1y=s['multi_tf_ensemble']['value'],
        multi_tf_ensemble_1y_return=_ret('multi_tf_ensemble'),
        final_correlation_ensemble_value_1y=s['correlation_ensemble']['value'],
        correlation_ensemble_1y_return=_ret('correlation_ensemble'),
        final_dynamic_pool_value_1y=s['dynamic_pool']['value'],
        dynamic_pool_1y_return=_ret('dynamic_pool'),
        final_sentiment_ensemble_value_1y=s['sentiment_ensemble']['value'],
        sentiment_ensemble_1y_return=_ret('sentiment_ensemble'),
        final_voting_ensemble_value_1y=s['voting_ensemble']['value'],
        voting_ensemble_1y_return=_ret('voting_ensemble'),
                # Ensemble transaction costs
        adaptive_ensemble_transaction_costs=s['adaptive_ensemble']['costs'],
        volatility_ensemble_transaction_costs=s['volatility_ensemble']['costs'],
        ai_volatility_ensemble_transaction_costs=s['ai_volatility_ensemble']['costs'],
        multi_tf_ensemble_transaction_costs=s['multi_tf_ensemble']['costs'],
        correlation_ensemble_transaction_costs=s['correlation_ensemble']['costs'],
        dynamic_pool_transaction_costs=s['dynamic_pool']['costs'],
        sentiment_ensemble_transaction_costs=s['sentiment_ensemble']['costs'],
        voting_ensemble_transaction_costs=s['voting_ensemble']['costs'],
                # Ensemble cash deployed
        adaptive_ensemble_cash_deployed=s['adaptive_ensemble']['cash_deployed'],
        volatility_ensemble_cash_deployed=s['volatility_ensemble']['cash_deployed'],
        ai_volatility_ensemble_cash_deployed=s['ai_volatility_ensemble']['cash_deployed'],
        multi_tf_ensemble_cash_deployed=s['multi_tf_ensemble']['cash_deployed'],
        correlation_ensemble_cash_deployed=s['correlation_ensemble']['cash_deployed'],
        dynamic_pool_cash_deployed=s['dynamic_pool']['cash_deployed'],
        sentiment_ensemble_cash_deployed=s['sentiment_ensemble']['cash_deployed'],
        voting_ensemble_cash_deployed=s['voting_ensemble']['cash_deployed'],
                # Rebalance horizon optimization results
        static_bh_1y_best_horizon=rebalance_optimization_results['1Y']['best_horizon'] if rebalance_optimization_results and '1Y' in rebalance_optimization_results else None,
        static_bh_6m_best_horizon=rebalance_optimization_results['6M']['best_horizon'] if rebalance_optimization_results and '6M' in rebalance_optimization_results else None,
        static_bh_3m_best_horizon=rebalance_optimization_results['3M']['best_horizon'] if rebalance_optimization_results and '3M' in rebalance_optimization_results else None,
        static_bh_1m_best_horizon=rebalance_optimization_results['1M']['best_horizon'] if rebalance_optimization_results and '1M' in rebalance_optimization_results else None,
        final_rule_value_ytd=None,
        rule_ytd_return=None,
        final_rule_value_3month=None,
        rule_3month_return=None,
        final_rule_value_1month=None,
        rule_1month_return=None,
        # New Advanced Strategies
        final_mom_accel_value_1y=s['mom_accel']['value'],
        mom_accel_1y_return=_ret('mom_accel'),
        mom_accel_transaction_costs=s['mom_accel']['costs'],
        mom_accel_cash_deployed=s['mom_accel']['cash_deployed'],
        final_concentrated_3m_value_1y=s['concentrated_3m']['value'],
        concentrated_3m_1y_return=_ret('concentrated_3m'),
        concentrated_3m_transaction_costs=s['concentrated_3m']['costs'],
        concentrated_3m_cash_deployed=s['concentrated_3m']['cash_deployed'],
        final_dual_mom_value_1y=s['dual_momentum']['value'],
        dual_mom_1y_return=_ret('dual_momentum'),
        dual_mom_transaction_costs=s['dual_momentum']['costs'],
        dual_mom_cash_deployed=s['dual_momentum']['cash_deployed'],
        final_trend_atr_value_1y=s['trend_atr']['value'],
        trend_atr_1y_return=_ret('trend_atr'),
        trend_atr_transaction_costs=s['trend_atr']['costs'],
        trend_atr_cash_deployed=s['trend_atr']['cash_deployed'],
        # Enhanced Volatility
        final_enhanced_volatility_value_1y=s['enhanced_volatility']['value'],
        enhanced_volatility_1y_return=_ret('enhanced_volatility'),
        enhanced_volatility_transaction_costs=s['enhanced_volatility']['costs'],
        enhanced_volatility_cash_deployed=s['enhanced_volatility']['cash_deployed'],
        # Elite Hybrid
        final_elite_hybrid_value_1y=s['elite_hybrid']['value'],
        elite_hybrid_1y_return=_ret('elite_hybrid'),
        elite_hybrid_transaction_costs=s['elite_hybrid']['costs'],
        elite_hybrid_cash=s['elite_hybrid']['cash_deployed'],
        # AI Elite
        final_ai_elite_value_1y=s['ai_elite']['value'],
        ai_elite_1y_return=_ret('ai_elite'),
        ai_elite_transaction_costs=s['ai_elite']['costs'],
        ai_elite_cash=s['ai_elite']['cash_deployed'],
        # Elite Risk
        final_elite_risk_value_1y=s['elite_risk']['value'],
        elite_risk_1y_return=_ret('elite_risk'),
        elite_risk_transaction_costs=s['elite_risk']['costs'],
        elite_risk_cash=s['elite_risk']['cash_deployed'],
        # Risk-Adj Mom 6M
        final_risk_adj_mom_6m_value_1y=s['risk_adj_mom_6m']['value'],
        risk_adj_mom_6m_1y_return=_ret('risk_adj_mom_6m'),
        risk_adj_mom_6m_transaction_costs=s['risk_adj_mom_6m']['costs'],
        risk_adj_mom_6m_cash=s['risk_adj_mom_6m']['cash_deployed'],
        # Risk-Adj Mom 3M
        final_risk_adj_mom_3m_value_1y=s['risk_adj_mom_3m']['value'],
        risk_adj_mom_3m_1y_return=_ret('risk_adj_mom_3m'),
        risk_adj_mom_3m_transaction_costs=s['risk_adj_mom_3m']['costs'],
        risk_adj_mom_3m_cash=s['risk_adj_mom_3m']['cash_deployed'],
        # Risk-Adj Mom 3M Monthly
        final_risk_adj_mom_3m_monthly_value_1y=s['risk_adj_mom_3m_monthly']['value'],
        risk_adj_mom_3m_monthly_1y_return=_ret('risk_adj_mom_3m_monthly'),
        risk_adj_mom_3m_monthly_transaction_costs=s['risk_adj_mom_3m_monthly']['costs'],
        risk_adj_mom_3m_monthly_cash=s['risk_adj_mom_3m_monthly']['cash_deployed'],
        # Risk-Adj Mom Sentiment
        final_risk_adj_mom_sentiment_value_1y=s['risk_adj_mom_sentiment']['value'],
        risk_adj_mom_sentiment_1y_return=_ret('risk_adj_mom_sentiment'),
        risk_adj_mom_sentiment_transaction_costs=s['risk_adj_mom_sentiment']['costs'],
        risk_adj_mom_sentiment_cash=s['risk_adj_mom_sentiment']['cash_deployed'],
        # Monthly rebalance variants
        final_bh_1y_monthly_value_1y=s['bh_1y_monthly']['value'],
        bh_1y_monthly_1y_return=_ret('bh_1y_monthly'),
        bh_1y_monthly_transaction_costs=s['bh_1y_monthly']['costs'],
        bh_1y_monthly_cash=s['bh_1y_monthly']['cash_deployed'],
        final_bh_6m_monthly_value_1y=s['bh_6m_monthly']['value'],
        bh_6m_monthly_1y_return=_ret('bh_6m_monthly'),
        bh_6m_monthly_transaction_costs=s['bh_6m_monthly']['costs'],
        bh_6m_monthly_cash=s['bh_6m_monthly']['cash_deployed'],
        final_bh_3m_monthly_value_1y=s['bh_3m_monthly']['value'],
        bh_3m_monthly_1y_return=_ret('bh_3m_monthly'),
        bh_3m_monthly_transaction_costs=s['bh_3m_monthly']['costs'],
        bh_3m_monthly_cash=s['bh_3m_monthly']['cash_deployed']
    )
    print("\n✅ Final summary prepared and printed.")

    # --- Print Meta-Strategy Results ---
    if 'meta_strategy_ml' in s:
        ml_val = s['meta_strategy_ml']['value']
        mom_val = s['meta_strategy_mom']['value']
        ml_ret = ((ml_val - portfolio_initial_capital) / abs(portfolio_initial_capital)) * 100 if portfolio_initial_capital != 0 else 0
        mom_ret = ((mom_val - portfolio_initial_capital) / abs(portfolio_initial_capital)) * 100 if portfolio_initial_capital != 0 else 0
        ml_costs = s['meta_strategy_ml']['costs']
        print("\n" + "=" * 80)
        print("              🧠 AI META-STRATEGY RESULTS 🧠")
        print("=" * 80)
        print(f"   Meta-Strategy ML:       ${ml_val:>10,.0f}  ({ml_ret:+.1f}%)  costs: ${ml_costs:,.0f}")
        print(f"   Meta-Strategy Momentum: ${mom_val:>10,.0f}  ({mom_ret:+.1f}%)")
        print("=" * 80)

    # --- Print Rebalance Horizon Optimization Results ---
    if rebalance_optimization_results:
        print("\n" + "=" * 80)
        print("              🔄 REBALANCE HORIZON OPTIMIZATION RESULTS 🔄")
        print("=" * 80)
        print(f"   Tested horizons: {REBALANCE_HORIZON_MIN} to {REBALANCE_HORIZON_MAX} days (daily)")
        print("-" * 80)
        for strategy_type in ['1Y', '6M', '3M', '1M']:
            if strategy_type in rebalance_optimization_results:
                r = rebalance_optimization_results[strategy_type]
                print(f"   Static BH {strategy_type}:")
                print(f"      🏆 Best horizon: {r['best_horizon']} days")
                print(f"      📈 Best return: {r['best_return']:+.1f}%")
                print(f"      💰 Transaction costs: ${r['best_txn_cost']:.0f}")
        print("=" * 80)
        print("   💡 Use these optimal horizons in config.py for best performance:")
        for strategy_type in ['1Y', '6M', '3M', '1M']:
            if strategy_type in rebalance_optimization_results:
                r = rebalance_optimization_results[strategy_type]
                config_var = f"STATIC_BH_{strategy_type}_REBALANCE_DAYS"
                print(f"      {config_var} = {r['best_horizon']}")
        print("=" * 80)
        
        # --- Print detailed table with all horizons and returns ---
        print("\n" + "=" * 100)
        print("              📊 DETAILED HORIZON vs RETURN TABLE 📊")
        print("=" * 100)
        
        # Build sorted results for each strategy
        strategy_results = {}
        for strategy_type in ['1Y', '6M', '3M', '1M']:
            if strategy_type in rebalance_optimization_results:
                results = rebalance_optimization_results[strategy_type]['all_results']
                # Sort by horizon
                strategy_results[strategy_type] = sorted(results, key=lambda x: x[0])
        
        # Print header
        header = f"{'Horizon':<10}"
        for st in ['1Y', '6M', '3M', '1M']:
            if st in strategy_results:
                header += f" | {'Static BH ' + st:>15}"
        print(header)
        print("-" * 100)
        
        # Print each horizon row
        horizons = list(range(REBALANCE_HORIZON_MIN, REBALANCE_HORIZON_MAX + 1))
        for horizon in horizons:
            row = f"{horizon:>3} days   "
            for st in ['1Y', '6M', '3M', '1M']:
                if st in strategy_results:
                    # Find return for this horizon
                    ret = None
                    for h, r, _ in strategy_results[st]:
                        if h == horizon:
                            ret = r
                            break
                    if ret is not None:
                        # Mark best with star
                        best_h = rebalance_optimization_results[st]['best_horizon']
                        marker = " 🏆" if horizon == best_h else "   "
                        row += f" | {ret:>+12.1f}%{marker}"
                    else:
                        row += f" | {'N/A':>15}"
            print(row)
        
        print("-" * 100)
        print("   🏆 = Best performing horizon for each strategy")
        print("=" * 100)

    # Send backtesting completion notification
    if backtest_start_time:
        backtest_time_minutes = (time.time() - backtest_start_time) / 60
        
        # Prepare strategy results for email
        strategy_results = {}
        if 'final_buy_hold_value_1y' in locals():
            portfolio_cap = INVESTMENT_PER_STOCK * PORTFOLIO_SIZE
            strategy_results['Buy & Hold'] = {'return': ((final_buy_hold_value_1y - portfolio_cap) / portfolio_cap) * 100 if portfolio_cap != 0 else 0.0}
        
        send_backtesting_notification(
            strategy_results=strategy_results,
            backtest_time_minutes=backtest_time_minutes
        )
    
    # Send final completion notification
    total_time_minutes = (time.time() - script_start_time) / 60
    
    # Use the generic send_completion_notification function
    from notifications import send_completion_notification
    send_completion_notification(
        subject="AI Stock Advisor - Complete",
        message_body=f"""
🎉 AI Stock Advisor execution completed successfully!

📊 Summary:
• Total Runtime: {total_time_minutes:.1f} minutes
• Training: {'Completed' if training_start_time else 'Skipped'}
• Backtesting: {'Completed' if backtest_start_time else 'Skipped'}
• Models Available: {len(models) if 'models' in locals() else 0}

📈 Ready for live trading or analysis!
        """.strip(),
        success=True
    )

    # --- Select and save best performing models for live trading ---
    # Determine which period had the highest portfolio return
    performance_values = {
        actual_period_name: final_strategy_value_1y
    }
    
    best_period_name = max(performance_values, key=performance_values.get)
    
    # Get the models and scalers corresponding to the best period (only 1-Year available)
    best_models_dict = models  # Single model per stock
    best_scalers_dict = scalers

    # Save the best models and scalers for each ticker to the paths used by live_trading.py
    models_dir = Path("logs/models")
    _ensure_dir(models_dir) # Ensure the directory exists

    print(f"\n🏆 Saving best performing models for live trading from {best_period_name} period...")

    for ticker in best_models_dict.keys():
        try:
            model = best_models_dict[ticker]
            scaler = best_scalers_dict[ticker]
            
            # Skip saving - models are already saved by training_phase.py with proper serialization
            # Re-saving here with joblib would corrupt PyTorch models
            print(f"  ✅ Using model for {ticker} from {best_period_name} period (already saved by training phase).")
            
        except Exception as e:
            print(f"  ⚠️ Error processing model for {ticker} from {best_period_name} period: {e}")
    
    # Return all_tickers_data for performance table
    return all_tickers_data

# ============================
# Main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Stock Advisor - Backtesting and Live Trading")
    parser.add_argument("--num-stocks", type=int, default=None,
                       help="Number of stocks to select (overrides PORTFOLIO_SIZE from config)")
    parser.add_argument("--live-trading", action="store_true", 
                       help="Run in live trading mode instead of backtesting")
    parser.add_argument("--strategy", type=str, default="volatility_ensemble",
                       help="Strategy for live trading. Available: volatility_ensemble, enhanced_volatility, correlation_ensemble, momentum_breakout, factor_rotation, pairs_trading, earnings_momentum, insider_trading, options_sentiment, ml_ensemble, risk_adj_mom, dynamic_bh_1y, quality_momentum, momentum_ai_hybrid, elite_hybrid")
    
    args = parser.parse_args()
    
    if args.live_trading:
        # Set strategy in config for live trading
        import src.config as config
        config.LIVE_TRADING_STRATEGY = args.strategy
        
        # Override PORTFOLIO_SIZE if --num-stocks is specified
        if args.num_stocks is not None:
            config.PORTFOLIO_SIZE = args.num_stocks
            config.INVESTMENT_PER_STOCK = config.TOTAL_CAPITAL / args.num_stocks
            print(f"📊 Overriding portfolio size to {args.num_stocks} stocks (${config.INVESTMENT_PER_STOCK:,.0f} per stock)")
        
        print(f"🚀 Starting Live Trading with Strategy: {args.strategy}")
        print(f"📋 Available strategies:")
        print(f"   🏆 volatility_ensemble  - Vol Ens (+106% in backtest)")
        print(f"   🚀 enhanced_volatility  - Enhanced Vol Trader (ATR stops + take profits)")
        print(f"   🤖 ai_volatility_ensemble - AI Vol Ens (NEW - AI-enhanced)")
        print(f"   🏆 correlation_ensemble - Corr Ens (+106% in backtest)")
        print(f"   🆕 momentum_breakout    - 52-week high breakouts")
        print(f"   🆕 factor_rotation      - Value/Growth/Mom/Quality rotation")
        print(f"   🆕 pairs_trading        - Statistical arbitrage")
        print(f"   🆕 earnings_momentum    - Post-earnings drift (PEAD)")
        print(f"   🆕 insider_trading      - Follow insider buying")
        print(f"   🆕 options_sentiment    - Put/call ratio signals")
        print(f"   🆕 ml_ensemble          - Multi-model voting")
        print(f"   risk_adj_mom, dynamic_bh_1y, quality_momentum")
        print(f"💡 Example: python src/main.py --live-trading --strategy volatility_ensemble")
        print("=" * 80)
        
        # Download data using same shared function as backtesting
        print("\n🔍 Loading market data (same as backtesting)...")
        from ticker_selection import get_all_tickers
        from data_utils import load_all_market_data
        
        all_available_tickers = get_all_tickers()
        print(f"   Found {len(all_available_tickers)} tickers in universe")
        
        all_tickers_data = load_all_market_data(all_available_tickers)
        
        if all_tickers_data is None or all_tickers_data.empty:
            print("   ❌ No data available")
            sys.exit(1)
        
        # Get all available tickers from data
        if 'ticker' in all_tickers_data.columns:
            all_available_tickers = sorted(all_tickers_data['ticker'].unique().tolist())
        else:
            all_available_tickers = sorted(all_tickers_data.columns.tolist())
        
        print(f"   ✅ Loaded data for {len(all_available_tickers)} tickers from backtesting")
        
        # Filter to top performers using backtest data
        try:
            from ticker_selection import find_top_performers
            
            # Get latest date from data
            if 'date' in all_tickers_data.columns:
                latest_date = all_tickers_data['date'].max()
                naive_end_date = pd.Timestamp(latest_date).replace(tzinfo=None)
            else:
                naive_end_date = datetime.now().replace(tzinfo=None)
            
            market_selected_performers = find_top_performers(
                all_available_tickers=all_available_tickers,
                all_tickers_data=all_tickers_data,
                return_tickers=True,
                n_top=N_TOP_TICKERS,
                performance_end_date=naive_end_date
            )
            
            if market_selected_performers:
                market_selected_performers = [ticker for ticker, _ in market_selected_performers]
                print(f"   ✅ Filtered to {len(market_selected_performers)} top performers")
            else:
                print(f"   ⚠️ No top performers found, using all tickers")
                market_selected_performers = all_available_tickers[:N_TOP_TICKERS]
                
        except Exception as e:
            print(f"   ❌ Ticker filtering failed: {e}")
            import traceback
            traceback.print_exc()
            market_selected_performers = all_available_tickers[:N_TOP_TICKERS]
        
        # Check if multiple strategies requested (comma-separated)
        strategies = [s.strip() for s in args.strategy.split(',')]
        
        # Use the live trading implementation with filtered tickers
        try:
            from live_trading import run_live_trading_with_filtered_tickers, get_strategy_tickers
            
            if len(strategies) > 1:
                # Multi-strategy mode: just show selected tickers for each
                print("\n" + "=" * 80)
                print("📊 MULTI-STRATEGY TICKER SELECTION")
                print("=" * 80)
                
                # Prepare ticker_data_grouped once (same format as backtesting and single-strategy)
                print(f"\n   📦 Preparing ticker data grouped by ticker...")
                ticker_data_grouped = {}
                for ticker in market_selected_performers:
                    ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                    if not ticker_data.empty:
                        if 'date' in ticker_data.columns:
                            ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                            ticker_data = ticker_data.set_index('date')
                        ticker_data = ticker_data.drop('ticker', axis=1, errors='ignore')
                        ticker_data_grouped[ticker] = ticker_data
                print(f"   ✅ Prepared data for {len(ticker_data_grouped)} tickers")
                
                all_selections = {}
                for strategy in strategies:
                    print(f"\n🎯 Strategy: {strategy}")
                    print("-" * 40)
                    selected = get_strategy_tickers(strategy, market_selected_performers, ticker_data_grouped)
                    all_selections[strategy] = set(selected) if selected else set()
                    if selected:
                        print(f"   Selected {len(selected)} tickers: {selected}")
                    else:
                        print(f"   ⚠️ No tickers selected")
                
                # Show comparison
                print("\n" + "=" * 80)
                print("📊 STRATEGY COMPARISON")
                print("=" * 80)
                
                # Find common tickers (intersection of all)
                if all_selections:
                    common = set.intersection(*all_selections.values()) if all(all_selections.values()) else set()
                    print(f"\n✅ COMMON to all strategies ({len(common)}): {sorted(common) if common else 'None'}")
                    
                    # Show unique tickers for each strategy
                    print(f"\n🔀 UNIQUE to each strategy:")
                    for strategy in strategies:
                        others = set.union(*[s for name, s in all_selections.items() if name != strategy]) if len(strategies) > 1 else set()
                        unique = all_selections[strategy] - others
                        if unique:
                            print(f"   {strategy}: {sorted(unique)}")
                        else:
                            print(f"   {strategy}: None (all shared)")
                
                # === CONSENSUS TABLE ===
                print("\n" + "=" * 80)
                print("📊 CONSENSUS RANKING TABLE")
                print("=" * 80)
                print(f"Stocks ranked by number of strategy votes (out of {len(strategies)} strategies)\n")
                
                # Count votes for each ticker
                from collections import Counter
                ticker_votes = Counter()
                ticker_strategies = {}  # Track which strategies voted for each ticker
                
                for strategy, tickers in all_selections.items():
                    for ticker in tickers:
                        ticker_votes[ticker] += 1
                        if ticker not in ticker_strategies:
                            ticker_strategies[ticker] = []
                        ticker_strategies[ticker].append(strategy)
                
                # Sort by vote count (descending), then alphabetically
                sorted_tickers = sorted(ticker_votes.items(), key=lambda x: (-x[1], x[0]))
                
                # Print header
                print(f"{'Rank':<6} {'Ticker':<12} {'Votes':<8} {'Strategies':<40}")
                print("-" * 80)
                
                # Print each ticker with vote info - show ALL with 2+ votes
                from config import PORTFOLIO_SIZE
                min_votes = 2  # Show only stocks with 2 or more votes
                filtered_tickers = [(t, v) for t, v in sorted_tickers if v >= min_votes]
                
                for rank, (ticker, votes) in enumerate(filtered_tickers, 1):
                    vote_bar = "█" * votes + "░" * (len(strategies) - votes)
                    strategies_short = ", ".join([s[:8] for s in ticker_strategies[ticker]])
                    print(f"{rank:<6} {ticker:<12} {votes}/{len(strategies)} {vote_bar:<8}  {strategies_short}")
                
                if not filtered_tickers:
                    print(f"   ⚠️ No stocks with {min_votes}+ strategy votes found")
                
                # Summary statistics
                print("\n" + "-" * 80)
                print("📈 CONSENSUS SUMMARY:")
                for vote_count in range(len(strategies), 0, -1):
                    count = sum(1 for v in ticker_votes.values() if v == vote_count)
                    tickers_with_votes = [t for t, v in ticker_votes.items() if v == vote_count]
                    if count > 0:
                        emoji = "🟢" if vote_count == len(strategies) else "🔵" if vote_count >= len(strategies)//2 + 1 else "⚪"
                        print(f"   {emoji} {vote_count} vote{'s' if vote_count > 1 else ''}: {count} stocks {tickers_with_votes if count <= 5 else ''}")
                
                # Recommended portfolio (top 10 by consensus)
                print("\n" + "=" * 80)
                print("🎯 RECOMMENDED CONSENSUS PORTFOLIO (Top 10 by votes)")
                print("=" * 80)
                top_consensus = [t for t, v in sorted_tickers[:10]]
                if top_consensus:
                    print(f"   {top_consensus}")
                    print(f"\n   💡 These stocks have the highest agreement across strategies")
                    print(f"   💡 Consider running: python src/main.py --live-trading --strategy consensus")
                else:
                    print("   ⚠️ No consensus stocks found")
                
                print("\n" + "=" * 80)
                print("💡 To execute trades with a single strategy, run:")
                print(f"   python src/main.py --live-trading --strategy <strategy_name>")
                print("=" * 80)
            else:
                # Single strategy: run full live trading
                # Prepare ticker_data_grouped once (same format as backtesting)
                print(f"\n   📦 Preparing ticker data grouped by ticker...")
                ticker_data_grouped = {}
                for ticker in market_selected_performers:
                    ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                    if not ticker_data.empty:
                        if 'date' in ticker_data.columns:
                            ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                            ticker_data = ticker_data.set_index('date')
                        ticker_data = ticker_data.drop('ticker', axis=1, errors='ignore')
                        ticker_data_grouped[ticker] = ticker_data
                print(f"   ✅ Prepared data for {len(ticker_data_grouped)} tickers")
                
                run_live_trading_with_filtered_tickers(market_selected_performers, ticker_data_grouped)
        except Exception as e:
            print(f"❌ Live trading failed: {e}")
    else:
        # Run normal backtesting and get the data
        all_tickers_data = main()
    
    # Add 3M performance table at the end (runs in both live trading and backtesting modes)
    try:
        print("\n" + "=" * 100)
        print("📊 UNIFIED TICKER PERFORMANCE TABLE (Sorted by 3M Performance)")
        print("=" * 100)
        
        from ticker_selection import _calculate_1y_return_from_dataframe
        
        today = datetime.now(timezone.utc)
        
        # Use the already loaded all_tickers_data from backtest instead of fetching fresh data
        if all_tickers_data is not None and not all_tickers_data.empty:
            print(f"🔍 Using backtest data for performance calculation...")
            print(f"   Data shape: {all_tickers_data.shape}")
            print(f"   Columns: {list(all_tickers_data.columns)[:5]}...")  # Show first 5 columns
            
            # Data is in long format
            print("   Data is in long format")
            
            # Check if date is in index or column
            if 'date' not in all_tickers_data.columns:
                if hasattr(all_tickers_data.index, 'names') and 'date' in all_tickers_data.index.names:
                    # Date is in index, reset it to make it a column
                    all_tickers_data = all_tickers_data.reset_index()
                    print("   Reset index to make 'date' a column")
                else:
                    print("   ⚠️ No 'date' column found in data")
                    raise ValueError("Date column not found in data")
            
            # Ensure we have a 'ticker' column in long format
            if 'ticker' not in all_tickers_data.columns:
                print("   ⚠️ No 'ticker' column found in long format data")
                # Try to infer ticker from other columns or raise a clear error
                possible_cols = [col for col in all_tickers_data.columns if 'ticker' in col.lower() or 'symbol' in col.lower()]
                if possible_cols:
                    ticker_col = possible_cols[0]
                    print(f"   Using column '{ticker_col}' as ticker")
                    all_tickers_data = all_tickers_data.rename(columns={ticker_col: 'ticker'})
                else:
                    raise ValueError("Data is in long format but missing 'ticker' column and cannot infer it")
            
            # Get unique tickers from the backtest data
            if 'ticker' in all_tickers_data.columns:
                all_available_tickers = all_tickers_data['ticker'].unique().tolist()
            else:
                print("   ⚠️ No 'ticker' column found in data")
                # Fallback to fetching fresh data
                raise ValueError("Ticker column not found")
            print(f"   Found {len(all_available_tickers)} tickers in backtest data")
            
            # Group data by ticker for easier processing
            ticker_data_dict = {}
            for ticker in all_available_tickers:
                ticker_df = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                if not ticker_df.empty:
                    if 'date' in ticker_df.columns:
                        ticker_df = ticker_df.set_index('date')
                    ticker_data_dict[ticker] = ticker_df
            
            # Calculate 3M and 1Y performance for all tickers in backtest data
            performance_data = []
            
            print(f"🔍 Calculating performance metrics for {len(all_available_tickers)} tickers...")
            
            for ticker in all_available_tickers:
                try:
                    ticker_df = ticker_data_dict[ticker]
                    
                    # Calculate 1Y performance
                    perf_1y = _calculate_1y_return_from_dataframe(ticker_df, today, 365)
                    
                    # Calculate 3M performance
                    perf_3m = _calculate_1y_return_from_dataframe(ticker_df, today, 90)
                    
                    if perf_1y is not None and perf_3m is not None:
                        performance_data.append({
                            'Ticker': ticker,
                            '3M_Perf': perf_3m,
                            '1Y_Perf': perf_1y
                        })
                        
                except Exception as e:
                    # Skip tickers with errors
                    continue
        else:
            print("⚠️ No backtest data available, fetching fresh data...")
            # Fallback to original behavior if no backtest data
            from ticker_selection import get_all_tickers, _get_current_1y_return_from_cache
            
            try:
                all_available_tickers = get_all_tickers()
            except Exception as e:
                print(f"⚠️ Could not fetch tickers: {e}")
                all_available_tickers = []
            
            # Calculate 3M and 1Y performance for all available tickers
            performance_data = []
            
            print(f"🔍 Calculating performance metrics for {len(all_available_tickers)} tickers...")
            
            for ticker in all_available_tickers:
                try:
                    # Get 1Y performance
                    perf_1y = _get_current_1y_return_from_cache(ticker, {}, today, 365)
                    
                    # Get 3M performance (reuse the same function with 90 days)
                    perf_3m = _get_current_1y_return_from_cache(ticker, {}, today, 90)
                    
                    if perf_1y is not None and perf_3m is not None:
                        performance_data.append({
                            'Ticker': ticker,
                            '3M_Perf': perf_3m,
                            '1Y_Perf': perf_1y
                        })
                        
                except Exception as e:
                    # Skip tickers with errors
                    continue
        
        # Sort by 3M performance (descending)
        performance_data.sort(key=lambda x: x['3M_Perf'], reverse=True)
        
        # Print table
        print(f"{'Ticker':<10} {'3M_Perf':>12} {'1Y_Perf':>12}")
        print("-" * 38)
        
        for data in performance_data:  # Show ALL tickers
            ticker = data['Ticker']
            perf_3m = data['3M_Perf']
            perf_1y = data['1Y_Perf']
            
            # Color coding for performance
            if perf_3m >= 20:
                color_3m = "🟢"  # Strong positive
            elif perf_3m >= 10:
                color_3m = "🔵"  # Moderate positive
            elif perf_3m >= 0:
                color_3m = "⚪"  # Neutral
            else:
                color_3m = "🔴"  # Negative
            
            print(f"{ticker:<10} {color_3m} {perf_3m:+10.2f}% {perf_1y:+10.2f}%")
        
        print(f"\n📈 Summary:")
        print(f"   Total tickers analyzed: {len(performance_data)}")
        if performance_data:
            avg_3m = sum(d['3M_Perf'] for d in performance_data) / len(performance_data)
            avg_1y = sum(d['1Y_Perf'] for d in performance_data) / len(performance_data)
            print(f"   Average 3M performance: {avg_3m:+.2f}%")
            print(f"   Average 1Y performance: {avg_1y:+.2f}%")
            print(f"   Best 3M performer: {performance_data[0]['Ticker']} ({performance_data[0]['3M_Perf']:+.2f}%)")
            print(f"   Worst 3M performer: {performance_data[-1]['Ticker']} ({performance_data[-1]['3M_Perf']:+.2f}%)")
        
        print("=" * 100)
        
    except Exception as e:
        print(f"⚠️ Error generating performance table: {e}")
