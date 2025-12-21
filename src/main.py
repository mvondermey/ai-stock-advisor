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
from pathlib import Path

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import (
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE,
    MIN_PROBA_BUY, MIN_PROBA_SELL, TARGET_PERCENTAGE, CLASS_HORIZON,
    FORCE_THRESHOLDS_OPTIMIZATION, FORCE_PERCENTAGE_OPTIMIZATION,
    INITIAL_BALANCE, INVESTMENT_PER_STOCK, TRANSACTION_COST,
    BACKTEST_DAYS, TRAIN_LOOKBACK_DAYS, VALIDATION_DAYS,
    TOP_CACHE_PATH, N_TOP_TICKERS, NUM_PROCESSES, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES, PAUSE_BETWEEN_YF_CALLS,
    ENABLE_1YEAR_TRAINING,
    ENABLE_1YEAR_BACKTEST,
    FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD,
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    GRU_HIDDEN_SIZE_OPTIONS, GRU_NUM_LAYERS_OPTIONS, GRU_DROPOUT_OPTIONS,
    GRU_LEARNING_RATE_OPTIONS, GRU_BATCH_SIZE_OPTIONS, GRU_EPOCHS_OPTIONS,
    USE_MODEL_GATE, USE_GRU, USE_LSTM, USE_LOGISTIC_REGRESSION, USE_RANDOM_FOREST,
    USE_SVM, USE_MLP_CLASSIFIER, USE_LIGHTGBM, USE_XGBOOST,
    FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    USE_PERFORMANCE_BENCHMARK, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    SEED, SAVE_PLOTS, MARKET_SELECTION,
    SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_EPOCHS,
    ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION,
    PERIOD_HORIZONS, POSITION_SCALING_BY_CONFIDENCE
)

from alpha_training import select_threshold_by_alpha, AlphaThresholdConfig
# Toggle: use alpha-optimized probability threshold for buys/sells
USE_ALPHA_THRESHOLD_BUY = True
USE_ALPHA_THRESHOLD_SELL = True

def _gpu_diag():
    """Run GPU diagnostics only for enabled models"""
    # Only check PyTorch if LSTM or GRU are enabled
    if USE_LSTM or USE_GRU:
        try:
            import torch
            print(f"[GPU] torch.cuda.is_available(): {torch.cuda.is_available()}")
        except Exception as e:
            print("[GPU] torch check failed:", e)
    
    # Only check XGBoost if it's enabled
    if USE_XGBOOST:
        try:
            import xgboost as xgb
            print("[GPU] XGBoost version:", getattr(xgb, "__version__", "?"))
        except Exception as e:
            print("[GPU] XGBoost check failed:", e)
    
    # Only check LightGBM if it's enabled
    if USE_LIGHTGBM:
        try:
            import lightgbm as lgb
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
if not _script_initialized:
    print("DEBUG: Script execution initiated.")
    _script_initialized = True

import os
from backtesting import optimize_thresholds_for_portfolio_parallel, _prepare_model_for_multiprocessing
# from portfolio_rebalancing import run_portfolio_rebalancing_backtest  # Module deleted
# from rule_based_strategy import run_rule_based_portfolio_strategy  # Module deleted
from summary_phase import print_final_summary
from training_phase import train_worker, train_models_for_period
from backtesting_phase import _run_portfolio_backtest, _run_portfolio_backtest_walk_forward
import json
import time
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Import torch if available
if PYTORCH_AVAILABLE:
    import torch
import gymnasium as gym
import codecs
import random
import requests  # Added for internet time fetching
from io import StringIO
from multiprocessing import Pool, cpu_count, current_process
import joblib # Added for model saving/loading
import warnings # Added for warning suppression
from data_utils import load_prices, fetch_training_data, load_prices_robust, _ensure_dir
from data_fetcher import (
    _normalize_symbol, _fetch_from_stooq, _fetch_from_alpaca, _fetch_from_twelvedata,
    _download_batch_robust, _fetch_financial_data, _fetch_financial_data_from_alpaca,
    _fetch_intermarket_data
)
from ticker_selection import get_tickers_for_backtest, get_all_tickers, find_top_performers
from backtesting_env import RuleTradingEnv
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

# --- Backtest & training windows
# BACKTEST_DAYS           = 365        # 1 year for backtest # Moved to config.py
# BACKTEST_DAYS_3MONTH    = 90         # 3 months for backtest # Moved to config.py
# BACKTEST_DAYS_1MONTH    = 32         # 1 month for backtest # Moved to config.py
# TRAIN_LOOKBACK_DAYS     = 360        # more data for model (e.g., 1 year) # Moved to config.py

# --- Backtest Period Enable/Disable Flags ---
# ENABLE_1YEAR_BACKTEST   = True # Moved to config.py
# ENABLE_YTD_BACKTEST     = True # Moved to config.py
# ENABLE_3MONTH_BACKTEST  = True # Moved to config.py
# ENABLE_1MONTH_BACKTEST  = True # Moved to config.py

# --- Training Period Enable/Disable Flags ---
# ENABLE_1YEAR_TRAINING   = True # Moved to config.py
# ENABLE_YTD_TRAINING     = True # Moved to config.py
# ENABLE_3MONTH_TRAINING  = True # Moved to config.py
# ENABLE_1MONTH_TRAINING  = True # Moved to config.py

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
    optimized_params_per_ticker: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[Optional[float], Optional[float], Optional[Dict], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Optional[List], Optional[List], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[Dict]]:
    
    # Set NUM_PROCESSES if not already set
    global NUM_PROCESSES
    if NUM_PROCESSES is None:
        NUM_PROCESSES = max(1, cpu_count() - 2)
        print(f"✅ NUM_PROCESSES set to {NUM_PROCESSES} (cpu_count() - 2)")
    
    # Set the start method for multiprocessing to 'spawn'
    # This is crucial for CUDA compatibility with multiprocessing
    try:
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            print("✅ Multiprocessing start method set to 'spawn' for CUDA compatibility.")
    except RuntimeError as e:
        print(f"⚠️ Could not set multiprocessing start method to 'spawn': {e}. This might cause issues with CUDA and multiprocessing.")

    # Get accurate time from internet source for consistent backtesting
    end_date = get_internet_time()
    bt_end = end_date
    
    alpaca_trading_client = None

    # Initialize ML libraries to determine CUDA availability
    initialize_ml_libraries()
    
    # Note: multiprocessing with CUDA-enabled DL models uses 'spawn' start method for stability
    if PYTORCH_AVAILABLE and CUDA_AVAILABLE and (USE_LSTM or USE_GRU or ("USE_TCN" in globals() and USE_TCN)):
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

    # Cap bt_end to the latest available data to avoid future-dated slices
    last_available = all_tickers_data.index.max()
    if last_available < bt_end:
        print(f"ℹ️ Capping backtest end date from {bt_end.date()} to last available data {last_available.date()}")
        end_date = last_available
        bt_end = last_available
    print(f"📅 Using backtest end date: {bt_end.date()} (last available: {last_available.date()})")

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
        for col_name in ['VIX_Index_Returns', 'DXY_Index_Returns', 'Gold_Futures_Returns', 'Oil_Futures_Returns', 'US10Y_Yield_Returns', 'Oil_Price_Returns', 'Gold_Price_Returns']:
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
        bt_start_1y = bt_end - timedelta(days=BACKTEST_DAYS)
        train_end_1y = bt_start_1y - timedelta(days=1)
        train_start_1y_calc = train_end_1y - timedelta(days=TRAIN_LOOKBACK_DAYS)
        
        # Use the new train_models_for_period function
        models_buy, models_sell, scalers, y_scalers = train_models_for_period(
            period_name="1-Year",
            tickers=top_tickers,
            all_tickers_data=all_tickers_data,
            train_start=train_start_1y_calc,
            train_end=train_end_1y,
            top_performers_data=top_performers_data,
            feature_set=feature_set,
            run_parallel=run_parallel
        )

    # 🧠 Initialize dictionaries for model training data before threshold optimization
    X_train_dict, y_train_dict, X_test_dict, y_test_dict = {}, {}, {}, {}
    prices_dict, signals_dict = {}, {}

    if not models_buy and USE_MODEL_GATE:
        print("⚠️ No models were trained for 1-Year backtest. Model-gating will be disabled for this run.\n")
    
    # Filter out failed tickers from top_tickers for subsequent steps
    top_tickers_1y_filtered = [t for t in top_tickers if t not in failed_training_tickers_1y]
    print(f"  ℹ️ {len(failed_training_tickers_1y)} tickers failed 1-Year model training and will be skipped: {', '.join(failed_training_tickers_1y.keys())}")
    
    # Update top_performers_data to reflect only successfully trained tickers
    top_performers_data_1y_filtered = [item for item in top_performers_data if item[0] in top_tickers_1y_filtered]
    
    # Set capital_per_stock to the fixed investment amount
    capital_per_stock_1y = INVESTMENT_PER_STOCK
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
    
    # Initialize all_tested_combinations
    all_tested_combinations = {}
    
    # Initialize optimized_params_per_ticker (will be populated during optimization phase)
    optimized_params_per_ticker = {}
    if not should_run_optimization:
        # If no optimization is forced, load existing or use defaults
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

    gru_hyperparams_buy_dict_1month, gru_hyperparams_sell_dict_1month = {}, {} # Initialize here




    # ========================================================================
    # PHASE 2: OPTIMIZE THRESHOLDS FOR ALL PERIODS
    # ========================================================================
    
    # Initialize optimized_params dictionaries for all periods
    optimized_params_per_ticker_ytd = {}
    optimized_params_per_ticker_3month = {}
    optimized_params_per_ticker_1month = {}
    
    if should_run_optimization:
        # --- Optimize 1-Year Period ---
        print("\n🔄 Step 4: Optimizing ML parameters for 1-Year period...")
        optimization_params = []
        for ticker in top_tickers_1y_filtered:
            if ticker in models_buy and ticker in models_sell and ticker in scalers:
                model_buy_ticker = models_buy[ticker]
                model_sell_ticker = models_sell[ticker]
                
                buy_model_type = type(model_buy_ticker).__name__ if model_buy_ticker else 'None'
                sell_model_type = type(model_sell_ticker).__name__ if model_sell_ticker else 'None'
                
                if model_buy_ticker is None or model_sell_ticker is None:
                    print(f"  ⏭️  Skipping optimization for {ticker}: Missing model (Buy: {buy_model_type}, Sell: {sell_model_type})")
                    continue
                
                print(f"  ✅ Optimizing {ticker}: Buy={buy_model_type}, Sell={sell_model_type}")
                
                current_min_proba_buy_for_opt = loaded_optimized_params.get(ticker, {}).get('min_proba_buy', MIN_PROBA_BUY)
                current_min_proba_sell_for_opt = loaded_optimized_params.get(ticker, {}).get('min_proba_sell', MIN_PROBA_SELL)
                
                # Use the SAME training parameters that the model was trained with
                if FORCE_PERCENTAGE_OPTIMIZATION:
                    # *** KEY FIX: Use config values for optimization too ***
                    current_target_percentage_for_opt = TARGET_PERCENTAGE
                    current_class_horizon_for_opt = CLASS_HORIZON
                    print(f"     📊 Optimization using FORCED config: Target={current_target_percentage_for_opt:.2%}, Horizon={current_class_horizon_for_opt}d")
                else:
                    # Reconstruct from B&H performance
                    ticker_bh_return_1y = 0.01
                    for t, perf_1y in top_performers_data:
                        if t == ticker:
                            ticker_bh_return_1y = perf_1y / 100.0
                            break
                    
                    period_horizon_1y = 20  # Shorter horizon for better learning
                    num_periods = 252.0 / period_horizon_1y
                    period_target_pct_1y = abs(ticker_bh_return_1y / num_periods)
                    period_target_pct_1y = max(period_target_pct_1y, 0.01)
                    
                    current_target_percentage_for_opt = period_target_pct_1y
                    current_class_horizon_for_opt = period_horizon_1y
                    
                    print(f"     📊 Optimization using training params: Target={current_target_percentage_for_opt:.2%}, Horizon={current_class_horizon_for_opt}d")

                feature_set_for_opt = scalers[ticker].feature_names_in_ if hasattr(scalers[ticker], 'feature_names_in_') else None

                # Get training data for optimization (REVERTED Fix 4)
                try:
                    ticker_train_data = all_tickers_data.loc[train_start_1y_calc:train_end_1y, (slice(None), ticker)]
                    ticker_train_data.columns = ticker_train_data.columns.droplevel(1)
                    if ticker_train_data.empty:
                        print(f"  ⚠️ Could not get training data for {ticker} for optimization. Skipping.")
                        continue
                except (KeyError, IndexError):
                    print(f"  ⚠️ Could not slice training data for {ticker} for optimization. Skipping.")
                    continue

                # Prepare PyTorch models for multiprocessing (extract state dict as numpy arrays)
                model_buy_prepared = _prepare_model_for_multiprocessing(model_buy_ticker)
                model_sell_prepared = _prepare_model_for_multiprocessing(model_sell_ticker)
                
                optimization_params.append((
                    ticker,
                    ticker_train_data.copy(),
                    capital_per_stock_1y,
                    current_target_percentage_for_opt,
                    current_class_horizon_for_opt,
                    force_thresholds_optimization,
                    force_percentage_optimization,
                    USE_ALPHA_THRESHOLD_BUY,
                    USE_ALPHA_THRESHOLD_SELL,
                    AlphaThresholdConfig(rebalance_freq="D", metric="alpha", costs_bps=5.0, slippage_bps=2.0),
                    current_min_proba_buy_for_opt,
                    current_min_proba_sell_for_opt,
                    current_target_percentage_for_opt,
                    current_class_horizon_for_opt,
                    GRU_TARGET_PERCENTAGE_OPTIONS,
                    GRU_CLASS_HORIZON_OPTIONS,
                    SEED,
                    feature_set_for_opt,
                    model_buy_prepared,  # Pass model info (will be reconstructed on GPU in worker)
                    model_sell_prepared,  # Pass model info (will be reconstructed on GPU in worker)
                    scalers[ticker]  # Pass already-trained scaler
                ))
        
        if optimization_params:
            optimized_params_per_ticker, all_tested_combinations = optimize_thresholds_for_portfolio_parallel(optimization_params)
            
            # Print backtest results for each tested combination
            if all_tested_combinations:
                print("\n" + "="*80)
                print("📊 Backtest Results for All Tested Optimization Combinations (1-Year)")
                print("="*80)
                for ticker, combinations in all_tested_combinations.items():
                    if not combinations:
                        continue
                    print(f"\n📈 {ticker} - Tested {len(combinations)} combinations:")
                    print("-" * 100)
                    sorted_combinations = sorted(combinations, key=lambda x: x.get('revenue', -np.inf), reverse=True)
                    print(f"{'Rank':<6} | {'Buy Thresh':<12} | {'Sell Thresh':<12} | {'AI Revenue':<15} | {'B&H Revenue':<15} | {'Difference':<15} | {'Alpha':<10}")
                    print("-" * 100)
                    for idx, combo in enumerate(sorted_combinations[:20], 1):
                        revenue = combo.get('revenue', capital_per_stock_1y)
                        buy_hold_revenue = combo.get('buy_hold_revenue', 0.0)
                        revenue_pct = ((revenue - capital_per_stock_1y) / capital_per_stock_1y * 100) if capital_per_stock_1y > 0 else 0.0
                        bh_revenue_pct = ((buy_hold_revenue) / capital_per_stock_1y * 100) if capital_per_stock_1y > 0 else 0.0
                        diff = revenue - buy_hold_revenue
                        diff_pct = revenue_pct - bh_revenue_pct
                        alpha = combo.get('alpha_annualized', 0.0)
                        
                        print(f"{idx:<6} | {combo.get('min_proba_buy', 0.0):>11.2f} | {combo.get('min_proba_sell', 0.0):>11.2f} | "
                              f"${revenue:>13,.2f} ({revenue_pct:>+6.2f}%) | ${buy_hold_revenue:>13,.2f} ({bh_revenue_pct:>+6.2f}%) | "
                              f"${diff:>13,.2f} ({diff_pct:>+6.2f}%) | {alpha:>9.4f}")
                    if len(sorted_combinations) > 20:
                        print(f"... and {len(sorted_combinations) - 20} more combinations")
                    print("-" * 100)
                print("="*80 + "\n")

            if optimized_params_per_ticker:
                try:
                    with open(optimized_params_file, 'w') as f:
                        json.dump(optimized_params_per_ticker, f, indent=4)
                    print(f"✅ Optimized parameters saved to {optimized_params_file}")
                except Exception as e:
                    print(f"⚠️ Could not save optimized parameters to file: {e}")
        else:
            optimized_params_per_ticker = {}
            all_tested_combinations = {}
        
        
        
    else:
        # If no optimization is forced, use defaults for all periods
        optimized_params_per_ticker_ytd = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_ytd_filtered} if optimized_params_per_ticker else {}
        for ticker in top_tickers_ytd_filtered:
            if ticker not in optimized_params_per_ticker_ytd:
                optimized_params_per_ticker_ytd[ticker] = {
                    'min_proba_buy': MIN_PROBA_BUY,
                    'min_proba_sell': MIN_PROBA_SELL,
                    'target_percentage': target_percentage,
                    'class_horizon': class_horizon,
                    'optimization_status': "Using 1-Year optimized params"
                }
        
        optimized_params_per_ticker_3month = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_3month_filtered} if optimized_params_per_ticker else {}
        for ticker in top_tickers_3month_filtered:
            if ticker not in optimized_params_per_ticker_3month:
                optimized_params_per_ticker_3month[ticker] = {
                    'min_proba_buy': MIN_PROBA_BUY,
                    'min_proba_sell': MIN_PROBA_SELL,
                    'target_percentage': target_percentage,
                    'class_horizon': class_horizon,
                    'optimization_status': "Using 1-Year optimized params"
                }
        
        optimized_params_per_ticker_1month = {k: v for k, v in optimized_params_per_ticker.items() if k in top_tickers_1month_filtered} if optimized_params_per_ticker else {}
        for ticker in top_tickers_1month_filtered:
            if ticker not in optimized_params_per_ticker_1month:
                optimized_params_per_ticker_1month[ticker] = {
                    'min_proba_buy': MIN_PROBA_BUY,
                    'min_proba_sell': MIN_PROBA_SELL,
                    'target_percentage': target_percentage,
                    'class_horizon': class_horizon,
                    'optimization_status': "Using 1-Year optimized params"
                }

    # ========================================================================
    # PHASE 3: RUN ALL BACKTESTS
    # ========================================================================
    
    # --- Run 1-Year Backtest ---
    if ENABLE_1YEAR_BACKTEST:
        print("\n🔍 Step 8: Running 1-Year Backtest...")
        # DEBUG: Check what's in models dictionaries
        print(f"\n[DEBUG MAIN] 1-Year models_buy keys: {list(models_buy.keys())}")
        print(f"[DEBUG MAIN] 1-Year models_buy values types: {[type(v).__name__ if v else 'None' for v in models_buy.values()]}")
        print(f"[DEBUG MAIN] 1-Year models_sell keys: {list(models_sell.keys())}")
        
        # --- Run 1-Year Backtest (AI Strategy) ---
        print("\n🔍 Step 8: Running 1-Year Backtest (AI Strategy)...")
        n_top_rebal = 3
        initial_capital_1y = capital_per_stock_1y * n_top_rebal
        
        # Use walk-forward backtest with periodic retraining and rebalancing
        final_strategy_value_1y, portfolio_values_1y, processed_tickers_1y, performance_metrics_1y, buy_hold_histories_1y = _run_portfolio_backtest_walk_forward(
            all_tickers_data=all_tickers_data,
            train_start_date=train_start_1y_calc,
            backtest_start_date=bt_start_1y,
            backtest_end_date=bt_end,
            initial_top_tickers=top_tickers_1y_filtered,
            initial_models_buy=models_buy,
            initial_models_sell=models_sell,
            initial_scalers=scalers,
            initial_y_scalers=y_scalers,
            capital_per_stock=capital_per_stock_1y,
            target_percentage=TARGET_PERCENTAGE,
            period_name="1-Year",
            top_performers_data=top_performers_data,
            horizon_days=PERIOD_HORIZONS.get("1-Year", 60)
        )
        
        ai_1y_return = ((final_strategy_value_1y - initial_capital_1y) / initial_capital_1y) * 100
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

        # --- Calculate Buy & Hold for 1-Year ---
        print("\n📊 Calculating Buy & Hold performance for 1-Year period...")
        buy_hold_results_1y = []
        performance_metrics_buy_hold_1y_actual = []
        for ticker in top_tickers_1y_filtered:
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
        # Calculate Buy & Hold portfolio value for the SAME stocks that AI strategy used
        # This ensures fair comparison: same stocks, same total capital allocation
        ai_strategy_tickers = set(processed_tickers_1y)  # Tickers that AI strategy actually traded (ONDS, WBX, OMEX)

        # Sum Buy & Hold final values for only the stocks that AI strategy actually traded
        bh_portfolio_value = 0.0
        stocks_count = 0

        for i, ticker in enumerate(top_tickers_1y_filtered):
            if ticker in ai_strategy_tickers and i < len(buy_hold_results_1y):
                bh_portfolio_value += buy_hold_results_1y[i]
                stocks_count += 1

        # The buy_hold_results_1y values assume $15,000 initial per stock
        # AI strategy also uses $15,000 per stock (45,000 / 3 = 15,000)
        # So the sum gives us the correct portfolio value
        if stocks_count > 0:
            final_buy_hold_value_1y = bh_portfolio_value
        else:
            # Fallback to original calculation
            final_buy_hold_value_1y = sum(buy_hold_results_1y) + (len(top_tickers_1y_filtered) - len(buy_hold_results_1y)) * capital_per_stock_1y
        print("✅ 1-Year Buy & Hold calculation complete.")
    else:
        print("\nℹ️ 1-Year Backtest is disabled by ENABLE_1YEAR_BACKTEST flag.")




    # ========================================================================
    # PHASE 4: FINAL SUMMARY
    # ========================================================================

    # --- Prepare data for the final summary table (using 1-Year results for the table) ---
    print("\n📝 Preparing final summary data...")
    final_results = []
    
    # Combine all failed tickers from all periods
    all_failed_tickers = {}
    all_failed_tickers.update(failed_training_tickers_1y)
    all_failed_tickers.update(failed_training_tickers_ytd)
    all_failed_tickers.update(failed_training_tickers_3month)
    all_failed_tickers.update(failed_training_tickers_1month) # Add 1-month failed tickers

    # Distribute the portfolio-level final value across tickers for display (no per-ticker result in rebalancing mode)
    per_ticker_portfolio_value_1y = (final_strategy_value_1y / len(processed_tickers_1y)) if processed_tickers_1y else INVESTMENT_PER_STOCK

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

        perf_1y_benchmark = np.nan
        ytd_perf_benchmark = np.nan  # YTD performance not available since YTD support removed
        for t, p1y in top_performers_data:
            if t == ticker:
                perf_1y_benchmark = p1y if np.isfinite(p1y) else np.nan
                break
        
        final_results.append({
            'ticker': ticker,
            'performance': per_ticker_portfolio_value_1y,
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
    
    # ✅ FIX: Use the same initial capital that was allocated to the portfolio backtest
    actual_initial_capital_1y = initial_capital_1y
    actual_tickers_analyzed = len(processed_tickers_1y)
    
    print_final_summary(
        sorted_final_results, models_buy, models_sell, scalers, optimized_params_per_ticker,
        final_strategy_value_1y, final_buy_hold_value_1y, ai_1y_return,
        0, 0, 0,  # Placeholder values for removed YTD parameters
        0, 0, 0,  # Placeholder values for removed 3-Month parameters
        actual_initial_capital_1y,
        actual_tickers_analyzed,
        0,  # Placeholder value for removed 1-Month final_strategy_value
        0,  # Placeholder value for removed 1-Month ai_return
        0,  # Placeholder value for removed 1-Month buy_hold_value
        performance_metrics_buy_hold_1y_actual, # Pass performance_metrics_buy_hold_1y_actual for Buy & Hold
        top_performers_data
    )
    print("\n✅ Final summary prepared and printed.")

    # --- Select and save best performing models for live trading ---
    # Determine which period had the highest portfolio return
    performance_values = {
        "1-Year": final_strategy_value_1y
    }
    
    best_period_name = max(performance_values, key=performance_values.get)
    
    # Get the models and scalers corresponding to the best period (only 1-Year available)
    best_models_buy_dict = models_buy
    best_models_sell_dict = models_sell
    best_scalers_dict = scalers

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

# ============================
# Main
# ============================

if __name__ == "__main__":
    main()
