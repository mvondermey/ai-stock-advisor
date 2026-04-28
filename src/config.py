from operator import truediv
from typing import Dict

import os

from pathlib import Path

from pickle import FALSE

from unittest.runner import TextTestRunner



# ============================

# Configuration / Hyperparams

# ============================



SEED                    = 42



# --- Provider & caching

DATA_PROVIDER           = 'yahoo' # 'stooq', 'yahoo', 'alpaca', or 'twelvedata'

USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Twelvedata thin

DATA_INTERVAL           = '1h'       # '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'

DATA_CACHE_DIR          = Path("data_cache")

# Master switch to enable/disable data downloads
ENABLE_DATA_DOWNLOAD    = True      # Set to False to use only cached data (no new downloads)



# --- JSON output control ---

ENABLE_JSON_OUTPUT       = True       # Control whether to write strategy_selections.json and live_trading_selections.json



def get_data_lookback_days():

    """Get the number of days to look back for data downloads based on DATA_INTERVAL."""

    MAX_LOOKBACK_DAYS = 730  # 2 years of historical data

    if DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:

        return 728  # Leave extra headroom for Yahoo's brittle intraday boundary checks

    else:

        return MAX_LOOKBACK_DAYS  # Removed buffer for weekends/holidays

TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")

VALID_TICKERS_CACHE_PATH = Path("logs/valid_tickers.json")

CACHE_DAYS              = 7



# --- Hybrid Data Configuration ---

AGGREGATE_TO_DAILY      = True       # Aggregate hourly data to create perfect daily data

USE_INTRADAY_FEATURES   = True       # Enable intraday features for AI models

USE_DAILY_FEATURES       = True       # Keep existing daily features

FEATURE_COMBINATION      = 'merged'    # 'merged' or 'separate'



# --- Cache Configuration ---

# Master switch: enable price caching to reduce downloads (uses `data_cache/` CSVs)

ENABLE_PRICE_CACHE       = True



# Alpaca API credentials

ALPACA_API_KEY          = "PK3FDQLRMEVFOAOU7VHD3Z6THE"

ALPACA_SECRET_KEY       = "8By7ituNTmspLsWc191hQfviD3xaNNdd2opB8tJAfmK6"



# TwelveData API credentials

TWELVEDATA_API_KEY      = "aed912386d7c47939ebc28a86a96a021"

TWELVEDATA_MAX_WORKERS  = 5  # Max parallel API requests (free tier: 800 credits/day)



# Alpha Vantage API credentials (FREE tier - 500 calls/day, 25 calls/minute)

ALPHA_VANTAGE_API_KEY   = "6WMZFFL86AE6QK2R"  # Get free key from https://www.alphavantage.co/support/#api-key

ALPHA_VANTAGE_MAX_CALLS_PER_MINUTE = 25  # Free tier limit



# --- Push Notifications (ntfy.sh) ---

# Setup: Install ntfy app on phone, subscribe to your topic

NTFY_ENABLED = True  # Enable push notifications

NTFY_TOPIC = "ai-stock-mvondermey"  # Your unique topic name (change this!)

NTFY_SERVER = "https://ntfy.sh"  # Default ntfy server



# --- ML Library Availability Flags (initialized to False, updated in main.py) ---

ALPACA_AVAILABLE = False

TWELVEDATA_SDK_AVAILABLE = True  # Runtime detection will confirm availability

# --- GPU vs CPU Control (must be set BEFORE imports) ---

# ============================================

# GPU Control: Independent flags for each framework

# ============================================



# PyTorch models (LSTM/TCN/GRU) GPU control

# For 164 tickers: GPU may be faster per model, but reduces parallelism (3 workers vs 15)

# For 5000 tickers: CPU is often FASTER due to higher parallelism

PYTORCH_USE_GPU = True  # True = use GPU for LSTM/TCN/GRU, False = CPU only



# XGBoost GPU control

# XGBoost GPU is stable and fast - recommended to keep enabled

XGBOOST_USE_GPU = True  # True = use device='cuda', False = use device='cpu'



# Legacy flag - kept for backward compatibility, derived from PYTORCH_USE_GPU

FORCE_CPU = not PYTORCH_USE_GPU  # Deprecated: use PYTORCH_USE_GPU instead



try:

    import torch

    PYTORCH_AVAILABLE = True

    # Detect if CUDA is actually available (independent of FORCE_CPU)

    # FORCE_CPU only controls PyTorch model placement, not XGBoost

    CUDA_AVAILABLE = torch.cuda.is_available()

except ImportError:

    PYTORCH_AVAILABLE = False

    CUDA_AVAILABLE = False

USE_LSTM = True

USE_GRU = False



# --- Volatility Ensemble Parameters ---

MAX_PORTFOLIO_VOLATILITY = 0.80  # 80% annualized max portfolio volatility (allows 10 positions with volatile stocks)

MAX_SINGLE_STOCK_VOLATILITY = 0.50  # 50% annualized max for any single stock

VOLATILITY_LOOKBACK_DAYS = 30  # Days to calculate volatility



# --- Universe / selection

MARKET_SELECTION = {

    "ALPACA_STOCKS": False,    # DISABLED - Using curated indices instead

    "NASDAQ_ALL": False,

    "NASDAQ_100": True,        # ENABLED - ~100 tech stocks for growth

    "SP500": True,             # ~500 stocks

    "SP400_MIDCAP": True,      # ENABLED - ~400 US mid-cap stocks

    "SP600_SMALLCAP": True,    # ENABLED - ~600 US small-cap stocks (quality filtered)

    "DOW_JONES": True,         # ~30 stocks

    "POPULAR_ETFS": True,      # ENABLED - Include popular ETFs (includes sector ETFs)

    "CRYPTO": True,            # ENABLED - Bitcoin, Ethereum, and crypto stocks

    "DAX": True,

    "MDAX": True,

    "SMI": True,

    "FTSE_MIB": True,         # ENABLED - Italian market

    "CAC_40": True,           # ENABLED - French market

    "IBEX_35": True,          # ENABLED - Spanish market

    "SWISS_MTI": True,        # ENABLED - Swiss market

}

# Explicitly exclude symbols from the candidate universe returned by
# `ticker_selection.get_all_tickers()`. Examples:
# - "SPHR"
# - "BRK.B" (normalized to "BRK-B")
# - "ENR.DE"
EXCLUDED_TICKERS = [
]

# If ALPACA_STOCKS is enabled, Alpaca can return thousands of symbols.

# Set very high to effectively include (almost) all Alpaca-tradable US equities.

ALPACA_STOCKS_LIMIT = 20000  # High limit = train models for ALL tradable stocks



# Exchange filter for Alpaca asset list. Use ["NASDAQ"] to restrict to NASDAQ only.

ALPACA_STOCKS_EXCHANGES = []  # NASDAQ only

N_TOP_TICKERS           = 20000     # Reduced from 2000 - balance quality and quantity

TOP_TICKER_SELECTION_LOOKBACK = "1Y"     # Try 1-month for more responsive selection

BATCH_DOWNLOAD_SIZE     = 10000     # Download in batches of 1000

PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability

PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals



# --- Parallel Processing

from multiprocessing import cpu_count

def _auto_detect_num_processes():
    """Automatically detect a safe general-purpose worker count based on available RAM."""
    cpu_safe_processes = max(1, cpu_count() - 2)
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        reserve_ram_gb = 4.0 if total_ram_gb >= 16 else 2.0
        ram_per_worker_gb = 3.0 if total_ram_gb >= 32 else 2.0
        usable_ram_gb = max(ram_per_worker_gb, available_ram_gb - reserve_ram_gb)
        ram_safe_processes = max(1, int(usable_ram_gb // ram_per_worker_gb))
        safe_processes = max(1, min(cpu_safe_processes, ram_safe_processes))

        print(f"   💾 Auto-detected: {total_ram_gb:.1f}GB total RAM, {available_ram_gb:.1f}GB available")
        print(
            f"   🔧 Safe processes: {safe_processes} "
            f"(cpu limit={cpu_safe_processes}, ram limit={ram_safe_processes})"
        )
        return safe_processes
    except ImportError:
        return cpu_safe_processes
    except Exception:
        return cpu_safe_processes

# Auto-detect a safe multiprocessing worker count based on RAM
NUM_PROCESSES = _auto_detect_num_processes()  # For data validation, ticker selection, and backtest data collection

NUM_PROCESSES = 16

# Parallel processing threshold - only use parallel processing for ticker lists larger than this

# Below this threshold, sequential processing is faster due to lower overhead

PARALLEL_THRESHOLD     = 200      # Use parallel only for >200 tickers



# Enable parallel strategy execution within each day (experimental)

# This runs strategy selections in parallel to improve CPU usage

ENABLE_PARALLEL_STRATEGIES = True   # Run strategies in parallel within each day

_PARALLEL_STRATEGY_PILOT_CONFIG: Dict[str, Dict[str, object]] = {
    "adaptive_ensemble": {"enable_flag": "ENABLE_ADAPTIVE_STRATEGY", "use_buffer": True},
    "risk_adj_mom_sentiment": {"enable_flag": "ENABLE_RISK_ADJ_MOM_SENTIMENT", "use_buffer": True},
    "multi_tf_ensemble": {"enable_flag": "ENABLE_MULTI_TIMEFRAME_ENSEMBLE", "use_buffer": True},
    "quality_momentum": {"enable_flag": "ENABLE_QUALITY_MOM", "use_buffer": False},
    "ratio_3m_1y": {"enable_flag": "ENABLE_3M_1Y_RATIO", "use_buffer": True},
    "ratio_1m_3m": {"enable_flag": "ENABLE_1M_3M_RATIO", "use_buffer": True},
    "ratio_1y_3m": {"enable_flag": "ENABLE_3M_1Y_RATIO", "use_buffer": True},
    "sma50_1y_delta_ratio": {"enable_flag": "ENABLE_SMA50_1Y_DELTA_RATIO", "use_buffer": True},
    "multi_timeframe_intraday": {"enable_flag": "ENABLE_MULTI_TIMEFRAME_INTRADAY_ENSEMBLE", "use_buffer": True},
    "dynamic_bh_1y": {"enable_flag": "ENABLE_DYNAMIC_BH_1Y", "use_buffer": True},
    "dynamic_bh_6m": {"enable_flag": "ENABLE_DYNAMIC_BH_6M", "use_buffer": True},
    "dynamic_bh_3m": {"enable_flag": "ENABLE_DYNAMIC_BH_3M", "use_buffer": True},
    "dynamic_bh_1m": {"enable_flag": "ENABLE_DYNAMIC_BH_1M", "use_buffer": True},
    "dynamic_bh_1y_trailing_stop": {"enable_flag": "ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP", "use_buffer": True},
    "risk_adj_mom": {"enable_flag": "ENABLE_RISK_ADJ_MOM", "use_buffer": True},
    "risk_adj_mom_6m": {"enable_flag": "ENABLE_RISK_ADJ_MOM_6M", "use_buffer": True},
    "risk_adj_mom_6m_monthly": {"enable_flag": "ENABLE_RISK_ADJ_MOM_6M_MONTHLY", "use_buffer": True},
    "risk_adj_mom_3m": {"enable_flag": "ENABLE_RISK_ADJ_MOM_3M", "use_buffer": True},
    "risk_adj_mom_3m_monthly": {"enable_flag": "ENABLE_RISK_ADJ_MOM_3M_MONTHLY", "use_buffer": True},
    "risk_adj_mom_1m": {"enable_flag": "ENABLE_RISK_ADJ_MOM_1M", "use_buffer": True},
    "risk_adj_mom_1m_monthly": {"enable_flag": "ENABLE_RISK_ADJ_MOM_1M_MONTHLY", "use_buffer": True},
    "risk_adj_mom_3m_sentiment": {"enable_flag": "ENABLE_RISK_ADJ_MOM_3M_SENTIMENT", "use_buffer": True},
    "vol_sweet_mom": {"enable_flag": "ENABLE_VOL_SWEET_MOM", "use_buffer": True},
    "risk_adj_mom_3m_market_up": {"enable_flag": "ENABLE_RISK_ADJ_MOM_3M_MARKET_UP", "use_buffer": True},
    "risk_adj_mom_3m_with_stops": {"enable_flag": "ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS", "use_buffer": True},
    "risk_adj_mom_1m_vol_sweet": {"enable_flag": "ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET", "use_buffer": True},
    "bh_1y_volsweet_accel": {"enable_flag": "ENABLE_BH_1Y_VOL_SWEET_ACCEL", "use_buffer": True},
    "bh_1y_dynamic_accel": {"enable_flag": "ENABLE_BH_1Y_DYNAMIC_ACCEL", "use_buffer": True},
    "trend_atr": {"enable_flag": "ENABLE_TREND_FOLLOWING_ATR", "use_buffer": True},
    "confirmed_leaders": {"enable_flag": "ENABLE_CONFIRMED_LEADERS", "use_buffer": True},
    "foresight_mimic": {"enable_flag": "ENABLE_FORESIGHT_MIMIC", "use_buffer": True},
    "bh_1y_1m_rank": {"enable_flag": "ENABLE_BH_1Y_1M_RANK", "use_buffer": True},
    "bh_1y_6m_rank": {"enable_flag": "ENABLE_BH_1Y_6M_RANK", "use_buffer": True},
    "bh_1y_6m_blend": {"enable_flag": "ENABLE_BH_1Y_6M_BLEND", "use_buffer": True},
    "early_leader_accel": {"enable_flag": "ENABLE_EARLY_LEADER_ACCEL", "use_buffer": True},
    "bh_1y_sma200": {"enable_flag": "ENABLE_BH_1Y_SMA200", "use_buffer": True},
    "bh_1y_fcf_rank": {"enable_flag": "ENABLE_BH_1Y_FCF_RANK", "use_buffer": True},
    "defensive_momentum": {"enable_flag": "ENABLE_DEFENSIVE_MOMENTUM", "use_buffer": True},
    "turnaround": {"enable_flag": "ENABLE_TURNAROUND", "use_buffer": True},
    "deep_recovery": {"enable_flag": "ENABLE_DEEP_RECOVERY", "use_buffer": False},
    "price_curvature": {"enable_flag": "ENABLE_PRICE_CURVATURE", "use_buffer": True},
    "price_curvature_1y_slope": {"enable_flag": "ENABLE_PRICE_CURVATURE_1Y_SLOPE", "use_buffer": True},
    "momentum_volatility_hybrid": {"enable_flag": "ENABLE_MOMENTUM_VOLATILITY_HYBRID", "use_buffer": True},
    "momentum_volatility_hybrid_6m": {"enable_flag": "ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M", "use_buffer": True},
    "momentum_volatility_hybrid_1y": {"enable_flag": "ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y", "use_buffer": True},
    "momentum_volatility_hybrid_1y3m": {"enable_flag": "ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M", "use_buffer": True},
    "mom_accel": {"enable_flag": "ENABLE_MOMENTUM_ACCELERATION", "use_buffer": True},
    "concentrated_3m": {"enable_flag": "ENABLE_CONCENTRATED_3M", "use_buffer": True},
    "dual_momentum": {"enable_flag": "ENABLE_DUAL_MOMENTUM", "use_buffer": True},
    "elite_hybrid": {"enable_flag": "ENABLE_ELITE_HYBRID", "use_buffer": True},
    "elite_risk": {"enable_flag": "ENABLE_ELITE_RISK", "use_buffer": True},
    "volatility_adj_mom": {"enable_flag": "ENABLE_VOLATILITY_ADJ_MOM", "use_buffer": False},
    "bb_mean_reversion": {"enable_flag": "ENABLE_BB_MEAN_REVERSION", "use_buffer": False},
    "bb_breakout": {"enable_flag": "ENABLE_BB_BREAKOUT", "use_buffer": False},
    "bb_squeeze_breakout": {"enable_flag": "ENABLE_BB_SQUEEZE_BREAKOUT", "use_buffer": False},
    "bb_rsi_combo": {"enable_flag": "ENABLE_BB_RSI_COMBO", "use_buffer": False},
    "trend_breakout": {"enable_flag": "ENABLE_TREND_BREAKOUT", "use_buffer": False},
    "ai_elite": {"enable_flag": "ENABLE_AI_ELITE", "use_buffer": True},
}



# Training batch size for parallel processing (how many tasks per batch)

TRAINING_BATCH_SIZE     = 100       # Smaller batches for better progress tracking



# Per-ticker prediction timeout (seconds). If a prediction takes longer, it will be skipped.

# - Set to 30 for normal use (predictions should be fast)

# - Set to None to disable timeout (not recommended - can hang forever)

PREDICTION_TIMEOUT = 30  # 30 seconds max per ticker prediction


# --- Backtest windows

BACKTEST_DAYS           =   50 # Backtest period in calendar days (~63=3mo, ~180=6mo, ~365=1yr)
BACKTEST_END_DATE       = False  # False = use current last trading day, or set "YYYY-MM-DD" to freeze runs for debugging

# Note: When RUN_BACKTEST_UNTIL_TODAY=True, actual backtest runs until today - 63 days



# --- Calendar days ---

CALENDAR_DAYS_PER_YEAR = 365



# --- Data Granularity ---

# Use daily data by default (recommended for momentum strategies)

# Set to True to enhance with intraday features (experimental)

ENABLE_INTRADAY_ENHANCEMENT = True    # Add 5m/15m data for enhanced features - ENABLED for AI Elite

INTRADAY_INTERVAL = "1h"              # Use 1-hour intervals for AI Elite training

INTRADAY_LOOKBACK_DAYS = 30            # Only use recent intraday data

# Set to True to run backtest until today - prediction horizon (ensures future data for validation)

# Set to False to subtract prediction horizon from end date (ensures future data for validation)

RUN_BACKTEST_UNTIL_TODAY = True   # Run backtest until today - horizon



# --- Walk-Forward Backtesting ---

# Models are trained during walk-forward backtest with periodic retraining



# --- Live Trading

LIVE_TRADING_ENABLED     = False       # ⚠️ Set to True to execute real orders (start with False for dry-run)

MODEL_MAX_AGE_DAYS        = 1           # Only use models trained in last X days

USE_PAPER_TRADING        = True        # True = paper trading (fake money), False = REAL MONEY ⚠️

TOP_N_STOCKS             = 10         # Number of stocks to hold in portfolio (should be much smaller than N_TOP_TICKERS)



# --- Live Trading Strategy Selection ---

# Choose which strategy to use for live trading:

# 'ai_volatility_ensemble' = 🤖 AI Vol Ens - AI-enhanced volatility ensemble (NEW)

# 'dynamic_bh_1y'          = Dynamic BH 1Y - Annual rebalancing

# 'dynamic_bh_3m'          = Dynamic BH 3M - Quarterly rebalancing

# 'risk_adj_mom'           = Risk-Adjusted Momentum

# 'quality_momentum'       = Quality + Momentum

# 'adaptive_ensemble'      = Adaptive Meta-Ensemble

# 'sentiment_ensemble'     = Sentiment-Enhanced Ensemble

LIVE_TRADING_STRATEGY    = None  # Set explicitly for live trading; no default strategy



# --- Walk-Forward Retraining Frequency ---

# How often to retrain models during walk-forward backtest

# Options:

#   5  = Weekly retraining (aggressive, best for high volatility/penny stocks)

#   10 = Bi-weekly retraining (balanced, recommended for volatile stocks)

#   20 = Monthly retraining (conservative, recommended for S&P 500 / stable large-caps)

#   60 = Quarterly retraining (rare, only for very stable/long-term strategies)

RETRAIN_FREQUENCY_DAYS = 1  # Retrain every 10 days - aligned with prediction horizon

ENABLE_WALK_FORWARD_RETRAINING = False  # Enable AI retraining during walk-forward backtests



# --- Backtest Period Enable/Disable Flags ---

ENABLE_1YEAR_BACKTEST   = True   # Enabled - For simulation and strategy validation



# --- Training Period Enable/Disable Flags ---

# ENABLE_1YEAR_TRAINING removed - models are now trained during walk-forward backtest



# --- Portfolio Strategy Enable/Disable Flags ---

# Set to False to disable specific portfolios in the backtest

ENABLE_MOMENTUM_AI_HYBRID = False  # DISABLED - Momentum+AI Hybrid (just 3M momentum, no actual AI)

ENABLE_AI_VOLATILITY_ENSEMBLE = True  # ENABLED - AI Volatility Ensemble



# Traditional strategies

ENABLE_STATIC_BH        = True   # ENABLED - Static Buy & Hold benchmark

ENABLE_DYNAMIC_BH_1Y    = True   # ENABLED - Dynamic BH 1-year

ENABLE_DYNAMIC_BH_3M    = True   # ENABLED - Dynamic BH 3-month

ENABLE_DYNAMIC_BH_1M    = True   # ENABLED - Dynamic BH 1-month

ENABLE_DYNAMIC_BH_6M    = True   # ENABLED - Dynamic BH 6-month

ENABLE_STATIC_BH_6M     = True   # ENABLED - Static BH 6-month

ENABLE_RISK_ADJ_MOM     = True   # ENABLED - Risk-Adjusted Momentum

ENABLE_MEAN_REVERSION   = True   # ENABLED - Mean Reversion

ENABLE_SEASONAL         = True   # ENABLED - Seasonal strategy

ENABLE_QUALITY_MOM      = True   # ENABLED - Quality + Momentum

ENABLE_VOLATILITY_ADJ_MOM = True  # ENABLED - Volatility-Adjusted Momentum strategy

ENABLE_DYNAMIC_BH_1Y_VOL_FILTER = True  # NEW - Dynamic BH 1Y with Volatility Filter

ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP = True   # ENABLED - Dynamic BH 1Y with trailing stop

ENABLE_SECTOR_ROTATION = True   # ENABLED - Sector Rotation Strategy

ENABLE_3M_1Y_RATIO = True   # ENABLED - 3M/1Y Ratio Strategy

ENABLE_1M_3M_RATIO = True   # ENABLED - 1M/3M Ratio Strategy (short-term momentum acceleration)

ENABLE_SMA50_1Y_DELTA_RATIO = True   # NEW - rank tickers by (Close - SMA50) / (6M return now - 6M return 50d ago)

ENABLE_MOMENTUM_VOLATILITY_HYBRID = True   # ENABLED - Momentum-Volatility Hybrid Strategy

ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M = True   # ENABLED - Momentum-Volatility Hybrid 6M Strategy (6-month lookback)

ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y = True   # ENABLED - Momentum-Volatility Hybrid 1Y Strategy (1-year lookback)

ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M = True   # ENABLED - Momentum-Volatility Hybrid 1Y/3M Ratio Strategy (strong 1Y, weak 3M)

ENABLE_ADAPTIVE_STRATEGY = True   # ENABLED - Adaptive Meta-Ensemble Strategy

ENABLE_VOLATILITY_ENSEMBLE = False  # Removed - Volatility Ensemble is treated as a meta strategy

ENABLE_ENHANCED_VOLATILITY = True   # ENABLED - Enhanced Volatility Trader (ATR stops + take profits)

ENABLE_RISK_ADJ_MOM_SENTIMENT = True   # NEW - Risk-Adjusted Momentum + Sentiment Strategy

ENABLE_SENTIMENT_ENSEMBLE = True   # ENABLED - Sentiment-Enhanced Ensemble Strategy



# --- Multi-Horizon Ensemble Strategy ---

# Combines signals from different timeframes for better entry/exit timing

ENABLE_MULTI_TIMEFRAME_ENSEMBLE = True   # NEW - Multi-Horizon Ensemble Strategy
ENABLE_MULTI_TIMEFRAME_INTRADAY_ENSEMBLE = True   # NEW - Multi-Horizon Ensemble using hourly data for medium/short horizons



# Additional strategy enable flags

ENABLE_TURNAROUND = True   # ENABLED - Turnaround Strategy (buy depressed stocks)

ENABLE_DEEP_RECOVERY = True   # ENABLED - Deep Recovery Strategy (contrarian: beaten-down stocks with reversal signals, uses full universe)
DEEP_RECOVERY_MIN_DATA_DAYS = 253   # Require roughly 1Y of history before calling it a deep recovery setup
DEEP_RECOVERY_MIN_YEAR_WINDOW_ROWS = 200   # Require a well-populated trailing year before using the 52W-high drawdown

ENABLE_MOMENTUM_VOLATILITY_HYBRID = True   # ENABLED - Momentum-Volatility Hybrid Strategy

ENABLE_3M_1Y_RATIO = True   # ENABLED - 3M/1Y Ratio Strategy

ENABLE_PRICE_CURVATURE = True   # SG slope strategy on daily closes with positive 1M filter
ENABLE_STATIC_PRICE_CURVATURE = True   # Static Price Curvature with periodic rebalancing
ENABLE_PRICE_CURVATURE_1Y_SLOPE = True   # SG slope strategy on a 1Y performance-prefiltered pool
ENABLE_STATIC_PRICE_CURVATURE_1Y_SLOPE = True   # Static 1Y-prefiltered SG slope strategy

ENABLE_VOTING_ENSEMBLE = True   # ENABLED - Voting Ensemble Strategy



# --- New Advanced Strategies ---

ENABLE_MOMENTUM_ACCELERATION = True   # NEW - Momentum Acceleration (3M momentum + acceleration filter)

ENABLE_CONCENTRATED_3M = True   # NEW - Concentrated 3M + Vol Filter (fewer positions, volatility filtered)

ENABLE_DUAL_MOMENTUM = True   # NEW - Dual Momentum (absolute + relative momentum)

ENABLE_TREND_FOLLOWING_ATR = True   # NEW - Trend Following with ATR Trailing Stop

ENABLE_TREND_BREAKOUT = True   # NEW - Trend Breakout (same entry as Trend ATR but no ATR selling, uses smart_rebalance)
TREND_BREAKOUT_USE_ATR_STOP = True  # If True, Trend Breakout uses ATR trailing stop in smart_rebalance
TREND_BREAKOUT_ATR_MULTIPLIER = 2.0  # ATR multiplier for trailing stop

ENABLE_CONFIRMED_LEADERS = True   # NEW - Confirmed Leaders (1Y leaders + 6M/3M/1M continuation + vol cap)
CONFIRMED_LEADERS_SMA_DAYS = 200   # Require price above 200-day SMA
CONFIRMED_LEADERS_MIN_1Y_RETURN = 25.0   # Minimum 1Y return for established leadership
CONFIRMED_LEADERS_MIN_6M_RETURN = 10.0   # Minimum 6M continuation
CONFIRMED_LEADERS_MIN_3M_RETURN = 5.0   # Minimum 3M continuation
CONFIRMED_LEADERS_MIN_1M_RETURN = 0.0   # 1M must still be positive
CONFIRMED_LEADERS_MAX_VOLATILITY = 0.85   # Annualized volatility cap

ENABLE_FORESIGHT_MIMIC = True   # NEW - 1Y leaders re-ranked by short-term acceleration
FORESIGHT_MIMIC_MIN_1Y_RETURN = 25.0   # Minimum 1Y return to qualify for the accel pool
FORESIGHT_MIMIC_MIN_ACCELERATION = 5.0   # Require improving 1M momentum versus the prior month
FORESIGHT_MIMIC_MIN_10D_RETURN = 2.0   # Require some recent thrust, not just a statistical accel uptick

ENABLE_BH_1Y_1M_RANK = True   # NEW - select top 1Y performers and rank that pool by 1M momentum
ENABLE_BH_1Y_6M_RANK = True   # NEW - select top 1Y performers and rank that pool by 6M performance
ENABLE_BH_1Y_6M_BLEND = True   # NEW - positive 1Y/6M leaders above SMA200 ranked by a 50/50 1Y+6M percentile blend
ENABLE_EARLY_LEADER_ACCEL = True   # NEW - earlier winners with 3M strength, positive 1M/6M/1Y, above SMA200, excluding extreme 1Y rockets
ENABLE_BH_1Y_SMA200 = True   # NEW - select top 1Y performers that are above their SMA200
BH_1Y_SMA200_DAYS = 200   # Require price above the trailing 200-day simple moving average
ENABLE_BH_1Y_FCF_RANK = True   # NEW - select top 1Y performers and rank that pool by free cash flow
BH_1Y_FCF_RANK_REQUIRE_POSITIVE = True   # Ignore tickers whose latest reported FCF is non-positive

ENABLE_DEFENSIVE_MOMENTUM = True   # NEW - Defensive Momentum (cash-first trend filter + momentum ranking)
DEFENSIVE_MOMENTUM_SMA_DAYS = 200   # Require price above 200-day SMA
DEFENSIVE_MOMENTUM_MIN_1Y_RETURN = 15.0   # Minimum 1Y return to consider a stock
DEFENSIVE_MOMENTUM_MIN_3M_RETURN = 5.0   # Minimum 3M return to consider a stock
DEFENSIVE_MOMENTUM_MAX_VOLATILITY = 0.35   # Annualized volatility cap
DEFENSIVE_MOMENTUM_MIN_MARKET_20D_RETURN = 0.0   # Market must have positive 20D momentum

ENABLE_ELITE_HYBRID = True   # NEW - Elite Hybrid (Mom-Vol 6M + 1Y/3M Ratio - combines top 2 most consistent strategies)

ENABLE_ELITE_RISK = True   # NEW - Elite Risk (Risk-Adj Mom base + Elite Hybrid dip/vol bonuses)

ENABLE_RISK_ADJ_MOM_6M = True   # NEW - Risk-Adj Mom 6M (same as Risk-Adj Mom but 6M window)

ENABLE_RISK_ADJ_MOM_6M_MONTHLY = True   # NEW - Risk-Adj Mom 6M Monthly (same scoring, rebalance start of month only)

ENABLE_RISK_ADJ_MOM_3M = True   # NEW - Risk-Adj Mom 3M (same as Risk-Adj Mom but 3M window)

ENABLE_RISK_ADJ_MOM_3M_MONTHLY = True   # NEW - Risk-Adj Mom 3M Monthly (same scoring, rebalance start of month only)

ENABLE_RISK_ADJ_MOM_3M_SENTIMENT = True   # NEW - Risk-Adj Mom 3M + Sentiment (momentum + price-derived sentiment)

ENABLE_RISK_ADJ_MOM_3M_MARKET_UP = True   # NEW - Risk-Adj Mom 3M Market-Up Only (rebalance only when market is up)

ENABLE_RISK_ADJ_MOM_3M_WITH_STOPS = True   # NEW - Risk-Adj Mom 3M with Stops (5% stop loss, 15% take profit)

ENABLE_VOL_SWEET_MOM = True        # NEW - Vol-Sweet Momentum (3M momentum + vol sweet spot + sentiment)

ENABLE_ULTIMATE = True            # NEW - Ultimate Strategy (multi-factor hybrid: 1M/3M momentum + vol sweet spot + acceleration + trend + volume)

ENABLE_RISK_ADJ_MOM_1M = True   # NEW - Risk-Adj Mom 1M (same as Risk-Adj Mom but 1M window)

ENABLE_RISK_ADJ_MOM_1M_MONTHLY = True   # NEW - Risk-Adj Mom 1M Monthly (same scoring, rebalance start of month only)

ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET = True   # NEW - Risk-Adj Mom 1M + Vol-Sweet (best Sharpe + vol filter)
ENABLE_BH_1Y_VOL_SWEET_ACCEL = True        # NEW - BH 1Y VolSweet Accel (Static BH 1Y + 1M VolSweet, ranked by acceleration)
ENABLE_BH_1Y_DYNAMIC_ACCEL = True          # NEW - BH 1Y Dynamic Accel (dynamic rebalancing based on market conditions)

ENABLE_AI_ELITE = True   # NEW - AI Elite (ML-powered scoring of momentum + dip opportunities)

ENABLE_VOTING_ENSEMBLE_AI_REBALANCE = True  # NEW - Voting Ensemble selection + AI rebalance decisions

ENABLE_AI_ELITE_MONTHLY_SHARED = True   # NEW - AI Elite Monthly Shared (monthly rebalance using daily AI Elite shared model)

ENABLE_AI_ELITE_FILTERED = True   # NEW - AI Elite Filtered (Risk-Adj Mom 3M pre-filter + AI Elite re-rank)

ENABLE_AI_ELITE_MARKET_UP = True   # NEW - AI Elite Market-Up (only rebalances when market is up)
ENABLE_AI_ELITE_ENSEMBLE = False   # NEW - AI Elite Ensemble (weighted avg of top 3 positive-R² models)
ENABLE_AI_ELITE_RANK_ENSEMBLE = False   # NEW - AI Elite Rank Ensemble (weighted rank avg of top 3 positive-R² models)

ENABLE_TOP5_CONSISTENCY_BLEND = True   # NEW - Blend current top strategies using prior-day top-5 consistency weights

ENABLE_LLM_STRATEGY = False   # DISABLED - LLM Strategy (not implemented)

# --- Meta-Strategy Selectors (10 proposals for combining strategies) ---
# Bollinger Bands Strategies
ENABLE_BB_MEAN_REVERSION = True      # BB Mean Reversion (buy at lower band, sell at upper)
ENABLE_BB_BREAKOUT = True            # BB Breakout (buy when price breaks above upper band with volume)
ENABLE_BB_SQUEEZE_BREAKOUT = True    # BB Squeeze + Breakout (wait for squeeze, trade breakout)
ENABLE_BB_RSI_COMBO = True           # BB + RSI Combo (buy at lower band AND RSI < 30)

# AI Elite Parameters

AI_ELITE_RETRAIN_DAYS = 1  # Retrain model every 1 days

AI_ELITE_TRAINING_LOOKBACK = 180  # Days of history to use for training

AI_ELITE_FORWARD_DAYS = 5  # Predict performance over next N days

AI_REBALANCE_RETRAIN_DAYS = 1  # Retrain/continue training every day
AI_REBALANCE_TRAINING_LOOKBACK = AI_ELITE_TRAINING_LOOKBACK  # Use same lookback as AI Elite
AI_REBALANCE_FORWARD_DAYS = 5  # Evaluate replace-vs-keep over the next N trading days
AI_REBALANCE_MIN_SAMPLES = 80  # Minimum hold-vs-replace samples before fitting
AI_REBALANCE_MIN_PREDICTED_EDGE = 0.0  # Require positive predicted net switch advantage
AI_REBALANCE_FORCE_FRESH_TRAIN = False  # False = load existing model and continue where possible

AI_VOTING_REBALANCE_RETRAIN_DAYS = AI_REBALANCE_RETRAIN_DAYS
AI_VOTING_REBALANCE_TRAINING_LOOKBACK = AI_REBALANCE_TRAINING_LOOKBACK
AI_VOTING_REBALANCE_FORWARD_DAYS = AI_REBALANCE_FORWARD_DAYS
AI_VOTING_REBALANCE_MIN_PREDICTED_EDGE = AI_REBALANCE_MIN_PREDICTED_EDGE
AI_VOTING_REBALANCE_FORCE_FRESH_TRAIN = AI_REBALANCE_FORCE_FRESH_TRAIN

AI_ELITE_FORCE_FRESH_TRAIN = False  # False = load existing model and do incremental training; True = always fresh train
AI_ELITE_CATBOOST_USED_RAM_LIMIT = "24gb"  # Raise CatBoost CPU RAM budget to avoid continuation crashes
AI_ELITE_CATBOOST_GPU_RAM_PART = 0.95  # Let CatBoost use most of GPU memory when training fresh on GPU

# AI Champion Parameters
AI_CHAMPION_RETRAIN_DAYS = 1  # Retrain/continue training every day
AI_REGIME_RETRAIN_DAYS = 1  # Retrain/continue training every day
UNIVERSAL_MODEL_RETRAIN_DAYS = 1  # Retrain/continue training every day
AI_CHAMPION_FORWARD_DAYS = 1  # Predict the next trading day's winner
AI_CHAMPION_CONFIDENCE_THRESHOLD = 0.55  # Require moderate confidence before switching
AI_CHAMPION_HOLD_MARGIN = 0.08  # Keep current strategy unless the edge is meaningful

# AI Elite Intraday Configuration

AI_ELITE_INTRADAY_INTERVAL = "1h"  # Use 1-hour data for intraday features



# Momentum Acceleration Parameters

MOM_ACCEL_LOOKBACK_DAYS = 90  # 3-month momentum lookback

MOM_ACCEL_SHORT_LOOKBACK = 21  # 1-month for acceleration calculation

MOM_ACCEL_MIN_ACCELERATION = 0.0  # Minimum acceleration (current 1M > previous 1M)



# Concentrated 3M Parameters

# CONCENTRATED_3M_POSITIONS removed - now uses PORTFOLIO_SIZE

CONCENTRATED_3M_MAX_VOLATILITY = 0.40  # 40% max annualized volatility

CONCENTRATED_3M_REBALANCE_DAYS = 21  # Monthly rebalancing



# Dual Momentum Parameters

DUAL_MOM_LOOKBACK_DAYS = 90  # 3-month momentum for relative comparison

DUAL_MOM_ABSOLUTE_THRESHOLD = 0.0  # Must have positive absolute momentum

# DUAL_MOM_POSITIONS removed - now uses PORTFOLIO_SIZE

DUAL_MOM_RISK_OFF_TICKER = None  # Set to 'TLT' or 'SHY' for bonds, None for cash



# --- Inverse ETFs (for bear market protection) ---

ENABLE_INVERSE_ETFS = True  # Include inverse ETFs in ticker universe

INVERSE_ETFS = [

    # Standard inverse ETFs (1x short)

    'SH',    # ProShares Short S&P 500

    'PSQ',   # ProShares Short QQQ (Nasdaq-100)

    'DOG',   # ProShares Short Dow 30

    'RWM',   # ProShares Short Russell 2000

    # 2x leveraged inverse ETFs (use with caution - decay over time)

    'SDS',   # ProShares UltraShort S&P 500 (2x)

    'QID',   # ProShares UltraShort QQQ (2x)

    'DXD',   # ProShares UltraShort Dow 30 (2x)

    'TWM',   # ProShares UltraShort Russell 2000 (2x)

    # 3x leveraged inverse ETFs (VERY HIGH RISK - decay rapidly, only for hedging)

    'SOXS',  # Direxion Daily Semiconductor Bear 3X

    'SQQQ',  # ProShares UltraPro Short QQQ (3x)

    'SPXU',  # ProShares UltraPro Short S&P 500 (3x)

    'FAZ',   # Direxion Daily Financial Bear 3X

    'TZA',   # Direxion Daily Small Cap Bear 3X

    'TECS',  # Direxion Daily Technology Bear 3X

]



# Inverse ETF Performance Filters (more lenient than regular stocks)

# These ETFs should be selected when market is falling, so we use shorter lookbacks

INVERSE_ETF_MIN_PERFORMANCE_1M = 0.02   # 2% minimum 1-month performance (market falling)

INVERSE_ETF_MIN_PERFORMANCE_3M = 0.0    # 0% minimum 3-month (just needs to be flat/positive)

INVERSE_ETF_SKIP_1Y_FILTER = True       # Skip 1Y filter (inverse ETFs lose in bull markets)



# Trend Following ATR Parameters

TREND_ATR_LOOKBACK_DAYS = 90  # 3-month for trend detection

TREND_ATR_PERIOD = 14  # ATR calculation period

TREND_ATR_TRAILING_MULT = 2.0  # ATR multiplier for trailing stop

TREND_ATR_ENTRY_BREAKOUT = 20  # Days for breakout detection



MULTI_TIMEFRAMES = ["long_term", "medium_term", "short_term"]    # Analysis horizons to score

MULTI_TIMEFRAME_LOOKBACK = {

    "long_term": 365,    # Long-term horizon: 1 year lookback

    "medium_term": 30,     # Medium-term horizon: 30 days lookback

    "short_term": 7       # Short-term horizon: 7 days lookback

}

MULTI_TIMEFRAME_WEIGHTS = {

    "long_term": 0.6,    # Long-term gets 60% weight (trend direction)

    "medium_term": 0.3,    # Medium-term gets 30% weight (momentum confirmation)

    "short_term": 0.1     # Short-term gets 10% weight (timing)

}

# Require consensus from at least this many timeframes

MULTI_TIMEFRAME_MIN_CONSENSUS = 2  # At least 2 timeframes must agree



# --- Strategy (separate from feature windows)

STRAT_SMA_SHORT         = 10

STRAT_SMA_LONG          = 20

ATR_PERIOD              = 14

ATR_MULT_TRAIL          = 2.0

ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing

PORTFOLIO_SIZE          = 5      # Number of stocks to hold in portfolio

PORTFOLIO_BUFFER_SIZE    = 10         # Sell when stock not in top X (buffer for stability)

TOTAL_CAPITAL           = 300000     # Total capital to invest ($300,000)

INVESTMENT_PER_STOCK    = TOTAL_CAPITAL / PORTFOLIO_SIZE  # Automatically calculated

TRANSACTION_COST        = 0.011      # 1.1% per trade leg (buy or sell)

DEBUG_REBALANCE_LOGS   = True       # Print detailed portfolio state before/after smart rebalancing

ENABLE_STOP_LOSS        = False      # Global enable/disable stop loss protection (overridden by per-strategy settings)

STOP_LOSS_PCT           = 0.05       # Default stop loss percentage (5%)

ENABLE_PROFIT_GUARD     = False      # Sell stocks when not in top 10 (no profit/loss consideration)



# --- Inverse ETF Hedge Strategy ---

# Instead of stop losses, add inverse ETFs when the shared trailing market regime turns down

# Hybrid approach: activate on trailing 5-day market weakness, then scale by decline size

ENABLE_INVERSE_ETF_HEDGE = True     # Add inverse ETFs when market crashes

# Gradual scaling thresholds after trailing 5-day market return turns non-positive
# Thresholds are stored as decimals in config but applied as percentage declines in code

# Example: trailing 5d return <= 0 starts hedge mode; -5% -> 20% hedge,
# -10% -> 50% hedge, -15% -> 80% hedge

INVERSE_ETF_HEDGE_THRESHOLD_LOW = 0.05    # 5% decline = 20% hedge

INVERSE_ETF_HEDGE_THRESHOLD_MED = 0.10      # 10% decline = 50% hedge

INVERSE_ETF_HEDGE_THRESHOLD_HIGH = 0.15     # 15% decline = 80% hedge

INVERSE_ETF_HEDGE_BASE_ALLOCATION = 0.20    # Always keep 20% in equity (never 100% hedge)

INVERSE_ETF_HEDGE_MAX_ALLOCATION = 0.80     # Max 80% in hedge (never 100%)

INVERSE_ETF_HEDGE_PREFERENCE = ['SOXS', 'SQQQ', 'SPXU', 'FAZ', 'SH', 'PSQ']  # Preferred hedge ETFs

INVERSE_ETF_HEDGE_MIN_HOLD_DAYS = 5         # Minimum days to hold hedge before exiting (prevents whipsaw)



# --- Analyst Recommendation Strategy ---

ENABLE_ANALYST_RECOMMENDATION = False       # Disable analyst recommendation strategy

ANALYST_LOOKBACK_DAYS = 60                  # Days to look back for analyst actions

ANALYST_MIN_ACTIONS = 1                     # Minimum analyst actions required

ANALYST_REBALANCE_DAYS = 7                  # Rebalance weekly (analyst data doesn't change daily)



# --- Strategy-Specific Stop Loss Configuration ---

# All stop loss values respect the global ENABLE_STOP_LOSS flag

# Based on backtest analysis: Some strategies benefit from stop loss, others don't



# Strategies with 5% stop loss (show positive improvement with stop loss protection)

AI_STRATEGY_STOP_LOSS = 0.05           # +17.7% improvement (+31.9% vs +14.2%)

VOLATILITY_ENSEMBLE_STOP_LOSS = 0.05   # +6.8% improvement (-5.1% vs -11.9%)

MOM_VOL_HYBRID_STOP_LOSS = 0.05        # +5.5% improvement (+45.6% vs +40.1%)

RATIO_3M_1Y_STOP_LOSS = 0.05           # +4.0% improvement (+21.0% vs +17.0%)

STATIC_BH_6M_STOP_LOSS = 0.05          # +3.5% improvement (+30.5% vs +27.0%)

TURNAROUND_STOP_LOSS = 0.05            # +2.5% improvement (+20.0% vs +17.5%)

PRICE_CURVATURE_STOP_LOSS = 0.05       # Default stop loss for upward-curvature trend strategy

ADAPTIVE_ENSEMBLE_STOP_LOSS = 0.05     # +2.4% improvement (+18.0% vs +15.6%)

CONCENTRATED_3M_STOP_LOSS = 0.05       # +1.4% improvement (+5.6% vs +4.2%)

STATIC_BH_1M_STOP_LOSS = 0.05          # +1.2% improvement (+7.7% vs +6.5%)

AI_VOLATILITY_ENSEMBLE_STOP_LOSS = 0.05 # +1.0% improvement (-7.8% vs -8.8%)

STATIC_BH_3M_STOP_LOSS = 0.05          # +0.8% improvement (+30.1% vs +29.3%)

STATIC_BH_1Y_STOP_LOSS = 0.05          # +0.7% improvement (+17.1% vs +16.4%)

SECTOR_ROTATION_STOP_LOSS = 0.05       # +0.7% improvement (+4.1% vs +3.4%)

MOM_ACCELERATION_STOP_LOSS = 0.05      # +0.4% improvement (-3.6% vs -4.0%)



# Strategies without stop loss (perform better with profit guard only)

MEAN_REVERSION_STOP_LOSS = 0.0         # -10.2% worse with stop (+2.0% vs -8.2%)

RISK_ADJ_MOM_STOP_LOSS = 0.0           # -3.1% worse with stop (+15.9% vs +12.8%)

QUALITY_MOMENTUM_STOP_LOSS = 0.0       # -2.8% worse with stop (+25.8% vs +23.0%)

MOMENTUM_6M_STOP_LOSS = 0.0            # -2.4% worse with stop (+16.3% vs +13.9%)

TREND_FOLLOWING_ATR_STOP_LOSS = 0.0   # -1.8% worse with stop (+20.0% vs +18.2%)

AI_ELITE_STOP_LOSS = 0.0               # -1.3% worse with stop (+19.3% vs +18.0%)

VOLATILITY_TRADER_STOP_LOSS = 0.0      # -1.1% worse with stop (+18.1% vs +17.0%)

ENHANCED_VOLATILITY_STOP_LOSS = 0.0    # -0.7% worse with stop (+18.6% vs +17.9%)

BUY_HOLD_STOP_LOSS = 0.0               # -0.6% worse with stop (+17.1% vs +16.5%)



# Dictionary for backward compatibility (used by backtesting.py)

STRATEGY_STOP_LOSS_PCT = {

    'AI Strategy': AI_STRATEGY_STOP_LOSS,

    'Volatility Ensemble': VOLATILITY_ENSEMBLE_STOP_LOSS,

    'Mom-Vol Hybrid': MOM_VOL_HYBRID_STOP_LOSS,

    '3M/1Y Ratio': RATIO_3M_1Y_STOP_LOSS,

    'Static BH 6M': STATIC_BH_6M_STOP_LOSS,

    'Turnaround': TURNAROUND_STOP_LOSS,

    'Static Price Curvature': PRICE_CURVATURE_STOP_LOSS,

    'Price Curvature': PRICE_CURVATURE_STOP_LOSS,

    'Static Price Curvature 1Y Slope': PRICE_CURVATURE_STOP_LOSS,

    'Price Curvature 1Y Slope': PRICE_CURVATURE_STOP_LOSS,

    'Adaptive Ensemble': ADAPTIVE_ENSEMBLE_STOP_LOSS,

    'Concentrated 3M': CONCENTRATED_3M_STOP_LOSS,

    'Static BH 1M': STATIC_BH_1M_STOP_LOSS,

    'AI Volatility Ensemble': AI_VOLATILITY_ENSEMBLE_STOP_LOSS,

    'Static BH 3M': STATIC_BH_3M_STOP_LOSS,

    'Static BH 1Y': STATIC_BH_1Y_STOP_LOSS,

    'Sector Rotation': SECTOR_ROTATION_STOP_LOSS,

    'Mom Acceleration': MOM_ACCELERATION_STOP_LOSS,

    'Mean Reversion': MEAN_REVERSION_STOP_LOSS,

    'Risk-Adj Mom': RISK_ADJ_MOM_STOP_LOSS,

    'Risk-Adj Mom Sentiment': RISK_ADJ_MOM_STOP_LOSS,

    'Quality Momentum': QUALITY_MOMENTUM_STOP_LOSS,

    'Momentum 6M': MOMENTUM_6M_STOP_LOSS,

    'Trend Following ATR': TREND_FOLLOWING_ATR_STOP_LOSS,

    'AI Elite': AI_ELITE_STOP_LOSS,

    'Volatility Trader': VOLATILITY_TRADER_STOP_LOSS,

    'Enhanced Volatility': ENHANCED_VOLATILITY_STOP_LOSS,

    'Buy & Hold': BUY_HOLD_STOP_LOSS,

    '1Y/3M Ratio': 0.0,            # -0.2% worse (+7.6% vs +7.4%)

}



# --- Strategy Performance Filters ---

# Minimum performance thresholds for stock selection

# These can be used to filter out underperformers before strategy scoring

MIN_PERFORMANCE_1M = 0.0       # Require positive 1-month performance (>= 0%)

MIN_PERFORMANCE_1Y = -1.0      # Effectively disabled (-100%) while preserving the shared filter path

MIN_PERFORMANCE_6M = -1.0      # Effectively disabled (-100%) while preserving the shared filter path

MIN_PERFORMANCE_3M = -1.0      # Effectively disabled (-100%) while preserving the shared filter path

ENABLE_PERFORMANCE_FILTERS = True  # Enable the unified performance prefilter across strategies



# --- Universal Data Requirements ---

# Single source of truth for minimum data requirements across all strategies

MIN_DATA_DAYS_1Y = 180         # Minimum days required for 1-year calculations (reduced from 252)

MIN_DATA_DAYS_6M = 90          # Minimum days required for 6-month calculations (reduced from 126)

MIN_DATA_DAYS_3M = 45          # Minimum days required for 3-month calculations (reduced from 63)

MIN_DATA_DAYS_1M = 15          # Minimum days required for 1-month calculations (reduced from 21)

MIN_DATA_DAYS_GENERAL = 60     # General minimum data requirement for most strategies (reduced from 90)



# --- Strategy-Specific Data Requirements ---

# Additional minimum data requirements for specific strategies

MIN_DATA_DAYS_AI_ELITE_VOLATILITY = 30      # AI Elite volatility calculation

MIN_TRAINING_SAMPLES_AI_ELITE = 100         # AI Elite training samples

MIN_DATA_DAYS_ENHANCED_VOLATILITY = 10      # Enhanced volatility trader

MIN_DATA_DAYS_FACTOR_VALUE = 252            # Factor rotation value calculation

MIN_DATA_DAYS_PREDICTION_HORIZON = 1        # Minimum prediction horizon days

MIN_DATA_DAYS_SEQUENCE_LENGTH = 50          # Minimum for sequence models

MIN_DATA_DAYS_MOMENTUM_CONFIRM = 50         # Risk-adj momentum confirmation

MIN_DATA_DAYS_PERFORMANCE_DATA = 50         # Performance data validation

MIN_DATA_DAYS_VALID_CLOSE = 10              # Minimum valid close prices

MIN_DATA_DAYS_DAILY_RETURNS = 5             # Minimum daily returns for volatility

MIN_DATA_DAYS_VOLUME_CONFIRM = 20           # Volume confirmation window

MIN_DATA_DAYS_THREE_MONTH_POINTS = 10       # Three month data points

MIN_DATA_DAYS_ONE_YEAR_POINTS = 50          # One year data points

MIN_DATA_DAYS_PERIOD_DATA = 30              # Period data for calculations



# --- Dynamic BH 1Y + Volatility Filter Parameters ---

# Maximum allowed annualized volatility for stock selection (as percentage)

# Higher values allow more volatile stocks, lower values are more conservative

DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY = 120.0  # Maximum 120% annualized volatility (allow most stocks)



# --- Dynamic BH 1Y + Trailing Stop Parameters ---

# Trailing stop to protect gains and limit downside

DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT = 20.0  # Sell if price drops 20% from peak

# Note: Uses AI_REBALANCE_FREQUENCY_DAYS for rebalancing frequency



# --- Sector Rotation Strategy Parameters ---

# PROPOSAL 2: Rotate between sector ETFs based on momentum

# SECTOR_ROTATION_TOP_N removed - now uses PORTFOLIO_SIZE

# Note: Uses AI_REBALANCE_FREQUENCY_DAYS for rebalancing frequency

SECTOR_ROTATION_MOMENTUM_WINDOW = 60  # 60-day momentum for sector selection

SECTOR_ROTATION_MIN_MOMENTUM = 0.0  # TEMPORARILY reduced to 0% for debugging - was 5.0%



# --- Data Freshness Check ---

# Maximum age of price data before rejecting it as stale (in days)

# Set to 5 to allow for weekends (Fri close -> Mon open = 3 days) + 1 holiday

DATA_FRESHNESS_MAX_DAYS = 5  # Maximum data age in days (allows weekends + holidays)



# --- Risk-Adjusted Momentum Improvements ---

# The existing Risk-Adjusted Momentum already uses return/volatility scoring

# PROPOSAL 1: Enhanced parameters for better performance

RISK_ADJ_MOM_PERFORMANCE_WINDOW = 365  # Days to look back for performance (1 year)

RISK_ADJ_MOM_VOLATILITY_WINDOW = 10   # REDUCED from 15 - less volatility penalty, more aggressive



# --- Momentum Confirmation Filter ---

# PROPOSAL 1: Less restrictive momentum confirmation

RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION = False  # DISABLED - too restrictive, allow more stocks

RISK_ADJ_MOM_CONFIRM_SHORT = True   # Available if re-enabled

RISK_ADJ_MOM_CONFIRM_MEDIUM = True  # Available if re-enabled

RISK_ADJ_MOM_CONFIRM_LONG = False    # DISABLED - reduce filtering

RISK_ADJ_MOM_MIN_CONFIRMATIONS = 1  # Only 1 timeframe needed if re-enabled



# --- Volume Confirmation Filter ---

# Require increasing volume to confirm price momentum strength

RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION = False  # TEMPORARILY DISABLED - too restrictive

RISK_ADJ_MOM_VOLUME_WINDOW = 20  # Days to compare recent volume vs average

RISK_ADJ_MOM_VOLUME_MULTIPLIER = 1.2  # Recent volume must be 20% above average



# --- Risk-Adjusted Momentum Minimum Score ---

# PROPOSAL 1: Lower minimum score to allow more high-potential stocks

RISK_ADJ_MOM_MIN_SCORE = 10.0  # FIXED: Lowered from 30.0 to allow more candidates



# --- Static Buy & Hold (BH) rebalancing period ---

# Set to 0 to maintain true buy & hold behavior (no rebalancing)

# These are default values - if OPTIMIZE_REBALANCE_HORIZON is True, the system will

# test multiple horizons (30-90 days) and pick the best one

STATIC_BH_1Y_REBALANCE_DAYS = 22  # Optimal rebalance period for 1Y selection

STATIC_BH_3M_REBALANCE_DAYS = 37  # Optimal rebalance period for 3M selection

STATIC_BH_1M_REBALANCE_DAYS = 27  # Optimal rebalance period for 1M selection

STATIC_BH_6M_REBALANCE_DAYS = 34  # Optimal rebalance period for 6M selection

STATIC_PRICE_CURVATURE_REBALANCE_DAYS = 22  # Static cadence for SG-slope price curvature picks

STATIC_PRICE_CURVATURE_1Y_SLOPE_REBALANCE_DAYS = 22  # Static cadence for 1Y-prefiltered SG slope picks

# --- Adaptive Rebalancing Strategies (based on Static BH 1Y) ---
# These strategies use adaptive triggers instead of fixed periodic rebalancing

ENABLE_STATIC_BH_1Y_VOLATILITY = True   # Rebalance when portfolio volatility > threshold
STATIC_BH_1Y_VOLATILITY_THRESHOLD = 0.02  # 2% volatility threshold

ENABLE_STATIC_BH_1Y_PERFORMANCE = True   # Rebalance when position weights deviate > threshold
STATIC_BH_1Y_PERFORMANCE_THRESHOLD = 0.25  # 25% deviation threshold

ENABLE_STATIC_BH_6M_PERFORMANCE = True   # BH 6M Perf Trig: same trigger, 6M ranking window
STATIC_BH_6M_PERFORMANCE_THRESHOLD = 0.25  # 25% deviation threshold

ENABLE_STATIC_BH_9M_PERFORMANCE = True   # BH 9M Perf Trig: same trigger, 9M ranking window
STATIC_BH_9M_PERFORMANCE_THRESHOLD = 0.25  # 25% deviation threshold

ENABLE_STATIC_BH_1Y_MOMENTUM = True   # Rebalance when portfolio momentum turns negative
STATIC_BH_1Y_MOMENTUM_LOOKBACK = 60   # Days for momentum calculation

ENABLE_STATIC_BH_1Y_ATR = True   # Rebalance when cumulative ATR change > threshold
STATIC_BH_1Y_ATR_THRESHOLD = 0.05  # 5% ATR change threshold

ENABLE_STATIC_BH_1Y_HYBRID = True   # Hybrid approach combining multiple triggers
STATIC_BH_1Y_HYBRID_MAX_DAYS = 15  # Safety net: rebalance at least every N days

# --- Static BH Monthly Rebalance Variants ---

# These are separate strategies that rebalance on the first trading day of each month

ENABLE_STATIC_BH_1Y_MONTHLY = True   # Static BH 1Y with monthly rebalance

ENABLE_STATIC_BH_6M_MONTHLY = True   # Static BH 6M with monthly rebalance

ENABLE_STATIC_BH_3M_MONTHLY = True   # Static BH 3M with monthly rebalance

ENABLE_STATIC_BH_1M_MONTHLY = True   # Static BH 1M with monthly rebalance

ENABLE_STATIC_BH_1Y_WEEKLY = True   # Static BH 1Y with weekly rebalance on the first trading day of the week

# --- Static BH 3M with Acceleration ---
ENABLE_STATIC_BH_3M_ACCEL = True   # Static BH 3M with accelerating momentum
STATIC_BH_3M_ACCEL_LOOKBACK_SHORT = 10   # Short momentum window (2 weeks)
STATIC_BH_3M_ACCEL_LOOKBACK_LONG = 21    # Long momentum window (1 month)
STATIC_BH_3M_ACCEL_WEIGHT = 0.3          # Weight for acceleration score (0-1)

# --- Enhanced Static BH 1Y Strategies ---

# These strategies add smart filters and triggers to the base Static BH 1Y

ENABLE_STATIC_BH_1Y_VOLUME_FILTER = True   # Volume confirmation filter
STATIC_BH_1Y_VOLUME_MIN_DAILY = 1_000_000  # Minimum $1M daily volume

ENABLE_STATIC_BH_1Y_SECTOR_ROTATION = True   # Sector rotation enhancement
STATIC_BH_1Y_SECTOR_MAX_PER_SECTOR = 3  # Maximum 3 stocks per sector

ENABLE_STATIC_BH_1Y_PERFORMANCE_THRESHOLD = True   # Performance threshold trigger
STATIC_BH_1Y_PERFORMANCE_MIN_IMPROVEMENT = 0.02  # 2% minimum improvement

ENABLE_STATIC_BH_1Y_MARKET_REGIME = True   # Market-regime based rebalancing
STATIC_BH_1Y_MARKET_REGIME_BASE_DAYS = 22  # Base rebalancing frequency
STATIC_BH_1Y_MARKET_REGIME_HIGH_VOL_DAYS = 15  # High volatility: rebalance every 15 days
STATIC_BH_1Y_MARKET_REGIME_LOW_VOL_DAYS = 30   # Low volatility: rebalance every 30 days
STATIC_BH_1Y_MARKET_REGIME_VOL_THRESHOLD = 0.20  # 20% volatility threshold

ENABLE_STATIC_BH_1Y_MOMENTUM_PERSIST = True   # Momentum persistence strategy
STATIC_BH_1Y_MOMENTUM_PERSIST_DAYS = 5  # Require top 10 to be stable for N consecutive days
STATIC_BH_1Y_MOMENTUM_PERSIST_MIN_OVERLAP = 0.8  # 80% overlap required for "stable"

ENABLE_STATIC_BH_1Y_OVERLAP = True   # Overlap-based rebalancing
STATIC_BH_1Y_OVERLAP_THRESHOLD = 0.7  # Rebalance when overlap drops below 70%

ENABLE_STATIC_BH_1Y_RANK_DRIFT = True   # Rank drift rebalancing
STATIC_BH_1Y_RANK_DRIFT_THRESHOLD = 3.0  # Rebalance when avg rank change > 3 positions

ENABLE_STATIC_BH_1Y_DRAWDOWN = True   # Portfolio drawdown trigger
STATIC_BH_1Y_DRAWDOWN_THRESHOLD = 0.05  # Rebalance when portfolio down 5% from peak

ENABLE_STATIC_BH_1Y_SMART_MONTHLY = True   # Smart monthly + conditional
STATIC_BH_1Y_SMART_MONTHLY_DRAWDOWN = 0.03  # Early rebalance if down 3% from peak

# --- Smart Rebalancing Strategies (based on Static BH 1Y) ---
# These strategies use the same stock selection as Static BH 1Y but different sell logic

# 1. Momentum-Based Sell: only sell when momentum declining
ENABLE_BH_1Y_MOM_SELL = True
BH_1Y_MOM_SELL_LOOKBACK_SHORT = 10  # Short momentum window (days)
BH_1Y_MOM_SELL_LOOKBACK_LONG = 20   # Long momentum window (days)

# 2. Relative Strength Ranking: sell when rank drops significantly
ENABLE_BH_1Y_RANK_SELL = True
BH_1Y_RANK_SELL_DROP_THRESHOLD = 5  # Sell if rank dropped by this many positions
BH_1Y_RANK_SELL_MAX_RANK = 15       # Sell if rank is now outside this threshold

# 3. Trailing Stop + Momentum: combined stops
ENABLE_BH_1Y_TRAILING_MOM = True
BH_1Y_TRAILING_MOM_SOFT_STOP = 0.10  # Soft stop: sell if down 10% AND momentum negative
BH_1Y_TRAILING_MOM_HARD_STOP = 0.15  # Hard stop: always sell if down 15% from peak

# 4. Volume Confirmation: sell only if volume confirms weakness
ENABLE_BH_1Y_VOLUME_CONFIRM = True
BH_1Y_VOLUME_CONFIRM_MULTIPLIER = 1.5  # Sell if volume > 1.5x average AND not in top 12

# 5. Sector Rotation Awareness: don't sell all stocks from a sector at once
ENABLE_BH_1Y_SECTOR_AWARE = True
BH_1Y_SECTOR_AWARE_MIN_PER_SECTOR = 1  # Keep at least 1 stock per sector if in top 20
BH_1Y_SECTOR_AWARE_MAX_RANK = 20       # Consider stocks in top 20 for sector protection

# 6. Accelerating Momentum Buy: prioritize stocks with accelerating momentum
ENABLE_BH_1Y_ACCEL_BUY = True
BH_1Y_ACCEL_BUY_WEIGHT = 0.3  # Weight for acceleration score (0-1)

# =====================================================================
# 10 NEW REBALANCING STRATEGIES (Based on Static BH 1Y stock selection)
# =====================================================================

# 1. Volatility-Adjusted Rebalancing: rebalance more often in high-vol periods
ENABLE_BH_1Y_VOL_ADJ_REBAL = True
BH_1Y_VOL_ADJ_REBAL_LOW_VOL_THRESH = 0.15   # Annualized vol below this = reduce rebalance freq
BH_1Y_VOL_ADJ_REBAL_HIGH_VOL_THRESH = 0.35  # Annualized vol above this = increase rebalance freq
BH_1Y_VOL_ADJ_REBAL_MIN_DAYS = 5            # Minimum days between rebalances
BH_1Y_VOL_ADJ_REBAL_MAX_DAYS = 30           # Maximum days between rebalances

# 1b. Volatility-Adjusted Rebalancing with 1M/3M Ratio ranking (instead of 1Y performance)
ENABLE_RATIO_1M3M_VOL_ADJ_REBAL = True      # Uses same vol thresholds as above

# 2. Correlation-Based Filtering: avoid holding highly correlated stocks
ENABLE_BH_1Y_CORR_FILTER = True
BH_1Y_CORR_FILTER_THRESH = 0.7              # Max correlation between any two holdings
BH_1Y_CORR_FILTER_LOOKBACK = 60             # Days for correlation calculation

# 3. Market Regime Detection: different rebalancing for bull/bear/sideways
ENABLE_BH_1Y_REGIME_AWARE = True
BH_1Y_REGIME_BULL_MA = 50                   # MA for bull detection
BH_1Y_REGIME_BEAR_MA = 200                  # MA for bear detection
BH_1Y_REGIME_REBAL_BULL = 30                # Rebalance every N days in bull
BH_1Y_REGIME_REBAL_BEAR = 10                # Rebalance every N days in bear (faster exit)
BH_1Y_REGIME_REBAL_SIDEWAYS = 20            # Rebalance every N days in sideways

# 4. Risk Parity Allocation: size by inverse volatility
ENABLE_BH_1Y_RISK_PARITY = True
BH_1Y_RISK_PARITY_VOL_LOOKBACK = 60         # Days for volatility calculation
BH_1Y_RISK_PARITY_MIN_WEIGHT = 0.02         # Minimum weight per position
BH_1Y_RISK_PARITY_MAX_WEIGHT = 0.25         # Maximum weight per position

# 5. Adaptive Drift Threshold: only rebalance when drift exceeds threshold
ENABLE_BH_1Y_DRIFT_THRESH = True
BH_1Y_DRIFT_THRESH_TARGET = 0.10            # Target weight per position (10%)
BH_1Y_DRIFT_THRESH_TRIGGER = 0.05           # Rebalance if any position drifts 5%+ from target

# 6. Momentum Quality Score: combine momentum strength + consistency
ENABLE_BH_1Y_MOM_QUALITY = True
BH_1Y_MOM_QUALITY_MOM_WEIGHT = 0.6          # Weight for momentum return
BH_1Y_MOM_QUALITY_CONS_WEIGHT = 0.4         # Weight for consistency (lower vol of returns)
BH_1Y_MOM_QUALITY_MIN_QUALITY = 0.3         # Minimum quality score to hold

# 7. Liquidity-Based Sizing: larger positions in more liquid stocks
ENABLE_BH_1Y_LIQUIDITY = True
BH_1Y_LIQUIDITY_AVG_VOL_DAYS = 20           # Days to average volume
BH_1Y_LIQUIDITY_MIN_DOLLAR_VOL = 1000000    # Minimum daily dollar volume
BH_1Y_LIQUIDITY_MAX_WEIGHT = 0.20           # Max weight in high-liquidity stocks

# 8. Earnings Avoidance: skip rebalancing near earnings
ENABLE_BH_1Y_EARNINGS_AVOID = True
BH_1Y_EARNINGS_AVOID_DAYS_BEFORE = 5        # Days to avoid before earnings
BH_1Y_EARNINGS_AVOID_DAYS_AFTER = 2         # Days to avoid after earnings

# 9. Multi-Factor Composite: blend momentum + value + quality
ENABLE_BH_1Y_MULTI_FACTOR = True
BH_1Y_MULTI_FACTOR_MOM_WEIGHT = 0.5         # Momentum weight
BH_1Y_MULTI_FACTOR_VAL_WEIGHT = 0.25        # Value weight (P/E inverse)
BH_1Y_MULTI_FACTOR_QUAL_WEIGHT = 0.25       # Quality weight (ROE)

# 10. Time-Decay Holdings: gradually reduce positions over time
ENABLE_BH_1Y_TIME_DECAY = True
BH_1Y_TIME_DECAY_EXIT_DAYS = 10             # Days to fully exit a position
BH_1Y_TIME_DECAY_EXIT_PCT_DAILY = 0.15      # Sell 15% of position per day when exiting



# --- Rebalance Horizon Optimization ---

# If True, test all rebalance horizons from 1 to 40 days for static strategies

# and report the best performing horizon in the final summary

OPTIMIZE_REBALANCE_HORIZON = True

REBALANCE_HORIZON_MIN = 1  # Minimum rebalance period to test

REBALANCE_HORIZON_MAX = 40  # Maximum rebalance period to test

# All horizons from MIN to MAX will be tested in parallel



# --- Dynamic Buy & Hold (BH) rebalancing guard ---

# Dynamic BH checks candidates daily, but only trades if:

#   expected_gain_from_price_diff - estimated_total_transaction_cost > 0

# This is computed from recent price performance (lookback depends on 1Y/3M/1M variant).



# --- Feature windows (for ML only)

FEAT_SMA_SHORT          = 5

FEAT_SMA_LONG           = 20

FEAT_VOL_WINDOW         = 10

# CLASS_HORIZON removed - use PERIOD_HORIZONS instead



# How many days of historical data to use when making predictions

# Must be >= 120 to have enough data after feature calculation (indicators need 50+ days lookback)

# Higher = more stable predictions, Lower = more reactive to recent changes

# Recommended: 120-180 days to ensure sufficient valid rows after feature engineering

PREDICTION_LOOKBACK_DAYS = 120



# --- UNIFIED REBALANCING FREQUENCY FOR ALL DYNAMIC STRATEGIES ---

# This controls rebalancing frequency for ALL dynamic strategies:

#   - Dynamic BH (1Y/3M/1M)

#   - AI Strategy

#   - Risk-Adj Mom, Mean Reversion, Quality+Mom, Vol-Adj Mom

#   - Momentum+AI Hybrid, Sector Rotation, 3M/1Y Ratio

# Note: Static BH strategies remain static (no rebalancing) as intended

# Set to:

#   1 = Daily rebalancing

#   7 = Weekly rebalancing

#   30 = Monthly rebalancing

AI_REBALANCE_FREQUENCY_DAYS = 1  # Rebalancing frequency for all dynamic strategies



# --- AI Strategy (3-stock daily selection) Rebalancing Threshold (REMOVED)

# AI Strategy now uses transaction cost guard like other strategies (Dynamic BH, Risk-Adj Mom, etc.)

# No arbitrary annual thresholds - rebalances based on portfolio value growth since last rebalance



# --- Momentum + AI Hybrid Strategy Parameters ---

# MOMENTUM_AI_HYBRID_TOP_N removed - now uses PORTFOLIO_SIZE

MOMENTUM_AI_HYBRID_PORTFOLIO_SIZE = PORTFOLIO_SIZE  # Use same portfolio size as other strategies

MOMENTUM_AI_HYBRID_BUY_THRESHOLD = 0.02  # Buy if AI predicts >2% return

MOMENTUM_AI_HYBRID_SELL_THRESHOLD = -0.01  # Sell if AI predicts <-1% return

# Note: Uses AI_REBALANCE_FREQUENCY_DAYS for rebalancing frequency

MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK = 90  # 3-month momentum for stock ranking

MOMENTUM_AI_HYBRID_STOP_LOSS = 0.10  # 10% stop loss from entry

MOMENTUM_AI_HYBRID_TRAILING_STOP = 0.08  # 8% trailing stop once in profit



# --- Momentum-Volatility Hybrid Strategy Parameters ---

MOMENTUM_VOLATILITY_HYBRID_BUY_THRESHOLD = 0.02  # Buy if predicted return > 2%

MOMENTUM_VOLATILITY_HYBRID_SELL_THRESHOLD = -0.01  # Sell if predicted return < -1%

MOMENTUM_VOLATILITY_HYBRID_STOP_LOSS = 0.10  # 10% stop loss

MOMENTUM_VOLATILITY_HYBRID_TRAILING_STOP = 0.08  # 8% trailing stop



# --- Price Curvature Strategy Parameters ---

PRICE_CURVATURE_LOOKBACK_DAYS = 60  # Fit curvature over roughly one quarter of trading days

PRICE_CURVATURE_SG_WINDOW = 15  # SG smoothing window for local trend/acceleration ranking

# --- Volatility-Adjusted Momentum Strategy Parameters ---

VOLATILITY_ADJ_MOM_LOOKBACK = 90  # 90-day momentum lookback

VOLATILITY_ADJ_MOM_VOL_WINDOW = 20  # 20-day volatility window

VOLATILITY_ADJ_MOM_MIN_SCORE = 0.5  # Minimum volatility-adjusted score threshold



# REGRESSION MODE: Using regression models that predict actual returns

# TARGET_PERCENTAGE removed - not used in regression mode (models predict exact returns)

# USE_MODEL_GATE removed - using simplified buy-and-hold logic

USE_MARKET_FILTER       = False      # market filter removed as per user request

MARKET_FILTER_TICKER    = 'SPY'

MARKET_FILTER_SMA       = 200

USE_PERFORMANCE_BENCHMARK = False  # Disable strict benchmark filtering for small universes



# --- ML Model Selection Flags ---

USE_LOGISTIC_REGRESSION = False      # Not needed - too simple

USE_SVM                 = False      # SVR slower than XGBoost, usually worse

USE_MLP_CLASSIFIER      = False      # Less effective than LSTM/TCN for time series

USE_LIGHTGBM            = True       # ENABLED - Best gradient boosting

USE_XGBOOST             = True       # KEEP - Best traditional ML

USE_LSTM                = True       # KEEP - Best deep learning for sequences

USE_GRU                 = True        # Enabled - GRU is faster and sometimes better than LSTM

USE_RANDOM_FOREST       = True       # KEEP - Good ensemble baseline

USE_TCN                 = True       # KEEP - Fast temporal model

USE_ELASTIC_NET         = False      # Too simple - linear models don't capture patterns

USE_RIDGE               = False      # Too simple - linear models don't capture patterns



# --- ML Training Settings ---

USE_ALPHA_WEIGHTS       = True       # Use alpha-based sample weights for training



# Simple Rule-Based Strategy removed - using AI strategy only



# --- Deep Learning specific hyperparameters

SEQUENCE_LENGTH         = 60          # Reduced from 120 - less noise, faster training

LSTM_HIDDEN_SIZE        = 64          # Reduced from 128 - prevent overfitting

LSTM_NUM_LAYERS         = 2           # Reduced from 3 - simpler model generalizes better

LSTM_DROPOUT            = 0.3         # Keep dropout for regularization

LSTM_EPOCHS             = 50          # Reduced from 100 - faster, less overfit

LSTM_BATCH_SIZE         = 32          # Keep batch size

LSTM_LEARNING_RATE      = 0.001       # Increased from 0.0005 - faster convergence



# --- GRU Hyperparameter Search Ranges ---

GRU_HIDDEN_SIZE_OPTIONS = [32, 64]         # Simplified for small datasets

GRU_NUM_LAYERS_OPTIONS  = [1, 2]           # Shallow networks for small data

GRU_DROPOUT_OPTIONS     = [0.1, 0.2, 0.3]  # Stable range

GRU_LEARNING_RATE_OPTIONS = [0.0005, 0.001, 0.002]  # Slightly tighter top-end

GRU_BATCH_SIZE_OPTIONS  = [64, 128]        # Practical batch sizes for GPU

GRU_EPOCHS_OPTIONS      = [50, 70, 90]     # Allow a bit more training

GRU_CLASS_HORIZON_OPTIONS = [10, 20, 40, 60]  # Match period horizons for optimization

GRU_TARGET_PERCENTAGE_OPTIONS = [0.005, 0.006, 0.007]  # Narrow range for short-term moves

ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = False  # Enable hyperparameter search for new features



# --- Misc

INITIAL_BALANCE         = 100_000.0

SAVE_PLOTS              = False     # Disable SHAP (causes errors with XGBoost Regressor)

FORCE_TRAINING          = False

CONTINUE_TRAINING_FROM_EXISTING = False  # Force fresh training to avoid loading corrupted PyTorch models

# Threshold optimization removed - system uses simplified buy-and-hold



# --- Live Trading Model Selection ---

# Which period's model to use for live trading

# Options: "Best" (auto-select highest performer), "1-Year"

LIVE_TRADING_MODEL_PERIOD = "Best"



# --- Regression-Based Return Prediction ---

# Regression is now the default and only approach - removed USE_REGRESSION_MODEL flag



# Period-specific horizons (calendar days) - matched to period scale

PERIOD_HORIZONS = {

    # Prediction horizon in calendar days

    "1-Year": 10      # 10-day prediction horizon - aligned with retraining frequency

}



USE_SINGLE_REGRESSION_MODEL = True  # Use single regression model instead of buy/sell pair

POSITION_SCALING_BY_CONFIDENCE = True  # Scale position size by predicted return magnitude





# Architecture options

TRY_LSTM_INSTEAD_OF_GRU = False  # Set to True to try LSTM instead of GRU for comparison
