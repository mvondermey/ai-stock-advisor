from operator import truediv
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

def get_data_lookback_days():
    """Get the number of days to look back for data downloads based on DATA_INTERVAL."""
    MAX_LOOKBACK_DAYS = 730  # 2 years of historical data
    if DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:
        return 729  # Stay within Yahoo's 730-day limit for intraday
    else:
        return MAX_LOOKBACK_DAYS + 60  # Add buffer for weekends/holidays
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

# If ALPACA_STOCKS is enabled, Alpaca can return thousands of symbols.
# Set very high to effectively include (almost) all Alpaca-tradable US equities.
ALPACA_STOCKS_LIMIT = 20000  # High limit = train models for ALL tradable stocks

# Exchange filter for Alpaca asset list. Use ["NASDAQ"] to restrict to NASDAQ only.
ALPACA_STOCKS_EXCHANGES = []  # NASDAQ only
N_TOP_TICKERS           = 2000     # Reduced from 2000 - balance quality and quantity
TOP_TICKER_SELECTION_LOOKBACK = "1Y"     # Try 1-month for more responsive selection
BATCH_DOWNLOAD_SIZE     = 10000     # Download in batches of 1000
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
from multiprocessing import cpu_count
# Conservative CPU utilization to prevent system overload and process termination
# Use only half of CPU cores to avoid memory exhaustion with large datasets
NUM_PROCESSES           = max(1, cpu_count() -4)

# Parallel processing threshold - only use parallel processing for ticker lists larger than this
# Below this threshold, sequential processing is faster due to lower overhead
PARALLEL_THRESHOLD     = 200      # Use parallel only for >200 tickers

# Enable parallel strategy execution within each day (experimental)
# This runs strategy selections in parallel to improve CPU usage
ENABLE_PARALLEL_STRATEGIES = True   # Run strategies in parallel within each day

# Training batch size for parallel processing (how many tasks per batch)
TRAINING_BATCH_SIZE     = 100       # Smaller batches for better progress tracking

# --- Dynamic GPU Slot Allocation ---
# Estimated VRAM requirements per model (in GB)
GPU_MEMORY_PER_MODEL = {
    'LSTM': 1.5,   # GB per LSTM/TCN/GRU model
    'XGBoost': 0.5 # GB per XGBoost model
}

def auto_configure_gpu_slots():
    try:
        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            # Only print in main process, not in spawned workers
            import multiprocessing as _mp
            if _mp.current_process().name == 'MainProcess':
                print(f" Detected {total_vram:.1f}GB VRAM - configuring slots dynamically")
            
            # Calculate max slots for each model type (leave 20% VRAM headroom)
            lstm_slots = max(1, int(total_vram * 0.8 / GPU_MEMORY_PER_MODEL['LSTM']))
            xgb_slots = max(1, int(total_vram * 0.8 / GPU_MEMORY_PER_MODEL['XGBoost']))
            return {'LSTM': lstm_slots, 'XGBoost': xgb_slots}
    except ImportError:
        pass
    return {'LSTM': 0, 'XGBoost': 0}  # Fallback to CPU

GPU_MODEL_SLOTS = auto_configure_gpu_slots()

# Limit GPU memory per training worker process (PyTorch)
GPU_PER_PROCESS_MEMORY_FRACTION = 1  # Auto: 0.95/4 = 23.75% per worker with 4 concurrent

# --- GPU memory cache behavior (PyTorch) ---
GPU_CLEAR_CACHE_ON_WORKER_INIT = False
GPU_CLEAR_CACHE_AFTER_EACH_TICKER = False

# --- GPU Concurrency Control for PyTorch ---
# When using multiprocessing with PyTorch on GPU, too many concurrent trainers can cause OOM.
# This limits how many worker processes can run PyTorch models on GPU simultaneously.
# Only applies when PYTORCH_USE_GPU = True (PyTorch uses GPU)
# Does NOT apply to XGBoost GPU (XGBoost manages its own GPU memory)
GPU_MAX_CONCURRENT_TRAINING_WORKERS = max(4, GPU_MODEL_SLOTS['LSTM'])  # Dynamic: ~5 for 8GB VRAM RTX 5060

# Multiprocessing stability: recycle worker processes periodically to avoid RAM creep / leaked semaphores
# when training many tickers under WSL + spawn.
# - Set to 1 for max stability (one ticker per worker process).
# - Set to None to disable recycling (faster, but may accumulate memory/semaphores).
TRAINING_POOL_MAXTASKSPERCHILD = 5  # Moderate recycling for better memory utilization

# Per-ticker training timeout (seconds). If a ticker takes longer, it will be skipped.
# - Set to 600 (10 min) for normal use (handles slow XGBoost GridSearchCV)
# - Set to 1800 (30 min) for very large datasets or complex models
# - Set to None to disable timeout (not recommended - can hang forever)
PER_TICKER_TIMEOUT = 60  # 60 seconds per ticker training task

# Per-ticker prediction timeout (seconds). If a prediction takes longer, it will be skipped.
# - Set to 30 for normal use (predictions should be fast)
# - Set to None to disable timeout (not recommended - can hang forever)
PREDICTION_TIMEOUT = 30  # 30 seconds max per ticker prediction

# Training worker process count (separate from global NUM_PROCESSES).
# For 5000 tickers, use parallel training. Models are saved to disk and loaded back (no pickling overhead).
#
# Worker count strategy:
# - CPU only: Use half of CPU cores (conservative to prevent memory exhaustion)
# - PyTorch CPU + XGBoost GPU: Use half CPU cores (GPU handles XGBoost, CPU handles PyTorch)
# - PyTorch GPU + XGBoost GPU: Very limited (GPU bottleneck)
# - PyTorch GPU + XGBoost CPU: Use quarter CPU for XGBoost while GPU handles PyTorch
#

TRAINING_NUM_PROCESSES = max(1, cpu_count() - 4)  # Use more CPU cores

# --- Unified Parallel Training System ---
# Enable the new parallel training system that trains models by model-type instead of by ticker.
# Benefits:
#   - Better GPU utilization (GPU models train while CPU models train in parallel)
#   - Faster overall training time (~18x speedup for large universes)
#   - More granular progress tracking
# Set to False to use the legacy sequential training system (train all models for one ticker, then move to next)
# ENABLED: Trains all models in parallel by model type for maximum efficiency
USE_UNIFIED_PARALLEL_TRAINING = True

# --- Backtest windows
BACKTEST_DAYS           =   2   # Backtest period in calendar days (~63=3mo, ~180=6mo, ~365=1yr)
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
# 'volatility_ensemble'    = 🏆 Vol Ens - Volatility-adjusted position sizing (+106% in backtest)
# 'ai_volatility_ensemble' = 🤖 AI Vol Ens - AI-enhanced volatility ensemble (NEW)
# 'correlation_ensemble'   = 🏆 Corr Ens - Correlation-filtered diversification (+106% in backtest)
# 'dynamic_bh_1y'          = Dynamic BH 1Y - Annual rebalancing
# 'dynamic_bh_3m'          = Dynamic BH 3M - Quarterly rebalancing
# 'risk_adj_mom'           = Risk-Adjusted Momentum
# 'quality_momentum'       = Quality + Momentum
# 'adaptive_ensemble'      = Adaptive Meta-Ensemble
# 'dynamic_pool'           = Dynamic Strategy Pool
# 'sentiment_ensemble'     = Sentiment-Enhanced Ensemble
LIVE_TRADING_STRATEGY    = 'volatility_ensemble'  # 🏆 Best performer from backtest

# --- Walk-Forward Retraining Frequency ---
# How often to retrain models during walk-forward backtest
# Options:
#   5  = Weekly retraining (aggressive, best for high volatility/penny stocks)
#   10 = Bi-weekly retraining (balanced, recommended for volatile stocks)
#   20 = Monthly retraining (conservative, recommended for S&P 500 / stable large-caps)
#   60 = Quarterly retraining (rare, only for very stable/long-term strategies)
RETRAIN_FREQUENCY_DAYS = 10  # Retrain every 10 days - aligned with prediction horizon
ENABLE_WALK_FORWARD_RETRAINING = False   # Set to False to use only saved models, no retraining

# --- Backtest Period Enable/Disable Flags ---
ENABLE_1YEAR_BACKTEST   = True   # Enabled - For simulation and strategy validation

# --- Training Period Enable/Disable Flags ---
# ENABLE_1YEAR_TRAINING removed - models are now trained during walk-forward backtest

# --- Portfolio Strategy Enable/Disable Flags ---
# Set to False to disable specific portfolios in the backtest
ENABLE_MOMENTUM_AI_HYBRID = True  # ENABLED - Momentum+AI Hybrid
ENABLE_MULTITASK_LEARNING = False # DISABLED - Multi-task learning
ENABLE_AI_VOLATILITY_ENSEMBLE = False  # DISABLED - AI Volatility Ensemble

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
ENABLE_MOMENTUM_VOLATILITY_HYBRID = True   # ENABLED - Momentum-Volatility Hybrid Strategy
ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M = True   # ENABLED - Momentum-Volatility Hybrid 6M Strategy (6-month lookback)
ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y = True   # ENABLED - Momentum-Volatility Hybrid 1Y Strategy (1-year lookback)
ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M = True   # ENABLED - Momentum-Volatility Hybrid 1Y/3M Ratio Strategy (strong 1Y, weak 3M)
ENABLE_ADAPTIVE_STRATEGY = True   # ENABLED - Adaptive Meta-Ensemble Strategy
ENABLE_VOLATILITY_ENSEMBLE = True   # ENABLED - Volatility-Adjusted Ensemble Strategy (risk-managed position sizing)
ENABLE_ENHANCED_VOLATILITY = True   # ENABLED - Enhanced Volatility Trader (ATR stops + take profits)
ENABLE_CORRELATION_ENSEMBLE = True   # ENABLED - Correlation-Filtered Ensemble Strategy (diversification-focused)
ENABLE_RISK_ADJ_MOM_SENTIMENT = True   # NEW - Risk-Adjusted Momentum + Sentiment Strategy
ENABLE_DYNAMIC_POOL = True   # ENABLED - Dynamic Strategy Pool Strategy

# --- Multi-Timeframe Ensemble Strategy ---
# Combines signals from different timeframes for better entry/exit timing
ENABLE_MULTI_TIMEFRAME_ENSEMBLE = True   # NEW - Multi-Timeframe Ensemble Strategy

# Additional strategy enable flags
ENABLE_TURNAROUND = True   # ENABLED - Turnaround Strategy (buy depressed stocks)
ENABLE_MOMENTUM_VOLATILITY_HYBRID = True   # ENABLED - Momentum-Volatility Hybrid Strategy
ENABLE_3M_1Y_RATIO = True   # ENABLED - 3M/1Y Ratio Strategy
ENABLE_PRICE_ACCELERATION = True   # ENABLED - Price Acceleration Strategy (physics-based velocity/acceleration)
ENABLE_VOTING_ENSEMBLE = False   # DISABLED - Voting Ensemble Strategy (not performing well)

# --- New Advanced Strategies ---
ENABLE_MOMENTUM_ACCELERATION = True   # NEW - Momentum Acceleration (3M momentum + acceleration filter)
ENABLE_CONCENTRATED_3M = True   # NEW - Concentrated 3M + Vol Filter (fewer positions, volatility filtered)
ENABLE_DUAL_MOMENTUM = True   # NEW - Dual Momentum (absolute + relative momentum)
ENABLE_TREND_FOLLOWING_ATR = True   # NEW - Trend Following with ATR Trailing Stop
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
ENABLE_RISK_ADJ_MOM_1M = True   # NEW - Risk-Adj Mom 1M (same as Risk-Adj Mom but 1M window)
ENABLE_RISK_ADJ_MOM_1M_MONTHLY = True   # NEW - Risk-Adj Mom 1M Monthly (same scoring, rebalance start of month only)
ENABLE_RISK_ADJ_MOM_1M_VOL_SWEET = True   # NEW - Risk-Adj Mom 1M + Vol-Sweet (best Sharpe + vol filter)
ENABLE_AI_ELITE = True   # NEW - AI Elite (ML-powered scoring of momentum + dip opportunities)
ENABLE_AI_ELITE_MONTHLY = True   # NEW - AI Elite Monthly (same ML scoring, retrain + rebalance start of month only)
ENABLE_AI_ELITE_FILTERED = True   # NEW - AI Elite Filtered (Risk-Adj Mom 3M pre-filter + AI Elite re-rank)
ENABLE_AI_REGIME = True   # NEW - AI Regime (ML predicts which strategy to use based on market conditions)
ENABLE_AI_REGIME_MONTHLY = True   # NEW - AI Regime Monthly (same as AI Regime but rebalance start of month only)
ENABLE_UNIVERSAL_MODEL = True   # NEW - Universal Model (single ML model for all tickers)
ENABLE_LLM_STRATEGY = False   # DISABLED - LLM Strategy (not implemented)

# AI Elite Parameters
AI_ELITE_RETRAIN_DAYS = 1  # Retrain model every 1 days
AI_ELITE_TRAINING_LOOKBACK = 90  # Days of history to use for training
AI_ELITE_FORWARD_DAYS = 5  # Predict performance over next N days
AI_ELITE_FORCE_FRESH_TRAIN = False  # False = load existing model and do incremental training; True = always fresh train

# AI Elite Intraday Configuration
AI_ELITE_INTRADAY_INTERVAL = "1h"  # Use 1-hour data for intraday features
AI_ELITE_INTRADAY_LOOKBACK = 10  # Days of hourly data to use (240 data points per stock)

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

MULTI_TIMEFRAMES = ["1d", "4h", "1h"]    # Timeframes to analyze
MULTI_TIMEFRAME_LOOKBACK = {
    "1d": 365,    # Daily: 1 year lookback
    "4h": 30,     # 4-hour: 30 days lookback  
    "1h": 7       # 1-hour: 7 days lookback
}
MULTI_TIMEFRAME_WEIGHTS = {
    "1d": 0.6,    # Daily gets 60% weight (trend direction)
    "4h": 0.3,    # 4-hour gets 30% weight (momentum confirmation)
    "1h": 0.1     # 1-hour gets 10% weight (timing)
}
# Require consensus from at least this many timeframes
MULTI_TIMEFRAME_MIN_CONSENSUS = 2  # At least 2 timeframes must agree

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 10
STRAT_SMA_LONG          = 20
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 2.0
ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing
PORTFOLIO_SIZE          = 10        # Number of stocks to hold in portfolio
PORTFOLIO_BUFFER_SIZE    = 12        # Sell when stock not in top X (buffer for stability)
TOTAL_CAPITAL           = 300000     # Total capital to invest ($300,000)
INVESTMENT_PER_STOCK    = TOTAL_CAPITAL / PORTFOLIO_SIZE  # Automatically calculated
TRANSACTION_COST        = 0.011      # 1.1% per trade leg (buy or sell)
ENABLE_STOP_LOSS        = False      # Global enable/disable stop loss protection (overridden by per-strategy settings)
STOP_LOSS_PCT           = 0.05       # Default stop loss percentage (5%)
ENABLE_PROFIT_GUARD     = False      # Sell stocks when not in top 10 (no profit/loss consideration)

# --- Inverse ETF Hedge Strategy ---
# Instead of stop losses, add inverse ETFs during market downturns
# NEW: Hybrid approach - gradual scaling based on market stress
ENABLE_INVERSE_ETF_HEDGE = True     # Add inverse ETFs when market crashes
# Gradual scaling thresholds (market decline % -> hedge allocation %)
# Example: market down 5% -> 20% hedge, down 10% -> 50% hedge, down 15% -> 80% hedge
INVERSE_ETF_HEDGE_THRESHOLD_LOW = 0.05    # 5% decline = 20% hedge
INVERSE_ETF_HEDGE_THRESHOLD_MED = 0.10      # 10% decline = 50% hedge  
INVERSE_ETF_HEDGE_THRESHOLD_HIGH = 0.15     # 15% decline = 80% hedge
INVERSE_ETF_HEDGE_BASE_ALLOCATION = 0.20    # Always keep 20% in equity (never 100% hedge)
INVERSE_ETF_HEDGE_MAX_ALLOCATION = 0.80     # Max 80% in hedge (never 100%)
INVERSE_ETF_HEDGE_PREFERENCE = ['SOXS', 'SQQQ', 'SPXU', 'FAZ', 'SH', 'PSQ']  # Preferred hedge ETFs
INVERSE_ETF_HEDGE_MIN_HOLD_DAYS = 5         # Minimum days to hold hedge before exiting (prevents whipsaw)

# --- Analyst Recommendation Strategy ---
ENABLE_ANALYST_RECOMMENDATION = True        # Enable analyst recommendation strategy
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
PRICE_ACCELERATION_STOP_LOSS = 0.05    # +2.4% improvement (+2.1% vs -0.3%)
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
CORRELATION_ENSEMBLE_STOP_LOSS = 0.0   # -0.9% worse with stop (+18.9% vs +18.0%)
DYNAMIC_POOL_STOP_LOSS = 0.0           # -0.8% worse with stop (+17.6% vs +16.8%)
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
    'Price Acceleration': PRICE_ACCELERATION_STOP_LOSS,
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
    'Correlation Ensemble': CORRELATION_ENSEMBLE_STOP_LOSS,
    'Dynamic Pool': DYNAMIC_POOL_STOP_LOSS,
    'Enhanced Volatility': ENHANCED_VOLATILITY_STOP_LOSS,
    'Buy & Hold': BUY_HOLD_STOP_LOSS,
    '1Y/3M Ratio': 0.0,            # -0.2% worse (+7.6% vs +7.4%)
}

# --- Strategy Performance Filters ---
# Minimum performance thresholds for stock selection
# These can be used to filter out underperformers before strategy scoring
MIN_PERFORMANCE_1Y = 0.10      # 10% minimum 1-year performance
MIN_PERFORMANCE_6M = 0.05      # 5% minimum 6-month performance (10%/2)
MIN_PERFORMANCE_3M = 0.025     # 2.5% minimum 3-month performance (10%/4)
ENABLE_PERFORMANCE_FILTERS = True   # Set to True to enable these filters

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

# --- Static BH Monthly Rebalance Variants ---
# These are separate strategies that rebalance on the first trading day of each month
ENABLE_STATIC_BH_1Y_MONTHLY = True   # Static BH 1Y with monthly rebalance
ENABLE_STATIC_BH_6M_MONTHLY = True   # Static BH 6M with monthly rebalance
ENABLE_STATIC_BH_3M_MONTHLY = True   # Static BH 3M with monthly rebalance
ENABLE_STATIC_BH_1M_MONTHLY = True   # Static BH 1M with monthly rebalance

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
