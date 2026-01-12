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
DATA_PROVIDER           = 'alpaca'    # 'stooq', 'yahoo', 'alpaca', or 'twelvedata'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_INTERVAL           = '1d'       # '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
VALID_TICKERS_CACHE_PATH = Path("logs/valid_tickers.json")
CACHE_DAYS              = 7

# Master switch: enable price caching to reduce downloads (uses `data_cache/` CSVs)
ENABLE_PRICE_CACHE       = True

# Alpaca API credentials
ALPACA_API_KEY          = "PK3FDQLRMEVFOAOU7VHD3Z6THE"
ALPACA_SECRET_KEY       = "8By7ituNTmspLsWc191hQfviD3xaNNdd2opB8tJAfmK6"

# TwelveData API credentials
TWELVEDATA_API_KEY      = "aed912386d7c47939ebc28a86a96a021"
TWELVEDATA_MAX_WORKERS  = 5  # Max parallel API requests (free tier: 800 credits/day)

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

# --- Universe / selection
MARKET_SELECTION = {
    "ALPACA_STOCKS": False,    # DISABLED - Using curated indices instead
    "NASDAQ_ALL": False,
    "NASDAQ_100": True,        # ENABLED - ~100 tech stocks for growth
    "SP500": True,             # ~500 stocks  
    "DOW_JONES": True,         # ~30 stocks
    "POPULAR_ETFS": True,      # ENABLED - Include popular ETFs (includes sector ETFs)
    "CRYPTO": False,
    "DAX": True,
    "MDAX": True,
    "SMI": True,
    "FTSE_MIB": False,        # DISABLED - focus on core markets
}

# If ALPACA_STOCKS is enabled, Alpaca can return thousands of symbols.
# Set very high to effectively include (almost) all Alpaca-tradable US equities.
ALPACA_STOCKS_LIMIT = 20000  # High limit = train models for ALL tradable stocks

# Exchange filter for Alpaca asset list. Use ["NASDAQ"] to restrict to NASDAQ only.
ALPACA_STOCKS_EXCHANGES = []  # NASDAQ only
N_TOP_TICKERS           = 199     # Reduced from 2000 - balance quality and quantity
TOP_TICKER_SELECTION_LOOKBACK = "1Y"     # Try 1-month for more responsive selection
BATCH_DOWNLOAD_SIZE     = 10000     # Download in batches of 1000
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
from multiprocessing import cpu_count
# Limit to 10 processes when using GPU to avoid GPU memory exhaustion
# PyTorch models (LSTM/GRU/TCN) load on GPU, and too many parallel processes cause OOM kills
# Use all but 5 CPU cores (keep some headroom for OS, data fetch, and GPU driver overhead)
NUM_PROCESSES           = max(1, cpu_count() - 5)

# Parallel processing threshold - only use parallel processing for ticker lists larger than this
# Below this threshold, sequential processing is faster due to lower overhead
PARALLEL_THRESHOLD     = 200      # Use parallel only for >200 tickers

# Training batch size for parallel processing (how many tasks per batch)
TRAINING_BATCH_SIZE     = 10000      # Smaller batches for better timeout handling

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
GPU_MAX_CONCURRENT_TRAINING_WORKERS = GPU_MODEL_SLOTS['LSTM'] # Use dynamic calculation

# Multiprocessing stability: recycle worker processes periodically to avoid RAM creep / leaked semaphores
# when training many tickers under WSL + spawn.
# - Set to 1 for max stability (one ticker per worker process).
# - Set to None to disable recycling (faster, but may accumulate memory/semaphores).
TRAINING_POOL_MAXTASKSPERCHILD = 1  # Enable recycling for maximum stability

# Per-ticker training timeout (seconds). If a ticker takes longer, it will be skipped.
# - Set to 600 (10 min) for normal use (handles slow XGBoost GridSearchCV)
# - Set to 1800 (30 min) for very large datasets or complex models
# - Set to None to disable timeout (not recommended - can hang forever)
PER_TICKER_TIMEOUT = 30  # 30 seconds max per ticker (very aggressive for stuck tickers)

# Training worker process count (separate from global NUM_PROCESSES).
# For 5000 tickers, use parallel training. Models are saved to disk and loaded back (no pickling overhead).
#
# Worker count strategy:
# - All CPU (PYTORCH_USE_GPU=False, XGBOOST_USE_GPU=False): 15 workers
# - PyTorch CPU + XGBoost GPU (PYTORCH_USE_GPU=False, XGBOOST_USE_GPU=True): 15 workers ← YOUR CURRENT SETUP
# - PyTorch GPU + XGBoost CPU (PYTORCH_USE_GPU=True, XGBOOST_USE_GPU=False): 3 workers
# - PyTorch GPU + XGBoost GPU (PYTORCH_USE_GPU=True, XGBOOST_USE_GPU=True): 3 workers (may cause OOM)
#
if not PYTORCH_USE_GPU and not XGBOOST_USE_GPU:
    TRAINING_NUM_PROCESSES = NUM_PROCESSES  # Full CPU, no GPU bottleneck
elif not PYTORCH_USE_GPU and XGBOOST_USE_GPU:
    TRAINING_NUM_PROCESSES = NUM_PROCESSES  # PyTorch on CPU, XGBoost on GPU (current setup)
else:
    TRAINING_NUM_PROCESSES = GPU_MODEL_SLOTS['LSTM']  # PyTorch on GPU (limited by VRAM)

# --- Unified Parallel Training System ---
# Enable the new parallel training system that trains models by model-type instead of by ticker.
# Benefits:
#   - Better GPU utilization (GPU models train while CPU models train in parallel)
#   - Faster overall training time (~18x speedup for large universes)
#   - More granular progress tracking
# Set to False to use the legacy sequential training system (train all models for one ticker, then move to next)
# ENABLED: Trains all models in parallel by model type for maximum efficiency
USE_UNIFIED_PARALLEL_TRAINING = True

# --- Backtest & training windows
BACKTEST_DAYS           = 500         # Backtest period in trading days (~60=2mo, ~125=6mo, ~250=1yr)
TRAIN_LOOKBACK_DAYS     = 365        # Train on ~1 year of history (user request)
VALIDATION_DAYS         = 90         # FIX 4: Validation period for threshold optimization

# --- Live Trading
LIVE_TRADING_ENABLED     = False       # ⚠️ Set to True to execute real orders (start with False for dry-run)
MODEL_MAX_AGE_DAYS        = 1           # Only use models trained in last X days
USE_PAPER_TRADING        = True        # True = paper trading (fake money), False = REAL MONEY ⚠️
TOP_N_STOCKS             = N_TOP_TICKERS  # Number of stocks to hold (from config)

# --- Walk-Forward Retraining Frequency ---
# How often to retrain models during walk-forward backtest
# Options:
#   5  = Weekly retraining (aggressive, best for high volatility/penny stocks)
#   10 = Bi-weekly retraining (balanced, recommended for volatile stocks)
#   20 = Monthly retraining (conservative, recommended for S&P 500 / stable large-caps)
#   60 = Quarterly retraining (rare, only for very stable/long-term strategies)
RETRAIN_FREQUENCY_DAYS = 300  # Train only once per year (effectively once for 1-year backtest)

# --- Backtest Period Enable/Disable Flags ---
ENABLE_1YEAR_BACKTEST   = True   # Enabled - For simulation and strategy validation

# --- Training Period Enable/Disable Flags ---
ENABLE_1YEAR_TRAINING   = True  # ENABLED - Train models for AI Strategy and individual ticker predictions

# --- Portfolio Stratebgy Enable/Disable Flags ---
# Set to False to disable specific portfolios in the backtest
# AI Portfolio + traditional strategies (no AI Strategy or AI Hybrid)
ENABLE_AI_STRATEGY      = True   # ENABLED - Use existing saved models (no new training)
ENABLE_AI_PORTFOLIO     = False   # ENABLED - AI Portfolio meta-learning
ENABLE_STATIC_BH        = True   # ENABLED - Static Buy & Hold benchmark
ENABLE_DYNAMIC_BH_1Y    = True   # ENABLED - Dynamic BH 1-year
ENABLE_DYNAMIC_BH_3M    = True   # ENABLED - Dynamic BH 3-month
ENABLE_DYNAMIC_BH_1M    = True   # ENABLED - Dynamic BH 1-month
ENABLE_RISK_ADJ_MOM     = True   # ENABLED - Risk-Adjusted Momentum
ENABLE_MEAN_REVERSION   = True   # ENABLED - Mean Reversion
ENABLE_SEASONAL         = True   # ENABLED - Seasonal strategy
ENABLE_QUALITY_MOM      = True   # ENABLED - Quality + Momentum
ENABLE_MOMENTUM_AI_HYBRID = True   # ENABLED - Use existing saved models
ENABLE_VOLATILITY_ADJ_MOM = True  # ENABLED - Volatility-Adjusted Momentum strategy
ENABLE_DYNAMIC_BH_1Y_VOL_FILTER = True  # NEW - Dynamic BH 1Y with Volatility Filter
ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP = True   # ENABLED - Dynamic BH 1Y with trailing stop
ENABLE_SECTOR_ROTATION = True   # NEW - Sector Rotation Strategy
ENABLE_MULTITASK_LEARNING = False   # NEW - Multi-Task Learning Strategy
ENABLE_3M_1Y_RATIO = True   # NEW - 3M/1Y Ratio Strategy

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 10
STRAT_SMA_LONG          = 20
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 2.0
ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing
PORTFOLIO_SIZE          = 10        # Number of stocks to hold in portfolio
TOTAL_CAPITAL           = 100000     # Total capital to invest ($100,000)
INVESTMENT_PER_STOCK    = TOTAL_CAPITAL / PORTFOLIO_SIZE  # Automatically calculated
TRANSACTION_COST        = 0.01       # 1% per trade leg (buy or sell)

# --- Dynamic BH 1Y + Volatility Filter Parameters ---
# Maximum allowed annualized volatility for stock selection (as percentage)
# Higher values allow more volatile stocks, lower values are more conservative
DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY = 120.0  # Maximum 120% annualized volatility (allow most stocks)

# --- Dynamic BH 1Y + Trailing Stop Parameters ---
# Trailing stop to protect gains and limit downside
DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT = 20.0  # Sell if price drops 20% from peak
DYNAMIC_BH_1Y_TRAILING_STOP_REBALANCE_DAYS = 1  # Check daily for stop triggers

# --- Sector Rotation Strategy Parameters ---
# PROPOSAL 2: Rotate between sector ETFs based on momentum
SECTOR_ROTATION_TOP_N = 5  # Number of top sectors to hold
SECTOR_ROTATION_REBALANCE_DAYS = 21  # Monthly rebalancing (21 trading days)
SECTOR_ROTATION_MOMENTUM_WINDOW = 60  # 60-day momentum for sector selection
SECTOR_ROTATION_MIN_MOMENTUM = 0.0  # TEMPORARILY reduced to 0% for debugging - was 5.0%

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
RISK_ADJ_MOM_MIN_SCORE = 30.0  # REVERTED - too low was hurting performance

# --- Static Buy & Hold (BH) rebalancing period ---
# How often to rebalance Static BH portfolios (in trading days)
# Set to 0 or None to disable rebalancing (buy once at start, hold until end)
# Recommended values: 20 (monthly), 60 (quarterly), 0 (no rebalancing)
STATIC_BH_1Y_REBALANCE_DAYS = 0   # Static BH 1Y: rebalance every N days (0 = no rebalancing)
STATIC_BH_3M_REBALANCE_DAYS = 0   # Static BH 3M: rebalance every N days (0 = no rebalancing)

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
# Recommended: 180-250 days to ensure sufficient valid rows after feature engineering
PREDICTION_LOOKBACK_DAYS = 252

# --- AI Portfolio Rebalancing Strategy knobs ---
# Check portfolio daily but only rebalance when stocks actually change (cost-effective).
# Set to 1 for daily checking, higher values for less frequent monitoring.
AI_REBALANCE_FREQUENCY_DAYS = 1  # Daily rebalancing

# AI Portfolio Rebalancing Threshold
AI_PORTFOLIO_MIN_IMPROVEMENT_THRESHOLD_ANNUAL = 0.05  # 5% annualized improvement required
# Only rebalance if new portfolio is expected to outperform current by 5% annually
# This prevents excessive trading on marginal improvements

# --- AI Strategy (3-stock daily selection) Rebalancing Threshold ---
# Only rebalance the 3-stock AI Strategy portfolio if (expected improvement - estimated transaction costs)
# clears this annualized threshold (converted to the model horizon in days).
AI_STRATEGY_MIN_IMPROVEMENT_THRESHOLD_ANNUAL = 0.05  # 5% annualized improvement required
# Formula: Converts to probability score difference based on evaluation window

# AI Portfolio Training Parameters
AI_PORTFOLIO_EVALUATION_WINDOW = 60  # Days to evaluate portfolio performance during training (more stable)
AI_PORTFOLIO_STEP_SIZE = 7  # Days between training samples (weekly = more training data)
AI_PORTFOLIO_PERFORMANCE_THRESHOLD_ANNUAL = 0.50  # ANNUALIZED return threshold (0.50 = 50% per year AFTER costs)
# The code automatically converts this to the evaluation window:
#    Formula: period_threshold = (1 + annual)^(days/365) - 1
#    Example: 50% annual → (1.50)^(30/365) - 1 = 3.39% for 30-day window
#    Higher values = more selective (fewer "good" portfolios), lower = more examples

# --- Momentum + AI Hybrid Strategy Parameters ---
MOMENTUM_AI_HYBRID_TOP_N = 20  # Select top N stocks by momentum
MOMENTUM_AI_HYBRID_PORTFOLIO_SIZE = 5  # Hold 5 stocks at a time (diversification)
MOMENTUM_AI_HYBRID_BUY_THRESHOLD = 0.02  # Buy if AI predicts >2% return
MOMENTUM_AI_HYBRID_SELL_THRESHOLD = -0.01  # Sell if AI predicts <-1% return
MOMENTUM_AI_HYBRID_REBALANCE_DAYS = 7  # Check weekly (not daily = less transaction costs)
MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK = 90  # 3-month momentum for stock ranking
MOMENTUM_AI_HYBRID_STOP_LOSS = 0.10  # 10% stop loss from entry
MOMENTUM_AI_HYBRID_TRAILING_STOP = 0.08  # 8% trailing stop once in profit

# --- Volatility-Adjusted Momentum Strategy Parameters ---
VOLATILITY_ADJ_MOM_LOOKBACK = 90  # 90-day momentum lookback
VOLATILITY_ADJ_MOM_VOL_WINDOW = 20  # 20-day volatility window
VOLATILITY_ADJ_MOM_MIN_SCORE = 0.5  # Minimum volatility-adjusted score threshold

# REGRESSION MODE: Probability thresholds removed - using simplified trading logic
TARGET_PERCENTAGE       = 0.006       # 0.6% target for buy/sell classification (balanced for 3-day moves)
# USE_MODEL_GATE removed - using simplified buy-and-hold logic
USE_MARKET_FILTER       = False      # market filter removed as per user request
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = False  # Disable strict benchmark filtering for small universes

# --- ML Model Selection Flags ---
USE_LOGISTIC_REGRESSION = False      # Not needed - too simple
USE_SVM                 = False      # SVR slower than XGBoost, usually worse
USE_MLP_CLASSIFIER      = False      # Less effective than LSTM/TCN for time series
USE_LIGHTGBM            = True       # ENABLED - Best for AI Portfolio meta-learning
USE_XGBOOST             = True       # KEEP - Best traditional ML
USE_LSTM                = True       # KEEP - Best deep learning for sequences
USE_GRU                 = False      # Redundant - LSTM is enough
USE_RANDOM_FOREST       = True       # KEEP - Good ensemble baseline
USE_TCN                 = True       # KEEP - Fast temporal model
USE_ELASTIC_NET         = False      # Too simple - linear models don't capture patterns
USE_RIDGE               = False      # Too simple - linear models don't capture patterns

# --- ML Training Settings ---
USE_ALPHA_WEIGHTS       = True       # Use alpha-based sample weights for training

# Simple Rule-Based Strategy removed - using AI strategy only

# --- Deep Learning specific hyperparameters
SEQUENCE_LENGTH         = 60         # 60 days lookback for longer horizon predictions
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.2
LSTM_EPOCHS             = 50
LSTM_BATCH_SIZE         = 64
LSTM_LEARNING_RATE      = 0.001

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
FORCE_TRAINING          = True
CONTINUE_TRAINING_FROM_EXISTING = False  # Force fresh training to avoid loading corrupted PyTorch models
# Threshold optimization removed - system uses simplified buy-and-hold

# --- Live Trading Model Selection ---
# Which period's model to use for live trading
# Options: "Best" (auto-select highest performer), "1-Year"
LIVE_TRADING_MODEL_PERIOD = "Best"

# --- Regression-Based Return Prediction ---
# Regression is now the default and only approach - removed USE_REGRESSION_MODEL flag

# Period-specific horizons (trading days) - matched to period scale
PERIOD_HORIZONS = {
    # Prediction horizon in trading days
    "1-Year": 63     # Predict 63 trading days ahead (~3 months)
}

USE_SINGLE_REGRESSION_MODEL = True  # Use single regression model instead of buy/sell pair
POSITION_SCALING_BY_CONFIDENCE = True  # Scale position size by predicted return magnitude


# Architecture options
TRY_LSTM_INSTEAD_OF_GRU = False  # Set to True to try LSTM instead of GRU for comparison
