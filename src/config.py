import os
from pathlib import Path

# ============================
# Configuration / Hyperparams
# ============================

SEED                    = 42

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

# --- ML Library Availability Flags (initialized to False, updated in main.py) ---
ALPACA_AVAILABLE = False
TWELVEDATA_SDK_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
USE_LSTM = False
USE_GRU = False

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
N_TOP_TICKERS           = 3        # Always pick top 3 predicted performers
BATCH_DOWNLOAD_SIZE     = 20000       # Reduced batch size for stability
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
NUM_PROCESSES           = None  # Will be set to cpu_count() - 2 in main.py

# --- Backtest & training windows
BACKTEST_DAYS           = 365        # 1 year for backtest
BACKTEST_DAYS_3MONTH    = 90         # 3 months for backtest
BACKTEST_DAYS_1MONTH    = 32         # 1 month for backtest
TRAIN_LOOKBACK_DAYS     = 400        # Training data (increased for more samples with shorter horizons)
VALIDATION_DAYS         = 90         # ✅ FIX 4: Validation period for threshold optimization

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
CLASS_HORIZON           = 20         # days ahead for return prediction (aligned with holding period)
# ✅ REGRESSION MODE: Thresholds based on predicted return percentages, not probabilities
MIN_PROBA_BUY           = -1.0      # Disable buy threshold (always eligible)
MIN_PROBA_BUY_OPTIONS   = [-1.0]    # No optimization, keep disabled
MIN_PROBA_SELL          = 1.0       # Disable sell threshold via gate
MIN_PROBA_SELL_OPTIONS  = [1.0]     # No optimization, keep disabled
TARGET_PERCENTAGE       = 0.006       # 0.6% target for buy/sell classification (balanced for 3-day moves)
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = False      # market filter removed as per user request
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True   # Set to True to enable benchmark filtering

# --- ML Model Selection Flags ---
USE_LOGISTIC_REGRESSION = False
USE_SVM                 = False
USE_MLP_CLASSIFIER      = False
USE_LIGHTGBM            = False
USE_XGBOOST             = False
USE_LSTM                = False
USE_GRU                 = True
USE_RANDOM_FOREST       = False

# --- Simple Rule-Based Strategy specific hyperparameters
USE_SIMPLE_RULE_STRATEGY = False
SIMPLE_RULE_TRAILING_STOP_PERCENT = 0.10 # 10% trailing stop
SIMPLE_RULE_TAKE_PROFIT_PERCENT = 0.10   # 10% take profit

# --- Deep Learning specific hyperparameters
SEQUENCE_LENGTH         = 30         # Number of past days to consider for LSTM/GRU (balanced for learning)
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.2
LSTM_EPOCHS             = 50
LSTM_BATCH_SIZE         = 64
LSTM_LEARNING_RATE      = 0.001

# --- GRU Hyperparameter Search Ranges ---
GRU_HIDDEN_SIZE_OPTIONS = [32, 64, 128]  # Focus on mid-range (removed 16, 256)
GRU_NUM_LAYERS_OPTIONS  = [1, 2, 3]      # Reduced (removed 4 - too deep for short sequences)
GRU_DROPOUT_OPTIONS     = [0.1, 0.2, 0.3]  # Narrowed range
GRU_LEARNING_RATE_OPTIONS = [0.0005, 0.001, 0.005]  # Focus on effective range
GRU_BATCH_SIZE_OPTIONS  = [32, 64, 128]  # Removed 16, 256
GRU_EPOCHS_OPTIONS      = [30, 50, 70]   # Reduced for faster iteration
GRU_CLASS_HORIZON_OPTIONS = [2, 3, 5]  # Short-term focus (removed 1 - too noisy, removed 7+)
GRU_TARGET_PERCENTAGE_OPTIONS = [0.005, 0.006, 0.007]  # Narrow range for short-term moves
ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = True  # Enable hyperparameter search for new features

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = True
FORCE_TRAINING          = True
CONTINUE_TRAINING_FROM_EXISTING = False
FORCE_THRESHOLDS_OPTIMIZATION = False  # ✅ Kelly Criterion makes threshold optimization unnecessary
FORCE_PERCENTAGE_OPTIMIZATION = False  # Use B&H-based targets for each period

# --- Live Trading Model Selection ---
# Which period's model to use for live trading
# Options: "Best" (auto-select highest performer), "3-Month", "1-Month", "YTD", "1-Year"
LIVE_TRADING_MODEL_PERIOD = "Best"

# --- Regression-Based Return Prediction ---
USE_REGRESSION_MODEL = True  # Use regression to predict actual returns instead of classification

# Period-specific horizons (trading days)
PERIOD_HORIZONS = {
    "1-Year": 252,   # Trading days in a year
    "YTD": None,     # Will be calculated dynamically based on YTD days
    "3-Month": 63,   # ~3 months trading days
    "1-Month": 21    # ~1 month trading days
}

MIN_PREDICTED_RETURN = 0.05  # Only buy if predicted return > 5%
MIN_SELL_RETURN = -0.02  # Sell if predicted return drops below -2% (cut losses)
POSITION_SCALING_BY_CONFIDENCE = True  # Scale position size by predicted return magnitude
