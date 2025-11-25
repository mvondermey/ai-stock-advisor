import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
NUM_PROCESSES           = max(1, os.cpu_count() - 5) # Use all but one CPU core for parallel processing

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
MIN_PROBA_BUY           = 0.70      # ML gate threshold for buy model
MIN_PROBA_SELL          = 0.30       # ML gate threshold for sell model
TARGET_PERCENTAGE       = 0.002       # 0.8% target for buy/sell classification
CLASS_HORIZON_OPTIONS   = [5, 10, 20, 30, 40] # Days ahead for classification target
CLASS_HORIZON           = CLASS_HORIZON_OPTIONS[0] # Default CLASS_HORIZON
ENABLE_CLASS_HORIZON_OPTIMIZATION = True # Set to True to enable CLASS_HORIZON optimization
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
USE_XGBOOST             = False # Enable XGBoost
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
ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = True # Set to True to enable GRU hyperparameter search

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = True
FORCE_TRAINING          = True      # Set to True to force re-training of ML models
CONTINUE_TRAINING_FROM_EXISTING = False # Set to True to load existing models and continue training
FORCE_THRESHOLDS_OPTIMIZATION = True # Set to True to force re-optimization of ML thresholds
FORCE_PERCENTAGE_OPTIMIZATION = True # Set to True to force re-optimization of TARGET_PERCENTAGE
