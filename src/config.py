import os
from pathlib import Path

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

# Alpaca API credentials
ALPACA_API_KEY          = "PK3FDQLRMEVFOAOU7VHD3Z6THE"
ALPACA_SECRET_KEY       = "8By7ituNTmspLsWc191hQfviD3xaNNdd2opB8tJAfmK6"

# TwelveData API credentials
TWELVEDATA_API_KEY      = "aed912386d7c47939ebc28a86a96a021"

# --- ML Library Availability Flags (initialized to False, updated in main.py) ---
ALPACA_AVAILABLE = False
TWELVEDATA_SDK_AVAILABLE = True  # Runtime detection will confirm availability
try:
    import torch
    PYTORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
USE_LSTM = True
USE_GRU = False

# --- Universe / selection
MARKET_SELECTION = {
    "ALPACA_STOCKS": False,   # ✅ ENABLED - ALL Alpaca tradable stocks
    "NASDAQ_ALL": False,
    "NASDAQ_100": True,
    "SP500": False,          # Disabled - using ALL Alpaca stocks instead
    "DOW_JONES": False,
    "POPULAR_ETFS": False,
    "CRYPTO": False,
    "DAX": False,
    "MDAX": False,
    "SMI": False,
    "FTSE_MIB": False,
}

# If ALPACA_STOCKS is enabled, Alpaca can return thousands of symbols.
# Set very high to effectively include (almost) all Alpaca-tradable US equities.
ALPACA_STOCKS_LIMIT = 100  # High limit = train models for ALL tradable stocks

# Exchange filter for Alpaca asset list. Use ["NASDAQ"] to restrict to NASDAQ only.
ALPACA_STOCKS_EXCHANGES = ["NASDAQ"]  # NASDAQ only
N_TOP_TICKERS           = 10       # Testing with 1 stock to verify predictions work
BATCH_DOWNLOAD_SIZE     = 20000       # Reduced batch size for stability
PAUSE_BETWEEN_BATCHES   = 5.0       # Pause between batches for stability
PAUSE_BETWEEN_YF_CALLS  = 0.5        # Pause between individual yfinance calls for fundamentals

# --- Parallel Processing
from multiprocessing import cpu_count
NUM_PROCESSES           = max(1, cpu_count() - 5)  # Use all but 2 CPU cores for parallel processing

# --- Backtest & training windows
BACKTEST_DAYS           = 90         # Backtest period in trading days (~60=2mo, ~125=6mo, ~250=1yr)
TRAIN_LOOKBACK_DAYS     = 365        # Train on ~1 year of history (user request)
VALIDATION_DAYS         = 90         # ✅ FIX 4: Validation period for threshold optimization

# --- Walk-Forward Retraining Frequency ---
# How often to retrain models during walk-forward backtest
# Options:
#   5  = Weekly retraining (aggressive, best for high volatility/penny stocks)
#   10 = Bi-weekly retraining (balanced, recommended for volatile stocks)
#   20 = Monthly retraining (conservative, recommended for S&P 500 / stable large-caps)
#   60 = Quarterly retraining (rare, only for very stable/long-term strategies)
RETRAIN_FREQUENCY_DAYS = 5  # Bi-weekly retraining - consider 20 for S&P 500

# --- Backtest Period Enable/Disable Flags ---
ENABLE_1YEAR_BACKTEST   = True   # ✅ Enabled - For simulation and strategy validation

# --- Training Period Enable/Disable Flags ---
ENABLE_1YEAR_TRAINING   = True

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 10
STRAT_SMA_LONG          = 20
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 2.0
ATR_MULT_TP             = 2.0        # 0 disables hard TP; rely on trailing
INVESTMENT_PER_STOCK    = 15000.0    # Fixed amount to invest per stock
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
# CLASS_HORIZON removed - use PERIOD_HORIZONS instead

# How many days of historical data to use when making predictions
# Must be >= 120 to have enough data after feature calculation (indicators need 50+ days lookback)
# Higher = more stable predictions, Lower = more reactive to recent changes
# Recommended: 60-250 days (120 = ~6 months, good balance)
PREDICTION_LOOKBACK_DAYS = 120

# --- AI Portfolio Rebalancing Strategy knobs ---
# Check portfolio daily but only rebalance when stocks actually change (cost-effective).
# Set to 1 for daily checking, higher values for less frequent monitoring.
AI_REBALANCE_FREQUENCY_DAYS = 1
# ✅ REGRESSION MODE: Probability thresholds removed - using simplified trading logic
TARGET_PERCENTAGE       = 0.006       # 0.6% target for buy/sell classification (balanced for 3-day moves)
# USE_MODEL_GATE removed - using simplified buy-and-hold logic
USE_MARKET_FILTER       = False      # market filter removed as per user request
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200
USE_PERFORMANCE_BENCHMARK = True  # Disable strict benchmark filtering for small universes

# --- ML Model Selection Flags ---
USE_LOGISTIC_REGRESSION = False
USE_SVM                 = True       # SVR for regression, SVM for classification
USE_MLP_CLASSIFIER      = False      # MLPRegressor for regression, MLPClassifier for classification (disabled - using regression)
USE_LIGHTGBM            = True       # enable LightGBM
USE_XGBOOST             = True       # enable XGBoost
USE_LSTM                = True
USE_GRU                 = True      # GRU disabled
USE_RANDOM_FOREST       = True       # enable Random Forest
USE_TCN                 = True       # Temporal Convolutional Network (sequence)
USE_ELASTIC_NET         = True       # Baseline linear regressor (tabular)
USE_RIDGE               = True       # Baseline linear regressor (tabular)

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
ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION = True  # Enable hyperparameter search for new features

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = False     # Disable SHAP (causes errors with XGBoost Regressor)
FORCE_TRAINING          = True
CONTINUE_TRAINING_FROM_EXISTING = False
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
    "1-Year": 20     # Predict 20 trading days ahead (~1 month)
}

USE_SINGLE_REGRESSION_MODEL = True  # Use single regression model instead of buy/sell pair
POSITION_SCALING_BY_CONFIDENCE = True  # Scale position size by predicted return magnitude


# Architecture options
TRY_LSTM_INSTEAD_OF_GRU = False  # Set to True to try LSTM instead of GRU for comparison
