import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# TwelveData SDK client
try:
    from twelvedata import TDClient
    TWELVEDATA_SDK_AVAILABLE = True
except ImportError:
    TWELVEDATA_SDK_AVAILABLE = False

# Import configuration from config.py
from config import (
    DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_CACHE_DIR, CACHE_DAYS,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, TWELVEDATA_API_KEY,
    MARKET_SELECTION, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES,
    PAUSE_BETWEEN_YF_CALLS, NUM_PROCESSES, SEED
)
