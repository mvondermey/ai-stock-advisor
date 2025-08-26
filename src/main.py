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

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
import sys
import codecs

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

# Optional Stooq provider
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

from datetime import datetime, timedelta, timezone

# ============================
# Configuration / Hyperparams
# ============================

SEED                    = 42
np.random.seed(SEED)

# --- Provider & caching
DATA_PROVIDER           = 'yahoo'    # 'stooq' or 'yahoo'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
CACHE_DAYS              = 7

# --- Universe / selection
MARKET_SELECTION = {
    "NASDAQ_ALL": False,
    "SP500": False,
    "DAX": False,
    "MDAX": False,
    "SMI": False,
    "FTSE_MIB": True,
}
N_TOP_TICKERS           = 0          # Set to 0 to disable the limit and run on all performers
BATCH_DOWNLOAD_SIZE     = 10
PAUSE_BETWEEN_BATCHES   = 1.5

# --- Backtest & training windows
BACKTEST_DAYS           = 160
TRAIN_LOOKBACK_DAYS     = 360        # more data for model

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 20
STRAT_SMA_LONG          = 100
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 3.5
ATR_MULT_TP             = 0.0        # 0 disables hard TP; rely on trailing
RISK_PER_TRADE          = 0.01       # 1% of capital
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_UP            = 0.55       # ML gate threshold
USE_MODEL_GATE          = True       # ENABLE ML gate
USE_MARKET_FILTER       = True       # re-enable market filter
MARKET_FILTER_TICKER    = 'SPY'
MARKET_FILTER_SMA       = 200

# --- Misc
INITIAL_BALANCE         = 100_000.0
SAVE_PLOTS              = False

# ============================
# Helpers
# ============================

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _normalize_symbol(symbol: str, provider: str) -> str:
    if provider.lower() == 'yahoo':
        return symbol.replace('.', '-')
    if provider.lower() == 'stooq':
        return symbol.replace('-', '.')
    return symbol

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

# ============================
# Data access
# ============================

def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """A wrapper for load_prices that handles rate limiting with retries."""
    import time
    max_retries = 5
    base_wait_time = 10  # seconds

    for attempt in range(max_retries):
        try:
            return load_prices(ticker, start, end)
        except Exception as e:
            # This is a simplistic check; a more robust solution would inspect the exception type/message
            if "YFRateLimitError" in str(e) or "rate limit" in str(e).lower():
                wait_time = base_wait_time * (attempt + 1)
                print(f"  ⚠️ Rate limited trying to fetch {ticker}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # For other errors, fail immediately
                print(f"  ⚠️ Data load failed for {ticker}: {e}. No more retries for this ticker.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to load data for {ticker} after {max_retries} retries.")
    return pd.DataFrame()

def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and clean data from the selected provider, with an improved local caching mechanism."""
    _ensure_dir(DATA_CACHE_DIR)
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
    
    # --- Check cache first ---
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):
            try:
                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')
                return cached_df.loc[(cached_df.index >= _to_utc(start)) & (cached_df.index <= _to_utc(end))].copy()
            except Exception as e:
                print(f"⚠️ Could not read or slice cache file for {ticker}: {e}. Refetching.")

    # --- If not in cache or cache is old, fetch a broad range of data ---
    fetch_start = datetime.now(timezone.utc) - timedelta(days=1000) # Fetch a generous amount of data
    fetch_end = datetime.now(timezone.utc)
    start_utc = _to_utc(fetch_start)
    end_utc   = _to_utc(fetch_end)
    
    provider = DATA_PROVIDER.lower()
    final_df = pd.DataFrame()

    if provider == 'stooq':
        stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
        if stooq_df.empty and not ticker.upper().endswith('.US'):
            stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
        if not stooq_df.empty:
            final_df = stooq_df.copy()
        elif USE_YAHOO_FALLBACK:
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None:
                    final_df = downloaded_df.dropna()
            except Exception as e:
                print(f"⚠️ yfinance fallback failed for {ticker}: {e}")
    else:
        try:
            downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
            if downloaded_df is not None:
                final_df = downloaded_df.dropna()
        except Exception as e:
            print(f"⚠️ yfinance failed for {ticker}: {e}")
        if final_df.empty and pdr is not None:
            stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
            if not stooq_df.empty:
                final_df = stooq_df.copy()

    if final_df.empty:
        return pd.DataFrame()

    # Clean and normalize the downloaded data
    if isinstance(final_df.columns, pd.MultiIndex):
        final_df.columns = final_df.columns.get_level_values(0)
    final_df.columns = [str(col).capitalize() for col in final_df.columns]
    if "Close" not in final_df.columns and "Adj close" in final_df.columns:
        final_df = final_df.rename(columns={"Adj close": "Close"})

    if "Close" not in final_df.columns:
        return pd.DataFrame()

    final_df.index = pd.to_datetime(final_df.index, utc=True)
    final_df.index.name = "Date"
    final_df["Close"] = pd.to_numeric(final_df["Close"], errors="coerce")
    final_df = final_df.dropna(subset=["Close"])
    final_df = final_df.ffill().bfill()

    # --- Save the entire fetched data to cache ---
    if not final_df.empty:
        try:
            final_df.to_csv(cache_file)
        except Exception as e:
            print(f"⚠️ Could not write cache file for {ticker}: {e}")
            
    # Return the specifically requested slice
    return final_df.loc[(final_df.index >= _to_utc(start)) & (final_df.index <= _to_utc(end))].copy()

# ============================
# Ticker discovery
# ============================

def get_tickers_for_backtest(n: int = 10) -> List[str]:
    """Gets a list of n random tickers from the S&P 500."""
    fallback = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "LRCX"]
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in table[col].tolist()]
    except Exception as e:
        print(f"⚠️ Could not fetch S&P 500 list ({e}). Using static fallback.")
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

    # --- US Tickers ---
    if MARKET_SELECTION.get("NASDAQ_ALL"):
        try:
            url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
            df = pd.read_csv(url, sep='|')
            df_clean = df.iloc[:-1]
            nasdaq_tickers = df_clean[(df_clean['Test Issue'] == 'N') & (df_clean['ETF'] == 'N')]['Symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            print(f"✅ Fetched {len(nasdaq_tickers)} tickers from NASDAQ.")
        except Exception as e:
            print(f"⚠️ Could not fetch full NASDAQ list ({e}).")

    if MARKET_SELECTION.get("SP500"):
        try:
            import requests
            url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response_sp500 = requests.get(url_sp500, headers=headers)
            response_sp500.raise_for_status()
            table_sp500 = pd.read_html(response_sp500.text)[0]
            col = "Symbol" if "Symbol" in table_sp500.columns else table_sp500.columns[0]
            sp500_tickers = [s.replace('.', '-') for s in table_sp500[col].tolist()]
            all_tickers.update(sp500_tickers)
            print(f"✅ Fetched {len(sp500_tickers)} tickers from S&P 500.")
        except Exception as e:
            print(f"⚠️ Could not fetch S&P 500 list ({e}).")

    # --- German Tickers ---
    if MARKET_SELECTION.get("DAX"):
        try:
            import requests
            url_dax = "https://en.wikipedia.org/wiki/DAX"
            response_dax = requests.get(url_dax, headers=headers)
            response_dax.raise_for_status()
            table_dax = pd.read_html(response_dax.text)[4]
            dax_tickers = [f"{s}.DE" for s in table_dax['Ticker'].tolist()]
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
            table_mdax = pd.read_html(response_mdax.text)[3]
            mdax_tickers = [f"{s}.DE" for s in table_mdax['Ticker'].tolist()]
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
            table_smi = pd.read_html(response_smi.text)[2]
            smi_tickers = [s for s in table_smi['Ticker'].tolist()]
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
            table_mib = pd.read_html(response_mib.text)[2]
            # Find the correct column for tickers by checking for a known Italian ticker
            ticker_col = None
            for col in table_mib.columns:
                if table_mib[col].astype(str).str.contains('ENI.MI').any():
                    ticker_col = col
                    break
            if ticker_col:
                mib_tickers = [s for s in table_mib[ticker_col].tolist()]
                all_tickers.update(mib_tickers)
                print(f"✅ Fetched {len(mib_tickers)} tickers from FTSE MIB.")
            else:
                print("⚠️ Could not find ticker column for FTSE MIB.")
        except Exception as e:
            print(f"⚠️ Could not fetch FTSE MIB list ({e}).")

    if not all_tickers:
        print("⚠️ No tickers fetched. Using static fallback.")
        fallback = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "LRCX", "SPY", "QQQ"]
        all_tickers.update(fallback)

    string_tickers = {str(s) for s in all_tickers if pd.notna(s)}
    cleaned_tickers = {s.strip() for s in string_tickers if '$' not in s}
    
    normalized_tickers = [_normalize_symbol(sym, DATA_PROVIDER) for sym in cleaned_tickers]
    print(f"Total unique tickers to analyze: {len(normalized_tickers)}")
    return sorted(list(normalized_tickers))

# ============================
# Feature prep & model
# ============================

def fetch_training_data(ticker: str, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
    """Fetch prices and compute ML features. Default window is TRAIN_LOOKBACK_DAYS up to 'end' (now if None)."""
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=TRAIN_LOOKBACK_DAYS)

    df = load_prices(ticker, start, end)
    if df.empty or len(df) < FEAT_SMA_LONG + 10:
        print(f"⚠️ Insufficient data for {ticker} from {start.date()} to {end.date()}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    # Fix for missing OHLC columns (Yahoo fallback)
    if "High" not in df.columns and "Close" in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns and "Close" in df.columns:
        df["Low"] = df["Close"]
    if "Open" not in df.columns and "Close" in df.columns:
        df["Open"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    # Ensure 'Close' is numeric and drop rows with NaN in 'Close'
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # Fill missing values in other columns
    df = df.ffill().bfill()

    df["Returns"]    = df["Close"].pct_change(fill_method=None)
    df["SMA_F_S"]    = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    df["SMA_F_L"]    = df["Close"].rolling(FEAT_SMA_LONG).mean()
    df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()

    # --- Additional Features ---
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
    
    df["Target"]     = df["Close"].shift(-1)

    # Classification label: 5-day forward > +1%
    fwd = df["Close"].shift(-CLASS_HORIZON)
    df["TargetClass"] = ((fwd / df["Close"] - 1.0) > 0.01).astype(float)

    # Progress info
    req_cols = ["Close","Returns","SMA_F_S","SMA_F_L","Volatility", "RSI_feat", "MACD", "BB_upper", "Target"]
    ready = df[req_cols].dropna()
    print(f"   ↳ rows after features available: {len(ready)}")
    return df

def train_and_evaluate_models(df: pd.DataFrame):
    """Train and compare multiple classifiers, returning the best one."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")
        LGBMClassifier = None

    df = df.copy()
    # Feature generation (same as before)
    if "Returns" not in df.columns and "Close" in df.columns:
        df["Returns"] = df["Close"].pct_change()
    if "SMA_F_S" not in df.columns and "Close" in df.columns:
        df["SMA_F_S"] = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    if "SMA_F_L" not in df.columns and "Close" in df.columns:
        df["SMA_F_L"] = df["Close"].rolling(FEAT_SMA_LONG).mean()
    if "Volatility" not in df.columns and "Returns" in df.columns:
        df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()
    
    # --- Ensure additional features are present ---
    if 'RSI_feat' not in df.columns:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs = gain / loss
        df['RSI_feat'] = 100 - (100 / (1 + rs))
    if 'MACD' not in df.columns:
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
    if 'BB_upper' not in df.columns:
        df['BB_mid'] = df["Close"].rolling(window=20).mean()
        df['BB_std'] = df["Close"].rolling(window=20).std()
        df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)

    if "TargetClass" not in df.columns and "Close" in df.columns:
        fwd = df["Close"].shift(-CLASS_HORIZON)
        df["TargetClass"] = ((fwd / df["Close"] - 1.0) > 0.01).astype(float)

    req = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper", "TargetClass"]
    if any(c not in df.columns for c in req):
        print("⚠️ Missing columns for model comparison. Skipping.")
        return None

    d = df[req].dropna()
    if len(d) < 50:  # Increased requirement for cross-validation
        print("⚠️ Not enough rows after feature prep to compare models (need ≥ 50). Skipping.")
        return None

    # --- Check for class balance ---
    if d["TargetClass"].nunique() < 2:
        print("⚠️ Not enough class diversity for training (e.g., only 'up' or 'down' days). Skipping model.")
        return None

    feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper"]
    X_df = d[feature_names]
    y = d["TargetClass"].values

    # Scale features for models that are sensitive to scale (like Logistic Regression and SVC)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_df), columns=feature_names, index=X_df.index)

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced"),
        "SVM": SVC(probability=True, random_state=SEED, class_weight="balanced")
    }
    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation):")
    for name, model in models.items():
        try:
            # Set n_jobs=1 to avoid multiprocessing issues on Windows
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
            results[name] = np.mean(scores)
            print(f"    - {name}: {results[name]:.4f} (std: {np.std(scores):.4f})")
        except Exception as e:
            print(f"    - {name}: Failed evaluation. Error: {e}")
            results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best model: {best_model_name} with AUC = {results[best_model_name]:.4f}")

    # Train the best model on all available data and return it
    best_model_instance = models[best_model_name]
    best_model_instance.fit(X, y)
    
    # We need to return the scaler as well to process live data
    return best_model_instance, scaler

# ============================
# Rule-based backtester (ATR & ML gate)
# ============================

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Optional ML gate to allow buys."""
    def __init__(self, df: pd.DataFrame, initial_balance: float, transaction_cost: float,
                 model=None, scaler=None, min_proba: float = MIN_PROBA_UP, use_gate: bool = USE_MODEL_GATE,
                 market_data: Optional[pd.DataFrame] = None, use_market_filter: bool = USE_MARKET_FILTER):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model = model
        self.scaler = scaler
        self.min_proba = float(min_proba)
        self.use_gate = bool(use_gate) and (model is not None) and (scaler is not None)
        self.market_data = market_data
        self.use_market_filter = use_market_filter and market_data is not None

        self.reset()

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

        close = self.df["Close"]
        
        # --- Strategy Indicators ---
        # 1. Trend Filter: 200-day SMA
        self.df['SMA_200'] = close.rolling(window=200).mean()

        # 2. Crossover SMAs
        self.df['SMA_S'] = close.rolling(window=STRAT_SMA_SHORT).mean()
        self.df['SMA_L'] = close.rolling(window=STRAT_SMA_LONG).mean()

        # --- Other Indicators (for reference or potential future use) ---
        # Momentum: 14-day RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=ATR_PERIOD - 1, adjust=False).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # 3. Volume: On-Balance Volume (OBV) and its SMAs
        self.df['OBV'] = (np.sign(close.diff()) * self.df['Volume']).fillna(0).cumsum()
        self.df['OBV_SMA_S'] = self.df['OBV'].rolling(window=10).mean()
        self.df['OBV_SMA_L'] = self.df['OBV'].rolling(window=30).mean()
        # ------------------------------------

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
        
        # Bollinger Bands for features
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.df['BB_upper'] = bb_mid + (bb_std * 2)

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

    def _allow_buy_by_model(self, i: int) -> bool:
        if not self.use_gate:
            return True
        row = self.df.loc[i]
        feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper"]
        if any(pd.isna(row.get(f)) for f in feature_names):
            return False
        
        # Prepare feature vector as a DataFrame and scale it
        X_df = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
        X_scaled_np = self.scaler.transform(X_df)
        
        # Reconstruct DataFrame for the model to avoid warnings
        X = pd.DataFrame(X_scaled_np, columns=feature_names)
        
        try:
            # The model expects a DataFrame, not a NumPy array
            proba_up = float(self.model.predict_proba(X)[0][1])
        except Exception as e:
            # Add logging to see why it fails
            # print(f"DEBUG: Model prediction failed at step {i}. Error: {e}")
            return False
        return proba_up >= self.min_proba

    def _position_size_from_atr(self, price: float, atr: float) -> int:
        if atr is None or np.isnan(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollars = self.initial_balance * RISK_PER_TRADE
        per_share_risk = ATR_MULT_TRAIL * atr
        qty = int(risk_dollars / per_share_risk)
        return max(qty, 0)

    def _buy(self, price: float, atr: Optional[float], date: str):
        if self.cash <= 0:
            return
        qty = self._position_size_from_atr(price, atr if atr is not None else np.nan)
        if qty <= 0:
            return
        fee = price * qty * self.transaction_cost
        cost = price * qty + fee
        if cost > self.cash:
            qty = int(self.cash / (price * (1 + self.transaction_cost)))
            if qty <= 0:
                return
            fee = price * qty * self.transaction_cost
            cost = price * qty + fee
            if cost > self.cash:
                return

        self.cash -= cost
        self.shares += qty
        self.entry_price = price
        self.entry_atr = atr if atr is not None and not np.isnan(atr) else None
        self.highest_since_entry = price
        self.holding_bars = 0
        self.trade_log.append((date, "BUY", price, qty, "TICKER", {"fee": fee}, fee))

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
        self.trade_log.append((date, "SELL", price, qty, "TICKER", {"fee": fee}, fee))

    def step(self):
        if self.current_step < 1: # Need previous row for signal
            self.current_step += 1
            self.portfolio_history.append(self.initial_balance)
            return False

        if self.current_step >= len(self.df):
            return True

        # Current and previous data rows
        row = self.df.iloc[self.current_step]
        prev_row = self.df.iloc[self.current_step - 1]
        
        price = float(row["Close"])
        date = self._date_at(self.current_step)
        atr = float(row.get("ATR", np.nan)) if pd.notna(row.get("ATR", np.nan)) else None

        # --- Market Filter ---
        if self.use_market_filter:
            current_date = row['Date'].normalize()
            # Use asof to find the latest market data point on or before the current date
            market_slice = self.market_data.loc[:current_date]
            if not market_slice.empty:
                latest_market_data = market_slice.iloc[-1]
                market_close = latest_market_data['Close']
                market_sma = latest_market_data['SMA_L_MKT']
                if pd.notna(market_close) and pd.notna(market_sma) and market_close < market_sma:
                    # Market is in a downtrend. Sell any open position and do not open new ones.
                    if self.shares > 0:
                        self._sell(price, date)
                    
                    port_val = self.cash + self.shares * price
                    self.portfolio_history.append(port_val)
                    self.current_step += 1
                    return self.current_step >= len(self.df)

        # --- Entry Signal ---
        # Filter: Trend must be up (price above 200-day SMA)
        sma_200 = row.get('SMA_200')
        trend_ok = price > sma_200 if sma_200 and not np.isnan(sma_200) else False

        # Trigger: Price above long SMA (no crossover)
        sma_l = row.get('SMA_L')

        # --- Trend-Following Entry Signal ---
        sma_s = row.get('SMA_S')
        sma_l = row.get('SMA_L')
        sma_200 = row.get('SMA_200')

        # Condition: Enter if the medium-term trend is positive (20 SMA > 100 SMA).
        # The stricter long-term trend filter (price > 200 SMA) has been removed to make the system more responsive.
        buy_signal = (
            pd.notna(sma_s) and pd.notna(sma_l) and
            sma_s > sma_l
        )

        if self.shares == 0 and buy_signal:
            if self._allow_buy_by_model(self.current_step):
                self._buy(price, atr, date)
        
        # --- Simplified Exit Signal: SMA Crossunder ---
        prev_sma_s = prev_row.get('SMA_S')
        prev_sma_l = prev_row.get('SMA_L')
        sell_signal = (
            pd.notna(sma_s) and pd.notna(sma_l) and
            pd.notna(prev_sma_s) and pd.notna(prev_sma_l) and
            prev_sma_s >= prev_sma_l and  # Was above or equal
            sma_s < sma_l                # Now below
        )
        if self.shares > 0 and sell_signal:
            self._sell(price, date)
            
        # Original Exit Logic (ATR-based) is kept as a fallback/stop-loss

        # --- Exit Logic (ATR-based, unchanged) ---
        if self.shares > 0:
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            self.holding_bars += 1
            
            tp_level = self.entry_price * (1 + ATR_MULT_TP * (atr / price)) if (atr and price > 0) else self.entry_price * (1 + 0.12)
            tsl_level = None
            if self.highest_since_entry is not None and atr is not None:
                tsl_level = self.highest_since_entry - ATR_MULT_TRAIL * atr

            hit_tp = price >= tp_level
            hit_trail = (tsl_level is not None) and (price <= tsl_level)
            
            if hit_tp or hit_trail:
                self._sell(price, date)

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple]]:
        done = False
        while not done:
            done = self.step()
        if self.shares > 0:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log

# ============================
# Analytics
# ============================

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
    strat_returns = pd.Series(strategy_history).pct_change().dropna()
    bh_returns = pd.Series(buy_hold_history).pct_change().dropna()

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

# ============================
# Top Performer Analysis
# ============================

def find_top_performers(return_tickers: bool = False, n_top: int = N_TOP_TICKERS, fcf_min_threshold: float = 0.0):
    """
    Fetches S&P 500 & NASDAQ tickers, screens for the top N performers,
    and returns a list of (ticker, performance) tuples.
    """
    tickers = get_all_tickers()
    if not tickers:
        print("❌ No tickers to process. Exiting.")
        return []

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)
    
    # --- Step 1: Calculate Benchmark Performances ---
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
        
    # Determine the higher of the two benchmarks
    market_benchmark_perf = max(benchmark_perfs.values()) if benchmark_perfs else 0.0
    final_benchmark_perf = max(market_benchmark_perf, 50.0)
    print(f"  📈 Using final 1-Year performance benchmark of {final_benchmark_perf:.2f}% (max of market and 50%)")

    # --- Step 2: Calculate YTD Benchmarks ---
    print("- Calculating YTD Performance Benchmarks...")
    ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)
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

    # --- Step 3: Find stocks that beat both benchmarks ---
    performance_data = {}
    for ticker in tqdm(tickers, desc="Analyzing stock performance vs benchmarks"):
        try:
            # 1-Year Performance
            df_1y = load_prices_robust(ticker, start_date, end_date)
            perf_1y = -np.inf
            if df_1y is not None and not df_1y.empty and len(df_1y) > 200:
                start_price = df_1y['Close'].iloc[0]
                end_price = df_1y['Close'].iloc[-1]
                if start_price > 0:
                    perf_1y = ((end_price - start_price) / start_price) * 100
            
            # YTD Performance
            df_ytd = load_prices_robust(ticker, ytd_start_date, end_date)
            perf_ytd = -np.inf
            if df_ytd is not None and not df_ytd.empty:
                start_price = df_ytd['Close'].iloc[0]
                end_price = df_ytd['Close'].iloc[-1]
                if start_price > 0:
                    perf_ytd = ((end_price - start_price) / start_price) * 100

            if perf_1y > final_benchmark_perf and perf_ytd > ytd_benchmark_perf:
                performance_data[ticker] = perf_1y
        except Exception:
            pass

    if not performance_data:
        print("\n❌ Could not calculate performance for any stock.")
        return []

    # Filter for stocks that beat the high benchmark
    strong_performers = performance_data
    
    sorted_strong_performers = sorted(strong_performers.items(), key=lambda item: item[1], reverse=True)
    
    if n_top > 0 and len(sorted_strong_performers) > n_top:
        print(f"  ✅ Found {len(sorted_strong_performers)} stocks outperforming the benchmark. Selecting top {n_top}.")
        sorted_strong_performers = sorted_strong_performers[:n_top]
    else:
        print(f"  ✅ Found {len(sorted_strong_performers)} stocks outperforming the benchmark.")

    final_performers = sorted_strong_performers

    # --- Step 3: Fundamental Screen (Optional Free Cash Flow for the last fiscal year) ---
    if fcf_min_threshold is not None:
        print(f"  🔍 Screening {len(final_performers)} strong performers for positive FCF (if available)...")
        screened_performers = []
        
        pbar = tqdm(final_performers, desc="Applying FCF screen")
        
        for ticker, perf in pbar:
            try:
                cashflow = yf.Ticker(ticker).cashflow
                if not cashflow.empty:
                    latest_cashflow = cashflow.iloc[:, 0]
                    fcf_keys = ['Free Cash Flow', 'freeCashflow']
                    fcf = None
                    for key in fcf_keys:
                        if key in latest_cashflow.index:
                            fcf = latest_cashflow[key]
                            break
                    
                    # If FCF is found, it must be positive. If not found, we assume it's not applicable (e.g., a bank).
                    if fcf is None or fcf > 0:
                        screened_performers.append((ticker, perf))
                else:
                    # If no cashflow statement is available, we let it pass
                    screened_performers.append((ticker, perf))
            except Exception:
                # If there's any error fetching financials, we let it pass
                screened_performers.append((ticker, perf))

        print(f"  ✅ Found {len(screened_performers)} stocks passing the FCF screen.")
        final_performers = screened_performers

    if return_tickers:
        return final_performers
    
    # If not returning for backtest, just print the list
    print(f"\n\n🏆 Stocks Outperforming {high_benchmark_ticker} ({high_benchmark_perf:.2f}%) 🏆")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Performance':>15}")
    print("-" * 60)
    
    for i, (ticker, perf) in enumerate(sorted_strong_performers, 1):
        print(f"{i:<5} | {ticker:<10} | {perf:14.2f}%")
    
    print("-" * 60)
    return final_tickers


# ============================
# Main
# ============================

def main(fcf_threshold: float = 0.0):
    if pdr is None and DATA_PROVIDER.lower() == 'stooq':
        print("⚠️ pandas-datareader not installed; run: pip install pandas-datareader")

    print(f"🚀 AI-Powered Momentum & Trend Strategy (FCF > ${fcf_threshold:,.0f})\n" + "="*50 + "\n")
    
    # --- Step 1: Find top momentum stocks ---
    print("🔍 Step 1: Identifying stocks outperforming market benchmarks...")
    top_performers_data = find_top_performers(return_tickers=True, fcf_min_threshold=fcf_threshold)
    if not top_performers_data:
        print("❌ Could not identify top tickers. Aborting backtest.")
        return None, None, None, None, None, None, None, None
    
    top_tickers = [ticker for ticker, _ in top_performers_data]
    print(f"\n✅ Identified {len(top_tickers)} stocks for backtesting.\n")

    # --- Step 2: Run the AI-gated backtest on these stocks ---
    print("🔍 Step 2: Running AI-gated backtest on momentum stocks...")

    # Define backtest window
    bt_end = datetime.now(timezone.utc)
    bt_start = bt_end - timedelta(days=BACKTEST_DAYS)

    # Train models using data BEFORE backtest (avoid leakage)
    models: Dict[str, object] = {}
    train_end = bt_start - timedelta(days=1)
    train_start = train_end - timedelta(days=TRAIN_LOOKBACK_DAYS)

    scalers: Dict[str, object] = {}
    for ticker in top_tickers:
        print(f"🔄 Fetching training data for {ticker}...")
        training_data = fetch_training_data(ticker, train_start, train_end)
        if training_data.empty:
            print(f"⚠️ Training data for {ticker} is empty. Skipping.\n")
            continue
        print(f"✅ Training data fetched with {len(training_data)} rows for {ticker}.")
        
        model_and_scaler = train_and_evaluate_models(training_data)
        if model_and_scaler is not None:
            model, scaler = model_and_scaler
            models[ticker] = model
            scalers[ticker] = scaler
            print(f"✅ Best model selected and trained for {ticker}.\n")
        else:
            print(f"⚠️ Skipped model for {ticker} due to insufficient data or evaluation failure.\n")

    if not models and USE_MODEL_GATE:
        print("⚠️ No models were trained. Model-gating will be disabled for this run.\n")

    # --- Prepare Market Filter Data ---
    market_data = None
    if USE_MARKET_FILTER:
        print(f"🔄 Fetching market data for filter ({MARKET_FILTER_TICKER})...")
        # Fetch data over a longer period to ensure the long-term SMA is available
        market_start = train_start - timedelta(days=MARKET_FILTER_SMA)
        market_data = load_prices_robust(MARKET_FILTER_TICKER, market_start, bt_end)
        if not market_data.empty:
            market_data['SMA_L_MKT'] = market_data['Close'].rolling(MARKET_FILTER_SMA).mean()
            print("✅ Market data prepared.\n")
        else:
            print(f"⚠️ Could not load market data for {MARKET_FILTER_TICKER}. Filter will be disabled.\n")

    print("🔄 Running rule-based backtest...\n")
    capital_per_stock = INITIAL_BALANCE / max(len(top_tickers), 1)
    
    strategy_results = []
    buy_hold_results = []
    processed_tickers = []
    performance_metrics = []

    for ticker in top_tickers:
        print(f"▶ {ticker}: preparing data...")
        import time
        max_retries = 3
        
        # Load data with a warm-up period for indicators
        warmup_days = max(STRAT_SMA_LONG, 200) + 50
        data_start = bt_start - timedelta(days=warmup_days)

        for attempt in range(max_retries):
            try:
                df = load_prices_robust(ticker, data_start, bt_end)
                if df.empty or len(df.loc[bt_start:]) < STRAT_SMA_SHORT + 5:
                    print(f"  ⚠️ Not enough data for backtest (need >{STRAT_SMA_SHORT + 5}, got {len(df.loc[bt_start:])}). Skipping {ticker}.")
                    df = None
                break
            except Exception as e:
                if "YFRateLimitError" in str(e) or "rate limit" in str(e).lower():
                    wait_time = 10 * (attempt + 1)
                    print(f"  ⚠️ Rate limited by yfinance. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠️ Data load failed for {ticker}: {e}. Skipping.")
                    df = None
                    break
        if df is None or df.empty or len(df) < STRAT_SMA_SHORT + 5:
            print(f"  ⚠️ Skipping {ticker}: Insufficient or invalid data after loading.")
            continue

        # Ensure no NaN values in critical columns
        if df["Close"].isna().any():
            print(f"  ⚠️ Skipping {ticker}: 'Close' column contains NaN values.")
            continue

        model = models.get(ticker) if USE_MODEL_GATE else None
        scaler = scalers.get(ticker) if USE_MODEL_GATE else None
        env = RuleTradingEnv(df, initial_balance=capital_per_stock, transaction_cost=TRANSACTION_COST,
                             model=model, scaler=scaler, min_proba=MIN_PROBA_UP, use_gate=USE_MODEL_GATE,
                             market_data=market_data, use_market_filter=USE_MARKET_FILTER)
        final_val, log = env.run()
        
        # --- Analysis over backtest period only ---
        df_backtest = df.loc[df.index >= bt_start]
        
        # Slice strategy history to match backtest period
        strategy_history = env.portfolio_history[-len(df_backtest):]

        # Buy & Hold baseline over backtest period
        start_price = float(df_backtest["Close"].iloc[0])
        shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
        cash_bh = capital_per_stock - shares_bh * start_price
        buy_hold_history = (cash_bh + shares_bh * df_backtest["Close"]).tolist()
        bh_val = buy_hold_history[-1]

        # If no trades occurred, default to buy & hold for this ticker to avoid idle cash
        made_trades = any(t[1] == "BUY" or t[1] == "SELL" for t in log)
        
        # Use B&H history for analysis if no trades were made, otherwise use the actual strategy history
        strategy_history_for_analysis = strategy_history
        if not made_trades:
            final_val = bh_val
            strategy_history_for_analysis = buy_hold_history
            print(f"  ✅ No trades taken; defaulting to Buy&Hold value for performance analysis.")

        print(f"  ✅ Final strategy value: ${final_val:,.2f} | Buy&Hold: ${bh_val:,.2f}")
        perf_data = analyze_performance(log, strategy_history_for_analysis, buy_hold_history, ticker)

        strategy_results.append(final_val)
        buy_hold_results.append(bh_val)
        processed_tickers.append(ticker)
        performance_metrics.append(perf_data)

    # --- Final Portfolio Summary ---
    num_processed = len(processed_tickers)
    num_skipped = len(top_tickers) - num_processed
    skipped_capital = num_skipped * capital_per_stock

    final_strategy_value = sum(strategy_results) + skipped_capital
    final_buy_hold_value = sum(buy_hold_results) + skipped_capital

    if final_strategy_value > 0:
        print(f"\n--- PORTFOLIO SUMMARY (FCF > ${fcf_threshold:,.0f}) ---")
        print(f"  Processed {num_processed}/{len(top_tickers)} stocks.")
        print("  Final Combined Portfolio Value: ${:,.2f}".format(final_strategy_value))
        print("  Final Buy-and-Hold Portfolio Value: ${:,.2f}".format(final_buy_hold_value))
        if skipped_capital > 0:
            print(f"  (Includes ${skipped_capital:,.2f} of unprocessed capital for {num_skipped} skipped stock(s))")
    
        if SAVE_PLOTS:
            _ensure_dir(Path("plots"))
            fig = plt.figure(figsize=(8, 4))
            plt.title(f"Combined Portfolio (FCF > ${fcf_threshold:,.0f})")
            plt.plot([0, 1], [INITIAL_BALANCE, final_strategy_value], label="Strategy")
            plt.plot([0, 1], [INITIAL_BALANCE, final_buy_hold_value], label="Buy & Hold")
            plt.xlabel("Period"); plt.ylabel("Value ($)"); plt.legend()
            fig.savefig(f"plots/combined_portfolio_fcf_{fcf_threshold}.png")
            plt.close(fig)
    
    return final_strategy_value, final_buy_hold_value, models, scalers, top_performers_data, strategy_results, processed_tickers, performance_metrics

# ============================
# AI Recommendation Engine
# ============================

def get_ai_recommendations():
    """
    Identifies top-performing stocks and uses the AI model to generate
    a list of current "buy" recommendations based on the latest data.
    """
    print("🚀 AI Recommendation Engine\n" + "="*50 + "\n")

    # --- Step 1: Find top momentum stocks ---
    print("🔍 Step 1: Identifying stocks outperforming market benchmarks...")
    top_tickers = find_top_performers(return_tickers=True)
    if not top_tickers:
        print("❌ Could not identify top tickers. Aborting.")
        return
    print(f"\n✅ Identified {len(top_tickers)} stocks for analysis.\n")

    # --- Step 2: Train models for each stock ---
    print("🔍 Step 2: Training AI models on recent historical data...")
    models: Dict[str, object] = {}
    scalers: Dict[str, object] = {}
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=TRAIN_LOOKBACK_DAYS)

    for ticker in top_tickers:
        print(f"🔄 Fetching training data for {ticker}...")
        training_data = fetch_training_data(ticker, train_start, train_end)
        if training_data.empty:
            print(f"⚠️ Training data for {ticker} is empty. Skipping.\n")
            continue
        
        model_and_scaler = train_and_evaluate_models(training_data)
        if model_and_scaler is not None:
            model, scaler = model_and_scaler
            models[ticker] = model
            scalers[ticker] = scaler
            print(f"✅ Model trained for {ticker}.\n")
        else:
            print(f"⚠️ Skipped model for {ticker} due to insufficient data.\n")

    if not models:
        print("❌ No models could be trained. Cannot generate recommendations.")
        return

    # --- Step 3: Analyze latest data for buy signals ---
    print("🔍 Step 3: Analyzing latest market data for AI-approved buy signals...")
    recommendations = []
    analysis_end = datetime.now(timezone.utc)
    analysis_start = analysis_end - timedelta(days=max(STRAT_SMA_LONG, 200) + 50) # Need warmup

    for ticker in tqdm(models.keys(), desc="Scanning for recommendations"):
        df = load_prices_robust(ticker, analysis_start, analysis_end)
        if df.empty or len(df) < STRAT_SMA_LONG + 2:
            continue

        # Calculate all necessary indicators and features
        env = RuleTradingEnv(df.copy(), 1, 1, models[ticker], scalers[ticker])
        df_processed = env.df # Get the df with all indicators calculated

        if len(df_processed) < 2:
            continue

        # Check for buy signal on the last day
        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2]

        buy_signal = (
            pd.notna(last_row['SMA_S']) and pd.notna(last_row['SMA_L']) and
            pd.notna(prev_row['SMA_S']) and pd.notna(prev_row['SMA_L']) and
            prev_row['SMA_S'] <= prev_row['SMA_L'] and
            last_row['SMA_S'] > last_row['SMA_L']
        )

        if buy_signal:
            # Now, check if the AI model approves this signal
            if env._allow_buy_by_model(len(df_processed) - 1):
                recommendations.append(ticker)

    # --- Step 4: Print recommendations ---
    print("\n\n" + "="*50)
    if recommendations:
        print("✅ AI-Approved Buy Recommendations for Today:")
        for i, ticker in enumerate(recommendations, 1):
            print(f"  {i}. {ticker}")
    else:
        print("ℹ️ No AI-approved buy signals found among top performers for today.")
    print("="*50)


# ============================
# Final Analysis & Recommendations
# ============================

def print_final_summary(sorted_results, models, scalers):
    """
    Analyzes the latest data for the sorted list of stocks and prints a final
    summary table with performance, Sharpe ratio, and a detailed AI recommendation.
    """
    print("\n\n🔍 Final AI Recommendations for Screened Stock List 🔍\n" + "="*115)
    
    recommendation_details = {}
    analysis_end = datetime.now(timezone.utc)
    analysis_start = analysis_end - timedelta(days=max(STRAT_SMA_LONG, 200) + 50)

    tickers_to_check = [res['ticker'] for res in sorted_results]

    for ticker in tqdm(tickers_to_check, desc="Generating final recommendations"):
        model = models.get(ticker)
        scaler = scalers.get(ticker)
        
        # Default values
        recommendation_details[ticker] = {'trend_signal': 'No', 'ai_prob': 'N/A', 'recommendation': 'HOLD / SELL'}

        if not model or not scaler:
            recommendation_details[ticker]['ai_prob'] = 'No Model'
            continue

        df = load_prices_robust(ticker, analysis_start, analysis_end)
        if df.empty or len(df) < STRAT_SMA_LONG + 2:
            recommendation_details[ticker]['ai_prob'] = 'Data Error'
            continue

        env = RuleTradingEnv(df.copy(), 1, 1, model, scaler)
        df_processed = env.df

        if len(df_processed) < 2:
            recommendation_details[ticker]['ai_prob'] = 'Data Error'
            continue

        last_row = df_processed.iloc[-1]

        # --- Condition 1: Check for Trend-Following Signal ---
        sma_s = last_row.get('SMA_S')
        sma_l = last_row.get('SMA_L')

        trend_signal_active = (
            pd.notna(sma_s) and pd.notna(sma_l) and
            sma_s > sma_l
        )
        recommendation_details[ticker]['trend_signal'] = "Yes" if trend_signal_active else "No"

        # --- Condition 2: Get AI Model's Probability ---
        feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "BB_upper"]
        if any(pd.isna(last_row.get(f)) for f in feature_names):
            recommendation_details[ticker]['ai_prob'] = 'Missing Feat.'
            ai_approved = False
        else:
            X_df = pd.DataFrame([[last_row[f] for f in feature_names]], columns=feature_names)
            X_scaled = scaler.transform(X_df)
            X = pd.DataFrame(X_scaled, columns=feature_names)
            proba_up = float(model.predict_proba(X)[0][1])
            recommendation_details[ticker]['ai_prob'] = f"{proba_up:.2%}"
            ai_approved = proba_up >= MIN_PROBA_UP
        
        # --- Final Recommendation ---
        if trend_signal_active and ai_approved:
            recommendation_details[ticker]['recommendation'] = "BUY"
        else:
            recommendation_details[ticker]['recommendation'] = "HOLD / SELL"

    print("\n" + "="*115)
    header = (f"{'Ticker':<10} | {'1Y Performance':>18} | {'Backtest Performance':>22} | {'Sharpe Ratio':>15} | {'Trend Signal?':>15} | {'AI Confidence':>15} | {'Recommendation':>15}")
    print(header)
    print("-" * len(header))
    for res in sorted_results:
        ticker = res['ticker']
        value = res['performance']
        sharpe = res['sharpe']
        one_year_perf = res['one_year_perf']
        
        rec_data = recommendation_details.get(ticker, {})
        trend_signal = rec_data.get('trend_signal', 'N/A')
        ai_prob = rec_data.get('ai_prob', 'N/A')
        recommendation = rec_data.get('recommendation', 'N/A')
        
        print(f"{ticker:<10} | {one_year_perf:>17.2f}% | ${value:>21,.2f} | {sharpe:15.2f} | {trend_signal:>15} | {ai_prob:>15} | {recommendation:>15}")
    print("="*115)

# ============================
# FCF Optimization & Main Execution
# ============================

def run_backtest_and_optimize():
    """
    Runs the backtest with a positive FCF threshold, sorts results by Sharpe Ratio,
    and prints a final summary with AI recommendations.
    """
    import time
    start_time = time.time()
    print("⚙️  Starting Backtest for Stocks with Positive FCF...\n" + "="*50)
    
    fcf_threshold = 0.0
    
    print(f"\n--- Running backtest for FCF > ${fcf_threshold:,.0f} ---")
    res = main(fcf_threshold=fcf_threshold)
    
    if res is None or res[0] is None:
        print("\n❌ Backtest could not be completed as no tickers were found or processed.")
        return

    final_strategy_val, final_buy_hold_val, models, scalers, top_performers_data, strategy_results, processed_tickers, performance_metrics = res

    print("\n\n🏆 Backtest Results 🏆\n" + "="*50)
    print(f"   - Final Portfolio Value: ${final_strategy_val:,.2f}")
    print(f"   - Equivalent Buy & Hold Value: ${final_buy_hold_val:,.2f}")
    
    if processed_tickers:
        # Combine results into a list of dictionaries for easy sorting and processing
        perf_map = dict(top_performers_data)
        combined_results = []
        for i, ticker in enumerate(processed_tickers):
            combined_results.append({
                'ticker': ticker,
                'performance': strategy_results[i],
                'sharpe': performance_metrics[i].get('sharpe_ratio', 0.0),
                'one_year_perf': perf_map.get(ticker, 0.0)
            })

        # Sort by Sharpe Ratio in descending order
        combined_results.sort(key=lambda x: x['sharpe'], reverse=True)
        
        # Call the final summary function with the sorted results
        print_final_summary(combined_results, models, scalers)

    print("="*50)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n⏱️  Total Execution Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")


if __name__ == "__main__":
    # The original backtesting functionality can be run by uncommenting the next line
    # main() 
    
    # Run the backtest, optimization, and recommendation flow
    run_backtest_and_optimize()

    # The standalone AI recommendation engine can be run by uncommenting the next line
    # get_ai_recommendations()
