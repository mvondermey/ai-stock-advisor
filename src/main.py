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
DATA_PROVIDER           = 'stooq'    # 'stooq' or 'yahoo'
USE_YAHOO_FALLBACK      = True       # let Yahoo fill gaps if Stooq thin
DATA_CACHE_DIR          = Path("data_cache")
TOP_CACHE_PATH          = Path("logs/top_tickers_cache.json")
CACHE_DAYS              = 7

# --- Universe / selection
N_TOP_TICKERS           = 6
BATCH_DOWNLOAD_SIZE     = 10
PAUSE_BETWEEN_BATCHES   = 1.5

# --- Backtest & training windows
BACKTEST_DAYS           = 160
TRAIN_LOOKBACK_DAYS     = 360        # more data for model

# --- Strategy (separate from feature windows)
STRAT_SMA_SHORT         = 20
STRAT_SMA_LONG          = 100
ATR_PERIOD              = 14
ATR_MULT_TRAIL          = 4.0
ATR_MULT_TP             = 6.0
RISK_PER_TRADE          = 0.01       # 1% of capital
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_UP            = 0.55       # ML gate threshold
USE_MODEL_GATE          = True       # require model proba >= threshold for buys

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
# Data access with cache
# ============================

def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load from cache; then provider-specific fetch; then optional fallback; write back to cache."""
    _ensure_dir(DATA_CACHE_DIR)
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"

    # Normalize times
    start_utc = _to_utc(start)
    end_utc   = _to_utc(end)

    df_cached = pd.DataFrame()
    if cache_file.exists():
        try:
            df_cached = pd.read_csv(cache_file, parse_dates=["Date"]).set_index("Date").sort_index()
            df_cached.index = pd.to_datetime(df_cached.index, utc=True)
        except Exception:
            df_cached = pd.DataFrame()

    if not df_cached.empty and df_cached.index.max() >= (end_utc - timedelta(days=2)):
        out = df_cached.loc[(df_cached.index >= start_utc) & (df_cached.index <= end_utc)].copy()
        if not out.empty:
            return out

    dl_start = (df_cached.index.max() + timedelta(days=1)) if not df_cached.empty else start_utc
    new_df = pd.DataFrame()
    provider = DATA_PROVIDER.lower()

    if provider == 'stooq':
        stooq_df = _fetch_from_stooq(ticker, dl_start, end_utc)
        if stooq_df.empty and not ticker.upper().endswith('.US'):
            stooq_df = _fetch_from_stooq(f"{ticker}.US", dl_start, end_utc)
        if not stooq_df.empty:
            new_df = stooq_df.copy()
        elif USE_YAHOO_FALLBACK:
            try:
                downloaded_df = yf.download(ticker, start=dl_start, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None:
                    new_df = downloaded_df.dropna()
            except Exception as e:
                print(f"⚠️ yfinance fallback failed for {ticker}: {e}")
    else:
        try:
            downloaded_df = yf.download(ticker, start=dl_start, end=end_utc, auto_adjust=True, progress=False)
            if downloaded_df is not None:
                new_df = downloaded_df.dropna()
        except Exception as e:
            print(f"⚠️ yfinance failed for {ticker}: {e}")
        if new_df.empty and pdr is not None:
            stooq_df = _fetch_from_stooq(ticker, dl_start, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", dl_start, end_utc)
            if not stooq_df.empty:
                new_df = stooq_df.copy()

    if not new_df.empty:
        new_df = new_df.copy()

        # Flatten MultiIndex columns, which can occur with yfinance
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)

        # Normalize column names to handle provider inconsistencies (e.g., 'close' vs 'Close')
        new_df.columns = [str(col).capitalize() for col in new_df.columns]

        # Ensure a 'Close' column exists, using 'Adj close' as a fallback
        if "Close" not in new_df.columns and "Adj close" in new_df.columns:
            new_df = new_df.rename(columns={"Adj close": "Close"})

    # If 'Close' column now exists, process and save the data.
    if "Close" in new_df.columns:
        # Ensure 'Close' is numeric before processing
        new_df["Close"] = pd.to_numeric(new_df["Close"], errors="coerce")
        new_df = new_df.dropna(subset=["Close"])

        new_df.index = pd.to_datetime(new_df.index, utc=True)
        new_df.index.name = "Date"
        # Fill missing values in other columns if necessary
        new_df = new_df.ffill().bfill()
        if not df_cached.empty:
            combined = pd.concat([df_cached, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined.index = pd.to_datetime(combined.index, utc=True)
        else:
            combined = new_df
        try:
            combined.reset_index().to_csv(cache_file, index=False)
        except Exception:
            pass
        return combined.loc[(combined.index >= start_utc) & (combined.index <= end_utc)].copy()

    if not df_cached.empty:
        df_filtered = df_cached.loc[(df_cached.index >= start_utc) & (df_cached.index <= end_utc)].copy()
        if "Close" in df_filtered.columns:
            return df_filtered.dropna(subset=["Close"])
        return df_filtered
    return pd.DataFrame()

# ============================
# Ticker discovery
# ============================

def get_top_performing_stocks_ytd(sp500: bool = True, n: int = N_TOP_TICKERS) -> List[str]:
    fallback = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "LRCX"]

    # Cache first
    try:
        if TOP_CACHE_PATH.exists():
            data = json.loads(TOP_CACHE_PATH.read_text(encoding="utf-8"))
            ts_val = data.get("timestamp")
            ts = datetime.fromisoformat(ts_val) if isinstance(ts_val, str) else None
            if ts:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                if (now - ts).days <= CACHE_DAYS and data.get("tickers"):
                    return data["tickers"][:n]
    except Exception as e:
        print(f"⚠️ Cache read failed: {e}")

    # Get S&P 500 symbol list
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in table[col].tolist()]
    except Exception as e:
        print(f"⚠️ Could not fetch S&P 500 list ({e}). Using static fallback.")
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in fallback]

    end_date = datetime.now(timezone.utc)
    start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    def compute_growth_for_batch(batch: List[str]) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []

        # Stooq first
        if DATA_PROVIDER.lower() == 'stooq' and pdr is not None:
            for t in (batch if isinstance(batch, list) else [batch]):
                d = _fetch_from_stooq(t, start_date, end_date)
                if (d.empty or "Close" not in d.columns or len(d) < 2) and not t.upper().endswith('.US'):
                    d = _fetch_from_stooq(f"{t}.US", start_date, end_date)
                if d.empty or "Close" not in d.columns or len(d) < 2:
                    continue
                sp, ep = float(d["Close"].iat[0]), float(d["Close"].iat[-1])
                if sp > 0:
                    results.append((t, (ep - sp) / sp))
            if results:
                return results

        # Yahoo polite batch
        try:
            df = yf.download(batch, period="ytd", interval="1d", progress=False, auto_adjust=True, group_by="ticker", threads=False)
        except Exception:
            df = pd.DataFrame()

        # Single ticker
        if isinstance(batch, str) or (isinstance(batch, list) and len(batch) == 1):
            key = batch[0] if isinstance(batch, list) else batch
            d = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            s = pd.Series(dtype=float)
            try:
                if isinstance(d.columns, pd.MultiIndex):
                    if (key, "Close") in d.columns:
                        s = d[(key, "Close")].dropna()
                    elif (key, "Adj Close") in d.columns:
                        s = d[(key, "Adj Close")].dropna()
                else:
                    s = d["Close"].dropna() if "Close" in d.columns else d.get("Adj Close", pd.Series(dtype=float)).dropna()
            except Exception:
                s = pd.Series(dtype=float)
            if not s.empty and len(s) >= 2:
                sp, ep = float(s.iat[0]), float(s.iat[-1])
                if sp > 0:
                    results.append((key, (ep - sp) / sp))
            return results

        # MultiIndex (many tickers)
        if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
            close_levels = {c[1] for c in df.columns}
            level = "Close" if "Close" in close_levels else ("Adj Close" if "Adj Close" in close_levels else None)
            if level:
                for t in (batch if isinstance(batch, list) else [batch]):
                    try:
                        s = df[t][level].dropna()
                        if len(s) < 2:
                            continue
                        sp, ep = float(s.iat[0]), float(s.iat[-1])
                        if sp > 0:
                            results.append((t, (ep - sp) / sp))
                    except Exception:
                        continue

        return results

    performances: List[Tuple[str, float]] = []
    import time
    for i in tqdm(range(0, min(len(tickers_all), 500), BATCH_DOWNLOAD_SIZE), desc="Processing S&P 500 Tickers"):
        batch = tickers_all[i:i+BATCH_DOWNLOAD_SIZE]
        performances.extend(compute_growth_for_batch(batch))
        time.sleep(PAUSE_BETWEEN_BATCHES)

    if not performances:
        print("⚠️ Using fallback tickers due to download issues.")
        top = [_normalize_symbol(sym, DATA_PROVIDER) for sym in fallback][:n]
    else:
        top = [t for t, _ in sorted(performances, key=lambda x: x[1], reverse=True)[:n]]

    try:
        _ensure_dir(TOP_CACHE_PATH.parent)
        TOP_CACHE_PATH.write_text(
            json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tickers": top}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass
    return top

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
    df["Target"]     = df["Close"].shift(-1)

    # Classification label: 5-day forward > +1%
    fwd = df["Close"].shift(-CLASS_HORIZON)
    df["TargetClass"] = ((fwd / df["Close"] - 1.0) > 0.01).astype(float)

    # Progress info
    req_cols = ["Close","Returns","SMA_F_S","SMA_F_L","Volatility","Target"]
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
    if "TargetClass" not in df.columns and "Close" in df.columns:
        fwd = df["Close"].shift(-CLASS_HORIZON)
        df["TargetClass"] = ((fwd / df["Close"] - 1.0) > 0.01).astype(float)

    req = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "TargetClass"]
    if any(c not in df.columns for c in req):
        print("⚠️ Missing columns for model comparison. Skipping.")
        return None

    d = df[req].dropna()
    if len(d) < 50:  # Increased requirement for cross-validation
        print("⚠️ Not enough rows after feature prep to compare models (need ≥ 50). Skipping.")
        return None

    feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility"]
    X_df = d[feature_names]
    y = d["TargetClass"].values

    # Scale features for models that are sensitive to scale (like Logistic Regression and SVC)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_df), columns=feature_names, index=X_df.index)

    models = {
        "Logistic Regression": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced", min_samples_leaf=2),
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
                 model=None, scaler=None, min_proba: float = MIN_PROBA_UP, use_gate: bool = USE_MODEL_GATE):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model = model
        self.scaler = scaler
        self.min_proba = float(min_proba)
        self.use_gate = bool(use_gate) and (model is not None) and (scaler is not None)

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
        # Strategy MAs
        self.df["SMA_S"] = close.rolling(STRAT_SMA_SHORT).mean()
        self.df["SMA_L"] = close.rolling(STRAT_SMA_LONG).mean()

        # ATR (prefer HL/prevClose; else fallback from returns)
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
            # Fallback: use volatility of returns * price
            ret = close.pct_change(fill_method=None)
            self.df["ATR"] = (ret.rolling(ATR_PERIOD).std() * close).rolling(2).mean()

        # Feature columns for ML gate (align with training)
        self.df["Returns"]    = close.pct_change(fill_method=None)
        self.df["SMA_F_S"]    = close.rolling(FEAT_SMA_SHORT).mean()
        self.df["SMA_F_L"]    = close.rolling(FEAT_SMA_LONG).mean()
        self.df["Volatility"] = self.df["Returns"].rolling(FEAT_VOL_WINDOW).std()

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

    def _allow_buy_by_model(self, i: int) -> bool:
        if not self.use_gate:
            return True
        row = self.df.loc[i]
        feature_names = ["Close", "Returns", "SMA_F_S", "SMA_F_L", "Volatility"]
        if any(pd.isna(row.get(f)) for f in feature_names):
            return False
        
        # Prepare feature vector and scale it
        X_raw = np.array([[row[f] for f in feature_names]], dtype=float)
        X = self.scaler.transform(X_raw)
        
        try:
            proba_up = float(self.model.predict_proba(X)[0][1])
        except Exception:
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
        if self.current_step >= len(self.df):
            return True

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        date = self._date_at(self.current_step)
        sma_s = float(row["SMA_S"]) if pd.notna(row["SMA_S"]) else None
        sma_l = float(row["SMA_L"]) if pd.notna(row["SMA_L"]) else None
        atr   = float(row["ATR"]) if pd.notna(row.get("ATR", np.nan)) else None

        if self.shares > 0:
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            self.holding_bars += 1

        # Entry: SMA cross AND (optional) ML gate
        if sma_s is not None and sma_l is not None:
            if self.shares == 0 and sma_s > sma_l:
                if self._allow_buy_by_model(self.current_step):
                    self._buy(price, atr, date)

            # Exit logic if in position
            if self.shares > 0:
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
# Main
# ============================

def main():
    if pdr is None and DATA_PROVIDER.lower() == 'stooq':
        print("⚠️ pandas-datareader not installed; run: pip install pandas-datareader")

    print("🚀 Rule-Based Trading System\n" + "="*50 + "\n")
    print("🔍 Fetching top-performing stocks from S&P 500...")
    top_tickers = get_top_performing_stocks_ytd(n=N_TOP_TICKERS)
    print("📈 Selected tickers:", ", ".join(top_tickers), "\n")

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

    print("🔄 Running rule-based backtest...\n")
    capital_per_stock = INITIAL_BALANCE / max(len(top_tickers), 1)
    combined_value = 0.0
    buy_hold_value = 0.0

    for ticker in top_tickers:
        print(f"▶ {ticker}: preparing data...")
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = load_prices(ticker, bt_start, bt_end)
                if df.empty or len(df) < STRAT_SMA_LONG + 5:
                    print(f"  ⚠️ Not enough data for backtest. Skipping {ticker}.")
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
        if df is None or df.empty or len(df) < STRAT_SMA_LONG + 5:
            print(f"  ⚠️ Skipping {ticker}: Insufficient or invalid data.")
            continue

        # Ensure no NaN values in critical columns
        if df["Close"].isna().any():
            print(f"  ⚠️ Skipping {ticker}: 'Close' column contains NaN values.")
            continue

        model = models.get(ticker) if USE_MODEL_GATE else None
        scaler = scalers.get(ticker) if USE_MODEL_GATE else None
        env = RuleTradingEnv(df, initial_balance=capital_per_stock, transaction_cost=TRANSACTION_COST,
                             model=model, scaler=scaler, min_proba=MIN_PROBA_UP, use_gate=USE_MODEL_GATE)
        final_val, log = env.run()
        strategy_history = env.portfolio_history

        # Buy & Hold baseline
        start_price = float(df["Close"].iloc[0])
        shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
        cash_bh = capital_per_stock - shares_bh * start_price
        buy_hold_history = (cash_bh + shares_bh * df["Close"]).tolist()
        bh_val = buy_hold_history[-1]

        print(f"  ✅ Final strategy value: ${final_val:,.2f} | Buy&Hold: ${bh_val:,.2f}")
        analyze_performance(log, strategy_history, buy_hold_history, ticker)

        combined_value += final_val
        buy_hold_value += bh_val

    if combined_value > 0:
        print("\n💰 Final Combined Portfolio Value: ${:,.2f}".format(combined_value))
        print("💰 Final Buy-and-Hold Portfolio Value: ${:,.2f}".format(buy_hold_value))
    
        if SAVE_PLOTS:
            _ensure_dir(Path("plots"))
            fig = plt.figure(figsize=(8, 4))
            plt.title("Combined Portfolio (placeholder)")
            plt.plot([0, 1], [INITIAL_BALANCE, combined_value])
            plt.xlabel("Period"); plt.ylabel("Value ($)")
            fig.savefig("plots/combined_portfolio.png")
            plt.close(fig)
    
        return combined_value, buy_hold_value, models
    
if __name__ == "__main__":
    main()
