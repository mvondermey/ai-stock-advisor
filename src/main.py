<<<<<<< HEAD


=======
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
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff
import pandas as pd
import gym
import numpy as np

# ---------- Matplotlib (headless-safe) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

<<<<<<< HEAD
# Extend your existing TradingEnv to be gym-compatible
class RLTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000, transaction_cost=0.001):
        super(RLTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
=======
import yfinance as yf
from tqdm import tqdm
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

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
ATR_MULT_TRAIL          = 3.0
ATR_MULT_TP             = 4.0
RISK_PER_TRADE          = 0.01       # 1% of capital
TRANSACTION_COST        = 0.001      # 0.1%

# --- Feature windows (for ML only)
FEAT_SMA_SHORT          = 5
FEAT_SMA_LONG           = 20
FEAT_VOL_WINDOW         = 10
CLASS_HORIZON           = 5          # days ahead for classification target
MIN_PROBA_UP            = 0.60       # ML gate threshold
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
                new_df = yf.download(ticker, start=dl_start, end=end_utc, auto_adjust=True, progress=False).dropna()
            except Exception as e:
                print(f"⚠️ yfinance fallback failed for {ticker}: {e}")
    else:
        try:
            new_df = yf.download(ticker, start=dl_start, end=end_utc, auto_adjust=True, progress=False).dropna()
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
        new_df.index = pd.to_datetime(new_df.index, utc=True)
        new_df.index.name = "Date"
        # Drop rows with NaN values in critical columns
        new_df = new_df.dropna(subset=["Close"])
        # Fill missing values in other columns if necessary
        new_df = new_df.fillna(method="ffill").fillna(method="bfill")
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
        return df_cached.loc[(df_cached.index >= start_utc) & (df_cached.index <= end_utc)].dropna(subset=["Close"]).copy()
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
    df = df.fillna(method="ffill").fillna(method="bfill")

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

def train_predictive_model(df: pd.DataFrame):
    """Binary classifier predicting probability of +1% over next H days."""
    from sklearn.ensemble import RandomForestClassifier

    df = df.copy()
    # Ensure features exist
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

    req = ["Close","Returns","SMA_F_S","SMA_F_L","Volatility","TargetClass"]
    if any(c not in df.columns for c in req):
        print("⚠️ Missing columns for model. Skipping.")
        return None

    d = df[req].dropna()
    if len(d) < 30:
        print("⚠️ Not enough rows after feature prep to train classifier (need ≥ 30). Skipping model.")
        return None

    X = d[["Close","Returns","SMA_F_S","SMA_F_L","Volatility"]].values
    y = d["TargetClass"].values

    clf = RandomForestClassifier(n_estimators=400, random_state=SEED, class_weight="balanced", min_samples_leaf=2)
    clf.fit(X, y)
    return clf

# ============================
# Rule-based backtester (ATR & ML gate)
# ============================

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Optional ML gate to allow buys."""
    def __init__(self, df: pd.DataFrame, initial_balance: float, transaction_cost: float,
                 model=None, min_proba: float = MIN_PROBA_UP, use_gate: bool = USE_MODEL_GATE):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model = model
        self.min_proba = float(min_proba)
        self.use_gate = bool(use_gate) and (model is not None)

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
        self.df["RET_F"]   = close.pct_change(fill_method=None)
        self.df["SMA_F_S"] = close.rolling(FEAT_SMA_SHORT).mean()
        self.df["SMA_F_L"] = close.rolling(FEAT_SMA_LONG).mean()
        self.df["VOL_F"]   = self.df["RET_F"].rolling(FEAT_VOL_WINDOW).std()

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

<<<<<<< HEAD
# Train PPO agent
def train_ppo_agent(df):
    env = DummyVecEnv([lambda: RLTradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model
=======
    def _allow_buy_by_model(self, i: int) -> bool:
        if not self.use_gate:
            return True
        row = self.df.loc[i]
        feats = ["Close", "RET_F", "SMA_F_S", "SMA_F_L", "VOL_F"]
        if any(pd.isna(row.get(f)) for f in feats):
            return False
        X = np.array([[row["Close"], row["RET_F"], row["SMA_F_S"], row["SMA_F_L"], row["VOL_F"]]], dtype=float)
        try:
            proba_up = float(self.model.predict_proba(X)[0][1])
        except Exception:
            return False
        return proba_up >= self.min_proba
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

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

<<<<<<< HEAD
# Use a cross-platform interactive backend
matplotlib.use("Agg")  # Or use "Agg" for non-interactive environments

# --- Constants ---
INITIAL_BALANCE = 20000
TRANSACTION_COST = 0.0015
POSITION_SIZE = 1.0  # Fixierte Positionsgröße auf 1
BACKTEST_PERIOD = 60
STOP_LOSS = 0.2  # Increased stop-loss threshold
TAKE_PROFIT = 0.2  # Increased take-profit threshold
TRAILING_STOP = 0.02 # Default trailing stop
MIN_HOLDING_PERIOD = 10000000  # Minimum holding period in steps
DEBUG_STEPS = False  # Debug switch for step method

# Define fixed stop-loss and take-profit thresholds
FIXED_STOP_LOSS = {}
FIXED_TAKE_PROFIT = {}

# --- Trading environment ---
class RuleBasedTradingEnv:
    """Custom trading environment for rule-based trading."""
    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE, transaction_cost: float = TRANSACTION_COST, stop_loss: float = STOP_LOSS, take_profit: float = TAKE_PROFIT, trailing_stop: float = TRAILING_STOP):
        self.df = df.reset_index(drop=True)  # Ensure the DataFrame is reset and uses only the passed data
        print(f"Number of steps included in the backtest: {len(self.df)}")  # Print the number of steps
        self.current_step = 0
        self.cash = initial_balance
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.portfolio_history = [initial_balance]
        self.trade_log = []
        self.returns = []
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.dynamic_stop_loss = stop_loss
        self.dynamic_take_profit = take_profit
        self.trailing_stop_price = None  # Add a variable to track the trailing stop price
        self.holding_period = 0  # Track the holding period for shares

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash = INITIAL_BALANCE
        self.shares = 0
        self.portfolio_history = [self.cash]
        self.trade_log = []
        self.returns = []
        self.trailing_stop_price = None  # Reset the trailing stop price
        self.holding_period = 0  # Reset holding period

    def detect_market_trend(self):
        """Detect market trend using moving averages."""
        short_window = 10  # Short-term moving average window
        long_window = 30  # Long-term moving average window

        # Calculate moving averages
        self.df['SMA_Short'] = self.df['Close'].rolling(window=short_window).mean()
        self.df['SMA_Long'] = self.df['Close'].rolling(window=long_window).mean()

        # Determine market trend
        if self.df['SMA_Short'].iloc[self.current_step] > self.df['SMA_Long'].iloc[self.current_step]:
            return "uptrend"
        elif self.df['SMA_Short'].iloc[self.current_step] < self.df['SMA_Long'].iloc[self.current_step]:
            return "downtrend"
        else:
            return "sideways"

    def step(self):
        """Execute one step with dynamic adjustments based on market trends."""
        current_price = self.df["Close"].iloc[self.current_step].item()
        previous_price = (
            self.df["Close"].iloc[self.current_step - 1].item()
            if self.current_step > 0
            else current_price
        )
=======
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
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        date = self._date_at(self.current_step)
        sma_s = float(row["SMA_S"]) if pd.notna(row["SMA_S"]) else None
        sma_l = float(row["SMA_L"]) if pd.notna(row["SMA_L"]) else None
        atr   = float(row["ATR"]) if pd.notna(row.get("ATR", np.nan)) else None

<<<<<<< HEAD
        # Detect market trend
        market_trend = self.detect_market_trend()
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Market Trend: {market_trend}")

        # Debug: Print dynamic thresholds
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Adjusted Stop-Loss: {self.dynamic_stop_loss:.2%}, Adjusted Take-Profit: {self.dynamic_take_profit:.2%}")

        # Take Profit Logic
        if self.shares > 0 and self.trailing_stop_price is not None:
            take_profit_price = self.trailing_stop_price / (1 - self.dynamic_take_profit)
            if current_price >= take_profit_price:
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Take Profit Triggered! Current Price: {current_price:.2f}, Take Profit Price: {take_profit_price:.2f}")
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "SELL", current_price, self.shares))
                self.shares = 0
                self.trailing_stop_price = None
                self.holding_period = 0  # Reset holding period after selling
=======
        if self.shares > 0:
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            self.holding_bars += 1
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

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

<<<<<<< HEAD
# --- Predictive Model Functions ---
def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features to the DataFrame."""
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['Target'] = df['Close'].shift(-1)
    return df

def train_predictive_model(df: pd.DataFrame):
    """
    Train a predictive model using the provided DataFrame.
    Assumes the DataFrame already has features and has been cleaned.
    :param df: DataFrame containing features and target variable.
    :return: Trained model.
    """
    print(f"Training dataset size: {len(df)} rows")
    
    X = df[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].values
    y = df['Target'].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_next_price(model, df):
    """Predict the next day's price."""
    df_features = _add_features(df)
    latest_data = df_features[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].iloc[-1].values.reshape(1, -1)
    return model.predict(latest_data)[0]

# --- Parameter Optimization Function ---
def optimize_parameters(strategy, param_grid, X_train, y_train, cv=3, scoring='neg_mean_squared_error'):
    """
    Optimize parameters for the given strategy using GridSearchCV.
    :param strategy: The trading strategy to optimize.
    :param param_grid: Dictionary of parameters to search.
    :param X_train: Training features.
    :param y_train: Training target.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring metric for optimization.
    :return: Best parameters found by GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=strategy,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,  # Use the scoring metric passed to the function
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# --- Data fetching and preprocessing ---
def prepare_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True).dropna()
    if df.empty or len(df) < 20:
        raise ValueError(f"Insufficient data for {ticker} from {start} to {end}")
    return df

def calculate_volatility(df: pd.DataFrame) -> float:
    returns = df["Close"].pct_change().dropna()
    return returns.std().item()

def get_top_performing_stocks_ytd(sp500: bool = True, n: int = 10) -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    symbol_column = "Symbol"
    index_data = pd.read_html(url)[0]
    if symbol_column not in index_data.columns:
        raise KeyError(f"Column '{symbol_column}' not found in the table fetched from {url}")
    tickers = [symbol.replace('.', '-') for symbol in index_data[symbol_column].tolist()]
    performances = []
    end_date = datetime.today()
    start_date = datetime(end_date.year, 1, 1)
    for ticker in tqdm(tickers[:50], desc="Processing S&P 500 Tickers"):
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty or len(df) < 20 or "Close" not in df.columns:
            continue
        start_price = df["Close"].iloc[0].item()
        end_price = df["Close"].iloc[-1].item()
        growth = (end_price - start_price) / start_price
        performances.append((ticker, growth))
    top_tickers = [ticker for ticker, _ in sorted(performances, key=lambda x: x[1], reverse=True)[:n]]
    return top_tickers

def analyze_trades(trade_log: List[tuple]):
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    completed_trades = min(len(buys), len(sells))
    profits = [sells[i][2] - buys[i][2] for i in range(completed_trades)]
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    avg_profit = total_profit / len(profits) if profits else 0
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Wins: {len([p for p in profits if p > 0])}, Losses: {len([p for p in profits if p <= 0])}")
    print(f"Win Rate: {win_rate:.2%}, Avg Profit: ${avg_profit:.2f}")

def analyze_trades_per_stock(trade_log: List[tuple], ticker: str, final_price: float):
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
=======
def analyze_trades_per_stock(trade_log: List[tuple], ticker: str, final_price: float) -> Dict[str, float]:
    buys  = [t for t in trade_log if t[1] == "BUY"]
    sells = [t for t in trade_log if t[1] == "SELL"]
    profits = []
    n = min(len(buys), len(sells))
    for i in range(n):
        pb, sb = float(buys[i][2]), float(sells[i][2])
        qb, qs = float(buys[i][3]), float(sells[i][3])
        qty = min(qb, qs)
        fee_b = float(buys[i][6]) if len(buys[i])>6 else 0.0
        fee_s = float(sells[i][6]) if len(sells[i])>6 else 0.0
        profits.append((sb - pb) * qty - (fee_b + fee_s))

>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff
    if len(buys) > len(sells):
        last_buy = buys[len(sells)]
        unreal = (float(final_price) - float(last_buy[2])) * float(last_buy[3])
        profits.append(unreal)
        print(f"  Unrealized Profit for Open Position: ${unreal:.2f}")

    total_pnl = float(sum(profits))
    win_rate = (sum(1 for p in profits if p > 0) / len(profits)) if profits else 0.0
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  Wins: {sum(1 for p in profits if p > 0)}, Losses: {sum(1 for p in profits if p <= 0)}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Final Price: ${float(final_price):.2f}")
    return {"trades": n, "win_rate": win_rate, "total_pnl": total_pnl}

# ============================
# Main
# ============================

<<<<<<< HEAD
def pad_portfolio_history(portfolio_history: List[float], max_steps: int) -> List[float]:
    if len(portfolio_history) < max_steps:
        last_value = portfolio_history[-1] if portfolio_history else 0
        portfolio_history.extend([last_value] * (max_steps - len(portfolio_history)))
    return portfolio_history

def fetch_training_data(ticker: str) -> pd.DataFrame:
    """Fetch historical stock data for the last 30 days to account for rolling calculations."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)  # Fetch data for the last 30 days

    # Ensure end_date is not in the future
    if end_date > datetime.now():
        end_date = datetime.now()

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True).dropna()
    if df.empty or len(df) < 30:  # Ensure at least 30 rows for rolling features
        print(f"⚠️ Insufficient data for {ticker} from {start_date} to {end_date}. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if insufficient data

    # Calculate additional features and clean data
    df = _add_features(df)
    df = df.dropna()

    # Debug: Check if all required columns exist
    required_columns = ['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility', 'Target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Missing columns in DataFrame: {missing_columns}. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if columns are missing

    print(f"✅ Fetched {len(df)} rows of data for {ticker} from {start_date} to {end_date}.")
    return df

# --- Main function ---
=======
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff
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

    for ticker in top_tickers:
<<<<<<< HEAD
        print(f"\n🔄 Fetching and preparing training data for {ticker}...")
        training_data = fetch_training_data(ticker)
=======
        print(f"🔄 Fetching training data for {ticker}...")
        training_data = fetch_training_data(ticker, train_start, train_end)
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff
        if training_data.empty:
            print(f"⚠️ Training data for {ticker} is empty. Skipping.\n")
            continue
<<<<<<< HEAD
        print(f"✅ Training data prepared with {len(training_data)} data points for {ticker}.\n")

        print(f"🔄 Training predictive model for {ticker}...")
        model = train_predictive_model(training_data)
        models[ticker] = model
        print(f"✅ Predictive model training completed for {ticker}.\n")

    if not models:
        print("⚠️ No models were trained. Exiting program.")
        return None, None, None  # Exit early if no models were trained

    print("🔄 Starting parameter optimization...")
    # Define the custom strategy
    strategy = CustomTradingStrategy()

    # Refine parameter grid
    param_grid = {
        'STOP_LOSS': [0.05, 0.1, 0.15],
        'TAKE_PROFIT': [0.05, 0.1, 0.15],
        # Entferne POSITION_SIZE aus der Optimierung
        'TRAILING_STOP': [0.01, 0.02, 0.03, 0.04, 0.05],  # Refined range for trailing stop
    }

    # Example: Use the first stock's data for parameter optimization
    first_ticker = next(iter(models.keys()))
    optimization_data = fetch_training_data(first_ticker)
    X_opt = optimization_data[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].values
    y_opt = optimization_data['Target'].values

    best_params = optimize_parameters(
        strategy, param_grid, X_opt, y_opt, cv=3, scoring='neg_mean_squared_error'  # Use regression scoring
    )
    print(f"🔧 Optimized Parameters: {best_params}")
    
    # Use optimized parameters for the backtest
    stop_loss_opt = best_params['STOP_LOSS']
    take_profit_opt = best_params['TAKE_PROFIT']
    trailing_stop_opt = best_params['TRAILING_STOP']

    print("✅ Parameter optimization completed.\n")

    start_date = datetime.today() - timedelta(days=BACKTEST_PERIOD + 365)  # Extend backtest period by 1 year
    end_date = datetime.today()

    # Initialize combined portfolio and individual stock contributions
    combined_portfolio = [0] * (BACKTEST_PERIOD + 365)
    buy_and_hold_portfolio = [0] * (BACKTEST_PERIOD + 365)
    individual_portfolios = {}
    trade_logs = {}

    # Allocate the initial balance equally across all selected stocks
    balance_per_stock = INITIAL_BALANCE / len(top_tickers)
    for ticker in top_tickers:
        print(f"\n📈 Fetching data for {ticker}...")
        df = prepare_data(ticker, start_date, end_date)
        print(f"🔄 Running backtest for {ticker}...")
        env = RuleBasedTradingEnv(
            df, 
            initial_balance=balance_per_stock,
            stop_loss=stop_loss_opt,
            take_profit=take_profit_opt,
            trailing_stop=trailing_stop_opt
        )
        env.run()
        print(f"✅ Backtest completed for {ticker}.")
=======
        print(f"✅ Training data fetched with {len(training_data)} rows for {ticker}.")
        model = train_predictive_model(training_data)
        if model is not None:
            models[ticker] = model
            print(f"✅ Classifier trained for {ticker}.\n")
        else:
            print(f"⚠️ Skipped model for {ticker} due to insufficient data.\n")

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
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

        # Ensure no NaN values in critical columns
        if df["Close"].isna().any():
            print(f"  ⚠️ Skipping {ticker}: 'Close' column contains NaN values.")
            continue

<<<<<<< HEAD
        # Analyze trades for the current stock
        print(f"🔍 Analyzing trades for {ticker}...")
        analyze_trades_per_stock(env.trade_log, ticker, final_price=df["Close"].iloc[-1].item())
        print(f"✅ Trade analysis completed for {ticker}.")
=======
        model = models.get(ticker) if USE_MODEL_GATE else None
        env = RuleTradingEnv(df, initial_balance=capital_per_stock, transaction_cost=TRANSACTION_COST,
                             model=model, min_proba=MIN_PROBA_UP, use_gate=USE_MODEL_GATE)
        final_val, log = env.run()
        final_price = float(df["Close"].iloc[-1])
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

        # Buy & Hold baseline
        start_price = float(df["Close"].iloc[0])
        shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
        cash_bh = capital_per_stock - shares_bh * start_price
        bh_val = cash_bh + shares_bh * final_price

<<<<<<< HEAD
        # Calculate buy-and-hold portfolio value for this stock
        initial_price = df["Close"].iloc[0].item()
        shares_held = balance_per_stock / initial_price
        stock_buy_and_hold_value = [(shares_held * df["Close"].iloc[i].item()) for i in range(len(df))]
        stock_buy_and_hold_value = pad_portfolio_history(stock_buy_and_hold_value, BACKTEST_PERIOD + 365)
        buy_and_hold_portfolio = [
            buy_and_hold_portfolio[i] + stock_buy_and_hold_value[i]
            for i in range(len(buy_and_hold_portfolio))
        ]
=======
        print(f"  ✅ Final strategy value: ${final_val:,.2f} | Buy&Hold: ${bh_val:,.2f}")
        analyze_trades_per_stock(log, ticker, final_price)
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff

        combined_value += final_val
        buy_hold_value += bh_val

<<<<<<< HEAD
    # Highlight trailing stop-loss triggers and buy/sell actions
    for ticker, log in trade_logs.items():
        for step, action, price, shares in log:
            if action == "SELL":
                plt.scatter(step, combined_portfolio[step], color="red", label="Sell Action", zorder=5)
            elif action == "BUY":
                plt.scatter(step, combined_portfolio[step], color="green", label="Buy Action", zorder=5)

    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Combined Portfolio Value Over Time with Individual Contributions")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.ylim(0, max(max(combined_portfolio), max(buy_and_hold_portfolio)) * 1.1)
    plt.savefig("plots/combined_portfolio_with_individuals.png")
    plt.show()

    print("✅ Combined portfolio rendering completed.\n")
    return combined_portfolio, buy_and_hold_portfolio, models

if __name__ == "__main__":
    main_result = main()
    if main_result:
        combined_portfolio, buy_and_hold_portfolio, models = main_result
        final_combined_value = combined_portfolio[-1] if combined_portfolio else 0
        final_buy_and_hold_value = buy_and_hold_portfolio[-1] if buy_and_hold_portfolio else 0
        print(f"\n💰 Final Combined Portfolio Value: ${final_combined_value:.2f}")
        print(f"💰 Final Buy-and-Hold Portfolio Value: ${final_buy_and_hold_value:.2f}")

        # Plot the results
        if combined_portfolio and buy_and_hold_portfolio:
            plt.figure(figsize=(12, 6))
            plt.plot(combined_portfolio, label="Combined Portfolio", linewidth=2, color="black")
            plt.plot(buy_and_hold_portfolio, label="Buy-and-Hold Portfolio", linestyle="--", color="blue")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Steps")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.savefig("plots/final_portfolio_comparison.png")
            plt.show()

        # Ensure models dictionary is accessible
        if models:
            print("\n📊 Predictions for Trained Stocks:")
            for ticker, model in models.items():
                latest_data = fetch_training_data(ticker) # Fetch the latest data
                if latest_data.empty:
                    print(f"⚠️ No data available for {ticker}. Skipping prediction.")
                    continue
                
                # Ensure the dataframe passed to predict_next_price has enough data for feature calculation
                prediction = predict_next_price(model, latest_data)
                latest_close = latest_data['Close'].iloc[-1].item()
                action = "BUY" if prediction > latest_close else "SELL"
                print(f"🔮 {ticker}: Predicted Action = {action}")
        else:
            print("⚠️ No trained models found. Skipping predictions.")
    else:
        print("Program exited without generating results.")

# --- RSI Calculation ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
=======
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
>>>>>>> 7268e6c556c182b7683acc981e04cbf88771aaff
