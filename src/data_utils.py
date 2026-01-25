# data_utils.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path # Import Path for _ensure_dir
from utils import _to_utc
import yfinance as yf
import time

# Import pandas_datareader for Stooq
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# Import config values
try:
    from config import (
        PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_INTERVAL, DATA_CACHE_DIR, CACHE_DAYS,
        TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE,
        FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD
    )
except ImportError:
    # Fallback values if config is not available
    PAUSE_BETWEEN_YF_CALLS = 0.5
    DATA_PROVIDER = 'yahoo'
    USE_YAHOO_FALLBACK = True
    DATA_CACHE_DIR = Path("data_cache")
    CACHE_DAYS = 7
    TWELVEDATA_API_KEY = None
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None
    ALPACA_AVAILABLE = False
    TWELVEDATA_SDK_AVAILABLE = False
    FEAT_SMA_LONG = 20
    FEAT_SMA_SHORT = 5
    FEAT_VOL_WINDOW = 10
    ATR_PERIOD = 14

# Optional Stooq provider
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# Define technical indicators calculation function here to avoid circular imports
def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and adds them to the DataFrame."""
    close = df["Close"]
    high = df["High"] if "High" in df.columns else None
    low  = df["Low"]  if "Low" in df.columns else None
    prev_close = close.shift(1)
    
    # Initialize all new columns with 0 to prevent all-NaN rows
    new_columns = [
        "ATR", "ATR_MED", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "RSI_feat", "MACD", "MACD_signal",
        "BB_upper", "BB_lower", "%K", "%D", "ADX", "OBV", "CMF", "ROC", "ROC_20", "ROC_60", "CMO", "KAMA",
        "EFI", "KC_TR", "KC_ATR", "KC_Middle", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower", "DC_Middle",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Log_Returns",
        "Historical_Volatility", "Close_to_SMA20", "Close_to_SMA50", "Close_Position_in_Range",
        "Intraday_Range_Pct", "Open_to_Close_Ratio", "Close_to_20D_High", "Close_to_20D_Low", "Volume_Normalized",
        "Momentum_3d", "Momentum_5d", "Momentum_10d", "Momentum_20d", "Momentum_40d", "Dist_From_SMA10",
        "Dist_From_SMA20", "Dist_From_SMA50", "SMA20_Slope", "SMA50_Slope", "Price_Accel_5d", "Price_Accel_20d",
        "Vol_Regime", "Vol_Spike", "Volume_Ratio_5d", "Volume_Ratio_20d", "Volume_Trend", "Range_Expansion",
        "Range_vs_Avg", "Daily_Direction", "Streak",
        "Relative_Strength_vs_SPY", "Vol_Adjusted_Momentum", "Mean_Reversion_Signal", "Trend_Strength"
    ]
    for col in new_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    # Only calculate indicators if we have sufficient data
    if len(df) > 5:  # Minimum 5 days of data
        # ATR for risk management
        if high is not None and low is not None and len(df) > 1:
            hl = (high - low).abs()
            h_pc = (high - prev_close).abs()
            l_pc = (low  - prev_close).abs()
            tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(ATR_PERIOD, min_periods=5).mean().fillna(0)
        else:
            df["ATR"] = 0
        
        # Low-volatility filter reference: rolling median ATR
        df['ATR_MED'] = df['ATR'].rolling(50, min_periods=10).median().fillna(0)
        
        # --- Features for ML Gate ---
        df["Returns"]    = close.pct_change(fill_method=None).fillna(0)
        df["SMA_F_S"]    = close.rolling(FEAT_SMA_SHORT, min_periods=5).mean().fillna(0)
        df["SMA_F_L"]    = close.rolling(FEAT_SMA_LONG, min_periods=10).mean().fillna(0)
        df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW, min_periods=5).std().fillna(0)
        
        # RSI for features
        delta_feat = close.diff()
        gain_feat = (delta_feat.where(delta_feat > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss_feat = (-delta_feat.where(delta_feat < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs_feat = gain_feat / loss_feat
        df['RSI_feat'] = 100 - (100 / (1 + rs_feat))
        
        # MACD for features
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands for features
        bb_mid = close.rolling(window=20, min_periods=5).mean().fillna(0)
        bb_std = close.rolling(window=20, min_periods=5).std().fillna(0)
        df['BB_upper'] = (bb_mid + (bb_std * 2)).fillna(0)
        df['BB_lower'] = (bb_mid - (bb_std * 2)).fillna(0)
        
        # --- NEW FEATURES FOR AI IMPROVEMENT ---
        # Volatility-Adjusted Momentum: momentum normalized by volatility (Sharpe-like)
        mom_20 = close.pct_change(20).fillna(0)
        vol_20 = df["Returns"].rolling(20, min_periods=5).std().fillna(0.01)
        df['Vol_Adjusted_Momentum'] = (mom_20 / vol_20.replace(0, 0.01)).clip(-10, 10).fillna(0)
        
        # Mean Reversion Signal: distance from 20-day mean in std devs (z-score)
        sma_20 = close.rolling(20, min_periods=5).mean()
        std_20 = close.rolling(20, min_periods=5).std().replace(0, 1)
        df['Mean_Reversion_Signal'] = ((close - sma_20) / std_20).clip(-3, 3).fillna(0)
        
        # Trend Strength: ADX-like measure using price vs moving averages
        sma_10 = close.rolling(10, min_periods=5).mean()
        sma_50 = close.rolling(50, min_periods=10).mean()
        trend_alignment = ((close > sma_10) & (sma_10 > sma_20) & (sma_20 > sma_50)).astype(float)
        df['Trend_Strength'] = trend_alignment.rolling(5, min_periods=1).mean().fillna(0)
        
        # Relative Strength vs SPY: stock momentum minus market momentum
        # This will be calculated later when SPY data is available, initialize to 0
        df['Relative_Strength_vs_SPY'] = 0.0
    
    # Fill any remaining NaNs at the end
    df = df.fillna(0)
    return df

# Define financial data fetching functions here to avoid circular imports
def _fetch_financial_data(ticker: str) -> pd.DataFrame:
    """Fetch key financial metrics from yfinance and prepare them for merging."""
    time.sleep(PAUSE_BETWEEN_YF_CALLS)
    yf_ticker = yf.Ticker(ticker)
    
    financial_data = {}
    
    try:
        income_statement = yf_ticker.quarterly_income_stmt
        if not income_statement.empty:
            metrics = ['Total Revenue', 'Net Income', 'EBITDA']
            for metric in metrics:
                if metric in income_statement.index:
                    financial_data[metric] = income_statement.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch income statement for {ticker}: {e}")

    try:
        balance_sheet = yf_ticker.quarterly_balance_sheet
        if not balance_sheet.empty:
            metrics = ['Total Assets', 'Total Liabilities']
            for metric in metrics:
                if metric in balance_sheet.index:
                    financial_data[metric] = balance_sheet.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch balance sheet for {ticker}: {e}")

    try:
        cash_flow = yf_ticker.quarterly_cash_flow
        if not cash_flow.empty:
            metrics = ['Free Cash Flow']
            for metric in metrics:
                if metric in cash_flow.index:
                    financial_data[metric] = cash_flow.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch cash flow for {ticker}: {e}")

    if not financial_data:
        return pd.DataFrame()

    df_financial = pd.DataFrame(financial_data).T
    df_financial.index.name = 'Metric'
    df_financial = df_financial.T
    df_financial.index = pd.to_datetime(df_financial.index, utc=True)
    df_financial.index.name = "Date"
    
    df_financial = df_financial.rename(columns={
        'Total Revenue': 'Fin_Revenue',
        'Net Income': 'Fin_NetIncome',
        'Total Assets': 'Fin_TotalAssets',
        'Total Liabilities': 'Fin_TotalLiabilities',
        'Free Cash Flow': 'Fin_FreeCashFlow',
        'EBITDA': 'Fin_EBITDA'
    })
    
    for col in df_financial.columns:
        df_financial[col] = pd.to_numeric(df_financial[col], errors='coerce')

    return df_financial.sort_index()

def _fetch_financial_data_from_alpaca(ticker: str) -> pd.DataFrame:
    """Placeholder for fetching financial metrics from Alpaca."""
    return pd.DataFrame()

# Assuming these are defined in config.py or passed as arguments
# For now, hardcode or import from a config if available
# from config import DATA_CACHE_DIR, DATA_PROVIDER, USE_YAHOO_FALLBACK, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, INVESTMENT_PER_STOCK, TRANSACTION_COST, CLASS_HORIZON, SEQUENCE_LENGTH, PYTORCH_AVAILABLE, CUDA_AVAILABLE, SHAP_AVAILABLE, SAVE_PLOTS, SEED

# Placeholder for config values if not imported
DATA_CACHE_DIR = Path("data_cache")
DATA_PROVIDER = 'yahoo'
USE_YAHOO_FALLBACK = True
DATA_INTERVAL = '1d'
CACHE_DAYS = 7
TWELVEDATA_API_KEY = "YOUR_DEFAULT_KEY_OR_EMPTY_STRING"
ALPACA_API_KEY = None
ALPACA_SECRET_KEY = None
ALPACA_AVAILABLE = False
TWELVEDATA_SDK_AVAILABLE = False
FEAT_SMA_LONG = 20
FEAT_SMA_SHORT = 5
FEAT_VOL_WINDOW = 10
ATR_PERIOD = 14
INVESTMENT_PER_STOCK = 15000.0
TRANSACTION_COST = 0.001
# USE_MODEL_GATE removed - simplified buy-and-hold logic
# Probability thresholds removed - using simplified trading logic
TARGET_PERCENTAGE = 0.008
CLASS_HORIZON = 5
SEQUENCE_LENGTH = 32
PYTORCH_AVAILABLE = False
CUDA_AVAILABLE = False
SHAP_AVAILABLE = False
SAVE_PLOTS = False
SEED = 42


# _ensure_dir moved to utils.py to avoid duplication
from utils import _ensure_dir

def _is_market_day_complete(date: datetime) -> bool:
    """
    Check if we should expect data for a given date.
    Markets close at 4pm ET (9pm UTC), data available ~6pm ET (11pm UTC).
    """
    now_utc = datetime.now(timezone.utc)
    target_date = date.date() if isinstance(date, datetime) else date
    
    # If date is in the future, we can't have data yet
    if target_date > now_utc.date():
        return False
    
    # If date is today, check if market close + data processing time has passed
    if target_date == now_utc.date():
        # Market data typically available after 11pm UTC (6pm ET + 1hr processing)
        market_data_ready_hour = 23  # 11pm UTC
        if now_utc.hour < market_data_ready_hour:
            return False
    
    # If it's a weekend, no new data
    weekday = now_utc.weekday()
    if target_date == now_utc.date() and weekday >= 5:  # Saturday=5, Sunday=6
        return False
    
    # For past dates (including yesterday), data should be available
    return True

CLASS_HORIZON           = 5          # days ahead for classification target
def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download and clean data with INCREMENTAL local caching.
    
    - If cache exists, only fetches NEW data since the last cached date
    - Appends new data to existing cache file
    - Much faster on subsequent runs
    """
    _ensure_dir(DATA_CACHE_DIR)
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
    
    cached_df = pd.DataFrame()
    new_df = pd.DataFrame()
    fetch_start = None
    fetch_end = datetime.now(timezone.utc)
    needs_fetch = True
    
    # --- Step 1: Check existing cache and determine what to fetch ---
    if cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            if cached_df.index.tzinfo is None:
                cached_df.index = cached_df.index.tz_localize('UTC')
            else:
                cached_df.index = cached_df.index.tz_convert('UTC')
            
            cached_df = cached_df.sort_index()
            
            if not cached_df.empty:
                last_cached_date = cached_df.index[-1]
                today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Check if we should expect new data (considering market hours/weekends)
                if _is_market_day_complete(today):
                    # Market data should be available - check if cache is current
                    if last_cached_date < today:
                        fetch_start = last_cached_date + timedelta(days=1)
                    else:
                        needs_fetch = False
                else:
                    # Market data not available yet - check against yesterday
                    yesterday = today - timedelta(days=1)
                    if last_cached_date < yesterday:
                        fetch_start = last_cached_date + timedelta(days=1)
                    else:
                        needs_fetch = False
                        print(f"  ℹ️  {ticker}: Cache is current (today's market data not available yet)")
                    
        except Exception as e:
            print(f"  Warning: Could not read cache for {ticker}: {e}. Will refetch all.")
            cached_df = pd.DataFrame()
    
    # If no cache exists, fetch historical data
    if cached_df.empty:
        fetch_start = datetime.now(timezone.utc) - timedelta(days=1000)
    
    # --- Step 2: Fetch new data if needed ---
    if needs_fetch and fetch_start is not None:
        start_utc = _to_utc(fetch_start)
        end_utc = _to_utc(fetch_end)
        
        days_to_fetch = (fetch_end - fetch_start).days
        if days_to_fetch > 5:
            print(f"  Fetching {days_to_fetch} days of data for {ticker}...")
        elif days_to_fetch > 0:
            print(f"  Updating {ticker} (+{days_to_fetch} days)...")
        
        provider = DATA_PROVIDER.lower()
        
        # Try Alpaca first
        if provider == 'alpaca' and ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            new_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
        
        # Try TwelveData second
        if new_df.empty and provider == 'twelvedata' and TWELVEDATA_API_KEY:
            new_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
        
        # Try Stooq
        if new_df.empty and provider == 'stooq':
            new_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if new_df.empty and not ticker.upper().endswith('.US'):
                new_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
        
        # Yahoo as final fallback
        if new_df.empty and USE_YAHOO_FALLBACK:
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, 
                                           interval=DATA_INTERVAL, auto_adjust=True, progress=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    new_df = downloaded_df.dropna()
            except Exception:
                pass
        
        # Clean up new data
        if not new_df.empty:
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = new_df.columns.get_level_values(0)
            new_df.columns = [str(col).capitalize() for col in new_df.columns]
            if "Close" not in new_df.columns and "Adj close" in new_df.columns:
                new_df = new_df.rename(columns={"Adj close": "Close"})
            if "Volume" in new_df.columns:
                new_df["Volume"] = new_df["Volume"].fillna(0).astype(int)
            else:
                new_df["Volume"] = 0
            
            if new_df.index.tzinfo is None:
                new_df.index = new_df.index.tz_localize('UTC')
            else:
                new_df.index = new_df.index.tz_convert('UTC')
    
    # --- Step 3: Merge cached and new data ---
    if not cached_df.empty and not new_df.empty:
        price_df = pd.concat([cached_df, new_df])
        price_df = price_df[~price_df.index.duplicated(keep='last')]
        price_df = price_df.sort_index()
    elif not cached_df.empty:
        price_df = cached_df
    elif not new_df.empty:
        price_df = new_df
    else:
        return pd.DataFrame()
    
    # --- Step 4: Save updated cache ---
    if needs_fetch and not new_df.empty:
        try:
            price_df.to_csv(cache_file)
        except Exception as e:
            print(f"  Warning: Could not save cache for {ticker}: {e}")
    
    # --- Step 5: Return filtered data for requested range ---
    result = price_df.loc[(price_df.index >= _to_utc(start)) & (price_df.index <= _to_utc(end))].copy()
    return result if not result.empty else pd.DataFrame()



def _download_batch_robust(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download multiple tickers with incremental caching.
    Returns DataFrame with MultiIndex columns (Field, Ticker) like the original implementation.
    """
    all_data_frames = []
    
    print(f"  📂 Processing {len(tickers)} tickers with incremental caching...")
    
    for ticker in tickers:
        try:
            df = load_prices(ticker, start, end)
            if not df.empty:
                # Convert to MultiIndex format (Field, Ticker)
                df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
                all_data_frames.append(df)
        except Exception as e:
            print(f"  ⚠️ Failed to download {ticker}: {e}")
    
    if all_data_frames:
        # Concatenate all tickers into wide format with MultiIndex columns
        combined_df = pd.concat(all_data_frames, axis=1)
        return combined_df
    else:
        return pd.DataFrame()


def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """A wrapper for load_prices that handles rate limiting with retries and other common API errors."""
    import time
    import random
    max_retries = 5
    base_wait_time = 5  # seconds, increased for more tolerance

    for attempt in range(max_retries):
        try:
            return load_prices(ticker, start, end)
        except Exception as e:
            error_str = str(e).lower()
            # Handle YFTzMissingError for delisted stocks gracefully
            if "yftzmissingerror" in error_str or "no timezone found" in error_str:
                print(f"  ℹ️ Skipping {ticker}: Data not available (possibly delisted).")
                return pd.DataFrame()
            
            # Handle rate limiting with exponential backoff
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⚠️ Rate limited trying to fetch {ticker}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # For other unexpected errors, log it and fail for this ticker
                print(f"  ⚠️ An unexpected error occurred for {ticker}: {e}. Skipping.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to load data for {ticker} after {max_retries} retries due to persistent rate limiting.")
    return pd.DataFrame()


def fetch_training_data(ticker: str, data: pd.DataFrame, class_horizon: int = CLASS_HORIZON) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ML features from a given DataFrame."""
    print(f"  [DIAGNOSTIC] {ticker}: fetch_training_data - Initial data rows: {len(data)}")
    if data.empty or len(data) < FEAT_SMA_LONG + 10:
        print(f"  [DIAGNOSTIC] {ticker}: Skipping feature prep. Initial data has {len(data)} rows, required > {FEAT_SMA_LONG + 10}.")
        return pd.DataFrame(), []

    df = data.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []

    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []

    # Fill missing values in other columns
    df = df.ffill().bfill()

    df["Returns"]    = df["Close"].pct_change(fill_method=None)
    df["SMA_F_S"]    = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    df["SMA_F_L"]    = df["Close"].rolling(FEAT_SMA_LONG).mean()
    df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()

    # --- Additional Features ---
    # ATR (Average True Range)
    high = df["High"] if "High" in df.columns else None
    low  = df["Low"]  if "Low" in df.columns else None
    prev_close = df["Close"].shift(1)
    if high is not None and low is not None:
        hl = (high - low).abs()
        h_pc = (high - prev_close).abs()
        l_pc = (low  - prev_close).abs()
        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(ATR_PERIOD, min_periods=1).mean().fillna(0)
    else:
        # Fallback for ATR if High/Low are not available (though they should be after load_prices)
        ret = df["Close"].pct_change(fill_method=None)
        df["ATR"] = (ret.rolling(ATR_PERIOD, min_periods=1).std() * df["Close"]).rolling(2, min_periods=1).mean().fillna(0)
    df["ATR"] = df["ATR"].fillna(0) # Fill any NaNs from initial ATR calculation

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

    # Stochastic Oscillator
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    denominator_k = high_14 - low_14
    df['%K'] = np.where(denominator_k != 0, (df['Close'] - low_14) / denominator_k * 100, 0)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%D'].fillna(0)

    # Average Directional Index (ADX)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['+DM'] = df['+DM'].fillna(0)

    # Calculate True Range (TR)
    high_low_diff = df['High'] - df['Low']
    high_prev_close_diff_abs = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close_diff_abs = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([high_low_diff, high_prev_close_diff_abs, low_prev_close_diff_abs], axis=1).max(axis=1)
    df['TR'] = df['TR'].fillna(0)

    # Calculate Smoothed DM and TR
    alpha = 1/14
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM14'].fillna(0)
    df['-DM14'] = df['-DM14'].fillna(0)
    df['TR14'] = df['TR14'].fillna(0)

    # Calculate Directional Index (DX)
    denominator_dx = (df['+DM14'] + df['-DM14'])
    df['DX'] = np.where(denominator_dx != 0, (abs(df['+DM14'] - df['-DM14']) / denominator_dx) * 100, 0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    df['ADX'] = df['ADX'].fillna(0)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['CMF'] = df['CMF'].fillna(0)

    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # Keltner Channels
    df['KC_TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['KC_ATR'] = df['KC_TR'].rolling(window=10).mean()
    df['KC_Middle'] = df["Close"].rolling(window=20).mean()
    df['KC_Upper'] = df['KC_Middle'] + (df['KC_ATR'] * 2)
    df['KC_Lower'] = df['KC_Middle'] - (df['KC_ATR'] * 2)

    # Donchian Channels
    df['DC_Upper'] = df['High'].rolling(window=20).max()
    df['DC_Lower'] = df['Low'].rolling(window=20).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2

    # Parabolic SAR (PSAR)
    psar = df['Close'].copy()
    af = 0.02
    max_af = 0.2

    uptrend = True if df['Close'].iloc[0] > df['Open'].iloc[0] else False
    ep = df['High'].iloc[0] if uptrend else df['Low'].iloc[0]
    sar = df['Low'].iloc[0] if uptrend else df['High'].iloc[0]

    for i in range(1, len(df)):
        if uptrend:
            sar = sar + af * (ep - sar)
            if df.iloc[i, df.columns.get_loc('Low')] < sar:
                uptrend = False
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('Low')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('High')] > ep:
                    ep = df.iloc[i, df.columns.get_loc('High')]
                    af = min(max_af, af + 0.02)
        else:
            sar = sar + af * (ep - sar)
            if df.iloc[i, df.columns.get_loc('High')] > sar:
                uptrend = True
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('High')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('Low')] < ep:
                    ep = df.iloc[i, df.columns.get_loc('Low')]
                    af = min(max_af, af + 0.02)
        psar.iloc[i] = sar
    df['PSAR'] = psar

    # Accumulation/Distribution Line (ADL)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['ADL'] = mf_volume.cumsum()
    df['ADL'] = df['ADL'].fillna(0)

    # Commodity Channel Index (CCI)
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
    df['CCI'] = df['CCI'].fillna(0)

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=FEAT_VOL_WINDOW).sum() / df['Volume'].rolling(window=FEAT_VOL_WINDOW).sum()
    df['VWAP'] = df['VWAP'].fillna(df['Close'])

    # ATR Percentage
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['ATR_Pct'] = df['ATR_Pct'].fillna(0)

    # Chaikin Oscillator
    adl_fast = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    adl_slow = adl_fast.ewm(span=10, adjust=False).mean()
    adl_fast = adl_fast.ewm(span=3, adjust=False).mean()
    df['Chaikin_Oscillator'] = adl_fast - adl_slow
    df['Chaikin_Oscillator'] = df['Chaikin_Oscillator'].fillna(0)

    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfi_ratio = positive_mf.rolling(window=14).sum() / negative_mf.rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['MFI'] = df['MFI'].fillna(0)

    # OBV Moving Average
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()
    df['OBV_SMA'] = df['OBV_SMA'].fillna(0)

    # Historical Volatility (e.g., 20-day rolling standard deviation of log returns)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Historical_Volatility'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Historical_Volatility'] = df['Historical_Volatility'].fillna(0)

    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]

    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Target"]     = df["Close"].shift(-1)

    # Create forward-looking data for regression target
    fwd = df["Close"].shift(-class_horizon)

    # Regression target: X-day forward return percentage
    df["TargetReturn"] = (fwd / df["Close"] - 1.0) * 100

    # Dynamically build the list of features that are actually present in the DataFrame
    # This is the most critical part to ensure consistency

    # Define a base set of expected technical features
    expected_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility",
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Log_Returns",
        "Historical_Volatility", "Market_Momentum_SPY",
        "Sentiment_Score",
        "VIX_Index_Returns", "DXY_Index_Returns", "Gold_Futures_Returns", "Oil_Futures_Returns", "US10Y_Yield_Returns",
        "Oil_Price_Returns", "Gold_Price_Returns",
        "Relative_Strength_vs_SPY", "Vol_Adjusted_Momentum", "Mean_Reversion_Signal", "Trend_Strength"
    ]

    # Filter to only include technical features that are actually in df.columns
    present_technical_features = [col for col in expected_technical_features if col in df.columns]

    # Combine with financial features
    all_present_features = present_technical_features + financial_features

    # Also include target columns for the initial DataFrame selection before dropna
    target_cols = ["Target", "TargetReturn"]
    cols_for_ready = all_present_features + target_cols

    # Filter cols_for_ready to ensure all are actually in df.columns (redundant but safe)
    cols_for_ready_final = [col for col in cols_for_ready if col in df.columns]

    ready = df[cols_for_ready_final].dropna()

    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in target_cols]

    print(f"   ↳ {ticker}: rows after features available: {len(ready)}")
    return ready, final_training_features


# Data provider functions
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


def _fetch_from_alpaca(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV from Alpaca."""
    if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame()
    
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=_to_utc(start),
            end=_to_utc(end)
        )
        
        bars = client.get_stock_bars(request_params)
        df = bars.df
        
        if df.empty:
            return pd.DataFrame()
        
        # Reset index to get date as column
        df = df.reset_index()
        df['Date'] = df['timestamp']
        df = df.set_index('Date')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame()


def _fetch_from_twelvedata(ticker: str, start: datetime, end: datetime, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV from TwelveData using the SDK."""
    if not TWELVEDATA_SDK_AVAILABLE or not (TWELVEDATA_API_KEY or api_key):
        return pd.DataFrame()
    
    try:
        from twelvedata import TDClient
        client = TDClient(apikey=api_key or TWELVEDATA_API_KEY)
        
        # Convert to required format
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Get time series data
        ts = client.time_series(
            symbol=ticker,
            interval='1day',
            start_date=start_str,
            end_date=end_str,
            outputsize=5000
        )
        
        df = ts.as_pandas()
        
        if df.empty:
            return pd.DataFrame()
        
        # Ensure proper column names
        df.columns = [col.capitalize() for col in df.columns]
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
            df = df.set_index('Date')
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        return pd.DataFrame()


def _fetch_intermarket_data(start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """Fetch intermarket data for analysis."""
    try:
        # Define intermarket symbols
        intermarket_symbols = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ ETF', 
            'VXX': 'VIX ETF',
            'UUP': 'US Dollar ETF',
            'GLD': 'Gold ETF',
            'USO': 'Oil ETF',
            'TNX': '10Y Treasury ETF'
        }
        
        all_data = []
        today = datetime.now(timezone.utc)
        start_date = start or (today - timedelta(days=365))
        end_date = end or today
        
        for symbol, name in intermarket_symbols.items():
            try:
                df = load_prices(symbol, start_date, end_date)
                if not df.empty:
                    df_renamed = df[['Close']].copy()
                    df_renamed.columns = [f'{symbol}_Close']
                    all_data.append(df_renamed)
            except Exception as e:
                print(f"  ⚠️ Failed to fetch {symbol}: {e}")
        
        if all_data:
            return pd.concat(all_data, axis=1)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
