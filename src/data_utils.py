# data_utils.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path # Import Path for _ensure_dir
from utils import _to_utc
import yfinance as yf
import time
import warnings
import logging

# Suppress yfinance warnings for delisted tickers
warnings.filterwarnings('ignore', message='.*possibly delisted.*')
warnings.filterwarnings('ignore', message='.*no timezone found.*')
warnings.filterwarnings('ignore', message='.*Failed download.*')
logging.getLogger('yfinance').setLevel(logging.ERROR)

# Suppress pandas FutureWarnings that yfinance might trigger
warnings.filterwarnings('ignore', category=FutureWarning)

# Context manager to suppress yfinance stderr output
import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_yfinance_output():
    """Context manager to suppress yfinance stderr output for delisted tickers."""
    # Save original stderr
    original_stderr = sys.stderr
    try:
        # Create a null file descriptor
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            yield
    finally:
        # Restore original stderr
        sys.stderr = original_stderr

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

# Resolve cache directory relative to repo root (not current working directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESOLVED_DATA_CACHE_DIR = DATA_CACHE_DIR if Path(DATA_CACHE_DIR).is_absolute() else (_REPO_ROOT / DATA_CACHE_DIR)
_MISSING_TICKER_CACHE_DIR = _RESOLVED_DATA_CACHE_DIR / "_missing"
_MISSING_TICKER_RETRY_HOURS = 24

# Optional Stooq provider
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# Import data fetching functions from data_fetcher (single source of truth)
try:
    from data_fetcher import _fetch_from_alpaca, _fetch_from_twelvedata, _fetch_from_stooq
except ImportError:
    # Define stubs if data_fetcher not available
    def _fetch_from_alpaca(ticker, start, end):
        return pd.DataFrame()
    def _fetch_from_twelvedata(ticker, start, end, api_key=None):
        return pd.DataFrame()
    def _fetch_from_stooq(ticker, start, end):
        return pd.DataFrame()

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
        "Momentum_3d", "Momentum_5d", "Momentum_10d", "Momentum_20d", "Momentum_40d", "Momentum_63d", "Momentum_126d", "Dist_From_SMA10",
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

        # --- MOMENTUM FEATURES (CRITICAL FOR PERFORMANCE) ---
        # Calculate momentum at different timeframes (percentage returns)
        df['Momentum_3d'] = (close.pct_change(3) * 100).fillna(0)
        df['Momentum_5d'] = (close.pct_change(5) * 100).fillna(0)
        df['Momentum_10d'] = (close.pct_change(10) * 100).fillna(0)
        df['Momentum_20d'] = (close.pct_change(20) * 100).fillna(0)
        df['Momentum_40d'] = (close.pct_change(40) * 100).fillna(0)
        df['Momentum_63d'] = (close.pct_change(63) * 100).fillna(0)  # 3-month momentum
        df['Momentum_126d'] = (close.pct_change(126) * 100).fillna(0)  # 6-month momentum

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
        print(f"  [WARN] Could not fetch income statement for {ticker}: {e}")

    try:
        balance_sheet = yf_ticker.quarterly_balance_sheet
        if not balance_sheet.empty:
            metrics = ['Total Assets', 'Total Liabilities']
            for metric in metrics:
                if metric in balance_sheet.index:
                    financial_data[metric] = balance_sheet.loc[metric]
    except Exception as e:
        print(f"  [WARN] Could not fetch balance sheet for {ticker}: {e}")

    try:
        cash_flow = yf_ticker.quarterly_cash_flow
        if not cash_flow.empty:
            metrics = ['Free Cash Flow']
            for metric in metrics:
                if metric in cash_flow.index:
                    financial_data[metric] = cash_flow.loc[metric]
    except Exception as e:
        print(f"  [WARN] Could not fetch cash flow for {ticker}: {e}")

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
# DATA_INTERVAL is imported from config, not hardcoded here
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
# Import from config
from config import PERIOD_HORIZONS
CLASS_HORIZON = PERIOD_HORIZONS.get("1-Year", 10)  # Default to 10 days
SEQUENCE_LENGTH = 32
PYTORCH_AVAILABLE = False
CUDA_AVAILABLE = False
SHAP_AVAILABLE = False
SAVE_PLOTS = False
SEED = 42


# _ensure_dir moved to utils.py to avoid duplication
from utils import _ensure_dir

def _is_market_day_complete(date):
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


def _get_market_trading_days(start_date, end_date, market='US'):
    """
    Get trading days for a specific market using pandas_market_calendars.

    Args:
        start_date: Start date for trading days
        end_date: End date for trading days
        market: Market identifier ('US', 'DE', 'UK', 'CA')

    Returns:
        List of trading days (pandas Timestamps)
    """
    try:
        import pandas_market_calendars as mcal

        # Map market to calendar
        calendar_map = {
            'US': 'NYSE',      # US stocks (NASDAQ, NYSE, etc.)
            'DE': 'XETR',      # German stocks (DAX)
            'UK': 'LSE',       # UK stocks
            'CA': 'TSX'        # Canadian stocks
        }

        if market not in calendar_map:
            # Fallback to simple weekday filter
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            return [d for d in date_range if d.weekday() < 5]

        calendar = mcal.get_calendar(calendar_map[market])
        trading_days = calendar.valid_days(start_date=start_date, end_date=end_date)
        return list(trading_days)

    except ImportError:
        # Fallback to simple weekday filter
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return [d for d in date_range if d.weekday() < 5]


def _get_last_trading_day(ticker_symbol=None):
    """
    Get the last trading day (excludes weekends and market-specific holidays).

    Args:
        ticker_symbol: Optional ticker to determine market for market-specific holidays
    """
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()

    # Determine market based on ticker suffix
    market = 'US'  # Default
    if ticker_symbol:
        if ticker_symbol.endswith('.DE') or ticker_symbol.endswith('.F'):
            market = 'DE'
        elif ticker_symbol.endswith('.L'):
            market = 'UK'
        elif ticker_symbol.endswith(('.TO', '.V')):
            market = 'CA'

    # Get trading days for the last 7 days
    start_date = today - timedelta(days=7)
    end_date = today

    trading_days = _get_market_trading_days(start_date, end_date, market)

    # Find the most recent trading day
    if trading_days:
        # Convert to date objects for comparison
        trading_dates = [d.date() for d in trading_days]
        # Filter out future dates (today if market hasn't closed yet)
        past_trading_days = [d for d in trading_dates if d <= today]
        if past_trading_days:
            return max(past_trading_days)

    # Fallback to simple logic if no trading days found
    weekday = today.weekday()
    if weekday == 5:  # Saturday
        return today - timedelta(days=1)  # Friday
    elif weekday == 6:  # Sunday
        return today - timedelta(days=2)  # Friday
    elif weekday == 0:  # Monday
        if now_utc.hour < 23:  # Before 11pm UTC
            return today - timedelta(days=3)  # Friday
        else:
            return today
    else:  # Tuesday-Friday
        if now_utc.hour < 23:  # Before 11pm UTC
            return today - timedelta(days=1)  # Yesterday
        else:
            return today


def _get_last_trading_day_old():
    """
    Get the last trading day (excludes weekends and US holidays).
    For US markets: Mon-Fri are trading days except holidays.
    """
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    weekday = today.weekday()

    # Use pandas US Federal Holiday Calendar
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        # Create calendar for current year
        cal = USFederalHolidayCalendar()
        # Get holidays for current year
        us_holidays = cal.holidays(start=f'{today.year}-01-01', end=f'{today.year}-12-31')
        # Convert to set for fast lookup
        holiday_set = set(us_holidays.date)

        def is_us_holiday(date):
            return date in holiday_set
    except ImportError:
        # Fallback to basic check if pandas not available
        def is_us_holiday(date):
            # Presidents' Day (third Monday in February)
            if date.month == 2 and date.weekday() == 0:  # Monday in February
                first_day = date.replace(day=1)
                first_monday = first_day + timedelta(days=(7 - first_day.weekday()) % 7)
                if first_monday.day <= date.day <= first_monday.day + 14:
                    return True
            return False

    # If today is Saturday (5), last trading day was Friday (1 day ago)
    # If today is Sunday (6), last trading day was Friday (2 days ago)
    # If today is Monday-Friday before market close, last trading day was previous weekday

    if weekday == 5:  # Saturday
        return today - timedelta(days=1)  # Friday
    elif weekday == 6:  # Sunday
        return today - timedelta(days=2)  # Friday
    elif weekday == 0:  # Monday
        # Check if today is a holiday
        if is_us_holiday(today):
            return today - timedelta(days=3)  # Friday
        # Check if market data for today is available yet
        elif now_utc.hour < 23:  # Before 11pm UTC
            return today - timedelta(days=3)  # Friday
        else:
            return today
    else:  # Tuesday-Friday
        # Check if today is a holiday
        if is_us_holiday(today):
            # Find the last non-holiday weekday
            days_back = 1
            while days_back <= 7:  # Don't loop forever
                check_date = today - timedelta(days=days_back)
                if check_date.weekday() < 5 and not is_us_holiday(check_date):  # Weekday and not holiday
                    return check_date
                days_back += 1
        # Check if yesterday was a holiday
        yesterday = today - timedelta(days=1)
        if is_us_holiday(yesterday):
            # Find the last non-holiday weekday
            days_back = 2
            while days_back <= 7:  # Don't loop forever
                check_date = today - timedelta(days=days_back)
                if check_date.weekday() < 5 and not is_us_holiday(check_date):  # Weekday and not holiday
                    return check_date
                days_back += 1
        elif now_utc.hour < 23:  # Before 11pm UTC
            return today - timedelta(days=1)  # Yesterday
        else:
            return today


def _is_cache_current(last_cached_date, ticker_symbol=None):
    """
    Check if cache is current (has data up to the last trading day).

    Args:
        last_cached_date: The date of the last cached data
        ticker_symbol: Optional ticker symbol to determine exchange/market
    """
    last_trading_day = _get_last_trading_day(ticker_symbol)

    # Convert cached_date to proper date object
    try:
        if isinstance(last_cached_date, datetime):
            cached_date = last_cached_date.date()
        elif hasattr(last_cached_date, 'date'):
            cached_date = last_cached_date.date()
        else:
            # Handle pandas Timestamp or other formats
            import pandas as pd
            if isinstance(last_cached_date, pd.Timestamp):
                cached_date = last_cached_date.date()
            else:
                # Convert from string or other format
                cached_date = pd.to_datetime(last_cached_date).date()
    except Exception as e:
        print(f"  [DEBUG] Cannot convert cached_date {last_cached_date} (type: {type(last_cached_date)}) to date: {e}")
        # Simple fallback for type errors
        try:
            import pandas as pd
            cached_date = pd.to_datetime(last_cached_date).date()
        except:
            return False

    # Check if cache has data up to or after last trading day
    is_current = cached_date >= last_trading_day
    print(f"  [DEBUG] {ticker_symbol}: Cache check {cached_date} >= {last_trading_day} = {is_current}")
    return is_current

# CLASS_HORIZON is now imported from config above
def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download and clean data with INCREMENTAL local caching.
    Thread-safe for parallel downloads.

    - If cache exists, only fetches NEW data since the last cached date
    - Appends new data to existing cache file
    - Much faster on subsequent runs
    """
    import threading

    # Global cache lock for thread safety
    if not hasattr(load_prices, '_cache_lock'):
        load_prices._cache_lock = threading.Lock()

    _ensure_dir(_RESOLVED_DATA_CACHE_DIR)
    _ensure_dir(_MISSING_TICKER_CACHE_DIR)
    cache_file = _RESOLVED_DATA_CACHE_DIR / f"{ticker}.csv"
    missing_marker_file = _MISSING_TICKER_CACHE_DIR / f"{ticker}.txt"

    cached_df = pd.DataFrame()
    new_df = pd.DataFrame()
    fetch_start = None
    fetch_end = datetime.now(timezone.utc)
    needs_fetch = True

    # --- Step 1: Check existing cache and determine what to fetch ---
    with load_prices._cache_lock:
        # If we recently failed to fetch this ticker and have no local cache, skip re-fetch for a while
        if not cache_file.exists() and missing_marker_file.exists():
            try:
                marker_mtime = datetime.fromtimestamp(missing_marker_file.stat().st_mtime, tz=timezone.utc)
                marker_age_hours = (datetime.now(timezone.utc) - marker_mtime).total_seconds() / 3600
                if marker_age_hours < _MISSING_TICKER_RETRY_HOURS:
                    return pd.DataFrame()
            except Exception:
                pass

        if cache_file.exists():
            try:
                # Check if cache has 'Date' or 'Datetime' column, or use first column as index
                temp_df = pd.read_csv(cache_file, nrows=1)
                if 'Date' in temp_df.columns:
                    index_col = 'Date'
                elif 'Datetime' in temp_df.columns:
                    index_col = 'Datetime'
                else:
                    index_col = 0  # Use first column (unnamed index)
                cached_df = pd.read_csv(cache_file, index_col=index_col, parse_dates=True)
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')

                # Clean up tuple columns from yfinance MultiIndex (e.g., "('close', 'aapl')")
                tuple_cols = [c for c in cached_df.columns if c.startswith("('")]
                if tuple_cols:
                    cached_df = cached_df.drop(columns=tuple_cols)

                # Drop rows where core OHLCV data is missing (bad cache entries)
                if 'Close' in cached_df.columns:
                    cached_df = cached_df.dropna(subset=['Close'])

                cached_df = cached_df.sort_index()

                if not cached_df.empty:
                    last_cached_date = cached_df.index[-1]
                    # Debug: Show cache file info for specific tickers
                    if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                        print(f"  [INFO] Cache {ticker}: shape={cached_df.shape}, Close[0]={cached_df['Close'].iloc[0]:.2f}, Close[-1]={cached_df['Close'].iloc[-1]:.2f}")

                    # [PASS] FIX: Use proper trading day check to avoid fetching on weekends
                    is_current = _is_cache_current(last_cached_date, ticker)
                    # Log cache status (use tqdm.write to avoid progress bar conflicts)
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"  [DEBUG] {ticker}: cache={last_cached_date.date()}, current={is_current}")
                    except:
                        print(f"  [DEBUG] {ticker}: cache={last_cached_date.date()}, current={is_current}")
                    if is_current:
                        # Cache already has data up to the last trading day
                        needs_fetch = False
                    else:
                        # Need to fetch data from last cached date + 1
                        fetch_start = last_cached_date + timedelta(days=1)

            except Exception as e:
                print(f"  Warning: Could not read cache for {ticker}: {e}. Will refetch all.")
                cached_df = pd.DataFrame()

    # If no cache exists, fetch historical data from the requested start date
    if cached_df.empty:
        # Use the start date passed to the function (respects the requested range)
        fetch_start = start

    # --- Step 2: Fetch new data if needed ---
    new_df = pd.DataFrame()
    if needs_fetch and fetch_start is not None:
        if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
            print(f"  [DEBUG] Fetching new data for {ticker} from {fetch_start.date()}")
        start_utc = _to_utc(fetch_start)
        end_utc = _to_utc(fetch_end)

        days_to_fetch = (fetch_end - fetch_start).days
        if days_to_fetch > 5:
            # Always show hours for 1h data - DATA_INTERVAL should be '1h' from config
            hours_to_fetch = days_to_fetch * 24
            print(f"  Fetching {hours_to_fetch} hours of data for {ticker}... (DATA_INTERVAL={DATA_INTERVAL})")
        elif days_to_fetch > 0:
            hours_to_fetch = days_to_fetch * 24
            print(f"  Updating {ticker} (+{hours_to_fetch} hours)... (DATA_INTERVAL={DATA_INTERVAL})")

        # Track which providers we tried
        providers_tried = []

        # [PASS] FIX: Try providers in order: Alpaca → TwelveData → Yahoo (cascade fallback)
        # Try Alpaca first (if available)
        if new_df.empty and ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            providers_tried.append("Alpaca")
            try:
                new_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
                if not new_df.empty:
                    print(f"  [SUCCESS] {ticker}: Got {len(new_df)} rows from Alpaca")
            except Exception as e:
                print(f"  [ERROR] {ticker}: Alpaca failed - {str(e)[:100]}")

        # Try TwelveData second (if available)
        if new_df.empty and TWELVEDATA_SDK_AVAILABLE and TWELVEDATA_API_KEY:
            providers_tried.append("TwelveData")
            try:
                new_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
                if not new_df.empty:
                    print(f"  [SUCCESS] {ticker}: Got {len(new_df)} rows from TwelveData")
            except Exception as e:
                print(f"  [ERROR] {ticker}: TwelveData failed - {str(e)[:100]}")

        # Yahoo as final fallback (always try if others fail)
        if new_df.empty:
            providers_tried.append("Yahoo")
            try:
                # Suppress yfinance stderr output for delisted tickers
                with suppress_yfinance_output():
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc,
                                               interval=DATA_INTERVAL, auto_adjust=True, progress=False,
                                               multi_level_index=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    new_df = downloaded_df.dropna()
                else:
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"  [ERROR] {ticker}: Yahoo returned empty data")
                    except:
                        print(f"  [ERROR] {ticker}: Yahoo returned empty data")
            except Exception as e:
                try:
                    from tqdm import tqdm
                    tqdm.write(f"  [ERROR] {ticker}: Yahoo failed - {str(e)[:100]}")
                except:
                    print(f"  [ERROR] {ticker}: Yahoo failed - {str(e)[:100]}")

        # If all providers failed, silently use stale cache (reduces log noise)

        # No fallback to daily data - use only the configured DATA_INTERVAL
        if new_df.empty:
            print(f"  [WARNING] {ticker}: No {DATA_INTERVAL} data available")
    else:
        print(f"  [DEBUG] Skipping fetch for {ticker} - cache is current")

    # Clean up new data (only if we fetched anything)
    if not new_df.empty:
        # Handle MultiIndex columns from yfinance (e.g., ('Close', 'AAPL') -> 'Close')
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = [col[0] if isinstance(col, tuple) else col for col in new_df.columns]
        new_df.columns = [str(col).capitalize() for col in new_df.columns]
        if "Close" not in new_df.columns and "Adj close" in new_df.columns:
            new_df = new_df.rename(columns={"Adj close": "Close"})
        if "Volume" in new_df.columns:
            new_df.loc[:, "Volume"] = new_df["Volume"].fillna(0).astype(int)
        else:
            new_df.loc[:, "Volume"] = 0

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
        # Persist a short-lived marker for unavailable/delisted tickers to avoid re-fetching every run
        if cached_df.empty:
            with load_prices._cache_lock:
                try:
                    missing_marker_file.write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
                except Exception:
                    pass
        return pd.DataFrame()

    # --- Step 4: Save updated cache (always save original 1h data) ---
    if needs_fetch and not new_df.empty:
        with load_prices._cache_lock:
            try:
                # Debug: Show cache save info for specific tickers
                if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                    print(f"  💾 Saving {ticker}: shape={price_df.shape}, Close[0]={price_df['Close'].iloc[0]:.2f}, Close[-1]={price_df['Close'].iloc[-1]:.2f}")
                price_df.to_csv(cache_file)
            except Exception as e:
                print(f"  Warning: Could not save cache for {ticker}: {e}")

    # Clear missing-marker once we have valid price data
    if not price_df.empty and missing_marker_file.exists():
        with load_prices._cache_lock:
            try:
                missing_marker_file.unlink(missing_ok=True)
            except Exception:
                pass

    # --- Step 5: Convert 1h data to daily data if DATA_INTERVAL is 1h ---
    if DATA_INTERVAL == '1h' and not price_df.empty:
        try:
            # --- Step 5.1: Calculate Core 1h Features before aggregation ---
            # Calculate intraday returns
            price_df['Hourly_Return'] = price_df['Close'].pct_change()

            # Intraday volatility patterns
            price_df['Intraday_Range_Pct'] = (price_df['High'] - price_df['Low']) / price_df['Open'] * 100
            price_df['Hourly_Volatility'] = price_df['Hourly_Return'].rolling(5).std()
            price_df['Intraday_Vol_Ratio'] = price_df['Hourly_Volatility'] / price_df['Hourly_Return'].rolling(20).std()

            # Volume distribution analysis
            price_df['Volume_Weighted_Price'] = (price_df['Volume'] * price_df['Close']).cumsum() / price_df['Volume'].cumsum()
            price_df['Volume_Concentration'] = price_df['Volume'] / price_df['Volume'].rolling(8).sum()
            price_df['Buying_Pressure'] = (price_df['Close'] > price_df['Open']).astype(int) * price_df['Volume']
            price_df['Selling_Pressure'] = (price_df['Close'] < price_df['Open']).astype(int) * price_df['Volume']

            # Price discovery metrics
            price_df['Price_Discovery_Efficiency'] = abs(price_df['Close'] - price_df['Volume_Weighted_Price']) / price_df['Volume_Weighted_Price'] * 100
            price_df['Information_Shock'] = abs(price_df['Hourly_Return']).rolling(20).mean() * price_df['Volume_Concentration']

            # Intraday momentum cycles
            price_df['Hourly_Direction'] = (price_df['Hourly_Return'] > 0).astype(int)
            price_df['Momentum_Persistence'] = price_df['Hourly_Direction'].rolling(4).sum()
            price_df['Intraday_Trend_Strength'] = price_df['Close'].rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)

            # Market microstructure features
            price_df['Liquidity_Detection'] = price_df['Volume'] / price_df['Intraday_Range_Pct']
            price_df['Net_Order_Flow'] = (price_df['Buying_Pressure'] - price_df['Selling_Pressure']) / price_df['Volume']
            price_df['Market_Impact'] = abs(price_df['Hourly_Return']) / price_df['Volume']

            # Volatility clustering and regime detection
            price_df['Vol_Clustering'] = (price_df['Hourly_Volatility'] > price_df['Hourly_Volatility'].rolling(20).mean()).astype(int)
            price_df['Vol_Regime_Change'] = price_df['Hourly_Volatility'].rolling(8).std() / price_df['Hourly_Volatility'].rolling(20).std()

            # Resample 1h data to daily data with enhanced features
            # Use 'B' for business days to preserve trading day alignment
            daily_df = price_df.resample('B').agg({
                # Basic OHLCV
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                # 1h Features - aggregated appropriately
                'Intraday_Range_Pct': 'mean',
                'Hourly_Volatility': 'mean',
                'Intraday_Vol_Ratio': 'mean',
                'Volume_Weighted_Price': 'last',
                'Volume_Concentration': 'mean',
                'Buying_Pressure': 'sum',
                'Selling_Pressure': 'sum',
                'Price_Discovery_Efficiency': 'mean',
                'Information_Shock': 'sum',
                'Momentum_Persistence': 'mean',
                'Intraday_Trend_Strength': 'mean',
                'Liquidity_Detection': 'mean',
                'Net_Order_Flow': 'mean',
                'Market_Impact': 'mean',
                'Vol_Clustering': 'sum',
                'Vol_Regime_Change': 'mean'
            }).dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])  # Only drop if core OHLCV is missing

            # Calculate additional daily features from 1h aggregates
            daily_df['Net_Buying_Pressure'] = daily_df['Buying_Pressure'] - daily_df['Selling_Pressure']
            daily_df['Buying_Pressure_Ratio'] = daily_df['Buying_Pressure'] / (daily_df['Buying_Pressure'] + daily_df['Selling_Pressure'])
            daily_df['Total_Order_Flow'] = daily_df['Buying_Pressure'] + daily_df['Selling_Pressure']

            # Calculate technical indicators for AI training features
            daily_df = _calculate_technical_indicators(daily_df)

            # Debug: Show conversion info for specific tickers
            if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                original_features = len(price_df.columns)
                daily_features = len(daily_df.columns)
                start_date = daily_df.index[0].strftime('%Y-%m-%d')
                end_date = daily_df.index[-1].strftime('%Y-%m-%d')
                print(f"  [INFO] Converted {ticker}: 1h ({price_df.shape[0]} rows, {original_features} features) -> daily ({daily_df.shape[0]} rows, {daily_features} features)")
                print(f"     [INFO] Date range: {start_date} to {end_date} ({daily_df.shape[0]} trading days)")

            price_df = daily_df
        except Exception as e:
            print(f"  Warning: Could not convert {ticker} from 1h to daily: {e}")
            # Fall back to original 1h data if conversion fails

    # --- Step 6: Return filtered data for requested range ---
    result = price_df.loc[(price_df.index >= _to_utc(start)) & (price_df.index <= _to_utc(end))].copy()
    return result if not result.empty else pd.DataFrame()



def _download_batch_robust(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download multiple tickers with incremental caching.
    Returns DataFrame in long format (no MultiIndex).
    """
    all_data_frames = []

    print(f"  [INFO] Processing {len(tickers)} tickers with incremental caching...")

    # Add progress bar
    try:
        from tqdm import tqdm
        ticker_iterator = tqdm(tickers, desc="  [INFO] Downloading tickers", ncols=100)
    except ImportError:
        # Fallback to no progress bar if tqdm not available
        ticker_iterator = tickers
        print("  [INFO] Note: Install tqdm for progress bars: pip install tqdm")

    for ticker in ticker_iterator:
        try:
            df = load_prices(ticker, start, end)
            if not df.empty:
                # Add ticker column and keep in long format
                df = df.copy()
                df['ticker'] = ticker
                # Reset index to make date a column, and standardize name to 'date'
                df = df.reset_index()
                # Rename index column to 'date' for consistency
                # Handle various possible names: 'Date', 'Datetime', 'index', or unnamed (level_0)
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'date'})
                elif 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'date'})
                elif 'index' in df.columns:
                    df = df.rename(columns={'index': 'date'})
                elif 'level_0' in df.columns:
                    df = df.rename(columns={'level_0': 'date'})
                all_data_frames.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to download {ticker}: {e}")

    if all_data_frames:
        # Concatenate all tickers into long format
        combined_df = pd.concat(all_data_frames, axis=0, ignore_index=True)
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
                print(f"  [INFO] Skipping {ticker}: Data not available (possibly delisted).")
                return pd.DataFrame()

            # Handle rate limiting with exponential backoff
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f"  [WARN] Rate limited trying to fetch {ticker}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # For other unexpected errors, log it and fail for this ticker
                print(f"  [WARN] An unexpected error occurred for {ticker}: {e}. Skipping.")
                return pd.DataFrame()

    print(f"  [FAIL] Failed to load data for {ticker} after {max_retries} retries due to persistent rate limiting.")
    return pd.DataFrame()


def fetch_training_data(ticker: str, data: pd.DataFrame, class_horizon: int = CLASS_HORIZON, train_start: pd.Timestamp = None, train_end: pd.Timestamp = None) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ML features from a given DataFrame.

    Args:
        ticker: Stock ticker symbol
        data: DataFrame with OHLCV data (includes all available data for TargetReturn calculation)
        class_horizon: Number of calendar days for forward return calculation
        train_start: Optional start date - only rows from this date onwards will be kept for training
        train_end: Optional cutoff date - only rows up to this date will be kept for training
                   (after TargetReturn is calculated using all available data)
    """
    print(f"  [DIAGNOSTIC] {ticker}: fetch_training_data - Initial data rows: {len(data)}")
    if ticker in ['AAPL', 'WDC', 'SNDK']:
        print(f"  [DEBUG] {ticker}: Input columns: {list(data.columns)[:10]}")
    if data.empty or len(data) < FEAT_SMA_LONG + 10:
        print(f"  [DIAGNOSTIC] {ticker}: Skipping feature prep. Initial data has {len(data)} rows, required > {FEAT_SMA_LONG + 10}.")
        return pd.DataFrame(), []

    df = data.copy()

    # [PASS] FIX: Handle long format data (with 'date' and 'ticker' columns)
    if 'date' in df.columns and 'ticker' in df.columns:
        # Convert from long format to wide format
        df = df.set_index('date')
        # Remove ticker column since we know which ticker this is
        if 'ticker' in df.columns:
            df = df.drop('ticker', axis=1)
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

    # Log Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Returns'] = df['Log_Returns'].fillna(0)

    # Historical Volatility (rolling standard deviation of returns)
    df['Historical_Volatility'] = df['Returns'].rolling(window=FEAT_VOL_WINDOW).std()
    df['Historical_Volatility'] = df['Historical_Volatility'].fillna(0)

    # Volatility-Adjusted Momentum: momentum normalized by volatility (Sharpe-like)
    mom_20 = df['Close'].pct_change(20).fillna(0)
    vol_20 = df['Returns'].rolling(20, min_periods=5).std().fillna(0.01)
    df['Vol_Adjusted_Momentum'] = (mom_20 / vol_20.replace(0, 0.01)).clip(-10, 10).fillna(0)

    # Mean Reversion Signal: distance from 20-day mean in std devs (z-score)
    sma_20 = df['Close'].rolling(20, min_periods=5).mean()
    std_20 = df['Close'].rolling(20, min_periods=5).std().replace(0, 1)
    df['Mean_Reversion_Signal'] = ((df['Close'] - sma_20) / std_20).clip(-3, 3).fillna(0)

    # Trend Strength: ADX-like measure using price vs moving averages
    sma_10 = df['Close'].rolling(10, min_periods=5).mean()
    sma_50 = df['Close'].rolling(50, min_periods=10).mean()
    trend_alignment = ((df['Close'] > sma_10) & (sma_10 > df['SMA_F_S']) & (df['SMA_F_S'] > sma_50)).astype(float)
    df['Trend_Strength'] = trend_alignment.rolling(5, min_periods=1).mean().fillna(0)

    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]

    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Target"]     = df["Close"].shift(-1)

    # [PASS] FIX: Sort by date index before calculating forward returns
    if hasattr(df, 'index') and hasattr(df.index, 'sort_values'):
        df = df.sort_index()

    # [PASS] NEW: Date-based forward return calculation (replaces shift-based approach)
    # Calculate TargetReturn using actual calendar days instead of row-based shift
    df["TargetReturn"] = np.nan

    for idx in df.index:
        # Calculate target date: current date + class_horizon calendar days
        target_date = idx + pd.Timedelta(days=class_horizon)

        # Find the closest available price on or after target_date
        future_prices = df[df.index >= target_date]["Close"]

        if len(future_prices) > 0:
            future_price = future_prices.iloc[0]
            current_price = df.loc[idx, "Close"]
            df.loc[idx, "TargetReturn"] = (future_price / current_price - 1.0) * 100

    # Filter to only keep rows within training period (if specified)
    # This removes rows outside the training window
    if train_start is not None:
        df = df[df.index >= train_start]
        if ticker in ['SNDK', 'WDC', 'MU']:
            print(f"  [DEBUG] DEBUG {ticker}: Filtered to train_start {train_start.date()}, now {len(df)} rows")

    if train_end is not None:
        df = df[df.index <= train_end]
        if ticker in ['SNDK', 'WDC', 'MU']:
            print(f"  [DEBUG] DEBUG {ticker}: Filtered to train_end {train_end.date()}, now {len(df)} rows")

    # DEBUG: Check TargetReturn calculation
    if ticker in ['SNDK', 'WDC', 'MU']:
        print(f"  [DEBUG] DEBUG {ticker}: df shape after TargetReturn calc: {df.shape}")
        if hasattr(df, 'index') and len(df.index) > 0:
            print(f"  [DEBUG] DEBUG {ticker}: Date range: {df.index.min()} to {df.index.max()}")
        print(f"  [DEBUG] DEBUG {ticker}: TargetReturn non-NaN count: {df['TargetReturn'].notna().sum()}")
        print(f"  [DEBUG] DEBUG {ticker}: Total rows: {len(df)}")
        print(f"  [DEBUG] DEBUG {ticker}: Last 10 dates: {df.index[-10:].tolist()}")
        print(f"  [DEBUG] DEBUG {ticker}: Close tail: {df['Close'].tail(10).tolist()}")
        print(f"  [DEBUG] DEBUG {ticker}: TargetReturn tail: {df['TargetReturn'].tail(10).tolist()}")
        print(f"  [DEBUG] DEBUG {ticker}: Is index sorted: {df.index.is_monotonic_increasing}")

    # Dynamically build the list of features that are actually present in the DataFrame
    # This is the most critical part to ensure consistency

    # Define a base set of expected technical features
    # Core technical features that are ALWAYS calculated in this function
    core_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility",
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Log_Returns",
        "Historical_Volatility",
        "Vol_Adjusted_Momentum", "Mean_Reversion_Signal", "Trend_Strength"
    ]

    # Optional features that may or may not be present (merged from external sources)
    optional_features = [
        "Market_Momentum_SPY", "Sentiment_Score",
        "VIX_Index_Returns", "DXY_Index_Returns", "Gold_Futures_Returns", "Oil_Futures_Returns", "US10Y_Yield_Returns",
        "Oil_Price_Returns", "Gold_Price_Returns", "Relative_Strength_vs_SPY"
    ]

    # Combine: core features + optional features that are actually present
    expected_technical_features = core_technical_features + [f for f in optional_features if f in df.columns]

    # Filter to only include technical features that are actually in df.columns
    present_technical_features = [col for col in expected_technical_features if col in df.columns]

    # Combine with financial features
    all_present_features = present_technical_features + financial_features

    # Also include target columns for the initial DataFrame selection before dropna
    target_cols = ["Target", "TargetReturn"]
    cols_for_ready = all_present_features + target_cols

    # Filter cols_for_ready to ensure all are actually in df.columns (redundant but safe)
    cols_for_ready_final = [col for col in cols_for_ready if col in df.columns]

    # DEBUG: Print which features are missing
    missing_features = [col for col in expected_technical_features if col not in df.columns]
    if missing_features and ticker in ['TTD', 'SOXS']:  # Debug for first few tickers
        print(f"   ↳ {ticker}: Missing features: {missing_features[:5]}...")  # Limit output

    # [PASS] FIX: Fill NaN values before dropping to preserve more rows
    # First, forward-fill and back-fill numeric columns
    ready = df[cols_for_ready_final].copy()
    for col in ready.columns:
        if col not in target_cols:  # Don't fill target columns
            ready[col] = ready[col].ffill().bfill().fillna(0)

    # [PASS] FIX: Only drop rows where the ACTIVE target is NaN
    # In regression mode (default): only TargetReturn matters
    # In classification mode: only Target matters
    # We use TargetReturn for regression, so only require that column
    active_target_col = "TargetReturn" if "TargetReturn" in ready.columns else "Target"

    # DEBUG: Check TargetReturn before dropna
    if ticker in ['SNDK', 'WDC', 'AAPL']:
        print(f"   [DEBUG] DEBUG {ticker}: cols_for_ready_final has {len(cols_for_ready_final)} cols")
        print(f"   [DEBUG] DEBUG {ticker}: 'TargetReturn' in cols_for_ready_final: {'TargetReturn' in cols_for_ready_final}")
        print(f"   [DEBUG] DEBUG {ticker}: 'TargetReturn' in df.columns: {'TargetReturn' in df.columns}")
        if active_target_col in ready.columns:
            non_nan_count = ready[active_target_col].notna().sum()
            print(f"   [DEBUG] DEBUG {ticker}: {active_target_col} has {non_nan_count} non-NaN values out of {len(ready)}")

    ready = ready.dropna(subset=[active_target_col])

    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in target_cols]

    print(f"   ↳ {ticker}: rows after features available: {len(ready)}")
    return ready, final_training_features


# Data provider functions are now imported from data_fetcher.py (single source of truth)
# See: _fetch_from_alpaca, _fetch_from_twelvedata, _fetch_from_stooq


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
                print(f"  [WARN] Failed to fetch {symbol}: {e}")

        if all_data:
            return pd.concat(all_data, axis=1)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"  [ERROR] Failed to fetch intermarket data: {e}")
        return pd.DataFrame()


def load_all_market_data(all_tickers: list, end_date: datetime = None) -> pd.DataFrame:
    """
    Load market data for all tickers.
    Shared function used by both backtesting and live trading.

    Args:
        all_tickers: List of ticker symbols to download
        end_date: End date for data (defaults to last trading day)

    Returns:
        DataFrame with all ticker data in long format (with 'ticker' and 'date' columns)
    """
    from config import DATA_INTERVAL, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES
    from utils import _to_utc
    import time

    if end_date is None:
        end_date = _get_last_trading_day()
        end_date = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    # Use shared function to determine lookback period
    from config import get_data_lookback_days
    lookback_days = get_data_lookback_days()
    start_date = end_date - timedelta(days=lookback_days)
    days_back = (end_date - start_date).days

    print(f"🚀 Batch downloading data for {len(all_tickers)} tickers from {start_date.date()} to {end_date.date()}...")
    print(f"  (Requesting {days_back} days of data - fixed maximum range for consistency)")

    all_tickers_data_list = []

    # Add progress bar for batch downloads
    try:
        from tqdm import tqdm
        batch_iterator = range(0, len(all_tickers), BATCH_DOWNLOAD_SIZE)
        batch_iterator = tqdm(batch_iterator, desc="  [INFO] Downloading batches", ncols=100)
    except ImportError:
        batch_iterator = range(0, len(all_tickers), BATCH_DOWNLOAD_SIZE)
        print("  [INFO] Note: Install tqdm for progress bars: pip install tqdm")

    for i in batch_iterator:
        batch = all_tickers[i:i + BATCH_DOWNLOAD_SIZE]
        batch_num = i//BATCH_DOWNLOAD_SIZE + 1
        total_batches = (len(all_tickers) + BATCH_DOWNLOAD_SIZE - 1)//BATCH_DOWNLOAD_SIZE
        print(f"  - Batch {batch_num}/{total_batches}: {len(batch)} tickers")
        batch_data = _download_batch_robust(batch, start=start_date, end=end_date)
        if not batch_data.empty:
            # Ensure date column is timezone-aware
            if 'date' in batch_data.columns:
                batch_data['date'] = pd.to_datetime(batch_data['date'], utc=True)

            filtered_batch_data = batch_data[
                (batch_data['date'] >= _to_utc(start_date)) &
                (batch_data['date'] <= _to_utc(end_date))
            ]
            if not filtered_batch_data.empty:
                all_tickers_data_list.append(filtered_batch_data)

        if i + BATCH_DOWNLOAD_SIZE < len(all_tickers):
            print(f"  - Pausing for {PAUSE_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    if not all_tickers_data_list:
        print("❌ Batch download failed - no data retrieved")
        return pd.DataFrame()

    all_tickers_data = pd.concat(all_tickers_data_list, axis=0)

    if all_tickers_data.empty:
        print("❌ Batch download failed - empty data")
        return pd.DataFrame()

    print(f"   📊 Data shape: {all_tickers_data.shape}")
    print(f"   📊 Columns: {list(all_tickers_data.columns[:5])}")

    return all_tickers_data
