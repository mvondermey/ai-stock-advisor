# data_utils.py
import pandas as pd
import numpy as np
import threading
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

# Suppress yfinance logging noise (thread-safe approach)
import contextlib
import io
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress yfinance warnings globally (thread-safe, no stderr manipulation)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

@contextlib.contextmanager
def suppress_yfinance_output():
    """
    Context manager for yfinance output suppression.

    Now a no-op since we suppress via logging level globally above.
    Kept for backward compatibility with existing code that uses it.
    """
    yield

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
        FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD, AGGREGATE_TO_DAILY, NUM_PROCESSES
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
    AGGREGATE_TO_DAILY = True
    NUM_PROCESSES = 16

# Resolve cache directory relative to repo root (not current working directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESOLVED_DATA_CACHE_DIR = DATA_CACHE_DIR if Path(DATA_CACHE_DIR).is_absolute() else (_REPO_ROOT / DATA_CACHE_DIR)

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


def _fundamental_cache_artifact_path(ticker: str) -> Path:
    from strategy_disk_cache import get_cache_dir

    cache_dir = get_cache_dir("fundamental_history", {"ticker": ticker}, create=False)
    return cache_dir / "series.json"


def _serialize_fundamental_history_rows(fundamental_df: pd.DataFrame) -> list[dict]:
    if fundamental_df is None or fundamental_df.empty:
        return []

    serialized_rows: list[dict] = []
    for idx, row in fundamental_df.sort_index().iterrows():
        serialized_row = {"Date": pd.Timestamp(idx).isoformat()}
        for col, value in row.items():
            serialized_row[col] = None if pd.isna(value) else float(value)
        serialized_rows.append(serialized_row)
    return serialized_rows


def _save_fundamental_history_cache(ticker: str, fundamental_df: pd.DataFrame) -> None:
    from strategy_disk_cache import save_json_cache

    serialized_rows = _serialize_fundamental_history_rows(fundamental_df)
    save_json_cache("fundamental_history", {"ticker": ticker}, serialized_rows, filename="series.json")


def _load_fundamental_history_cache_from_disk(ticker: str) -> Optional[pd.DataFrame]:
    from strategy_disk_cache import load_json_cache

    disk_result = load_json_cache("fundamental_history", {"ticker": ticker}, filename="series.json")
    if disk_result is None:
        return None
    if not isinstance(disk_result, list):
        return None
    if not disk_result:
        return pd.DataFrame()

    df = pd.DataFrame(disk_result)
    if "Date" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fundamental_cache_recent_enough(ticker: str, max_age_days: int) -> bool:
    cache_artifact = _fundamental_cache_artifact_path(ticker)
    if not cache_artifact.exists():
        return False
    cache_age_seconds = time.time() - cache_artifact.stat().st_mtime
    return cache_age_seconds <= max(0, max_age_days) * 86400


def _fundamental_cache_matches_recent_yahoo_history(
    ticker: str,
    cached_df: pd.DataFrame,
) -> tuple[bool, str, pd.DataFrame]:
    fresh_df = _fetch_financial_data(ticker)
    if fresh_df is None or fresh_df.empty:
        return True, "recent Yahoo validation returned empty data", pd.DataFrame()

    recent_fresh_df = fresh_df.sort_index().tail(2)
    if recent_fresh_df.empty:
        return True, "no recent Yahoo quarters to compare", fresh_df

    if cached_df is None or cached_df.empty:
        missing_dates = ", ".join(idx.date().isoformat() for idx in recent_fresh_df.index)
        return False, f"missing recent Yahoo quarter(s): {missing_dates}", fresh_df

    missing_recent_dates = [idx for idx in recent_fresh_df.index if idx not in cached_df.index]
    if missing_recent_dates:
        missing_dates = ", ".join(idx.date().isoformat() for idx in missing_recent_dates)
        return False, f"missing recent Yahoo quarter(s): {missing_dates}", fresh_df

    compared = 0
    for idx, fresh_row in recent_fresh_df.iterrows():
        cached_row = cached_df.loc[idx]
        if isinstance(cached_row, pd.DataFrame):
            cached_row = cached_row.iloc[-1]

        comparable_columns = sorted(set(fresh_df.columns).intersection(cached_df.columns))
        for column_name in comparable_columns:
            fresh_value = pd.to_numeric(pd.Series([fresh_row.get(column_name)]), errors="coerce").iloc[0]
            if pd.isna(fresh_value):
                continue

            cached_value = pd.to_numeric(pd.Series([cached_row.get(column_name)]), errors="coerce").iloc[0]
            if pd.isna(cached_value):
                return False, (
                    f"recent quarter {idx.date().isoformat()} missing cached {column_name}"
                ), fresh_df

            rel_diff = abs(float(cached_value) - float(fresh_value)) / max(abs(float(fresh_value)), 1e-9)
            if rel_diff > 0.005:
                return False, (
                    f"recent quarter {idx.date().isoformat()} mismatch for {column_name} "
                    f"(cache={float(cached_value):.6f}, yahoo={float(fresh_value):.6f}, rel_diff={rel_diff:.4%})"
                ), fresh_df
            compared += 1

    if compared == 0:
        return True, "recent quarter dates present; no comparable fields", fresh_df
    return True, "ok", fresh_df


def _refresh_fundamental_history_cache_worker(ticker: str, max_age_days: int) -> Tuple[str, str, int, str]:
    try:
        cached_df = _load_fundamental_history_cache_from_disk(ticker)
        fetched = None
        refresh_reason = "cache missing or stale"

        if cached_df is not None and _fundamental_cache_recent_enough(ticker, max_age_days):
            cache_valid, validation_detail, fresh_df = _fundamental_cache_matches_recent_yahoo_history(
                ticker,
                cached_df,
            )
            if cache_valid:
                cached_rows = 0 if cached_df.empty else len(cached_df)
                return ticker, "current", cached_rows, validation_detail
            fetched = fresh_df
            refresh_reason = validation_detail

        if fetched is None:
            fetched = _fetch_financial_data(ticker)

        _save_fundamental_history_cache(ticker, fetched)
        if fetched is None or fetched.empty:
            return ticker, "empty", 0, refresh_reason
        return ticker, "saved", len(fetched), refresh_reason
    except Exception as e:
        return ticker, "error", 0, str(e)


def _download_and_update_fundamental_cache(all_tickers: list) -> None:
    from config import CACHE_DAYS

    tickers_to_refresh = [
        ticker for ticker in all_tickers
        if (
            not _fundamental_cache_recent_enough(ticker, CACHE_DAYS)
            or _load_fundamental_history_cache_from_disk(ticker) is not None
        )
    ]
    if not tickers_to_refresh:
        print("  ✅ Fundamental cache already current")
        return

    worker_count = max(1, min(4, NUM_PROCESSES, len(tickers_to_refresh)))
    print(
        f"  📚 Updating fundamentals cache for {len(tickers_to_refresh)} tickers "
        f"({worker_count} Yahoo workers)..."
    )

    saved_count = 0
    current_count = 0
    empty_count = 0
    error_count = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_refresh_fundamental_history_cache_worker, ticker, CACHE_DAYS): ticker
            for ticker in tickers_to_refresh
        }
        for future in as_completed(future_map):
            processed_count += 1
            ticker = future_map[future]
            try:
                _, status, row_count, detail = future.result()
            except Exception as e:
                status = "error"
                row_count = 0
                detail = str(e)
            if status == "current":
                current_count += 1
            elif status == "saved":
                saved_count += 1
                if detail != "cache missing or stale":
                    print(f"  [INFO] Refreshed fundamentals for {ticker}: {detail}")
            elif status == "empty":
                empty_count += 1
                if detail != "cache missing or stale":
                    print(f"  [INFO] Refreshed empty fundamentals for {ticker}: {detail}")
            else:
                error_count += 1
                print(f"  [WARNING] Fundamentals fetch failed for {ticker}: {detail}")

            if processed_count % 100 == 0 or processed_count == len(tickers_to_refresh):
                print(
                    f"  [INFO] Fundamentals progress: {processed_count}/{len(tickers_to_refresh)} "
                    f"(current={current_count}, saved={saved_count}, empty={empty_count}, errors={error_count})"
                )

    print(
        "  ✅ Fundamentals cache update complete "
        f"(current={current_count}, saved={saved_count}, empty={empty_count}, errors={error_count})"
    )

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


def _is_cache_current(last_cached_date, ticker_symbol=None, requested_end_date=None):
    """
    Check if cache is current (has data up to the last trading day).

    Args:
        last_cached_date: The date of the last cached data
        ticker_symbol: Optional ticker symbol to determine exchange/market
        requested_end_date: Optional requested end date for the load request
    """
    last_trading_day = _get_last_trading_day(ticker_symbol)
    target_date = last_trading_day

    if requested_end_date is not None:
        try:
            requested_end_date = _to_utc(requested_end_date).date()
            target_date = min(requested_end_date, last_trading_day)
        except Exception as e:
            print(f"  [DEBUG] {ticker_symbol}: Could not normalize requested_end_date {requested_end_date}: {e}")

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
        except Exception as fallback_error:
            print(f"  [DEBUG] Fallback cached_date conversion failed for {ticker_symbol}: {fallback_error}")
            return False

    # Check if cache has data up to or after the requested target date
    is_current = cached_date >= target_date
    print(f"  [DEBUG] {ticker_symbol}: Cache check {cached_date} >= {target_date} = {is_current}")
    return is_current

# Global lock for yfinance calls - yfinance has threading bugs that cause data corruption
# when called concurrently. This lock serializes all yfinance API calls.
_yfinance_global_lock = threading.Lock()


def _is_intraday_interval(interval: str) -> bool:
    return str(interval).lower() in {"1m", "5m", "15m", "30m", "1h", "60m", "90m"}


def _effective_requested_end_utc(
    ticker_symbol: Optional[str],
    requested_end_utc: pd.Timestamp,
) -> pd.Timestamp:
    """Clamp any requested end date to the end of the last trading day for the ticker's market."""
    try:
        last_trading_day = _get_last_trading_day(ticker_symbol)
        last_trading_day_ts = pd.Timestamp(last_trading_day)
        if last_trading_day_ts.tzinfo is None:
            last_trading_day_ts = last_trading_day_ts.tz_localize("UTC")
        else:
            last_trading_day_ts = last_trading_day_ts.tz_convert("UTC")
        last_trading_day_end = last_trading_day_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        requested_end_utc = pd.Timestamp(requested_end_utc)
        if requested_end_utc.tzinfo is None:
            requested_end_utc = requested_end_utc.tz_localize("UTC")
        else:
            requested_end_utc = requested_end_utc.tz_convert("UTC")
        if (
            requested_end_utc.date() == last_trading_day_ts.date()
            and requested_end_utc.time() == datetime.min.time()
        ):
            return last_trading_day_end
        return min(requested_end_utc, last_trading_day_end)
    except Exception:
        return pd.Timestamp(requested_end_utc)


def _extract_yahoo_error_detail(output_text: str) -> str:
    lines = [line.strip() for line in str(output_text).splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if "Yahoo error" in line or "Failed download" in line or "possibly delisted" in line:
            return line
    return ""


def _normalize_downloaded_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [col[0] if isinstance(col, tuple) else col for col in normalized.columns]
    normalized.columns = [str(col).capitalize() for col in normalized.columns]
    normalized = normalized.loc[:, ~normalized.columns.duplicated()]

    if "Close" not in normalized.columns and "Adj close" in normalized.columns:
        normalized = normalized.rename(columns={"Adj close": "Close"})
    if "Volume" in normalized.columns:
        normalized["Volume"] = normalized["Volume"].fillna(0).astype(int)
    else:
        normalized["Volume"] = 0

    if normalized.index.tzinfo is None:
        normalized.index = normalized.index.tz_localize("UTC")
    else:
        normalized.index = normalized.index.tz_convert("UTC")

    if "Close" in normalized.columns:
        normalized = normalized.dropna(subset=["Close"])

    return normalized.sort_index()


def _download_from_yahoo_with_logging(
    ticker: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    import config as config_module

    start_utc = pd.Timestamp(start_utc)
    end_utc = pd.Timestamp(end_utc)
    if start_utc.tzinfo is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")

    # Yahoo's intraday boundary checks are brittle near the 730-day limit.
    # Trim the actual request window instead of relying on period=Nd, which can
    # still fail for some symbols like AHR even when an equivalent explicit
    # start/end range succeeds.
    if _is_intraday_interval(DATA_INTERVAL):
        max_intraday_span = pd.Timedelta(days=config_module.get_data_lookback_days())
        if end_utc - start_utc > max_intraday_span:
            start_utc = end_utc - max_intraday_span

    request_kwargs = dict(
        interval=DATA_INTERVAL,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
        start=start_utc,
        end=end_utc,
    )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with _yfinance_global_lock:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            downloaded_df = yf.download(ticker, **request_kwargs)

    error_detail = _extract_yahoo_error_detail(
        "\n".join([stdout_buffer.getvalue(), stderr_buffer.getvalue()])
    )
    return downloaded_df, error_detail


def _latest_row_on_or_before(df: pd.DataFrame, anchor_ts: pd.Timestamp) -> Optional[tuple[pd.Timestamp, pd.Series]]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    window = df[df.index <= anchor_ts]
    if window.empty:
        return None
    return window.index[-1], window.iloc[-1]


def _cache_matches_recent_yahoo_history(
    ticker: str,
    cached_df: pd.DataFrame,
    requested_end_utc: pd.Timestamp,
) -> tuple[bool, str, pd.DataFrame]:
    if cached_df is None or cached_df.empty or "Close" not in cached_df.columns:
        return True, "no cached close history", pd.DataFrame()

    effective_end_utc = _effective_requested_end_utc(ticker, requested_end_utc)
    compare_start = effective_end_utc - timedelta(days=200)
    fresh_df, yahoo_error_detail = _download_from_yahoo_with_logging(
        ticker=ticker,
        start_utc=compare_start,
        end_utc=effective_end_utc,
    )
    fresh_df = _normalize_downloaded_price_df(fresh_df)
    if fresh_df.empty:
        detail = yahoo_error_detail or "recent Yahoo validation returned empty data"
        return True, detail, fresh_df

    recent_week_start = effective_end_utc - timedelta(days=7)
    cached_recent_df = cached_df.loc[
        (cached_df.index >= recent_week_start) & (cached_df.index <= effective_end_utc)
    ].copy()
    fresh_recent_df = fresh_df.loc[
        (fresh_df.index >= recent_week_start) & (fresh_df.index <= effective_end_utc)
    ].copy()

    if not fresh_recent_df.empty:
        missing_recent_timestamps = fresh_recent_df.index.difference(cached_recent_df.index)
        if len(missing_recent_timestamps) > 0:
            first_missing = missing_recent_timestamps[0].isoformat()
            last_missing = missing_recent_timestamps[-1].isoformat()
            return False, (
                f"missing {len(missing_recent_timestamps)} Yahoo bars in last week "
                f"(first_missing={first_missing}, last_missing={last_missing})"
            ), fresh_df

        recent_joined = cached_recent_df[["Close"]].join(
            fresh_recent_df[["Close"]],
            how="inner",
            lsuffix="_cache",
            rsuffix="_yahoo",
        )
        if not recent_joined.empty:
            rel_diff = (
                (recent_joined["Close_cache"] - recent_joined["Close_yahoo"]).abs()
                / recent_joined["Close_yahoo"].abs().clip(lower=1e-9)
            )
            bad_recent = rel_diff[rel_diff > 0.002]
            if not bad_recent.empty:
                mismatch_ts = bad_recent.index[0]
                mismatch_rel_diff = float(bad_recent.iloc[0])
                return False, (
                    f"last-week close mismatch at {mismatch_ts.isoformat()} "
                    f"(cache={recent_joined.loc[mismatch_ts, 'Close_cache']:.6f}, "
                    f"yahoo={recent_joined.loc[mismatch_ts, 'Close_yahoo']:.6f}, "
                    f"rel_diff={mismatch_rel_diff:.4%})"
                ), fresh_df

    anchors = (
        ("6mo_a", effective_end_utc - timedelta(days=182)),
        ("6mo_b", effective_end_utc - timedelta(days=175)),
        ("1mo", effective_end_utc - timedelta(days=30)),
        ("1w", effective_end_utc - timedelta(days=7)),
    )
    compared = 0
    for label, anchor_ts in anchors:
        cached_row = _latest_row_on_or_before(cached_df, anchor_ts)
        fresh_row = _latest_row_on_or_before(fresh_df, anchor_ts)
        if cached_row is None or fresh_row is None:
            continue

        cached_idx, cached_values = cached_row
        fresh_idx, fresh_values = fresh_row
        if abs(cached_idx - fresh_idx) > pd.Timedelta(days=2):
            return False, (
                f"{label} anchor timestamp mismatch "
                f"(cache={cached_idx.isoformat()}, yahoo={fresh_idx.isoformat()})"
            ), fresh_df

        cached_close = float(cached_values["Close"])
        fresh_close = float(fresh_values["Close"])
        if not np.isfinite(cached_close) or not np.isfinite(fresh_close):
            continue

        rel_diff = abs(cached_close - fresh_close) / max(abs(fresh_close), 1e-9)
        if rel_diff > 0.002:
            return False, (
                f"{label} anchor close mismatch "
                f"(cache={cached_close:.6f}, yahoo={fresh_close:.6f}, rel_diff={rel_diff:.4%})"
            ), fresh_df
        compared += 1

    if compared == 0:
        return True, "no comparable recent anchors", fresh_df
    return True, "ok", fresh_df

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
    import config as config_module

    # Get ENABLE_DATA_DOWNLOAD directly from module (not cached import)
    ENABLE_DATA_DOWNLOAD = config_module.ENABLE_DATA_DOWNLOAD

    # Per-ticker locks for fine-grained thread safety (allows parallel downloads)
    if not hasattr(load_prices, '_ticker_locks'):
        load_prices._ticker_locks = {}
        load_prices._locks_lock = threading.Lock()  # Lock for creating new ticker locks

    # Get or create per-ticker lock
    with load_prices._locks_lock:
        if ticker not in load_prices._ticker_locks:
            load_prices._ticker_locks[ticker] = threading.Lock()
        ticker_lock = load_prices._ticker_locks[ticker]

    _ensure_dir(_RESOLVED_DATA_CACHE_DIR)
    cache_file = _RESOLVED_DATA_CACHE_DIR / f"{ticker}.csv"

    cached_df = pd.DataFrame()
    new_df = pd.DataFrame()
    fetch_start = None
    requested_end_utc = _to_utc(end)
    fetch_end = _effective_requested_end_utc(ticker, requested_end_utc)
    needs_fetch = True if ENABLE_DATA_DOWNLOAD else False
    last_cached_date = None
    cache_is_current = None
    recent_yahoo_validation_df = pd.DataFrame()

    # --- Step 1: Check existing cache and determine what to fetch ---
    with ticker_lock:
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
                    should_validate_cache = bool(
                        ENABLE_DATA_DOWNLOAD and (DATA_PROVIDER == 'yahoo' or USE_YAHOO_FALLBACK)
                    )
                    if should_validate_cache:
                        cache_valid, validation_detail, recent_yahoo_validation_df = _cache_matches_recent_yahoo_history(
                            ticker=ticker,
                            cached_df=cached_df,
                            requested_end_utc=requested_end_utc,
                        )
                        if not cache_valid:
                            print(
                                f"  [WARNING] {ticker}: Cached CSV mismatches recent Yahoo data "
                                f"({validation_detail}); deleting cache and redownloading full history"
                            )
                            try:
                                cache_file.unlink(missing_ok=True)
                            except Exception as delete_exc:
                                print(f"  [WARNING] {ticker}: Could not delete mismatched cache: {delete_exc}")
                            cached_df = pd.DataFrame()
                            recent_yahoo_validation_df = pd.DataFrame()

                if not cached_df.empty:
                    last_cached_date = cached_df.index[-1]
                    # Debug: Show cache file info for specific tickers
                    if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                        print(f"  [INFO] Cache {ticker}: shape={cached_df.shape}, Close[0]={cached_df['Close'].iloc[0]:.2f}, Close[-1]={cached_df['Close'].iloc[-1]:.2f}")

                    # [PASS] FIX: Use proper trading day check to avoid fetching on weekends
                    cache_is_current = _is_cache_current(last_cached_date, ticker, requested_end_utc)
                    # Log cache status (use tqdm.write to avoid progress bar conflicts)
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"  [DEBUG] {ticker}: cache={last_cached_date.date()}, current={cache_is_current}")
                    except:
                        print(f"  [DEBUG] {ticker}: cache={last_cached_date.date()}, current={cache_is_current}")
                    if cache_is_current:
                        # Cache already has data up to the last trading day
                        needs_fetch = False
                    else:
                        # For intraday data, restart at the next calendar day boundary
                        # instead of carrying forward the last cached bar's clock time.
                        if _is_intraday_interval(DATA_INTERVAL):
                            fetch_start = pd.Timestamp(last_cached_date).normalize() + timedelta(days=1)
                            if fetch_start.tzinfo is None:
                                fetch_start = fetch_start.tz_localize("UTC")
                            else:
                                fetch_start = fetch_start.tz_convert("UTC")
                        else:
                            # Daily/longer intervals can safely continue from the next day.
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

        if not recent_yahoo_validation_df.empty and not cached_df.empty:
            validation_tail_df = recent_yahoo_validation_df.loc[
                (recent_yahoo_validation_df.index >= start_utc)
                & (recent_yahoo_validation_df.index <= end_utc)
            ].copy()
            if not validation_tail_df.empty:
                new_df = validation_tail_df
                if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                    print(
                        f"  [INFO] Reusing recent Yahoo validation data for {ticker} "
                        f"({len(new_df)} rows)"
                    )

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
            yahoo_error_detail = ""
            try:
                downloaded_df, yahoo_error_detail = _download_from_yahoo_with_logging(
                    ticker=ticker,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
                if downloaded_df is not None and not downloaded_df.empty:
                    new_df = downloaded_df.dropna()
                else:
                    if last_cached_date is not None:
                        error_message = (
                            f"  [ERROR] {ticker}: No today's {DATA_INTERVAL} data available. "
                            f"Last cached date: {last_cached_date.date()}"
                        )
                    else:
                        error_message = f"  [ERROR] {ticker}: Yahoo returned empty data"
                    detail_suffix = f" ({yahoo_error_detail})" if yahoo_error_detail else ""
                    try:
                        from tqdm import tqdm
                        tqdm.write(f"{error_message}{detail_suffix}")
                    except:
                        print(f"{error_message}{detail_suffix}")
            except Exception as e:
                detail_suffix = f" | {yahoo_error_detail}" if yahoo_error_detail else ""
                try:
                    from tqdm import tqdm
                    tqdm.write(f"  [ERROR] {ticker}: Yahoo failed - {str(e)[:100]}{detail_suffix}")
                except:
                    print(f"  [ERROR] {ticker}: Yahoo failed - {str(e)[:100]}{detail_suffix}")

        # If all providers failed, silently use stale cache (reduces log noise)

        # No fallback to daily data - use only the configured DATA_INTERVAL
        if new_df.empty:
            print(f"  [WARNING] {ticker}: No {DATA_INTERVAL} data available")
    else:
        if cached_df.empty:
            print(f"  [WARNING] Skipping fetch for {ticker} - downloads disabled and no cache is available")
        elif cache_is_current is False and last_cached_date is not None:
            print(f"  [WARNING] Skipping fetch for {ticker} - downloads disabled, using stale cache from {last_cached_date.date()}")
        else:
            print(f"  [DEBUG] Skipping fetch for {ticker} - cache is current")

    # Clean up new data (only if we fetched anything)
    if not new_df.empty:
        # Make an explicit copy to avoid SettingWithCopyWarning
        new_df = new_df.copy()

        # Handle MultiIndex columns from yfinance (e.g., ('Close', 'AAPL') -> 'Close')
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = [col[0] if isinstance(col, tuple) else col for col in new_df.columns]
        new_df.columns = [str(col).capitalize() for col in new_df.columns]

        # Remove duplicate columns (can happen with MultiIndex flattening)
        new_df = new_df.loc[:, ~new_df.columns.duplicated()]

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

    # --- Step 4: Save updated cache (always save original 1h data) ---
    if needs_fetch and not new_df.empty:
        with ticker_lock:
            try:
                # Debug: Show cache save info for specific tickers
                if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                    print(f"  💾 Saving {ticker}: shape={price_df.shape}, Close[0]={price_df['Close'].iloc[0]:.2f}, Close[-1]={price_df['Close'].iloc[-1]:.2f}")
                # Ensure index has a name for proper CSV loading later
                if price_df.index.name is None:
                    price_df.index.name = 'Datetime'
                price_df.to_csv(cache_file)
            except Exception as e:
                print(f"  Warning: Could not save cache for {ticker}: {e}")

    # --- Step 5: Convert 1h data to daily data if DATA_INTERVAL is 1h ---
    if DATA_INTERVAL == '1h' and not price_df.empty:
        try:
            converted_df = _convert_hourly_frame_to_daily(price_df)
            if converted_df is not None and not converted_df.empty:
                if ticker in ['SNDK', 'SLV', 'MU', 'NEM', 'AAPL']:
                    original_features = len(price_df.columns)
                    daily_features = len(converted_df.columns)
                    start_date = converted_df.index[0].strftime('%Y-%m-%d')
                    end_date = converted_df.index[-1].strftime('%Y-%m-%d')
                    print(f"  [INFO] Converted {ticker}: 1h ({price_df.shape[0]} rows, {original_features} features) -> daily ({converted_df.shape[0]} rows, {daily_features} features)")
                    print(f"     [INFO] Date range: {start_date} to {end_date} ({converted_df.shape[0]} trading days)")
                price_df = converted_df
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
    import config as config_module

    # Get ENABLE_DATA_DOWNLOAD directly from module (not cached import)
    ENABLE_DATA_DOWNLOAD = config_module.ENABLE_DATA_DOWNLOAD

    # Check if data downloads are disabled
    if not ENABLE_DATA_DOWNLOAD:
        print(f"  [INFO] Data downloads disabled (ENABLE_DATA_DOWNLOAD=False)")
        print(f"  [INFO] Using only cached data for {len(tickers)} tickers...")

        # Load from cache with checking (but no downloads)
        all_data_frames = []
        for ticker in tickers:
            try:
                df = load_prices(ticker, start, end)
                if not df.empty:
                    df = df.copy()
                    df['ticker'] = ticker
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df = df.rename(columns={'Date': 'date'})
                    elif 'Datetime' in df.columns:
                        df = df.rename(columns={'Datetime': 'date'})
                    elif 'index' in df.columns:
                        df = df.rename(columns={'index': 'date'})
                    elif df.index.name in [None, '']:
                        df = df.rename(columns={df.columns[0]: 'date'})
                    all_data_frames.append(df)
            except Exception:
                pass  # Skip silently - no downloads allowed

        if all_data_frames:
            return pd.concat(all_data_frames, ignore_index=True)
        else:
            print(f"  [INFO] No cached data found")
            return pd.DataFrame()

    all_data_frames = []

    print(f"  [INFO] Processing {len(tickers)} tickers with incremental caching (parallel)...")

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

    # Global rate limiter for API calls (shared across threads)
    import threading
    import time
    if not hasattr(_download_batch_robust, '_rate_limiter'):
        _download_batch_robust._rate_limiter = threading.Semaphore(8)  # Max 8 concurrent API calls
        _download_batch_robust._last_call_time = 0
        _download_batch_robust._call_lock = threading.Lock()

    def download_single_ticker(ticker):
        """Download a single ticker with rate limiting"""
        # Acquire rate limiter (limits concurrent API calls)
        with _download_batch_robust._rate_limiter:
            # Add small delay between API calls to avoid burst requests
            with _download_batch_robust._call_lock:
                elapsed = time.time() - _download_batch_robust._last_call_time
                if elapsed < 0.1:  # Minimum 100ms between API calls
                    time.sleep(0.1 - elapsed)
                _download_batch_robust._last_call_time = time.time()

            try:
                # Use load_prices_robust for automatic retry on rate limits
                df = load_prices_robust(ticker, start, end)
                if not df.empty:
                    df = df.copy()
                    df['ticker'] = ticker
                    df = df.reset_index()
                    if 'Date' in df.columns:
                        df = df.rename(columns={'Date': 'date'})
                    elif 'Datetime' in df.columns:
                        df = df.rename(columns={'Datetime': 'date'})
                    elif 'index' in df.columns:
                        df = df.rename(columns={'index': 'date'})
                    elif 'level_0' in df.columns:
                        df = df.rename(columns={'level_0': 'date'})
                    return df
            except Exception as e:
                pass  # Errors logged inside load_prices_robust
            return None

    # Use 16 workers for downloads, but rate limiter caps concurrent API calls to 8
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_single_ticker, ticker): ticker for ticker in tickers}

        pbar = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tickers), desc="  [INFO] Downloading tickers", ncols=100, file=sys.stdout)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        all_data_frames.append(result)
                except Exception:
                    pass  # Ignore individual ticker errors
                pbar.update(1)
        except ImportError:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        all_data_frames.append(result)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [WARN] Download progress tracking error: {e}", flush=True)
        finally:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()

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


def _fetch_intermarket_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch intermarket data for analysis.

    Args:
        start: Start date for data (required)
        end: End date for data (required)

    Returns:
        DataFrame with intermarket close prices, or empty DataFrame if no data
    """
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

    for symbol, name in intermarket_symbols.items():
        df = load_prices(symbol, start, end)
        if not df.empty:
            df_renamed = df[['Close']].copy()
            df_renamed.columns = [f'{symbol}_Close']
            all_data.append(df_renamed)

    if all_data:
        return pd.concat(all_data, axis=1)
    return pd.DataFrame()


def _load_ticker_csv_from_cache_worker(args: Tuple[str, datetime, datetime, str]) -> Optional[pd.DataFrame]:
    """Process-safe worker for loading one ticker CSV from cache."""
    ticker, start_utc, end_utc, cache_dir_str = args
    cache_dir = Path(cache_dir_str)
    try:
        cache_file = cache_dir / f"{ticker}.csv"
        if not cache_file.exists():
            return None

        df = pd.read_csv(cache_file)

        if 'Datetime' in df.columns:
            date_col = 'Datetime'
        elif 'Date' in df.columns:
            date_col = 'Date'
        elif 'Unnamed: 0' in df.columns:
            date_col = 'Unnamed: 0'
        else:
            return None

        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index(date_col)
        df = df.loc[(df.index >= start_utc) & (df.index <= end_utc)].copy()
        if DATA_INTERVAL == '1h' and AGGREGATE_TO_DAILY and not df.empty:
            converted_df = _convert_hourly_frame_to_daily(df)
            if converted_df is not None and not converted_df.empty:
                df = converted_df
        if df.empty:
            return None

        df['ticker'] = ticker
        df = df.reset_index()
        if df.columns[0] != 'date':
            df = df.rename(columns={df.columns[0]: 'date'})
        return df
    except Exception:
        return None


def _load_from_cache_parallel(all_tickers: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load data from cache files in parallel. Fast path - no API calls, no locks.

    Args:
        all_tickers: List of ticker symbols
        start_date: Start date for data filtering
        end_date: End date for data filtering

    Returns:
        DataFrame with all ticker data in long format
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context
    from utils import _to_utc

    worker_count = max(1, min(NUM_PROCESSES, len(all_tickers))) if all_tickers else 1
    print(
        f"  [INFO] Loading price history cache for {len(all_tickers)} tickers "
        f"(parallel, {worker_count} fork workers)..."
    )

    start_utc = _to_utc(start_date)
    end_utc = _to_utc(end_date)
    cache_dir = _RESOLVED_DATA_CACHE_DIR

    all_data_frames = []
    worker_args = [
        (ticker, start_utc, end_utc, str(cache_dir))
        for ticker in all_tickers
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("fork"),
    ) as executor:
        futures = {
            executor.submit(_load_ticker_csv_from_cache_worker, worker_arg): worker_arg[0]
            for worker_arg in worker_args
        }
        pending_futures = set(futures)
        completed_count = 0
        from tqdm import tqdm
        progress_bar = tqdm(
            total=len(futures),
            desc="  [INFO] Loading price history cache",
            unit="ticker",
        )
        try:
            while pending_futures:
                done, pending_futures = wait(
                    pending_futures,
                    timeout=10.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    print(
                        "  [INFO] Price history cache heartbeat: "
                        f"{completed_count}/{len(futures)} tickers completed..."
                    )
                    continue
                for future in done:
                    result = future.result()
                    completed_count += 1
                    progress_bar.update(1)
                    if result is not None:
                        all_data_frames.append(result)
        finally:
            progress_bar.close()

    print(
        f"  [INFO] Loaded {len(all_data_frames)}/{len(all_tickers)} tickers "
        "from price history cache"
    )

    if all_data_frames:
        all_tickers_data = pd.concat(all_data_frames, ignore_index=True)
        if 'date' in all_tickers_data.columns:
            all_tickers_data['date'] = pd.to_datetime(all_tickers_data['date'], utc=True)
        return all_tickers_data
    else:
        return pd.DataFrame()


def _convert_hourly_frame_to_daily(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert cached hourly OHLCV data into one daily bar plus aggregated intraday features."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    working = price_df.copy()
    working.columns = [str(col).strip().capitalize() for col in working.columns]
    working = working.loc[:, ~working.columns.duplicated()]

    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(set(working.columns)):
        return pd.DataFrame()

    if isinstance(working['Close'], pd.DataFrame):
        working['Close'] = working['Close'].iloc[:, 0]

    working['Hourly_Return'] = working['Close'].pct_change()
    working['Intraday_Range_Pct'] = (working['High'] - working['Low']) / working['Open'] * 100
    working['Hourly_Volatility'] = working['Hourly_Return'].rolling(5).std()
    working['Intraday_Vol_Ratio'] = working['Hourly_Volatility'] / working['Hourly_Return'].rolling(20).std()
    working['Volume_Weighted_Price'] = (working['Volume'] * working['Close']).cumsum() / working['Volume'].cumsum()
    working['Volume_Concentration'] = working['Volume'] / working['Volume'].rolling(8).sum()
    working['Buying_Pressure'] = (working['Close'] > working['Open']).astype(int) * working['Volume']
    working['Selling_Pressure'] = (working['Close'] < working['Open']).astype(int) * working['Volume']
    working['Price_Discovery_Efficiency'] = abs(working['Close'] - working['Volume_Weighted_Price']) / working['Volume_Weighted_Price'] * 100
    working['Information_Shock'] = abs(working['Hourly_Return']).rolling(20).mean() * working['Volume_Concentration']
    working['Hourly_Direction'] = (working['Hourly_Return'] > 0).astype(int)
    working['Momentum_Persistence'] = working['Hourly_Direction'].rolling(4).sum()
    # Preserve the original rolling slope semantics without Python-level rolling apply.
    trend_window = 8
    close_values = pd.to_numeric(working['Close'], errors='coerce').to_numpy(dtype=float, copy=False)
    trend_strength = np.full(close_values.shape, np.nan, dtype=float)
    if close_values.size >= trend_window:
        x = np.arange(trend_window, dtype=float)
        x_centered = x - x.mean()
        denominator = np.square(x_centered).sum()
        valid_close_values = np.nan_to_num(close_values, nan=0.0)
        slope_numerators = np.convolve(valid_close_values, x_centered[::-1], mode='valid')
        trend_strength[trend_window - 1:] = slope_numerators / denominator
        valid_window_mask = (
            pd.Series(close_values).rolling(trend_window).count().to_numpy() == trend_window
        )
        trend_strength[~valid_window_mask] = np.nan
    working['Intraday_Trend_Strength'] = trend_strength
    working['Liquidity_Detection'] = working['Volume'] / working['Intraday_Range_Pct']
    working['Net_Order_Flow'] = (working['Buying_Pressure'] - working['Selling_Pressure']) / working['Volume']
    working['Market_Impact'] = abs(working['Hourly_Return']) / working['Volume']
    working['Vol_Clustering'] = (
        working['Hourly_Volatility'] > working['Hourly_Volatility'].rolling(20).mean()
    ).astype(int)
    working['Vol_Regime_Change'] = (
        working['Hourly_Volatility'].rolling(8).std() / working['Hourly_Volatility'].rolling(20).std()
    )

    daily_df = working.resample('B').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
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
        'Vol_Regime_Change': 'mean',
    }).dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    if daily_df.empty:
        return daily_df

    daily_df['Net_Buying_Pressure'] = daily_df['Buying_Pressure'] - daily_df['Selling_Pressure']
    daily_df['Buying_Pressure_Ratio'] = daily_df['Buying_Pressure'] / (
        daily_df['Buying_Pressure'] + daily_df['Selling_Pressure']
    )
    daily_df['Total_Order_Flow'] = daily_df['Buying_Pressure'] + daily_df['Selling_Pressure']
    return _calculate_technical_indicators(daily_df)


def _download_and_update_cache(all_tickers: list, start_date: datetime, end_date: datetime) -> None:
    """
    Download data and update cache files. Does NOT return data - just updates cache.
    After this, use _load_from_cache_parallel() to read the data.

    Args:
        all_tickers: List of ticker symbols to download
        start_date: Start date for data
        end_date: End date for data
    """
    from config import BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES
    import time

    print(f"🚀 Downloading/updating cache for {len(all_tickers)} tickers...")

    # Use parallel downloads within _download_batch_robust (already parallelized)
    # This calls load_prices() which checks cache freshness and downloads if needed
    _download_batch_robust(all_tickers, start=start_date, end=end_date)
    _download_and_update_fundamental_cache(all_tickers)

    print(f"  ✅ Cache update complete")


def load_all_market_data(all_tickers: list, end_date: datetime = None) -> pd.DataFrame:
    """
    Load market data for all tickers.

    Two-phase approach:
    1. If downloads enabled: update cache files (download new data where needed)
    2. Always: load from cache in parallel (fast path)

    Args:
        all_tickers: List of ticker symbols
        end_date: End date for data (defaults to last trading day)

    Returns:
        DataFrame with all ticker data in long format (with 'ticker' and 'date' columns)
    """
    import config as config_module

    # Get ENABLE_DATA_DOWNLOAD directly from module (not cached import)
    ENABLE_DATA_DOWNLOAD = config_module.ENABLE_DATA_DOWNLOAD

    if end_date is None:
        end_date = _get_last_trading_day()
        end_date = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    # Use shared function to determine lookback period
    from config import get_data_lookback_days
    lookback_days = get_data_lookback_days()
    start_date = end_date - timedelta(days=lookback_days)
    days_back = (end_date - start_date).days

    print(f"📊 Loading data for {len(all_tickers)} tickers ({start_date.date()} to {end_date.date()})")
    print(f"  (Requesting {days_back} days of data)")

    # Phase 1: Update cache if downloads enabled
    if ENABLE_DATA_DOWNLOAD:
        _download_and_update_cache(all_tickers, start_date, end_date)
    else:
        print(f"  📦 Using cached data only (--no-download)")

    # Phase 2: Load from cache (always - fast parallel path)
    all_tickers_data = _load_from_cache_parallel(all_tickers, start_date, end_date)

    if all_tickers_data.empty:
        print("❌ No data found in cache")
        return pd.DataFrame()

    print(f"   📊 Data shape: {all_tickers_data.shape}")
    print(f"   📊 Columns: {list(all_tickers_data.columns[:5])}")

    return all_tickers_data
