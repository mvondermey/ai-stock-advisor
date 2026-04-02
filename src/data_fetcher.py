import os

import time

import random

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import List, Dict, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed



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

    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

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



# Import from config and utils

from config import (

    DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_INTERVAL, DATA_CACHE_DIR, CACHE_DAYS,
    ENABLE_PRICE_CACHE,

    ALPACA_API_KEY, ALPACA_SECRET_KEY, TWELVEDATA_API_KEY, TWELVEDATA_MAX_WORKERS,

    PAUSE_BETWEEN_BATCHES, PAUSE_BETWEEN_YF_CALLS, INVESTMENT_PER_STOCK

)

from utils import _ensure_dir, _to_utc

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


def _effective_end_utc_for_daily(end: datetime) -> pd.Timestamp:
    """
    For daily bars, avoid requesting "today" (which is often incomplete intraday and causes cache misses).
    Returns a UTC-normalized Timestamp for the last *completed* business day.
    """
    end_utc = _to_utc(end).normalize()
    now_day = _to_utc(datetime.now(timezone.utc)).normalize()

    # If caller asks for today or beyond, clamp to last business day.
    if end_utc >= now_day:
        end_utc = (now_day - pd.tseries.offsets.BDay(1)).normalize()

    # If end lands on a weekend, roll back to previous business day.
    if end_utc.weekday() >= 5:
        end_utc = (end_utc - pd.tseries.offsets.BDay(1)).normalize()

    return end_utc





def _normalize_symbol(symbol: str, provider: str) -> str:

    """Normalize ticker symbol based on provider requirements."""

    symbol = symbol.upper().strip()



    if provider == 'stooq':

        # Stooq uses .US suffix for US stocks

        if not symbol.endswith('.US'):

            return f"{symbol}.US"

    elif provider in ['alpaca', 'twelvedata', 'yahoo']:

        # Remove .US suffix if present

        if symbol.endswith('.US'):

            return symbol[:-3]



    return symbol





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

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        # Convert DATA_INTERVAL to Alpaca TimeFrame (no fallback to daily)
        interval_map = {
            '1d': TimeFrame.Day, '1day': TimeFrame.Day,
            '1h': TimeFrame.Hour, '1hour': TimeFrame.Hour,
            '30m': TimeFrame(30, TimeFrameUnit.Minute), '30min': TimeFrame(30, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute), '15min': TimeFrame(15, TimeFrameUnit.Minute),
            '5m': TimeFrame(5, TimeFrameUnit.Minute), '5min': TimeFrame(5, TimeFrameUnit.Minute),
            '1m': TimeFrame.Minute, '1min': TimeFrame.Minute,
        }
        timeframe = interval_map.get(DATA_INTERVAL, TimeFrame.Hour)  # Default to Hour, not Day

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment='split'  # ✅ Apply split adjustments to prevent fake gains
        )

        bars = client.get_stock_bars(request_params)

        df = bars.df



        if df.empty:

            return pd.DataFrame()



        # Alpaca returns a MultiIndex DataFrame, we need to flatten it

        df = df.loc[ticker]

        df = df.rename(columns={

            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'

        })

        df.index.name = "Date"

        df.index = pd.to_datetime(df.index, utc=True)

        return df

    except Exception as e:

        error_msg = str(e)

        if "subscription does not permit querying recent SIP data" in error_msg:

            print(f"  ℹ️ Alpaca (free tier) does not provide recent data for {ticker}. Attempting fallback provider.")

        else:

            print(f"  ⚠️ Could not fetch data from Alpaca for {ticker}: {e}")

        return pd.DataFrame()


def _fetch_batch_from_alpaca(tickers: List[str], start: datetime, end: datetime) -> Tuple[pd.DataFrame, List[str]]:
    """
    Batch fetch OHLCV from Alpaca for multiple tickers.
    Returns: (DataFrame with MultiIndex columns like yfinance, list of failed tickers)
    """
    if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return pd.DataFrame(), tickers  # All failed

    if not tickers:
        return pd.DataFrame(), []

    try:
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        # Convert DATA_INTERVAL to Alpaca TimeFrame
        if DATA_INTERVAL in ['1d', '1day']:
            timeframe = TimeFrame.Day
        elif DATA_INTERVAL in ['1h', '1hour']:
            timeframe = TimeFrame.Hour
        elif DATA_INTERVAL in ['1m', '1min']:
            timeframe = TimeFrame.Minute
        else:
            timeframe = TimeFrame.Day

        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment='split'  # ✅ Apply split adjustments to prevent fake gains
        )

        bars = client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            return pd.DataFrame(), tickers  # All failed

        # Convert Alpaca format to yfinance-like MultiIndex format
        # Alpaca returns MultiIndex (symbol, timestamp) on rows
        # We need MultiIndex (Field, Ticker) on columns
        result_frames = []
        successful_tickers = []
        failed_tickers = []

        for ticker in tickers:
            try:
                if ticker in df.index.get_level_values(0):
                    ticker_df = df.loc[ticker].copy()
                    ticker_df = ticker_df.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    })
                    ticker_df.index = pd.to_datetime(ticker_df.index, utc=True)
                    ticker_df.index.name = "Date"

                    # Convert to yfinance-like MultiIndex columns
                    ticker_df.columns = pd.MultiIndex.from_product([ticker_df.columns, [ticker]])
                    result_frames.append(ticker_df)
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
            except Exception:
                failed_tickers.append(ticker)

        if result_frames:
            combined = pd.concat(result_frames, axis=1)
            return combined, failed_tickers
        else:
            return pd.DataFrame(), tickers

    except Exception as e:
        error_msg = str(e)
        if "subscription does not permit querying recent SIP data" in error_msg:
            print(f"  ℹ️ Alpaca (free tier) does not provide recent data. Falling back to yfinance...")
        else:
            print(f"  ⚠️ Alpaca batch fetch failed: {e}. Falling back to yfinance...")
        return pd.DataFrame(), tickers  # All failed



def _fetch_from_twelvedata(ticker: str, start: datetime, end: datetime, api_key: Optional[str] = None) -> pd.DataFrame:

    """Fetch OHLCV from TwelveData using the SDK."""

    key_to_use = api_key if api_key else TWELVEDATA_API_KEY

    if not TWELVEDATA_SDK_AVAILABLE or not key_to_use:

        return pd.DataFrame()



    try:

        tdc = TDClient(apikey=key_to_use)



        # Convert DATA_INTERVAL to TwelveData format (no fallback to daily)
        interval_map = {
            '1d': '1day', '1day': '1day',
            '1h': '1h', '1hour': '1h',
            '30m': '30min', '30min': '30min',
            '15m': '15min', '15min': '15min',
            '5m': '5min', '5min': '5min',
            '1m': '1min', '1min': '1min',
        }
        twelivedata_interval = interval_map.get(DATA_INTERVAL, '1h')  # Default to 1h, not 1day

        ts = tdc.time_series(
            symbol=ticker,
            interval=twelivedata_interval,
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),

            outputsize=5000

        ).as_pandas()



        if ts.empty:

            print(f"  ℹ️ No data found for {ticker} from TwelveData SDK.")

            return pd.DataFrame()



        df = ts.copy()



        df.index = pd.to_datetime(df.index, utc=True)

        df.index.name = "Date"



        df = df.rename(columns={

            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'

        })



        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:

            if col in df.columns:

                df[col] = pd.to_numeric(df[col], errors='coerce')



        return df.sort_index()

    except Exception as e:
        error_msg = str(e).lower()
        # Re-raise rate limit errors so caller can detect them
        if "run out of api credits" in error_msg or "api credits were used" in error_msg:
            raise

        print(f"  ⚠️ An error occurred while fetching data from TwelveData SDK for {ticker}: {e}")

        return pd.DataFrame()



def _fetch_batch_multi_provider(tickers: List[str], start: datetime, end: datetime, show_progress: bool = True) -> List[pd.DataFrame]:
    """
    Fetch batch data using multi-provider strategy: Alpaca → TwelveData → yfinance.
    Returns list of DataFrames with yfinance-compatible MultiIndex columns.
    """
    remaining_tickers = list(tickers)
    result_frames = []
    max_retries = 7
    base_wait_time = 30

    # 1. Try Alpaca first (if available and configured)
    if DATA_PROVIDER == 'alpaca' and ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        print(f"  📡 Trying Alpaca for {len(remaining_tickers)} tickers...")
        alpaca_data, failed_tickers = _fetch_batch_from_alpaca(remaining_tickers, start, end)
        if not alpaca_data.empty:
            result_frames.append(alpaca_data)
            success_count = len(remaining_tickers) - len(failed_tickers)
            print(f"  ✅ Alpaca: {success_count}/{len(remaining_tickers)} tickers")
        remaining_tickers = failed_tickers

    # 2. Try TwelveData for remaining tickers (PARALLEL with rate limit detection)
    if remaining_tickers and TWELVEDATA_SDK_AVAILABLE and TWELVEDATA_API_KEY:
        print(f"  📡 Trying TwelveData for {len(remaining_tickers)} remaining tickers in parallel...")
        twelvedata_frames = []
        twelvedata_failed = []
        rate_limit_hit = False

        def _fetch_td_single(ticker):
            """Helper for parallel TwelveData fetch"""
            try:
                td_df = _fetch_from_twelvedata(ticker, start, end)
                if not td_df.empty:
                    # Convert to yfinance-like MultiIndex format
                    td_df.columns = pd.MultiIndex.from_product([td_df.columns, [ticker]])
                    return (ticker, td_df, True, None)
                return (ticker, None, False, None)
            except Exception as e:
                error_msg = str(e).lower()
                # Detect rate limit errors
                if "run out of api credits" in error_msg or "api credits were used" in error_msg:
                    return (ticker, None, False, "RATE_LIMIT")
                return (ticker, None, False, str(e))

        # Use thread pool for parallel API requests (I/O bound)
        max_workers = min(TWELVEDATA_MAX_WORKERS, len(remaining_tickers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_td_single, ticker): ticker for ticker in remaining_tickers}

            pbar = tqdm(total=len(remaining_tickers), desc="TwelveData", unit="ticker")
            for future in as_completed(futures):
                try:
                    ticker, df, success, error = future.result()

                    # Check for rate limit
                    if error == "RATE_LIMIT":
                        if not rate_limit_hit:
                            print(f"\n  ⚠️ TwelveData API rate limit reached. Skipping remaining TwelveData requests...")
                            rate_limit_hit = True
                        twelvedata_failed.append(ticker)
                        # Cancel pending futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                    elif success:
                        twelvedata_frames.append(df)
                    else:
                        twelvedata_failed.append(ticker)
                except Exception:
                    ticker = futures[future]
                    twelvedata_failed.append(ticker)
                finally:
                    pbar.update(1)
            pbar.close()

        if twelvedata_frames:
            result_frames.extend(twelvedata_frames)
            success_count = len(remaining_tickers) - len(twelvedata_failed)
            if rate_limit_hit:
                print(f"  ⚠️ TwelveData: {success_count}/{len(remaining_tickers)} tickers (rate limit hit, rest will use yfinance)")
            else:
                print(f"  ✅ TwelveData: {success_count}/{len(remaining_tickers)} tickers")
        remaining_tickers = twelvedata_failed

    # 3. Fall back to yfinance for remaining tickers
    if remaining_tickers:
        if DATA_PROVIDER == 'alpaca':
            print(f"  📡 Falling back to yfinance for {len(remaining_tickers)} remaining tickers...")

        for attempt in range(max_retries):
            try:
                # CRITICAL: threads=False prevents yfinance internal threading bugs
                # that cause data corruption (wrong ticker's data returned)
                yf_data = yf.download(
                    remaining_tickers,
                    start=start,
                    end=end,
                    interval=DATA_INTERVAL,
                    auto_adjust=True,
                    progress=show_progress,
                    threads=False,  # DISABLED: yfinance threading causes data corruption
                    keepna=False,
                    multi_level_index=False
                )

                if not yf_data.empty and not yf_data.isnull().all().all():
                    print(f"  ✅ yfinance: {len(remaining_tickers)} tickers")
                    result_frames.append(yf_data)
                    break
                else:
                    if attempt == 0:
                        print(f"  ℹ️  yfinance returned empty data")
                    break

            except Exception as e:
                error_str = str(e).lower()
                if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                    wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  ⚠️ yfinance failed (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠️ yfinance error: {e}")
                    break

    return result_frames


# NOTE: _download_batch_robust function has been moved to data_utils.py
# This file now only contains provider-specific functions and utilities


def _fetch_financial_data(ticker: str) -> pd.DataFrame:
    """Fetch key financial metrics from yfinance and prepare them for merging."""
    time.sleep(PAUSE_BETWEEN_YF_CALLS)
    yf_ticker = yf.Ticker(ticker)

    financial_data = {}

    # Get income statement data (EBITDA)
    try:
        income_stmt = yf_ticker.quarterly_income_stmt
        if not income_stmt.empty and 'EBITDA' in income_stmt.index:
            latest_ebitda = income_stmt.loc['EBITDA'].iloc[-1]
            if not pd.isna(latest_ebitda):
                financial_data['EBITDA'] = float(latest_ebitda)
    except Exception:
        pass

    # Get cash flow data (Free Cash Flow)
    try:
        cash_flow = yf_ticker.quarterly_cash_flow
        if not cash_flow.empty and 'Free Cash Flow' in cash_flow.index:
            latest_fcf = cash_flow.loc['Free Cash Flow'].iloc[-1]
            if not pd.isna(latest_fcf):
                financial_data['FCF'] = float(latest_fcf)
    except Exception:
        pass

    if financial_data:
        df_financial = pd.DataFrame(financial_data, index=[pd.Timestamp.now(tz=timezone.utc)])
        df_financial = df_financial.sort_index()
        return df_financial
    else:
        return pd.DataFrame()


def _fetch_financial_data_from_alpaca(ticker: str) -> pd.DataFrame:
    """Placeholder for fetching financial metrics from Alpaca."""
    return pd.DataFrame()


# NOTE: The rest of this file has been cleaned up.
# All data loading functions (load_prices, load_prices_robust, _download_batch_robust)
# have been moved to data_utils.py for better incremental caching and timezone handling.
