import os

import time

import random

from datetime import datetime, timedelta, timezone

from pathlib import Path

from typing import List, Dict, Tuple, Optional



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



# Import from config and utils

from config import (

    DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_INTERVAL, DATA_CACHE_DIR, CACHE_DAYS,
    ENABLE_PRICE_CACHE,

    ALPACA_API_KEY, ALPACA_SECRET_KEY, TWELVEDATA_API_KEY,

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

        # Convert DATA_INTERVAL to Alpaca TimeFrame
        if DATA_INTERVAL in ['1d', '1day']:
            timeframe = TimeFrame.Day
        elif DATA_INTERVAL in ['1h', '1hour']:
            timeframe = TimeFrame.Hour
        elif DATA_INTERVAL in ['1m', '1min']:
            timeframe = TimeFrame.Minute
        else:
            # Default to Day for unsupported intervals
            timeframe = TimeFrame.Day

        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start,
            end=end
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



def _fetch_from_twelvedata(ticker: str, start: datetime, end: datetime, api_key: Optional[str] = None) -> pd.DataFrame:

    """Fetch OHLCV from TwelveData using the SDK."""

    key_to_use = api_key if api_key else TWELVEDATA_API_KEY

    if not TWELVEDATA_SDK_AVAILABLE or not key_to_use:

        return pd.DataFrame()



    try:

        tdc = TDClient(apikey=key_to_use)

        

        # Convert DATA_INTERVAL to TwelveData format
        if DATA_INTERVAL in ['1d', '1day']:
            twelivedata_interval = "1day"
        elif DATA_INTERVAL in ['1h', '1hour']:
            twelivedata_interval = "1h"
        elif DATA_INTERVAL in ['30m', '30min']:
            twelivedata_interval = "30min"
        elif DATA_INTERVAL in ['15m', '15min']:
            twelivedata_interval = "15min"
        elif DATA_INTERVAL in ['5m', '5min']:
            twelivedata_interval = "5min"
        elif DATA_INTERVAL in ['1m', '1min']:
            twelivedata_interval = "1min"
        else:
            # Default to 1day for unsupported intervals
            twelivedata_interval = "1day"

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

        print(f"  ⚠️ An error occurred while fetching data from TwelveData SDK for {ticker}: {e}")

        return pd.DataFrame()



def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:

    """A wrapper for load_prices that handles rate limiting with retries and other common API errors."""

    max_retries = 5

    base_wait_time = 5



    for attempt in range(max_retries):

        try:

            result = load_prices(ticker, start, end)
            # Ensure we always return a DataFrame
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

        except Exception as e:

            error_str = str(e).lower()

            if "yftzmissingerror" in error_str or "no timezone found" in error_str:

                print(f"  ℹ️ Skipping {ticker}: Data not available (possibly delisted).")

                return pd.DataFrame()



            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:

                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)

                print(f"  ⚠️ Rate limited trying to fetch {ticker}. Retrying in {wait_time:.2f} seconds...")

                time.sleep(wait_time)

            else:

                print(f"  ⚠️ An unexpected error occurred for {ticker}: {e}. Skipping.")

                return pd.DataFrame()

    

    print(f"  ❌ Failed to load data for {ticker} after {max_retries} retries due to persistent rate limiting.")

    return pd.DataFrame()



def _download_batch_robust(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:

    """Wrapper for yf.download for batches with retry logic and cache checking."""

    # Check cache for each ticker first
    cached_data_frames = []
    cache_summary = []  # Track cache hit info for summary table
    tickers_to_download_full = []  # Tickers needing full download
    tickers_to_download_incremental = {}  # Tickers needing only recent data: {ticker: last_cache_date}

    # Check if we should expect new data today
    now_utc = datetime.now(timezone.utc)
    if not _is_market_day_complete(end):
        print(f"  ℹ️  Market data not available yet (markets closed or processing). Using cached data.")

    if not ENABLE_PRICE_CACHE:
        print(f"  📂 Cache disabled - downloading {len(tickers)} tickers...", flush=True)
        tickers_to_download_full = list(tickers)
        cached_data_frames = []
    else:
        print(f"  📂 Checking cache for {len(tickers)} tickers...")

    for ticker in tickers:
        if not ENABLE_PRICE_CACHE:
            continue
        cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
        cache_valid = False

        # Skip benchmark tickers that should always be fresh
        if cache_file.exists() and ticker not in ['QQQ', 'SPY']:
            try:
                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)

                # Ensure cached DataFrame index is timezone-aware
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')

                # Check if cached data covers the required date range
                start_utc = _to_utc(start)
                end_utc = _to_utc(end)
                if str(DATA_INTERVAL).lower() in ["1d", "1day", "day"]:
                    end_utc = _effective_end_utc_for_daily(end)

                # Get available date range in cache
                if not cached_df.empty:
                    cache_start = cached_df.index.min()
                    cache_end = cached_df.index.max()

                    # ✅ Adjust expected end date based on market hours/weekends
                    # Don't expect today's data if market hasn't closed yet or if it's a weekend
                    expected_end_utc = end_utc
                    if not _is_market_day_complete(end_utc):
                        # If today's data isn't available yet, only expect data up to yesterday
                        expected_end_utc = end_utc - pd.Timedelta(days=1)
                        # Keep going back until we find a trading day
                        while not _is_market_day_complete(expected_end_utc) and expected_end_utc > cache_end:
                            expected_end_utc = expected_end_utc - pd.Timedelta(days=1)

                    # ✅ Calculate gap to determine if we need to download recent data
                    gap_days = max(0, (expected_end_utc - cache_end).days) if cache_end < expected_end_utc else 0
                    
                    # Full cache coverage (check against expected_end_utc, not end_utc)
                    if cache_start <= start_utc and cache_end >= expected_end_utc:
                        # Filter to requested date range
                        filtered_df = cached_df.loc[(cached_df.index >= start_utc) & (cached_df.index <= end_utc)].copy()

                        if not filtered_df.empty:
                            # Reformat column names to match yfinance format (ticker as column prefix)
                            if isinstance(filtered_df.columns, pd.MultiIndex):
                                filtered_df.columns = filtered_df.columns.get_level_values(0)
                            filtered_df.columns = [str(col).capitalize() for col in filtered_df.columns]

                            # Convert to yfinance-like MultiIndex columns: (Field, Ticker)
                            filtered_df.columns = pd.MultiIndex.from_product([filtered_df.columns, [ticker]])

                            cached_data_frames.append(filtered_df)
                            cache_valid = True
                            
                            # Track cache hit info
                            cache_summary.append({
                                'ticker': ticker,
                                'status': 'Cache Hit',
                                'rows': len(filtered_df),
                                'start_date': filtered_df.index.min().strftime('%Y-%m-%d') if not filtered_df.empty else 'N/A',
                                'end_date': filtered_df.index.max().strftime('%Y-%m-%d') if not filtered_df.empty else 'N/A'
                            })
                            
                            print(f"  ✅ Cache hit for {ticker} ({len(filtered_df)} rows, 100% coverage)")
                    
                    # ✅ Partial cache - download only missing recent data (incremental update)
                    elif cache_start <= start_utc and gap_days > 0:
                        # Use cached data + will download only the missing gap
                        filtered_df = cached_df.loc[cached_df.index >= start_utc].copy()
                        
                        if not filtered_df.empty:
                            # Reformat column names
                            if isinstance(filtered_df.columns, pd.MultiIndex):
                                filtered_df.columns = filtered_df.columns.get_level_values(0)
                            filtered_df.columns = [str(col).capitalize() for col in filtered_df.columns]
                            filtered_df.columns = pd.MultiIndex.from_product([filtered_df.columns, [ticker]])
                            
                            cached_data_frames.append(filtered_df)
                            # Track for incremental download (only missing days)
                            tickers_to_download_incremental[ticker] = cache_end
                            print(f"  🔄 Partial cache for {ticker} ({len(filtered_df)} rows, will download last {gap_days} days)")
                        else:
                            # Cache exists but empty after filtering - need full download
                            tickers_to_download_full.append(ticker)
                            print(f"  📥 Cache miss for {ticker} - will download (empty after filter)")
                    else:
                        # Cache doesn't cover the start date - need full download
                        tickers_to_download_full.append(ticker)
                        print(f"  📥 Cache miss for {ticker} - will download (insufficient coverage)")

            except Exception as e:
                print(f"  ⚠️ Could not read cache for {ticker}: {e}")
                tickers_to_download_full.append(ticker)
                print(f"  📥 Cache miss for {ticker} - will download (read error)")
        else:
            # No cache file exists
            tickers_to_download_full.append(ticker)
            if not cache_file.exists():
                print(f"  📥 Cache miss for {ticker} - will download (no cache)")

    # If we have cached data for all tickers, combine and return
    if not tickers_to_download_full and not tickers_to_download_incremental:
        print(f"  🎉 All {len(tickers)} tickers loaded from cache!")
        if cached_data_frames:
            # ✅ FIX: Normalize timezone info before concat
            normalized_frames = []
            for df in cached_data_frames:
                if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df = df.copy()
                    df.index = df.index.tz_localize(None)
                normalized_frames.append(df)
            combined_df = pd.concat(normalized_frames, axis=1, join='outer')
            return combined_df
        else:
            return pd.DataFrame()

    # Download missing tickers
    total_to_download = len(tickers_to_download_full) + len(tickers_to_download_incremental)
    print(f"  🔄 Downloading data for {total_to_download} tickers ({len(tickers_to_download_full)} full, {len(tickers_to_download_incremental)} incremental)...")

    max_retries = 7
    base_wait_time = 30

    fresh_data_frames = []
    
    # ✅ STEP 1: Download full range for tickers needing complete data
    if tickers_to_download_full:
        print(f"  📥 Downloading FULL range for {len(tickers_to_download_full)} tickers...")
        for attempt in range(max_retries):
            try:
                # For daily bars, treat end as inclusive day; yfinance end is exclusive, so add 1 day.
                download_end = end
                if str(DATA_INTERVAL).lower() in ["1d", "1day", "day"]:
                    end_utc_eff = _effective_end_utc_for_daily(end)
                    download_end = (end_utc_eff + pd.Timedelta(days=1)).to_pydatetime()

                fresh_data_full = yf.download(
                    tickers_to_download_full,
                    start=start,
                    end=download_end,
                    interval=DATA_INTERVAL,
                    auto_adjust=True,
                    progress=True,
                    threads=False,
                    keepna=False
                )

                if not fresh_data_full.empty and not fresh_data_full.isnull().all().all():
                    print(f"  ✅ Successfully downloaded full range for {len(tickers_to_download_full)} tickers")
                    fresh_data_frames.append(fresh_data_full)
                    break
                else:
                    raise ValueError("Batch download failed: DataFrame is empty or all-NaN.")

            except Exception as e:
                error_str = str(e).lower()
                if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "batch download failed" in error_str:
                    wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                    print(f"  ⚠️ Full batch download failed (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  ⚠️ Unexpected error during full batch download: {e}.")
                    break
    
    # ✅ STEP 2: Download only missing date ranges for tickers with partial cache (INCREMENTAL)
    if tickers_to_download_incremental:
        print(f"  🔄 Downloading INCREMENTAL updates for {len(tickers_to_download_incremental)} tickers...")
        
        # Group tickers by their last cache date to batch similar ranges together
        date_groups = {}
        for ticker, last_date in tickers_to_download_incremental.items():
            # Round to day to group tickers with similar last dates
            date_key = last_date.date()
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(ticker)
        
        # Download each group with its specific start date
        for cache_date, ticker_group in date_groups.items():
            incremental_start = pd.Timestamp(cache_date) + pd.Timedelta(days=1)
            download_end = end
            if str(DATA_INTERVAL).lower() in ["1d", "1day", "day"]:
                end_utc_eff = _effective_end_utc_for_daily(end)
                download_end = (end_utc_eff + pd.Timedelta(days=1)).to_pydatetime()
            
            # Only download if there's actually data to fetch
            if incremental_start >= download_end:
                print(f"  ℹ️  No new data expected for {len(ticker_group)} tickers (cache is current)")
                continue
            
            # Check if we should expect new data yet (market hours/weekends)
            if not _is_market_day_complete(download_end):
                print(f"  ℹ️  Skipping {len(ticker_group)} tickers - market data not available yet (market closed or weekend)")
                continue
            
            for attempt in range(max_retries):
                try:
                    fresh_data_incremental = yf.download(
                        ticker_group,
                        start=incremental_start,
                        end=download_end,
                        interval=DATA_INTERVAL,
                        auto_adjust=True,
                        progress=False,  # Less verbose for incremental
                        threads=False,
                        keepna=False
                    )

                    if not fresh_data_incremental.empty and not fresh_data_incremental.isnull().all().all():
                        days_downloaded = (pd.Timestamp(download_end) - incremental_start).days
                        print(f"  ✅ Downloaded {days_downloaded} days for {len(ticker_group)} tickers (from {cache_date})")
                        fresh_data_frames.append(fresh_data_incremental)
                        break
                    else:
                        # Empty result might be OK (no new data yet)
                        print(f"  ℹ️  No new data available for {len(ticker_group)} tickers since {cache_date}")
                        break

                except Exception as e:
                    error_str = str(e).lower()
                    if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                        wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                        print(f"  ⚠️ Incremental download failed (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ⚠️ Unexpected error during incremental download: {e}.")
                        break
    
    # Combine all downloaded data
    if not fresh_data_frames:
        print(f"  ⚠️ No new data downloaded.")
        # Return only cached data if download failed
        if cached_data_frames:
            normalized_frames = []
            for df in cached_data_frames:
                if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df = df.copy()
                    df.index = df.index.tz_localize(None)
                normalized_frames.append(df)
            combined_df = pd.concat(normalized_frames, axis=1, join='outer')
            return combined_df
        return pd.DataFrame()
    
    fresh_data = pd.concat(fresh_data_frames, axis=1) if len(fresh_data_frames) > 1 else fresh_data_frames[0]

    # Ensure yfinance batch output uses MultiIndex columns (Field, Ticker)
    if not isinstance(fresh_data.columns, pd.MultiIndex):
        if len(tickers_to_download) == 1:
            t = tickers_to_download[0]
            fresh_data.columns = pd.MultiIndex.from_product([fresh_data.columns, [t]])

    # ✅ Cache freshly downloaded data per ticker so future runs hit cache
    download_summary = []  # Track download info for summary table
    
    if ENABLE_PRICE_CACHE:
        _ensure_dir(DATA_CACHE_DIR)
        try:
            # Normalize index to tz-naive date index for CSV stability (we localize to UTC on read)
            if getattr(fresh_data.index, "tzinfo", None) is not None:
                fresh_data.index = fresh_data.index.tz_convert(None)
            fresh_data.index.name = "Date"

            # Combine both full and incremental download lists
            all_downloaded_tickers = tickers_to_download_full + list(tickers_to_download_incremental.keys())

            for ticker in all_downloaded_tickers:
                try:
                    # Extract single-ticker DF with single-level columns (Field)
                    if isinstance(fresh_data.columns, pd.MultiIndex):
                        if ticker not in fresh_data.columns.get_level_values(1):
                            continue
                        ticker_df = fresh_data.xs(ticker, axis=1, level=1, drop_level=True).copy()
                    else:
                        ticker_df = fresh_data.copy()

                    if ticker_df.empty:
                        continue

                    # Basic normalization for consistent cache reads
                    ticker_df.columns = [str(c).capitalize() for c in ticker_df.columns]
                    ticker_df.index.name = "Date"

                    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
                    
                    # ✅ INCREMENTAL UPDATE: Append to existing cache instead of overwriting
                    if cache_file.exists():
                        try:
                            existing_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                            # Combine old + new data, remove duplicates (keep new data)
                            combined = pd.concat([existing_cache, ticker_df])
                            combined = combined[~combined.index.duplicated(keep='last')]  # Keep latest data for duplicate dates
                            combined = combined.sort_index()
                            combined.to_csv(cache_file)
                            
                            # Track download info for summary
                            download_type = "Incremental" if ticker in tickers_to_download_incremental else "Full"
                            download_summary.append({
                                'ticker': ticker,
                                'type': download_type,
                                'new_rows': len(ticker_df),
                                'start_date': ticker_df.index.min().strftime('%Y-%m-%d') if not ticker_df.empty else 'N/A',
                                'end_date': ticker_df.index.max().strftime('%Y-%m-%d') if not ticker_df.empty else 'N/A',
                                'total_cached': len(combined)
                            })
                            
                            # Show appropriate message
                            if ticker in tickers_to_download_incremental:
                                new_rows = len(ticker_df)
                                print(f"      🔄 Incremental update for {ticker}: added {new_rows} new rows")
                            else:
                                print(f"      ✅ Updated cache for {ticker} (full download)")
                        except Exception as e:
                            # If append fails, just overwrite
                            ticker_df.to_csv(cache_file)
                            print(f"      ⚠️ Cache append failed for {ticker}, overwrote: {e}")
                    else:
                        # New cache file
                        ticker_df.to_csv(cache_file)
                        
                        # Track download info for summary
                        download_summary.append({
                            'ticker': ticker,
                            'type': 'New',
                            'new_rows': len(ticker_df),
                            'start_date': ticker_df.index.min().strftime('%Y-%m-%d') if not ticker_df.empty else 'N/A',
                            'end_date': ticker_df.index.max().strftime('%Y-%m-%d') if not ticker_df.empty else 'N/A',
                            'total_cached': len(ticker_df)
                        })
                        
                        print(f"      ✅ Created new cache for {ticker}")
                except Exception as e:
                    print(f"  ⚠️ Could not cache batch data for {ticker}: {e}")
        except Exception as e:
            print(f"  ⚠️ Could not finalize batch caching: {e}")

    # ✅ Print comprehensive data summary table
    total_tickers = len(cache_summary) + len(download_summary)
    if total_tickers > 0:
        print(f"\n  📊 Data Summary for {total_tickers} Tickers")
        print("  " + "=" * 95)
        print(f"  {'Ticker':<8} {'Status':<15} {'Rows':<8} {'Start Date':<12} {'End Date':<12} {'Info':<20}")
        print("  " + "-" * 95)
        
        # Combine and sort all tickers
        all_summary = []
        
        # Add cache hits
        for info in cache_summary:
            all_summary.append({
                'ticker': info['ticker'],
                'status': info['status'],
                'rows': info['rows'],
                'start_date': info['start_date'],
                'end_date': info['end_date'],
                'info': 'From cache'
            })
        
        # Add downloads
        for info in download_summary:
            all_summary.append({
                'ticker': info['ticker'],
                'status': f"{info['type']} DL",
                'rows': info['total_cached'],
                'start_date': info['start_date'],
                'end_date': info['end_date'],
                'info': f"+{info['new_rows']} new rows"
            })
        
        # Sort by ticker name
        all_summary.sort(key=lambda x: x['ticker'])
        
        # Print up to 25 tickers
        display_limit = 25
        for i, info in enumerate(all_summary):
            if i < display_limit:
                print(f"  {info['ticker']:<8} {info['status']:<15} {info['rows']:<8} {info['start_date']:<12} {info['end_date']:<12} {info['info']:<20}")
        
        if len(all_summary) > display_limit:
            remaining = len(all_summary) - display_limit
            print(f"  ... and {remaining} more tickers")
        
        print("  " + "=" * 95)
        
        # Summary statistics
        cache_hits = len(cache_summary)
        downloads = len(download_summary)
        
        if download_summary:
            total_new_rows = sum(info['new_rows'] for info in download_summary)
            incremental_count = sum(1 for info in download_summary if info['type'] == 'Incremental')
            full_count = sum(1 for info in download_summary if info['type'] == 'Full')
            new_count = sum(1 for info in download_summary if info['type'] == 'New')
            
            print(f"  ✅ {cache_hits} cache hits, {downloads} downloads ({total_new_rows} new rows)")
            print(f"  📋 Download types: {new_count} new, {full_count} full, {incremental_count} incremental")
        else:
            print(f"  ✅ All {cache_hits} tickers loaded from cache (no downloads needed)")
        
        print()

    # Combine cached and fresh data
    all_data_frames = cached_data_frames + [fresh_data] if not fresh_data.empty else cached_data_frames

    if all_data_frames:
        # ✅ FIX: Normalize timezone info before concat (remove timezone awareness)
        normalized_frames = []
        for df in all_data_frames:
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is not None:
                    # Remove timezone info
                    df = df.copy()
                    df.index = df.index.tz_localize(None)
            normalized_frames.append(df)
        
        combined_df = pd.concat(normalized_frames, axis=1, join='outer')
        print(f"  📊 Combined data: {len(cached_data_frames)} cached + {1 if not fresh_data.empty else 0} fresh = {len(combined_df.columns)} total columns")
        return combined_df

    return fresh_data



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



def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:

    """Download and clean data from the selected provider, with an improved local caching mechanism."""

    if ENABLE_PRICE_CACHE:
        _ensure_dir(DATA_CACHE_DIR)

    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"

    financial_cache_file = DATA_CACHE_DIR / f"{ticker}_financials.csv"

    

    price_df = pd.DataFrame()

    # For benchmark tickers (QQQ, SPY), always refetch to ensure fresh data
    cache_valid = False
    if ENABLE_PRICE_CACHE and cache_file.exists() and ticker not in ['QQQ', 'SPY']:

        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)

        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):

            cache_valid = True

            try:

                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)

                # ✅ FIX 5: Ensure cached DataFrame index is timezone-aware before comparison
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')

                start_utc = _to_utc(start)
                end_utc = _to_utc(end)
                if str(DATA_INTERVAL).lower() in ["1d", "1day", "day"]:
                    end_utc = _effective_end_utc_for_daily(end)

                price_df = cached_df.loc[(cached_df.index >= start_utc) & (cached_df.index <= end_utc)].copy()

            except Exception as e:

                print(f"⚠️ Could not read or slice price cache file for {ticker}: {e}. Refetching prices.")

                cache_valid = False

    if not cache_valid:

        if price_df.empty:
            fetch_start = datetime.now(timezone.utc) - timedelta(days=1000)
            fetch_end = datetime.now(timezone.utc)
            start_utc = _to_utc(fetch_start)
            end_utc   = _to_utc(fetch_end)

        

            # Try providers in order: Alpaca -> TwelveData -> Yahoo
            print(f"  🔄 Fetching data for {ticker} using multi-provider fallback...")

            # 1. Try Alpaca first (fastest, best quality)
            if ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
                alpaca_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
                if not alpaca_df.empty:
                    price_df = alpaca_df.copy()
                    print(f"  ✅ Successfully fetched data for {ticker} from Alpaca.")
                else:
                    print(f"  ℹ️ Alpaca fetch failed for {ticker}. Trying TwelveData...")
            else:
                print(f"  ℹ️ Alpaca not available. Trying TwelveData...")

            # 2. Try TwelveData second (re-enabled as backup)
            if price_df.empty and TWELVEDATA_SDK_AVAILABLE and TWELVEDATA_API_KEY:
                try:
                    twelivedata_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
                    if not twelivedata_df.empty:
                        price_df = twelivedata_df.copy()
                        print(f"  ✅ Successfully fetched data for {ticker} from TwelveData.")
                    else:
                        print(f"  ℹ️ TwelveData fetch failed for {ticker}. Trying Yahoo Finance...")
                except Exception as e:
                    print(f"  ⚠️ TwelveData error for {ticker}: {e}. Trying Yahoo Finance...")
            elif price_df.empty:
                print(f"  ℹ️ TwelveData not available. Trying Yahoo Finance...")






            # 3. Try Yahoo Finance as final fallback
            if price_df.empty and USE_YAHOO_FALLBACK:
                try:
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, interval=DATA_INTERVAL, auto_adjust=True, progress=False)
                    if downloaded_df is not None and not downloaded_df.empty:
                        price_df = downloaded_df
                        print(f"  ✅ Successfully fetched data for {ticker} from Yahoo Finance.")
                    else:
                        print(f"  ⚠️ Yahoo Finance returned empty data for {ticker}.")
                except Exception as e:
                    print(f"  ❌ Yahoo Finance failed for {ticker}: {e}")
            elif price_df.empty:
                print(f"  ❌ All data providers failed for {ticker}.")



            # Multi-provider fallback complete



            if price_df.empty:
                return pd.DataFrame()

            if isinstance(price_df.columns, pd.MultiIndex):
                price_df.columns = price_df.columns.get_level_values(0)
            price_df.columns = [str(col).capitalize() for col in price_df.columns]
            if "Close" not in price_df.columns and "Adj close" in price_df.columns:
                price_df = price_df.rename(columns={"Adj close": "Close"})

            if "Volume" in price_df.columns:
                price_df["Volume"] = price_df["Volume"].fillna(0).astype(int)
            else:
                price_df["Volume"] = 0



            # Add financial data if available
            if financial_cache_file.exists():
                try:
                    financial_df = pd.read_csv(financial_cache_file, index_col='Date', parse_dates=True)
                    if financial_df.index.tzinfo is None:
                        financial_df.index = financial_df.index.tz_localize('UTC')
                    else:
                        financial_df.index = financial_df.index.tz_convert('UTC')

                    # Merge financial data
                    price_df = price_df.join(financial_df, how='left')
                except Exception as e:
                    print(f"⚠️ Could not read financial cache file for {ticker}: {e}")

            if ENABLE_PRICE_CACHE:
                try:
                    price_df.to_csv(cache_file)
                    # Check if any column starts with 'Fin_'
                    if any(str(col).startswith('Fin_') for col in price_df.columns):
                        financial_only = price_df[[col for col in price_df.columns if str(col).startswith('Fin_')]].copy()
                        if not financial_only.empty:
                            financial_only.to_csv(financial_cache_file)
                except Exception as e:
                    print(f"⚠️ Could not cache data for {ticker}: {e}")



        result = price_df.loc[(price_df.index >= _to_utc(start)) & (price_df.index <= _to_utc(end))].copy()
        return result if not result.empty else pd.DataFrame()





def _fetch_intermarket_data(start: datetime, end: datetime) -> pd.DataFrame:

    """Fetches intermarket data (e.g., bond yields, commodities, currencies)."""

    intermarket_tickers = {

        '^VIX': 'VIX_Index',

        'DX-Y.NYB': 'DXY_Index',

        'GC=F': 'Gold_Futures',

        'CL=F': 'Oil_Futures',

        '^TNX': 'US10Y_Yield',

        'USO': 'Oil_Price',

        'GLD': 'Gold_Price',

    }



    all_intermarket_dfs = []

    for ticker, name in intermarket_tickers.items():

        try:

            df = load_prices_robust(ticker, start, end)

            if df is not None and not df.empty:

                single_ticker_df = df[['Close']].rename(columns={'Close': name})

                all_intermarket_dfs.append(single_ticker_df)

        except Exception as e:

            print(f"  ⚠️ Could not fetch intermarket data for {ticker} ({name}): {e}")



    if not all_intermarket_dfs:

        return pd.DataFrame()

    # ✅ FIX: Normalize timezone info before concat
    normalized_intermarket = []
    for df in all_intermarket_dfs:
        if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        normalized_intermarket.append(df)

    intermarket_df = pd.concat(normalized_intermarket, axis=1)

    intermarket_df.index = pd.to_datetime(intermarket_df.index, utc=True)

    intermarket_df.index.name = "Date"



    for col in intermarket_df.columns:

        intermarket_df[f"{col}_Returns"] = intermarket_df[col].pct_change(fill_method=None).fillna(0)



    return intermarket_df.ffill().bfill().fillna(0)
