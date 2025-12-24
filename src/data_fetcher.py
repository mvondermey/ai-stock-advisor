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

    ALPACA_API_KEY, ALPACA_SECRET_KEY, TWELVEDATA_API_KEY,

    PAUSE_BETWEEN_BATCHES, PAUSE_BETWEEN_YF_CALLS, INVESTMENT_PER_STOCK

)

from utils import _ensure_dir, _to_utc





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
    tickers_to_download = []

    print(f"  📂 Checking cache for {len(tickers)} tickers...")

    for ticker in tickers:
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

                # Get available date range in cache
                if not cached_df.empty:
                    cache_start = cached_df.index.min()
                    cache_end = cached_df.index.max()

                    # Check if cache contains the required date range
                    # Cache must cover 100% of requested period
                    requested_days = (end_utc - start_utc).days
                    overlap_start = max(start_utc, cache_start)
                    overlap_end = min(end_utc, cache_end)
                    overlap_days = max(0, (overlap_end - overlap_start).days)

                    coverage_ratio = overlap_days / requested_days if requested_days > 0 else 0

                    if coverage_ratio >= 1.0:  # Cache must cover 100% of requested period
                        # Filter to requested date range
                        filtered_df = cached_df.loc[(cached_df.index >= start_utc) & (cached_df.index <= end_utc)].copy()

                        if not filtered_df.empty:
                            # Reformat column names to match yfinance format (ticker as column prefix)
                            if isinstance(filtered_df.columns, pd.MultiIndex):
                                filtered_df.columns = filtered_df.columns.get_level_values(0)
                            filtered_df.columns = [str(col).capitalize() for col in filtered_df.columns]

                            # Add ticker prefix to match yfinance output format
                            filtered_df = filtered_df.add_prefix(f"{ticker} ")

                            cached_data_frames.append(filtered_df)
                            cache_valid = True
                            print(f"  ✅ Cache hit for {ticker} ({len(filtered_df)} rows, {coverage_ratio:.1%} coverage)")

            except Exception as e:
                print(f"  ⚠️ Could not read cache for {ticker}: {e}")

        if not cache_valid:
            tickers_to_download.append(ticker)
            print(f"  📥 Cache miss for {ticker} - will download")

    # If we have cached data for all tickers, combine and return
    if not tickers_to_download:
        print(f"  🎉 All {len(tickers)} tickers loaded from cache!")
        if cached_data_frames:
            combined_df = pd.concat(cached_data_frames, axis=1, join='outer')
            return combined_df
        else:
            return pd.DataFrame()

    # Download missing tickers
    print(f"  🔄 Downloading {len(tickers_to_download)} tickers from {len(tickers)} total...")

    max_retries = 7
    base_wait_time = 30

    fresh_data = pd.DataFrame()

    for attempt in range(max_retries):
        try:
            fresh_data = yf.download(tickers_to_download, start=start, end=end, interval=DATA_INTERVAL, auto_adjust=True, progress=True, threads=False, keepna=False)

            if fresh_data.empty or fresh_data.isnull().all().all():
                raise ValueError("Batch download failed: DataFrame is empty or all-NaN.")

            print(f"  ✅ Successfully downloaded {len(tickers_to_download)} tickers")
            break

        except Exception as e:
            error_str = str(e).lower()

            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "batch download failed" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                print(f"  ⚠️ Batch download failed for {len(tickers_to_download)} tickers (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ An unexpected error occurred during batch download for {len(tickers_to_download)} tickers: {e}. Skipping batch.")
                return pd.DataFrame()

    if fresh_data.empty:
        print(f"  ❌ Failed to download data for {len(tickers_to_download)} tickers after {max_retries} retries.")
        # Return only cached data if download failed
        if cached_data_frames:
            combined_df = pd.concat(cached_data_frames, axis=1, join='outer')
            return combined_df
        return pd.DataFrame()

    # Combine cached and fresh data
    all_data_frames = cached_data_frames + [fresh_data] if not fresh_data.empty else cached_data_frames

    if all_data_frames:
        combined_df = pd.concat(all_data_frames, axis=1, join='outer')
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

    _ensure_dir(DATA_CACHE_DIR)

    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"

    financial_cache_file = DATA_CACHE_DIR / f"{ticker}_financials.csv"

    

    price_df = pd.DataFrame()

    # For benchmark tickers (QQQ, SPY), always refetch to ensure fresh data
    cache_valid = False
    if cache_file.exists() and ticker not in ['QQQ', 'SPY']:

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

                price_df = cached_df.loc[(cached_df.index >= _to_utc(start)) & (cached_df.index <= _to_utc(end))].copy()

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

            try:
                price_df.to_csv(cache_file)
                if 'Fin_' in price_df.columns.any() if hasattr(price_df.columns, 'any') else any('Fin_' in str(col) for col in price_df.columns):
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



    intermarket_df = pd.concat(all_intermarket_dfs, axis=1)

    intermarket_df.index = pd.to_datetime(intermarket_df.index, utc=True)

    intermarket_df.index.name = "Date"



    for col in intermarket_df.columns:

        intermarket_df[f"{col}_Returns"] = intermarket_df[col].pct_change(fill_method=None).fillna(0)



    return intermarket_df.ffill().bfill().fillna(0)
