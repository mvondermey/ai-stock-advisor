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
    DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_CACHE_DIR, CACHE_DAYS,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, TWELVEDATA_API_KEY,
    PAUSE_BETWEEN_BATCHES, PAUSE_BETWEEN_YF_CALLS, INVESTMENT_PER_STOCK
)
from utils import _ensure_dir, _to_utc

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
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
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
        
        ts = tdc.time_series(
            symbol=ticker,
            interval="1day",
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
            return load_prices(ticker, start, end)
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
    """Wrapper for yf.download for batches with retry logic."""
    max_retries = 7
    base_wait_time = 30

    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True, threads=False, keepna=False)
            
            if data.empty or data.isnull().all().all():
                raise ValueError("Batch download failed: DataFrame is empty or all-NaN.")
                
            return data
        except Exception as e:
            error_str = str(e).lower()
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str or "batch download failed" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 2)
                print(f"  ⚠️ Batch download failed for {len(tickers)} tickers (attempt {attempt + 1}/{max_retries}): {error_str}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ An unexpected error occurred during batch download for {len(tickers)} tickers: {e}. Skipping batch.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to download batch data for {len(tickers)} tickers after {max_retries} retries.")
    return pd.DataFrame()

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
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):
            try:
                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')
                price_df = cached_df.loc[(cached_df.index >= _to_utc(start)) & (cached_df.index <= _to_utc(end))].copy()
            except Exception as e:
                print(f"⚠️ Could not read or slice price cache file for {ticker}: {e}. Refetching prices.")

    if price_df.empty:
        fetch_start = datetime.now(timezone.utc) - timedelta(days=1000)
        fetch_end = datetime.now(timezone.utc)
        start_utc = _to_utc(fetch_start)
        end_utc   = _to_utc(fetch_end)
        
        provider = DATA_PROVIDER.lower()
        
        if provider == 'twelvedata':
            if not TWELVEDATA_API_KEY:
                print("⚠️ TwelveData is selected but API key is missing. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                twelvedata_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
                if not twelvedata_df.empty:
                    price_df = twelvedata_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ TwelveData fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")

        elif provider == 'alpaca':
            if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                print("⚠️ Alpaca is selected but API keys are missing or SDK not available. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                alpaca_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
                if not alpaca_df.empty:
                    price_df = alpaca_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ Alpaca fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")
        
        elif provider == 'stooq':
            stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
            if not stooq_df.empty:
                price_df = stooq_df.copy()
            elif USE_YAHOO_FALLBACK:
                print(f"  ℹ️ Stooq data for {ticker} empty. Falling back to Yahoo.")
                try:
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                    if downloaded_df is not None and not downloaded_df.empty:
                        price_df = downloaded_df.dropna()
                    else:
                        print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                except Exception as e:
                    print(f"  ❌ Yahoo Finance fallback (after Stooq) failed for {ticker}: {e}")
        
        if price_df.empty:
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    price_df = downloaded_df.dropna()
                else:
                    print(f"  ⚠️ Final Yahoo download attempt returned empty data for {ticker}.")
            except Exception as e:
                print(f"  ❌ Final Yahoo download attempt failed for {ticker}: {e}")
            if price_df.empty and pdr is not None and DATA_PROVIDER.lower() != 'stooq':
                print(f"  ℹ️ Yahoo data for {ticker} empty. Falling back to Stooq.")
                stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
                if stooq_df.empty and not ticker.upper().endswith('.US'):
                    stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
                if not stooq_df.empty:
                    price_df = stooq_df.copy()

        if price_df.empty:
            return pd.DataFrame()

        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df.columns = [str(col).capitalize() for col in price_df.columns]
        if "Close" not in price_df.columns and "Adj close" in price_df.columns:
            price_df = price_df.rename(columns={"Adj close": "Close"})

        if "Close" not in price_df.columns:
            return pd.DataFrame()

        price_df.index = pd.to_datetime(price_df.index, utc=True)
        price_df.index.name = "Date"
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        price_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if "Volume" not in price_df.columns:
            price_df["Volume"] = 0
        if "High" not in price_df.columns:
            price_df["High"] = price_df["Close"]
        if "Low" not in price_df.columns:
            price_df["Low"] = price_df["Close"]
        if "Open" not in price_df.columns:
            price_df["Open"] = price_df["Close"]
            
        price_df = price_df.dropna(subset=["Close"])
        price_df = price_df.ffill().bfill()

        if not price_df.empty:
            try:
                price_df.to_csv(cache_file)
            except Exception as e:
                print(f"⚠️ Could not write price cache file for {ticker}: {e}")
                
    financial_df = pd.DataFrame()
    if financial_cache_file.exists():
        file_mod_time = datetime.fromtimestamp(financial_cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=CACHE_DAYS * 4):
            try:
                financial_df = pd.read_csv(financial_cache_file, index_col='Date', parse_dates=True)
                if financial_df.index.tzinfo is None:
                    financial_df.index = financial_df.index.tz_localize('UTC')
                else:
                    financial_df.index = financial_df.index.tz_convert('UTC')
            except Exception as e:
                print(f"⚠️ Could not read financial cache file for {ticker}: {e}. Refetching financials.")
    
    if financial_df.empty:
        if DATA_PROVIDER.lower() == 'alpaca':
            financial_df = _fetch_financial_data_from_alpaca(ticker)
            if financial_df.empty:
                financial_df = _fetch_financial_data(ticker)
        else:
            financial_df = _fetch_financial_data(ticker)
            
        if not financial_df.empty:
            try:
                financial_df.to_csv(financial_cache_file)
            except Exception as e:
                print(f"⚠️ Could not write financial cache file for {ticker}: {e}")

    if not financial_df.empty and not price_df.empty:
        full_date_range = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='D', tz='UTC')
        financial_df_reindexed = financial_df.reindex(full_date_range)
        financial_df_reindexed = financial_df_reindexed.ffill()
        
        final_df = price_df.merge(financial_df_reindexed, left_index=True, right_index=True, how='left')
        final_df.fillna(0, inplace=True)
    else:
        final_df = price_df.copy()

    if 'Sentiment_Score' not in final_df.columns:
        final_df['Sentiment_Score'] = np.random.uniform(-1, 1, len(final_df))
        final_df['Sentiment_Score'] = final_df['Sentiment_Score'].rolling(window=5).mean().fillna(0)

    return final_df.loc[(final_df.index >= _to_utc(start)) & (final_df.index <= _to_utc(end))].copy()

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
            if not df.empty:
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
