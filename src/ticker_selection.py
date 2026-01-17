import os
import sys
import json
import re
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import financial data fetching function
try:
    from data_fetcher import _fetch_financial_data
except ImportError:
    _fetch_financial_data = None

# Import from config and data_fetcher
from config import (
    DATA_PROVIDER, N_TOP_TICKERS, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES,
    PAUSE_BETWEEN_YF_CALLS, MARKET_SELECTION, USE_PERFORMANCE_BENCHMARK,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, TOP_CACHE_PATH, VALID_TICKERS_CACHE_PATH,
    ALPACA_STOCKS_LIMIT, ALPACA_STOCKS_EXCHANGES, NUM_PROCESSES, PARALLEL_THRESHOLD,
    TOP_TICKER_SELECTION_LOOKBACK
)
from data_fetcher import load_prices_robust, _download_batch_robust
from utils import _ensure_dir, _normalize_symbol, _to_utc

# Optional Alpaca provider for asset listing
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

def get_all_tickers() -> List[str]:
    """
    Gets a list of tickers from the markets selected in the configuration.
    """
    all_tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    if MARKET_SELECTION.get("ALPACA_STOCKS"):
        if ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
                search_params = GetAssetsRequest(
                    asset_class=AssetClass.US_EQUITY,
                    status=AssetStatus.ACTIVE
                )
                assets = trading_client.get_all_assets(search_params)
                
                # Use Alpaca's built-in attributes to filter common stocks
                # fractionable=True typically indicates common stocks (not warrants/preferred/rights)
                # If fractionable attribute doesn't exist, assume it's NOT a common stock (safer)
                tradable_assets = [
                    a for a in assets 
                    if a.tradable 
                    and getattr(a, 'fractionable', False)  # Default False = exclude if unknown
                ]
                
                # Filter by exchange if specified in config
                if ALPACA_STOCKS_EXCHANGES:
                    tradable_assets = [a for a in tradable_assets if a.exchange in ALPACA_STOCKS_EXCHANGES]
                
                # Get symbols
                alpaca_tickers = [asset.symbol for asset in tradable_assets]
                print(f"   After fractionable filter: {len(alpaca_tickers)} tickers")
                
                # Additional filtering: Remove foreign ADRs and special securities
                def is_us_common_stock(symbol: str) -> bool:
                    """Filter out foreign ADRs (ending in Y) and other special securities"""
                    symbol_upper = symbol.upper()
                    # Exclude ADRs (5-letter tickers ending in Y, e.g., BTVCY, ASAZY)
                    if len(symbol_upper) == 5 and symbol_upper.endswith('Y'):
                        return False
                    # Exclude very long symbols (usually special securities)
                    if len(symbol_upper) > 5:
                        return False
                    # Exclude symbols with special characters (except hyphen for class shares like BRK-A)
                    if '$' in symbol_upper or '/' in symbol_upper or '_' in symbol_upper:
                        return False
                    return True
                
                alpaca_tickers_before = len(alpaca_tickers)
                alpaca_tickers = [t for t in alpaca_tickers if is_us_common_stock(t)]
                filtered_count = alpaca_tickers_before - len(alpaca_tickers)
                print(f"   After ADR/special filter: {len(alpaca_tickers)} tickers (filtered out {filtered_count})")
                if filtered_count > 0:
                    print(f"   Removed {filtered_count} foreign ADRs/special securities (e.g., symbols ending in Y)")
                

                # Apply ALPACA_STOCKS_LIMIT to prevent downloading too many stocks
                exchange_filter_desc = f" ({', '.join(ALPACA_STOCKS_EXCHANGES)} only)" if ALPACA_STOCKS_EXCHANGES else ""
                if len(alpaca_tickers) > ALPACA_STOCKS_LIMIT:
                    alpaca_tickers = alpaca_tickers[:ALPACA_STOCKS_LIMIT]
                    print(f"[LIMITED] Fetched {len(alpaca_tickers)} tradable US equity tickers from Alpaca{exchange_filter_desc} (limited to {ALPACA_STOCKS_LIMIT}).")
                else:
                    print(f"[SUCCESS] Fetched {len(alpaca_tickers)} tradable US equity tickers from Alpaca{exchange_filter_desc}.")

                all_tickers.update(alpaca_tickers)
            except Exception as e:
                print(f"[WARNING] Could not fetch asset list from Alpaca ({e}).")
        else:
            print(" Alpaca stock selection is enabled, but SDK/API keys are not available.")

    if MARKET_SELECTION.get("NASDAQ_ALL"):
        try:
            url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
            df = pd.read_csv(url, sep='|')
            df_clean = df.iloc[:-1]
            # Filter: Test Issue = 'N' (not test), and exclude delisted (ETF column should be 'N' for stocks)
            # Also check if there's a delisting indicator
            nasdaq_tickers = df_clean[
                (df_clean['Test Issue'] == 'N') &  # Not a test issue
                (df_clean.get('Financial Status', 'N') != 'D')  # Not delisted (if column exists)
            ]['Symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            print(f" Fetched {len(nasdaq_tickers)} active NASDAQ tickers (delisted excluded).")
        except Exception as e:
            print(f" Could not fetch full NASDAQ list ({e}).")

    if MARKET_SELECTION.get("NASDAQ_100"):
        try:
            url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response_nasdaq = requests.get(url_nasdaq, headers=headers)
            response_nasdaq.raise_for_status()
            table_nasdaq = pd.read_html(StringIO(response_nasdaq.text))[4]
            nasdaq_100_tickers = [s.replace('.', '-') for s in table_nasdaq['Ticker'].tolist()]
            all_tickers.update(nasdaq_100_tickers)
            print(f" Fetched {len(nasdaq_100_tickers)} tickers from NASDAQ 100.")
        except Exception as e:
            print(f" Could not fetch NASDAQ 100 list ({e}). Using fallback list.")
            # Fallback list of popular NASDAQ 100 stocks
            nasdaq_100_tickers = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD',
                'INTC', 'CMCSA', 'PEP', 'COST', 'ADBE', 'AVGO', 'TXN', 'QCOM', 'HON', 'AMGN',
                'SBUX', 'INTU', 'AMD', 'ISRG', 'BKNG', 'MDLZ', 'GILD', 'REGN', 'VRTX', 'ILMN',
                'IDXX', 'LRCX', 'KLAC', 'AMAT', 'MU', 'WBD', 'PLTR', 'APP', 'SHOP', 'AZN',
                'ASML', 'MNST', 'CSCO', 'EA', 'TTWO', 'ADI', 'AEP', 'AMGN', 'CCEP', 'EXC'
            ]
            all_tickers.update(nasdaq_100_tickers)
            print(f" Using fallback list with {len(nasdaq_100_tickers)} NASDAQ 100 tickers.")

    if MARKET_SELECTION.get("SP500"):
        try:
            url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response_sp500 = requests.get(url_sp500, headers=headers)
            response_sp500.raise_for_status()
            table_sp500 = pd.read_html(StringIO(response_sp500.text))[0]
            col = "Symbol" if "Symbol" in table_sp500.columns else table_sp500.columns[0]
            sp500_tickers = [s.replace('.', '-') for s in table_sp500[col].tolist()]
            all_tickers.update(sp500_tickers)
            print(f" Fetched {len(sp500_tickers)} tickers from S%26P 500.")
        except Exception as e:
            print(f" Could not fetch S%26P 500 list ({e}).")

    if MARKET_SELECTION.get("DOW_JONES"):
        try:
            url_dow = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            response_dow = requests.get(url_dow, headers=headers)
            response_dow.raise_for_status()
            tables_dow = pd.read_html(StringIO(response_dow.text))
            table_dow = None
            for table in tables_dow:
                if 'Symbol' in table.columns:
                    table_dow = table
                    break
            if table_dow is None:
                raise ValueError("Could not find the ticker table on the Dow Jones Wikipedia page.")
            col = "Symbol"
            dow_tickers = [str(s).replace('.', '-') for s in table_dow[col].tolist()]
            all_tickers.update(dow_tickers)
            print(f" Fetched {len(dow_tickers)} tickers from Dow Jones. ")
        except Exception as e:
            print(f" Could not fetch Dow Jones list ({e}).")

    if MARKET_SELECTION.get("POPULAR_ETFS"):
        try:
            url_etf = "https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds"
            response_etf = requests.get(url_etf, headers=headers)
            response_etf.raise_for_status()
            
            soup = BeautifulSoup(response_etf.text, 'html.parser')
            etf_tickers = set()
            
            for li in soup.find_all('li'):
                text = li.get_text()
                match = re.search(r'\((?:NYSE\sArca|NASDAQ)[^)]*:([^)]+)\)', text)
                if match:
                    ticker = match.group(1).strip()
                    ticker = ticker.replace('.', '-')
                    etf_tickers.add(ticker)

            if not etf_tickers:
                raise ValueError("No ETF tickers found on the page.")

            all_tickers.update(etf_tickers)
            print(f" Fetched {len(etf_tickers)} tickers from Popular ETFs list.")
        except Exception as e:
            print(f" Could not fetch Popular ETFs list ({e}).")
        
        # Add custom ETF list including leveraged and popular ETFs
        custom_etfs = [
            # Leveraged ETFs
            'QQQ3', 'TQQQ', 'SQQQ',  # 3x NASDAQ-100 (long/short)
            'UPRO', 'SPXU',           # 3x S&P 500 (long/short)
            'TNA', 'TZA',             # 3x Russell 2000 (long/short)
            'TECL', 'TECS',           # 3x Technology (long/short)
            'FAS', 'FAZ',             # 3x Financials (long/short)
            'SOXL', 'SOXS',           # 3x Semiconductors (long/short)
            # Popular Index ETFs
            'SPY', 'QQQ', 'DIA', 'IWM',  # Major indices
            'VOO', 'VTI', 'VEA', 'VWO',  # Vanguard
            # Sector ETFs (for Sector Rotation strategy)
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLC', 'XLB',
            # Additional sector ETFs for rotation
            'GDX', 'USO', 'TLT',  # Gold, Oil, Treasury (used in sector rotation)
            # Bond ETFs
            'AGG', 'BND', 'LQD', 'HYG',
            # Commodity ETFs
            'GLD', 'SLV', 'UNG',
            # International ETFs
            'EFA', 'EEM', 'FXI', 'EWJ',
        ]
        all_tickers.update(custom_etfs)
        print(f" Added {len(custom_etfs)} custom ETFs (including leveraged ETFs like QQQ3, TQQQ)")

    if MARKET_SELECTION.get("CRYPTO"):
        try:
            url_crypto = "https://en.wikipedia.org/wiki/List_of_cryptocurrencies"
            response_crypto = requests.get(url_crypto, headers=headers)
            response_crypto.raise_for_status()
            tables_crypto = pd.read_html(StringIO(response_crypto.text))[0]
            table_crypto = None
            for table in tables_crypto:
                if 'Symbol' in table.columns:
                    table_crypto = table
                    break
            if table_crypto is None:
                raise ValueError("Could not find the ticker table on the Cryptocurrency Wikipedia page.")
            if 'Symbol' in table_crypto.columns:
                crypto_tickers = set()
                for s in table_crypto['Symbol'].tolist():
                    if isinstance(s, str):
                        match = re.match(r'([A-Z]+)', s)
                        if match:
                            crypto_tickers.add(f"{match.group(1)}-USD")
                all_tickers.update(crypto_tickers)
                print(f" Fetched {len(crypto_tickers)} tickers from Cryptocurrency list.")
        except Exception as e:
            print(f" Could not fetch Cryptocurrency list ({e}).")

    if MARKET_SELECTION.get("DAX"):
        try:
            url_dax = "https://en.wikipedia.org/wiki/DAX"
            response_dax = requests.get(url_dax, headers=headers)
            response_dax.raise_for_status()
            tables_dax = pd.read_html(StringIO(response_dax.text))
            table_dax = None
            for table in tables_dax:
                if 'Ticker' in table.columns:
                    table_dax = table
                    break
            if table_dax is None:
                raise ValueError("Could not find the ticker table on the DAX Wikipedia page.")
            dax_tickers = [s if '.' in s else f"{s}.DE" for s in table_dax['Ticker'].tolist()]
            all_tickers.update(dax_tickers)
            print(f" Fetched {len(dax_tickers)} tickers from DAX.")
        except Exception as e:
            print(f" Could not fetch DAX list ({e}).")

    if MARKET_SELECTION.get("MDAX"):
        try:
            url_mdax = "https://en.wikipedia.org/wiki/MDAX"
            response_mdax = requests.get(url_mdax, headers=headers)
            response_mdax.raise_for_status()
            tables_mdax = pd.read_html(StringIO(response_mdax.text))
            table_mdax = None
            for table in tables_mdax:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    table_mdax = table
                    break
            if table_mdax is None:
                raise ValueError("Could not find the ticker table on the MDAX Wikipedia page.")
            ticker_col = 'Ticker' if 'Ticker' in table_mdax.columns else 'Symbol'
            mdax_tickers = [s if '.' in s else f"{s}.DE" for s in table_mdax[ticker_col].tolist()]
            all_tickers.update(mdax_tickers)
            print(f" Fetched {len(mdax_tickers)} tickers from MDAX.")
        except Exception as e:
            print(f" Could not fetch MDAX list ({e}).")

    if MARKET_SELECTION.get("SMI"):
        try:
            url_smi = "https://en.wikipedia.org/wiki/Swiss_Market_Index"
            response_smi = requests.get(url_smi, headers=headers)
            response_smi.raise_for_status()
            tables_smi = pd.read_html(StringIO(response_smi.text))
            table_smi = None
            for table in tables_smi:
                if 'Ticker' in table.columns:
                    table_smi = table
                    break
            if table_smi is None:
                raise ValueError("Could not find the ticker table on the SMI Wikipedia page.")
            smi_tickers = [s if '.' in s else f"{s}.SW" for s in table_smi['Ticker'].tolist()]
            all_tickers.update(smi_tickers)
            print(f" Fetched {len(smi_tickers)} tickers from SMI.")
        except Exception as e:
            print(f" Could not fetch SMI list ({e}).")

    if MARKET_SELECTION.get("FTSE_MIB"):
        try:
            url_mib = "https://en.wikipedia.org/wiki/FTSE_MIB"
            response_mib = requests.get(url_mib, headers=headers)
            response_mib.raise_for_status()
            tables_mib = pd.read_html(StringIO(response_mib.text))
            table_mib = None
            for table in tables_mib:
                if 'Ticker' in table.columns:
                    table_mib = table
                    break
            if table_mib is None:
                raise ValueError("Could not find the ticker table on the FTSE MIB Wikipedia page.")
            ticker_col = 'Ticker'
            mib_tickers = [s if '.' in s else f"{s}.MI" for s in table_mib[ticker_col].tolist()]
            all_tickers.update(mib_tickers)
            print(f" Fetched {len(mib_tickers)} tickers from FTSE MIB.")
        except Exception as e:
            print(f" Could not fetch FTSE MIB list ({e}).")

    if not all_tickers:
        print(" No tickers fetched. Returning empty list.")
        return []

    string_tickers = {str(s) for s in all_tickers if pd.notna(s)}
    
    final_tickers = set()
    for ticker in string_tickers:
        s_ticker = ticker.strip()
        if '$' in s_ticker:
            continue
        
        if s_ticker.endswith(('.DE', '.MI', '.SW', '.PA', '.AS', '.HE', '.LS', '.BR', '.MC')):
            final_tickers.add(s_ticker)
        else:
            final_tickers.add(s_ticker.replace('.', '-'))

    # Always include benchmark tickers to ensure they're cached
    final_tickers.update(['QQQ', 'SPY'])
    
    print(f"Total unique tickers found: {len(final_tickers)}")
    return sorted(list(final_tickers))

def get_tickers_for_backtest(n: int = 10) -> List[str]:
    """Gets a list of n random tickers from the S&P 500."""
    fallback = ["NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "LRCX"]
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        col = "Symbol" if "Symbol" in table.columns else table.columns[0]
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in table[col].tolist()]
    except Exception as e:
        print(f" Could not fetch S%26P 500 list ({e}). Using static fallback.")
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in fallback]

    import random
    # random.seed(SEED) # SEED is in config, not directly accessible here
    if len(tickers_all) > n:
        selected_tickers = random.sample(tickers_all, n)
    else:
        selected_tickers = tickers_all
    
    print(f"Randomly selected {n} tickers: {', '.join(selected_tickers)}")
    return selected_tickers

def _get_comparison_return(ticker: str, end_date: datetime, lookback_days: int = 365) -> Optional[float]:
    """
    Fetch actual return using unified data fetcher (Alpaca -> TwelveData -> Yahoo).
    Returns the actual return % from live data.
    """
    try:
        start_date = end_date - timedelta(days=lookback_days)
        # Use unified data fetcher (Alpaca -> TwelveData -> Yahoo)
        hist = load_prices_robust(ticker, start_date, end_date + timedelta(days=1))
        
        if hist.empty or len(hist) < 10:
            return None
        
        # Find price column
        price_col = next((c for c in ['Close', 'Adj Close', 'Adj close', 'close'] if c in hist.columns), None)
        if price_col is None:
            return None
        
        start_price = hist[price_col].iloc[0]
        end_price = hist[price_col].iloc[-1]
        
        if start_price <= 0:
            return None
        
        return_pct = ((end_price - start_price) / start_price) * 100
        return float(return_pct)
    except Exception:
        return None


def _get_current_1y_return_from_cache(ticker: str, all_tickers_data: dict, today: datetime, lookback_days: int = 365) -> Optional[float]:
    """
    Calculate current 1-year return from cached data (ending today).
    If cached data doesn't have recent data, fetch fresh data.
    This shows the actual 1Y performance as of today, matching what Yahoo Finance shows.
    """
    try:
        # First try to use cached data
        if ticker in all_tickers_data and not all_tickers_data[ticker].empty:
            df = all_tickers_data[ticker]
            
            # Check if we have recent data (within last 7 days)
            latest_date = df.index.max()
            if latest_date >= (today - timedelta(days=7)):
                # We have recent cached data, use it
                return _calculate_1y_return_from_dataframe(df, today, lookback_days)
        
        # If no recent cached data, fetch fresh data
        print(f"   🔄 {ticker}: Fetching fresh data for current 1Y calculation...")
        from data_utils import load_prices_robust
        
        start_date = today - timedelta(days=lookback_days + 30)  # Extra buffer
        fresh_data = load_prices_robust(ticker, start_date, today)
        
        if fresh_data is not None and not fresh_data.empty:
            # Ensure fresh data is timezone-aware like today parameter
            if fresh_data.index.tzinfo is None:
                fresh_data.index = fresh_data.index.tz_localize('UTC')
            else:
                fresh_data.index = fresh_data.index.tz_convert('UTC')
            
            return _calculate_1y_return_from_dataframe(fresh_data, today, lookback_days)
        else:
            return None
            
    except Exception as e:
        print(f"   ⚠️ {ticker}: Error calculating current 1Y return: {e}")
        return None


def _calculate_1y_return_from_dataframe(df: pd.DataFrame, today: datetime, lookback_days: int = 365) -> Optional[float]:
    """Helper function to calculate 1Y return from a DataFrame."""
    try:
        # Find price column
        candidates = ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']
        price_col = next((c for c in candidates if c in df.columns), None)
        if price_col is None:
            return None
        
        # Ensure DataFrame index is timezone-aware like today parameter
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        # Get data for last lookback_days ending today
        start_date = today - timedelta(days=lookback_days)
        
        # Filter to date range
        mask = (df.index >= start_date) & (df.index <= today)
        df_1y = df.loc[mask]
        
        if df_1y.empty or len(df_1y) < 10:
            return None
        
        s = pd.to_numeric(df_1y[price_col], errors='coerce').ffill().bfill().dropna()
        if s.empty or len(s) < 2:
            return None
        
        start_price = s.iloc[0]
        end_price = s.iloc[-1]
        
        if start_price <= 0:
            return None
        
        return_pct = ((end_price / start_price) - 1.0) * 100.0
        return float(return_pct) if np.isfinite(return_pct) else None
    except Exception:
        return None

def _prepare_ticker_data_worker(args: Tuple) -> Optional[Tuple[str, pd.DataFrame]]:
    """Worker function to prepare data for a single ticker (for parallel processing)."""
    ticker, ticker_data_slice = args
    
    try:
        # Find Close column
        close_col = None
        for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
            if attr in ticker_data_slice.columns:
                close_col = attr
                break
        
        if close_col is None:
            return None
        
        # Create time series
        s = pd.to_numeric(ticker_data_slice[close_col], errors='coerce')
        s = s.ffill().bfill()
        
        if s.dropna().shape[0] < 2:
            return None
        
        ticker_df = pd.DataFrame({'Close': s})
        return (ticker, ticker_df)
    
    except Exception:
        return None


def _calculate_performance_worker(params: Tuple[str, pd.DataFrame, int]) -> Optional[Tuple[str, float, pd.DataFrame]]:
    """Robust 1Y performance: tolerate gaps, weekends, and column variants."""
    ticker, df_1y, min_points = params
    try:
        candidates = ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']
        price_col = next((c for c in candidates if c in df_1y.columns), None)
        if price_col is None:
            lower = {c.lower(): c for c in df_1y.columns}
            for key in ['close', 'adj close']:
                if key in lower:
                    price_col = lower[key]
                    break
        if price_col is None:
            return None

        s = pd.to_numeric(df_1y[price_col], errors='coerce').ffill().bfill()
        s = s.dropna()
        if s.empty or len(s) < 2:
            return None
        
        # Data quality check: Require enough points for the configured lookback window
        if len(s) < min_points:
            return None

        start_price = s.iloc[0]
        end_price = s.iloc[-1]
        if start_price <= 0:
            return None

        perf_1y = (end_price / start_price - 1.0) * 100.0
        
        # Data quality check: Flag extreme returns (likely data issues or penny stocks)
        # Filter out stocks with >300% annual returns (usually data quality issues)
        if perf_1y > 300.0:
            start_date = s.index[0]
            end_date = s.index[-1]
            days_of_data = (end_date - start_date).days
            # Only log, don't exclude - let the user see what's happening
            print(f"  High return: {ticker}: {perf_1y:.1f}% | ${start_price:.4f} → ${end_price:.2f} | {days_of_data} days ({len(s)} data points)")
        
        if np.isfinite(perf_1y):
            out = df_1y.copy()
            if price_col != 'Close':
                out = out.rename(columns={price_col: 'Close'})
            return (ticker, float(perf_1y), out)
    except Exception:
        return None
    return None

def find_top_performers(
    all_available_tickers: List[str],
    all_tickers_data: pd.DataFrame,
    return_tickers: bool = False,
    n_top: int = N_TOP_TICKERS,
    fcf_min_threshold: float = 0.0,
    ebitda_min_threshold: float = 0.0,
    performance_end_date: datetime = None
):
    """
    Screens pre-fetched data for the top N performers and returns a list of (ticker, performance) tuples.
    """
    if all_tickers_data.empty:
        print(" No ticker data provided to find_top_performers. Exiting.")
        return []

    # Use provided performance end date, or fall back to data's max date
    if performance_end_date is not None:
        end_date = performance_end_date
    else:
        # FIX: Handle both long-format (with 'date' column) and wide-format (DatetimeIndex)
        if 'date' in all_tickers_data.columns:
            # Long format: dates are in 'date' column
            end_date = pd.to_datetime(all_tickers_data['date']).max()
        else:
            # Wide format: dates are in index
            end_date = all_tickers_data.index.max()

    lookback_setting = str(TOP_TICKER_SELECTION_LOOKBACK).strip().upper() if TOP_TICKER_SELECTION_LOOKBACK is not None else "1Y"
    if lookback_setting in {"3M", "3-M", "3MONTH", "3MONTHS", "90", "90D", "90DAYS"}:
        lookback_days = 90
        lookback_label = "3-Month"
    else:
        lookback_days = 365
        lookback_label = "1-Year"

    start_date = end_date - timedelta(days=lookback_days)
    ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    final_benchmark_perf = -np.inf
    ytd_benchmark_perf = -np.inf
    if USE_PERFORMANCE_BENCHMARK:
        print(f"- Calculating {lookback_label} Performance Benchmarks...")
        benchmark_perfs = {}
        
        # Use pre-fetched data from all_tickers_data instead of re-downloading
        for bench_ticker in ['QQQ', 'SPY']:
            try:
                # Extract benchmark data from all_tickers_data (long format)
                if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
                    # Check if ticker exists in dataset
                    ticker_check = all_tickers_data[all_tickers_data['ticker'] == bench_ticker]
                    if ticker_check.empty:
                        print(f"  {bench_ticker}: Not in dataset (available tickers: {sorted(all_tickers_data['ticker'].unique())[:10]}...)")
                        continue
                    
                    bench_data = all_tickers_data[
                        (all_tickers_data['ticker'] == bench_ticker) &
                        (all_tickers_data['date'] >= start_date) &
                        (all_tickers_data['date'] <= end_date)
                    ].sort_values('date')
                    
                    if bench_data.empty:
                        print(f"  {bench_ticker}: No data in date range {start_date.date()} to {end_date.date()}")
                        print(f"      Available date range: {ticker_check['date'].min().date()} to {ticker_check['date'].max().date()}")
                        continue
                    
                    if 'Close' not in bench_data.columns:
                        print(f"  {bench_ticker}: 'Close' column not found (columns: {list(bench_data.columns)})")
                        continue
                    
                    # Drop NaN values
                    valid_prices = bench_data['Close'].dropna()
                    if len(valid_prices) < 2:
                        print(f"  {bench_ticker}: Insufficient valid prices ({len(valid_prices)} non-NaN values)")
                        continue
                    
                    start_price = valid_prices.iloc[0]
                    end_price = valid_prices.iloc[-1]
                    
                    if pd.isna(start_price) or pd.isna(end_price):
                        print(f"  {bench_ticker}: NaN prices (start={start_price}, end={end_price})")
                        continue
                    
                    if start_price > 0:
                        perf = ((end_price - start_price) / start_price) * 100
                        benchmark_perfs[bench_ticker] = perf
                        print(f"  {bench_ticker} {lookback_label} Performance: {perf:.2f}% (${start_price:.2f} → ${end_price:.2f})")
                    else:
                        print(f"  {bench_ticker}: Invalid start price ({start_price})")
                else:
                    # Fallback to old method if data is in wide format
                    df = load_prices_robust(bench_ticker, start_date, end_date)
                    if df is not None and not df.empty:
                        start_price = df['Close'].iloc[0]
                        end_price = df['Close'].iloc[-1]
                        if start_price > 0:
                            perf = ((end_price - start_price) / start_price) * 100
                            benchmark_perfs[bench_ticker] = perf
                            print(f"  {bench_ticker} {lookback_label} Performance: {perf:.2f}%")
            except Exception as e:
                print(f"  Could not calculate {bench_ticker} performance: {e}")
                import traceback
                traceback.print_exc()
        
        if not benchmark_perfs:
            print(" Could not calculate any benchmark performance. Cannot proceed.")
            return []
            
        final_benchmark_perf = max(benchmark_perfs.values())
        print(f"  Using final {lookback_label} performance benchmark of {final_benchmark_perf:.2f}%")
    else:
        print(" Performance benchmark is disabled. All tickers will be considered.")

    print(f" Calculating {lookback_label} performance from pre-fetched data...")
    
    # FIX: Handle both long-format and wide-format data
    if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
        # Long format: First filter by data quality on FULL dataset, then filter to lookback window
        
        # Step 1: Group by ticker on FULL dataset to check data quality
        print(f"   Checking data quality on full dataset...", flush=True)
        full_grouped = all_tickers_data.groupby('ticker')
        
        # Filter tickers that have at least 252 data points in the FULL dataset
        min_points_quality = 252  # Require 1 year of data for quality
        quality_tickers = []
        for ticker in full_grouped.groups.keys():
            ticker_data = full_grouped.get_group(ticker)
            if len(ticker_data) >= min_points_quality:
                quality_tickers.append(ticker)
        
        print(f"   Found {len(quality_tickers)} tickers with {min_points_quality}+ data points", flush=True)
        
        # Step 2: Now filter to lookback period for performance calculation
        print(f"   Filtering data for period {start_date.date()} to {end_date.date()}...", flush=True)
        all_data = all_tickers_data[
            (all_tickers_data['date'] >= start_date) & 
            (all_tickers_data['date'] <= end_date) &
            (all_tickers_data['ticker'].isin(quality_tickers))
        ].copy()
        
        # Use groupby to split data in ONE operation
        print(f"   Splitting data by ticker using groupby (fast)...", flush=True)
        sys.stdout.flush()
        
        grouped = all_data.groupby('ticker')
        valid_tickers = list(grouped.groups.keys())
        print(f"   Found {len(valid_tickers)} tickers with data in lookback period", flush=True)
        
        # Build prep_args using pre-grouped data (very fast!)
        print(f"   Building parameter list...", flush=True)
        prep_args = []
        for ticker in tqdm(valid_tickers, desc="Building params", ncols=100):
            try:
                ticker_data = grouped.get_group(ticker).copy()
                ticker_data = ticker_data.set_index('date')
                prep_args.append((ticker, ticker_data))
            except KeyError:
                pass
        
        # Parallelize actual data preparation (finding Close column, cleaning data)
        num_prep_workers = min(NUM_PROCESSES, len(prep_args)) if prep_args else 1
        prep_chunksize = max(1, len(prep_args) // (num_prep_workers * 2))
        
        print(f"   Processing {len(prep_args)} tickers with {num_prep_workers} workers (chunksize={prep_chunksize})", flush=True)
        sys.stdout.flush()
        
        params = []
        with Pool(processes=num_prep_workers) as pool:
            prep_results = list(tqdm(
                pool.imap(_prepare_ticker_data_worker, prep_args, chunksize=prep_chunksize),
                total=len(prep_args),
                desc="Processing ticker data",
                ncols=100
            ))
            params = [r for r in prep_results if r is not None]

        # min_points for performance calculation - just need enough points in the lookback window
        # Since we already filtered for quality (252 points in full data), use a lower threshold here
        min_points = 10  # Just need some data points in the lookback window
        params = [(ticker, ticker_df, min_points) for ticker, ticker_df in params]
    else:
        # Wide format: First filter by data quality on FULL dataset, then filter to lookback window
        
        # Step 1: Check data quality on FULL dataset
        print(f"   Checking data quality on full dataset...", flush=True)
        all_tickers_in_data = list(all_tickers_data.columns.get_level_values(1).unique())
        min_points_quality = 252  # Require 1 year of data for quality
        
        def check_single_ticker_quality(ticker):
            try:
                close_key = None
                for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
                    if (attr, ticker) in all_tickers_data.columns:
                        close_key = (attr, ticker)
                        break
                if close_key is not None:
                    s = all_tickers_data.loc[:, close_key].dropna()
                    if len(s) >= min_points_quality:
                        return ticker
            except KeyError:
                pass
            return None
        
        # Use parallel processing for large ticker sets
        if len(all_tickers_in_data) > PARALLEL_THRESHOLD:
            print(f"   Using parallel quality check for {len(all_tickers_in_data)} tickers...")
            num_workers = min(NUM_PROCESSES, len(all_tickers_in_data))
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(check_single_ticker_quality, all_tickers_in_data)
            quality_tickers = [t for t in results if t is not None]
        else:
            # Sequential for small lists
            quality_tickers = []
            for ticker in all_tickers_in_data:
                result = check_single_ticker_quality(ticker)
                if result:
                    quality_tickers.append(result)
        
        print(f"   Found {len(quality_tickers)} tickers with {min_points_quality}+ data points", flush=True)
        
        # Step 2: Now filter to lookback period for performance calculation
        print(f"   Filtering data for period {start_date.date()} to {end_date.date()}...", flush=True)
        all_data = all_tickers_data.loc[start_date:end_date]
        valid_tickers = [t for t in quality_tickers if t in all_data.columns.get_level_values(1)]

        print(f"   Building parameters for {len(valid_tickers)} tickers...", flush=True)
        
        # Prepare data slices (wide format is already column-based, so this is fast)
        prep_args = []
        for ticker in tqdm(valid_tickers, desc="Building params", ncols=100):
            try:
                # Extract close column
                close_key = None
                for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
                    if (attr, ticker) in all_data.columns:
                        close_key = (attr, ticker)
                        break
                
                if close_key is not None:
                    s = all_data.loc[:, close_key]
                    ticker_df = pd.DataFrame({'Close': s})
                    ticker_df.index = all_data.index
                    prep_args.append((ticker, ticker_df))
            except KeyError:
                pass
        
        # Parallelize data processing
        num_prep_workers = min(NUM_PROCESSES, len(prep_args)) if prep_args else 1
        prep_chunksize = max(1, len(prep_args) // (num_prep_workers * 2))
        
        print(f"   Processing {len(prep_args)} tickers with {num_prep_workers} workers (chunksize={prep_chunksize})", flush=True)
        sys.stdout.flush()
        
        params = []
        with Pool(processes=num_prep_workers) as pool:
            prep_results = list(tqdm(
                pool.imap(_prepare_ticker_data_worker, prep_args, chunksize=prep_chunksize),
                total=len(prep_args),
                desc="Processing ticker data",
                ncols=100
            ))
            params = [r for r in prep_results if r is not None]

        # min_points for performance calculation - just need enough points in the lookback window
        # Since we already filtered for quality (252 points in full data), use a lower threshold here
        min_points = 10  # Just need some data points in the lookback window
        params = [(ticker, ticker_df, min_points) for ticker, ticker_df in params]
    
    # Prepare for parallel processing
    if not params:
        print("   No valid tickers found for performance calculation")
        return []
    
    print(f"   Prepared {len(params)} tickers for performance calculation", flush=True)
    
    all_tickers_performance_with_df = []
    # Use configured number of processes for optimal performance
    num_workers = min(NUM_PROCESSES, len(params)) if params else 1
    chunksize = max(1, len(params) // (num_workers * 4)) if params else 1  # Optimal chunking
    
    print(f"   Starting parallel calculation with {num_workers} workers (chunksize={chunksize})", flush=True)
    sys.stdout.flush()
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_calculate_performance_worker, params, chunksize=chunksize), 
            total=len(params), 
            desc=f"Calculating {lookback_label} Performance",
            ncols=100
        ))
        for res in results:
            if res:
                all_tickers_performance_with_df.append(res)
    
    print(f"   Performance calculation complete! Processed {len(all_tickers_performance_with_df)}/{len(params)} tickers")

    if not all_tickers_performance_with_df:
        print(f" No tickers with valid {lookback_label} performance found. Aborting.")
        return []

    sorted_all_tickers_performance_with_df = sorted(all_tickers_performance_with_df, key=lambda item: item[1], reverse=True)
    
    # Filter out extreme positive returns (>1000%) - negative performers will be filtered by benchmark
    sorted_all_tickers_performance_with_df = [item for item in sorted_all_tickers_performance_with_df if item[1] < 1000]

    if n_top > 0:
        final_performers_for_selection = sorted_all_tickers_performance_with_df[:n_top]
        print(f"\n Selected top {len(final_performers_for_selection)} tickers based on {lookback_label} performance.")
    else:
        final_performers_for_selection = sorted_all_tickers_performance_with_df
        print(f"\n Analyzing all {len(final_performers_for_selection)} tickers (N_TOP_TICKERS is {n_top}).")

    print(f"\n Applying performance benchmarks for selected tickers in parallel...")
    
    finalize_params = [
        (ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, USE_PERFORMANCE_BENCHMARK)
        for ticker, perf_1y, df_1y in final_performers_for_selection
    ]
    performance_data = []

    # Use configured number of processes
    num_workers_bench = min(NUM_PROCESSES, len(finalize_params)) if finalize_params else 1
    chunksize_bench = max(1, len(finalize_params) // (num_workers_bench * 4)) if finalize_params else 1
    
    print(f"   Using {num_workers_bench} parallel workers with chunksize={chunksize_bench}")
    
    with Pool(processes=num_workers_bench) as pool:
        results = list(tqdm(
            pool.imap(_finalize_single_ticker_performance, finalize_params, chunksize=chunksize_bench), 
            total=len(finalize_params), 
            desc="Finalizing Top Performers",
            ncols=100
        ))
        for res in results:
            if res:
                performance_data.append(res)

    if USE_PERFORMANCE_BENCHMARK:
        print(f"\n Found {len(performance_data)} stocks that passed the performance benchmarks.")
    else:
        print(f"\n Found {len(performance_data)} stocks for analysis (performance benchmark disabled).")
        
    if not performance_data:
        return []

    final_performers = performance_data

    if fcf_min_threshold is not None or ebitda_min_threshold is not None:
        print(f"  Screening {len(final_performers)} strong performers for fundamental metrics in parallel...")
        
        fundamental_screen_params = [
            (ticker, perf_1y, fcf_min_threshold, ebitda_min_threshold)
            for ticker, perf_1y in final_performers
        ]
        screened_performers = []

        # Use configured number of processes
        num_workers_fund = min(NUM_PROCESSES, len(fundamental_screen_params)) if fundamental_screen_params else 1
        chunksize_fund = max(1, len(fundamental_screen_params) // (num_workers_fund * 4)) if fundamental_screen_params else 1
        
        print(f"   Using {num_workers_fund} parallel workers with chunksize={chunksize_fund}")
        
        with Pool(processes=num_workers_fund) as pool:
            results = list(tqdm(
                pool.imap(_apply_fundamental_screen_worker, fundamental_screen_params, chunksize=chunksize_fund), 
                total=len(fundamental_screen_params), 
                desc="Applying fundamental screens",
                ncols=100
            ))
            for res in results:
                if res:
                    screened_performers.append(res)

        print(f"  Found {len(screened_performers)} stocks passing the fundamental screens.")
        final_performers = screened_performers

    # Fetch actual Yahoo Finance 1-year returns for comparison (parallel)
    print(f"\n  Fetching actual {lookback_label} returns for comparison (top {min(N_TOP_TICKERS, len(final_performers))} tickers)...")
    yahoo_returns = {}
    
    # Fetch for top N_TOP_TICKERS (uses unified Alpaca -> TwelveData -> Yahoo fetcher)
    comparison_limit = min(N_TOP_TICKERS, len(final_performers))
    tickers_to_check = [ticker for ticker, _ in final_performers[:comparison_limit]]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(_get_comparison_return, ticker, end_date, lookback_days): ticker 
            for ticker in tickers_to_check
        }
        
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers_to_check), desc="Fetching comparison returns", ncols=100):
            ticker = future_to_ticker[future]
            try:
                yahoo_return = future.result()
                if yahoo_return is not None:
                    yahoo_returns[ticker] = yahoo_return
            except Exception:
                pass
    
    # Calculate "Current" returns from cached data (ending today, not at backtest start)
    today = datetime.now(timezone.utc)
    current_1y_returns = {}
    
    # Build a dict lookup for ticker data
    ticker_data_dict = {}
    if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
        # Long format
        for ticker in tickers_to_check:
            ticker_df = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
            if not ticker_df.empty:
                ticker_df = ticker_df.set_index('date')
                ticker_data_dict[ticker] = ticker_df
    else:
        # Wide format - data is already in dict-like structure
        ticker_data_dict = all_tickers_data if isinstance(all_tickers_data, dict) else {}
    
    # Calculate current return for each ticker
    for ticker in tickers_to_check:
        current_1y = _get_current_1y_return_from_cache(ticker, ticker_data_dict, today, lookback_days)
        if current_1y is not None:
            current_1y_returns[ticker] = current_1y
    
    # ALWAYS display comparison table (even when return_tickers=True)
    print(f"\n\n Top Performers with Yahoo Finance Comparison ")
    print(f"  NOTE: 'Historical {lookback_label}' = window ending at backtest start ({end_date.strftime('%Y-%m-%d')})")
    print(f"        'Current {lookback_label}' = window ending today ({today.strftime('%Y-%m-%d')}) - matches Yahoo Finance")
    print("-" * 140)
    print(f"{'Rank':<5} | {'Ticker':<10} | {('Historical ' + lookback_label):>14} | {('Current ' + lookback_label):>12} | {('Yahoo ' + lookback_label):>12} | {'Curr vs Yahoo':>14} | {'Status':>12}")
    print("-" * 140)
    
    # Show top 25 for comparison
    for i, (ticker, perf) in enumerate(final_performers[:25], 1):
        yahoo_perf = yahoo_returns.get(ticker)
        current_perf = current_1y_returns.get(ticker)
        
        current_str = f"{current_perf:11.2f}%" if current_perf is not None else "N/A"
        yahoo_str = f"{yahoo_perf:11.2f}%" if yahoo_perf is not None else "N/A"
        
        if current_perf is not None and yahoo_perf is not None:
            diff = current_perf - yahoo_perf
            # Flag suspicious discrepancies > 20% between current and Yahoo
            status = "LARGE DIFF" if abs(diff) > 20 else "Match"
            diff_str = f"{diff:+13.2f}%"
        else:
            diff_str = "N/A"
            status = "No data"
        
        print(f"{i:<5} | {ticker:<10} | {perf:13.2f}% | {current_str:>12} | {yahoo_str:>12} | {diff_str:>14} | {status:>12}")
    
    if len(final_performers) > 25:
        print(f"   ... and {len(final_performers) - 25} more tickers")
    
    print("-" * 140)
    
    # Summary stats
    if yahoo_returns:
        matched_tickers = [t for t in tickers_to_check if t in yahoo_returns]
        if matched_tickers:
            avg_historical = np.mean([perf for ticker, perf in final_performers if ticker in matched_tickers])
            avg_yahoo = np.mean([yahoo_returns[t] for t in matched_tickers])
            
            # Calculate average current 1Y for matched tickers
            current_matched = [t for t in matched_tickers if t in current_1y_returns]
            if current_matched:
                avg_current = np.mean([current_1y_returns[t] for t in current_matched])
                print(f"\n Summary ({len(matched_tickers)} tickers):")
                print(f"   Historical {lookback_label} (ending {end_date.strftime('%Y-%m-%d')}): Avg = {avg_historical:.1f}%")
                print(f"   Current {lookback_label} (ending today):                    Avg = {avg_current:.1f}%")
                print(f"   Yahoo {lookback_label}:                                     Avg = {avg_yahoo:.1f}%")
                print(f"   Current vs Yahoo difference:                  Avg = {avg_current - avg_yahoo:+.1f}%")
            else:
                print(f"\n Summary ({len(matched_tickers)} tickers): Avg Historical = {avg_historical:.1f}%, Avg Yahoo = {avg_yahoo:.1f}%")
            
            # Count large discrepancies between current and Yahoo
            large_diffs = sum(1 for t in current_matched if abs(current_1y_returns.get(t, 0) - yahoo_returns[t]) > 20)
            if large_diffs > 0:
                print(f"  ⚠️ WARNING: {large_diffs} tickers have >20% discrepancy between Current {lookback_label} and Yahoo - may indicate data issues!")
    
    if return_tickers:
        return final_performers
    
    print(f"\n\n Stocks Outperforming {final_benchmark_perf:.2f}% (Full List) ")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Performance':>15}")
    print("-" * 60)
    
    for i, (ticker, perf) in enumerate(final_performers, 1):
        print(f"{i:<5} | {ticker:<10} | {perf:14.2f}%")
    
    print("-" * 60)
    
    return [ticker for ticker, perf in final_performers]

def _finalize_single_ticker_performance(params: Tuple) -> Optional[Tuple[str, float]]:
    """Worker to apply performance benchmarks."""
    ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, use_performance_benchmark = params

    if use_performance_benchmark and perf_1y < final_benchmark_perf:
        return None

    # YTD performance calculation removed since YTD support was removed
    return (ticker, perf_1y)

def _apply_fundamental_screen_worker(params: Tuple) -> Optional[Tuple[str, float]]:
    """Worker to apply fundamental screens using yfinance with proper fallback."""
    ticker, perf_1y, fcf_min_threshold, ebitda_min_threshold = params

    # Use yfinance directly for fundamental data - same as _fetch_financial_data but simplified
    try:
        import yfinance as yf
        import time

        # Small delay to be respectful to APIs
        time.sleep(0.1)  # Shorter delay for multiprocessing

        yf_ticker = yf.Ticker(ticker)
        financial_data = {}

        # Get income statement data (EBITDA)
        try:
            income_stmt = yf_ticker.quarterly_income_stmt
            if not income_stmt.empty and 'EBITDA' in income_stmt.index:
                latest_ebitda = income_stmt.loc['EBITDA'].iloc[-1]
                if not pd.isna(latest_ebitda):
                    financial_data['EBITDA'] = float(latest_ebitda)
        except Exception as e:
            pass  # Silently continue if this fails

        # Get cash flow data (Free Cash Flow)
        try:
            cash_flow = yf_ticker.quarterly_cash_flow
            if not cash_flow.empty and 'Free Cash Flow' in cash_flow.index:
                latest_fcf = cash_flow.loc['Free Cash Flow'].iloc[-1]
                if not pd.isna(latest_fcf):
                    financial_data['FCF'] = float(latest_fcf)
        except Exception as e:
            pass  # Silently continue if this fails

        # Apply thresholds (fcf_min_threshold and ebitda_min_threshold are 0.0 by default)
        should_exclude = False

        if 'EBITDA' in financial_data and ebitda_min_threshold is not None:
            if financial_data['EBITDA'] < ebitda_min_threshold:
                should_exclude = True

        if 'FCF' in financial_data and fcf_min_threshold is not None:
            if financial_data['FCF'] < fcf_min_threshold:
                should_exclude = True

        if should_exclude:
            return None  # Exclude stocks that don't meet financial criteria

        # Include stocks that pass financial screening or have no data available
        return (ticker, perf_1y)

    except Exception as e:
        # If anything fails, include the stock by default (fail-open approach)
        return (ticker, perf_1y)
