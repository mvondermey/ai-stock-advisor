import os
import json
import re
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional
from io import StringIO

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
    ALPACA_STOCKS_LIMIT, ALPACA_STOCKS_EXCHANGES
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
                
                # ✅ Use Alpaca's built-in attributes to filter common stocks
                # fractionable=True typically indicates common stocks (not warrants/preferred/rights)
                # If fractionable attribute doesn't exist, assume it's NOT a common stock (safer)
                tradable_assets = [
                    a for a in assets 
                    if a.tradable 
                    and getattr(a, 'fractionable', False)  # ✅ Default False = exclude if unknown
                ]
                
                # Filter by exchange if specified in config
                if ALPACA_STOCKS_EXCHANGES:
                    tradable_assets = [a for a in tradable_assets if a.exchange in ALPACA_STOCKS_EXCHANGES]
                
                # Get symbols
                alpaca_tickers = [asset.symbol for asset in tradable_assets]
                print(f"   📊 After fractionable filter: {len(alpaca_tickers)} tickers")
                
                # ✅ Additional filtering: Remove foreign ADRs and special securities
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
                print(f"   📊 After ADR/special filter: {len(alpaca_tickers)} tickers (filtered out {filtered_count})")
                if filtered_count > 0:
                    print(f"   ✅ Removed {filtered_count} foreign ADRs/special securities (e.g., symbols ending in Y)")
                

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
            print("⚠️ Alpaca stock selection is enabled, but SDK/API keys are not available.")

    if MARKET_SELECTION.get("NASDAQ_ALL"):
        try:
            url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
            df = pd.read_csv(url, sep='|')
            df_clean = df.iloc[:-1]
            # ✅ Filter: Test Issue = 'N' (not test), and exclude delisted (ETF column should be 'N' for stocks)
            # Also check if there's a delisting indicator
            nasdaq_tickers = df_clean[
                (df_clean['Test Issue'] == 'N') &  # Not a test issue
                (df_clean.get('Financial Status', 'N') != 'D')  # Not delisted (if column exists)
            ]['Symbol'].tolist()
            all_tickers.update(nasdaq_tickers)
            print(f"✅ Fetched {len(nasdaq_tickers)} active NASDAQ tickers (delisted excluded).")
        except Exception as e:
            print(f"⚠️ Could not fetch full NASDAQ list ({e}).")

    if MARKET_SELECTION.get("NASDAQ_100"):
        try:
            url_nasdaq = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response_nasdaq = requests.get(url_nasdaq, headers=headers)
            response_nasdaq.raise_for_status()
            table_nasdaq = pd.read_html(StringIO(response_nasdaq.text))[4]
            nasdaq_100_tickers = [s.replace('.', '-') for s in table_nasdaq['Ticker'].tolist()]
            all_tickers.update(nasdaq_100_tickers)
            print(f"✅ Fetched {len(nasdaq_100_tickers)} tickers from NASDAQ 100.")
        except Exception as e:
            print(f"⚠️ Could not fetch NASDAQ 100 list ({e}). Using fallback list.")
            # Fallback list of popular NASDAQ 100 stocks
            nasdaq_100_tickers = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD',
                'INTC', 'CMCSA', 'PEP', 'COST', 'ADBE', 'AVGO', 'TXN', 'QCOM', 'HON', 'AMGN',
                'SBUX', 'INTU', 'AMD', 'ISRG', 'BKNG', 'MDLZ', 'GILD', 'REGN', 'VRTX', 'ILMN',
                'IDXX', 'LRCX', 'KLAC', 'AMAT', 'MU', 'WBD', 'PLTR', 'APP', 'SHOP', 'AZN',
                'ASML', 'MNST', 'CSCO', 'EA', 'TTWO', 'ADI', 'AEP', 'AMGN', 'CCEP', 'EXC'
            ]
            all_tickers.update(nasdaq_100_tickers)
            print(f"✅ Using fallback list with {len(nasdaq_100_tickers)} NASDAQ 100 tickers.")

    if MARKET_SELECTION.get("SP500"):
        try:
            url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response_sp500 = requests.get(url_sp500, headers=headers)
            response_sp500.raise_for_status()
            table_sp500 = pd.read_html(StringIO(response_sp500.text))[0]
            col = "Symbol" if "Symbol" in table_sp500.columns else table_sp500.columns[0]
            sp500_tickers = [s.replace('.', '-') for s in table_sp500[col].tolist()]
            all_tickers.update(sp500_tickers)
            print(f"✅ Fetched {len(sp500_tickers)} tickers from S%26P 500.")
        except Exception as e:
            print(f"⚠️ Could not fetch S%26P 500 list ({e}).")

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
            print(f"✅ Fetched {len(dow_tickers)} tickers from Dow Jones. ")
        except Exception as e:
            print(f"⚠️ Could not fetch Dow Jones list ({e}).")

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
            print(f"✅ Fetched {len(etf_tickers)} tickers from Popular ETFs list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Popular ETFs list ({e}).")

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
                print(f"✅ Fetched {len(crypto_tickers)} tickers from Cryptocurrency list.")
        except Exception as e:
            print(f"⚠️ Could not fetch Cryptocurrency list ({e}).")

    if MARKET_SELECTION.get("DAX"):
        try:
            url_dax = "https://en.wikipedia.org/wiki/DAX"
            response_dax = requests.get(url_dax, headers=headers)
            response_dax.raise_for_status()
            tables_dax = pd.read_html(StringIO(response_dax.text))[0]
            table_dax = None
            for table in tables_dax:
                if 'Ticker' in table.columns:
                    table_dax = table
                    break
            if table_dax is None:
                raise ValueError("Could not find the ticker table on the DAX Wikipedia page.")
            dax_tickers = [s if '.' in s else f"{s}.DE" for s in table_dax['Ticker'].tolist()]
            all_tickers.update(dax_tickers)
            print(f"✅ Fetched {len(dax_tickers)} tickers from DAX.")
        except Exception as e:
            print(f"⚠️ Could not fetch DAX list ({e}).")

    if MARKET_SELECTION.get("MDAX"):
        try:
            url_mdax = "https://en.wikipedia.org/wiki/MDAX"
            response_mdax = requests.get(url_mdax, headers=headers)
            response_mdax.raise_for_status()
            tables_mdax = pd.read_html(StringIO(response_mdax.text))[0]
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
            print(f"✅ Fetched {len(mdax_tickers)} tickers from MDAX.")
        except Exception as e:
            print(f"⚠️ Could not fetch MDAX list ({e}).")

    if MARKET_SELECTION.get("SMI"):
        try:
            url_smi = "https://en.wikipedia.org/wiki/Swiss_Market_Index"
            response_smi = requests.get(url_smi, headers=headers)
            response_smi.raise_for_status()
            tables_smi = pd.read_html(StringIO(response_smi.text))[0]
            table_smi = None
            for table in tables_smi:
                if 'Ticker' in table.columns:
                    table_smi = table
                    break
            if table_smi is None:
                raise ValueError("Could not find the ticker table on the SMI Wikipedia page.")
            smi_tickers = [s if '.' in s else f"{s}.SW" for s in table_smi['Ticker'].tolist()]
            all_tickers.update(smi_tickers)
            print(f"✅ Fetched {len(smi_tickers)} tickers from SMI.")
        except Exception as e:
            print(f"⚠️ Could not fetch SMI list ({e}).")

    if MARKET_SELECTION.get("FTSE_MIB"):
        try:
            url_mib = "https://en.wikipedia.org/wiki/FTSE_MIB"
            response_mib = requests.get(url_mib, headers=headers)
            response_mib.raise_for_status()
            tables_mib = pd.read_html(StringIO(response_mib.text))[0]
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
            print(f"✅ Fetched {len(mib_tickers)} tickers from FTSE MIB.")
        except Exception as e:
            print(f"⚠️ Could not fetch FTSE MIB list ({e}).")

    if not all_tickers:
        print("⚠️ No tickers fetched. Returning empty list.")
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
        print(f"⚠️ Could not fetch S%26P 500 list ({e}). Using static fallback.")
        tickers_all = [_normalize_symbol(sym, DATA_PROVIDER) for sym in fallback]

    import random
    # random.seed(SEED) # SEED is in config, not directly accessible here
    if len(tickers_all) > n:
        selected_tickers = random.sample(tickers_all, n)
    else:
        selected_tickers = tickers_all
    
    print(f"Randomly selected {n} tickers: {', '.join(selected_tickers)}")
    return selected_tickers

def _calculate_performance_worker(params: Tuple[str, pd.DataFrame]) -> Optional[Tuple[str, float, pd.DataFrame]]:
    """Robust 1Y performance: tolerate gaps, weekends, and column variants."""
    ticker, df_1y = params
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

        start_price = s.iloc[0]
        end_price = s.iloc[-1]
        if start_price <= 0:
            return None

        perf_1y = (end_price / start_price - 1.0) * 100.0
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
        print("❌ No ticker data provided to find_top_performers. Exiting.")
        return []

    # Use provided performance end date, or fall back to data's max date
    if performance_end_date is not None:
        end_date = performance_end_date
    else:
        # ✅ FIX: Handle both long-format (with 'date' column) and wide-format (DatetimeIndex)
        if 'date' in all_tickers_data.columns:
            # Long format: dates are in 'date' column
            end_date = pd.to_datetime(all_tickers_data['date']).max()
        else:
            # Wide format: dates are in index
            end_date = all_tickers_data.index.max()

    start_date = end_date - timedelta(days=365)
    ytd_start_date = datetime(end_date.year, 1, 1, tzinfo=timezone.utc)

    final_benchmark_perf = -np.inf
    ytd_benchmark_perf = -np.inf
    if USE_PERFORMANCE_BENCHMARK:
        print("- Calculating 1-Year Performance Benchmarks...")
        benchmark_perfs = {}
        for bench_ticker in ['QQQ', 'SPY']:
            try:
                df = load_prices_robust(bench_ticker, start_date, end_date)
                if df is not None and not df.empty:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    if start_price > 0:
                        perf = ((end_price - start_price) / start_price) * 100
                        benchmark_perfs[bench_ticker] = perf
                        print(f"  ✅ {bench_ticker} 1-Year Performance: {perf:.2f}%")
            except Exception as e:
                print(f"⚠️ Could not calculate {bench_ticker} performance: {e}.")
        
        if not benchmark_perfs:
            print("❌ Could not calculate any benchmark performance. Cannot proceed.")
            return []
            
        final_benchmark_perf = max(benchmark_perfs.values())
        print(f"  📈 Using final 1-Year performance benchmark of {final_benchmark_perf:.2f}%")
    else:
        print("ℹ️ Performance benchmark is disabled. All tickers will be considered.")

    print("🔍 Calculating 1-Year performance from pre-fetched data...")
    
    # ✅ FIX: Handle both long-format and wide-format data
    if 'date' in all_tickers_data.columns and 'ticker' in all_tickers_data.columns:
        # Long format: filter using boolean indexing
        all_data = all_tickers_data[
            (all_tickers_data['date'] >= start_date) & 
            (all_tickers_data['date'] <= end_date)
        ].copy()
        valid_tickers = all_data['ticker'].unique()
        
        params = []
        for ticker in valid_tickers:
            try:
                # Filter data for this ticker
                ticker_data = all_data[all_data['ticker'] == ticker].copy()
                
                # Find Close column
                close_col = None
                for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
                    if attr in ticker_data.columns:
                        close_col = attr
                        break
                
                if close_col is None:
                    continue
                
                # Create time series with date index
                ticker_data = ticker_data.set_index('date')
                s = pd.to_numeric(ticker_data[close_col], errors='coerce')
                s = s.ffill().bfill()
                
                if s.dropna().shape[0] < 2:
                    continue
                    
                ticker_df = pd.DataFrame({'Close': s})
                params.append((ticker, ticker_df))
            except (KeyError, ValueError) as e:
                pass
    else:
        # Wide format: use original logic
        all_data = all_tickers_data.loc[start_date:end_date]
        valid_tickers = all_data.columns.get_level_values(1).unique()

        params = []
        for ticker in valid_tickers:
            try:
                close_key = None
                for attr in ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']:
                    if (attr, ticker) in all_data.columns:
                        close_key = (attr, ticker)
                        break
                if close_key is None:
                    continue
                s = pd.to_numeric(all_data.loc[:, close_key], errors='coerce')
                s = s.ffill().bfill()
                s = s.loc[start_date:end_date]
                if s.dropna().shape[0] < 2:
                    continue
                ticker_data = pd.DataFrame({'Close': s})
                params.append((ticker, ticker_data))
            except KeyError:
                pass
    
    print("...performance calculation complete.")

    all_tickers_performance_with_df = []
    with Pool() as pool: # Use default number of processes
        results = list(tqdm(pool.imap(_calculate_performance_worker, params), total=len(params), desc="Calculating 1Y Performance"))
        for res in results:
            if res:
                all_tickers_performance_with_df.append(res)

    if not all_tickers_performance_with_df:
        print("❌ No tickers with valid 1-Year performance found. Aborting.")
        return []

    sorted_all_tickers_performance_with_df = sorted(all_tickers_performance_with_df, key=lambda item: item[1], reverse=True)
    
    sorted_all_tickers_performance_with_df = [item for item in sorted_all_tickers_performance_with_df if item[1] < 1000]

    if n_top > 0:
        final_performers_for_selection = sorted_all_tickers_performance_with_df[:n_top]
        print(f"\n✅ Selected top {len(final_performers_for_selection)} tickers based on 1-Year performance.")
    else:
        final_performers_for_selection = sorted_all_tickers_performance_with_df
        print(f"\n✅ Analyzing all {len(final_performers_for_selection)} tickers (N_TOP_TICKERS is {n_top}).")

    print(f"🔍 Applying performance benchmarks for selected tickers in parallel...")
    
    finalize_params = [
        (ticker, perf_1y, df_1y, ytd_start_date, end_date, final_benchmark_perf, ytd_benchmark_perf, USE_PERFORMANCE_BENCHMARK)
        for ticker, perf_1y, df_1y in final_performers_for_selection
    ]
    performance_data = []

    with Pool() as pool: # Use default number of processes
        results = list(tqdm(pool.imap(_finalize_single_ticker_performance, finalize_params), total=len(finalize_params), desc="Finalizing Top Performers"))
        for res in results:
            if res:
                performance_data.append(res)

    if USE_PERFORMANCE_BENCHMARK:
        print(f"\n✅ Found {len(performance_data)} stocks that passed the performance benchmarks.")
    else:
        print(f"\n✅ Found {len(performance_data)} stocks for analysis (performance benchmark disabled).")
        
    if not performance_data:
        return []

    final_performers = performance_data

    if fcf_min_threshold is not None or ebitda_min_threshold is not None:
        print(f"  🔍 Screening {len(final_performers)} strong performers for fundamental metrics in parallel...")
        
        fundamental_screen_params = [
            (ticker, perf_1y, fcf_min_threshold, ebitda_min_threshold)
            for ticker, perf_1y in final_performers
        ]
        screened_performers = []

        with Pool() as pool: # Use default number of processes
            results = list(tqdm(pool.imap(_apply_fundamental_screen_worker, fundamental_screen_params), total=len(fundamental_screen_params), desc="Applying fundamental screens"))
            for res in results:
                if res:
                    screened_performers.append(res)

        print(f"  ✅ Found {len(screened_performers)} stocks passing the fundamental screens.")
        final_performers = screened_performers

    if return_tickers:
        return final_performers
    
    print(f"\n\n🏆 Stocks Outperforming {final_benchmark_perf:.2f}%) 🏆")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Ticker':<10} | {'Performance':>15}")
    print("-" * 60)
    
    for i, (ticker, perf) in enumerate(final_performers, 1):
        print(f"{i:<5} | {ticker:<10} | {perf:14.2f}%")
    
    print("-" * 60)
    return list(final_tickers)

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
