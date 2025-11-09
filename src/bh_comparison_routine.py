import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
from multiprocessing import Pool # Added this import
import time # Added this import, as time.sleep is used later
from tqdm import tqdm # Added this import

# Import necessary components from main.py
from main import (
    load_prices_robust,
    find_top_performers,
    _run_portfolio_backtest,
    get_all_tickers,
    _download_batch_robust, # Added this import
    _calculate_performance_worker, # Added this import
    INVESTMENT_PER_STOCK,
    N_TOP_TICKERS,
    BACKTEST_DAYS,
    BACKTEST_DAYS_3MONTH,
    BACKTEST_DAYS_1MONTH,
    TRAIN_LOOKBACK_DAYS,
    MARKET_SELECTION,
    NUM_PROCESSES,
    BATCH_DOWNLOAD_SIZE,
    PAUSE_BETWEEN_BATCHES,
    USE_PERFORMANCE_BENCHMARK,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    TWELVEDATA_API_KEY,
    ALPACA_AVAILABLE,
    TWELVEDATA_SDK_AVAILABLE,
    _ensure_dir,
    TOP_CACHE_PATH
)

def run_buy_hold_comparison_routine():
    print("\n" + "="*80)
    print("             📈 Buy & Hold Performance Comparison Routine 📈")
    print("="*80)

    end_date = datetime.now(timezone.utc)
    
    # Determine the absolute earliest date needed for any calculation
    # This needs to cover the longest selection period (1 year) plus the longest backtest period (1 year)
    # and any training lookback (TRAIN_LOOKBACK_DAYS)
    earliest_date_needed = end_date - timedelta(days=max(BACKTEST_DAYS, TRAIN_LOOKBACK_DAYS) + 365 + 1) # 365 for 1-year selection

    print(f"🚀 Step 1: Batch downloading data for all available tickers from {earliest_date_needed.date()} to {end_date.date()}...")
    all_available_tickers = get_all_tickers()
    if not all_available_tickers:
        print("❌ No tickers found from market selection. Aborting comparison.")
        return

    all_tickers_data_list = []
    for i in range(0, len(all_available_tickers), BATCH_DOWNLOAD_SIZE):
        batch = all_available_tickers[i:i + BATCH_DOWNLOAD_SIZE]
        print(f"  - Downloading batch {i//BATCH_DOWNLOAD_SIZE + 1}/{(len(all_available_tickers) + BATCH_DOWNLOAD_SIZE - 1)//BATCH_DOWNLOAD_SIZE} ({len(batch)} tickers)...")
        batch_data = _download_batch_robust(batch, start=earliest_date_needed, end=end_date)
        if not batch_data.empty:
            all_tickers_data_list.append(batch_data)
        if i + BATCH_DOWNLOAD_SIZE < len(all_available_tickers):
            print(f"  - Pausing for {PAUSE_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

    if not all_tickers_data_list:
        print("❌ Comprehensive batch download failed. Aborting comparison.")
        return

    all_tickers_data = pd.concat(all_tickers_data_list, axis=1)
    if all_tickers_data.empty:
        print("❌ Comprehensive batch download failed. Aborting comparison.")
        return
    
    if all_tickers_data.index.tzinfo is None:
        all_tickers_data.index = all_tickers_data.index.tz_localize('UTC')
    else:
        all_tickers_data.index = all_tickers_data.index.tz_convert('UTC')
    print("✅ Comprehensive data download complete.")

    selection_timeframes = {
        "1-Year": BACKTEST_DAYS,
        "YTD": (end_date - datetime(end_date.year, 1, 1, tzinfo=timezone.utc)).days,
        "3-Month": BACKTEST_DAYS_3MONTH,
        "1-Month": BACKTEST_DAYS_1MONTH
    }

    comparison_results = {}

    for name, days_for_selection in selection_timeframes.items():
        print(f"\n🔍 Identifying top {N_TOP_TICKERS} performers based on {name} performance...")
        
        # Calculate the start date for the selection period
        selection_start_date = end_date - timedelta(days=days_for_selection)
        
        # Filter all_tickers_data for the current selection period
        selection_data = all_tickers_data.loc[selection_start_date:end_date]

        # Calculate 1-year performance for all available tickers within this selection_data
        # This is a simplified version of find_top_performers for just performance calculation
        all_tickers_performance_for_selection = []
        
        params_for_selection = []
        valid_tickers_in_selection_data = selection_data.columns.get_level_values(1).unique()
        for ticker in valid_tickers_in_selection_data:
            try:
                ticker_data = selection_data.loc[:, (slice(None), ticker)]
                ticker_data.columns = ticker_data.columns.droplevel(1)
                params_for_selection.append((ticker, ticker_data.copy()))
            except KeyError:
                pass
        
        with Pool(processes=NUM_PROCESSES) as pool:
            results_for_selection = list(tqdm(pool.imap(_calculate_performance_worker, params_for_selection), total=len(params_for_selection), desc=f"Calculating {name} Performance for Selection"))
            for res in results_for_selection:
                if res:
                    all_tickers_performance_for_selection.append(res)

        if not all_tickers_performance_for_selection:
            print(f"  ⚠️ No tickers with valid {name} performance found for selection. Skipping this timeframe.")
            continue

        # Sort and select top N_TOP_TICKERS
        sorted_performers = sorted(all_tickers_performance_for_selection, key=lambda item: item[1], reverse=True)
        top_performers_for_bh = [item[0] for item in sorted_performers[:N_TOP_TICKERS]]

        if not top_performers_for_bh:
            print(f"  ❌ Could not identify top tickers for {name} selection. Skipping.")
            continue

        print(f"  ✅ Top {N_TOP_TICKERS} tickers for {name} selection: {', '.join(top_performers_for_bh)}")

        # Run Buy & Hold backtest for these selected tickers
        print(f"  📈 Running Buy & Hold backtest for {name} selected tickers...")
        
        bh_results_for_selection = []
        total_initial_capital_bh = len(top_performers_for_bh) * INVESTMENT_PER_STOCK

        for ticker in tqdm(top_performers_for_bh, desc=f"Buy & Hold for {name} Selection"):
            df_bh = all_tickers_data.loc[selection_start_date:end_date, (slice(None), ticker)]
            df_bh.columns = df_bh.columns.droplevel(1)
            
            if not df_bh.empty:
                start_price = float(df_bh["Close"].iloc[0])
                shares_bh = int(INVESTMENT_PER_STOCK / start_price) if start_price > 0 else 0
                cash_bh = INVESTMENT_PER_STOCK - shares_bh * start_price
                bh_results_for_selection.append(cash_bh + shares_bh * df_bh["Close"].iloc[-1])
            else:
                bh_results_for_selection.append(INVESTMENT_PER_STOCK) # Assume initial capital if no data

        final_bh_value = sum(bh_results_for_selection)
        bh_return_percentage = ((final_bh_value - total_initial_capital_bh) / total_initial_capital_bh) * 100 if total_initial_capital_bh != 0 else 0.0
        
        # Calculate annualized return percentage
        # Ensure days_for_selection is not zero to avoid division by zero
        if days_for_selection > 0:
            # Convert percentage to decimal for calculation
            decimal_return = bh_return_percentage / 100.0
            # Annualize: (1 + total_return)^(365 / number_of_days) - 1
            annualized_return_percentage = ((1 + decimal_return)**(365 / days_for_selection) - 1) * 100
        else:
            annualized_return_percentage = 0.0 # Or handle as appropriate for 0 days

        comparison_results[name] = {
            "final_value": final_bh_value,
            "return_percentage": bh_return_percentage,
            "annualized_return_percentage": annualized_return_percentage, # Added annualized return
            "top_tickers": top_performers_for_bh
        }
        print(f"  ✅ Buy & Hold for {name} selection: ${final_bh_value:,.2f} ({bh_return_percentage:+.2f}%)")

    print("\n" + "="*80)
    print("             📊 Final Buy & Hold Comparison Summary 📊")
    print("="*80)
    print(f"{'Selection Timeframe':<20} | {'Final Value':>15} | {'Return Percentage':>20} | {'Annualized Return':>20} | {'Top Tickers':<50}")
    print("-" * 132) # Adjusted length for new column

    for name, results in comparison_results.items():
        tickers_str = ", ".join(results["top_tickers"][:5]) + ("..." if len(results["top_tickers"]) > 5 else "")
        print(f"{name:<20} | ${results['final_value']:>13,.2f} | {results['return_percentage']:>18.2f}% | {results['annualized_return_percentage']:>18.2f}% | {tickers_str:<50}")
    print("-" * 132) # Adjusted length for new column
    print("\nNote: 'Top Tickers' shows up to the first 5 tickers for brevity.")

if __name__ == "__main__":
    run_buy_hold_comparison_routine()
