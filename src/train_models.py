"""
Training-Only Script
Trains models for top N stocks (by momentum) WITHOUT running backtests.
Models are saved to logs/models/ for use in live trading.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ticker_selection import get_all_tickers, find_top_performers
from training_phase import train_models_for_period
from data_fetcher import _download_batch_robust
from config import TRAIN_LOOKBACK_DAYS, N_TOP_TICKERS, BATCH_DOWNLOAD_SIZE, PAUSE_BETWEEN_BATCHES, TOP_TICKER_SELECTION_LOOKBACK

print("=" * 80)
print("🤖 AI STOCK ADVISOR - MODEL TRAINING (No Backtest)")
print("=" * 80)
print(f"📅 {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
print("=" * 80)

# 1. Get all available tickers from config
print(f"\n🔍 Step 1: Fetching available tickers from config...")
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=TRAIN_LOOKBACK_DAYS)

try:
    all_available_tickers = get_all_tickers()
    print(f"✅ Found {len(all_available_tickers)} tickers from market selection")
    
    if len(all_available_tickers) == 0:
        print("❌ No tickers found. Check MARKET_SELECTION in config.py")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error fetching tickers: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Download historical data for ALL tickers (needed for momentum filtering)
print(f"\n📥 Step 2: Downloading data for {len(all_available_tickers)} tickers...")
print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"  Lookback: {TRAIN_LOOKBACK_DAYS} days")
print(f"  (Need data for all tickers to calculate momentum)\n")

try:
    all_tickers_data_list = []
    
    # Download in batches
    for i in range(0, len(all_available_tickers), BATCH_DOWNLOAD_SIZE):
        batch = all_available_tickers[i:i + BATCH_DOWNLOAD_SIZE]
        batch_num = i // BATCH_DOWNLOAD_SIZE + 1
        total_batches = (len(all_available_tickers) + BATCH_DOWNLOAD_SIZE - 1) // BATCH_DOWNLOAD_SIZE
        
        print(f"  - Batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
        
        batch_data = _download_batch_robust(batch, start=start_date, end=end_date)
        
        if not batch_data.empty:
            all_tickers_data_list.append(batch_data)
        
        # Pause between batches (except after last batch)
        if i + BATCH_DOWNLOAD_SIZE < len(all_available_tickers):
            print(f"  - Pausing {PAUSE_BETWEEN_BATCHES}s...")
            time.sleep(PAUSE_BETWEEN_BATCHES)
    
    if not all_tickers_data_list:
        print("❌ No data downloaded")
        sys.exit(1)
    
    # Concatenate all batch data
    all_tickers_data = pd.concat(all_tickers_data_list, axis=1)
    
    if all_tickers_data.empty:
        print("❌ Downloaded data is empty")
        sys.exit(1)
    
    # Ensure timezone-aware index
    if all_tickers_data.index.tzinfo is None:
        all_tickers_data.index = all_tickers_data.index.tz_localize('UTC')
    else:
        all_tickers_data.index = all_tickers_data.index.tz_convert('UTC')
    
    print(f"✅ Downloaded data for all tickers")

except Exception as e:
    print(f"❌ Error downloading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Filter to top N by momentum
print(f"\n🎯 Step 3: Filtering to top {N_TOP_TICKERS} by {TOP_TICKER_SELECTION_LOOKBACK} momentum...")

try:
    # For training only (not backtesting), we can use all available data
    # since we're not simulating real-time trading decisions
    top_performers_data = find_top_performers(
        all_available_tickers=all_available_tickers,
        all_tickers_data=all_tickers_data,
        return_tickers=False,
        n_top=N_TOP_TICKERS
    )
    
    tickers = [ticker for ticker, _, _ in top_performers_data]
    print(f"✅ Selected {len(tickers)} top performers")
    
    # Show top 10
    print(f"\n📊 Top 10 by {TOP_TICKER_SELECTION_LOOKBACK} performance:")
    for i, (ticker, perf_1y, _) in enumerate(top_performers_data[:10], 1):
        print(f"  {i}. {ticker}: {perf_1y:+.2f}%")
    
    if len(tickers) == 0:
        print("❌ No valid tickers found")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error filtering top performers: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Train models
print(f"\n🤖 Step 4: Training models for {len(tickers)} tickers...")
print("⏱️  This may take 2-5 hours for 50 tickers...\n")

try:
    training_results = train_models_for_period(
        period_name="Live Trading",
        tickers=tickers,
        all_tickers_data=all_tickers_data,
        train_start=start_date,
        train_end=end_date,
        top_performers_data=top_performers_data,
        feature_set=None,
        run_parallel=True
    )
    
    # ✅ FIX: Convert list of results to dictionaries
    models = {}
    scalers = {}
    y_scalers = {}
    for result in training_results:
        if result and result.get('status') in ['trained', 'loaded']:
            ticker = result['ticker']
            models[ticker] = result['model']
            scalers[ticker] = result['scaler']
            if result.get('y_scaler'):
                y_scalers[ticker] = result['y_scaler']
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"📊 Models trained: {len(models)}")
    print(f"📁 Saved to: logs/models/")
    print(f"🎯 Ready for live trading: python src/live_trading.py")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

