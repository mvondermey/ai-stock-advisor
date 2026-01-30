#!/usr/bin/env python3
"""
Check data coverage for hybrid 1h data processing
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def check_data_requirements():
    """Check current data requirements vs what we'll get with 1h data"""
    try:
        from config import DATA_INTERVAL, AGGREGATE_TO_DAILY, TRAIN_LOOKBACK_DAYS, BACKTEST_DAYS
        from data_validation import MIN_DAYS_FOR_TRAINING, MIN_ROWS_AFTER_FEATURES
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("📊 Data Coverage Analysis for Hybrid 1h Processing")
    print("=" * 60)
    
    # Current requirements
    print(f"\n📋 Current Requirements:")
    print(f"   MIN_DAYS_FOR_TRAINING: {MIN_DAYS_FOR_TRAINING} calendar days")
    print(f"   TRAIN_LOOKBACK_DAYS: {TRAIN_LOOKBACK_DAYS} calendar days")
    print(f"   BACKTEST_DAYS: {BACKTEST_DAYS} calendar days")
    print(f"   DATA_INTERVAL: {DATA_INTERVAL}")
    print(f"   AGGREGATE_TO_DAILY: {AGGREGATE_TO_DAILY}")
    
    # Calculate expected data points
    print(f"\n🔢 Expected Data Points:")
    
    if AGGREGATE_TO_DAILY and DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:
        # Hybrid processing
        print(f"   🕰️ Hybrid Mode: {DATA_INTERVAL} → Daily aggregation + intraday features")
        
        # Daily data (after aggregation)
        trading_days_per_year = 252
        daily_points = TRAIN_LOOKBACK_DAYS * (trading_days_per_year / 365)
        print(f"   📅 Daily data points: ~{daily_points:.0f} trading days")
        
        # Intraday data points
        hours_per_trading_day = 6.5
        if DATA_INTERVAL == '1h':
            hourly_points_per_day = int(hours_per_trading_day)
            total_hourly_points = daily_points * hourly_points_per_day
            print(f"   ⏰ Hourly data points: ~{total_hourly_points:.0f} hourly bars")
            
            # Feature calculation requirements
            print(f"   🧮 Feature Requirements:")
            print(f"      - Daily features need: {MIN_DAYS_FOR_TRAINING * 0.5:.0f}+ daily rows")
            print(f"      - Intraday features need: {MIN_DAYS_FOR_TRAINING * 0.5 * hourly_points_per_day:.0f}+ hourly rows")
            
        # Validation thresholds
        min_daily_rows = MIN_DAYS_FOR_TRAINING * 0.5
        min_hourly_rows = min_daily_rows * hourly_points_per_day
        
        print(f"\n✅ Validation Thresholds:")
        print(f"   Daily data: Need ≥{min_daily_rows:.0f} rows (expect {daily_points:.0f})")
        print(f"   Hourly data: Need ≥{min_hourly_rows:.0f} rows (expect {total_hourly_points:.0f})")
        
        # Coverage assessment
        daily_coverage = (daily_points / min_daily_rows) * 100
        hourly_coverage = (total_hourly_points / min_hourly_rows) * 100
        
        print(f"\n📈 Coverage Assessment:")
        print(f"   Daily coverage: {daily_coverage:.1f}% ({'✅' if daily_coverage >= 100 else '❌'})")
        print(f"   Hourly coverage: {hourly_coverage:.1f}% ({'✅' if hourly_coverage >= 100 else '❌'})")
        
        return daily_coverage >= 100 and hourly_coverage >= 100
        
    else:
        # Traditional daily processing
        print(f"   📅 Daily Mode: Traditional daily data")
        daily_points = TRAIN_LOOKBACK_DAYS * (trading_days_per_year / 365)
        min_daily_rows = MIN_DAYS_FOR_TRAINING * 0.5
        coverage = (daily_points / min_daily_rows) * 100
        
        print(f"   Expected daily points: ~{daily_points:.0f}")
        print(f"   Minimum required: {min_daily_rows:.0f}")
        print(f"   Coverage: {coverage:.1f}% ({'✅' if coverage >= 100 else '❌'})")
        
        return coverage >= 100

def check_actual_data():
    """Check actual data in cache for a sample ticker"""
    try:
        from data_utils import load_hybrid_features
        from config import DATA_INTERVAL, AGGREGATE_TO_DAILY
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    if not AGGREGATE_TO_DAILY:
        print("❌ Hybrid processing not enabled")
        return False
    
    print(f"\n🔍 Checking Actual Data (Sample: SNDK)")
    print("-" * 40)
    
    # Try to get data for SNDK
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year back
    
    try:
        daily_data, intraday_data = load_hybrid_features('SNDK', start_date, end_date)
        
        if daily_data.empty and intraday_data.empty:
            print("❌ No data found - need to run data fetch first")
            return False
        
        print(f"📅 Daily Data:")
        print(f"   Rows: {len(daily_data)}")
        print(f"   Date range: {daily_data.index.min().date()} to {daily_data.index.max().date()}")
        print(f"   Columns: {list(daily_data.columns)}")
        
        print(f"\n⏰ Intraday Data:")
        print(f"   Rows: {len(intraday_data)}")
        print(f"   Date range: {intraday_data.index.min().date()} to {intraday_data.index.max().date()}")
        print(f"   Columns: {list(intraday_data.columns)}")
        
        # Check against requirements
        from data_validation import MIN_DAYS_FOR_TRAINING
        min_daily_rows = MIN_DAYS_FOR_TRAINING * 0.5
        min_hourly_rows = min_daily_rows * 6.5  # Approximate for 1h data
        
        daily_ok = len(daily_data) >= min_daily_rows
        hourly_ok = len(intraday_data) >= min_hourly_rows
        
        print(f"\n✅ Validation:")
        print(f"   Daily: {len(daily_data)} ≥ {min_daily_rows:.0f} → {'✅ PASS' if daily_ok else '❌ FAIL'}")
        print(f"   Hourly: {len(intraday_data)} ≥ {min_hourly_rows:.0f} → {'✅ PASS' if hourly_ok else '❌ FAIL'}")
        
        return daily_ok and hourly_ok
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def main():
    """Run all checks"""
    print("🧪 Hybrid Data Coverage Check")
    print("=" * 60)
    
    # Check theoretical requirements
    theory_ok = check_data_requirements()
    
    # Check actual data if available
    actual_ok = check_actual_data()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Theoretical coverage: {'✅ PASS' if theory_ok else '❌ FAIL'}")
    print(f"   Actual data check: {'✅ PASS' if actual_ok else '❌ FAIL' if actual_ok is False else '⏭️ SKIP'}")
    
    if theory_ok:
        print("\n🎉 Hybrid data processing should provide sufficient coverage!")
        print("   You should have more than enough data for training.")
    else:
        print("\n⚠️  Data coverage may be insufficient.")
        print("   Consider increasing the training period or checking data quality.")
    
    return theory_ok

if __name__ == "__main__":
    main()
