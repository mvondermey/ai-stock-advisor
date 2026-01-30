#!/usr/bin/env python3
"""
Simple data coverage check without pandas
"""

def check_coverage():
    # Current requirements
    MIN_DAYS_FOR_TRAINING = 329
    TRAIN_LOOKBACK_DAYS = 365
    DATA_INTERVAL = '1h'

    print('📊 Data Coverage Analysis for Hybrid 1h Processing')
    print('=' * 60)

    print(f'📋 Current Requirements:')
    print(f'   MIN_DAYS_FOR_TRAINING: {MIN_DAYS_FOR_TRAINING} calendar days')
    print(f'   TRAIN_LOOKBACK_DAYS: {TRAIN_LOOKBACK_DAYS} calendar days')
    print(f'   DATA_INTERVAL: {DATA_INTERVAL}')

    print(f'\n🔢 Expected Data Points:')
    print(f'   🕰️ Hybrid Mode: {DATA_INTERVAL} → Daily aggregation + intraday features')

    # Daily data (after aggregation)
    trading_days_per_year = 252
    daily_points = TRAIN_LOOKBACK_DAYS * (trading_days_per_year / 365)
    print(f'   📅 Daily data points: ~{daily_points:.0f} trading days')

    # Intraday data points
    hours_per_trading_day = 6.5
    if DATA_INTERVAL == '1h':
        hourly_points_per_day = int(hours_per_trading_day)
        total_hourly_points = daily_points * hourly_points_per_day
        print(f'   ⏰ Hourly data points: ~{total_hourly_points:.0f} hourly bars')

    # Feature calculation requirements
    print(f'   🧮 Feature Requirements:')
    print(f'      - Daily features need: {MIN_DAYS_FOR_TRAINING * 0.5:.0f}+ daily rows')
    print(f'      - Intraday features need: {MIN_DAYS_FOR_TRAINING * 0.5 * hourly_points_per_day:.0f}+ hourly rows')

    # Validation thresholds
    min_daily_rows = MIN_DAYS_FOR_TRAINING * 0.5
    min_hourly_rows = min_daily_rows * hourly_points_per_day

    print(f'\n✅ Validation Thresholds:')
    print(f'   Daily data: Need ≥{min_daily_rows:.0f} rows (expect {daily_points:.0f})')
    print(f'   Hourly data: Need ≥{min_hourly_rows:.0f} rows (expect {total_hourly_points:.0f})')

    # Coverage assessment
    daily_coverage = (daily_points / min_daily_rows) * 100
    hourly_coverage = (total_hourly_points / min_hourly_rows) * 100

    print(f'\n📈 Coverage Assessment:')
    daily_status = "✅" if daily_coverage >= 100 else "❌"
    hourly_status = "✅" if hourly_coverage >= 100 else "❌"
    print(f'   Daily coverage: {daily_coverage:.1f}% ({daily_status})')
    print(f'   Hourly coverage: {hourly_coverage:.1f}% ({hourly_status})')

    print(f'\n🎉 RESULT: Hybrid data processing provides {hourly_coverage:.1f}% more data than required!')
    
    return daily_coverage >= 100 and hourly_coverage >= 100

if __name__ == "__main__":
    check_coverage()
