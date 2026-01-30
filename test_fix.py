#!/usr/bin/env python3
"""
Test the new calculation
"""

def test_fix():
    from datetime import datetime, timedelta
    
    # Simulate the new logic
    DATA_INTERVAL = '1h'
    MAX_LOOKBACK_DAYS = 730
    if DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:
        intraday_multiplier = 3.7
        MAX_LOOKBACK_DAYS = int(MAX_LOOKBACK_DAYS * intraday_multiplier)
    
    MAX_LOOKBACK_CALENDAR_DAYS = MAX_LOOKBACK_DAYS + 60
    fetch_start = datetime.now() - timedelta(days=MAX_LOOKBACK_CALENDAR_DAYS)
    fetch_end = datetime.now()
    days_to_fetch = (fetch_end - fetch_start).days
    
    print(f'✅ New calculation:')
    print(f'   DATA_INTERVAL: {DATA_INTERVAL}')
    print(f'   MAX_LOOKBACK_DAYS: {MAX_LOOKBACK_DAYS}')
    print(f'   MAX_LOOKBACK_CALENDAR_DAYS: {MAX_LOOKBACK_CALENDAR_DAYS}')
    print(f'   days_to_fetch: {days_to_fetch}')
    
    # Check if it will show hours
    if DATA_INTERVAL in ['1h', '30m', '15m', '5m', '1m']:
        hours_to_fetch = days_to_fetch * 24
        print(f'   hours_to_fetch: {hours_to_fetch}')
        print(f'   Will show: "Fetching {hours_to_fetch} hours of data"')
    else:
        print(f'   Will show: "Fetching {days_to_fetch} days of data"')

if __name__ == "__main__":
    test_fix()
