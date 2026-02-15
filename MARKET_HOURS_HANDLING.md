# Smart Market Hours & Weekend Handling

## Problem: What if today's data isn't available?

The system now intelligently handles scenarios where the latest market data isn't available yet:
- **Before market close** (4pm ET / 9pm UTC)
- **During data processing** (4pm-6pm ET / 9pm-11pm UTC)
- **Weekends** (Saturday & Sunday)
- **Holidays** (when markets are closed)

## Solution: Smart Market Day Detection

### New Function: `_is_market_day_complete()`

```python
def _is_market_day_complete(date: datetime) -> bool:
    """
    Check if we should expect data for a given date.
    Markets close at 4pm ET (9pm UTC), data available ~6pm ET (11pm UTC).
    """
```

**Logic:**
1. **Future dates**: No data available yet → return `False`
2. **Today before 11pm UTC**: Market data still processing → return `False`
3. **Weekends (Sat/Sun)**: No trading → return `False`
4. **Past dates**: Data should be available → return `True`

## How It Works

### Scenario 1: Running During Market Hours (Monday 10am ET)
```
Current time: Monday 2pm UTC (10am ET)
Cache last date: Friday

Expected behavior:
  ℹ️  Market data not available yet (markets closed or processing). Using cached data.
  ✅ Cache hit for AAPL (250 rows, 100% coverage)
  ✅ Cache hit for MSFT (250 rows, 100% coverage)
  
Result: Uses Friday's data, no download attempts
```

### Scenario 2: Running After Market Close (Monday 7pm ET)
```
Current time: Tuesday 12am UTC (8pm ET Monday)
Cache last date: Friday

Expected behavior:
  📂 Checking cache for 500 tickers...
  🔄 Partial cache for AAPL (250 rows, will download last 1 days)
  📥 Downloading INCREMENTAL updates for 500 tickers...
  ✅ Downloaded 1 days for 500 tickers (from Friday)
  
Result: Downloads Monday's data only
```

### Scenario 3: Running on Weekend (Saturday)
```
Current time: Saturday 10am UTC
Cache last date: Friday

Expected behavior:
  ℹ️  Market data not available yet (markets closed or processing). Using cached data.
  ✅ Cache hit for AAPL (250 rows, 100% coverage)
  
Result: Uses Friday's data, no download attempts
```

### Scenario 4: Running Monday After Weekend
```
Current time: Monday 12am UTC (8pm ET Sunday)
Cache last date: Friday

Expected behavior:
  ℹ️  Market data not available yet (markets closed or processing). Using cached data.
  (Monday's data not available until ~11pm UTC Monday)
  
Result: Uses Friday's data until Monday evening
```

### Scenario 5: Stale Cache (Haven't run in weeks)
```
Current time: Monday 12am UTC (8pm ET Sunday)
Cache last date: 2 weeks ago

Expected behavior:
  📂 Checking cache for 500 tickers...
  🔄 Partial cache for AAPL (250 rows, will download last 10 days)
  📥 Downloading INCREMENTAL updates for 500 tickers...
  ✅ Downloaded 10 days for 500 tickers
  
Result: Downloads all missing trading days
```

## Fallback Behavior

### If Download Fails (Network Issue, API Limit, etc.)

The system **gracefully falls back to cached data**:

```python
if not fresh_data_frames:
    print(f"  ⚠️ No new data downloaded.")
    # Return cached data
    if cached_data_frames:
        combined_df = pd.concat(cached_data_frames, axis=1, join='outer')
        return combined_df
```

**Example:**
```
  📥 Downloading INCREMENTAL updates for 500 tickers...
  ⚠️ Incremental download failed: Rate limit exceeded
  ⚠️ No new data downloaded.
  ✅ Using cached data (last updated: 2024-12-27)
  
Analysis will proceed with cached data.
```

## Messages You'll See

### ✅ Good (Normal Operation)
- `ℹ️  Market data not available yet (markets closed or processing). Using cached data.`
  - **Meaning**: It's too early/weekend, using cache is expected
  
- `✅ Cache hit for TICKER (250 rows, 100% coverage)`
  - **Meaning**: Cache has all needed data

- `🔄 Partial cache for TICKER (250 rows, will download last 3 days)`
  - **Meaning**: Cache is good, just needs recent days

### ⚠️ Informational
- `ℹ️  No new data expected for 100 tickers (cache is current)`
  - **Meaning**: Cache is up to date, no download needed

- `ℹ️  Skipping 100 tickers - market data not available yet`
  - **Meaning**: Not attempting download because market hasn't closed

### ⚠️ Warning (Still OK)
- `⚠️ No new data downloaded.`
  - **Meaning**: Download attempted but failed, using cache

## Configuration

**No configuration needed!** The system automatically:
- Detects current time (UTC)
- Checks market hours (US markets: 9:30am-4pm ET)
- Waits for data processing (~2 hours after market close)
- Handles weekends and holidays
- Falls back to cache gracefully

## Benefits

1. **No Wasted API Calls**: Doesn't attempt downloads when data isn't available
2. **No False Cache Misses**: Doesn't mark cache as "stale" on weekends
3. **Always Works**: Falls back to cached data if download fails
4. **Informative Messages**: Clear status about what's happening
5. **Time Zone Aware**: Works correctly regardless of your local time zone

## Technical Details

### Market Hours Reference
- **US Markets**: 9:30am - 4:00pm ET (Mon-Fri)
- **Data Processing**: ~1-2 hours after close
- **Safe Check Time**: 11pm UTC (6pm ET + 1hr buffer)

### Time Zone Conversions
- **9am ET** = 2pm UTC (1pm UTC in summer/DST)
- **4pm ET** = 9pm UTC (8pm UTC in summer/DST)
- **6pm ET** = 11pm UTC (10pm UTC in summer/DST)

*Note: Using 11pm UTC as safe threshold to account for DST and processing delays*

## Files Modified

1. **`src/data_fetcher.py`**: Added `_is_market_day_complete()` and market-aware cache checking
2. **`src/data_utils.py`**: Added same function and logic for single-ticker downloads

