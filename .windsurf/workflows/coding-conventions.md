---
description: Coding conventions and standards for the ai-stock-advisor project
---

# Coding Conventions for ai-stock-advisor

## Timezone Handling

**All dates and timestamps MUST be in UTC.**

When working with dates/timestamps:
1. Always convert timestamps to UTC using `tz_localize('UTC')` or `tz_convert('UTC')`
2. Always convert DataFrame indexes to UTC before comparisons
3. Never adapt to the data's timezone - standardize everything to UTC

```python
# Correct pattern for timestamps:
current_ts = pd.Timestamp(current_date)
if current_ts.tz is None:
    current_ts = current_ts.tz_localize('UTC')
elif str(current_ts.tz) != 'UTC':
    current_ts = current_ts.tz_convert('UTC')

# Correct pattern for DataFrame indexes:
if ticker_data.index.tz is None:
    ticker_data = ticker_data.copy()
    ticker_data.index = ticker_data.index.tz_localize('UTC')
elif str(ticker_data.index.tz) != 'UTC':
    ticker_data = ticker_data.copy()
    ticker_data.index = ticker_data.index.tz_convert('UTC')
```

Use helper functions from `shared_strategies.py`:
- `_to_utc(ts)` - Convert a timestamp to UTC
- `_ensure_utc_index(df)` - Ensure DataFrame index is UTC

## No Fallbacks or Default Values

**Never use fallback values when data is insufficient.**

- Do NOT return artificial default values (e.g., `return 0.0`, `return True`)
- Do NOT use `else: return default_value` patterns
- Instead, return empty results (`return []`, `return {}`, `return ''`) or skip the item (`continue`)
- Let the caller handle missing data explicitly

```python
# WRONG - fallback to default
if len(data) < min_required:
    return 0.5  # artificial default

# CORRECT - return empty/skip
if len(data) < min_required:
    return None  # or continue in a loop
```

## No Silent Exceptions

**All exceptions must be logged or re-raised.**

- Never use `except Exception: pass`
- Never use `except Exception: continue` without logging
- Always print error messages or re-raise

```python
# WRONG - silent exception
except Exception as e:
    return None

# CORRECT - log the error
except Exception as e:
    print(f"   ⚠️ Error processing {ticker}: {e}")
    return None
```

## No Random Data

**All operations must be deterministic.**

- Do NOT use `random.sample()`, `random.uniform()`, `np.random.uniform()` in production code
- Use deterministic selection (e.g., `list[:n]` instead of `random.sample(list, n)`)
- Use fixed delays instead of random delays
- `random_state` parameters in ML models are acceptable (for reproducibility)

```python
# WRONG - random selection
selected = random.sample(tickers, n)

# CORRECT - deterministic selection
selected = tickers[:n]
```

## Meta-Strategy Requirements

**Meta-strategies must have minimum data requirements.**

- Require minimum 5 days of data for `_get_consistency()` and `_get_total_returns()`
- Require minimum 20 days for Sharpe ratio and volatility calculations
- Return empty `('', {})` when data is insufficient
- Never use warmup periods or default strategies

## Code Style

- Use type hints for function parameters and return values
- Use docstrings for all public functions
- Do not add or remove comments unless explicitly asked
- Follow existing code patterns in the file
