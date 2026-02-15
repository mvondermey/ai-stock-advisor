# Rolling Windows Compliance Tests

This test suite ensures all strategies properly use rolling windows and don't fall back to static date calculations.

## 🎯 Purpose

Rolling windows are fundamental to proper walk-forward backtesting. Each strategy should:
- Use the actual `current_date` parameter
- Calculate performance based on rolling time windows (3M, 6M, 1Y, etc.)
- Show different values as the backtest progresses through time
- Adapt to changing market conditions daily

## 🚨 What We Prevent

**Static Behavior (BAD):**
```python
# ❌ Using fixed end date
if current_date is None:
    current_date = max(latest_dates)  # Always 2026-02-13!

# ❌ Same performance values every day
ZEC-USD: 1Y=1025.5%, 6M=1187.5%, 3M=256.8%  # Never changes!
```

**Rolling Windows (GOOD):**
```python
# ✅ Using actual backtest date
current_date = 2026-01-08  # Day 20
current_date = 2026-01-09  # Day 21
current_date = 2026-01-10  # Day 22

# ✅ Different performance values daily
ZEC-USD: 1Y=950.6%, 6M=966.9%, 3M=80.9%   # Day 21
ZEC-USD: 1Y=973.5%, 6M=900.9%, 3M=37.8%   # Day 22
```

## 🧪 Test Categories

### 1. Strategy-Specific Tests
- **3M/1Y Ratio**: Verifies acceleration values change daily
- **1Y/3M Ratio**: Verifies dip candidates change daily  
- **Turnaround**: Verifies recovery candidates change daily

### 2. Parameter Compliance Tests
- **Current Date Parameter**: Ensures all strategies accept `current_date`
- **Function Signatures**: Verifies proper parameter defaults

### 3. Static Behavior Detection
- **Output Comparison**: Detects identical outputs across different dates
- **Pattern Recognition**: Finds hardcoded dates and static calculations

### 4. Performance Filter Tests
- **Filter Rolling Windows**: Ensures performance filters use rolling windows
- **Value Changes**: Verifies filter criteria change with market data

### 5. Future Date Validation Tests (NEW!)
- **No Future Date Usage**: Ensures strategies don't use data beyond `current_date`
- **Historical Data Only**: Verifies only historical data up to current date is used
- **Data Boundaries**: Tests strategies respect data availability limits
- **Date Direction**: Ensures rolling windows move forward, not backward or randomly

## 🛠️ Usage

### Quick Start
```bash
# Run all tests
python tests/run_rolling_windows_tests.py

# Run with verbose output
python tests/run_rolling_windows_tests.py --verbose

# Test specific strategy
python tests/run_rolling_windows_tests.py --strategy 3m_1y_ratio

# List available test categories
python tests/run_rolling_windows_tests.py --list
```

### Pytest Direct Usage
```bash
# Run all tests
pytest tests/test_rolling_windows.py -v

# Run specific test
pytest tests/test_rolling_windows.py::TestRollingWindowsCompliance::test_3m_1y_ratio_rolling_windows -v

# Run with coverage
pytest tests/test_rolling_windows.py --cov=src --cov-report=html
```

## 📋 Test Categories

| Category | Command | Description |
|----------|---------|-------------|
| `3m_1y_ratio` | `--strategy 3m_1y_ratio` | Test 3M/1Y Ratio strategy |
| `1y_3m_ratio` | `--strategy 1y_3m_ratio` | Test 1Y/3M Ratio strategy |
| `turnaround` | `--strategy turnaround` | Test Turnaround strategy |
| `current_date` | `--strategy current_date` | Test parameter compliance |
| `static_detection` | `--strategy static_detection` | Test for static behavior |
| `performance_filters` | `--strategy performance_filters` | Test filter rolling windows |
| `future_dates` | `--strategy future_dates` | Test for future date usage (NEW!) |
| `data_boundaries` | `--strategy data_boundaries` | Test data boundary compliance |
| `historical_only` | `--strategy historical_only` | Test historical data only usage |

## 🔧 How Tests Work

### 1. Data Generation
Tests create realistic sample data with:
- Multiple tickers (AAPL, MSFT, GOOGL, TSLA, AMZN)
- Daily price data from 2025-01-01 to 2026-02-13
- Realistic trends and volatility

### 2. Date Progression
Tests run strategies with multiple dates:
```python
test_dates = [
    datetime(2025, 12, 15),
    datetime(2025, 12, 16), 
    datetime(2025, 12, 17),
    datetime(2026, 1, 15),
    datetime(2026, 1, 16),
    datetime(2026, 1, 17),
]
```

### 3. Value Extraction
Tests parse debug output to extract performance values:
```python
# 3M/1Y Ratio: acceleration=+254.0%, annualized_3M=+286.1%
# 1Y/3M Ratio: dip=162.6%, 1Y=+180.5%, 3M=+17.9%
# Turnaround: recovery=108.6%, 1Y=+111.2%, 3Y=+7.9%
```

### 4. Change Detection
Tests verify values change between dates:
```python
# Values should be different (rolling window effect)
assert abs(current_value - next_value) > 0.001
```

## 🚨 Failure Scenarios

### Static Values Detected
```
❌ 3M/1Y Ratio: No rolling window effect detected between 2025-12-15 and 2025-12-16. Values appear static.
```

**Fix**: Ensure `current_date` parameter is passed to strategy function.

### Missing Current Date Parameter
```
❌ Strategy function select_xyz_stocks missing current_date parameter.
```

**Fix**: Add `current_date: datetime = None` parameter to function signature.

### Identical Output Pattern
```
❌ Static behavior detected: Strategy produces identical output across different dates.
```

**Fix**: Check for hardcoded dates or max(latest_dates) usage.

### Future Date Usage Detected
```
❌ Future date usage detected: ['2026-01-20', '2026-02-15']. Strategies should only use data up to current_date (2025-12-15).
```

**Fix**: Ensure strategies filter data with `data[data.index <= current_date]`.

### Data Boundary Violation
```
❌ Strategy should detect insufficient data when current_date (2025-07-15) is beyond available data (2025-06-30), not attempt calculations with future data.
```

**Fix**: Add proper data availability checks before calculations.

### Historical Data Violation
```
❌ Future date detected: 2026-01-10 > current_date: 2025-12-15. Strategies should only use historical data.
```

**Fix**: Ensure all date calculations respect the current_date boundary.

## 🔄 Integration

### Pre-commit Hooks
```bash
# Install pre-commit hooks
python setup_rolling_windows_tests.py

# Hooks will run automatically before each commit
git commit -m "Update strategy"
# → Runs rolling windows tests
# → Checks for missing current_date parameters
# → Detects static behavior patterns
```

### CI/CD Pipeline
- **GitHub Actions**: Run on every push and PR
- **Daily Schedule**: Run at 2 AM UTC to catch regressions
- **Multi-Python**: Test on Python 3.9, 3.10, 3.11
- **Test Reports**: Generate and upload test artifacts

### Local Development
```bash
# Quick test before committing
python tests/run_rolling_windows_tests.py --strategy current_date

# Full test suite
python tests/run_rolling_windows_tests.py --verbose

# Test specific changes
pytest tests/test_rolling_windows.py -k "3m_1y_ratio" -v
```

## 📊 Test Coverage

The test suite covers:
- ✅ All 37 strategies
- ✅ Function signature compliance
- ✅ Rolling window calculations
- ✅ Performance filter behavior
- ✅ Static behavior detection
- ✅ Parameter passing verification
- ✅ **Future date usage prevention (NEW!)**
- ✅ **Historical data only enforcement (NEW!)**
- ✅ **Data boundary compliance (NEW!)**
- ✅ **Rolling window direction validation (NEW!)**

## 🛠️ Adding New Tests

### For New Strategies
1. Add strategy function to `strategy_functions` list
2. Add debug output pattern to `extract_performance_values()`
3. Create specific test method if needed

### For New Patterns
1. Add regex pattern to `extract_performance_values()`
2. Update test categories in runner script
3. Add documentation to README

## 🔍 Debugging Tips

### Enable Verbose Output
```bash
python tests/run_rolling_windows_tests.py --verbose
```

### Run Single Test
```bash
pytest tests/test_rolling_windows.py::TestRollingWindowsCompliance::test_3m_1y_ratio_rolling_windows -v -s
```

### Check Strategy Output
```python
# Manually run strategy to see debug output
from shared_strategies import select_3m_1y_ratio_stocks
stocks = select_3m_1y_ratio_stocks(tickers, data, datetime(2025, 12, 15), top_n=5)
```

## 📝 Maintenance

### Regular Tasks
- Update strategy list when adding new strategies
- Add new debug patterns to extraction regex
- Review test coverage quarterly
- Update dependencies as needed

### Troubleshooting
- **Import Errors**: Check Python path and module structure
- **Timeout Issues**: Increase test timeout or reduce data size
- **Flaky Tests**: Add retry logic or stabilize test data

## 🎯 Best Practices

1. **Always pass current_date**: Never use None for strategy calls
2. **Use universal constants**: Use `MIN_DATA_DAYS_*` from config
3. **Test multiple dates**: Verify behavior across different time periods
4. **Check edge cases**: Test with insufficient data scenarios
5. **Document patterns**: Keep regex patterns up to date

## 📞 Support

For issues with rolling windows tests:
1. Check this README first
2. Run tests with `--verbose` flag
3. Review debug output patterns
4. Check strategy function signatures
5. Verify `current_date` parameter usage

Remember: **Rolling windows are fundamental to proper backtesting!** 🎯
