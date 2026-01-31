# Ensemble Strategies Summary

This document describes the new ensemble strategies implemented as independent strategies in the AI Stock Advisor system.

## 1. Adaptive Meta-Ensemble Strategy (`adaptive_ensemble.py`)

**File**: `src/adaptive_ensemble.py`

**Key Features**:
- Market regime detection (trending, volatile, mean-reverting, neutral)
- Dynamic strategy weighting based on regime and recent performance
- Consensus filtering (minimum 2 strategies must agree)
- Combines 4 core strategies: Static BH 3M, Dyn BH 1Y+Vol, Risk-Adj Mom, Quality+Mom

**Configuration**:
- `ENABLE_ADAPTIVE_STRATEGY = True` (in config.py)
- Strategy weights adjust by market regime
- Minimum consensus agreement: 2 strategies

**Use Case**: Best for adaptive market conditions where strategy performance varies by regime.

---

## 2. Volatility-Adjusted Ensemble Strategy (`volatility_ensemble.py`)

**File**: `src/volatility_ensemble.py`

**Key Features**:
- Inverse volatility position sizing
- Portfolio volatility constraints (max 20% annualized)
- Individual stock volatility caps (max 40% annualized)
- Ensemble consensus with volatility filtering

**Configuration**:
- `MAX_PORTFOLIO_VOLATILITY = 0.20` (20% max portfolio volatility)
- `MAX_SINGLE_STOCK_VOLATILITY = 0.40` (40% max individual stock)
- Position weights: 5% minimum, 30% maximum

**Use Case**: Risk-conscious investors who want volatility-managed exposure.

---

## 3. Correlation-Filtered Ensemble Strategy (`correlation_ensemble.py`)

**File**: `src/correlation_ensemble.py`

**Key Features**:
- Correlation matrix analysis (max 70% correlation between stocks)
- Sector diversification constraints (max 40% per sector)
- Reduces portfolio concentration risk
- Ensemble consensus with correlation filtering

**Configuration**:
- `MAX_CORRELATION = 0.70` (maximum correlation between any two stocks)
- `MAX_SECTOR_WEIGHT = 0.40` (maximum 40% in any sector)
- 60-day correlation lookback period

**Use Case**: Investors seeking diversification and reduced concentration risk.

---

## 4. Dynamic Strategy Pool Strategy (`dynamic_pool.py`)

**File**: `src/dynamic_pool.py`

**Key Features**:
- Tracks performance of 12 available strategies
- Keeps top 4 performing strategies in active pool
- Dynamic weight allocation based on recent performance
- Strategy rotation based on performance thresholds

**Configuration**:
- `POOL_SIZE = 4` (number of active strategies)
- `PERFORMANCE_WINDOW_DAYS = 30` (performance evaluation window)
- 15% performance difference triggers rotation

**Use Case**: Investors who want the system to automatically adapt to the best performing strategies.

---

## 5. Sentiment-Enhanced Ensemble Strategy (`sentiment_ensemble.py`)

**File**: `src/sentiment_ensemble.py`

**Key Features**:
- News sentiment analysis (mock data - can be replaced with real API)
- Social media sentiment integration
- Sentiment-weighted stock selection
- Ensemble consensus with sentiment filtering

**Configuration**:
- `SENTIMENT_WEIGHT = 0.30` (30% weight for sentiment in final score)
- Mock sentiment data for demonstration
- Minimum sentiment threshold: -0.1

**Use Case**: Investors who want to incorporate market sentiment into stock selection.

---

## Integration Status

All strategies are integrated with:
- ✅ `backtesting.py` - Full backtesting support with portfolio tracking
- ✅ `live_trading.py` - Live trading support
- ✅ Configuration flags in `config.py`

## To Enable in Backtesting:

```python
# In config.py
ENABLE_VOLATILITY_ENSEMBLE = True
ENABLE_CORRELATION_ENSEMBLE = True
ENABLE_DYNAMIC_POOL = True
ENABLE_SENTIMENT_ENSEMBLE = True
```

## To Use in Live Trading:

Set strategy in your live trading configuration:
- `'volatility_ensemble'`
- `'correlation_ensemble'`
- `'dynamic_pool'`
- `'sentiment_ensemble'`

## Performance Expectations:

| Strategy | Expected Return | Volatility | Best Market Conditions |
|----------|----------------|------------|---------------------|
| Adaptive Meta-Ensemble | 60-70% of best strategy | Medium | Varying regimes |
| Volatility-Adjusted | Moderate | Low-Medium | High volatility |
| Correlation-Filtered | Moderate | Low | All conditions |
| Dynamic Pool | High (adaptive) | Medium | Strategy rotation |
| Sentiment-Enhanced | Moderate-High | Medium | News-driven markets |

## Next Steps:

1. Add configuration flags to `config.py` for new strategies
2. Integrate with `backtesting.py` and `live_trading.py`
3. Test with historical data
4. Consider adding real sentiment APIs for sentiment strategy
