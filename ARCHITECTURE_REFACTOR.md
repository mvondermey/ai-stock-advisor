# Architecture Refactor Plan

## Problem
Strategy selection logic is duplicated in 3 places:
1. `backtesting.py` - has ALL 95+ strategies (source of truth)
2. `main.py` `_execute_strategy_live()` - duplicates ~10 strategies
3. `live_trading.py` `get_strategy_tickers()` - duplicates ~30 strategies

When a new strategy is added to backtesting, it must be manually added to main.py and live_trading.py for live execution to work.

## Solution
Move ALL strategy selection functions to `shared_strategies.py`. Both backtesting and live trading import from there.

## Current State of shared_strategies.py
Already has:
- `select_top_performers()` - Static BH 1Y/6M/3M/1M
- `select_top_performers_with_scores()`
- `select_top_performers_vol_filtered()` - Dynamic BH 1Y+Vol
- `select_risk_adj_mom_stocks()`
- `select_momentum_ai_hybrid_stocks()`
- `select_ai_elite_with_training()`
- `select_bh_1y_volsweet_accel_stocks()`
- `select_bh_1y_dynamic_accel_stocks()`

## Missing from shared_strategies.py (need to add)
From backtesting.py strategy blocks:
- [ ] `select_mean_reversion_stocks()`
- [ ] `select_quality_momentum_stocks()`
- [ ] `select_volatility_adj_mom_stocks()`
- [ ] `select_sector_rotation_stocks()`
- [ ] `select_3m_1y_ratio_stocks()`
- [ ] `select_1y_3m_ratio_stocks()`
- [ ] `select_momentum_volatility_hybrid_stocks()` (all variants: 6m, 1y, 1y3m)
- [ ] `select_price_acceleration_stocks()`
- [ ] `select_turnaround_stocks()`
- [ ] `select_adaptive_ensemble_stocks()`
- [ ] `select_volatility_ensemble_stocks()`
- [ ] `select_enhanced_volatility_stocks()`
- [ ] `select_correlation_ensemble_stocks()`
- [ ] `select_dynamic_pool_stocks()`
- [ ] `select_voting_ensemble_stocks()`
- [ ] `select_dual_momentum_stocks()`
- [ ] `select_trend_atr_stocks()`
- [ ] `select_concentrated_3m_stocks()`
- [ ] `select_mom_accel_stocks()`
- [ ] `select_elite_hybrid_stocks()`
- [ ] `select_elite_risk_stocks()`
- [ ] `select_risk_adj_mom_6m_stocks()`
- [ ] `select_risk_adj_mom_3m_stocks()`
- [ ] `select_risk_adj_mom_1m_stocks()`
- [ ] `select_bb_mean_reversion_stocks()`
- [ ] `select_bb_breakout_stocks()`
- [ ] `select_bb_squeeze_stocks()`
- [ ] `select_bb_rsi_combo_stocks()`
- [ ] `select_trend_breakout_stocks()`
- [ ] All Static BH 1Y variants (vol, perf, mom, atr, hybrid, volume, sector, etc.)
- [ ] All BH 1Y smart rebalancing variants (mom_sell, rank_sell, trailing_mom, etc.)
- [ ] All Rebal 1Y variants (vol_adj, corr_filter, regime, risk_parity, etc.)
- [ ] AI Elite variants (monthly, filtered, market_up)
- [ ] AI Regime variants
- [ ] Universal Model
- [ ] Analyst Rec
- [ ] Inverse ETF Hedge

## Refactor Steps
1. For each strategy in backtesting.py:
   - Extract selection logic into a function in shared_strategies.py
   - Update backtesting.py to call the shared function
   - Remove duplicate code from main.py and live_trading.py

2. Update main.py `_execute_strategy_live()`:
   - Replace if/elif chain with single call to shared function
   - Or: use a strategy registry dict mapping name -> function

3. Update live_trading.py `get_strategy_tickers()`:
   - Same approach - call shared functions

## Strategy Registry Pattern (recommended)
```python
# In shared_strategies.py
STRATEGY_SELECTORS = {
    'static_bh_1y': lambda tickers, data, date, n: select_top_performers(tickers, data, date, 365, n),
    'static_bh_6m': lambda tickers, data, date, n: select_top_performers(tickers, data, date, 180, n),
    'risk_adj_mom': select_risk_adj_mom_stocks,
    'bh_1y_accel_buy': select_bh_1y_accel_buy_stocks,
    # ... all strategies
}

def get_strategy_tickers(strategy_name, tickers, data, date, top_n):
    if strategy_name in STRATEGY_SELECTORS:
        return STRATEGY_SELECTORS[strategy_name](tickers, data, date, top_n)
    raise ValueError(f"Unknown strategy: {strategy_name}")
```

Then main.py and live_trading.py just call:
```python
from shared_strategies import get_strategy_tickers
tickers = get_strategy_tickers(strategy, all_tickers, ticker_data, current_date, portfolio_size)
```

## Priority
High - this is blocking live execution of many strategies.
