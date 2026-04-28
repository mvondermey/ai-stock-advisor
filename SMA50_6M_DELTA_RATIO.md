# SMA50 / 6M Delta Ratio

## Purpose
This strategy ranks tickers by how far price is above the `SMA50` relative to the change in their trailing `6M` return over the last `50` trading days.

Formula:

```text
(close - SMA50) / (6M_return_today - 6M_return_50d_ago)
```

## Selection Rules
- Require `close > SMA50`
- Allow negative `6M` delta values
- Skip tickers where the `6M` delta cannot be computed
- Skip tickers where the denominator is effectively zero

## Return Windows
- `SMA50`: computed from daily resampled closes
- `6M_return_today`: based on roughly `126` trading days
- `6M_return_50d_ago`: computed from the close `50` trading days ago versus roughly `126` trading days before that

In practice, the implementation needs at least about `177` daily closes for a ticker to qualify.

## Implementation
The strategy was added to:

- `src/shared_strategies.py`
- `src/backtesting.py`
- `src/config.py`
- `src/strategy_registry.py`

## Strategy Names
Current internal canonical key:

- `sma50_1y_delta_ratio`

Supported CLI aliases:

- `sma50_6m_delta_ratio`
- `sma50_6m_ratio`
- `sma50_ratio`

The internal key still contains `1y` for compatibility, but the actual logic and displayed label now use `6M`.

## Backtest Usage
Example smoke test:

```bash
python src/main.py --strategy sma50_6m_delta_ratio --backtest-days 5 --no-download --num-stocks 5
```

## Observed Behavior
- The strategy runs end-to-end in backtesting
- It selects a buffer list, then holds up to `PORTFOLIO_SIZE`
- It participates in the normal smart rebalance flow

## Caveats
- Newer tickers can still be skipped if they do not have enough daily history
- Crypto and ETFs may appear unless the trading universe is filtered elsewhere
- The strategy can turn over quickly because the ratio can move sharply day to day
