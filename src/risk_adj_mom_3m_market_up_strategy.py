"""
Risk-Adjusted Momentum 3M - Market-Up Only Strategy
Same as Risk-Adj Mom 3M but only rebalances when the trailing market regime is up.
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


def select_risk_adj_mom_3m_market_up_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """Select stocks using Risk-Adj Mom 3M scoring, but only when market is up."""
    from market_regime import get_trailing_market_regime
    from shared_strategies import select_risk_adj_mom_stocks
    from strategy_cache_adapter import ensure_price_history_cache

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    market_return, is_market_up, proxy = get_trailing_market_regime(
        ticker_data_grouped,
        current_date,
        lookback_days=5,
    )

    if market_return is None:
        # On first day or when market data unavailable, assume market is up to allow initial investment
        print(f"   📊 Risk-Adj Mom 3M Market-Up: Market data unavailable, allowing initial investment")
        market_return = 0.0
        is_market_up = True

    if not is_market_up:
        print(f"   📊 Risk-Adj Mom 3M Market-Up: Market is down ({market_return:+.1f}% over trailing 5d via {proxy}), skipping rebalance")
        return []  # Don't rebalance when market is down

    print(f"   📊 Risk-Adj Mom 3M Market-Up: Market is up ({market_return:+.1f}% over trailing 5d via {proxy}), proceeding with selection")

    # Delegate to shared parallel implementation
    return select_risk_adj_mom_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        lookback_days=90,
        strategy_name="Risk-Adj Mom 3M Market-Up",
        price_history_cache=price_history_cache,
    )
