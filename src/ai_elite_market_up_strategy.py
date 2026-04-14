"""
AI Elite Market-Up Shared Strategy
Same as AI Elite but only rebalances when the trailing market regime is up,
reusing the daily AI Elite shared model.
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd


def select_ai_elite_market_up_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None,
    price_history_cache=None,
    hourly_history_cache=None,
    feature_cache_context=None,
    market_context_map=None,
) -> List[str]:
    """Select stocks using the daily AI Elite shared model, but only when market is up."""
    from ai_elite_strategy import select_ai_elite_stocks
    from market_regime import get_trailing_market_regime
    from strategy_cache_adapter import ensure_price_history_cache

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    market_return, is_market_up, proxy = get_trailing_market_regime(
        ticker_data_grouped,
        current_date,
        lookback_days=5,
    )

    if market_return is None:
        # On first day or when market data unavailable, assume market is up to allow initial investment
        print(f"   📊 AI Elite Market-Up Shared: Market data unavailable, allowing initial investment")
        market_return = 0.0
        is_market_up = True

    if not is_market_up:
        print(f"   📊 AI Elite Market-Up Shared: Market is down ({market_return:+.1f}% over trailing 5d via {proxy}), skipping rebalance")
        return []  # Don't rebalance when market is down

    print(f"   📊 AI Elite Market-Up Shared: Market is up ({market_return:+.1f}% over trailing 5d via {proxy}), proceeding with shared-model selection")

    # Delegate to AI Elite stock selection
    return select_ai_elite_stocks(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        per_ticker_models=per_ticker_models,
        price_history_cache=price_history_cache,
        hourly_history_cache=hourly_history_cache,
        feature_cache_context=feature_cache_context,
        market_context_map=market_context_map,
    )
