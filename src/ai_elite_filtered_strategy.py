"""
AI Elite Filtered Strategy: Risk-Adj Mom 3M + AI Elite confirmation filter

Approach:
1. Get top 20 stocks by Risk-Adj Mom 3M score (proven simple signal)
2. Use AI Elite model to filter/re-rank down to top 10
3. Return [] if no AI model is available

This combines the best of both:
- Risk-Adj Mom 3M: Proven, no overfitting, captures recent momentum
- AI Elite: ML refinement to filter out false positives
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


def select_ai_elite_filtered_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    per_ticker_models: Dict[str, any] = None,
    pre_filter_n: int = 20,
    price_history_cache=None,
    hourly_history_cache=None,
    feature_cache_context=None,
    market_context_map=None,
) -> List[str]:
    """
    AI Elite Filtered Strategy: Risk-Adj Mom 3M base + AI Elite filter
    
    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select (final output)
        per_ticker_models: Dict of ticker -> trained AI Elite model
        pre_filter_n: Number of stocks to pre-select with Risk-Adj Mom 3M
        
    Returns:
        List of selected ticker symbols
    """
    from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
    from ai_elite_strategy import select_ai_elite_stocks
    from strategy_cache_adapter import ensure_price_history_cache

    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    
    # Step 1: Get top N candidates using Risk-Adj Mom 3M (proven signal)
    pre_filter_count = max(pre_filter_n, top_n * 2)  # At least 2x final count
    
    risk_adj_candidates = select_risk_adj_mom_3m_stocks(
        all_tickers,
        ticker_data_grouped,
        current_date=current_date,
        top_n=pre_filter_count,
        price_history_cache=price_history_cache,
    )
    
    if not risk_adj_candidates:
        print(f"   ⚠️ AI Elite Filtered: No Risk-Adj Mom 3M candidates, returning empty")
        return []
    
    print(f"   🔍 AI Elite Filtered: Pre-filtered to {len(risk_adj_candidates)} stocks via Risk-Adj Mom 3M")
    
    # Step 2: Check if we have AI Elite models
    if per_ticker_models is None or len(per_ticker_models) == 0:
        print(f"   ⚠️ AI Elite Filtered: No AI models available, returning empty")
        return []
    
    # Check how many candidates have models
    candidates_with_models = [t for t in risk_adj_candidates if per_ticker_models.get(t) is not None]
    
    if len(candidates_with_models) < top_n:
        print(f"   ⚠️ AI Elite Filtered: Only {len(candidates_with_models)} candidates have models, returning empty")
        return []
    
    # Step 3: Use AI Elite to re-rank the pre-filtered candidates
    print(f"   🤖 AI Elite Filtered: Applying AI filter to {len(risk_adj_candidates)} candidates...")
    
    ai_elite_picks = select_ai_elite_stocks(
        risk_adj_candidates,  # Only score the pre-filtered candidates
        ticker_data_grouped,
        current_date=current_date,
        top_n=top_n,
        per_ticker_models=per_ticker_models,
        price_history_cache=price_history_cache,
        hourly_history_cache=hourly_history_cache,
        feature_cache_context=feature_cache_context,
        market_context_map=market_context_map,
    )
    
    if not ai_elite_picks:
        print(f"   ⚠️ AI Elite Filtered: AI Elite returned empty, returning empty")
        return []
    
    # Step 4: Blend - use AI Elite ranking but ensure we don't stray too far from Risk-Adj Mom 3M
    # If AI Elite picks are all from the pre-filtered set, we're good
    final_picks = ai_elite_picks[:top_n]
    
    print(f"   ✅ AI Elite Filtered: Selected {len(final_picks)} stocks")
    print(f"   📊 AI Elite Filtered: Combining Risk-Adj Mom 3M (pre-filter) + AI Elite (re-rank)")
    
    # Show overlap with pure Risk-Adj Mom 3M top picks
    pure_risk_adj_top = set(risk_adj_candidates[:top_n])
    ai_filtered_set = set(final_picks)
    overlap = pure_risk_adj_top & ai_filtered_set
    print(f"   📊 AI Elite Filtered: {len(overlap)}/{top_n} overlap with pure Risk-Adj Mom 3M")
    
    return final_picks
