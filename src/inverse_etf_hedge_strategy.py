"""
Inverse ETF Hedge Strategy

Selects inverse ETFs when market is down to hedge against declines.
Only holds inverse ETFs during market downturns, stays in cash otherwise.
"""

from typing import List, Dict
from datetime import datetime
import pandas as pd

def select_inverse_etf_hedge_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 4
) -> List[str]:
    """
    Select inverse ETFs based on current market conditions.
    
    Args:
        all_tickers: List of available ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame with price data
        current_date: Current date for analysis
        top_n: Number of ETFs to select (max 4)
        
    Returns:
        List of selected inverse ETF symbols (empty list if market is up)
    """
    from config import (
        INVERSE_ETF_HEDGE_THRESHOLD_LOW, INVERSE_ETF_HEDGE_THRESHOLD_MED,
        INVERSE_ETF_HEDGE_THRESHOLD_HIGH, INVERSE_ETF_HEDGE_PREFERENCE
    )
    from shared_strategies import get_market_conditions
    
    # Get market conditions
    market_conditions = get_market_conditions(ticker_data_grouped, current_date)
    
    # Check if market is down significantly
    market_decline = 0
    for metric, value in market_conditions.items():
        if '_3m' in metric and value < 0:  # Negative 3-month performance
            market_decline = max(market_decline, abs(value))
    
    # Determine if we should hedge
    if market_decline <= INVERSE_ETF_HEDGE_THRESHOLD_LOW:
        print(f"   🛡️ Market up {market_decline:.1%} (threshold {INVERSE_ETF_HEDGE_THRESHOLD_LOW:.1%}) - No hedge needed")
        return []
    
    # Calculate hedge level for logging
    if market_decline >= INVERSE_ETF_HEDGE_THRESHOLD_HIGH:
        hedge_level = "HIGH"
    elif market_decline >= INVERSE_ETF_HEDGE_THRESHOLD_MED:
        hedge_level = "MED"
    else:
        hedge_level = "LOW"
    
    print(f"   🛡️ Market down {market_decline:.1%} ({hedge_level} hedge) - Selecting inverse ETFs")
    
    # Score inverse ETFs by recent performance (better performers during decline)
    inverse_etf_scores = []
    for etf in INVERSE_ETF_HEDGE_PREFERENCE:
        if etf in ticker_data_grouped:
            try:
                etf_data = ticker_data_grouped[etf]
                
                # Convert current_date to pandas Timestamp with timezone
                current_ts = pd.Timestamp(current_date)
                if hasattr(etf_data.index, 'tz') and etf_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(etf_data.index.tz)
                    else:
                        current_ts = current_ts.tz_convert(etf_data.index.tz)
                
                # Filter data up to current_date
                etf_data = etf_data[etf_data.index <= current_ts]
                
                if len(etf_data) >= 20:  # Need at least 20 days of data
                    # Calculate 3-month performance
                    start_price = etf_data['Close'].iloc[-63] if len(etf_data) >= 63 else etf_data['Close'].iloc[0]
                    end_price = etf_data['Close'].iloc[-1]
                    
                    if start_price > 0:
                        perf_3m = (end_price / start_price - 1) * 100
                        inverse_etf_scores.append((etf, perf_3m))
                        print(f"      📈 {etf}: {perf_3m:+.1f}% (3M)")
                        
            except Exception as e:
                print(f"      ⚠️ Error scoring {etf}: {e}")
                continue
    
    if not inverse_etf_scores:
        print(f"   ⚠️ No inverse ETFs with valid data")
        return []
    
    # Sort by performance (best performers during decline)
    inverse_etf_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top ETFs
    num_etfs = min(top_n, len(inverse_etf_scores))
    selected_etfs = [etf for etf, _ in inverse_etf_scores[:num_etfs]]
    
    print(f"   ✅ Selected {num_etfs} inverse ETFs: {selected_etfs}")
    return selected_etfs
