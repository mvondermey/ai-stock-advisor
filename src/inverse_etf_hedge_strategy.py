"""
Inverse ETF Hedge Strategy

Selects inverse ETFs based on recent performance.
Always selects top performing inverse ETFs for comparison purposes.
"""

from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
from performance_filters import filter_tickers_by_performance
from strategy_cache_adapter import ensure_price_history_cache

def select_inverse_etf_hedge_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    top_n: int = 4,
    verbose: bool = True,
    price_history_cache=None,
) -> List[str]:
    """
    Select inverse ETFs based on recent performance.

    Args:
        all_tickers: List of available ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame with price data
        current_date: Current date for analysis
        top_n: Number of ETFs to select (max 4)

    Returns:
        List of selected inverse ETF symbols
    """
    from config import INVERSE_ETF_HEDGE_PREFERENCE
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)

    if verbose:
        print(f"   🛡️ Inverse ETF Strategy: Selecting top {top_n} inverse ETFs by 3M performance")

    candidate_etfs = filter_tickers_by_performance(
        [etf for etf in INVERSE_ETF_HEDGE_PREFERENCE if etf in ticker_data_grouped],
        current_date,
        "Inverse ETF Hedge",
        price_history_cache=price_history_cache,
    )
    if not candidate_etfs:
        if verbose:
            print("   ⚠️ Inverse ETF Hedge: No inverse ETFs passed the unified prefilter")
        return []

    # Score inverse ETFs by recent performance
    inverse_etf_scores = []
    for etf in candidate_etfs:
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
                    # Calculate 3-month performance using calendar days (90 days)
                    start_3m = current_ts - timedelta(days=90)
                    data_3m = etf_data[etf_data.index >= start_3m]

                    if len(data_3m) >= 10:  # Need at least 10 trading days
                        start_price = data_3m['Close'].iloc[0]
                        end_price = etf_data['Close'].iloc[-1]

                        if start_price > 0:
                            perf_3m = (end_price / start_price - 1) * 100
                            inverse_etf_scores.append((etf, perf_3m))
                            if verbose:
                                print(f"      📈 {etf}: {perf_3m:+.1f}% (3M)")

            except Exception as e:
                if verbose:
                    print(f"      ⚠️ Error scoring {etf}: {e}")
                continue

    if not inverse_etf_scores:
        if verbose:
            print(f"   ⚠️ No inverse ETFs with valid data")
        return []

    # Sort by performance (best performers during decline)
    inverse_etf_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top ETFs
    num_etfs = min(top_n, len(inverse_etf_scores))
    selected_etfs = [etf for etf, _ in inverse_etf_scores[:num_etfs]]

    if verbose:
        print(f"   ✅ Selected {num_etfs} inverse ETFs: {selected_etfs}")
    return selected_etfs
