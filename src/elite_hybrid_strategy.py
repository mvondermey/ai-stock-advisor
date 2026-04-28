"""
Elite Hybrid Strategy: Combines the two most consistent strategies
- Mom-Vol Hybrid 6M (95.4% consistency)
- 1Y/3M Ratio (93.1% consistency)

Strategy Logic:
1. Start with Mom-Vol Hybrid 6M candidates (strong 6M momentum + low volatility)
2. Apply 1Y/3M Ratio filter (buy-the-dip: strong 1Y, weak 3M)
3. Rank by composite score combining both signals
4. Select top N stocks

This creates a strategy that:
- Captures momentum with controlled volatility (Mom-Vol 6M)
- Identifies pullback opportunities in strong trends (1Y/3M Ratio)
- Combines consistency of both top performers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta, timezone
from strategy_cache_adapter import (
    ensure_price_history_cache,
    get_cached_history_up_to,
    get_cached_window,
    resolve_cache_current_date,
)

def select_elite_hybrid_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    price_history_cache=None,
) -> List[str]:
    """
    Elite Hybrid Strategy: Combines Mom-Vol Hybrid 6M + 1Y/3M Ratio
    
    Args:
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current date for analysis
        top_n: Number of stocks to select
        
    Returns:
        List of selected ticker symbols
    """
    price_history_cache = ensure_price_history_cache(ticker_data_grouped, price_history_cache)
    current_date = resolve_cache_current_date(price_history_cache, current_date, all_tickers)
    if current_date is None:
        return []
    
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    
    # Filter out inverse ETFs - they should only be in inverse_etf_hedge strategy
    from config import INVERSE_ETFS
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]
    
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers,
        current_date,
        "Elite Hybrid",
        price_history_cache=price_history_cache,
    )
    
    candidates = []
    
    print(f"   [INFO] Elite Hybrid: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")
    
    # Debug: Track specific tickers
    debug_tickers = ['SNDK', 'WDC', 'MU', 'AMAT', 'TER', 'GLW']
    
    for ticker in filtered_tickers:
        try:
            close_values = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="close",
                min_rows=60,
            )
            if close_values is None:
                continue
            close_prices = pd.Series(close_values)
            n_prices = len(close_prices)
            
            if n_prices < 60:  # Minimum 60 days
                continue
            
            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue
            
            # === PART 1: Mom-Vol Hybrid 6M Scoring (using calendar days) ===
            
            # Calculate 6M performance using calendar days (180 days)
            data_6m = get_cached_window(
                price_history_cache, ticker, current_date, 180, field_name="close", min_rows=40
            )
            if data_6m is None or len(data_6m) < 40:
                continue
            price_6m_ago = data_6m[0]
            if price_6m_ago <= 0:
                continue
            
            momentum_6m = (latest_price - price_6m_ago) / price_6m_ago
            
            # Calculate volatility (using all available data)
            daily_returns = close_prices.pct_change().dropna()
            if len(daily_returns) < 30:
                continue
            
            volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
            
            # Mom-Vol score: momentum adjusted by volatility
            if volatility > 0:
                mom_vol_score = momentum_6m / volatility
            else:
                mom_vol_score = 0
            
            # === PART 2: 1Y/3M Ratio Scoring (using calendar days) ===
            
            # Calculate 3M performance using calendar days (90 days)
            data_3m = get_cached_window(
                price_history_cache, ticker, current_date, 90, field_name="close", min_rows=10
            )
            if data_3m is None or len(data_3m) < 10:
                continue
            price_3m_ago = data_3m[0]
            if price_3m_ago <= 0:
                continue
            perf_3m = ((latest_price - price_3m_ago) / price_3m_ago) * 100
            
            # Calculate 1Y performance using calendar days (365 days)
            data_1y = get_cached_window(
                price_history_cache, ticker, current_date, 365, field_name="close", min_rows=60
            )
            if data_1y is None or len(data_1y) < 60:
                continue
            price_1y_ago = data_1y[0]
            if price_1y_ago <= 0:
                continue
            perf_1y = ((latest_price - price_1y_ago) / price_1y_ago) * 100
            
            # Calculate average volume (for volume confirmation)
            volume_values = get_cached_history_up_to(
                price_history_cache,
                ticker,
                current_date,
                field_name="volume",
                min_rows=1,
            )
            avg_volume = (
                float(np.mean(volume_values[-30:]))
                if volume_values is not None and len(volume_values) > 0
                else 1000000
            )
            
            # Dip score: for reference only (not used in scoring)
            dip_score = max(perf_1y - perf_3m, 0)
            
            # === PART 3: Combined Elite Score (IMPROVED with Risk-Adj Mom techniques) ===
            
            # Component 1: Risk-adjusted momentum (use sqrt volatility like Risk-Adj Mom)
            annualized_6m = momentum_6m * 2  # 6M -> 1Y
            if volatility > 0:
                # KEY IMPROVEMENT: Use sqrt(volatility) instead of full volatility
                # This is what makes Risk-Adj Mom so successful
                risk_adj_momentum = annualized_6m / ((volatility ** 0.5) + 0.001)
            else:
                risk_adj_momentum = 0
            
            # Component 2: Volume quality check (like Risk-Adj Mom)
            # Higher volume = more liquid and reliable
            volume_score = min(avg_volume / 1000000, 2.0)  # Cap at 2x bonus
            
            # Component 3: Dip ratio (1Y strength / 3M strength)
            # Tighter range than before (0.1 to 3.0 instead of 0.5 to 5.0)
            if perf_3m > 0 and perf_1y > 0:
                dip_ratio = (perf_1y - perf_3m) / perf_3m
                dip_ratio = max(min(dip_ratio, 3.0), 0.1)
            else:
                dip_ratio = 0.1
            
            # Elite Score: BEAT Risk-Adj Mom by exploiting dip opportunities
            # Risk-Adj Mom weakness: Ignores stocks with strong 1Y but weak 3M (dip opportunities)
            # Our advantage: Identify and capitalize on temporary dips in strong stocks
            
            base_score = risk_adj_momentum
            
            # INNOVATION 1: Smart Dip Detection (what Risk-Adj Mom misses)
            # Identify stocks with strong 1Y performance but temporary 3M weakness
            # These are high-quality stocks on sale!
            if perf_1y > 20 and perf_3m < 10 and perf_3m > 0:
                # Strong 1Y (>20%), weak 3M (<10%) = BUY THE DIP opportunity
                dip_opportunity_bonus = 1.3  # 30% bonus for dip opportunities
            elif perf_1y > 30 and perf_3m < 5:
                # Very strong 1Y (>30%), very weak 3M (<5%) = STRONG BUY
                dip_opportunity_bonus = 1.5  # 50% bonus for strong dips
            elif perf_1y > 10 and perf_3m < 0:
                # Good 1Y (>10%), negative 3M = Potential reversal
                dip_opportunity_bonus = 1.2  # 20% bonus
            else:
                # Normal momentum - use standard dip ratio
                dip_opportunity_bonus = 1 + dip_ratio * 0.1
            
            # INNOVATION 2: Volume Quality (better than Risk-Adj Mom's simple filter)
            # High volume + tight spread = institutional interest
            if avg_volume > 5000000:  # Very high volume
                volume_bonus = 1.15  # 15% bonus
            elif avg_volume > 2000000:  # High volume
                volume_bonus = 1.10  # 10% bonus
            elif avg_volume > 1000000:  # Good volume
                volume_bonus = 1.05  # 5% bonus
            else:
                volume_bonus = 1.0  # No bonus for low volume
            
            # INNOVATION 3: Volatility Sweet Spot (Risk-Adj Mom treats all vol equally)
            # Too low vol = no opportunity, too high vol = too risky
            # Sweet spot: 20-40% volatility
            if 0.20 <= volatility <= 0.40:
                volatility_bonus = 1.10  # 10% bonus for optimal volatility
            elif volatility < 0.15:
                volatility_bonus = 0.95  # Penalize too-low volatility (no opportunity)
            elif volatility > 0.60:
                volatility_bonus = 0.90  # Penalize too-high volatility (too risky)
            else:
                volatility_bonus = 1.0
            
            # Combined Elite Score: Base + 3 innovations
            elite_score = base_score * dip_opportunity_bonus * volume_bonus * volatility_bonus
            
            candidates.append({
                'ticker': ticker,
                'elite_score': elite_score,
                'momentum_6m': momentum_6m * 100,
                'volatility': volatility * 100,
                'perf_1y': perf_1y,
                'perf_3m': perf_3m,
                'dip_score': dip_score
            })
            
            # Debug: Show specific tickers
            if ticker in debug_tickers:
                print(f"   [DEBUG] {ticker}: Elite={elite_score:.3f}, 6M={momentum_6m*100:+.1f}%, "
                      f"Vol={volatility*100:.1f}%, 1Y={perf_1y:+.1f}%, 3M={perf_3m:+.1f}%, Dip={dip_score:.1f}")
            
        except Exception as e:
            if ticker in debug_tickers:
                print(f"   [FAIL] DEBUG {ticker}: FAILED - {e}")
            continue
    
    if not candidates:
        print(f"   [WARN] Elite Hybrid: No candidates found")
        return []
    
    # Sort by elite score (descending)
    candidates.sort(key=lambda x: x['elite_score'], reverse=True)
    
    # Debug: show top candidates
    print(f"   [INFO] Elite Hybrid: Found {len(candidates)} candidates")
    for i, c in enumerate(candidates[:5], 1):
        print(f"      {i}. {c['ticker']}: Elite={c['elite_score']:.3f}, "
              f"6M={c['momentum_6m']:+.1f}%, Vol={c['volatility']:.1f}%, "
              f"1Y={c['perf_1y']:+.1f}%, 3M={c['perf_3m']:+.1f}%, Dip={c['dip_score']:.1f}")
    
    # Return top N tickers
    selected = [c['ticker'] for c in candidates[:top_n]]
    return selected
