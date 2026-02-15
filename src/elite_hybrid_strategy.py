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

def select_elite_hybrid_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10
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
    if current_date is None:
        latest_dates = [ticker_data_grouped[t].index.max() 
                       for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
        if latest_dates:
            current_date = max(latest_dates)
        else:
            return []
    
    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=timezone.utc)
    
    # Apply performance filters if enabled
    from performance_filters import filter_tickers_by_performance
    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Elite Hybrid"
    )
    
    candidates = []
    
    print(f"   🏆 Elite Hybrid: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")
    
    # Debug: Track specific tickers
    debug_tickers = ['SNDK', 'WDC', 'MU', 'AMAT', 'TER', 'GLW']
    
    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker]
            if len(ticker_data) == 0:
                continue
            
            # ✅ FIX: Filter data up to current_date to avoid temporal leakage
            if current_date is not None:
                current_ts = pd.Timestamp(current_date)
                if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                    if current_ts.tz is None:
                        current_ts = current_ts.tz_localize(ticker_data.index.tz)
                ticker_data_filtered = ticker_data.loc[:current_ts]
            else:
                ticker_data_filtered = ticker_data
            
            # Use dropna'd Close series for all calculations (adaptive approach)
            close_prices = ticker_data_filtered['Close'].dropna()
            n_prices = len(close_prices)
            
            if n_prices < 60:  # Minimum 60 days
                continue
            
            # Get latest price
            latest_price = close_prices.iloc[-1]
            if latest_price <= 0:
                continue
            
            # === PART 1: Mom-Vol Hybrid 6M Scoring ===
            
            # Calculate 6M performance (adaptive: use up to 126 trading days)
            lookback_6m = min(126, n_prices - 1)
            if lookback_6m < 40:
                continue
            price_6m_ago = close_prices.iloc[-lookback_6m]
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
            
            # === PART 2: 1Y/3M Ratio Scoring ===
            
            # Calculate 3M performance (adaptive: use up to 63 trading days)
            lookback_3m = min(63, n_prices - 1)
            if lookback_3m < 10:
                continue
            price_3m_ago = close_prices.iloc[-lookback_3m]
            if price_3m_ago <= 0:
                continue
            perf_3m = ((latest_price - price_3m_ago) / price_3m_ago) * 100
            
            # Calculate 1Y performance (adaptive: use up to 252 trading days)
            lookback_1y = min(252, n_prices - 1)
            if lookback_1y < 60:
                continue
            price_1y_ago = close_prices.iloc[-lookback_1y]
            if price_1y_ago <= 0:
                continue
            perf_1y = ((latest_price - price_1y_ago) / price_1y_ago) * 100
            
            # Dip score: for reference only (not used in scoring)
            dip_score = max(perf_1y - perf_3m, 0)
            
            # === PART 3: Combined Elite Score ===
            
            # Component 1: Mom-Vol score (momentum/volatility)
            annualized_6m = momentum_6m * 2  # 6M -> 1Y
            if volatility > 0:
                mom_vol_score = annualized_6m / volatility
            else:
                mom_vol_score = 0
            
            # Component 2: Dip ratio (1Y strength / 3M strength)
            # Higher ratio = bigger dip opportunity (strong 1Y, weak 3M)
            if perf_3m > 0:
                dip_ratio = perf_1y / perf_3m
            else:
                # If 3M is negative, use 1Y performance directly
                dip_ratio = max(perf_1y / 10, 0.1)  # Scale down and avoid zero
            
            # Normalize dip_ratio to reasonable range (0.5 to 5.0)
            dip_ratio = max(min(dip_ratio, 5.0), 0.5)
            
            # Elite Score: Additive bonus approach
            # Base score is mom-vol, with dip providing up to 150% bonus (0.3 * 5.0 = 1.5)
            # This keeps momentum dominant while rewarding dip opportunities
            elite_score = mom_vol_score * (1 + dip_ratio * 0.3)
            
            # Bonus: Reward low volatility (< 50%)
            if volatility < 0.5:
                elite_score *= 1.1
            
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
                print(f"   🔍 DEBUG {ticker}: Elite={elite_score:.3f}, 6M={momentum_6m*100:+.1f}%, "
                      f"Vol={volatility*100:.1f}%, 1Y={perf_1y:+.1f}%, 3M={perf_3m:+.1f}%, Dip={dip_score:.1f}")
            
        except Exception as e:
            if ticker in debug_tickers:
                print(f"   ❌ DEBUG {ticker}: FAILED - {e}")
            continue
    
    if not candidates:
        print(f"   ⚠️ Elite Hybrid: No candidates found")
        return []
    
    # Sort by elite score (descending)
    candidates.sort(key=lambda x: x['elite_score'], reverse=True)
    
    # Debug: show top candidates
    print(f"   ✅ Elite Hybrid: Found {len(candidates)} candidates")
    for i, c in enumerate(candidates[:5], 1):
        print(f"      {i}. {c['ticker']}: Elite={c['elite_score']:.3f}, "
              f"6M={c['momentum_6m']:+.1f}%, Vol={c['volatility']:.1f}%, "
              f"1Y={c['perf_1y']:+.1f}%, 3M={c['perf_3m']:+.1f}%, Dip={c['dip_score']:.1f}")
    
    # Return top N tickers
    selected = [c['ticker'] for c in candidates[:top_n]]
    return selected
