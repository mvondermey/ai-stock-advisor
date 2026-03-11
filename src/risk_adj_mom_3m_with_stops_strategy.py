"""
Risk-Adjusted Momentum 3M with Stops Strategy
Same as Risk-Adj Mom 3M but includes stop loss and take profit logic.
- Stop loss: 5% loss from entry price
- Take profit: 15% gain from entry price
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime


def select_risk_adj_mom_3m_with_stops_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
) -> List[str]:
    """Select stocks using Risk-Adj Mom 3M scoring (same as original)."""
    from performance_filters import filter_tickers_by_performance
    from config import (
        RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION,
        RISK_ADJ_MOM_MIN_CONFIRMATIONS,
        RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION,
        RISK_ADJ_MOM_VOLUME_WINDOW,
        RISK_ADJ_MOM_VOLUME_MULTIPLIER,
        RISK_ADJ_MOM_MIN_SCORE,
        INVERSE_ETFS,
    )

    # Filter out inverse ETFs
    all_tickers = [t for t in all_tickers if t not in INVERSE_ETFS]

    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "Risk-Adj Mom 3M with Stops"
    )

    PERF_WINDOW = 90  # 3 months

    candidates = []
    print(f"   📊 Risk-Adj Mom 3M with Stops: Analyzing {len(filtered_tickers)} tickers")

    for ticker in filtered_tickers:
        try:
            if ticker not in ticker_data_grouped:
                continue

            data = ticker_data_grouped[ticker]
            if data is None or len(data) == 0:
                continue

            close = data['Close'].dropna()
            n = len(close)
            if n < 30:
                continue

            latest_price = close.iloc[-1]
            if latest_price <= 0:
                continue

            perf_window = min(PERF_WINDOW, n - 1)
            if perf_window < 30:
                continue

            start_price = close.iloc[-perf_window]
            if start_price <= 0:
                continue

            basic_return = (latest_price - start_price) / start_price * 100

            daily_returns = close.pct_change().dropna()
            if len(daily_returns) < 20:
                continue

            volatility_pct = daily_returns.std() * 100
            if volatility_pct <= 0:
                continue

            score = basic_return / (volatility_pct ** 0.5 + 0.001)

            if score <= RISK_ADJ_MOM_MIN_SCORE:
                continue

            # Momentum confirmation
            if RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION:
                confirmations = 0
                for days in [30, 60, 90]:
                    lookback = min(days, n - 1)
                    p = close.iloc[-lookback]
                    if p > 0 and (latest_price - p) / p > 0:
                        confirmations += 1
                if confirmations < RISK_ADJ_MOM_MIN_CONFIRMATIONS:
                    continue

            # Volume confirmation
            if RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION and 'Volume' in data.columns:
                vol_series = data['Volume'].dropna()
                if len(vol_series) >= RISK_ADJ_MOM_VOLUME_WINDOW + 10:
                    recent_vol = vol_series.tail(RISK_ADJ_MOM_VOLUME_WINDOW).mean()
                    avg_vol = vol_series.iloc[:-RISK_ADJ_MOM_VOLUME_WINDOW].mean()
                    if avg_vol > 0 and recent_vol < avg_vol * RISK_ADJ_MOM_VOLUME_MULTIPLIER:
                        continue

            candidates.append((ticker, score, basic_return, volatility_pct))

        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    if not candidates:
        print(f"   ⚠️ Risk-Adj Mom 3M with Stops: No candidates found")
        return []

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, s, r, v in candidates[:top_n]]

    print(f"   ✅ Risk-Adj Mom 3M with Stops: Found {len(candidates)} candidates, selected {len(selected)}")
    for t, s, r, v in candidates[:top_n]:
        print(f"      {t}: score={s:.2f}, return={r:.1f}%, vol={v:.1f}%")

    return selected


def check_risk_adj_mom_3m_stops(
    ticker: str,
    data: pd.DataFrame,
    entry_price: float,
    current_price: float,
    position_days: int,
) -> tuple[bool, str]:
    """
    Check if position should be closed based on stop loss or take profit.
    
    Returns:
        (should_close, reason)
    """
    STOP_LOSS_PCT = 5.0  # 5% stop loss
    TAKE_PROFIT_PCT = 15.0  # 15% take profit
    
    if current_price <= 0 or entry_price <= 0:
        return False, "Invalid prices"
    
    pnl_pct = (current_price - entry_price) / entry_price * 100
    
    # Stop loss: close if down 5% or more
    if pnl_pct <= -STOP_LOSS_PCT:
        return True, f"Stop loss triggered: {pnl_pct:.1f}% loss"
    
    # Take profit: close if up 15% or more
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, f"Take profit triggered: {pnl_pct:.1f}% gain"
    
    return False, "Hold"


def update_risk_adj_mom_3m_with_stops_positions(
    positions: Dict,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    transaction_cost: float,
) -> tuple[Dict, float, List[str]]:
    """
    Check and apply custom stops for Risk-Adj Mom 3M with Stops strategy.
    
    Returns:
        (updated_positions, transaction_costs, sold_tickers)
    """
    from backtesting import _last_valid_close_up_to
    
    total_costs = 0.0
    sold_tickers = []
    
    for ticker, pos_info in list(positions.items()):
        if pos_info.get('shares', 0) <= 0:
            continue
            
        # Get current price
        ticker_df = ticker_data_grouped.get(ticker)
        if ticker_df is None:
            continue
            
        current_price = _last_valid_close_up_to(ticker_df, current_date)
        if current_price is None:
            continue
        
        # Get entry price (stored as entry_price or avg_price)
        entry_price = pos_info.get('entry_price', pos_info.get('avg_price', 0))
        if entry_price <= 0:
            # Store entry price if not set
            pos_info['entry_price'] = current_price
            continue
        
        # Check if stop should be triggered
        should_close, reason = check_risk_adj_mom_3m_stops(
            ticker, ticker_df, entry_price, current_price, 
            pos_info.get('days_held', 0)
        )
        
        if should_close:
            # Sell the position
            shares = pos_info['shares']
            sale_value = shares * current_price
            cost = sale_value * transaction_cost
            net_value = sale_value - cost
            
            # Update position
            pos_info['shares'] = 0
            pos_info['value'] = 0
            pos_info['exit_price'] = current_price
            pos_info['exit_reason'] = reason
            pos_info['exit_date'] = current_date
            
            total_costs += cost
            sold_tickers.append((ticker, reason, net_value))
            
            print(f"   💰 RiskAdj 3M Stop: Selling {ticker}: {reason}")
    
    return positions, total_costs, sold_tickers
