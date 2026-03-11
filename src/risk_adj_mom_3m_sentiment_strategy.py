"""
Risk-Adjusted Momentum 3M + Sentiment Strategy

Combines the best performer (Risk-Adj Mom 3M: +60.2%) with price-derived sentiment signals.
score = (return_3m / sqrt(volatility)) * (1 + sentiment_boost)

Sentiment signals (price-derived, no API needed):
- Short-term reversal: 5-day return minus 20-day return (acceleration)
- Volume sentiment: Volume surge signed by price direction
- RSI momentum: Overbought/oversold conditions
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime


def calculate_sentiment_score(data: pd.DataFrame, current_date: datetime = None) -> float:
    """
    Calculate sentiment score from price data (no API calls needed).
    Returns a score between -1.0 (very bearish) and +1.0 (very bullish).
    """
    try:
        if data is None or len(data) < 30:
            return 0.0
        
        # Filter to current date if provided
        if current_date is not None:
            current_ts = pd.Timestamp(current_date)
            if current_ts.tz is None and data.index.tz is not None:
                current_ts = current_ts.tz_localize(data.index.tz)
            elif current_ts.tz is not None and data.index.tz is None:
                data = data.copy()
                data.index = data.index.tz_localize('UTC')
            data = data[data.index <= current_ts]
        
        if len(data) < 30:
            return 0.0
        
        close = data['Close'].dropna()
        if len(close) < 30:
            return 0.0
        
        # 1. Short-term reversal (acceleration signal)
        # Positive = recent acceleration (bullish), negative = fading (bearish)
        perf_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        perf_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        short_term_reversal = perf_5d - perf_20d
        # Normalize to [-1, 1] range (assume ±10% is extreme)
        reversal_score = np.clip(short_term_reversal / 10.0, -1.0, 1.0)
        
        # 2. Volume sentiment (conviction signal)
        volume_score = 0.0
        if 'Volume' in data.columns:
            vol_series = data['Volume'].dropna()
            if len(vol_series) >= 20:
                recent_5d_vol = vol_series.tail(5).mean()
                avg_20d_vol = vol_series.tail(20).mean()
                vol_surge = (recent_5d_vol / avg_20d_vol) - 1.0 if avg_20d_vol > 0 else 0.0
                # Sign by 5-day price direction
                price_direction = 1.0 if perf_5d > 0 else (-1.0 if perf_5d < 0 else 0.0)
                volume_score = np.clip(vol_surge * price_direction, -1.0, 1.0)
        
        # 3. RSI momentum (overbought/oversold)
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) >= 14:
            gains = daily_returns.where(daily_returns > 0, 0)
            losses = (-daily_returns).where(daily_returns < 0, 0)
            avg_gain = gains.tail(14).mean()
            avg_loss = losses.tail(14).mean()
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50
            # RSI score: >70 = overbought (slightly bearish), <30 = oversold (bullish)
            # Normalize: RSI 50 = 0, RSI 30 = +0.5, RSI 70 = -0.5
            rsi_score = (50 - rsi) / 40.0  # Range roughly [-1, 1]
            rsi_score = np.clip(rsi_score, -1.0, 1.0)
        else:
            rsi_score = 0.0
        
        # 4. Trend strength (SMA alignment)
        trend_score = 0.0
        if len(close) >= 50:
            sma_20 = close.tail(20).mean()
            sma_50 = close.tail(50).mean()
            current_price = close.iloc[-1]
            # Bullish: price > SMA20 > SMA50
            if current_price > sma_20 > sma_50:
                trend_score = 0.5
            elif current_price > sma_20:
                trend_score = 0.25
            # Bearish: price < SMA20 < SMA50
            elif current_price < sma_20 < sma_50:
                trend_score = -0.5
            elif current_price < sma_20:
                trend_score = -0.25
        
        # Combine scores with weights
        # Reversal (acceleration) is most important for momentum
        combined = (
            reversal_score * 0.40 +   # 40% weight - acceleration
            volume_score * 0.25 +     # 25% weight - conviction
            rsi_score * 0.15 +        # 15% weight - mean reversion
            trend_score * 0.20        # 20% weight - trend alignment
        )
        
        return np.clip(combined, -1.0, 1.0)
        
    except Exception as e:
        print(f"   ⚠️ Sentiment analysis error: {e}")
        return 0.0


def select_risk_adj_mom_3m_sentiment_stocks(
    all_tickers: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime = None,
    top_n: int = 10,
    sentiment_weight: float = 0.30,  # 30% sentiment, 70% momentum
) -> List[str]:
    """
    Select stocks using Risk-Adj Mom 3M + Sentiment scoring.
    
    Final score = base_score * (1 + sentiment_weight * sentiment_score)
    
    This boosts stocks with positive sentiment and penalizes those with negative sentiment.
    """
    from performance_filters import filter_tickers_by_performance
    from config import (
        RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION,
        RISK_ADJ_MOM_MIN_CONFIRMATIONS,
        RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION,
        RISK_ADJ_MOM_VOLUME_WINDOW,
        RISK_ADJ_MOM_VOLUME_MULTIPLIER,
        RISK_ADJ_MOM_MIN_SCORE,
    )

    filtered_tickers = filter_tickers_by_performance(
        all_tickers, ticker_data_grouped, current_date, "RiskAdj 3M Sent"
    )

    PERF_WINDOW = 90  # 3 months

    candidates = []
    sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    print(f"   📊 RiskAdj 3M Sent: Analyzing {len(filtered_tickers)} tickers (filtered from {len(all_tickers)})")

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

            # Base score: Risk-Adj Mom 3M
            base_score = basic_return / (volatility_pct ** 0.5 + 0.001)

            if base_score <= RISK_ADJ_MOM_MIN_SCORE:
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

            # Calculate sentiment score
            sentiment = calculate_sentiment_score(data, current_date)
            
            # Track sentiment distribution
            if sentiment > 0.1:
                sentiment_stats['positive'] += 1
            elif sentiment < -0.1:
                sentiment_stats['negative'] += 1
            else:
                sentiment_stats['neutral'] += 1
            
            # Final score: boost/penalize by sentiment
            # sentiment_weight=0.30 means ±30% adjustment for extreme sentiment
            final_score = base_score * (1 + sentiment_weight * sentiment)
            
            candidates.append({
                'ticker': ticker,
                'final_score': final_score,
                'base_score': base_score,
                'sentiment': sentiment,
                'return': basic_return,
                'volatility': volatility_pct
            })

        except Exception as e:
            print(f"   ⚠️ Error processing {ticker}: {e}")
            continue

    if not candidates:
        print(f"   ⚠️ RiskAdj 3M Sent: No candidates found")
        return []

    # Sort by final score (sentiment-adjusted)
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    selected = [c['ticker'] for c in candidates[:top_n]]

    print(f"   ✅ RiskAdj 3M Sent: Found {len(candidates)} candidates, selected {len(selected)}")
    print(f"   📈 Sentiment: {sentiment_stats['positive']} bullish, {sentiment_stats['negative']} bearish, {sentiment_stats['neutral']} neutral")
    
    for c in candidates[:top_n]:
        sent_emoji = "🟢" if c['sentiment'] > 0.1 else ("🔴" if c['sentiment'] < -0.1 else "⚪")
        print(f"      {c['ticker']}: score={c['final_score']:.2f} (base={c['base_score']:.2f}, sent={c['sentiment']:+.2f}{sent_emoji}), ret={c['return']:.1f}%, vol={c['volatility']:.1f}%")

    return selected
