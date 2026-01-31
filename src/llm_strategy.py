"""
LLM Strategy Module - DeepSeek via Ollama for stock selection.

This module provides an LLM-based stock selection strategy that uses
DeepSeek (via Ollama) to analyze stock data and make predictions.
"""

import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import configuration
try:
    from config import (
        LLM_OLLAMA_BASE_URL, LLM_OLLAMA_MODEL, LLM_OLLAMA_TIMEOUT,
        LLM_PARALLEL_WORKERS, LLM_MIN_SCORE
    )
    OLLAMA_BASE_URL = LLM_OLLAMA_BASE_URL
    OLLAMA_MODEL = LLM_OLLAMA_MODEL
    OLLAMA_TIMEOUT = LLM_OLLAMA_TIMEOUT
except ImportError:
    # Fallback defaults
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "deepseek-r1:8b"
    OLLAMA_TIMEOUT = 60
    LLM_PARALLEL_WORKERS = 4
    LLM_MIN_SCORE = 0.0


def check_ollama_available() -> bool:
    """Check if Ollama is running and the configured model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            # Check if configured model is available (exact or partial match)
            for name in model_names:
                if OLLAMA_MODEL in name or name in OLLAMA_MODEL:
                    print(f"   ✅ LLM model '{name}' available")
                    return True
            print(f"   ⚠️ Model '{OLLAMA_MODEL}' not found. Available: {model_names}")
            # Fall back to any available model
            if model_names:
                print(f"   💡 Will use first available model: {model_names[0]}")
                return True
            return False
        return False
    except Exception as e:
        print(f"   ⚠️ Ollama not available: {e}")
        return False


def _prepare_stock_summary(ticker: str, data: pd.DataFrame, full_year_data: pd.DataFrame = None) -> str:
    """
    Prepare a comprehensive summary of stock data for LLM analysis.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with OHLCV data (last 60 days for short-term metrics)
        full_year_data: DataFrame with full year data (for 1Y metrics)
    
    Returns:
        String summary of stock metrics
    """
    if data.empty or len(data) < 20:
        return None
    
    try:
        # Calculate key metrics
        close = data['Close'].values
        high = data['High'].values if 'High' in data.columns else close
        low = data['Low'].values if 'Low' in data.columns else close
        volume = data['Volume'].values if 'Volume' in data.columns else None
        
        # Price metrics
        current_price = close[-1]
        price_5d_ago = close[-5] if len(close) >= 5 else close[0]
        price_20d_ago = close[-20] if len(close) >= 20 else close[0]
        price_60d_ago = close[0]
        
        return_5d = ((current_price / price_5d_ago) - 1) * 100
        return_20d = ((current_price / price_20d_ago) - 1) * 100
        return_60d = ((current_price / price_60d_ago) - 1) * 100
        
        # 1-Year return (most important!)
        return_1y = "N/A"
        if full_year_data is not None and len(full_year_data) >= 200:
            try:
                year_close = full_year_data['Close'].dropna().values
                if len(year_close) >= 200:
                    price_1y_ago = year_close[0]
                    if price_1y_ago > 0:
                        return_1y = f"{((current_price / price_1y_ago) - 1) * 100:+.1f}%"
            except:
                pass
        
        # Volatility (20-day)
        returns = np.diff(close) / close[:-1]
        volatility_20d = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else 0
        
        # Volume analysis
        vol_trend = "N/A"
        avg_volume = "N/A"
        if volume is not None and len(volume) >= 20:
            recent_vol = np.mean(volume[-5:])
            avg_vol = np.mean(volume[-20:])
            avg_volume = f"{avg_vol/1e6:.1f}M" if avg_vol >= 1e6 else f"{avg_vol/1e3:.0f}K"
            if recent_vol > avg_vol * 1.5:
                vol_trend = "surging (+50%)"
            elif recent_vol > avg_vol * 1.2:
                vol_trend = "increasing"
            elif recent_vol < avg_vol * 0.8:
                vol_trend = "decreasing"
            else:
                vol_trend = "stable"
        
        # 52-week high/low (use available data)
        all_close = full_year_data['Close'].dropna().values if full_year_data is not None else close
        high_52w = np.max(all_close) if len(all_close) > 0 else current_price
        low_52w = np.min(all_close) if len(all_close) > 0 else current_price
        pct_from_52w_high = ((current_price / high_52w) - 1) * 100 if high_52w > 0 else 0
        pct_from_52w_low = ((current_price / low_52w) - 1) * 100 if low_52w > 0 else 0
        
        # Price position relative to 20-day range
        high_20d = np.max(close[-20:])
        low_20d = np.min(close[-20:])
        price_position = ((current_price - low_20d) / (high_20d - low_20d)) * 100 if high_20d != low_20d else 50
        
        # RSI (14-day)
        rsi = "N/A"
        if len(returns) >= 14:
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = f"{100 - (100 / (1 + rs)):.0f}"
            else:
                rsi = "100" if avg_gain > 0 else "50"
        
        # Moving averages
        sma_10 = np.mean(close[-10:])
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        
        # Trend analysis
        if sma_10 > sma_20 > sma_50:
            trend = "strong uptrend (SMA10 > SMA20 > SMA50)"
        elif sma_10 > sma_20:
            trend = "uptrend (SMA10 > SMA20)"
        elif sma_10 < sma_20 < sma_50:
            trend = "strong downtrend (SMA10 < SMA20 < SMA50)"
        elif sma_10 < sma_20:
            trend = "downtrend (SMA10 < SMA20)"
        else:
            trend = "sideways"
        
        # Price vs moving averages
        above_sma20 = "above" if current_price > sma_20 else "below"
        above_sma50 = "above" if current_price > sma_50 else "below"
        
        summary = f"""Ticker: {ticker}
Current Price: ${current_price:.2f}

RETURNS:
- 5-day: {return_5d:+.1f}%
- 20-day (1 month): {return_20d:+.1f}%
- 60-day (3 months): {return_60d:+.1f}%
- 1-year: {return_1y}

TECHNICAL INDICATORS:
- RSI (14): {rsi}
- Trend: {trend}
- Price vs SMA20: {above_sma20}
- Price vs SMA50: {above_sma50}
- Volatility (annualized): {volatility_20d:.1f}%

PRICE LEVELS:
- 52-week high: ${high_52w:.2f} ({pct_from_52w_high:+.1f}% from current)
- 52-week low: ${low_52w:.2f} ({pct_from_52w_low:+.1f}% from current)
- Position in 20d range: {price_position:.0f}% (0=low, 100=high)

VOLUME:
- Average daily volume: {avg_volume}
- Recent volume trend: {vol_trend}"""
        
        return summary
    
    except Exception as e:
        return None


def _query_llm_for_stock(ticker: str, summary: str, context: str = "") -> Tuple[str, float, str]:
    """
    Query the LLM for a stock prediction.
    
    Args:
        ticker: Stock ticker symbol
        summary: Stock data summary
        context: Additional market context
    
    Returns:
        Tuple of (ticker, score, reasoning)
        Score is between -1 (strong sell) and +1 (strong buy)
    """
    prompt = f"""You are a quantitative stock analyst. Analyze this stock and provide a prediction score.

{summary}

{context}

Based on the data above, rate this stock's expected performance over the next 10 trading days.
Respond with ONLY a JSON object in this exact format:
{{"score": <number between -1 and 1>, "reasoning": "<brief 1-sentence explanation>"}}

Where:
- score of 1.0 = strong buy (expect >5% gain)
- score of 0.5 = moderate buy (expect 2-5% gain)
- score of 0.0 = neutral (expect flat)
- score of -0.5 = moderate sell (expect 2-5% loss)
- score of -1.0 = strong sell (expect >5% loss)

Respond with ONLY the JSON, no other text."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent outputs
                    "num_predict": 150   # Limit response length
                }
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '').strip()
            
            # Parse JSON from response
            try:
                # Try to extract JSON from the response (handle thinking tags from DeepSeek)
                # Remove <think>...</think> tags if present
                import re
                text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                
                if '{' in text_clean and '}' in text_clean:
                    json_start = text_clean.index('{')
                    json_end = text_clean.rindex('}') + 1
                    json_str = text_clean[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    score = float(parsed.get('score', 0))
                    score = max(-1, min(1, score))  # Clamp to [-1, 1]
                    reasoning = parsed.get('reasoning', 'No reasoning provided')
                    
                    return (ticker, score, reasoning)
                
                # Fallback: try to find score in text using regex
                score_match = re.search(r'"score"\s*:\s*([-\d.]+)', text)
                if score_match:
                    score = float(score_match.group(1))
                    score = max(-1, min(1, score))
                    return (ticker, score, "Parsed from partial response")
                    
            except (json.JSONDecodeError, ValueError):
                pass
        
        return (ticker, 0.0, "Failed to parse LLM response")
    
    except requests.exceptions.Timeout:
        return (ticker, 0.0, "LLM request timed out")
    except Exception as e:
        return (ticker, 0.0, f"LLM error: {str(e)[:50]}")


def get_llm_predictions(
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    current_date: datetime,
    top_n: int = 10,
    parallel_workers: int = 4,
    verbose: bool = False
) -> List[Tuple[str, float, str]]:
    """
    Get LLM-based predictions for a list of tickers.
    
    Args:
        tickers: List of ticker symbols to analyze
        all_tickers_data: DataFrame with all ticker data
        current_date: Current backtest date
        top_n: Number of top predictions to return
        parallel_workers: Number of parallel LLM requests
        verbose: Print detailed progress
    
    Returns:
        List of (ticker, score, reasoning) tuples, sorted by score descending
    """
    if not check_ollama_available():
        print("   ⚠️ LLM Strategy: Ollama not available, skipping")
        return []
    
    predictions = []
    summaries = {}
    
    # Prepare summaries for all tickers
    for ticker in tickers:
        try:
            ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker]
            if ticker_data.empty:
                continue
            
            ticker_data = ticker_data.set_index('date')
            data_slice = ticker_data.loc[:current_date].tail(60)  # Last 60 days for short-term
            full_year_data = ticker_data.loc[:current_date].tail(365)  # Full year for 1Y metrics
            
            summary = _prepare_stock_summary(ticker, data_slice, full_year_data)
            if summary:
                summaries[ticker] = summary
        except Exception:
            continue
    
    if not summaries:
        return []
    
    if verbose:
        print(f"   🤖 LLM Strategy: Analyzing {len(summaries)} stocks...")
    
    # Query LLM in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(_query_llm_for_stock, ticker, summary): ticker
            for ticker, summary in summaries.items()
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            predictions.append(result)
            completed += 1
            
            if verbose and completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   📊 LLM Progress: {completed}/{len(summaries)} ({elapsed:.1f}s)")
    
    # Sort by score (descending) and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"   ✅ LLM Strategy: Completed {len(predictions)} predictions in {elapsed:.1f}s")
        if predictions:
            print(f"   🏆 Top pick: {predictions[0][0]} (score: {predictions[0][1]:.2f}) - {predictions[0][2]}")
    
    return predictions[:top_n]


def select_llm_portfolio(
    tickers: List[str],
    all_tickers_data: pd.DataFrame,
    current_date: datetime,
    portfolio_size: int = 10,
    min_score: float = 0.0
) -> List[str]:
    """
    Select a portfolio of stocks using LLM predictions.
    
    Args:
        tickers: List of candidate tickers
        all_tickers_data: DataFrame with all ticker data
        current_date: Current backtest date
        portfolio_size: Number of stocks to select
        min_score: Minimum score threshold for selection
    
    Returns:
        List of selected ticker symbols
    """
    predictions = get_llm_predictions(
        tickers, all_tickers_data, current_date,
        top_n=portfolio_size * 2,  # Get more than needed for filtering
        verbose=True
    )
    
    # Filter by minimum score and take top N
    selected = [
        ticker for ticker, score, _ in predictions
        if score >= min_score
    ][:portfolio_size]
    
    return selected


# For testing
if __name__ == "__main__":
    print("Testing LLM Strategy Module...")
    
    if check_ollama_available():
        print("✅ Ollama is available")
        
        # Test with dummy data
        test_summary = """Ticker: AAPL
Price: $185.50
20-day return: +3.2%
60-day return: +8.5%
Volatility (annualized): 22.5%
Volume trend: stable
Price position in 20d range: 75% (0=low, 100=high)
Momentum (SMA10 vs SMA20): bullish"""
        
        result = _query_llm_for_stock("AAPL", test_summary)
        print(f"Test result: {result}")
    else:
        print("❌ Ollama not available")
