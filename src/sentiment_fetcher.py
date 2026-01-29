"""
Real Sentiment Data Fetcher
Uses free APIs to get real sentiment data for stocks.

Supported APIs:
1. Alpha Vantage News & Sentiment (FREE - 500 calls/day)
2. Reddit API (FREE - via PRAW)
3. Mock data fallback
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict

# Import config
from config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_MAX_CALLS_PER_MINUTE

class SentimentDataFetcher:
    """Fetches real sentiment data from various free APIs."""
    
    def __init__(self):
        self.alpha_vantage_cache = {}
        self.last_alpha_vantage_call = 0
        self.alpha_vantage_call_count = 0
        self.alpha_vantage_reset_time = time.time() + 60  # Reset every minute
        
    def _rate_limit_alpha_vantage(self):
        """Implement rate limiting for Alpha Vantage API."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time > self.alpha_vantage_reset_time:
            self.alpha_vantage_call_count = 0
            self.alpha_vantage_reset_time = current_time + 60
        
        # Check if we're at the limit
        if self.alpha_vantage_call_count >= ALPHA_VANTAGE_MAX_CALLS_PER_MINUTE:
            sleep_time = self.alpha_vantage_reset_time - current_time
            if sleep_time > 0:
                print(f"   ⏰ Alpha Vantage rate limit reached, sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.alpha_vantage_call_count = 0
        
        # Minimum delay between calls
        if current_time - self.last_alpha_vantage_call < 2.4:  # 25 calls per minute = 2.4s between calls
            time.sleep(2.4 - (current_time - self.last_alpha_vantage_call))
        
        self.last_alpha_vantage_call = time.time()
        self.alpha_vantage_call_count += 1
    
    def get_alpha_vantage_news_sentiment(self, ticker: str, current_date: datetime) -> Dict:
        """
        Get news sentiment from Alpha Vantage API.
        
        Returns:
            Dict with sentiment data: {'sentiment': float, 'news_count': int, 'relevance': float}
        """
        # Check cache first (cache for 1 hour)
        cache_key = f"{ticker}_{current_date.strftime('%Y-%m-%d')}"
        if cache_key in self.alpha_vantage_cache:
            cached_time = self.alpha_vantage_cache[cache_key]['timestamp']
            if time.time() - cached_time < 3600:  # 1 hour cache
                return self.alpha_vantage_cache[cache_key]['data']
        
        # Skip if no API key
        if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
            return self._get_mock_sentiment(ticker)
        
        self._rate_limit_alpha_vantage()
        
        try:
            # Get news sentiment
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': ALPHA_VANTAGE_API_KEY,
                'limit': 50  # Get recent news
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                print(f"   ⚠️ Alpha Vantage API error: {data['Error Message']}")
                return self._get_mock_sentiment(ticker)
            
            # Parse sentiment data
            sentiment_score = 0.0
            news_count = 0
            relevance_score = 0.0
            
            if 'feed' in data:
                for article in data['feed']:
                    # Extract ticker-specific sentiment
                    ticker_sentiment = None
                    ticker_relevance = 0.0
                    
                    if 'ticker_sentiment' in article:
                        for ticker_data in article['ticker_sentiment']:
                            if ticker_data['ticker'] == ticker:
                                ticker_sentiment = float(ticker_data['ticker_sentiment_score'])
                                ticker_relevance = float(ticker_data['relevance_score'])
                                break
                    
                    if ticker_sentiment is not None:
                        sentiment_score += ticker_sentiment * ticker_relevance
                        relevance_score += ticker_relevance
                        news_count += 1
            
            # Calculate average sentiment
            if news_count > 0:
                sentiment_score = sentiment_score / relevance_score if relevance_score > 0 else 0.0
                relevance_score = relevance_score / news_count
            else:
                # No news found, use neutral sentiment
                sentiment_score = 0.0
                relevance_score = 0.0
            
            result = {
                'sentiment': sentiment_score,
                'news_count': news_count,
                'relevance': relevance_score,
                'source': 'alpha_vantage'
            }
            
            # Cache the result
            self.alpha_vantage_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Error fetching Alpha Vantage sentiment for {ticker}: {e}")
            return self._get_mock_sentiment(ticker)
    
    def get_reddit_sentiment(self, ticker: str, current_date: datetime) -> Dict:
        """
        Get sentiment from Reddit (using public API, no auth required).
        This is a simplified implementation.
        
        Returns:
            Dict with sentiment data: {'sentiment': float, 'post_count': int, 'source': 'reddit'}
        """
        try:
            # Search for ticker mentions in r/wallstreetbets
            url = "https://www.reddit.com/r/wallstreetbets/search.json"
            params = {
                'q': ticker,
                'sort': 'new',
                'limit': 100,
                't': 'week'  # Last week
            }
            
            headers = {'User-Agent': 'StockSentimentBot/1.0'}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                # Simple sentiment analysis based on keywords
                positive_words = ['moon', 'rocket', 'buy', 'hold', 'bull', 'diamond', 'hands', 'yolo', 'gain', 'profit']
                negative_words = ['sell', 'bear', 'crash', 'loss', 'dump', 'short', 'put', 'margin', 'liquidate']
                
                sentiment_score = 0.0
                post_count = len(posts)
                
                for post in posts:
                    title = post['data'].get('title', '').lower()
                    text = post['data'].get('selftext', '').lower()
                    combined_text = title + ' ' + text
                    
                    # Count positive and negative words
                    positive_count = sum(1 for word in positive_words if word in combined_text)
                    negative_count = sum(1 for word in negative_words if word in combined_text)
                    
                    # Simple sentiment calculation
                    if positive_count > 0 or negative_count > 0:
                        post_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                        sentiment_score += post_sentiment
                
                # Average sentiment
                if post_count > 0:
                    sentiment_score = sentiment_score / post_count
                else:
                    sentiment_score = 0.0
                
                return {
                    'sentiment': sentiment_score,
                    'post_count': post_count,
                    'source': 'reddit'
                }
            else:
                return self._get_mock_sentiment(ticker)
                
        except Exception as e:
            print(f"   ⚠️ Error fetching Reddit sentiment for {ticker}: {e}")
            return self._get_mock_sentiment(ticker)
    
    def _get_mock_sentiment(self, ticker: str) -> Dict:
        """
        Fallback mock sentiment data.
        """
        # Expanded mock sentiment data for more stocks
        mock_data = {
            # Tech - Positive
            'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.15, 'AMZN': 0.18, 'META': 0.22,
            'NVDA': 0.35, 'TSLA': 0.12, 'AMD': 0.28, 'NFLX': 0.10, 'CRM': 0.15,
            'ADBE': 0.20, 'INTC': 0.05, 'PYPL': 0.08, 'SQ': 0.18, 'SHOP': 0.22,
            
            # Finance - Mixed
            'JPM': 0.10, 'BAC': -0.05, 'WFC': -0.10, 'C': -0.08, 'GS': 0.12,
            'MS': 0.08, 'BLK': 0.15, 'SPGI': 0.10, 'ICE': 0.05, 'CME': 0.08,
            
            # Healthcare - Neutral to Positive
            'JNJ': 0.08, 'PFE': 0.05, 'UNH': 0.12, 'ABT': 0.10, 'TMO': 0.15,
            'DHR': 0.08, 'BMY': -0.05, 'AMGN': 0.10, 'GILD': 0.02, 'MRK': 0.08,
            
            # Consumer - Mixed
            'HD': 0.12, 'MCD': 0.08, 'NKE': 0.10, 'DIS': 0.05, 'KO': 0.02,
            'PEP': 0.05, 'WMT': 0.08, 'COST': 0.15, 'TGT': -0.05, 'LOW': 0.08,
            
            # Energy - Negative
            'XOM': -0.08, 'CVX': -0.05, 'COP': -0.10, 'EOG': -0.05, 'SLB': -0.12,
            
            # Industrial - Neutral
            'GE': -0.15, 'BA': 0.05, 'CAT': 0.08, 'HON': 0.10, 'MMM': 0.02,
            'UPS': 0.08, 'RTX': 0.05, 'LMT': 0.12, 'DE': 0.08, 'CAT': 0.08,
        }
        
        # Default sentiment for unknown stocks
        default_sentiment = 0.0
        
        return {
            'sentiment': mock_data.get(ticker, default_sentiment),
            'news_count': 5 if ticker in mock_data else 0,
            'relevance': 0.5 if ticker in mock_data else 0.0,
            'source': 'mock'
        }
    
    def get_combined_sentiment(self, ticker: str, current_date: datetime) -> Dict:
        """
        Get combined sentiment from multiple sources.
        
        Returns:
            Dict with combined sentiment data
        """
        # Get sentiment from different sources
        av_sentiment = self.get_alpha_vantage_news_sentiment(ticker, current_date)
        reddit_sentiment = self.get_reddit_sentiment(ticker, current_date)
        
        # Combine sentiments (weighted average)
        weights = {
            'alpha_vantage': 0.7,  # More reliable
            'reddit': 0.2,          # Social sentiment
            'mock': 0.1              # Fallback
        }
        
        # Determine which source to use as primary
        if av_sentiment['source'] != 'mock':
            primary_sentiment = av_sentiment['sentiment']
            primary_weight = weights['alpha_vantage']
        else:
            primary_sentiment = av_sentiment['sentiment']
            primary_weight = weights['mock']
        
        # Add Reddit sentiment if available
        if reddit_sentiment['source'] == 'reddit' and reddit_sentiment['post_count'] > 0:
            combined_sentiment = (
                primary_sentiment * primary_weight +
                reddit_sentiment['sentiment'] * weights['reddit']
            ) / (primary_weight + weights['reddit'])
        else:
            combined_sentiment = primary_sentiment
        
        # Calculate confidence based on data sources
        confidence = 0.0
        if av_sentiment['source'] != 'mock':
            confidence += 0.7
        if reddit_sentiment['source'] == 'reddit' and reddit_sentiment['post_count'] > 0:
            confidence += 0.3
        
        return {
            'combined': combined_sentiment,
            'confidence': confidence,
            'sources': [av_sentiment['source'], reddit_sentiment['source']],
            'news_count': av_sentiment.get('news_count', 0),
            'reddit_posts': reddit_sentiment.get('post_count', 0)
        }


# Global instance
_sentiment_fetcher = None

def get_sentiment_fetcher():
    """Get the global sentiment fetcher instance."""
    global _sentiment_fetcher
    if _sentiment_fetcher is None:
        _sentiment_fetcher = SentimentDataFetcher()
    return _sentiment_fetcher
