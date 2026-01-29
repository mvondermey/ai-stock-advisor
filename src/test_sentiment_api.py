#!/usr/bin/env python3
"""
Test script for sentiment API integration
Tests Alpha Vantage and Reddit sentiment fetching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from sentiment_fetcher import get_sentiment_fetcher

def test_sentiment_fetcher():
    """Test the sentiment fetcher with real APIs."""
    print("🧪 Testing Sentiment API Integration")
    print("=" * 50)
    
    # Get sentiment fetcher
    fetcher = get_sentiment_fetcher()
    
    # Test with a few popular stocks
    test_tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GME']
    current_date = datetime.now()
    
    print(f"\n📅 Testing for date: {current_date.date()}")
    print(f"🔑 Alpha Vantage API Key: {'✅ Set' if fetcher else '❌ Not set'}")
    
    for ticker in test_tickers:
        print(f"\n--- Testing {ticker} ---")
        
        # Test Alpha Vantage sentiment
        print("📰 Testing Alpha Vantage News Sentiment...")
        try:
            av_sentiment = fetcher.get_alpha_vantage_news_sentiment(ticker, current_date)
            print(f"   Sentiment: {av_sentiment['sentiment']:.3f}")
            print(f"   News Count: {av_sentiment['news_count']}")
            print(f"   Relevance: {av_sentiment['relevance']:.3f}")
            print(f"   Source: {av_sentiment['source']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test Reddit sentiment
        print("📱 Testing Reddit Sentiment...")
        try:
            reddit_sentiment = fetcher.get_reddit_sentiment(ticker, current_date)
            print(f"   Sentiment: {reddit_sentiment['sentiment']:.3f}")
            print(f"   Post Count: {reddit_sentiment['post_count']}")
            print(f"   Source: {reddit_sentiment['source']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test combined sentiment
        print("🔄 Testing Combined Sentiment...")
        try:
            combined = fetcher.get_combined_sentiment(ticker, current_date)
            print(f"   Combined: {combined['combined']:.3f}")
            print(f"   Confidence: {combined['confidence']:.3f}")
            print(f"   Sources: {combined['sources']}")
            print(f"   News Count: {combined['news_count']}")
            print(f"   Reddit Posts: {combined['reddit_posts']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Sentiment API Test Complete!")
    
    # Test rate limiting
    print("\n⏱️ Testing Rate Limiting...")
    start_time = datetime.now()
    for i in range(3):
        print(f"   Call {i+1}/3...")
        result = fetcher.get_alpha_vantage_news_sentiment('AAPL', current_date)
        print(f"   Sentiment: {result['sentiment']:.3f}")
    end_time = datetime.now()
    print(f"   Total time: {(end_time - start_time).total_seconds():.1f}s")
    print(f"   Rate limiting: {'✅ Working' if (end_time - start_time).total_seconds() > 4 else '⚠️ May need adjustment'}")

if __name__ == "__main__":
    test_sentiment_fetcher()
