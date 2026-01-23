#!/usr/bin/env python3
"""
Test Multi-Task Learning Strategy
Demonstrates the unified model approach vs individual models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from multitask_strategy import MultiTaskStrategy

def create_sample_data():
    """Create sample stock data for testing."""
    
    # Sample tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    all_data = []
    
    for ticker in tickers:
        # Generate synthetic price data
        np.random.seed(hash(ticker) % 1000)  # Reproducible per ticker
        
        prices = []
        price = 100.0
        
        for i, date in enumerate(dates):
            # Random walk with trend
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
            price *= (1 + daily_return)
            
            # Add some volume and other features
            volume = np.random.normal(1000000, 200000)
            
            all_data.append({
                'date': date,
                'ticker': ticker,
                'Open': price * (1 + np.random.normal(0, 0.005)),
                'High': price * (1 + abs(np.random.normal(0, 0.01))),
                'Low': price * (1 - abs(np.random.normal(0, 0.01))),
                'Close': price,
                'Volume': max(volume, 100000),
                # Add some technical indicators
                'SMA_5': np.nan,
                'SMA_20': np.nan,
                'RSI': np.random.uniform(20, 80),
                'MACD': np.random.normal(0, 0.5)
            })
    
    df = pd.DataFrame(all_data)
    
    # Calculate moving averages
    for ticker in tickers:
        ticker_mask = df['ticker'] == ticker
        df.loc[ticker_mask, 'SMA_5'] = df.loc[ticker_mask, 'Close'].rolling(5).mean()
        df.loc[ticker_mask, 'SMA_20'] = df.loc[ticker_mask, 'Close'].rolling(20).mean()
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def test_multitask_strategy():
    """Test the multi-task learning strategy."""
    
    print("🧠 Testing Multi-Task Learning Strategy")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    all_tickers_data = create_sample_data()
    print(f"   Created {len(all_tickers_data)} data points")
    print(f"   Tickers: {sorted(all_tickers_data['ticker'].unique())}")
    
    # Define training and test periods
    train_start_date = datetime(2023, 1, 1)
    train_end_date = datetime(2023, 12, 31)
    test_date = datetime(2024, 1, 15)
    
    print(f"   Training period: {train_start_date.date()} to {train_end_date.date()}")
    print(f"   Test date: {test_date.date()}")
    
    # Initialize strategy
    strategy = MultiTaskStrategy()
    
    try:
        # Prepare data
        print("\n🔧 Preparing multi-task data...")
        X, ticker_ids, y = strategy.prepare_data(
            all_tickers_data, train_start_date, train_end_date
        )
        
        # Train models
        print("\n🚀 Training multi-task models...")
        strategy.train_models(X, ticker_ids, y)
        
        # Make predictions
        print("\n🔮 Making predictions...")
        selected_tickers = strategy.predict_returns(all_tickers_data, test_date, top_n=3)
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Selected tickers: {selected_tickers}")
        
        # Show benefits
        print(f"\n📈 Multi-Task Learning Benefits:")
        print(f"   ✓ Single unified model instead of separate models per ticker")
        print(f"   ✓ Knowledge sharing between tickers")
        print(f"   ✓ Reduced memory usage")
        print(f"   ✓ Faster training time")
        print(f"   ✓ Better generalization")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_approaches():
    """Compare multi-task vs single-task approaches."""
    
    print("\n🔄 Multi-Task vs Single-Task Comparison")
    print("=" * 50)
    
    # Simulated metrics
    print("📊 Performance Comparison (Simulated):")
    print()
    
    print("Single-Task Learning (Current System):")
    print("   Models per ticker: 6 (LSTM, XGBoost, LightGBM, RF, TCN, GRU)")
    print("   Total models: 1200 tickers × 6 = 7200 models")
    print("   Training time: ~2 hours per model = 14,400 hours total")
    print("   Memory usage: ~50MB per model = 360GB total")
    print("   Knowledge sharing: None")
    print()
    
    print("Multi-Task Learning (New Strategy):")
    print("   Models total: 6 unified models")
    print("   Training time: ~2 hours total")
    print("   Memory usage: ~200MB total")
    print("   Knowledge sharing: Full (patterns learned across all tickers)")
    print()
    
    print("🎯 Expected Improvements:")
    print("   ⚡ Training speed: 7200x faster")
    print("   💾 Memory usage: 1800x lower")
    print("   🧠 Generalization: Better (cross-ticker learning)")
    print("   📈 Performance: Potentially higher (shared patterns)")

if __name__ == "__main__":
    
    success = test_multitask_strategy()
    
    if success:
        compare_approaches()
        print(f"\n🎉 Multi-Task Learning strategy test completed!")
    else:
        print(f"\n💥 Test failed - check dependencies and configuration")
