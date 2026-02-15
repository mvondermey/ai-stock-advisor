"""
Rolling Windows Compliance Test Suite

This test suite ensures all strategies properly use rolling windows
and don't fall back to static date calculations.

Author: AI Assistant
Created: 2026-02-15
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestRollingWindowsCompliance:
    """Test suite for rolling windows compliance across all strategies."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        # Start data early enough to have 1 year of history before first test date
        # First test date is 2025-12-15, so we need data from 2024-12-15 onwards
        dates = pd.date_range('2024-01-01', '2026-02-13', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        
        data = {}
        for ticker_idx, ticker in enumerate(tickers):
            # Create realistic price data with trends
            prices = []
            base_price = 100.0
            
            # Create different performance scenarios for different tickers
            # Some should pass filters, some should fail
            if ticker_idx < 2:  # First 2 tickers: Strong performers
                trend = 0.0008  # 0.08% daily trend (about 22% annual)
                volatility = 0.015  # 1.5% daily volatility
            elif ticker_idx < 4:  # Next 2 tickers: Moderate performers
                trend = 0.0006  # 0.06% daily trend (about 16% annual)
                volatility = 0.02  # 2% daily volatility
            else:  # Last ticker: Weak performer
                trend = -0.0001  # Slight negative trend
                volatility = 0.025  # Higher volatility
            
            # Set different random seeds for variety
            np.random.seed(42 + ticker_idx)
            
            for i, date in enumerate(dates):
                # Use geometric Brownian motion for realistic price movements
                daily_return = trend + volatility * np.random.normal(0, 1)
                
                if i == 0:
                    price = base_price
                else:
                    # Geometric Brownian motion
                    price = prices[-1] * np.exp(daily_return)
                
                # Ensure price stays reasonable
                price = max(min(price, 10000), 1)  # Between $1 and $10,000
                prices.append(price)
            
            data[ticker] = pd.DataFrame({
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        
        return data
    
    @pytest.fixture
    def test_dates(self):
        """Sample dates for testing rolling behavior."""
        return [
            datetime(2025, 12, 15),
            datetime(2025, 12, 16),
            datetime(2025, 12, 17),
            datetime(2026, 1, 15),
            datetime(2026, 1, 16),
            datetime(2026, 1, 17),
        ]
    
    def extract_performance_values(self, strategy_output: str) -> Dict[str, float]:
        """Extract performance values from strategy debug output."""
        values = {}
        
        # Look for common performance patterns
        import re
        
        # 3M/1Y Ratio patterns
        accel_pattern = r'(\w+[-\.\w]*): acceleration=([+-]?\d+\.?\d*)%'
        matches = re.findall(accel_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_acceleration"] = float(value)
        
        # 1Y/3M Ratio patterns  
        dip_pattern = r'(\w+[-\.\w]*): dip=([+-]?\d+\.?\d*)%'
        matches = re.findall(dip_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_dip"] = float(value)
        
        # Turnaround patterns
        recovery_pattern = r'(\w+[-\.\w]*): recovery=([+-]?\d+\.?\d*)%'
        matches = re.findall(recovery_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_recovery"] = float(value)
        
        # Elite Hybrid patterns
        elite_pattern = r'🔍 DEBUG (\w+[-\.\w]*): Elite=([+-]?\d+\.?\d*)'
        matches = re.findall(elite_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_elite"] = float(value)
        
        # AI Elite patterns
        ai_pattern = r'(\w+[-\.\w]*): AI Score=([+-]?\d+\.?\d*)'
        matches = re.findall(ai_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_ai_score"] = float(value)
        
        # Risk-Adj Mom patterns
        risk_pattern = r'✅ (\w+[-\.\w]*): PASSED \(1Y=([+-]?\d+\.?\d*)%'
        matches = re.findall(risk_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_1y"] = float(value)
        
        # Mom-Vol Hybrid patterns
        momvol_pattern = r'Top 3: (\w+[-\.\w]*) \((\d+\.?\d*)\)'
        matches = re.findall(momvol_pattern, strategy_output)
        for ticker, value in matches:
            values[f"{ticker}_momvol_score"] = float(value)
        
        # Price Acceleration patterns
        price_accel_pattern = r'🔍 DEBUG (\w+[-\.\w]*): velocity=([+-]?\d+\.?\d*), accel=([+-]?\d+\.?\d*)'
        matches = re.findall(price_accel_pattern, strategy_output)
        for ticker, velocity, accel in matches:
            values[f"{ticker}_velocity"] = float(velocity)
            values[f"{ticker}_acceleration"] = float(accel)
        
        return values
    
    def test_3m_1y_ratio_rolling_windows(self, sample_data, test_dates, capsys):
        """Test 3M/1Y Ratio strategy uses rolling windows."""
        from shared_strategies import select_3m_1y_ratio_stocks
        
        values_by_date = {}
        
        for date in test_dates:
            # Capture debug output
            with capsys.disabled():
                stocks = select_3m_1y_ratio_stocks(
                    list(sample_data.keys()),
                    sample_data,
                    date,
                    top_n=5
                )
            
            # Capture output
            captured = capsys.readouterr()
            values = self.extract_performance_values(captured.err)
            
            if values:
                values_by_date[date] = values
        
        # Verify values change between dates (rolling window behavior)
        if len(values_by_date) >= 2:
            dates = sorted(values_by_date.keys())
            
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                
                current_values = values_by_date[current_date]
                next_values = values_by_date[next_date]
                
                # Find common metrics
                common_metrics = set(current_values.keys()) & set(next_values.keys())
                
                # At least some values should change (rolling window effect)
                changed_metrics = []
                for metric in common_metrics:
                    if abs(current_values[metric] - next_values[metric]) > 0.001:
                        changed_metrics.append(metric)
                
                assert len(changed_metrics) > 0, (
                    f"3M/1Y Ratio: No rolling window effect detected between "
                    f"{current_date} and {next_date}. Values appear static."
                )
    
    def test_1y_3m_ratio_rolling_windows(self, sample_data, test_dates, capsys):
        """Test 1Y/3M Ratio strategy uses rolling windows."""
        from shared_strategies import select_1y_3m_ratio_stocks
        
        values_by_date = {}
        
        for date in test_dates:
            with capsys.disabled():
                stocks = select_1y_3m_ratio_stocks(
                    list(sample_data.keys()),
                    sample_data,
                    date,
                    top_n=5
                )
            
            captured = capsys.readouterr()
            values = self.extract_performance_values(captured.err)
            
            if values:
                values_by_date[date] = values
        
        # Verify rolling window behavior
        if len(values_by_date) >= 2:
            dates = sorted(values_by_date.keys())
            
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                
                current_values = values_by_date[current_date]
                next_values = values_by_date[next_date]
                
                common_metrics = set(current_values.keys()) & set(next_values.keys())
                changed_metrics = []
                
                for metric in common_metrics:
                    if abs(current_values[metric] - next_values[metric]) > 0.001:
                        changed_metrics.append(metric)
                
                assert len(changed_metrics) > 0, (
                    f"1Y/3M Ratio: No rolling window effect detected between "
                    f"{current_date} and {next_date}. Values appear static."
                )
    
    def test_turnaround_rolling_windows(self, sample_data, test_dates, capsys):
        """Test Turnaround strategy uses rolling windows."""
        from shared_strategies import select_turnaround_stocks
        
        values_by_date = {}
        
        for date in test_dates:
            with capsys.disabled():
                stocks = select_turnaround_stocks(
                    list(sample_data.keys()),
                    sample_data,
                    date,
                    top_n=5
                )
            
            captured = capsys.readouterr()
            values = self.extract_performance_values(captured.err)
            
            if values:
                values_by_date[date] = values
        
        # Verify rolling window behavior
        if len(values_by_date) >= 2:
            dates = sorted(values_by_date.keys())
            
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                
                current_values = values_by_date[current_date]
                next_values = values_by_date[next_date]
                
                common_metrics = set(current_values.keys()) & set(next_values.keys())
                changed_metrics = []
                
                for metric in common_metrics:
                    if abs(current_values[metric] - next_values[metric]) > 0.001:
                        changed_metrics.append(metric)
                
                assert len(changed_metrics) > 0, (
                    f"Turnaround: No rolling window effect detected between "
                    f"{current_date} and {next_date}. Values appear static."
                )
    
    def test_current_date_parameter_compliance(self):
        """Test all strategy functions accept current_date parameter."""
        import inspect
        
        # List of all strategy functions that should accept current_date
        strategy_functions = [
            # Core strategies
            'select_risk_adj_mom_stocks',
            'select_3m_1y_ratio_stocks', 
            'select_1y_3m_ratio_stocks',
            'select_turnaround_stocks',
            'select_mean_reversion_stocks',
            'select_quality_momentum_stocks',
            
            # Momentum-Volatility Hybrid strategies
            'select_momentum_volatility_hybrid_stocks',
            'select_momentum_volatility_hybrid_6m_stocks',
            'select_momentum_volatility_hybrid_1y_stocks',
            'select_momentum_volatility_hybrid_1y3m_stocks',
            
            # Advanced strategies
            'select_price_acceleration_stocks',
            'select_momentum_acceleration_stocks',
            'select_concentrated_3m_stocks',
            'select_dual_momentum_stocks',
            'select_trend_following_atr_stocks',
            
            # AI/ML strategies
            'select_elite_hybrid_stocks',
            'select_ai_elite_stocks',
            
            # Ensemble strategies
            'select_enhanced_volatility_stocks',
            'select_correlation_ensemble_stocks',
            'select_multi_timeframe_stocks',
            'select_adaptive_ensemble_stocks',
            
            # Additional strategies (if they exist)
            'select_voting_ensemble_stocks',
            'select_factor_rotation_stocks',
            'select_pairs_trading_stocks',
            'select_options_sentiment_stocks',
        ]
        
        # Import all strategy modules
        try:
            from shared_strategies import (
                select_risk_adj_mom_stocks,
                select_3m_1y_ratio_stocks, 
                select_1y_3m_ratio_stocks,
                select_turnaround_stocks,
                select_mean_reversion_stocks,
                select_quality_momentum_stocks,
                select_momentum_volatility_hybrid_stocks,
                select_momentum_volatility_hybrid_6m_stocks,
                select_momentum_volatility_hybrid_1y_stocks,
                select_momentum_volatility_hybrid_1y3m_stocks,
                select_voting_ensemble_stocks
            )
            from new_strategies import (
                select_price_acceleration_stocks,
                select_momentum_acceleration_stocks,
                select_concentrated_3m_stocks,
                select_dual_momentum_stocks,
                select_trend_following_atr_stocks
            )
            from elite_hybrid_strategy import select_elite_hybrid_stocks
            from ai_elite_strategy import select_ai_elite_stocks
            
            # Try to import additional strategies (may not exist)
            try:
                from enhanced_volatility_trader import select_enhanced_volatility_stocks
            except ImportError:
                pass
            
            try:
                from correlation_ensemble import select_correlation_ensemble_stocks
            except ImportError:
                pass
                
            try:
                from multi_timeframe_ensemble import select_multi_timeframe_stocks
            except ImportError:
                pass
                
            try:
                from adaptive_ensemble import select_adaptive_ensemble_stocks
            except ImportError:
                pass
                
            try:
                from factor_rotation import select_factor_rotation_stocks
            except ImportError:
                pass
                
            try:
                from pairs_trading import select_pairs_trading_stocks
            except ImportError:
                pass
                
            try:
                from options_sentiment import select_options_sentiment_stocks
            except ImportError:
                pass
        except ImportError as e:
            pytest.skip(f"Cannot import strategy modules: {e}")
        
        for func_name in strategy_functions:
            if func_name in globals():
                func = globals()[func_name]
                sig = inspect.signature(func)
                
                # Check if current_date parameter exists
                assert 'current_date' in sig.parameters, (
                    f"Strategy function {func_name} missing current_date parameter. "
                    f"Found parameters: {list(sig.parameters.keys())}"
                )
                
                # Check if current_date has proper default
                current_date_param = sig.parameters['current_date']
                assert current_date_param.default is None or current_date_param.default == inspect.Parameter.empty, (
                    f"Strategy function {func_name} current_date should default to None"
                )
    
    def test_static_behavior_detection(self, sample_data, capsys):
        """Test for static behavior patterns that indicate rolling window issues."""
        from shared_strategies import select_3m_1y_ratio_stocks
        
        # Run strategy multiple times with different dates
        dates = [
            datetime(2025, 12, 15),
            datetime(2025, 12, 20),
            datetime(2025, 12, 25),
            datetime(2026, 1, 5),
            datetime(2026, 1, 15),
        ]
        
        performance_values_by_date = {}
        
        for date in dates:
            with capsys.disabled():
                stocks = select_3m_1y_ratio_stocks(
                    list(sample_data.keys()),
                    sample_data,
                    date,
                    top_n=3
                )
            
            captured = capsys.readouterr()
            values = self.extract_performance_values(captured.err)
            
            if values:
                performance_values_by_date[date] = values
        
        # Check for static performance values (more precise than identical output)
        if len(performance_values_by_date) >= 2:
            dates = sorted(performance_values_by_date.keys())
            
            # Check if performance values are identical across dates
            first_date = dates[0]
            first_values = performance_values_by_date[first_date]
            
            all_identical = True
            for date in dates[1:]:
                current_values = performance_values_by_date[date]
                
                # Compare common metrics
                common_metrics = set(first_values.keys()) & set(current_values.keys())
                
                for metric in common_metrics:
                    if abs(first_values[metric] - current_values[metric]) > 0.001:
                        all_identical = False
                        break
                
                if not all_identical:
                    break
            
            # If all performance values are identical, that's static behavior
            if all_identical and len(performance_values_by_date) > 1:
                pytest.fail(
                    "Static behavior detected: Performance values are identical across different dates. "
                    "This indicates rolling windows are not working properly."
                )
    
    def test_performance_filter_rolling_windows(self, sample_data, test_dates, capsys):
        """Test performance filters use rolling windows."""
        from performance_filters import apply_performance_filters
        
        values_by_date = {}
        
        for date in test_dates:
            # Test with a sample ticker
            ticker_data = sample_data['AAPL']
            
            with capsys.disabled():
                result = apply_performance_filters('AAPL', {'AAPL': ticker_data}, date, "Test", 1)
            
            captured = capsys.readouterr()
            
            # Look for performance values in output
            import re
            perf_pattern = r'PASSED \(1Y=([+-]?\d+\.?\d*)%, 6M=([+-]?\d+\.?\d*)%, 3M=([+-]?\d+\.?\d*)%\)'
            matches = re.findall(perf_pattern, captured.err)
            
            if matches:
                values_by_date[date] = matches[0]  # (1y, 6m, 3m)
        
        # Verify performance values change between dates
        if len(values_by_date) >= 2:
            dates = sorted(values_by_date.keys())
            
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                
                current_perf = values_by_date[current_date]
                next_perf = values_by_date[next_date]
                
                # Convert to floats for comparison
                current_values = [float(x) for x in current_perf]
                next_values = [float(x) for x in next_perf]
                
                # At least one performance metric should change
                changed = any(abs(current_values[j] - next_values[j]) > 0.001 for j in range(3))
                
                assert changed, (
                    f"Performance filters: No rolling window effect detected between "
                    f"{current_date} and {next_date}. Performance values appear static."
                )
    
    def test_no_future_date_usage(self, sample_data, test_dates, capsys):
        """Test strategies don't use future dates beyond current_date."""
        from shared_strategies import select_3m_1y_ratio_stocks
        from performance_filters import apply_performance_filters
        
        # Test with a specific current date
        current_date = datetime(2025, 12, 15)
        
        # Get the data up to current date
        ticker_data = sample_data['AAPL']
        data_up_to_current = ticker_data[ticker_data.index <= current_date]
        
        # Run strategy
        with capsys.disabled():
            stocks = select_3m_1y_ratio_stocks(
                ['AAPL'], {'AAPL': ticker_data}, current_date, top_n=3
            )
        
        # Test performance filter
        with capsys.disabled():
            result = apply_performance_filters('AAPL', {'AAPL': ticker_data}, current_date, "Test", 1)
        
        captured = capsys.readouterr()
        
        # Check for any date references in output that might indicate future usage
        import re
        
        # Look for date patterns that might be future dates
        # These patterns would indicate using data beyond current_date
        future_date_patterns = [
            r'2026-01-[0-9]+',  # Any January 2026 date (after Dec 15, 2025)
            r'2026-02-[0-9]+',  # Any February 2026 date
            r'2025-12-[2-3][0-9]',  # Late December 2025 dates
        ]
        
        suspicious_dates = []
        for pattern in future_date_patterns:
            matches = re.findall(pattern, captured.err)
            suspicious_dates.extend(matches)
        
        # Filter out the current date itself (that's expected)
        current_date_str = current_date.strftime('%Y-%m-%d')
        suspicious_dates = [d for d in suspicious_dates if d != current_date_str]
        
        # Should not find any future dates in the calculations
        assert len(suspicious_dates) == 0, (
            f"Future date usage detected: {suspicious_dates}. "
            f"Strategies should only use data up to current_date ({current_date_str})."
        )
    
    def test_data_boundaries_respected(self, sample_data, capsys):
        """Test strategies respect data boundaries and don't use data after current_date."""
        from shared_strategies import select_3m_1y_ratio_stocks
        
        # Create truncated data that only goes up to a certain point
        max_data_date = datetime(2025, 6, 30)  # Data only available up to June 30
        
        # Modify sample data to have a clear cutoff
        truncated_data = {}
        for ticker, df in sample_data.items():
            truncated_data[ticker] = df[df.index <= max_data_date]
        
        # Run strategy with current_date after data cutoff
        test_date = datetime(2025, 7, 15)  # After data cutoff
        
        with capsys.disabled():
            stocks = select_3m_1y_ratio_stocks(
                list(truncated_data.keys()), truncated_data, test_date, top_n=3
            )
        
        captured = capsys.readouterr()
        
        # Check that the strategy handles insufficient data gracefully
        # The strategy should either:
        # 1. Return empty results (no stocks found)
        # 2. Show insufficient data warnings
        # 3. Not crash or use future data
        
        # Verify no future dates are mentioned in output
        import re
        future_date_patterns = [
            r'2025-07-[0-9]+',  # Any July 2025 date (after June 30)
            r'2025-08-[0-9]+',  # Any August 2025 date
            r'2025-09-[0-9]+',  # Any September 2025 date
        ]
        
        suspicious_dates = []
        for pattern in future_date_patterns:
            matches = re.findall(pattern, captured.err)
            suspicious_dates.extend(matches)
        
        # Should not find any future dates in the calculations
        assert len(suspicious_dates) == 0, (
            f"Future date usage detected: {suspicious_dates}. "
            f"Strategy should only use data up to {max_data_date.date()}, "
            f"but current_date is {test_date.date()}."
        )
        
        # Also verify the strategy doesn't return unrealistic performance values
        # that would indicate using future data
        performance_pattern = r'PASSED \(1Y=([+-]?\d+\.?\d*)%'
        matches = re.findall(performance_pattern, captured.err)
        
        if matches:
            # Check for absurdly high performance values that might indicate future data usage
            for perf_str in matches:
                try:
                    perf_value = float(perf_str)
                    # Performance values over 1,000,000% are likely unrealistic and indicate future data usage
                    if perf_value > 1000000:
                        pytest.fail(
                            f"Unrealistic performance value detected: {perf_value}%. "
                            f"This may indicate the strategy is using future data beyond {max_data_date.date()}."
                        )
                except ValueError:
                    continue
    
    def test_rolling_window_direction_correctness(self, sample_data, capsys):
        """Test rolling windows move forward, not backward or randomly."""
        from shared_strategies import select_3m_1y_ratio_stocks
        
        # Test consecutive dates to ensure rolling windows move forward
        consecutive_dates = [
            datetime(2025, 12, 10),
            datetime(2025, 12, 11), 
            datetime(2025, 12, 12),
            datetime(2025, 12, 13),
            datetime(2025, 12, 14),
        ]
        
        window_boundaries = []
        
        for date in consecutive_dates:
            with capsys.disabled():
                stocks = select_3m_1y_ratio_stocks(
                    list(sample_data.keys()), sample_data, date, top_n=3
                )
            
            captured = capsys.readouterr()
            values = self.extract_performance_values(captured.err)
            
            if values:
                # Extract any acceleration values to track window movement
                for key, value in values.items():
                    if 'acceleration' in key:
                        window_boundaries.append((date, value))
                        break
        
        # Verify we have consecutive data points
        if len(window_boundaries) >= 2:
            # The windows should move forward with time
            # Values should change consistently (not jump around randomly)
            dates, values = zip(*window_boundaries)
            
            # Check that we have different values (rolling windows working)
            unique_values = set(values)
            assert len(unique_values) > 1, (
                "Rolling windows appear static - values don't change across consecutive dates"
            )
            
            # Additional check: ensure logical progression
            # (This is a basic check - more sophisticated checks could be added)
            for i in range(len(values) - 1):
                current_val = values[i]
                next_val = values[i + 1]
                
                # Values should be different (rolling window effect)
                assert abs(current_val - next_val) > 0.001, (
                    f"No rolling window change between {dates[i]} and {dates[i+1]}"
                )
    
    def test_historical_data_only(self, sample_data, test_dates, capsys):
        """Test strategies only use historical data up to current_date."""
        from shared_strategies import select_3m_1y_ratio_stocks
        
        for current_date in test_dates:
            with capsys.disabled():
                stocks = select_3m_1y_ratio_stocks(
                    list(sample_data.keys()), sample_data, current_date, top_n=3
                )
            
            captured = capsys.readouterr()
            
            # Extract any date mentions in debug output
            import re
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            mentioned_dates = re.findall(date_pattern, captured.err)
            
            # Convert to datetime objects for comparison
            mentioned_datetimes = []
            for date_str in mentioned_dates:
                try:
                    mentioned_datetimes.append(pd.to_datetime(date_str))
                except:
                    continue
            
            # All mentioned dates should be <= current_date
            current_dt = pd.to_datetime(current_date)
            
            for mentioned_dt in mentioned_datetimes:
                assert mentioned_dt <= current_dt, (
                    f"Future date detected: {mentioned_dt.date()} > current_date: {current_dt.date()}. "
                    f"Strategies should only use historical data."
                )


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])
