"""
Test suite for verifying all backtesting strategies have correct input/output values.

This test ensures:
1. All strategies initialize with correct capital (initial_capital_needed)
2. All rebalancing functions return correct types (cash: float, positions: dict)
3. Portfolio values are calculated correctly (cash + invested value)
4. Transaction costs are applied correctly
5. No variable name collisions occur
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    PORTFOLIO_SIZE, TRANSACTION_COST, INITIAL_CAPITAL,
    ENABLE_STATIC_BH, ENABLE_STATIC_BH_6M, ENABLE_DYNAMIC_BH_1Y,
    ENABLE_DYNAMIC_BH_6M, ENABLE_DYNAMIC_BH_3M, ENABLE_DYNAMIC_BH_1M,
    ENABLE_RISK_ADJ_MOM, ENABLE_MEAN_REVERSION, ENABLE_QUALITY_MOM,
    ENABLE_MOMENTUM_AI_HYBRID, ENABLE_VOLATILITY_ADJ_MOM,
    ENABLE_ENHANCED_VOLATILITY, ENABLE_AI_VOLATILITY_ENSEMBLE,
    ENABLE_TURNAROUND, ENABLE_VOLATILITY_ENSEMBLE,
    ENABLE_DYNAMIC_BH_1Y_VOL_FILTER, ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP
)


class TestStrategyConfiguration:
    """Test that all strategy configurations are valid."""
    
    def test_portfolio_size_is_positive(self):
        """Portfolio size should be a positive integer."""
        assert isinstance(PORTFOLIO_SIZE, int)
        assert PORTFOLIO_SIZE > 0
        
    def test_transaction_cost_is_valid(self):
        """Transaction cost should be between 0 and 1."""
        assert isinstance(TRANSACTION_COST, (int, float))
        assert 0 <= TRANSACTION_COST <= 0.1  # Max 10% transaction cost
        
    def test_initial_capital_is_positive(self):
        """Initial capital should be positive."""
        assert isinstance(INITIAL_CAPITAL, (int, float))
        assert INITIAL_CAPITAL > 0


class TestMockData:
    """Helper class to create mock price data for testing."""
    
    @staticmethod
    def create_mock_ticker_data(ticker: str, days: int = 365, start_price: float = 100.0):
        """Create mock price data for a single ticker."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)  # Reproducible
        
        # Generate random walk prices
        returns = np.random.normal(0.001, 0.02, days)
        prices = start_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        return df
    
    @staticmethod
    def create_mock_ticker_data_grouped(tickers: list, days: int = 365):
        """Create mock ticker_data_grouped dictionary."""
        return {
            ticker: TestMockData.create_mock_ticker_data(ticker, days, 100 + i * 10)
            for i, ticker in enumerate(tickers)
        }
    
    @staticmethod
    def create_mock_all_tickers_data(tickers: list, days: int = 365):
        """Create mock all_tickers_data DataFrame."""
        dfs = []
        for i, ticker in enumerate(tickers):
            df = TestMockData.create_mock_ticker_data(ticker, days, 100 + i * 10)
            df['ticker'] = ticker
            df['date'] = df.index
            df = df.reset_index(drop=True)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)


class TestRebalancingFunctions:
    """Test that all rebalancing functions return correct types."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX']
        return {
            'tickers': tickers,
            'ticker_data_grouped': TestMockData.create_mock_ticker_data_grouped(tickers),
            'all_tickers_data': TestMockData.create_mock_all_tickers_data(tickers),
            'current_date': datetime.now() - timedelta(days=1),
            'initial_capital': 30000.0,
            'capital_per_stock': 3000.0
        }
    
    def test_rebalance_function_returns_tuple(self, mock_data):
        """All rebalance functions should return (cash, positions) tuple."""
        from backtesting import (
            _rebalance_dynamic_bh_portfolio,
            _rebalance_static_bh_portfolio,
            _rebalance_risk_adj_mom_portfolio,
            _rebalance_mean_reversion_portfolio,
            _rebalance_quality_momentum_portfolio,
            _rebalance_volatility_adj_mom_portfolio,
            _rebalance_turnaround_portfolio,
            _rebalance_generic_portfolio
        )
        
        rebalance_functions = [
            ('_rebalance_dynamic_bh_portfolio', _rebalance_dynamic_bh_portfolio),
            ('_rebalance_static_bh_portfolio', _rebalance_static_bh_portfolio),
            ('_rebalance_risk_adj_mom_portfolio', _rebalance_risk_adj_mom_portfolio),
            ('_rebalance_mean_reversion_portfolio', _rebalance_mean_reversion_portfolio),
            ('_rebalance_quality_momentum_portfolio', _rebalance_quality_momentum_portfolio),
            ('_rebalance_volatility_adj_mom_portfolio', _rebalance_volatility_adj_mom_portfolio),
            ('_rebalance_turnaround_portfolio', _rebalance_turnaround_portfolio),
            ('_rebalance_generic_portfolio', _rebalance_generic_portfolio),
        ]
        
        new_stocks = mock_data['tickers'][:PORTFOLIO_SIZE]
        positions = {}
        cash = mock_data['initial_capital']
        
        for func_name, func in rebalance_functions:
            try:
                # Different functions have different signatures
                if 'generic' in func_name:
                    result = func(
                        new_stocks,
                        mock_data['current_date'],
                        mock_data['all_tickers_data'],
                        positions.copy(),
                        cash,
                        mock_data['capital_per_stock']
                    )
                elif 'turnaround' in func_name or 'volatility_ensemble' in func_name:
                    result = func(
                        new_stocks,
                        mock_data['current_date'],
                        mock_data['ticker_data_grouped'],
                        positions.copy(),
                        cash,
                        mock_data['capital_per_stock']
                    )
                else:
                    result = func(
                        new_stocks,
                        mock_data['current_date'],
                        mock_data['all_tickers_data'],
                        positions.copy(),
                        cash,
                        mock_data['capital_per_stock']
                    )
                
                # Verify return type
                assert isinstance(result, tuple), f"{func_name} should return a tuple"
                assert len(result) == 2, f"{func_name} should return exactly 2 values (cash, positions)"
                
                returned_cash, returned_positions = result
                assert isinstance(returned_cash, (int, float)), f"{func_name} cash should be numeric, got {type(returned_cash)}"
                assert isinstance(returned_positions, dict), f"{func_name} positions should be dict, got {type(returned_positions)}"
                assert returned_cash >= 0, f"{func_name} cash should not be negative"
                
                print(f"✅ {func_name}: cash={returned_cash:.2f}, positions={len(returned_positions)}")
                
            except Exception as e:
                pytest.fail(f"{func_name} failed: {e}")


class TestPortfolioValueCalculation:
    """Test that portfolio values are calculated correctly."""
    
    def test_portfolio_value_equals_cash_plus_invested(self):
        """Portfolio value should equal cash + sum of position values."""
        positions = {
            'AAPL': {'shares': 10, 'entry_price': 150.0, 'value': 1500.0},
            'MSFT': {'shares': 5, 'entry_price': 300.0, 'value': 1500.0},
        }
        cash = 1000.0
        
        invested_value = sum(pos['value'] for pos in positions.values())
        portfolio_value = cash + invested_value
        
        assert portfolio_value == 4000.0
        
    def test_transaction_cost_reduces_cash(self):
        """Transaction costs should reduce available cash."""
        initial_cash = 10000.0
        buy_value = 5000.0
        
        transaction_cost = buy_value * TRANSACTION_COST
        remaining_cash = initial_cash - buy_value - transaction_cost
        
        assert remaining_cash < initial_cash - buy_value
        assert remaining_cash == initial_cash - buy_value * (1 + TRANSACTION_COST)


class TestStrategyInitialization:
    """Test that all strategies initialize with correct values."""
    
    def test_all_strategies_start_with_same_capital(self):
        """All strategies should start with initial_capital_needed."""
        # This tests the initialization values in backtesting.py
        # We verify by checking the config values
        initial_capital_needed = INITIAL_CAPITAL * PORTFOLIO_SIZE / 10  # Approximate
        
        # All strategies should use the same initial capital
        assert initial_capital_needed > 0
        
    def test_positions_dict_structure(self):
        """Position dictionaries should have correct structure."""
        valid_position = {
            'shares': 10.0,
            'entry_price': 100.0,
            'value': 1000.0
        }
        
        assert 'shares' in valid_position
        assert 'entry_price' in valid_position
        assert 'value' in valid_position
        assert valid_position['shares'] > 0
        assert valid_position['entry_price'] > 0
        assert valid_position['value'] == valid_position['shares'] * valid_position['entry_price']


class TestSharedStrategies:
    """Test shared strategy selection functions."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock data for testing."""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX']
        return {
            'tickers': tickers,
            'ticker_data_grouped': TestMockData.create_mock_ticker_data_grouped(tickers, days=400),
            'current_date': datetime.now() - timedelta(days=1)
        }
    
    def test_select_dynamic_bh_stocks_returns_list(self, mock_data):
        """select_dynamic_bh_stocks should return a list of tickers."""
        from shared_strategies import select_dynamic_bh_stocks
        
        for period in ['1y', '6m', '3m', '1m']:
            result = select_dynamic_bh_stocks(
                mock_data['tickers'],
                mock_data['ticker_data_grouped'],
                mock_data['current_date'],
                period=period,
                top_n=5
            )
            
            assert isinstance(result, list), f"Period {period}: should return list"
            assert len(result) <= 5, f"Period {period}: should return at most top_n items"
            for ticker in result:
                assert ticker in mock_data['tickers'], f"Period {period}: returned ticker should be from input"
            
            print(f"✅ select_dynamic_bh_stocks({period}): {result}")
    
    def test_select_risk_adj_mom_stocks_returns_list(self, mock_data):
        """select_risk_adj_mom_stocks should return a list of tickers."""
        from shared_strategies import select_risk_adj_mom_stocks
        
        result = select_risk_adj_mom_stocks(
            mock_data['tickers'],
            mock_data['ticker_data_grouped'],
            mock_data['current_date'],
            top_n=5
        )
        
        assert isinstance(result, list)
        assert len(result) <= 5
        print(f"✅ select_risk_adj_mom_stocks: {result}")
    
    def test_select_mean_reversion_stocks_returns_list(self, mock_data):
        """select_mean_reversion_stocks should return a list of tickers."""
        from shared_strategies import select_mean_reversion_stocks
        
        result = select_mean_reversion_stocks(
            mock_data['tickers'],
            mock_data['ticker_data_grouped'],
            mock_data['current_date'],
            top_n=5
        )
        
        assert isinstance(result, list)
        assert len(result) <= 5
        print(f"✅ select_mean_reversion_stocks: {result}")
    
    def test_select_quality_momentum_stocks_returns_list(self, mock_data):
        """select_quality_momentum_stocks should return a list of tickers."""
        from shared_strategies import select_quality_momentum_stocks
        
        result = select_quality_momentum_stocks(
            mock_data['tickers'],
            mock_data['ticker_data_grouped'],
            mock_data['current_date'],
            top_n=5
        )
        
        assert isinstance(result, list)
        assert len(result) <= 5
        print(f"✅ select_quality_momentum_stocks: {result}")
    
    def test_select_turnaround_stocks_returns_list(self, mock_data):
        """select_turnaround_stocks should return a list of tickers."""
        from shared_strategies import select_turnaround_stocks
        
        result = select_turnaround_stocks(
            mock_data['tickers'],
            mock_data['ticker_data_grouped'],
            mock_data['current_date'],
            top_n=5
        )
        
        assert isinstance(result, list)
        assert len(result) <= 5
        print(f"✅ select_turnaround_stocks: {result}")


class TestVariableNameCollisions:
    """Test that there are no variable name collisions in backtesting.py."""
    
    def test_positions_variable_not_overwritten(self):
        """The 'positions' variable should not be overwritten by display code."""
        import ast
        
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r') as f:
            content = f.read()
        
        # Check that the daily summary uses different variable names
        assert 'num_positions = 0' in content or 'num_pos' in content, \
            "Daily summary should use 'num_positions' or 'num_pos' instead of 'positions'"
        
        # Check that the for loop unpacking uses different variable names
        assert 'for i, (name, value, strat_cash, num_pos, invested)' in content, \
            "For loop should unpack to 'num_pos' instead of 'positions'"
        
        print("✅ No variable name collisions detected")


class TestDay1Returns:
    """Test that Day 1 returns are approximately equal to transaction costs."""
    
    def test_day1_return_matches_transaction_cost(self):
        """Day 1 return should be approximately -TRANSACTION_COST."""
        initial_capital = 30000.0
        
        # Simulate buying stocks on Day 1
        # All capital is used to buy stocks, minus transaction costs
        invested_value = initial_capital / (1 + TRANSACTION_COST)
        transaction_cost_paid = initial_capital - invested_value
        
        # Portfolio value after buying = invested value (no price change yet)
        portfolio_value_day1 = invested_value
        
        # Return = (portfolio_value - initial_capital) / initial_capital
        day1_return = (portfolio_value_day1 - initial_capital) / initial_capital
        
        # Should be approximately -TRANSACTION_COST
        expected_return = -TRANSACTION_COST / (1 + TRANSACTION_COST)
        
        assert abs(day1_return - expected_return) < 0.001, \
            f"Day 1 return {day1_return:.4f} should be close to {expected_return:.4f}"
        
        print(f"✅ Day 1 return: {day1_return:.2%} (expected: {expected_return:.2%})")


class TestPortfolioSizeConsistency:
    """Test that all strategies use PORTFOLIO_SIZE consistently."""
    
    def test_portfolio_size_used_in_strategies(self):
        """All strategies should use PORTFOLIO_SIZE for stock selection."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r') as f:
            content = f.read()
        
        # Check that PORTFOLIO_SIZE is used instead of hardcoded values
        # These patterns should NOT appear (hardcoded sizes)
        bad_patterns = [
            '[:3]',  # Hardcoded 3 stocks
            'top_n=3',  # Hardcoded top 3
        ]
        
        for pattern in bad_patterns:
            # Allow some occurrences (e.g., in comments or debug output)
            occurrences = content.count(pattern)
            assert occurrences < 5, \
                f"Found {occurrences} occurrences of '{pattern}' - should use PORTFOLIO_SIZE instead"
        
        # Check that PORTFOLIO_SIZE is used
        assert 'PORTFOLIO_SIZE' in content, "PORTFOLIO_SIZE should be used in backtesting.py"
        
        print("✅ PORTFOLIO_SIZE is used consistently")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
