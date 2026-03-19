"""
Precommit tests for 10 new rebalancing strategies.
Verifies that all strategies are properly integrated into the backtesting system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from config import (
    ENABLE_BH_1Y_VOL_ADJ_REBAL,
    ENABLE_BH_1Y_CORR_FILTER,
    ENABLE_BH_1Y_REGIME_AWARE,
    ENABLE_BH_1Y_RISK_PARITY,
    ENABLE_BH_1Y_DRIFT_THRESH,
    ENABLE_BH_1Y_MOM_QUALITY,
    ENABLE_BH_1Y_LIQUIDITY,
    ENABLE_BH_1Y_EARNINGS_AVOID,
    ENABLE_BH_1Y_MULTI_FACTOR,
    ENABLE_BH_1Y_TIME_DECAY,
)


class TestConfigFlags:
    """Test that all config flags exist and are properly set."""

    def test_all_config_flags_exist(self):
        """All 10 strategy config flags should exist."""
        flags = [
            ENABLE_BH_1Y_VOL_ADJ_REBAL,
            ENABLE_BH_1Y_CORR_FILTER,
            ENABLE_BH_1Y_REGIME_AWARE,
            ENABLE_BH_1Y_RISK_PARITY,
            ENABLE_BH_1Y_DRIFT_THRESH,
            ENABLE_BH_1Y_MOM_QUALITY,
            ENABLE_BH_1Y_LIQUIDITY,
            ENABLE_BH_1Y_EARNINGS_AVOID,
            ENABLE_BH_1Y_MULTI_FACTOR,
            ENABLE_BH_1Y_TIME_DECAY,
        ]
        assert len(flags) == 10, "Should have 10 config flags"

    def test_all_config_flags_are_boolean(self):
        """All config flags should be boolean."""
        flags = [
            ENABLE_BH_1Y_VOL_ADJ_REBAL,
            ENABLE_BH_1Y_CORR_FILTER,
            ENABLE_BH_1Y_REGIME_AWARE,
            ENABLE_BH_1Y_RISK_PARITY,
            ENABLE_BH_1Y_DRIFT_THRESH,
            ENABLE_BH_1Y_MOM_QUALITY,
            ENABLE_BH_1Y_LIQUIDITY,
            ENABLE_BH_1Y_EARNINGS_AVOID,
            ENABLE_BH_1Y_MULTI_FACTOR,
            ENABLE_BH_1Y_TIME_DECAY,
        ]
        for flag in flags:
            assert isinstance(flag, bool), f"Config flag should be boolean, got {type(flag)}"

    def test_all_config_flags_enabled(self):
        """All config flags should be True for testing."""
        flags = [
            ENABLE_BH_1Y_VOL_ADJ_REBAL,
            ENABLE_BH_1Y_CORR_FILTER,
            ENABLE_BH_1Y_REGIME_AWARE,
            ENABLE_BH_1Y_RISK_PARITY,
            ENABLE_BH_1Y_DRIFT_THRESH,
            ENABLE_BH_1Y_MOM_QUALITY,
            ENABLE_BH_1Y_LIQUIDITY,
            ENABLE_BH_1Y_EARNINGS_AVOID,
            ENABLE_BH_1Y_MULTI_FACTOR,
            ENABLE_BH_1Y_TIME_DECAY,
        ]
        for flag in flags:
            assert flag is True, f"Config flag should be True, got {flag}"


class TestStrategyFunctions:
    """Test that all strategy functions can be imported."""

    def test_all_strategy_functions_importable(self):
        """All 10 strategy functions should be importable."""
        from enhanced_static_bh_strategies import (
            get_vol_adjusted_rebalance_days,
            filter_by_correlation,
            detect_market_regime,
            get_risk_parity_weights,
            calculate_portfolio_drift,
            score_momentum_quality,
            get_liquidity_weights,
            is_near_earnings,
            get_multi_factor_score,
            get_time_decay_exit_pct,
        )
        funcs = [
            get_vol_adjusted_rebalance_days,
            filter_by_correlation,
            detect_market_regime,
            get_risk_parity_weights,
            calculate_portfolio_drift,
            score_momentum_quality,
            get_liquidity_weights,
            is_near_earnings,
            get_multi_factor_score,
            get_time_decay_exit_pct,
        ]
        assert len(funcs) == 10, "Should have 10 strategy functions"


class TestBacktestingIntegration:
    """Test that backtesting.py has all required components."""

    def test_backtesting_compiles(self):
        """backtesting.py should compile without errors."""
        import py_compile
        import tempfile
        
        # Read the backtesting.py file
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        # Compile check
        try:
            py_compile.compile(backtesting_path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"backtesting.py failed to compile: {e}")

    def test_all_strategies_in_daily_summary(self):
        """All 10 strategies should be in daily summary strategy_values list."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_strategies = [
            '"BH 1Y Vol Adj"',
            '"BH 1Y Corr Filt"',
            '"BH 1Y Regime"',
            '"BH 1Y Risk Par"',
            '"BH 1Y Drift"',
            '"BH 1Y Mom Qual"',
            '"BH 1Y Liquid"',
            '"BH 1Y Earn Avd"',
            '"BH 1Y MultiFact"',
            '"BH 1Y TimeDec"',
        ]
        
        for strategy in expected_strategies:
            assert strategy in content, f"Strategy {strategy} should be in daily summary"

    def test_all_strategies_in_cash_positions_display(self):
        """All 10 strategies should have cash/positions display logic."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_checks = [
            'name == "BH 1Y Vol Adj" and ENABLE_BH_1Y_VOL_ADJ_REBAL',
            'name == "BH 1Y Corr Filt" and ENABLE_BH_1Y_CORR_FILTER',
            'name == "BH 1Y Regime" and ENABLE_BH_1Y_REGIME_AWARE',
            'name == "BH 1Y Risk Par" and ENABLE_BH_1Y_RISK_PARITY',
            'name == "BH 1Y Drift" and ENABLE_BH_1Y_DRIFT_THRESH',
            'name == "BH 1Y Mom Qual" and ENABLE_BH_1Y_MOM_QUALITY',
            'name == "BH 1Y Liquid" and ENABLE_BH_1Y_LIQUIDITY',
            'name == "BH 1Y Earn Avd" and ENABLE_BH_1Y_EARNINGS_AVOID',
            'name == "BH 1Y MultiFact" and ENABLE_BH_1Y_MULTI_FACTOR',
            'name == "BH 1Y TimeDec" and ENABLE_BH_1Y_TIME_DECAY',
        ]
        
        for check in expected_checks:
            assert check in content, f"Check {check} should be in cash/positions display"

    def test_all_strategies_in_strategy_history_pairs(self):
        """All 10 strategies should be in strategy_history_pairs for std dev."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_pairs = [
            '("BH 1Y Vol Adj", bh_1y_vol_adj_rebal_portfolio_history)',
            '("BH 1Y Corr Filt", bh_1y_corr_filter_portfolio_history)',
            '("BH 1Y Regime", bh_1y_regime_aware_portfolio_history)',
            '("BH 1Y Risk Par", bh_1y_risk_parity_portfolio_history)',
            '("BH 1Y Drift", bh_1y_drift_thresh_portfolio_history)',
            '("BH 1Y Mom Qual", bh_1y_mom_quality_portfolio_history)',
            '("BH 1Y Liquid", bh_1y_liquidity_portfolio_history)',
            '("BH 1Y Earn Avd", bh_1y_earnings_avoid_portfolio_history)',
            '("BH 1Y MultiFact", bh_1y_multi_factor_portfolio_history)',
            '("BH 1Y TimeDec", bh_1y_time_decay_portfolio_history)',
        ]
        
        for pair in expected_pairs:
            assert pair in content, f"Pair {pair} should be in strategy_history_pairs"

    def test_all_strategies_in_final_results_dict(self):
        """All 10 strategies should be in final results dict."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_results = [
            "'bh_1y_vol_adj_rebal':",
            "'bh_1y_corr_filter':",
            "'bh_1y_regime_aware':",
            "'bh_1y_risk_parity':",
            "'bh_1y_drift_thresh':",
            "'bh_1y_mom_quality':",
            "'bh_1y_liquidity':",
            "'bh_1y_earnings_avoid':",
            "'bh_1y_multi_factor':",
            "'bh_1y_time_decay':",
        ]
        
        for result in expected_results:
            assert result in content, f"Result {result} should be in final results dict"

    def test_all_strategies_have_initialization(self):
        """All 10 strategies should have initialization variables."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_inits = [
            'bh_1y_vol_adj_rebal_positions = {}',
            'bh_1y_corr_filter_positions = {}',
            'bh_1y_regime_aware_positions = {}',
            'bh_1y_risk_parity_positions = {}',
            'bh_1y_drift_thresh_positions = {}',
            'bh_1y_mom_quality_positions = {}',
            'bh_1y_liquidity_positions = {}',
            'bh_1y_earnings_avoid_positions = {}',
            'bh_1y_multi_factor_positions = {}',
            'bh_1y_time_decay_positions = {}',
        ]
        
        for init in expected_inits:
            assert init in content, f"Initialization {init} should exist"

    def test_all_strategies_have_daily_logic(self):
        """All 10 strategies should have daily logic."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_logic = [
            'if ENABLE_BH_1Y_VOL_ADJ_REBAL:',
            'if ENABLE_BH_1Y_CORR_FILTER:',
            'if ENABLE_BH_1Y_REGIME_AWARE:',
            'if ENABLE_BH_1Y_RISK_PARITY:',
            'if ENABLE_BH_1Y_DRIFT_THRESH:',
            'if ENABLE_BH_1Y_MOM_QUALITY:',
            'if ENABLE_BH_1Y_LIQUIDITY:',
            'if ENABLE_BH_1Y_EARNINGS_AVOID:',
            'if ENABLE_BH_1Y_MULTI_FACTOR:',
            'if ENABLE_BH_1Y_TIME_DECAY:',
        ]
        
        for logic in expected_logic:
            assert logic in content, f"Daily logic {logic} should exist"

    def test_all_strategies_have_value_updates(self):
        """All 10 strategies should have portfolio value updates."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        expected_updates = [
            'bh_1y_vol_adj_rebal_portfolio_value = _update_smart_strategy_value',
            'bh_1y_corr_filter_portfolio_value = _update_smart_strategy_value',
            'bh_1y_regime_aware_portfolio_value = _update_smart_strategy_value',
            'bh_1y_risk_parity_portfolio_value = _update_smart_strategy_value',
            'bh_1y_drift_thresh_portfolio_value = _update_smart_strategy_value',
            'bh_1y_mom_quality_portfolio_value = _update_smart_strategy_value',
            'bh_1y_liquidity_portfolio_value = _update_smart_strategy_value',
            'bh_1y_earnings_avoid_portfolio_value = _update_smart_strategy_value',
            'bh_1y_multi_factor_portfolio_value = _update_smart_strategy_value',
            'bh_1y_time_decay_portfolio_value = _update_smart_strategy_value',
        ]
        
        for update in expected_updates:
            assert update in content, f"Value update {update} should exist"


class TestStrategyNaming:
    """Test that strategy names are consistent across the codebase."""

    def test_strategy_names_match_across_files(self):
        """Strategy names should be consistent in config and backtesting."""
        backtesting_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'backtesting.py')
        
        with open(backtesting_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that display names match enable flags
        name_mapping = [
            ('BH_1Y_VOL_ADJ_REBAL', 'BH 1Y Vol Adj'),
            ('BH_1Y_CORR_FILTER', 'BH 1Y Corr Filt'),
            ('BH_1Y_REGIME_AWARE', 'BH 1Y Regime'),
            ('BH_1Y_RISK_PARITY', 'BH 1Y Risk Par'),
            ('BH_1Y_DRIFT_THRESH', 'BH 1Y Drift'),
            ('BH_1Y_MOM_QUALITY', 'BH 1Y Mom Qual'),
            ('BH_1Y_LIQUIDITY', 'BH 1Y Liquid'),
            ('BH_1Y_EARNINGS_AVOID', 'BH 1Y Earn Avd'),
            ('BH_1Y_MULTI_FACTOR', 'BH 1Y MultiFact'),
            ('BH_1Y_TIME_DECAY', 'BH 1Y TimeDec'),
        ]
        
        for flag, name in name_mapping:
            assert f'ENABLE_{flag}' in content, f"Enable flag ENABLE_{flag} should exist"
            assert f'"{name}"' in content, f"Display name {name} should exist"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
