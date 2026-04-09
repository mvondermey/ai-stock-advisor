"""
Meta-Strategy Selector Module

Implements 10 different algorithms for selecting/combining strategies based on their performance.
These are "meta-strategies" that allocate capital across existing sub-strategies.

Proposal 1: Weighted Composite Score - Combine multiple metrics with weights
Proposal 2: Tiered Selection - Filter by consistency, rank by Sharpe
Proposal 3: Ensemble Allocation - Allocate proportionally to consistency
Proposal 4: Regime-Based Selection - Select based on market volatility regime
Proposal 5: Recency-Weighted Consistency - Weight recent performance higher
Proposal 6: Consistency-Return Efficiency - Efficiency ratio ranking
Proposal 7: Minimum Variance Ensemble - Minimize portfolio variance
Proposal 8: Bayesian Strategy Selector - Update probabilities based on wins
Proposal 9: Adaptive Convex Combination - Dynamic allocation based on regime
Proposal 10: Best of Best Consensus - Require agreement across multiple metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from strategy_universes import (
    META_ADAPTIVE_REBALANCING_STRATEGIES,
    META_ADAPTIVE_CONVEX_BALANCED,
    META_ADAPTIVE_CONVEX_DEFENSIVE,
    META_ADAPTIVE_CONVEX_MOMENTUM,
    META_AI_STRATEGIES,
    META_AI_REGIME_STRATEGIES,
    META_BB_STRATEGIES,
    META_DYNAMIC_BH_STRATEGIES,
    META_ENHANCED_VOLATILITY_STRATEGIES,
    META_ENSEMBLE_STRATEGIES,
    META_MONTHLY_BH_STRATEGIES,
    META_MOMENTUM_VOLATILITY_HYBRID_STRATEGIES,
    META_RATIO_STRATEGIES,
    META_REGIME_HIGH_VOL_CANDIDATES,
    META_REGIME_LOW_VOL_CANDIDATES,
    META_REGIME_MEDIUM_VOL_CANDIDATES,
    META_RISK_ADJ_MOM_MONTHLY_STRATEGIES,
    META_RISK_ADJ_MOM_STRATEGIES,
    META_RISK_ADJ_MOM_VARIANT_STRATEGIES,
    META_STATIC_BH_STRATEGIES,
    META_STRATEGY_SOURCES,
    get_enabled_strategy_aliases,
)


def get_all_enabled_strategies():
    """
    Dynamically generate list of all enabled strategies from config.
    Returns mapping of strategy key -> display name.
    """
    return get_enabled_strategy_aliases(META_STRATEGY_SOURCES)


# Get dynamic list of enabled strategies
META_SUB_STRATEGIES = get_all_enabled_strategies()


class MetaStrategyManager:
    """
    Manages 10 meta-strategy selection algorithms.
    Each tracks a virtual portfolio that follows selected sub-strategy(ies).
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.day_count = 0
        self.warmup_days = 2  # Days before meta-strategies start

        # Track daily values for each sub-strategy
        self.strategy_histories: Dict[str, List[float]] = defaultdict(list)
        self.strategy_returns: Dict[str, List[float]] = defaultdict(list)

        # Top-5 counts for consistency
        self.top_5_counts: Dict[str, int] = defaultdict(int)

        # Bayesian priors
        self.bayesian_priors: Dict[str, float] = {}

        # Current allocations for each meta-strategy
        self.allocations: Dict[str, Dict[str, float]] = {
            'meta_weighted_composite': {},
            'meta_tiered_selection': {},
            'meta_ensemble_alloc': {},
            'meta_regime_based': {},
            'meta_recency_weighted': {},
            'meta_efficiency_ratio': {},
            'meta_min_variance': {},
            'meta_bayesian': {},
            'meta_adaptive_convex': {},
            'meta_consensus': {},
        }

    def record_daily_values(self, strategy_values: Dict[str, float]):
        """Record daily portfolio values for all sub-strategies."""
        self.day_count += 1

        for name in META_SUB_STRATEGIES:
            if name in strategy_values and strategy_values[name] is not None:
                value = strategy_values[name]
                self.strategy_histories[name].append(value)

                # Calculate daily return
                if len(self.strategy_histories[name]) > 1:
                    prev = self.strategy_histories[name][-2]
                    ret = (value - prev) / prev if prev > 0 else 0.0
                else:
                    ret = 0.0
                self.strategy_returns[name].append(ret)

        # Update top-5 counts
        self._update_top_5_counts()

    def _update_top_5_counts(self):
        """Update top-5 consistency counts."""
        if self.day_count < 2:
            return

        today_values = []
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_histories and len(self.strategy_histories[name]) > 0:
                today_values.append((name, self.strategy_histories[name][-1]))

        if len(today_values) >= 5:
            today_values.sort(key=lambda x: x[1], reverse=True)
            for name, _ in today_values[:5]:
                self.top_5_counts[name] += 1

    def _get_consistency(self) -> Dict[str, float]:
        """Get consistency scores (% of days in top 5)."""
        total_days = max(1, self.day_count - 1)
        return {name: self.top_5_counts[name] / total_days for name in META_SUB_STRATEGIES}

    def _get_recent_consistency(self, days: int = 15) -> Dict[str, float]:
        """Get consistency over recent N days."""
        if self.day_count < days + 1:
            return self._get_consistency()

        recent_top_5 = defaultdict(int)
        for offset in range(days):
            idx = -(offset + 1)
            day_vals = []
            for name in META_SUB_STRATEGIES:
                if name in self.strategy_histories and len(self.strategy_histories[name]) >= abs(idx):
                    day_vals.append((name, self.strategy_histories[name][idx]))

            if len(day_vals) >= 5:
                day_vals.sort(key=lambda x: x[1], reverse=True)
                for n, _ in day_vals[:5]:
                    recent_top_5[n] += 1

        return {name: recent_top_5[name] / days for name in META_SUB_STRATEGIES}

    def _get_total_returns(self) -> Dict[str, float]:
        """Get total returns for each strategy."""
        returns = {}
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_histories and len(self.strategy_histories[name]) > 0:
                current = self.strategy_histories[name][-1]
                returns[name] = (current - self.initial_capital) / self.initial_capital
            else:
                returns[name] = 0.0
        return returns

    def _get_sharpe_ratios(self, lookback: int = 20) -> Dict[str, float]:
        """Get Sharpe ratios."""
        sharpes = {}
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_returns and len(self.strategy_returns[name]) >= lookback:
                rets = self.strategy_returns[name][-lookback:]
                mean_r = np.mean(rets)
                std_r = np.std(rets)
                sharpes[name] = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            else:
                sharpes[name] = 0.0
        return sharpes

    def _get_volatilities(self, lookback: int = 20) -> Dict[str, float]:
        """Get annualized volatilities."""
        vols = {}
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_returns and len(self.strategy_returns[name]) >= lookback:
                rets = self.strategy_returns[name][-lookback:]
                vols[name] = np.std(rets) * np.sqrt(252)
            else:
                vols[name] = 0.0
        return vols

    def _get_max_drawdowns(self) -> Dict[str, float]:
        """Get maximum drawdowns."""
        drawdowns = {}
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_histories and len(self.strategy_histories[name]) > 1:
                vals = self.strategy_histories[name]
                peak = vals[0]
                max_dd = 0.0
                for v in vals:
                    if v > peak:
                        peak = v
                    dd = (peak - v) / peak if peak > 0 else 0.0
                    max_dd = max(max_dd, dd)
                drawdowns[name] = max_dd
            else:
                drawdowns[name] = 0.0
        return drawdowns

    def _get_market_volatility(self) -> float:
        """Estimate market volatility from average strategy volatility."""
        vols = self._get_volatilities()
        valid_vols = [v for v in vols.values() if v > 0]
        return np.mean(valid_vols) if valid_vols else 0.15

    def _rank_dict(self, d: Dict[str, float], ascending: bool = False) -> Dict[str, int]:
        """Rank dictionary values (1 = best)."""
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=not ascending)
        return {name: rank + 1 for rank, (name, _) in enumerate(sorted_items)}

    # =========================================================================
    # PROPOSAL 1: Weighted Composite Score
    # =========================================================================
    def select_weighted_composite(self) -> Tuple[str, Dict[str, float]]:
        """
        Combine multiple metrics with weights:
        - 30% consistency rank
        - 25% total return rank
        - 20% Sharpe ratio rank
        - 15% max drawdown rank (inverted)
        - 10% win rate (positive days)
        """
        consistency = self._get_consistency()
        returns = self._get_total_returns()
        sharpes = self._get_sharpe_ratios()
        drawdowns = self._get_max_drawdowns()

        # Calculate win rates
        win_rates = {}
        for name in META_SUB_STRATEGIES:
            if name in self.strategy_returns and len(self.strategy_returns[name]) > 0:
                rets = self.strategy_returns[name]
                win_rates[name] = sum(1 for r in rets if r > 0) / len(rets)
            else:
                win_rates[name] = 0.0

        # Rank each metric (lower rank = better)
        cons_rank = self._rank_dict(consistency)
        ret_rank = self._rank_dict(returns)
        sharpe_rank = self._rank_dict(sharpes)
        dd_rank = self._rank_dict(drawdowns, ascending=True)  # Lower DD is better
        wr_rank = self._rank_dict(win_rates)

        # Composite score (lower = better)
        scores = {}
        for name in META_SUB_STRATEGIES:
            scores[name] = (
                0.30 * cons_rank.get(name, 100) +
                0.25 * ret_rank.get(name, 100) +
                0.20 * sharpe_rank.get(name, 100) +
                0.15 * dd_rank.get(name, 100) +
                0.10 * wr_rank.get(name, 100)
            )

        # Select best (lowest score)
        best = min(scores, key=scores.get)
        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 2: Tiered Selection
    # =========================================================================
    def select_tiered(self) -> Tuple[str, Dict[str, float]]:
        """
        1. Filter: Only strategies with >50% consistency
        2. Rank filtered by Sharpe ratio
        3. Select top 1
        """
        consistency = self._get_consistency()
        sharpes = self._get_sharpe_ratios()

        # Filter by consistency > 50%
        filtered = [name for name in META_SUB_STRATEGIES if consistency.get(name, 0) > 0.5]

        # If none pass, use top 3 by consistency
        if not filtered:
            sorted_cons = sorted(consistency.items(), key=lambda x: x[1], reverse=True)
            filtered = [name for name, _ in sorted_cons[:3]]

        # Rank by Sharpe
        filtered_sharpes = {name: sharpes.get(name, 0) for name in filtered}
        best = max(filtered_sharpes, key=filtered_sharpes.get)

        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 3: Ensemble Allocation
    # =========================================================================
    def select_ensemble_allocation(self) -> Tuple[str, Dict[str, float]]:
        """
        Allocate capital proportionally to consistency.
        Top 4 strategies get allocation based on their consistency %.
        """
        consistency = self._get_consistency()

        # Get top 4 by consistency
        sorted_cons = sorted(consistency.items(), key=lambda x: x[1], reverse=True)
        top_4 = sorted_cons[:4]

        # Normalize to sum to 1.0
        total_cons = sum(c for _, c in top_4)
        if total_cons > 0:
            allocation = {name: cons / total_cons for name, cons in top_4}
        else:
            allocation = {name: 0.25 for name, _ in top_4}

        # Return highest allocation as "selected"
        best = max(allocation, key=allocation.get)
        return best, allocation

    # =========================================================================
    # PROPOSAL 4: Regime-Based Selection
    # =========================================================================
    def select_regime_based(self) -> Tuple[str, Dict[str, float]]:
        """
        Select based on market volatility regime:
        - High vol (>25%): Defensive strategies (mean_reversion, risk_adj_mom)
        - Low vol (<15%): Momentum strategies (static_bh_1y, dynamic_bh_1y)
        - Medium: Balanced (trend_atr, dual_momentum)
        """
        market_vol = self._get_market_volatility()
        returns = self._get_total_returns()

        if market_vol > 0.25:  # High volatility
            candidates = list(META_REGIME_HIGH_VOL_CANDIDATES)
        elif market_vol < 0.15:  # Low volatility
            candidates = list(META_REGIME_LOW_VOL_CANDIDATES)
        else:  # Medium volatility
            candidates = list(META_REGIME_MEDIUM_VOL_CANDIDATES)

        # Filter to available strategies
        available = [c for c in candidates if c in returns and returns[c] != 0]
        if not available:
            available = list(returns.keys())[:3]

        # Select best by return
        best = max(available, key=lambda x: returns.get(x, 0))
        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 5: Recency-Weighted Consistency
    # =========================================================================
    def select_recency_weighted(self) -> Tuple[str, Dict[str, float]]:
        """
        Give more weight to recent performance (last 15 days):
        composite = 0.6 * recent_consistency + 0.4 * overall_consistency
        """
        overall = self._get_consistency()
        recent = self._get_recent_consistency(15)

        composite = {}
        for name in META_SUB_STRATEGIES:
            composite[name] = 0.6 * recent.get(name, 0) + 0.4 * overall.get(name, 0)

        best = max(composite, key=composite.get)
        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 6: Consistency-Return Efficiency
    # =========================================================================
    def select_efficiency_ratio(self) -> Tuple[str, Dict[str, float]]:
        """
        Efficiency = annual_return / (1 + volatility) * consistency
        Rewards consistent high returns with low volatility.
        """
        returns = self._get_total_returns()
        vols = self._get_volatilities()
        consistency = self._get_consistency()

        efficiency = {}
        for name in META_SUB_STRATEGIES:
            ret = returns.get(name, 0)
            vol = vols.get(name, 0.01)
            cons = consistency.get(name, 0)
            efficiency[name] = (ret / (1 + vol)) * (1 + cons)

        best = max(efficiency, key=efficiency.get)
        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 7: Minimum Variance Ensemble
    # =========================================================================
    def select_min_variance(self) -> Tuple[str, Dict[str, float]]:
        """
        Select strategies that minimize portfolio variance.
        Use correlation to find diversifying strategies.
        """
        if self.day_count < 30:
            return self.select_tiered()

        # Build return matrix
        valid_strategies = []
        return_matrix = []

        for name in META_SUB_STRATEGIES:
            if name in self.strategy_returns and len(self.strategy_returns[name]) >= 20:
                valid_strategies.append(name)
                return_matrix.append(self.strategy_returns[name][-20:])

        if len(valid_strategies) < 3:
            return self.select_tiered()

        # Calculate correlation matrix
        return_matrix = np.array(return_matrix)
        corr_matrix = np.corrcoef(return_matrix)

        # Find strategy with lowest average correlation (most diversifying)
        avg_corr = np.mean(np.abs(corr_matrix), axis=1)

        # Also consider volatility
        vols = self._get_volatilities()

        # Score = low correlation + low volatility
        scores = {}
        for i, name in enumerate(valid_strategies):
            vol = vols.get(name, 0.5)
            scores[name] = avg_corr[i] + vol  # Lower is better

        # Select top 3 with lowest scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        top_3 = [name for name, _ in sorted_scores[:3]]

        allocation = {name: 1.0 / 3 for name in top_3}
        best = top_3[0]
        return best, allocation

    # =========================================================================
    # PROPOSAL 8: Bayesian Strategy Selector
    # =========================================================================
    def select_bayesian(self) -> Tuple[str, Dict[str, float]]:
        """
        Track win probability over time:
        - Start with equal priors
        - Update: P(strategy | was in top 5 yesterday)
        """
        # Initialize priors if needed
        if not self.bayesian_priors:
            n = len(META_SUB_STRATEGIES)
            self.bayesian_priors = {name: 1.0 / n for name in META_SUB_STRATEGIES}

        # Update based on yesterday's top 5
        if self.day_count > 1:
            yesterday_vals = []
            for name in META_SUB_STRATEGIES:
                if name in self.strategy_histories and len(self.strategy_histories[name]) >= 2:
                    yesterday_vals.append((name, self.strategy_histories[name][-2]))

            if len(yesterday_vals) >= 5:
                yesterday_vals.sort(key=lambda x: x[1], reverse=True)
                top_5_names = {name for name, _ in yesterday_vals[:5]}

                # Bayesian update: increase probability for top 5, decrease for others
                alpha = 0.1  # Learning rate
                for name in META_SUB_STRATEGIES:
                    if name in top_5_names:
                        self.bayesian_priors[name] *= (1 + alpha)
                    else:
                        self.bayesian_priors[name] *= (1 - alpha * 0.5)

                # Normalize
                total = sum(self.bayesian_priors.values())
                if total > 0:
                    self.bayesian_priors = {k: v / total for k, v in self.bayesian_priors.items()}

        # Select highest posterior probability
        best = max(self.bayesian_priors, key=self.bayesian_priors.get)
        return best, {best: 1.0}

    # =========================================================================
    # PROPOSAL 9: Adaptive Convex Combination
    # =========================================================================
    def select_adaptive_convex(self) -> Tuple[str, Dict[str, float]]:
        """
        Dynamic allocation based on market regime:
        - High volatility: Defensive tilt (50% defensive, 30% balanced, 20% momentum)
        - Low volatility: Momentum tilt (50% momentum, 30% balanced, 20% defensive)
        - Medium: Balanced (equal weight)
        """
        market_vol = self._get_market_volatility()
        returns = self._get_total_returns()

        # Categorize strategies
        defensive = list(META_ADAPTIVE_CONVEX_DEFENSIVE)
        momentum = list(META_ADAPTIVE_CONVEX_MOMENTUM)
        balanced = list(META_ADAPTIVE_CONVEX_BALANCED)

        # Get best from each category
        def best_in_category(cat):
            available = [c for c in cat if c in returns]
            if available:
                return max(available, key=lambda x: returns.get(x, 0))
            return None

        best_def = best_in_category(defensive)
        best_mom = best_in_category(momentum)
        best_bal = best_in_category(balanced)

        # Allocate based on regime
        allocation = {}
        if market_vol > 0.25:  # High vol - defensive
            if best_def: allocation[best_def] = 0.50
            if best_bal: allocation[best_bal] = 0.30
            if best_mom: allocation[best_mom] = 0.20
        elif market_vol < 0.15:  # Low vol - momentum
            if best_mom: allocation[best_mom] = 0.50
            if best_bal: allocation[best_bal] = 0.30
            if best_def: allocation[best_def] = 0.20
        else:  # Medium - balanced
            if best_def: allocation[best_def] = 0.33
            if best_bal: allocation[best_bal] = 0.34
            if best_mom: allocation[best_mom] = 0.33

        # Normalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}

        best = max(allocation, key=allocation.get) if allocation else 'static_bh_1y'
        return best, allocation

    # =========================================================================
    # PROPOSAL 10: Best of Best Consensus
    # =========================================================================
    def select_consensus(self) -> Tuple[str, Dict[str, float]]:
        """
        Require agreement across multiple metrics:
        1. Must be in Top 3 for consistency
        2. Must be in Top 5 for total return
        3. Must have max drawdown < 15%

        Only strategies meeting ALL criteria are selected.
        """
        consistency = self._get_consistency()
        returns = self._get_total_returns()
        drawdowns = self._get_max_drawdowns()

        # Get top 3 by consistency
        sorted_cons = sorted(consistency.items(), key=lambda x: x[1], reverse=True)
        top_3_cons = {name for name, _ in sorted_cons[:3]}

        # Get top 5 by return
        sorted_ret = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_5_ret = {name for name, _ in sorted_ret[:5]}

        # Get strategies with DD < 15%
        low_dd = {name for name, dd in drawdowns.items() if dd < 0.15}

        # Find consensus (intersection)
        consensus = top_3_cons & top_5_ret & low_dd

        # If no consensus, relax criteria
        if not consensus:
            consensus = top_3_cons & top_5_ret
        if not consensus:
            consensus = top_3_cons
        if not consensus:
            consensus = {sorted_cons[0][0]}

        # Select best by return from consensus
        consensus_returns = {name: returns.get(name, 0) for name in consensus}
        best = max(consensus_returns, key=consensus_returns.get)

        return best, {best: 1.0}

    # =========================================================================
    # Main selection method
    # =========================================================================
    def get_all_selections(self) -> Dict[str, Tuple[str, Dict[str, float]]]:
        """Get selections from all 10 meta-strategies."""
        if self.day_count < self.warmup_days:
            # During warmup, use static_bh_1y as default
            default = ('static_bh_1y', {'static_bh_1y': 1.0})
            return {
                'meta_weighted_composite': default,
                'meta_tiered_selection': default,
                'meta_ensemble_alloc': default,
                'meta_regime_based': default,
                'meta_recency_weighted': default,
                'meta_efficiency_ratio': default,
                'meta_min_variance': default,
                'meta_bayesian': default,
                'meta_adaptive_convex': default,
                'meta_consensus': default,
            }

        return {
            'meta_weighted_composite': self.select_weighted_composite(),
            'meta_tiered_selection': self.select_tiered(),
            'meta_ensemble_alloc': self.select_ensemble_allocation(),
            'meta_regime_based': self.select_regime_based(),
            'meta_recency_weighted': self.select_recency_weighted(),
            'meta_efficiency_ratio': self.select_efficiency_ratio(),
            'meta_min_variance': self.select_min_variance(),
            'meta_bayesian': self.select_bayesian(),
            'meta_adaptive_convex': self.select_adaptive_convex(),
            'meta_consensus': self.select_consensus(),
        }


def calculate_meta_portfolio_value(
    allocation: Dict[str, float],
    strategy_values: Dict[str, float],
    initial_capital: float
) -> float:
    """
    Calculate meta-portfolio value based on allocation and sub-strategy values.

    Args:
        allocation: {strategy_name: weight} (weights sum to 1.0)
        strategy_values: {strategy_name: current_value}
        initial_capital: Starting capital

    Returns:
        Current meta-portfolio value
    """
    if not allocation:
        return initial_capital

    total_value = 0.0
    for name, weight in allocation.items():
        if name in strategy_values and strategy_values[name] is not None:
            # Calculate return of sub-strategy
            sub_value = strategy_values[name]
            sub_return = (sub_value - initial_capital) / initial_capital if initial_capital > 0 else 0
            # Apply weighted return
            total_value += weight * (initial_capital * (1 + sub_return))
        else:
            # If strategy not available, assume no change
            total_value += weight * initial_capital

    return total_value


def select_meta_strategy_stocks(
    selected_strategy: str,
    all_tickers: list,
    ticker_data_grouped: dict,
    current_date,
    top_n: int = 10,
    ai_elite_models: dict = None,
    price_history_cache=None,
) -> list:
    """
    Select actual tickers based on the meta-strategy's chosen sub-strategy.

    Args:
        selected_strategy: Name of the sub-strategy chosen by meta-strategy
        all_tickers: List of ticker symbols to select from
        ticker_data_grouped: Dict of ticker -> DataFrame
        current_date: Current analysis date
        top_n: Number of stocks to select
        ai_elite_models: AI Elite models (if needed)

    Returns:
        List of selected ticker symbols
    """
    if not selected_strategy:
        return []

    try:
        # Import shared function once
        from shared_strategies import select_top_performers

        # === STATIC BH STRATEGIES ===
        if selected_strategy in META_STATIC_BH_STRATEGIES:
            lookback = 365 if '1y' in selected_strategy else 90
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=lookback,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'static_bh_6m':
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=180,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'static_bh_1m':
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=30,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        # === MONTHLY REBALANCE VARIANTS ===
        elif selected_strategy in META_MONTHLY_BH_STRATEGIES:
            lookback_map = {'1y': 365, '6m': 180, '3m': 90, '1m': 30}
            lookback = lookback_map[selected_strategy.split('_')[1]]
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=lookback,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        # === DYNAMIC BH STRATEGIES ===
        elif selected_strategy in META_DYNAMIC_BH_STRATEGIES:
            lookback = 365 if '1y' in selected_strategy else 90
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=lookback,
                top_n=top_n,
                apply_performance_filter=True,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'dynamic_bh_6m':
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=180,
                top_n=top_n,
                apply_performance_filter=True,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'dynamic_bh_1m':
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=30,
                top_n=top_n,
                apply_performance_filter=True,
                price_history_cache=price_history_cache,
            )

        # === DYNAMIC BH VARIANTS ===
        elif selected_strategy == 'dynamic_bh_1y_vol_filter':
            from shared_strategies import select_top_performers_vol_filtered
            return select_top_performers_vol_filtered(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=365,
                top_n=top_n,
                max_volatility=0.4,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'dynamic_bh_1y_trailing_stop':
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=365,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        # === RISK-ADJ MOMENTUM STRATEGIES ===
        elif selected_strategy in META_RISK_ADJ_MOM_STRATEGIES:
            from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
            return select_risk_adj_mom_3m_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === RISK-ADJ MOM VARIANTS ===
        elif selected_strategy in META_RISK_ADJ_MOM_MONTHLY_STRATEGIES:
            from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
            return select_risk_adj_mom_3m_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy in META_RISK_ADJ_MOM_VARIANT_STRATEGIES:
            from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
            return select_risk_adj_mom_3m_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'risk_adj_mom_1m_vol_sweet':
            from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
            return select_risk_adj_mom_1m_vol_sweet_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'vol_sweet_mom':
            from risk_adj_mom_1m_vol_sweet_strategy import select_risk_adj_mom_1m_vol_sweet_stocks
            return select_risk_adj_mom_1m_vol_sweet_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === OTHER CORE STRATEGIES ===
        elif selected_strategy == 'mean_reversion':
            from mean_reversion_strategy import select_mean_reversion_stocks
            return select_mean_reversion_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'quality_momentum':
            from quality_momentum_strategy import select_quality_momentum_stocks
            return select_quality_momentum_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'momentum_ai_hybrid':
            from shared_strategies import select_momentum_ai_hybrid_stocks
            return select_momentum_ai_hybrid_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'volatility_adj_mom':
            from shared_strategies import select_volatility_adj_mom_stocks
            return select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        # === ENHANCED VOLATILITY STRATEGIES ===
        elif selected_strategy in META_ENHANCED_VOLATILITY_STRATEGIES:
            from enhanced_volatility_trader import select_enhanced_volatility_stocks
            return select_enhanced_volatility_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === TREND AND MOMENTUM STRATEGIES ===
        elif selected_strategy == 'trend_atr':
            from new_strategies import select_trend_following_atr_stocks
            result = select_trend_following_atr_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )
            if isinstance(result, tuple):
                return result[0]
            return result

        elif selected_strategy == 'dual_momentum':
            from new_strategies import select_dual_momentum_stocks
            return select_dual_momentum_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )[0]

        elif selected_strategy == 'momentum_acceleration':
            from momentum_acceleration_strategy import select_momentum_acceleration_stocks
            return select_momentum_acceleration_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'concentrated_3m':
            from concentrated_3m_strategy import select_concentrated_3m_stocks
            return select_concentrated_3m_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'price_acceleration':
            from price_acceleration_strategy import select_price_acceleration_stocks
            return select_price_acceleration_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'turnaround':
            from turnaround_strategy import select_turnaround_stocks
            return select_turnaround_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === ELITE STRATEGIES ===
        elif selected_strategy == 'elite_hybrid':
            from elite_hybrid_strategy import select_elite_hybrid_stocks
            return select_elite_hybrid_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        elif selected_strategy == 'elite_risk':
            from elite_risk_strategy import select_elite_risk_stocks
            return select_elite_risk_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === MOMENTUM-VOLATILITY HYBRIDS ===
        elif selected_strategy in META_MOMENTUM_VOLATILITY_HYBRID_STRATEGIES:
            from shared_strategies import (
                select_momentum_volatility_hybrid_1y3m_stocks,
                select_momentum_volatility_hybrid_1y_stocks,
                select_momentum_volatility_hybrid_6m_stocks,
                select_momentum_volatility_hybrid_stocks,
            )

            strategy_selectors = {
                'momentum_volatility_hybrid': select_momentum_volatility_hybrid_stocks,
                'momentum_volatility_hybrid_6m': select_momentum_volatility_hybrid_6m_stocks,
                'momentum_volatility_hybrid_1y': select_momentum_volatility_hybrid_1y_stocks,
                'momentum_volatility_hybrid_1y3m': select_momentum_volatility_hybrid_1y3m_stocks,
            }
            return strategy_selectors[selected_strategy](
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === RATIO STRATEGIES ===
        elif selected_strategy in META_RATIO_STRATEGIES:
            from ratio_strategies import select_ratio_stocks
            return select_ratio_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        # === ENSEMBLE STRATEGIES ===
        elif selected_strategy in META_ENSEMBLE_STRATEGIES:
            if selected_strategy == 'adaptive_ensemble':
                from adaptive_ensemble import select_adaptive_ensemble_stocks
                return select_adaptive_ensemble_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n,
                    price_history_cache=price_history_cache,
                )
            elif selected_strategy == 'volatility_ensemble':
                from volatility_ensemble import select_volatility_ensemble_stocks
                return select_volatility_ensemble_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n,
                    price_history_cache=price_history_cache,
                )
            elif selected_strategy == 'correlation_ensemble':
                from correlation_ensemble import select_correlation_ensemble_stocks
                return select_correlation_ensemble_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n,
                    price_history_cache=price_history_cache,
                )
            elif selected_strategy == 'dynamic_pool':
                from dynamic_pool import select_dynamic_pool_stocks
                return select_dynamic_pool_stocks(
                    all_tickers,
                    ticker_data_grouped,
                    current_date,
                    top_n,
                    price_history_cache=price_history_cache,
                )
            elif selected_strategy == 'voting_ensemble':
                from shared_strategies import select_voting_ensemble_stocks
                return select_voting_ensemble_stocks(all_tickers, ticker_data_grouped, current_date, top_n)
            return []

        # === AI STRATEGIES ===
        elif selected_strategy in META_AI_STRATEGIES:
            from shared_strategies import select_ai_elite_with_training
            return select_ai_elite_with_training(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy in META_AI_REGIME_STRATEGIES:
            from ai_regime_strategy import select_ai_regime_stocks
            return select_ai_regime_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'universal_model':
            from shared_strategies import _select_universal_model_stocks
            return _select_universal_model_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        # === SPECIAL STRATEGIES ===
        elif selected_strategy == 'inverse_etf_hedge':
            from inverse_etf_hedge_strategy import select_inverse_etf_hedge_stocks
            return select_inverse_etf_hedge_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        elif selected_strategy == 'analyst_recommendation':
            from analyst_recommendation_strategy import select_analyst_recommendation_stocks
            return select_analyst_recommendation_stocks(all_tickers, ticker_data_grouped, current_date, top_n)

        # === ADAPTIVE REBALANCING STRATEGIES ===
        elif selected_strategy in META_ADAPTIVE_REBALANCING_STRATEGIES:
            # These use the same base selection as Static BH 1Y
            return select_top_performers(
                all_tickers,
                ticker_data_grouped,
                current_date,
                lookback_days=365,
                top_n=top_n,
                price_history_cache=price_history_cache,
            )

        # === BOLLINGER BANDS STRATEGIES ===
        elif selected_strategy in META_BB_STRATEGIES:
            from bollinger_bands_strategy import (
                select_bb_breakout_stocks,
                select_bb_mean_reversion_stocks,
                select_bb_rsi_combo_stocks,
                select_bb_squeeze_breakout_stocks,
            )

            bb_selectors = {
                'bb_mean_reversion': select_bb_mean_reversion_stocks,
                'bb_breakout': select_bb_breakout_stocks,
                'bb_squeeze_breakout': select_bb_squeeze_breakout_stocks,
                'bb_rsi_combo': select_bb_rsi_combo_stocks,
            }
            selector = bb_selectors.get(selected_strategy)
            if selector is None:
                return []
            if selected_strategy == 'bb_rsi_combo':
                return selector(all_tickers, ticker_data_grouped, current_date, top_n)
            return selector(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        # === TREND STRATEGIES ===
        elif selected_strategy == 'trend_breakout':
            from new_strategies import select_trend_breakout_stocks
            return select_trend_breakout_stocks(
                all_tickers,
                ticker_data_grouped,
                current_date,
                top_n,
                price_history_cache=price_history_cache,
            )

        else:
            print(f"⚠️ Meta strategy: Unknown sub-strategy '{selected_strategy}', returning no selection")
            return []

    except Exception as e:
        print(f"⚠️ Meta strategy error for '{selected_strategy}': {e}")
        return []
