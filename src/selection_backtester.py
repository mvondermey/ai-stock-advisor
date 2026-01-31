"""
Selection Strategy Backtester

Compares multiple stock selection strategies by measuring their forward performance.
Each strategy selects top N stocks at the selection date, then we measure how those
picks performed going forward.

This helps identify which selection criteria leads to the best-performing portfolios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import N_TOP_TICKERS


def calculate_momentum(prices: pd.Series, days: int) -> Optional[float]:
    """Calculate momentum (return) over specified days."""
    if prices is None or len(prices) < days // 2:  # Need at least half the days
        return None
    
    # Get prices from 'days' ago to now
    try:
        # Use the exact lookback period, not just first/last prices
        if len(prices) >= days:
            start_price = prices.iloc[-days]  # Price 'days' ago
            end_price = prices.iloc[-1]       # Current price
        else:
            # If not enough data, use first available price
            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
        
        if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
            return None
        
        return ((end_price / start_price) - 1.0) * 100.0
    except Exception:
        return None


def calculate_sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.05) -> Optional[float]:
    """Calculate annualized Sharpe ratio from price series."""
    if prices is None or len(prices) < 20:
        return None
    
    try:
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return None
        
        # Annualize
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        
        if std_return == 0 or pd.isna(std_return):
            return None
        
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        excess_return = mean_return - risk_free_rate
        
        return excess_return / std_return
    except Exception:
        return None


def calculate_volatility(prices: pd.Series) -> Optional[float]:
    """Calculate annualized volatility (standard deviation of returns)."""
    if prices is None or len(prices) < 20:
        return None
    
    try:
        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return None
        
        # Annualized volatility
        return returns.std() * np.sqrt(252) * 100  # As percentage
    except Exception:
        return None


def calculate_relative_strength(ticker_prices: pd.Series, benchmark_prices: pd.Series) -> Optional[float]:
    """Calculate relative strength vs benchmark (ticker return - benchmark return)."""
    if ticker_prices is None or benchmark_prices is None:
        return None
    
    try:
        ticker_return = calculate_momentum(ticker_prices, len(ticker_prices))
        benchmark_return = calculate_momentum(benchmark_prices, len(benchmark_prices))
        
        if ticker_return is None or benchmark_return is None:
            return None
        
        return ticker_return - benchmark_return
    except Exception:
        return None


def get_price_series(ticker: str, all_tickers_data: dict, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
    """Extract price series for a ticker within date range."""
    try:
        if ticker not in all_tickers_data or all_tickers_data[ticker].empty:
            return None
        
        df = all_tickers_data[ticker]
        
        # Find price column
        candidates = ['Close', 'Adj Close', 'Adj close', 'close', 'adj close']
        price_col = next((c for c in candidates if c in df.columns), None)
        if price_col is None:
            return None
        
        # Filter to date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df.loc[mask]
        
        if df_filtered.empty:
            return None
        
        prices = pd.to_numeric(df_filtered[price_col], errors='coerce').ffill().bfill().dropna()
        return prices if len(prices) >= 2 else None
    except Exception:
        return None


def run_selection_strategy_comparison(
    all_tickers_data: dict,
    all_available_tickers: List[str],
    selection_date: datetime,
    evaluation_date: datetime,
    n_top: int = None,
    benchmark_ticker: str = 'SPY'
) -> Dict:
    """
    Run multiple selection strategies and compare their forward performance.
    
    Args:
        all_tickers_data: Dict of ticker -> DataFrame with price data
        all_available_tickers: List of all tickers to consider
        selection_date: Date when stocks are selected (historical)
        evaluation_date: Date to measure forward performance (usually today)
        n_top: Number of top stocks to select per strategy
        benchmark_ticker: Ticker to use for relative strength calculation
    
    Returns:
        Dict with strategy results and comparison table
    """
    if n_top is None:
        n_top = N_TOP_TICKERS
    
    print(f"\n{'='*80}")
    print(f"📊 SELECTION STRATEGY COMPARISON")
    print(f"   Selection Date: {selection_date.strftime('%Y-%m-%d')}")
    print(f"   Evaluation Date: {evaluation_date.strftime('%Y-%m-%d')}")
    print(f"   Forward Period: {(evaluation_date - selection_date).days} days")
    print(f"   Stocks per Strategy: {n_top}")
    print(f"{'='*80}")
    
    # Define lookback periods for each strategy (days before selection_date)
    strategies = {
        '1Y Momentum': {'lookback': 365, 'metric': 'momentum'},
        '6M Momentum': {'lookback': 180, 'metric': 'momentum'},
        '3M Momentum': {'lookback': 90, 'metric': 'momentum'},
        '1M Momentum': {'lookback': 30, 'metric': 'momentum'},
        'Risk-Adj (Sharpe)': {'lookback': 365, 'metric': 'sharpe'},
        'Low Volatility': {'lookback': 365, 'metric': 'low_volatility'},
        'Mean Reversion': {'lookback': 90, 'metric': 'mean_reversion'},
        'Rel Strength vs SPY': {'lookback': 365, 'metric': 'relative_strength'},
    }
    
    # Get benchmark data for relative strength
    benchmark_start = selection_date - timedelta(days=400)
    benchmark_prices_historical = get_price_series(benchmark_ticker, all_tickers_data, benchmark_start, selection_date)
    
    # Calculate metrics for all tickers at selection_date
    print(f"\n📈 Calculating selection metrics for {len(all_available_tickers)} tickers...")
    
    ticker_metrics = {}
    
    for ticker in tqdm(all_available_tickers, desc="Calculating metrics", ncols=100):
        ticker_metrics[ticker] = {}
        
        for strategy_name, config in strategies.items():
            lookback = config['lookback']
            metric_type = config['metric']
            
            # Get historical prices (before selection date)
            hist_start = selection_date - timedelta(days=lookback + 30)  # Extra buffer
            hist_prices = get_price_series(ticker, all_tickers_data, hist_start, selection_date)
            
            if hist_prices is None or len(hist_prices) < 10:
                ticker_metrics[ticker][strategy_name] = None
                continue
            
            # Trim to exact lookback period
            if len(hist_prices) > lookback:
                hist_prices = hist_prices.iloc[-lookback:]
            
            # Calculate metric based on strategy type
            if metric_type == 'momentum':
                ticker_metrics[ticker][strategy_name] = calculate_momentum(hist_prices, len(hist_prices))
            
            elif metric_type == 'sharpe':
                ticker_metrics[ticker][strategy_name] = calculate_sharpe_ratio(hist_prices)
            
            elif metric_type == 'low_volatility':
                vol = calculate_volatility(hist_prices)
                # Invert so lower volatility = higher score
                ticker_metrics[ticker][strategy_name] = -vol if vol is not None else None
            
            elif metric_type == 'mean_reversion':
                # Select stocks that are DOWN the most (contrarian)
                mom = calculate_momentum(hist_prices, len(hist_prices))
                # Invert so biggest losers get highest score
                ticker_metrics[ticker][strategy_name] = -mom if mom is not None else None
            
            elif metric_type == 'relative_strength':
                if benchmark_prices_historical is not None:
                    ticker_metrics[ticker][strategy_name] = calculate_relative_strength(
                        hist_prices, benchmark_prices_historical
                    )
                else:
                    ticker_metrics[ticker][strategy_name] = None
    
    # Select top N tickers for each strategy
    print(f"\n🎯 Selecting top {n_top} stocks for each strategy...")
    
    strategy_selections = {}
    
    for strategy_name in strategies.keys():
        # Get all tickers with valid metrics for this strategy
        valid_tickers = [
            (ticker, ticker_metrics[ticker][strategy_name])
            for ticker in all_available_tickers
            if ticker_metrics[ticker].get(strategy_name) is not None
        ]
        
        # Sort by metric (descending - highest is best)
        valid_tickers.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        strategy_selections[strategy_name] = [ticker for ticker, _ in valid_tickers[:n_top]]
    
    # Calculate forward performance for each strategy's selections
    print(f"\n📊 Calculating forward performance ({selection_date.strftime('%Y-%m-%d')} → {evaluation_date.strftime('%Y-%m-%d')})...")
    
    strategy_results = {}
    
    for strategy_name, selected_tickers in strategy_selections.items():
        forward_returns = []
        ticker_details = []
        
        for ticker in selected_tickers:
            # Get forward prices (from selection date to evaluation date)
            forward_prices = get_price_series(ticker, all_tickers_data, selection_date, evaluation_date)
            
            if forward_prices is not None and len(forward_prices) >= 2:
                forward_return = calculate_momentum(forward_prices, len(forward_prices))
                if forward_return is not None:
                    forward_returns.append(forward_return)
                    ticker_details.append({
                        'ticker': ticker,
                        'forward_return': forward_return,
                        'selection_score': ticker_metrics[ticker].get(strategy_name)
                    })
        
        # Calculate strategy statistics
        if forward_returns:
            avg_return = np.mean(forward_returns)
            median_return = np.median(forward_returns)
            best_return = max(forward_returns)
            worst_return = min(forward_returns)
            win_rate = sum(1 for r in forward_returns if r > 0) / len(forward_returns) * 100
            
            # Find best and worst tickers
            ticker_details.sort(key=lambda x: x['forward_return'], reverse=True)
            best_ticker = ticker_details[0]['ticker'] if ticker_details else 'N/A'
            worst_ticker = ticker_details[-1]['ticker'] if ticker_details else 'N/A'
            
            strategy_results[strategy_name] = {
                'avg_return': avg_return,
                'median_return': median_return,
                'best_return': best_return,
                'worst_return': worst_return,
                'win_rate': win_rate,
                'best_ticker': best_ticker,
                'worst_ticker': worst_ticker,
                'num_stocks': len(forward_returns),
                'selected_tickers': selected_tickers,
                'ticker_details': ticker_details
            }
        else:
            strategy_results[strategy_name] = {
                'avg_return': 0,
                'median_return': 0,
                'best_return': 0,
                'worst_return': 0,
                'win_rate': 0,
                'best_ticker': 'N/A',
                'worst_ticker': 'N/A',
                'num_stocks': 0,
                'selected_tickers': [],
                'ticker_details': []
            }
    
    # Sort strategies by average return
    sorted_strategies = sorted(
        strategy_results.items(),
        key=lambda x: x[1]['avg_return'],
        reverse=True
    )
    
    # Print comparison table
    print(f"\n{'='*120}")
    print(f"{'SELECTION STRATEGY COMPARISON - FORWARD PERFORMANCE':^120}")
    print(f"{'Selection: ' + selection_date.strftime('%Y-%m-%d') + ' → Evaluation: ' + evaluation_date.strftime('%Y-%m-%d'):^120}")
    print(f"{'='*120}")
    print(f"{'Rank':<6} | {'Strategy':<20} | {'Avg Return':>12} | {'Median':>10} | {'Best':>10} | {'Worst':>10} | {'Win Rate':>10} | {'Best Pick':<10} | {'Worst Pick':<10}")
    print(f"{'-'*120}")
    
    for rank, (strategy_name, results) in enumerate(sorted_strategies, 1):
        print(f"{rank:<6} | {strategy_name:<20} | {results['avg_return']:>+11.2f}% | {results['median_return']:>+9.2f}% | {results['best_return']:>+9.2f}% | {results['worst_return']:>+9.2f}% | {results['win_rate']:>9.1f}% | {results['best_ticker']:<10} | {results['worst_ticker']:<10}")
    
    print(f"{'='*120}")
    
    # Print winner summary
    winner = sorted_strategies[0]
    print(f"\n🏆 BEST SELECTION STRATEGY: {winner[0]}")
    print(f"   Average Forward Return: {winner[1]['avg_return']:+.2f}%")
    print(f"   Win Rate: {winner[1]['win_rate']:.1f}%")
    print(f"   Top Picks: {', '.join(winner[1]['selected_tickers'][:5])}")
    
    # Compare with current 1Y Momentum strategy
    current_strategy = strategy_results.get('1Y Momentum', {})
    if current_strategy and winner[0] != '1Y Momentum':
        improvement = winner[1]['avg_return'] - current_strategy.get('avg_return', 0)
        print(f"\n📈 vs Current (1Y Momentum): {improvement:+.2f}% improvement")
    
    print(f"{'='*80}\n")
    
    return {
        'strategy_results': dict(sorted_strategies),
        'selection_date': selection_date,
        'evaluation_date': evaluation_date,
        'n_top': n_top,
        'winner': winner[0]
    }


def print_strategy_stock_overlap(strategy_results: Dict) -> None:
    """Print overlap analysis between different strategies' stock selections."""
    print(f"\n{'='*80}")
    print(f"{'STRATEGY STOCK OVERLAP ANALYSIS':^80}")
    print(f"{'='*80}")
    
    strategies = list(strategy_results.keys())
    
    # Create overlap matrix
    print(f"\n{'Strategy':<20}", end="")
    for s in strategies:
        print(f" | {s[:8]:>8}", end="")
    print()
    print("-" * (22 + 11 * len(strategies)))
    
    for s1 in strategies:
        print(f"{s1:<20}", end="")
        tickers1 = set(strategy_results[s1].get('selected_tickers', []))
        
        for s2 in strategies:
            tickers2 = set(strategy_results[s2].get('selected_tickers', []))
            overlap = len(tickers1 & tickers2)
            print(f" | {overlap:>8}", end="")
        print()
    
    print(f"{'='*80}\n")
