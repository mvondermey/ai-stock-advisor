"""
Summary Phase Module
Handles the final summary printing for the AI stock advisor system.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from config import (
    MIN_PROBA_BUY, MIN_PROBA_SELL, TARGET_PERCENTAGE, CLASS_HORIZON,
    INVESTMENT_PER_STOCK, PERIOD_HORIZONS
)


def print_final_summary(
    sorted_final_results: List[Dict],
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    optimized_params_per_ticker: Dict[str, Dict[str, float]],
    final_strategy_value_1y: float,
    final_buy_hold_value_1y: float,
    ai_1y_return: float,
    final_strategy_value_ytd: float,
    final_buy_hold_value_ytd: float,
    ai_ytd_return: float,
    final_strategy_value_3month: float,
    final_buy_hold_value_3month: float,
    ai_3month_return: float,
    initial_balance_used: float,
    num_tickers_analyzed: int,
    final_strategy_value_1month: float,
    ai_1month_return: float,
    final_buy_hold_value_1month: float,
    performance_metrics_buy_hold_1y: List[Dict],
    top_performers_data: List[Tuple],
    strategy_results_ytd: List[Dict] = None,
    strategy_results_3month: List[Dict] = None,
    strategy_results_1month: List[Dict] = None,
    performance_metrics_buy_hold_ytd: List[Dict] = None,
    performance_metrics_buy_hold_3month: List[Dict] = None,
    performance_metrics_buy_hold_1month: List[Dict] = None,
    prediction_vs_bh_1y: List[Dict] = None,
    prediction_vs_bh_ytd: List[Dict] = None,
    prediction_vs_bh_3month: List[Dict] = None,
    prediction_vs_bh_1month: List[Dict] = None,
    final_rule_value_1y: float = None,
    rule_1y_return: float = None,
    final_rule_value_ytd: float = None,
    rule_ytd_return: float = None,
    final_rule_value_3month: float = None,
    rule_3month_return: float = None,
    final_rule_value_1month: float = None,
    rule_1month_return: float = None
) -> None:
    """Prints the final summary of the backtest results."""
    
    # 🔍 DEBUG: Check what the function received
    print(f"\n[DEBUG] Inside print_final_summary - Received parameters:")
    print(f"  - final_strategy_value_1y: ${final_strategy_value_1y:,.2f}")
    print(f"  - final_buy_hold_value_1y: ${final_buy_hold_value_1y:,.2f}")
    print(f"  - ai_1y_return: {ai_1y_return:.2f}%\n")

    def _print_pred_vs_bh(period_name: str, perf_metrics: List[Dict], horizon_days: Optional[int] = None):
        """Helper to show mean predicted vs B&H per ticker for a period, if available."""
        if not perf_metrics or not isinstance(perf_metrics, list):
            return
        rows = []
        for res in perf_metrics:
            if not isinstance(res, dict):
                continue  # skip non-dict entries (e.g., float portfolio values)
            # These metrics are stored only for AI strategy runs
            ticker = res.get("ticker", "N/A")
            pred_mean = res.get("pred_mean_pct")
            pred_min = res.get("pred_min_pct")
            pred_max = res.get("pred_max_pct")
            # Use horizon-matched B&H return if available, otherwise fall back to individual_bh_return
            bh_return = res.get("bh_horizon_return_pct", res.get("individual_bh_return"))
            if pred_mean is None and pred_min is None and pred_max is None:
                continue
            rows.append((ticker, pred_mean, pred_min, pred_max, bh_return))
        if not rows:
            return
        if horizon_days is not None:
            print(f"\n📊 Predictions vs Buy & Hold - {period_name} (Horizon: {horizon_days}d)")
            bh_header = f"B&H {horizon_days}d"
        else:
            print(f"\n📊 Predictions vs Buy & Hold - {period_name}")
            bh_header = "B&H Return"
        print("-" * 100)
        print(f"{'Ticker':<10} | {'Pred Mean':>10} | {'Pred Min':>10} | {'Pred Max':>10} | {bh_header:>12}")
        print("-" * 100)
        for (ticker, meanv, minv, maxv, bhv) in rows:
            mean_str = f"{meanv:>9.2f}%" if meanv is not None else "   N/A   "
            min_str = f"{minv:>9.2f}%" if minv is not None else "   N/A   "
            max_str = f"{maxv:>9.2f}%" if maxv is not None else "   N/A   "
            bh_str = f"{bhv:>11.2f}%" if bhv is not None else "    N/A    "
            print(f"{ticker:<10} | {mean_str} | {min_str} | {max_str} | {bh_str}")
        print("-" * 100)
    
    # Show prediction vs B&H tables for AI strategy (per period) when metrics include prediction stats
    _print_pred_vs_bh("1-Year", prediction_vs_bh_1y if prediction_vs_bh_1y is not None else sorted_final_results, PERIOD_HORIZONS.get("1-Year"))
    _print_pred_vs_bh("YTD", prediction_vs_bh_ytd if prediction_vs_bh_ytd is not None else strategy_results_ytd, PERIOD_HORIZONS.get("YTD"))
    _print_pred_vs_bh("3-Month", prediction_vs_bh_3month if prediction_vs_bh_3month is not None else strategy_results_3month, PERIOD_HORIZONS.get("3-Month"))
    _print_pred_vs_bh("1-Month", prediction_vs_bh_1month if prediction_vs_bh_1month is not None else strategy_results_1month, PERIOD_HORIZONS.get("1-Month"))
    
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}")
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 80)
    print(f"{'Period':<15} | {'AI Strategy':<25} | {'Rule-Based':<25} | {'Buy & Hold':<25}")
    print("-" * 80)
    
    # 1-Year
    rule_1y_str = f"${final_rule_value_1y:,.2f} ({rule_1y_return:+.2f}%)" if final_rule_value_1y is not None else "N/A"
    print(f"{'1-Year':<15} | ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)".ljust(43) + f"| {rule_1y_str}".ljust(28) + f"| ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    
    # YTD
    rule_ytd_str = f"${final_rule_value_ytd:,.2f} ({rule_ytd_return:+.2f}%)" if final_rule_value_ytd is not None else "N/A"
    print(f"{'YTD':<15} | ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)".ljust(43) + f"| {rule_ytd_str}".ljust(28) + f"| ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    
    # 3-Month
    rule_3m_str = f"${final_rule_value_3month:,.2f} ({rule_3month_return:+.2f}%)" if final_rule_value_3month is not None else "N/A"
    print(f"{'3-Month':<15} | ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)".ljust(43) + f"| {rule_3m_str}".ljust(28) + f"| ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    
    # 1-Month
    rule_1m_str = f"${final_rule_value_1month:,.2f} ({rule_1month_return:+.2f}%)" if final_rule_value_1month is not None else "N/A"
    bh_1month_return_str = "N/A" if final_buy_hold_value_1month == 0 else f"{((final_buy_hold_value_1month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%"
    ai_1month_return_str = "N/A" if final_strategy_value_1month == 0 else f"{ai_1month_return:+.2f}%"
    bh_1month_value_str = "N/A" if final_buy_hold_value_1month == 0 else f"${final_buy_hold_value_1month:,.2f}"
    
    print(f"{'1-Month':<15} | ${final_strategy_value_1month:,.2f} ({ai_1month_return_str})".ljust(43) + f"| {rule_1m_str}".ljust(28) + f"| {bh_1month_value_str} ({bh_1month_return_str})")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 190)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Max Shares Held':>25}")
    print("-" * 190)
    for res in sorted_final_results:
        ticker = str(res.get('ticker', 'N/A'))
        # ✅ FIX: Show $0 allocated capital for failed tickers (they weren't actually traded)
        if res.get('status') == 'failed':
            allocated_capital = 0.0
            strategy_gain = 0.0
        else:
            allocated_capital = INVESTMENT_PER_STOCK
            strategy_gain = res.get('performance', 0.0) - allocated_capital

        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        
        # ✅ FIX: Show N/A for shares if ticker failed (wasn't traded)
        # Now showing max_shares_held instead of shares_before_liquidation
        if res.get('status') == 'failed':
            max_shares_str = "N/A".rjust(25)
        else:
            max_shares_value = res.get('final_shares', res.get('shares_before_liquidation', 0.0))
            max_shares_str = f"{max_shares_value:>24.2f}"
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {max_shares_str}")
    print("-" * 190)

    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
        
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 136)

    # Show Buy & Hold results for tickers not selected by the AI strategy
    ai_tickers = {str(res.get('ticker')) for res in sorted_final_results}
    bh_only_results = [
        res for res in performance_metrics_buy_hold_1y
        if isinstance(res, dict) and str(res.get('ticker')) not in ai_tickers
    ]
    if bh_only_results:
        print("\n📊 Buy & Hold (Tickers not held by AI strategy):")
        print("-" * 136)
        print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
        print("-" * 136)
        for res in bh_only_results:
            ticker = str(res.get('ticker', 'N/A'))
            allocated_capital = INVESTMENT_PER_STOCK
            strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
            
            one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
            for t, p1y, pytd in top_performers_data:
                if t == ticker:
                    one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                    ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                    break

            one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
            ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
            sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
            shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

            print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
        print("-" * 136)

    # Print YTD, 3-Month, and 1-Month period tables if data is available
    for period_name, strategy_results, buy_hold_results in [
        ("YTD", strategy_results_ytd, performance_metrics_buy_hold_ytd),
        ("3-Month", strategy_results_3month, performance_metrics_buy_hold_3month),
        ("1-Month", strategy_results_1month, performance_metrics_buy_hold_1month)
    ]:
        if strategy_results and len(strategy_results) > 0:
            print(f"\n📈 Individual Ticker Performance ({period_name} - AI Strategy):")
            print("-" * 190)
            print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Max Shares Held':>25}")
            print("-" * 190)
            
            # Handle both dict and non-dict items safely
            if strategy_results and len(strategy_results) > 0 and isinstance(strategy_results[0], dict):
                sorted_period_results = sorted(strategy_results, key=lambda x: x.get('one_year_perf', -np.inf) if pd.notna(x.get('one_year_perf')) else -np.inf, reverse=True)
            else:
                sorted_period_results = []  # Empty list if invalid data
            
            for res in sorted_period_results:
                # Extra safety check
                if not isinstance(res, dict):
                    continue
                ticker = str(res.get('ticker', 'N/A'))
                
                if res.get('status') == 'failed':
                    allocated_capital = 0.0
                    strategy_gain = 0.0
                else:
                    allocated_capital = INVESTMENT_PER_STOCK
                    strategy_gain = res.get('performance', 0.0) - allocated_capital
                
                one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
                ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
                sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
                buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
                sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
                last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
                
                if res.get('status') == 'failed':
                    max_shares_str = "N/A".rjust(25)
                else:
                    max_shares_value = res.get('final_shares', res.get('shares_before_liquidation', 0.0))
                    max_shares_str = f"{max_shares_value:>24.2f}"
                
                print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {max_shares_str}")
            print("-" * 190)
        
        if buy_hold_results and len(buy_hold_results) > 0:
            print(f"\n📈 Individual Ticker Performance ({period_name} - Buy & Hold Strategy):")
            print("-" * 136)
            print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
            print("-" * 136)
            
            if buy_hold_results and isinstance(buy_hold_results[0], dict):
                sorted_buy_hold_period = sorted(buy_hold_results, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)
            else:
                sorted_buy_hold_period = buy_hold_results
            
            for res in sorted_buy_hold_period:
                ticker = str(res.get('ticker', 'N/A'))
                allocated_capital = INVESTMENT_PER_STOCK
                strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
                
                one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
                for t, p1y, pytd in top_performers_data:
                    if t == ticker:
                        one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                        ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                        break
                
                one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
                ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
                sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
                shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"
                
                print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
            print("-" * 136)

    print("\n🤖 ML Model Status:")
    # ✅ FIX: Only show unique tickers (avoid duplicates)
    seen_tickers = set()
    for ticker in sorted_final_results:
        t = ticker['ticker']
        if t not in seen_tickers:
            seen_tickers.add(t)
            buy_model_status = "✅ Trained" if models_buy.get(t) else "❌ Not Trained"
            sell_model_status = "✅ Trained" if models_sell.get(t) else "❌ Not Trained"
            print(f"  - {t}: Buy Model: {buy_model_status}, Sell Model: {sell_model_status}")
    print("="*80)

    print("\n💡 Next Steps:")
    print("  - Review individual ticker performance and trade logs for deeper insights.")
    print("  - Experiment with different `MARKET_SELECTION` options and `N_TOP_TICKERS`.")
    print("  - Adjust `TARGET_PERCENTAGE` and `RISK_PER_TRADE` for different risk appetites.")
    print("  - Consider enabling `USE_MARKET_FILTER` and `USE_PERFORMANCE_BENCHMARK` for additional filtering.")
    print("  - Explore advanced ML models or feature engineering for further improvements.")
    print("="*80)


def print_prediction_vs_actual_comparison(
    period_name: str,
    prediction_records: List[Dict],
    top_performers_data: List[Tuple]
) -> None:
    """
    Print comparison table showing AI predictions vs actual returns vs B&H.
    
    Args:
        period_name: Name of the period (e.g., "1-Year", "YTD", "3-Month", "1-Month")
        prediction_records: List of dicts with keys: ticker, predicted_return, actual_return, date
        top_performers_data: List of (ticker, 1y_perf, ytd_perf) tuples for B&H comparison
    """
    if not prediction_records:
        print(f"\n⚠️  No prediction data available for {period_name} period.")
        return
    
    print(f"\n{'='*120}")
    print(f"📊 AI PREDICTIONS vs ACTUAL RETURNS vs BUY & HOLD - {period_name.upper()}")
    print(f"{'='*120}")
    
    # Group by ticker and get the latest prediction
    ticker_predictions = {}
    for record in prediction_records:
        ticker = record['ticker']
        if ticker not in ticker_predictions:
            ticker_predictions[ticker] = record
        else:
            # Keep the latest prediction
            if record.get('date', '') > ticker_predictions[ticker].get('date', ''):
                ticker_predictions[ticker] = record
    
    # Print table header
    print(f"{'Ticker':<10} | {'AI Predicted':>15} | {'Actual Return':>15} | {'B&H Return':>15} | {'AI vs Actual':>15} | {'AI vs B&H':>15} | {'Status':>10}")
    print(f"{'-'*120}")
    
    total_ai_predicted = 0.0
    total_actual = 0.0
    total_bh = 0.0
    wins_vs_bh = 0
    total_stocks = 0
    
    # Sort by ticker
    for ticker in sorted(ticker_predictions.keys()):
        record = ticker_predictions[ticker]
        predicted_return = record.get('predicted_return', 0.0)
        actual_return = record.get('actual_return', 0.0)
        
        # Get B&H return for this ticker
        bh_return = 0.0
        for t, perf_1y, perf_ytd in top_performers_data:
            if t == ticker:
                if period_name == "1-Year":
                    bh_return = perf_1y / 100.0  # Convert to decimal
                elif period_name == "YTD":
                    bh_return = perf_ytd / 100.0
                break
        
        # Calculate differences
        ai_vs_actual = predicted_return - actual_return
        ai_vs_bh = actual_return - bh_return  # Actual performance vs B&H
        
        # Determine status
        if actual_return > bh_return:
            status = "✅ WIN"
            wins_vs_bh += 1
        elif actual_return < bh_return:
            status = "❌ LOSS"
        else:
            status = "➖ TIE"
        
        # Accumulate totals
        total_ai_predicted += predicted_return
        total_actual += actual_return
        total_bh += bh_return
        total_stocks += 1
        
        # Print row
        print(f"{ticker:<10} | {predicted_return:>14.2%} | {actual_return:>14.2%} | {bh_return:>14.2%} | {ai_vs_actual:>+14.2%} | {ai_vs_bh:>+14.2%} | {status:>10}")
    
    print(f"{'-'*120}")
    
    # Print summary
    avg_ai_predicted = total_ai_predicted / total_stocks if total_stocks > 0 else 0.0
    avg_actual = total_actual / total_stocks if total_stocks > 0 else 0.0
    avg_bh = total_bh / total_stocks if total_stocks > 0 else 0.0
    win_rate = (wins_vs_bh / total_stocks * 100) if total_stocks > 0 else 0.0
    
    print(f"{'AVERAGE':<10} | {avg_ai_predicted:>14.2%} | {avg_actual:>14.2%} | {avg_bh:>14.2%} | {avg_ai_predicted - avg_actual:>+14.2%} | {avg_actual - avg_bh:>+14.2%} | {win_rate:>9.1f}%")
    print(f"{'='*120}")
    
    print(f"\n📈 Summary:")
    print(f"  - Win Rate (AI Strategy vs B&H): {win_rate:.1f}% ({wins_vs_bh}/{total_stocks})")
    print(f"  - Average AI Prediction Error: {(avg_ai_predicted - avg_actual):.2%}")
    print(f"  - Average Outperformance vs B&H: {(avg_actual - avg_bh):.2%}")
    print(f"{'='*120}\n")


def print_horizon_validation_summary(
    periods_trained: List[str],
    horizons_used: Dict[str, int]
) -> None:
    """
    Print summary showing which horizons were used for each period.
    
    Args:
        periods_trained: List of period names that were trained
        horizons_used: Dict mapping period name to horizon days used
    """
    print(f"\n{'='*80}")
    print(f"📏 HORIZON VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Period':<15} | {'Expected Horizon':>20} | {'Actual Horizon':>20} | {'Status':>10}")
    print(f"{'-'*80}")
    
    all_correct = True
    for period in periods_trained:
        expected = PERIOD_HORIZONS.get(period, "N/A")
        actual = horizons_used.get(period, "N/A")
        
        if expected == "N/A" or actual == "N/A":
            status = "⚠️  N/A"
            all_correct = False
        elif period == "YTD":
            # YTD is dynamically calculated, so just check it's reasonable
            if isinstance(actual, int) and 150 <= actual <= 252:
                status = "✅ OK"
            else:
                status = "❌ ERROR"
                all_correct = False
        elif expected == actual:
            status = "✅ OK"
        else:
            status = "❌ ERROR"
            all_correct = False
        
        expected_str = f"{expected}d" if isinstance(expected, int) else str(expected)
        actual_str = f"{actual}d" if isinstance(actual, int) else str(actual)
        
        print(f"{period:<15} | {expected_str:>20} | {actual_str:>20} | {status:>10}")
    
    print(f"{'-'*80}")
    
    if all_correct:
        print("✅ All horizons are correctly configured!")
    else:
        print("⚠️  Some horizons may not be correctly configured. Review the errors above.")
    
    print(f"{'='*80}\n")


def print_training_phase_summary(
    models_buy: Dict,
    models_sell: Dict,
    scalers: Dict,
    failed_tickers: Dict[str, str] = None
) -> None:
    """
    Print summary of the training phase.
    
    Args:
        models_buy: Dict of trained buy models
        models_sell: Dict of trained sell models
        scalers: Dict of trained scalers
        failed_tickers: Dict mapping ticker to failure reason
    """
    print(f"\n{'='*80}")
    print(f"🤖 TRAINING PHASE SUMMARY")
    print(f"{'='*80}")
    
    total_tickers = len(models_buy) + len(failed_tickers) if failed_tickers else len(models_buy)
    successful = len(models_buy)
    failed = len(failed_tickers) if failed_tickers else 0
    success_rate = (successful / total_tickers * 100) if total_tickers > 0 else 0.0
    
    print(f"  Total Tickers: {total_tickers}")
    print(f"  ✅ Successfully Trained: {successful} ({success_rate:.1f}%)")
    print(f"  ❌ Failed to Train: {failed}")
    
    if failed_tickers and failed > 0:
        print(f"\n  Failed Tickers:")
        for ticker, reason in failed_tickers.items():
            print(f"    - {ticker}: {reason}")
    
    print(f"\n  Model Types:")
    if len(models_buy) > 0:
        sample_ticker = next(iter(models_buy))
        buy_model_type = type(models_buy[sample_ticker]).__name__
        sell_model_type = type(models_sell[sample_ticker]).__name__
        print(f"    - Buy Model: {buy_model_type}")
        print(f"    - Sell Model: {sell_model_type}")
    else:
        print(f"    - No models trained")
    
    print(f"{'='*80}\n")


def print_portfolio_comparison_summary(
    period_name: str,
    ai_strategy_return: float,
    bh_portfolio_return: float,
    ai_final_value: float,
    bh_final_value: float,
    initial_balance: float
) -> None:
    """
    Print portfolio-level comparison between AI strategy and Buy & Hold.
    
    Args:
        period_name: Name of the period
        ai_strategy_return: AI strategy return percentage
        bh_portfolio_return: Buy & Hold portfolio return percentage
        ai_final_value: AI strategy final portfolio value
        bh_final_value: Buy & Hold final portfolio value
        initial_balance: Initial balance
    """
    print(f"\n{'='*80}")
    print(f"💼 PORTFOLIO COMPARISON - {period_name.upper()}")
    print(f"{'='*80}")
    
    outperformance = ai_strategy_return - bh_portfolio_return
    
    print(f"  Initial Balance: ${initial_balance:,.2f}")
    print(f"  ")
    print(f"  AI Strategy:")
    print(f"    - Final Value: ${ai_final_value:,.2f}")
    print(f"    - Return: {ai_strategy_return:+.2f}%")
    print(f"  ")
    print(f"  Buy & Hold Portfolio:")
    print(f"    - Final Value: ${bh_final_value:,.2f}")
    print(f"    - Return: {bh_portfolio_return:+.2f}%")
    print(f"  ")
    print(f"  {'─'*40}")
    
    if outperformance > 0:
        print(f"  ✅ AI OUTPERFORMED by {outperformance:+.2f}%")
    elif outperformance < 0:
        print(f"  ❌ AI UNDERPERFORMED by {outperformance:.2f}%")
    else:
        print(f"  ➖ TIED with B&H")
    
    value_diff = ai_final_value - bh_final_value
    print(f"  Value Difference: ${value_diff:+,.2f}")
    
    print(f"{'='*80}\n")

