"""
Summary Phase Module
Handles the final summary printing for the AI stock advisor system.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from config import (
    MIN_PROBA_BUY, MIN_PROBA_SELL, TARGET_PERCENTAGE, CLASS_HORIZON,
    INVESTMENT_PER_STOCK
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
    final_simple_rule_value_1y: float,
    simple_rule_1y_return: float,
    final_simple_rule_value_ytd: float,
    simple_rule_ytd_return: float,
    final_simple_rule_value_3month: float,
    simple_rule_3month_return: float,
    final_simple_rule_value_1month: float,
    simple_rule_1month_return: float,
    performance_metrics_simple_rule_1y: List[Dict],
    performance_metrics_buy_hold_1y: List[Dict],
    top_performers_data: List[Tuple]
) -> None:
    """Prints the final summary of the backtest results."""
    
    # 🔍 DEBUG: Check what the function received
    print(f"\n[DEBUG] Inside print_final_summary - Received parameters:")
    print(f"  - final_strategy_value_1y: ${final_strategy_value_1y:,.2f}")
    print(f"  - final_buy_hold_value_1y: ${final_buy_hold_value_1y:,.2f}")
    print(f"  - ai_1y_return: {ai_1y_return:.2f}%\n")
    
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}")
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)")
    print(f"  1-Year Buy & Hold Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  YTD AI Strategy Value: ${final_strategy_value_ytd:,.2f} ({ai_ytd_return:+.2f}%)")
    print(f"  YTD Simple Rule Value: ${final_simple_rule_value_ytd:,.2f} ({simple_rule_ytd_return:+.2f}%)")
    print(f"  YTD Buy & Hold Value: ${final_buy_hold_value_ytd:,.2f} ({((final_buy_hold_value_ytd - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    print(f"  3-Month AI Strategy Value: ${final_strategy_value_3month:,.2f} ({ai_3month_return:+.2f}%)")
    print(f"  3-Month Simple Rule Value: ${final_simple_rule_value_3month:,.2f} ({simple_rule_3month_return:+.2f}%)")
    print(f"  3-Month Buy & Hold Value: ${final_buy_hold_value_3month:,.2f} ({((final_buy_hold_value_3month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    print("-" * 40)
    # ✅ FIX: Show 0% or N/A when no tickers were backtested (not -100%)
    bh_1month_return_str = "N/A" if final_buy_hold_value_1month == 0 else f"{((final_buy_hold_value_1month - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%"
    ai_1month_return_str = "N/A" if final_strategy_value_1month == 0 else f"{ai_1month_return:+.2f}%"
    simple_1month_return_str = "N/A" if final_simple_rule_value_1month == 0 else f"{simple_rule_1month_return:+.2f}%"
    
    print(f"  1-Month AI Strategy Value: ${final_strategy_value_1month:,.2f} ({ai_1month_return_str})")
    print(f"  1-Month Simple Rule Value: ${final_simple_rule_value_1month:,.2f} ({simple_1month_return_str})")
    print(f"  1-Month Buy & Hold Value: ${final_buy_hold_value_1month:,.2f} ({bh_1month_return_str})")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 290)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Max Shares Held':>25}")
    print("-" * 290)
    for res in sorted_final_results:
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', CLASS_HORIZON)
        opt_status = optimized_params.get('optimization_status', 'N/A')

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
            max_shares_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {max_shares_str}")
    print("-" * 290)

    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 136)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 136)
    
    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital
        
        one_year_perf_benchmark, ytd_perf_benchmark = np.nan, np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                ytd_perf_benchmark = pytd if pd.notna(pytd) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        ytd_perf_str = f"{ytd_perf_benchmark:>9.2f}%" if pd.notna(ytd_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 136)

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

