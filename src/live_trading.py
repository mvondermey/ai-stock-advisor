"""
Live Trading Module
Executes daily trades on Alpaca using the SAME logic as the backtest:
1. Load regression models
2. Predict returns for all stocks
3. Rank by predicted return
4. Buy top 3 stocks
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    INVESTMENT_PER_STOCK,
    CLASS_HORIZON
)
from ticker_selection import get_all_tickers
from data_fetcher import load_prices_robust
from prediction import (
    rank_tickers_by_predicted_return,
    load_models_for_tickers,
    get_feature_set_from_saved_model
)

# Alpaca API imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    print("⚠️ alpaca-py not installed. Run: pip install alpaca-py")
    ALPACA_AVAILABLE = False


# --- Configuration ---
LIVE_TRADING_ENABLED = True  # Set to False for dry-run (no actual orders)
INVESTMENT_PER_STOCK_LIVE = INVESTMENT_PER_STOCK  # Inherit from config.py
MODEL_MAX_AGE_DAYS = 90  # Only use models trained in last 90 days
USE_PAPER_TRADING = True  # True = paper trading (fake money), False = REAL MONEY
TOP_N_STOCKS = 3  # Number of stocks to hold (matches backtest)

# --- Strategy Selection ---
# Choose which strategy to use for live trading:
# 'ai' = AI predictions (requires trained models)
# 'static_bh' = Static Buy & Hold (top performers from backtest)
# 'dynamic_bh_1y' = Dynamic BH rebalancing annually
# 'dynamic_bh_3m' = Dynamic BH rebalancing quarterly
# 'dynamic_bh_1m' = Dynamic BH rebalancing monthly
# 'risk_adj_mom' = Risk-Adjusted Momentum
# 'mean_reversion' = Mean Reversion
LIVE_TRADING_STRATEGY = 'ai'  # Change this to select your strategy


def get_alpaca_client() -> Optional[TradingClient]:
    """Initialize Alpaca trading client."""
    if not ALPACA_AVAILABLE:
        print("❌ Alpaca library not available")
        return None
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ Alpaca API keys not set. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        return None
    
    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=USE_PAPER_TRADING)
        account = client.get_account()
        
        mode = "Paper" if USE_PAPER_TRADING else "LIVE"
        print(f"✅ Connected to Alpaca {mode} Trading")
        print(f"  💰 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  📈 Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  💵 Cash: ${float(account.cash):,.2f}")
        
        return client
    except Exception as e:
        print(f"❌ Error connecting to Alpaca: {e}")
        return None


def get_current_positions(client: TradingClient) -> Dict[str, float]:
    """
    Get current positions from Alpaca.
    
    Returns:
        Dict mapping ticker -> quantity held
    """
    try:
        positions = client.get_all_positions()
        return {pos.symbol: float(pos.qty) for pos in positions}
    except Exception as e:
        print(f"⚠️ Error fetching positions: {e}")
        return {}


def place_order(
    client: TradingClient,
    ticker: str,
    qty: int,
    side: OrderSide
) -> bool:
    """Place a market order on Alpaca."""
    if not LIVE_TRADING_ENABLED:
        print(f"  ℹ️  [DRY-RUN] Would {side.value} {qty} shares of {ticker}")
        return True
    
    if qty <= 0:
        print(f"  ⚠️ Cannot place order for {ticker}: qty={qty}")
        return False
    
    try:
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_request)
        print(f"  ✅ {side.value} {qty} shares of {ticker} (Order ID: {order.id})")
        return True
    except APIError as e:
        print(f"  ❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def rebalance_portfolio(
    client: TradingClient,
    target_tickers: List[str],
    current_positions: Dict[str, float],
    investment_per_stock: float
) -> None:
    """
    Rebalance portfolio to match target tickers.
    
    This matches the backtest logic:
    - Hold exactly TOP_N_STOCKS (default 3)
    - Each position sized at INVESTMENT_PER_STOCK
    - Sell stocks not in top 3
    - Buy stocks in top 3 that we don't hold
    """
    print("\n📊 Portfolio Rebalancing:")
    print(f"  Target: {target_tickers}")
    print(f"  Current: {list(current_positions.keys())}")
    
    # Determine actions
    tickers_to_buy = set(target_tickers) - set(current_positions.keys())
    tickers_to_sell = set(current_positions.keys()) - set(target_tickers)
    tickers_to_hold = set(target_tickers) & set(current_positions.keys())
    
    # 1. Sell positions not in target
    for ticker in tickers_to_sell:
        qty = int(current_positions[ticker])
        if qty > 0:
            print(f"\n  🔴 SELL {ticker} (not in top {TOP_N_STOCKS})")
            place_order(client, ticker, qty, OrderSide.SELL)
    
    # 2. Hold existing positions in target
    for ticker in tickers_to_hold:
        print(f"\n  ⚪ HOLD {ticker} ({int(current_positions[ticker])} shares)")
    
    # 3. Buy new positions in target
    for ticker in tickers_to_buy:
        # Get latest price to calculate quantity
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            latest_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            
            if latest_price is None or latest_price <= 0:
                print(f"\n  ⚠️ Could not get price for {ticker}. Skipping.")
                continue
            
            qty = int(investment_per_stock / latest_price)
            
            if qty > 0:
                print(f"\n  🟢 BUY {ticker} (${latest_price:.2f}/share)")
                place_order(client, ticker, qty, OrderSide.BUY)
            else:
                print(f"\n  ⚠️ Cannot buy {ticker}: price ${latest_price:.2f} too high (qty would be 0)")
        
        except Exception as e:
            print(f"\n  ❌ Error processing {ticker}: {e}")


def get_strategy_tickers(strategy: str, all_tickers: List[str]) -> List[str]:
    """Get the tickers to hold based on the selected strategy."""

    if strategy == 'ai':
        # AI Strategy: Use model predictions (existing logic)
        return get_ai_strategy_tickers(all_tickers)

    elif strategy.startswith('dynamic_bh_'):
        # Dynamic BH Strategy: Select based on performance period
        period = strategy.replace('dynamic_bh_', '')  # '1y', '3m', or '1m'
        return get_dynamic_bh_tickers(all_tickers, period)

    elif strategy == 'risk_adj_mom':
        # Risk-Adjusted Momentum Strategy
        return get_risk_adj_mom_tickers(all_tickers)

    elif strategy == 'mean_reversion':
        # Mean Reversion Strategy
        return get_mean_reversion_tickers(all_tickers)

    elif strategy == 'static_bh':
        # Static BH: Use the same top performers from backtest
        print("📊 Static BH: Using top performers from most recent backtest")
        # This would need to load from saved backtest results
        # For now, fall back to AI strategy
        print("⚠️  Static BH not fully implemented yet, using AI strategy")
        return get_ai_strategy_tickers(all_tickers)

    else:
        print(f"❌ Unknown strategy: {strategy}")
        return []


def get_ai_strategy_tickers(all_tickers: List[str]) -> List[str]:
    """AI Strategy: Rank by predicted return (existing logic)."""
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   Run 'python src/main.py' first to train models")
        return []

    print("🤖 Loading trained models...")
    current_time = datetime.now(timezone.utc)
    max_age = timedelta(days=MODEL_MAX_AGE_DAYS)

    valid_tickers = []
    for ticker in all_tickers:
        model_path = models_dir / f"{ticker}_model.joblib"
        if model_path.exists():
            model_age = current_time - datetime.fromtimestamp(model_path.stat().st_mtime, timezone.utc)
            if model_age <= max_age:
                valid_tickers.append(ticker)

    if len(valid_tickers) < TOP_N_STOCKS:
        print(f"⚠️  Only {len(valid_tickers)} tickers have recent models (need {TOP_N_STOCKS})")
        return []

    print(f"✅ Found {len(valid_tickers)} tickers with recent models")

    # Load models and make predictions
    print("🔮 Making predictions...")
    ranked_predictions = rank_tickers_by_predicted_return(valid_tickers)
    if not ranked_predictions:
        print("❌ No valid predictions generated")
        return []

    target_tickers = [ticker for ticker, _ in ranked_predictions[:TOP_N_STOCKS]]
    return target_tickers


def get_dynamic_bh_tickers(all_tickers: List[str], period: str) -> List[str]:
    """Dynamic BH Strategy: Select top performers based on specified period."""
    print(f"📊 Dynamic BH ({period}): Selecting top {TOP_N_STOCKS} performers...")

    # This would implement the same logic as the backtest
    # For now, use a simplified version
    performances = []
    for ticker in all_tickers[:50]:  # Limit to avoid too many API calls
        try:
            # Get data for the appropriate period
            if period == '1y':
                start_date = datetime.now() - timedelta(days=365)
            elif period == '3m':
                start_date = datetime.now() - timedelta(days=90)
            elif period == '1m':
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = datetime.now() - timedelta(days=365)

            df = load_prices_robust(ticker, start_date, datetime.now())
            if df is not None and len(df) >= 10:
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                if start_price > 0:
                    performance = ((end_price - start_price) / start_price) * 100
                    performances.append((ticker, performance))

        except Exception as e:
            continue

    # Sort by performance and get top N
    performances.sort(key=lambda x: x[1], reverse=True)
    target_tickers = [ticker for ticker, _ in performances[:TOP_N_STOCKS]]

    if target_tickers:
        print(f"🏆 Top {TOP_N_STOCKS} performers ({period}): {target_tickers}")
        for i, (ticker, perf) in enumerate(performances[:TOP_N_STOCKS], 1):
            print(".1f")

    return target_tickers


def get_risk_adj_mom_tickers(all_tickers: List[str]) -> List[str]:
    """Risk-Adjusted Momentum Strategy: Select based on risk-adjusted returns."""
    print(f"📊 Risk-Adjusted Momentum: Selecting top {TOP_N_STOCKS}...")

    # Simplified implementation - would need the full risk-adjusted logic
    # For now, use dynamic BH 6-month as approximation
    return get_dynamic_bh_tickers(all_tickers, '6m')


def get_mean_reversion_tickers(all_tickers: List[str]) -> List[str]:
    """Mean Reversion Strategy: Select worst recent performers."""
    print(f"📊 Mean Reversion: Selecting bottom {TOP_N_STOCKS} recent performers...")

    # Get 1-month performance and select WORST performers (opposite of momentum)
    performances = []
    for ticker in all_tickers[:50]:
        try:
            start_date = datetime.now() - timedelta(days=30)
            df = load_prices_robust(ticker, start_date, datetime.now())
            if df is not None and len(df) >= 10:
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                if start_price > 0:
                    performance = ((end_price - start_price) / start_price) * 100
                    performances.append((ticker, performance))

        except Exception as e:
            continue

    # Sort by performance (ascending = worst performers) and get bottom N
    performances.sort(key=lambda x: x[1])  # Ascending order
    target_tickers = [ticker for ticker, _ in performances[:TOP_N_STOCKS]]

    if target_tickers:
        print(f"🔄 Bottom {TOP_N_STOCKS} performers (1-month): {target_tickers}")
        for i, (ticker, perf) in enumerate(performances[:TOP_N_STOCKS], 1):
            print(".1f")

    return target_tickers


def run_live_trading():
    """Main live trading function."""
    print("=" * 80)
    print("🤖 AI STOCK ADVISOR - LIVE TRADING")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

    # Display selected strategy
    strategy_names = {
        'ai': 'AI Predictions',
        'static_bh': 'Static Buy & Hold',
        'dynamic_bh_1y': 'Dynamic BH (1-Year)',
        'dynamic_bh_3m': 'Dynamic BH (3-Month)',
        'dynamic_bh_1m': 'Dynamic BH (1-Month)',
        'risk_adj_mom': 'Risk-Adjusted Momentum',
        'mean_reversion': 'Mean Reversion'
    }

    strategy_name = strategy_names.get(LIVE_TRADING_STRATEGY, LIVE_TRADING_STRATEGY)
    print(f"🎯 Strategy: {strategy_name} → Hold top {TOP_N_STOCKS}")
    print(f"💰 Investment per stock: ${INVESTMENT_PER_STOCK_LIVE:,.2f}")

    mode = "🧪 DRY-RUN" if not LIVE_TRADING_ENABLED else ("📄 PAPER" if USE_PAPER_TRADING else "⚠️  LIVE")
    print(f"🔧 Mode: {mode}")
    print("=" * 80)
    
    # 1. Connect to Alpaca
    client = get_alpaca_client()
    if client is None:
        print("\n❌ Cannot proceed without Alpaca connection")
        return
    
    # 2. Get current positions
    current_positions = get_current_positions(client)
    print(f"\n📊 Current Portfolio: {len(current_positions)} positions")
    for ticker, qty in current_positions.items():
        print(f"  - {ticker}: {int(qty)} shares")
    
    # 3. Get universe of stocks (S&P 500 based on config.py)
    print("\n🔍 Fetching stock universe from config...")
    try:
        all_tickers = get_all_tickers()
        print(f"✅ Found {len(all_tickers)} tickers in universe")
    except Exception as e:
        print(f"❌ Error fetching tickers: {e}")
        return
    
    # 4. Filter to tickers with trained models
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print(f"\n❌ Models directory not found: {models_dir}")
        print("   Run 'python src/main.py' first to train models")
        return
    
    print(f"\n🤖 Loading trained models from {models_dir}...")
    current_time = datetime.now(timezone.utc)
    max_age = timedelta(days=MODEL_MAX_AGE_DAYS)
    
    valid_tickers = []
    for ticker in all_tickers:
        model_path = models_dir / f"{ticker}_model_buy.joblib"
        if model_path.exists():
            model_age = current_time - datetime.fromtimestamp(model_path.stat().st_mtime, timezone.utc)
            if model_age <= max_age:
                valid_tickers.append(ticker)
    
    print(f"✅ Found {len(valid_tickers)} tickers with trained models (< {MODEL_MAX_AGE_DAYS} days old)")
    
    if len(valid_tickers) == 0:
        print("\n❌ No valid models found. Train models first: python src/main.py")
        return
    
    # 5. Load models
    print("\n📦 Loading models, scalers, and feature sets...")
    models_buy, scalers, y_scalers = load_models_for_tickers(valid_tickers, models_dir)
    
    # Get feature set from first ticker
    feature_set = None
    for ticker in valid_tickers:
        feature_set = get_feature_set_from_saved_model(ticker, models_dir)
        if feature_set is not None:
            break
    
    if feature_set is None:
        print("❌ Could not determine feature set from models")
        return
    
    print(f"✅ Loaded {len(models_buy)} models")
    print(f"✅ Feature set: {len(feature_set)} features")
    
    # 6. Download recent data for all tickers
    print(f"\n📥 Downloading recent data for {len(valid_tickers)} tickers...")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)  # 1 year of history for features
    
    all_data_dict = {}
    failed_tickers = []

    for idx, ticker in enumerate(valid_tickers, 1):
        # Check if cache exists and is recent
        cache_file = Path("data_cache") / f"{ticker}.csv"
        cache_status = "cached"
        if cache_file.exists():
            try:
                cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
                today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                if (today - cache_mtime).days > 1:
                    cache_status = "updating"
                # else cache_status remains "cached"
            except:
                cache_status = "fetching"
        else:
            cache_status = "fetching"

        if cache_status == "cached":
            print(f"{idx}/{len(valid_tickers)} stocks - Loading {ticker} from cache...")
        else:
            print(f"{idx}/{len(valid_tickers)} stocks - Downloading {ticker} data...")

        try:
            data = load_prices_robust(ticker, start_date, end_date)
            if not data.empty and len(data) > 60:  # Need enough history for features
                all_data_dict[ticker] = data
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            failed_tickers.append(ticker)
    
    print(f"✅ Downloaded data for {len(all_data_dict)} tickers")
    if failed_tickers:
        print(f"⚠️ Skipped {len(failed_tickers)} tickers (no data)")
    
    # Convert to multi-index DataFrame (same format as backtesting)
    if len(all_data_dict) == 0:
        print("\n❌ No data available for any ticker")
        return
    
    all_data = pd.concat(
        {ticker: data for ticker, data in all_data_dict.items()},
        axis=1
    )
    
    # 7. Predict returns and rank tickers
    print(f"\n🔮 Generating predictions for {len(all_data_dict)} tickers...")
    
    ranked_predictions = rank_tickers_by_predicted_return(
        tickers=list(all_data_dict.keys()),
        all_data=all_data,
        models_buy=models_buy,
        scalers=scalers,
        y_scalers=y_scalers,
        feature_set=feature_set,
        horizon_days=CLASS_HORIZON,
        top_n=TOP_N_STOCKS
    )
    
    # 8. Display predictions
    print("\n" + "=" * 80)
    print(f"📊 TOP {TOP_N_STOCKS} STOCKS BY PREDICTED RETURN")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Ticker':<10} | {'Predicted Return':>18}")
    print("-" * 80)
    
    for i, (ticker, pred_return) in enumerate(ranked_predictions, 1):
        print(f"{i:<6} | {ticker:<10} | {pred_return:>17.2%}")
    
    print("=" * 80)
    
    # 9. Rebalance portfolio
    target_tickers = [ticker for ticker, _ in ranked_predictions]
    
    rebalance_portfolio(
        client,
        target_tickers,
        current_positions,
        INVESTMENT_PER_STOCK_LIVE
    )
    
    # 10. Summary
    print("\n" + "=" * 80)
    print("✅ LIVE TRADING COMPLETE")
    print("=" * 80)
    print(f"📊 Portfolio should now hold: {target_tickers}")
    print(f"🔄 Check Alpaca dashboard: https://app.alpaca.markets/{'paper' if USE_PAPER_TRADING else 'live'}/dashboard")
    print("=" * 80)


if __name__ == "__main__":
    run_live_trading()


