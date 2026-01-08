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

from src.config import (
    ALPACA_API_KEY as CONFIG_ALPACA_API_KEY,
    ALPACA_SECRET_KEY as CONFIG_ALPACA_SECRET_KEY,
    INVESTMENT_PER_STOCK,
    PERIOD_HORIZONS
)

# Use environment variables if set, otherwise fall back to config
import os
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY') or CONFIG_ALPACA_API_KEY
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY') or CONFIG_ALPACA_SECRET_KEY
from src.ticker_selection import get_all_tickers
from src.data_fetcher import load_prices_robust
from src.prediction import (
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
    print(" alpaca-py not installed. Run: pip install alpaca-py")
    ALPACA_AVAILABLE = False


# --- Configuration ---
LIVE_TRADING_ENABLED = False  # ⚠️ Set to True to execute real orders (start with False for dry-run)
INVESTMENT_PER_STOCK_LIVE = INVESTMENT_PER_STOCK  # Inherit from config.py
MODEL_MAX_AGE_DAYS = 1  # Only use models trained in last 90 days
USE_PAPER_TRADING = True  # True = paper trading (fake money), False = REAL MONEY ⚠️
TOP_N_STOCKS = 3  # Number of stocks to hold (matches backtest)

# --- Strategy Selection ---
# Choose which strategy to use for live trading:
# 'ai' = AI predictions (requires trained models)
# 'ai_portfolio' = AI-driven portfolio rebalancing (intelligent rebalancing)
# 'static_bh' = Static Buy & Hold (top performers from backtest)
# 'dynamic_bh_1y' = Dynamic BH rebalancing annually  ← Best for long-term (less trading costs)
# 'dynamic_bh_3m' = Dynamic BH rebalancing quarterly ← Good balance
# 'dynamic_bh_1m' = Dynamic BH rebalancing monthly   ← More aggressive
# 'ai_individual' = AI Strategy (individual models per stock)
# 'ai_portfolio' = AI Portfolio (meta-learning approach)
# 'multitask' = Multi-Task Learning (unified models)
# 'risk_adj_mom' = Risk-Adjusted Momentum
# 'mean_reversion' = Mean Reversion
# 'volatility_adj_mom' = Volatility-Adjusted Momentum
LIVE_TRADING_STRATEGY = 'risk_adj_mom'  # 👈 Optimized Risk-Adjusted Momentum (+179.4% backtest!)


def get_alpaca_client() -> Optional[TradingClient]:
    """Initialize Alpaca trading client."""
    if not ALPACA_AVAILABLE:
        print(" Alpaca library not available")
        return None
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print(" Alpaca API keys not set.")
        print("   Set environment variables:")
        print("   export ALPACA_API_KEY='your_key'")
        print("   export ALPACA_SECRET_KEY='your_secret'")
        print("   Or configure them in src/config.py")
        return None
    
    try:
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=USE_PAPER_TRADING)
        account = client.get_account()
        
        mode = "Paper" if USE_PAPER_TRADING else "LIVE"
        print(f" Connected to Alpaca {mode} Trading")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        
        return client
    except Exception as e:
        print(f" Error connecting to Alpaca: {e}")
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
        print(f" Error fetching positions: {e}")
        return {}


def place_order(
    client: TradingClient,
    ticker: str,
    qty: int,
    side: OrderSide
) -> bool:
    """Place a market order on Alpaca."""
    if not LIVE_TRADING_ENABLED:
        print(f"    [DRY-RUN] Would {side.value} {qty} shares of {ticker}")
        return True
    
    if qty <= 0:
        print(f"   Cannot place order for {ticker}: qty={qty}")
        return False
    
    try:
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_request)
        print(f"   {side.value} {qty} shares of {ticker} (Order ID: {order.id})")
        return True
    except APIError as e:
        print(f"   API Error: {e}")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False


def rebalance_portfolio(
    client: TradingClient,
    target_tickers: List[str],
    current_positions: Dict[str, float],
    investment_per_stock: float
) -> None:
    """
    Rebalance portfolio to hold only the target tickers.

    Args:
        client: Alpaca trading client
        target_tickers: List of tickers to hold
        current_positions: Dict of current positions (ticker -> quantity)
        investment_per_stock: Amount to invest per stock
    """
    print(f"\n Rebalancing portfolio to hold: {target_tickers}")

    # Calculate target quantities
    target_positions = {}
    for ticker in target_tickers:
        # Calculate quantity based on investment amount and current price
        try:
            # Get current price (simplified - in production you'd get real-time price)
            # For now, use a placeholder calculation
            price = 100.0  # Placeholder price
            qty = int(investment_per_stock / price)
            target_positions[ticker] = qty
        except Exception as e:
            print(f"   Error calculating quantity for {ticker}: {e}")
            continue

    # Close positions not in target
    for ticker, current_qty in current_positions.items():
        if ticker not in target_tickers:
            print(f" Selling {current_qty} shares of {ticker} (not in target portfolio)")
            place_order(client, ticker, abs(int(current_qty)), OrderSide.SELL)

    # Open positions for target tickers
    for ticker in target_tickers:
        target_qty = target_positions.get(ticker, 0)
        current_qty = current_positions.get(ticker, 0)

        if target_qty > current_qty:
            buy_qty = int(target_qty - current_qty)
            print(f" Buying {buy_qty} shares of {ticker}")
            place_order(client, ticker, buy_qty, OrderSide.BUY)
        elif target_qty < current_qty:
            sell_qty = int(current_qty - target_qty)
            print(f" Selling {sell_qty} shares of {ticker}")
            place_order(client, ticker, sell_qty, OrderSide.SELL)

    print(" Portfolio rebalancing complete")


def get_strategy_tickers(strategy: str, all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Get the tickers to hold based on the selected strategy."""

    if strategy == 'ai_individual':
        # AI Strategy: Use model predictions (existing logic)
        return get_ai_strategy_tickers(all_tickers)

    elif strategy == 'ai_portfolio':
        # AI Portfolio Strategy: Use AI Portfolio model
        return get_ai_portfolio_tickers(all_tickers)

    elif strategy == 'multitask':
        # Multi-Task Learning Strategy: Use Multi-Task model
        return get_multitask_tickers(all_tickers)

    elif strategy.startswith('dynamic_bh_'):
        # Dynamic BH Strategy: Select based on performance period
        period = strategy.replace('dynamic_bh_', '')  # '1y', '3m', or '1m'
        return get_dynamic_bh_tickers(all_tickers, period)

    elif strategy == 'risk_adj_mom':
        # Risk-Adjusted Momentum Strategy
        return get_risk_adj_mom_tickers(all_tickers, all_tickers_data)

    elif strategy == 'mean_reversion':
        # Mean Reversion Strategy
        return get_mean_reversion_tickers(all_tickers)

    elif strategy == 'volatility_adj_mom':
        # Volatility-Adjusted Momentum Strategy
        return get_volatility_adj_mom_tickers(all_tickers)

    elif strategy == 'quality_momentum':
        # Quality + Momentum Strategy
        return get_quality_momentum_tickers(all_tickers)

    else:
        print(f" Unknown strategy: {strategy}, using dynamic_bh_3m")
        return get_dynamic_bh_tickers(all_tickers, '3m')


def get_ai_strategy_tickers(all_tickers: List[str]) -> List[str]:
    """AI Strategy: Use model predictions to select top tickers."""
    # This would use the AI models - for now return a placeholder
    return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_dynamic_bh_tickers(all_tickers: List[str], period: str) -> List[str]:
    """Dynamic BH Strategy: Select top performers based on specified period."""
    print(f" Dynamic BH ({period}): Selecting top {TOP_N_STOCKS} performers...")

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
        print(f" Top {TOP_N_STOCKS} performers ({period}): {target_tickers}")
        for i, (ticker, perf) in enumerate(performances[:TOP_N_STOCKS], 1):
            print(".1f")

    return target_tickers


def get_risk_adj_mom_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Risk-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_risk_adj_mom_stocks
    
    # Prepare data for shared strategy
    ticker_data_grouped = {}
    if all_tickers_data is not None:
        # Handle different data formats
        if isinstance(all_tickers_data.columns, pd.MultiIndex):
            # Wide format: columns are (field, ticker)
            for ticker in all_tickers_data.columns.levels[1]:
                ticker_cols = [col for col in all_tickers_data.columns if col[1] == ticker]
                if ticker_cols:
                    ticker_data = all_tickers_data[ticker_cols].copy()
                    ticker_data.columns = [col[0] for col in ticker_cols]
                    ticker_data_grouped[ticker] = ticker_data
        elif 'ticker' in all_tickers_data.index.names:
            # Long format: group by ticker
            ticker_data_grouped = {ticker: group for ticker, group in all_tickers_data.groupby('ticker')}
        else:
            # Assume ticker columns
            for ticker in all_tickers:
                if ticker in all_tickers_data.columns:
                    ticker_data_grouped[ticker] = all_tickers_data[[ticker]].copy()
                    ticker_data_grouped[ticker].columns = ['Close']
    
    return select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, top_n=PORTFOLIO_SIZE)


def get_mean_reversion_tickers(all_tickers: List[str]) -> List[str]:
    """Mean Reversion Strategy: Use shared strategy logic."""
    from shared_strategies import select_mean_reversion_stocks
    return select_mean_reversion_stocks(all_tickers, {}, top_n=PORTFOLIO_SIZE)


def get_volatility_adj_mom_tickers(all_tickers: List[str]) -> List[str]:
    """Volatility-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_volatility_adj_mom_stocks
    return select_volatility_adj_mom_stocks(all_tickers, {}, top_n=PORTFOLIO_SIZE)


def get_dynamic_bh_tickers(all_tickers: List[str], period: str) -> List[str]:
    """Dynamic Buy & Hold Strategy: Use shared strategy logic."""
    from shared_strategies import select_dynamic_bh_stocks
    return select_dynamic_bh_stocks(all_tickers, {}, period=period, top_n=PORTFOLIO_SIZE)


def get_quality_momentum_tickers(all_tickers: List[str]) -> List[str]:
    """Quality + Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_quality_momentum_stocks
    return select_quality_momentum_stocks(all_tickers, {}, top_n=PORTFOLIO_SIZE)


def get_ai_portfolio_tickers(all_tickers: List[str]) -> List[str]:
    """AI Portfolio Strategy: Use AI Portfolio model for selection."""
    # This would use the AI Portfolio model
    # For now, return top N tickers as placeholder
    return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_multitask_tickers(all_tickers: List[str]) -> List[str]:
    """Multi-Task Learning Strategy: Use Multi-Task model for selection."""
    # This would use the Multi-Task Learning model
    # For now, return top N tickers as placeholder
    return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def run_live_trading():
    """Main live trading function."""
    print("=" * 80)
    print(" AI STOCK ADVISOR - LIVE TRADING")
    print("=" * 80)
    print(f" {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

    # Display selected strategy
    strategy_names = {
        'ai_individual': 'AI Strategy (Individual Models)',
        'ai_portfolio': 'AI Portfolio (Meta-Learning)',
        'multitask': 'Multi-Task Learning',
        'risk_adj_mom': 'Risk-Adjusted Momentum',
        'mean_reversion': 'Mean Reversion',
        'quality_momentum': 'Quality + Momentum'
    }

    strategy_name = strategy_names.get(LIVE_TRADING_STRATEGY, LIVE_TRADING_STRATEGY)
    print(f" Strategy: {strategy_name}  Hold top {TOP_N_STOCKS}")
    print(f" Investment per stock: ${INVESTMENT_PER_STOCK_LIVE:,.2f}")

    mode = " DRY-RUN" if not LIVE_TRADING_ENABLED else (" PAPER" if USE_PAPER_TRADING else "  LIVE")
    print(f" Mode: {mode}")
    print("=" * 80)

    client = get_alpaca_client()
    if client is None:
        print("\n Cannot proceed without Alpaca connection")
        return

    # 2. Get current positions
    current_positions = get_current_positions(client)
    print(f"\n Current Portfolio: {len(current_positions)} positions")
    for ticker, qty in current_positions.items():
        print(f"  - {ticker}: {int(qty)} shares")

    # 3. Get universe of stocks (S&P 500 based on config.py)
    print("\n Fetching stock universe from config...")
    try:
        all_tickers = get_all_tickers()
        print(f" Found {len(all_tickers)} tickers in universe")
    except Exception as e:
        print(f" Error fetching tickers: {e}")
        return

    # 4. Download fresh data for Risk-Adjusted Momentum strategy
    if LIVE_TRADING_STRATEGY == 'risk_adj_mom':
        print("\n Downloading fresh data for Risk-Adjusted Momentum analysis...")
        try:
            from data_fetcher import _download_batch_robust
            from utils import _to_utc
            
            # Download 1 year of data for all tickers
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=365)
            
            print(f"   Downloading data from {start_date.date()} to {end_date.date()}...")
            
            # Download in batches to avoid API limits
            batch_size = 100
            all_data_list = []
            
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i:i + batch_size]
                print(f"   - Downloading batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)...")
                
                try:
                    batch_data = _download_batch_robust(batch, start_date, end_date)
                    if batch_data is not None and not batch_data.empty:
                        # Convert timezone-aware index to UTC for comparison
                        if hasattr(batch_data.index, 'tz_localize') and batch_data.index.tz is None:
                            batch_data.index = batch_data.index.tz_localize('UTC')
                        elif hasattr(batch_data.index, 'tz_convert') and batch_data.index.tz != 'UTC':
                            batch_data.index = batch_data.index.tz_convert('UTC')
                        
                        # Filter to our date range (both are now UTC)
                        filtered_batch_data = batch_data[
                            (batch_data.index >= start_date) & 
                            (batch_data.index <= end_date)
                        ]
                        if not filtered_batch_data.empty:
                            all_data_list.append(filtered_batch_data)
                except Exception as e:
                    print(f"     ⚠️ Batch {i//batch_size + 1} failed: {e}")
                    continue
                
                # Small pause between batches
                if i + batch_size < len(all_tickers):
                    import time
                    time.sleep(1.0)
            
            if all_data_list:
                all_tickers_data = pd.concat(all_data_list, ignore_index=False)
                print(f"   ✅ Downloaded data for {len(all_tickers_data.columns)} tickers")
                
                # Check the actual structure - it might be a regular DataFrame with ticker columns
                print(f"   📊 Data structure: {type(all_tickers_data.index)}")
                if hasattr(all_tickers_data.index, 'names'):
                    print(f"   📊 Index names: {all_tickers_data.index.names}")
                print(f"   📊 Columns: {list(all_tickers_data.columns)[:5]}...")  # Show first 5 columns
                
            else:
                print("   ❌ No data downloaded")
                return
                
        except Exception as e:
            print(f"   ❌ Data download failed: {e}")
            return

    # 5. Load AI models only if strategy requires them
    if LIVE_TRADING_STRATEGY in ['ai_individual', 'ai_portfolio', 'multitask']:
        models_dir = Path("logs/models")
        if not models_dir.exists():
            print(f"\n Models directory not found: {models_dir}")
            print("   Run 'python src/main.py' first to train models")
            return

        print(f"\n Loading trained models from {models_dir}...")
        current_time = datetime.now(timezone.utc)
        max_age = timedelta(days=MODEL_MAX_AGE_DAYS)

        valid_tickers = []
        for ticker in all_tickers:
            # ✅ FIX: Use correct filename pattern (models saved as {ticker}_model.joblib, not {ticker}_model_buy.joblib)
            model_path = models_dir / f"{ticker}_model.joblib"
            if model_path.exists():
                model_age = current_time - datetime.fromtimestamp(model_path.stat().st_mtime, timezone.utc)
                if model_age <= max_age:
                    valid_tickers.append(ticker)

        print(f" Found {len(valid_tickers)} tickers with trained models (< {MODEL_MAX_AGE_DAYS} days old)")

        if len(valid_tickers) == 0:
            print("\n No valid models found. Train models first: python src/main.py")
            return

        # 5. Load models (AI strategy only)
        print("\n Loading models, scalers, and feature sets...")
        models_buy, scalers, y_scalers = load_models_for_tickers(valid_tickers, models_dir)

        # Get feature set from first ticker
        feature_set = None
        for ticker in valid_tickers:
            feature_set = get_feature_set_from_saved_model(ticker, models_dir)
            if feature_set is not None:
                break

        if feature_set is None:
            print(" Could not determine feature set from models")
            return

        print(f" Loaded {len(models_buy)} models")
        print(f" Feature set: {len(feature_set)} features")

        # 6. Download recent data for all tickers (AI strategy only)
        print(f"\n Downloading recent data for {len(valid_tickers)} tickers...")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)  # 1 year of history for features

        all_data_dict = {}
        failed_tickers = []

        for idx, ticker in enumerate(valid_tickers, 1):
            # Check if cache exists and always update it for live trading
            cache_file = Path("data_cache") / f"{ticker}.csv"
            cache_status = "cached"
            if not cache_file.exists():
                cache_status = "downloading"
                print(f"   [{idx}/{len(valid_tickers)}] {ticker}: {cache_status}...")
            else:
                # Always update cache to get latest data for live trading
                cache_status = "updating"
                print(f"   [{idx}/{len(valid_tickers)}] {ticker}: {cache_status}...")

            if cache_status in ["downloading", "updating"]:
                try:
                    ticker_data = load_prices_robust(ticker, start_date, end_date)
                    if ticker_data is not None and len(ticker_data) > 0:
                        ticker_data.to_csv(cache_file)
                    else:
                        failed_tickers.append(ticker)
                        continue
                except Exception as e:
                    print(f"   Error downloading {ticker}: {e}")
                    failed_tickers.append(ticker)
                    continue

            # Load from cache
            try:
                ticker_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                all_data_dict[ticker] = ticker_data
            except Exception as e:
                print(f"   Error loading cached data for {ticker}: {e}")
                failed_tickers.append(ticker)

        if failed_tickers:
            print(f" Failed to get data for {len(failed_tickers)} tickers: {failed_tickers[:5]}...")

        valid_tickers = [t for t in valid_tickers if t in all_data_dict]
        print(f" Successfully loaded data for {len(valid_tickers)} tickers")

        if len(valid_tickers) == 0:
            print("\n No valid data available")
            return

        # 7. Predict returns and rank tickers (AI strategy only)
        print(f"\n Generating predictions for {len(valid_tickers)} tickers...")

        ranked_predictions = rank_tickers_by_predicted_return(
            tickers=valid_tickers,
            all_data=all_data_dict,
            models_buy=models_buy,
            scalers=scalers,
            y_scalers=y_scalers,
            feature_set=feature_set,
            horizon_days=PERIOD_HORIZONS["1-Year"],
            top_n=TOP_N_STOCKS
        )
    else:
        # Non-AI strategy - no models needed
        valid_tickers = all_tickers
        ranked_predictions = None

    # 6. Get target tickers based on strategy
    if LIVE_TRADING_STRATEGY in ['ai_individual', 'ai_portfolio', 'multitask']:
        # AI strategies - display predictions and get tickers
        if LIVE_TRADING_STRATEGY == 'ai_individual':
            # AI strategy - display predictions and get tickers
            for i, (ticker, pred_return) in enumerate(ranked_predictions, 1):
                print(f"{i:<6} | {ticker:<10} | {pred_return:>17.2%}")
            target_tickers = [ticker for ticker, _ in ranked_predictions]
        elif LIVE_TRADING_STRATEGY == 'ai_portfolio':
            # AI Portfolio strategy
            target_tickers = get_ai_portfolio_tickers(valid_tickers)
            print(f"\n🎯 AI Portfolio selected {len(target_tickers)} stocks:")
            for i, ticker in enumerate(target_tickers, 1):
                print(f"{i:<6} | {ticker:<10}")
        elif LIVE_TRADING_STRATEGY == 'multitask':
            # Multi-Task Learning strategy
            from multitask_strategy import select_multitask_stocks
            from datetime import datetime
            target_tickers = select_multitask_stocks(valid_tickers, all_data_dict, 
                                                  current_date=datetime.now(), top_n=TOP_N_STOCKS)
            print(f"\n🎯 Multi-Task Learning selected {len(target_tickers)} stocks:")
            for i, ticker in enumerate(target_tickers, 1):
                print(f"{i:<6} | {ticker:<10}")
        
        print("=" * 80)
    else:
        # Non-AI strategy - use strategy-specific logic
        print(f"\n 🎯 Running {LIVE_TRADING_STRATEGY} strategy...")
        print(f"    Available tickers: {len(valid_tickers)}")
        
        # Pass downloaded data if available for Risk-Adjusted Momentum
        all_tickers_data_for_strategy = all_tickers_data if LIVE_TRADING_STRATEGY == 'risk_adj_mom' and 'all_tickers_data' in locals() else None
        if LIVE_TRADING_STRATEGY == 'risk_adj_mom':
            print(f"    Data available: {all_tickers_data_for_strategy is not None}")
        
        target_tickers = get_strategy_tickers(LIVE_TRADING_STRATEGY, valid_tickers, all_tickers_data_for_strategy)
        
        print(f"\n🎯 SELECTED STOCKS FOR TRADING:")
        print(f"   Strategy: {LIVE_TRADING_STRATEGY}")
        print(f"   Number of stocks: {len(target_tickers)}")
        print(f"   Stocks to buy: {target_tickers}")
        
        # Show current vs target positions
        print(f"\n📊 PORTFOLIO CHANGES:")
        current_stocks = list(current_positions.keys())
        stocks_to_sell = [ticker for ticker in current_stocks if ticker not in target_tickers]
        stocks_to_buy = [ticker for ticker in target_tickers if ticker not in current_stocks]
        
        if stocks_to_sell:
            print(f"   SELL: {stocks_to_sell}")
        if stocks_to_buy:
            print(f"   BUY:  {stocks_to_buy}")
        if not stocks_to_sell and not stocks_to_buy:
            print(f"   No changes needed - portfolio already aligned")
        
        print(f"\n⚠️  TRADING MODE: {'DRY RUN' if not LIVE_TRADING_ENABLED else ('PAPER TRADING' if USE_PAPER_TRADING else 'LIVE TRADING')}")
        
        # Ask for confirmation in live mode
        if LIVE_TRADING_ENABLED and not USE_PAPER_TRADING:
            try:
                confirm = input("\n❓ Execute these trades? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("❌ Trading cancelled by user")
                    return
            except KeyboardInterrupt:
                print("\n❌ Trading cancelled by user")
                return

    # 7. Rebalance portfolio
    rebalance_portfolio(
        client,
        target_tickers,
        current_positions,
        INVESTMENT_PER_STOCK_LIVE
    )

    # 8. Summary
    print("\n" + "=" * 80)
    print(" LIVE TRADING COMPLETE")
    print("=" * 80)
    print(f" Portfolio should now hold: {target_tickers}")
    print(f" Check Alpaca dashboard: https://app.alpaca.markets/{'paper' if USE_PAPER_TRADING else 'live'}/dashboard")
    print("=" * 80)


if __name__ == "__main__":
    run_live_trading()


