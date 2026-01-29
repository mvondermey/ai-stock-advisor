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

import src.config as config
from src.config import (
    ALPACA_API_KEY as CONFIG_ALPACA_API_KEY,
    ALPACA_SECRET_KEY as CONFIG_ALPACA_SECRET_KEY,
    INVESTMENT_PER_STOCK,
    PERIOD_HORIZONS,
    PORTFOLIO_SIZE,
    N_TOP_TICKERS,
    LIVE_TRADING_ENABLED,
    MODEL_MAX_AGE_DAYS,
    USE_PAPER_TRADING,
    TOP_N_STOCKS
)

# Use environment variables if set, otherwise fall back to config
import os
ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY') or CONFIG_ALPACA_API_KEY
ALPACA_SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY') or CONFIG_ALPACA_SECRET_KEY
from src.ticker_selection import get_all_tickers
from src.data_utils import load_prices_robust
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
# All parameters are imported from config.py

# --- Strategy Selection ---
# Choose which strategy to use for live trading:
# 'ai' = AI predictions (requires trained models)
# 'static_bh' = Static Buy & Hold (top performers from backtest)
# 'dynamic_bh_1y' = Dynamic BH rebalancing annually  ← Best for long-term (less trading costs)
# 'dynamic_bh_6m' = Dynamic BH rebalancing semi-annually ← Good balance
# 'dynamic_bh_3m' = Dynamic BH rebalancing quarterly ← Good balance
# 'dynamic_bh_1m' = Dynamic BH rebalancing monthly   ← More aggressive
# 'ai_individual' = AI Strategy (individual models per stock)
# 'multitask' = Multi-Task Learning (unified models)
# 'risk_adj_mom' = Risk-Adjusted Momentum
# 'mean_reversion' = Mean Reversion
# 'volatility_adj_mom' = Volatility-Adjusted Momentum
def get_live_trading_strategy():
    """Get the live trading strategy from config (set dynamically by main.py)."""
    return getattr(config, 'LIVE_TRADING_STRATEGY', 'risk_adj_mom')


def _prepare_ticker_data_grouped(all_tickers: List[str], all_tickers_data: pd.DataFrame, strategy_name: str = "Strategy") -> dict:
    """
    Prepare ticker data grouped by ticker with date as index.
    This is a shared helper for all strategies to ensure consistent data format.
    """
    ticker_data_grouped = {}
    if all_tickers_data is None:
        return ticker_data_grouped
    
    print(f"   🔍 {strategy_name}: Data shape: {all_tickers_data.shape}")
    print(f"   🔍 {strategy_name}: Columns: {list(all_tickers_data.columns[:5])}")
    
    # Debug: Check unique tickers in data
    if 'ticker' in all_tickers_data.columns:
        unique_tickers = all_tickers_data['ticker'].unique()
        print(f"   🔍 {strategy_name}: Unique tickers in data: {len(unique_tickers)}")
        # Check if requested tickers are in data
        matching = [t for t in all_tickers[:5] if t in unique_tickers]
        print(f"   🔍 {strategy_name}: First 5 requested tickers in data: {matching}")
    
    if isinstance(all_tickers_data.columns, pd.MultiIndex):
        # Wide format: columns are (field, ticker)
        print(f"   🔍 {strategy_name}: Using MultiIndex format")
        for ticker in all_tickers_data.columns.levels[1]:
            ticker_cols = [col for col in all_tickers_data.columns if col[1] == ticker]
            if ticker_cols:
                ticker_data = all_tickers_data[ticker_cols].copy()
                ticker_data.columns = [col[0] for col in ticker_cols]
                ticker_data_grouped[ticker] = ticker_data
    elif 'ticker' in all_tickers_data.columns:
        # Long format: group by ticker, set date as index
        print(f"   🔍 {strategy_name}: Using long format (ticker in columns)")
        for ticker, group in all_tickers_data.groupby('ticker'):
            group_copy = group.copy()
            if 'date' in group_copy.columns:
                group_copy['date'] = pd.to_datetime(group_copy['date'])
                group_copy = group_copy.set_index('date')
            ticker_data_grouped[ticker] = group_copy
    elif 'ticker' in all_tickers_data.index.names:
        # Long format: ticker in index, ensure date is index
        print(f"   🔍 {strategy_name}: Using long format (ticker in index)")
        for ticker, group in all_tickers_data.groupby('ticker'):
            group_copy = group.copy()
            if 'date' in group_copy.columns:
                group_copy['date'] = pd.to_datetime(group_copy['date'])
                group_copy = group_copy.set_index('date')
            ticker_data_grouped[ticker] = group_copy
    else:
        # Assume ticker columns
        print(f"   🔍 {strategy_name}: Using ticker columns format")
        for ticker in all_tickers:
            if ticker in all_tickers_data.columns:
                ticker_data_grouped[ticker] = all_tickers_data[[ticker]].copy()
                ticker_data_grouped[ticker].columns = ['Close']
    
    print(f"   🔍 {strategy_name}: Prepared {len(ticker_data_grouped)} ticker data groups")
    return ticker_data_grouped


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
    
    if not target_tickers:
        print(f"   ⚠️  NO TARGET TICKERS SELECTED!")
        print(f"   💡 Possible reasons:")
        print(f"      - No stocks met the Risk-Adjusted Momentum criteria (min score: 30.0)")
        print(f"      - Insufficient data for momentum calculation")
        print(f"      - Data format issue")
        print(f"   ℹ️  Skipping rebalancing - portfolio unchanged")
        return
    
    # Get account information
    try:
        account = client.get_account()
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        print(f"   💰 Account Info:")
        print(f"      Cash: ${cash:,.2f}")
        print(f"      Buying Power: ${buying_power:,.2f}")
        print(f"      Portfolio Value: ${portfolio_value:,.2f}")
        print(f"      Investment per stock: ${investment_per_stock:,.2f}")
        print(f"      Total needed for {len(target_tickers)} stocks: ${investment_per_stock * len(target_tickers):,.2f}")
        
        if buying_power < investment_per_stock * len(target_tickers):
            print(f"   ⚠️  INSUFFICIENT FUNDS: Need ${investment_per_stock * len(target_tickers):,.2f}, only have ${buying_power:,.2f}")
            print(f"   💡 Consider reducing INVESTMENT_PER_STOCK")
    except Exception as e:
        print(f"   ⚠️ Could not get account info: {e}")

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
            print(f"   📊 {ticker}: ${investment_per_stock:,.2f} → {qty} shares @ ${price:.2f}")
        except Exception as e:
            print(f"   Error calculating quantity for {ticker}: {e}")
            continue

    # Get all open orders once and group by ticker
    orders_by_ticker = {}
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        # Get ALL orders (not just 'open') to catch pending orders
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(filter=request)
        print(f"   📋 Found {len(orders)} open orders")
        for order in orders:
            if order.symbol not in orders_by_ticker:
                orders_by_ticker[order.symbol] = []
            orders_by_ticker[order.symbol].append(order)
        if orders_by_ticker:
            print(f"   📋 Active orders for {len(orders_by_ticker)} tickers")
    except Exception as e:
        print(f"   Could not get orders: {e}")

    # Close positions not in target
    for ticker, current_qty in current_positions.items():
        if ticker not in target_tickers:
            # Check if there are existing orders for this ticker
            if ticker in orders_by_ticker:
                open_orders = orders_by_ticker[ticker]
                print(f" ⏳ Skipping {ticker}: {len(open_orders)} open order(s) still pending")
                for order in open_orders:
                    print(f"      Order {order.id}: {order.side.value} {order.qty} shares (status: {order.status})")
                continue
            
            # Check available qty from position before selling
            try:
                position = client.get_open_position(ticker)
                qty_available = float(position.qty_available) if hasattr(position, 'qty_available') else float(position.qty)
                if qty_available <= 0:
                    print(f" ⏳ Skipping {ticker}: 0 shares available (held for pending orders)")
                    continue
                sell_qty = min(abs(int(current_qty)), int(qty_available))
                print(f" Selling {sell_qty} shares of {ticker} (not in target portfolio)")
                place_order(client, ticker, sell_qty, OrderSide.SELL)
            except Exception as e:
                print(f" ⚠️ Could not check position for {ticker}: {e}")

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
    print(f"   🎯 DEBUG: Strategy passed = '{strategy}'")

    if strategy == 'ai_individual':
        # AI Strategy: Use model predictions (existing logic)
        return get_ai_strategy_tickers(all_tickers)

    elif strategy == 'multitask':
        # Multi-Task Learning Strategy: Use Multi-Task model
        return get_multitask_tickers(all_tickers)

    elif strategy.startswith('dynamic_bh_'):
        # Dynamic BH Strategy: Select based on performance period
        period = strategy.replace('dynamic_bh_', '')  # '1y', '6m', '3m', or '1m'
        return get_dynamic_bh_tickers(all_tickers, period, all_tickers_data)

    elif strategy == 'risk_adj_mom':
        # Risk-Adjusted Momentum Strategy
        return get_risk_adj_mom_tickers(all_tickers, all_tickers_data)

    elif strategy == '3m_1y_ratio':
        # 3M/1Y Ratio Strategy
        return get_3m_1y_ratio_tickers(all_tickers, all_tickers_data)

    elif strategy == 'mean_reversion':
        # Mean Reversion Strategy
        return get_mean_reversion_tickers(all_tickers, all_tickers_data)

    elif strategy == 'volatility_adj_mom':
        # Volatility-Adjusted Momentum Strategy
        return get_volatility_adj_mom_tickers(all_tickers, all_tickers_data)

    elif strategy == 'quality_momentum':
        # Quality + Momentum Strategy
        return get_quality_momentum_tickers(all_tickers, all_tickers_data)

    elif strategy.startswith('static_bh_'):
        # Static BH Strategy: Select based on performance period but hold static
        period = strategy.replace('static_bh_', '')  # '1y', '6m', '3m', or '1m'
        return get_static_bh_tickers(all_tickers, period, all_tickers_data)
    
    elif strategy == 'turnaround':
        # Turnaround Strategy: Low 3Y but high 1Y performance
        return get_turnaround_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'ratio_3m_1y':
        # 3M/1Y Ratio Strategy: Strong 3M momentum vs 1Y performance
        return get_3m_1y_ratio_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'ratio_1y_3m':
        # 1Y/3M Ratio Strategy: Strong 1Y performance but weak 3M (buy on dip)
        return get_ratio_1y_3m_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'momentum_volatility_hybrid':
        # Momentum-Volatility Hybrid Strategy: Combines strong momentum with controlled volatility
        return get_momentum_volatility_hybrid_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'adaptive_ensemble':
        # Adaptive Ensemble Strategy: Meta-ensemble combining multiple strategies
        return get_adaptive_ensemble_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'volatility_ensemble':
        # Volatility-Adjusted Ensemble Strategy: Risk-managed position sizing
        return get_volatility_ensemble_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'ai_volatility_ensemble':
        # AI-Enhanced Volatility Ensemble Strategy: AI-optimized weights and volatility caps
        return get_ai_volatility_ensemble_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'correlation_ensemble':
        # Correlation-Filtered Ensemble Strategy: Diversification-focused
        return get_correlation_ensemble_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'dynamic_pool':
        # Dynamic Strategy Pool Strategy: Rotates strategies based on performance
        return get_dynamic_pool_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'sentiment_ensemble':
        # Sentiment-Enhanced Ensemble Strategy: Incorporates news sentiment
        return get_sentiment_ensemble_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'momentum_breakout':
        # Momentum Breakout Strategy: 52-week high breakouts with volume
        return get_momentum_breakout_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'factor_rotation':
        # Factor Rotation Strategy: Rotates between Value/Growth/Momentum/Quality
        return get_factor_rotation_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'pairs_trading':
        # Pairs Trading Strategy: Statistical arbitrage on correlated pairs
        return get_pairs_trading_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'earnings_momentum':
        # Earnings Momentum (PEAD) Strategy: Post-earnings drift capture
        return get_earnings_momentum_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'insider_trading':
        # Insider Trading Signal Strategy: Follow insider buying patterns
        return get_insider_trading_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'options_sentiment':
        # Options-Based Sentiment Strategy: Put/call ratios and unusual activity
        return get_options_sentiment_tickers(all_tickers, all_tickers_data)
    
    elif strategy == 'ml_ensemble':
        # ML Ensemble Strategy: Weighted voting from multiple ML models
        return get_ml_ensemble_tickers(all_tickers, all_tickers_data)

    else:
        print(f" Unknown strategy: {strategy}, using dynamic_bh_3m")
        return get_dynamic_bh_tickers(all_tickers, '3m')


def get_ai_strategy_tickers(all_tickers: List[str]) -> List[str]:
    """AI Strategy: Use the EXACT same logic as backtesting for consistency."""
    try:
        from prediction import load_models_for_tickers
        from data_utils import load_prices_robust, fetch_training_data
        from datetime import datetime, timedelta
        import numpy as np
        import pandas as pd
        
        print(f"   🤖 AI Strategy: Loading models for {len(all_tickers)} tickers...")
        models, scalers, y_scalers = load_models_for_tickers(all_tickers)
        
        if not models:
            print(f"   ⚠️ AI Strategy: No models available, falling back to top {TOP_N_STOCKS} tickers")
            return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers
        
        print(f"   🤖 AI Strategy: Making predictions using {len(models)} models...")
        predictions = []
        
        # Load SPY data for Market_Momentum_SPY feature (same as backtesting)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        spy_df = load_prices_robust('SPY', start_date, end_date)
        
        if not spy_df.empty:
            spy_df['SPY_Returns'] = spy_df['Close'].pct_change(fill_method=None)
            spy_df['Market_Momentum_SPY'] = spy_df['SPY_Returns'].rolling(window=20).mean()  # Same window as backtesting
            spy_df = spy_df[['Market_Momentum_SPY']].reset_index()
            spy_df.columns = ['date', 'Market_Momentum_SPY']
            spy_df['date'] = pd.to_datetime(spy_df['date'])
        else:
            print("   ⚠️ Could not fetch SPY data. Market Momentum feature will be 0.")
            # Create dummy SPY data
            spy_df = pd.DataFrame({'date': pd.date_range(start_date, end_date), 'Market_Momentum_SPY': 0.0})
        
        # Use the EXACT same logic as backtesting (lines 2675-2695 in backtesting.py)
        for ticker in all_tickers:
            if ticker in models and models[ticker] is not None:
                try:
                    # Load recent data (same as backtesting approach)
                    ticker_data = load_prices_robust(ticker, start_date, end_date)
                    
                    if ticker_data is not None and not ticker_data.empty:
                        # Reset index to have 'date' column for merging
                        ticker_data = ticker_data.reset_index()
                        
                        # Debug: Print columns to see what we have
                        print(f"   🔍 {ticker}: Columns after reset_index: {list(ticker_data.columns)}")
                        
                        # Handle different index name possibilities
                        if 'date' not in ticker_data.columns:
                            print(f"   🔍 {ticker}: 'date' not in columns, attempting to fix...")
                            # Index might be named differently, rename it to 'date'
                            if len(ticker_data.columns) > 0 and ticker_data.columns[0] in ['index', 'Date', 'datetime']:
                                ticker_data = ticker_data.rename(columns={ticker_data.columns[0]: 'date'})
                                print(f"   🔍 {ticker}: Renamed '{ticker_data.columns[0]}' to 'date'")
                            else:
                                # Index is unnamed, first column should be the date
                                old_cols = list(ticker_data.columns)
                                ticker_data.columns = ['date'] + list(ticker_data.columns[1:])
                                print(f"   🔍 {ticker}: Renamed first column from '{old_cols[0]}' to 'date'")
                        
                        print(f"   🔍 {ticker}: Columns after fix: {list(ticker_data.columns)}")
                        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                        
                        # Merge SPY data (same as backtesting)
                        ticker_data = ticker_data.merge(spy_df, on='date', how='left')
                        ticker_data['Market_Momentum_SPY'] = ticker_data['Market_Momentum_SPY'].ffill().bfill().fillna(0.0)
                        
                        # Set date back as index for feature engineering
                        ticker_data = ticker_data.set_index('date')
                        
                        # Remove 'date' column if it still exists (prevents KeyError: 'date')
                        if 'date' in ticker_data.columns:
                            ticker_data = ticker_data.drop(columns=['date'])
                        
                        # Ensure index name is not 'date' (prevents conflicts in calculations)
                        ticker_data.index.name = None
                        
                        # Ensure index is timezone-aware for prediction
                        if ticker_data.index.tzinfo is None:
                            ticker_data.index = ticker_data.index.tz_localize('UTC')
                        else:
                            ticker_data.index = ticker_data.index.tz_convert('UTC')
                        
                        # Use the same prediction logic as backtesting
                        from backtesting import _quick_predict_return
                        
                        # Need minimum lookback days for features (same as backtesting)
                        PREDICTION_LOOKBACK_DAYS = 120
                        if len(ticker_data) >= PREDICTION_LOOKBACK_DAYS:
                            try:
                                pred = _quick_predict_return(
                                    ticker, 
                                    ticker_data.tail(PREDICTION_LOOKBACK_DAYS),
                                    models[ticker],
                                    scalers.get(ticker),
                                    y_scalers.get(ticker),
                                    horizon_days=10
                                )
                                
                                if pred != -np.inf:
                                    predictions.append((ticker, pred))
                                    print(f"   📊 {ticker}: Prediction = {pred:.4f}")
                                    
                            except Exception as e:
                                print(f"   ⚠️ Error predicting {ticker}: {e}")
                                # Print full traceback to identify exact error location
                                import traceback
                                print(f"   🔍 Full traceback:")
                                traceback.print_exc()
                                continue
                        
                except Exception as e:
                    print(f"   ⚠️ Error predicting {ticker}: {e}")
                    # Print full traceback for outer exception too
                    import traceback
                    print(f"   🔍 Outer exception traceback:")
                    traceback.print_exc()
                    continue
        
        # Select top N by predicted return (same as backtesting line 2750-2754)
        if predictions:
            predictions.sort(key=lambda x: x[1], reverse=True)
            num_to_select = min(TOP_N_STOCKS, len(predictions))
            selected_stocks = [ticker for ticker, _ in predictions[:num_to_select]]
            
            print(f"   ✅ AI Strategy: Selected {len(selected_stocks)} tickers: {selected_stocks}")
            for ticker, pred in predictions[:num_to_select]:
                print(f"      {ticker}: {pred:.4f}")
            
            return selected_stocks
        else:
            print(f"   ⚠️ AI Strategy: No valid predictions, falling back to top {TOP_N_STOCKS} tickers")
            return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers
            
    except Exception as e:
        print(f"   ❌ AI Strategy failed: {e}. Falling back to top {TOP_N_STOCKS} tickers")
        return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_3m_1y_ratio_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """3M/1Y Ratio Strategy: Select stocks with strong 3M momentum vs 1Y performance."""
    from shared_strategies import select_3m_1y_ratio_stocks
    
    print(f"   🔍 3M/1Y Ratio: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "3M/1Y Ratio")
    
    current_date = datetime.now()
    return select_3m_1y_ratio_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_risk_adj_mom_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Risk-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_risk_adj_mom_stocks
    
    print(f"   🔍 Risk-Adj Mom: Processing {len(all_tickers)} tickers")
    print(f"   🔍 Risk-Adj Mom: Data available: {all_tickers_data is not None}")
    
    # Use shared helper to prepare data with date as index
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Risk-Adj Mom")
    
    # Pass required date parameters
    current_date = datetime.now()
    train_start_date = current_date - timedelta(days=365)
    
    selected = select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, 
                                          current_date=current_date, 
                                          train_start_date=train_start_date,
                                          top_n=PORTFOLIO_SIZE)
    
    if selected:
        print(f"   ✅ Risk-Adj Mom: Selected {len(selected)} stocks: {selected}")
    else:
        print(f"   ⚠️  Risk-Adj Mom: No stocks selected!")
        print(f"      - Check if data has 'Close' column")
        print(f"      - Check if stocks meet min score threshold (30.0)")
    
    return selected


def get_mean_reversion_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Mean Reversion Strategy: Use shared strategy logic."""
    from shared_strategies import select_mean_reversion_stocks
    
    print(f"   🔍 Mean Reversion: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Mean Reversion")
    
    current_date = datetime.now()
    return select_mean_reversion_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_volatility_adj_mom_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Volatility-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_volatility_adj_mom_stocks
    
    print(f"   🔍 Vol-Adj Mom: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Vol-Adj Mom")
    
    current_date = datetime.now()
    return select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_dynamic_bh_tickers(all_tickers: List[str], period: str, all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Dynamic Buy & Hold Strategy: Use shared strategy logic."""
    from shared_strategies import select_dynamic_bh_stocks
    
    print(f"   🔍 Dynamic BH ({period}): Processing {len(all_tickers)} tickers")
    print(f"   🔍 Dynamic BH ({period}): Data available: {all_tickers_data is not None}")
    
    # Use shared helper to prepare data with date as index
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, f"Dynamic BH ({period})")
    
    # Pass required date parameters
    current_date = datetime.now()
    
    selected = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, 
                                      period=period, 
                                      current_date=current_date,
                                      top_n=PORTFOLIO_SIZE)
    return selected


def get_static_bh_tickers(all_tickers: List[str], period: str, all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Static Buy & Hold Strategy: Select top performers based on period and hold them."""
    from shared_strategies import select_dynamic_bh_stocks
    
    print(f"   🔍 Static BH ({period}): Processing {len(all_tickers)} tickers")
    print(f"   🔍 Static BH ({period}): Data available: {all_tickers_data is not None}")
    
    # Use shared helper to prepare data with date as index
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, f"Static BH ({period})")
    
    # Pass required date parameters
    current_date = datetime.now()
    
    selected = select_dynamic_bh_stocks(all_tickers, ticker_data_grouped, 
                                      period=period, 
                                      current_date=current_date,
                                      top_n=PORTFOLIO_SIZE)
    return selected


def get_quality_momentum_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Quality + Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_quality_momentum_stocks
    
    print(f"   🔍 Quality+Mom: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Quality+Mom")
    
    current_date = datetime.now()
    return select_quality_momentum_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_multitask_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Multi-Task Learning Strategy: Use Multi-Task model for selection."""
    try:
        from multitask_strategy import select_multitask_stocks
        from datetime import datetime, timedelta
        
        print(f"   🤖 Multi-Task: Processing {len(all_tickers)} tickers")
        
        # Prepare data
        ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Multi-Task")
        
        # Set dates
        current_date = datetime.now()
        train_start_date = current_date - timedelta(days=365)
        train_end_date = current_date - timedelta(days=30)
        
        # Get Multi-Task selection
        selected = select_multitask_stocks(all_tickers, ticker_data_grouped, 
                                          current_date=current_date,
                                          train_start_date=train_start_date,
                                          train_end_date=train_end_date,
                                          top_n=PORTFOLIO_SIZE)
        
        if selected:
            print(f"   ✅ Multi-Task: Selected {len(selected)} stocks: {selected}")
            return selected
        else:
            print(f"   ⚠️ Multi-Task: No stocks selected, falling back to top {TOP_N_STOCKS} tickers")
            return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers
            
    except Exception as e:
        print(f"   ❌ Multi-Task failed: {e}. Falling back to top {TOP_N_STOCKS} tickers")
        return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_turnaround_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Turnaround Strategy: Select stocks with low 3Y but high 1Y performance."""
    from shared_strategies import select_turnaround_stocks
    
    print(f"   🔍 Turnaround: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Turnaround")
    
    current_date = datetime.now()
    return select_turnaround_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)




def get_ratio_1y_3m_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """1Y/3M Ratio Strategy: Select stocks with strong 1Y performance but weak 3M (buy on dip)."""
    from shared_strategies import select_1y_3m_ratio_stocks
    
    print(f"   🔍 1Y/3M Ratio: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "1Y/3M Ratio")
    
    current_date = datetime.now()
    return select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_volatility_hybrid_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Hybrid Momentum-Volatility Strategy: Combines strong momentum with controlled volatility."""
    from shared_strategies import select_momentum_volatility_hybrid_stocks
    
    print(f"   🎯 Momentum-Volatility Hybrid: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Momentum-Volatility Hybrid")
    
    current_date = datetime.now()
    return select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_adaptive_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Adaptive Ensemble Strategy: Meta-ensemble combining multiple strategies dynamically."""
    from adaptive_ensemble import select_adaptive_ensemble_stocks
    
    print(f"   🔍 Adaptive Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Adaptive Ensemble")
    
    current_date = datetime.now()
    return select_adaptive_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_volatility_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Volatility-Adjusted Ensemble Strategy: Risk-managed position sizing."""
    from volatility_ensemble import select_volatility_ensemble_stocks
    
    print(f"   🔍 Volatility Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Volatility Ensemble")
    
    current_date = datetime.now()
    return select_volatility_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_ai_volatility_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """AI-Enhanced Volatility Ensemble Strategy: AI-optimized weights and volatility caps."""
    from ai_volatility_ensemble import select_ai_volatility_ensemble_stocks
    
    print(f"   🤖 AI Volatility Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "AI Volatility Ensemble")
    
    current_date = datetime.now()
    return select_ai_volatility_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=TOP_N_STOCKS)


def get_correlation_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Correlation-Filtered Ensemble Strategy: Diversification-focused."""
    from correlation_ensemble import select_correlation_ensemble_stocks
    
    print(f"   🔍 Correlation Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Correlation Ensemble")
    
    current_date = datetime.now()
    return select_correlation_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_dynamic_pool_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Dynamic Strategy Pool Strategy: Rotates strategies based on performance."""
    from dynamic_pool import select_dynamic_pool_stocks
    
    print(f"   🔍 Dynamic Pool: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Dynamic Pool")
    
    current_date = datetime.now()
    return select_dynamic_pool_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_sentiment_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Sentiment-Enhanced Ensemble Strategy: Incorporates news sentiment."""
    from sentiment_ensemble import select_sentiment_ensemble_stocks
    
    print(f"   🔍 Sentiment Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Sentiment Ensemble")
    
    current_date = datetime.now()
    return select_sentiment_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_breakout_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Momentum Breakout Strategy: 52-week high breakouts with volume confirmation."""
    from momentum_breakout import select_momentum_breakout_stocks
    
    print(f"   🔍 Momentum Breakout: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Momentum Breakout")
    
    current_date = datetime.now()
    return select_momentum_breakout_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_factor_rotation_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Factor Rotation Strategy: Rotates between Value/Growth/Momentum/Quality based on regime."""
    from factor_rotation import select_factor_rotation_stocks
    
    print(f"   🔍 Factor Rotation: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Factor Rotation")
    
    current_date = datetime.now()
    return select_factor_rotation_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_pairs_trading_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Pairs Trading Strategy: Statistical arbitrage on correlated pairs."""
    from pairs_trading import select_pairs_trading_stocks
    
    print(f"   🔍 Pairs Trading: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Pairs Trading")
    
    current_date = datetime.now()
    return select_pairs_trading_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_earnings_momentum_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Earnings Momentum (PEAD) Strategy: Post-earnings announcement drift."""
    from earnings_momentum import select_earnings_momentum_stocks
    
    print(f"   🔍 Earnings Momentum: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Earnings Momentum")
    
    current_date = datetime.now()
    return select_earnings_momentum_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_insider_trading_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Insider Trading Signal Strategy: Follow insider buying patterns."""
    from insider_trading import select_insider_trading_stocks
    
    print(f"   🔍 Insider Trading: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Insider Trading")
    
    current_date = datetime.now()
    return select_insider_trading_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_options_sentiment_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """Options-Based Sentiment Strategy: Put/call ratios and unusual activity."""
    from options_sentiment import select_options_sentiment_stocks
    
    print(f"   🔍 Options Sentiment: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "Options Sentiment")
    
    current_date = datetime.now()
    return select_options_sentiment_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_ml_ensemble_tickers(all_tickers: List[str], all_tickers_data: pd.DataFrame = None) -> List[str]:
    """ML Ensemble Strategy: Weighted voting from multiple ML models."""
    from ml_ensemble import select_ml_ensemble_stocks
    
    print(f"   🔍 ML Ensemble: Processing {len(all_tickers)} tickers")
    ticker_data_grouped = _prepare_ticker_data_grouped(all_tickers, all_tickers_data, "ML Ensemble")
    
    current_date = datetime.now()
    return select_ml_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def run_live_trading_with_filtered_tickers(filtered_tickers: List[str], all_tickers_data: pd.DataFrame = None):
    """Live trading function that receives pre-filtered tickers from main.py."""
    # Use the filtered tickers instead of fetching all tickers
    valid_tickers = filtered_tickers
    all_tickers_data_for_strategy = all_tickers_data
    
    # Get strategy dynamically from config (set by main.py)
    LIVE_TRADING_STRATEGY = get_live_trading_strategy()
    
    # Continue with the rest of the live trading logic
    print("=" * 80)
    print(" AI STOCK ADVISOR - LIVE TRADING")
    print("=" * 80)
    print(f" {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")

    # Display selected strategy
    strategy_names = {
        'ai_individual': 'AI Strategy (Individual Models)',
        'multitask': 'Multi-Task Learning',
        'risk_adj_mom': 'Risk-Adjusted Momentum',
        'mean_reversion': 'Mean Reversion',
        'quality_momentum': 'Quality + Momentum',
        'dynamic_bh_1y': 'Dynamic Buy & Hold (1 Year)',
        'dynamic_bh_6m': 'Dynamic Buy & Hold (6 Months)',
        'dynamic_bh_3m': 'Dynamic Buy & Hold (3 Months)',
        'dynamic_bh_1m': 'Dynamic Buy & Hold (1 Month)',
        'static_bh_1y': 'Static Buy & Hold (1 Year)',
        'static_bh_6m': 'Static Buy & Hold (6 Months)',
        'static_bh_3m': 'Static Buy & Hold (3 Months)',
        'static_bh_1m': 'Static Buy & Hold (1 Month, 30-day rebalance)',
        'turnaround': 'Turnaround (Low 3Y, High 1Y)',
        'ratio_3m_1y': '3M/1Y Ratio (Momentum Acceleration)',
        'ratio_1y_3m': '1Y/3M Ratio (Buy on Dip)',
        'momentum_volatility_hybrid': 'Momentum-Volatility Hybrid (Controlled Momentum)',
        'adaptive_ensemble': 'Adaptive Ensemble (Meta-Strategy)',
        'volatility_ensemble': 'Volatility Ensemble (Risk-Managed)',
        'ai_volatility_ensemble': 'AI Volatility Ensemble (AI-Enhanced)',
        'correlation_ensemble': 'Correlation Ensemble (Diversified)',
        'dynamic_pool': 'Dynamic Pool (Adaptive)',
        'sentiment_ensemble': 'Sentiment Ensemble (News-Enhanced)',
        'momentum_breakout': 'Momentum Breakout (52-Week High)',
        'factor_rotation': 'Factor Rotation (Value/Growth/Mom/Quality)',
        'pairs_trading': 'Pairs Trading (Statistical Arbitrage)',
        'earnings_momentum': 'Earnings Momentum (PEAD)',
        'insider_trading': 'Insider Trading Signal',
        'options_sentiment': 'Options Sentiment (Put/Call)',
        'ml_ensemble': 'ML Ensemble (Multi-Model Voting)'
    }

    strategy_name = strategy_names.get(LIVE_TRADING_STRATEGY, LIVE_TRADING_STRATEGY)
    print(f" Strategy: {strategy_name}  Hold top {TOP_N_STOCKS}")
    print(f" Investment per stock: ${INVESTMENT_PER_STOCK:,.2f}")

    mode = " DRY-RUN" if not LIVE_TRADING_ENABLED else (" PAPER" if USE_PAPER_TRADING else "  LIVE")
    print(f" Mode: {mode}")
    print("=" * 80)

    client = get_alpaca_client()
    if client is None:
        print("\n Cannot proceed without Alpaca connection")
        return

    # Get current positions
    current_positions = {}
    try:
        positions = client.get_all_positions()
        for pos in positions:
            if float(pos.qty_available) != 0:  # Only include positions with available shares
                current_positions[pos.symbol] = float(pos.qty_available)
    except Exception as e:
        print(f"   ⚠️ Could not get current positions: {e}")

    print(f"\n Current positions: {len(current_positions)}")
    for ticker, qty in current_positions.items():
        print(f"  - {ticker}: {int(qty)} shares")

    # Get target tickers based on strategy
    print(f"\n 🎯 Running {LIVE_TRADING_STRATEGY} strategy...")
    print(f"    Available tickers: {len(valid_tickers)}")
    
    # Pass downloaded data if available for strategies that need it
    all_tickers_data_for_strategy = all_tickers_data_for_strategy if LIVE_TRADING_STRATEGY in ['risk_adj_mom', 'dynamic_bh_1y', 'dynamic_bh_6m', 'dynamic_bh_3m', 'dynamic_bh_1m', 'static_bh_6m', 'static_bh_3m', 'static_bh_1m', 'ratio_1y_3m', 'ratio_3m_1y', 'turnaround', 'momentum_volatility_hybrid'] and all_tickers_data_for_strategy is not None else None
    if LIVE_TRADING_STRATEGY in ['risk_adj_mom', 'dynamic_bh_1y', 'dynamic_bh_6m', 'dynamic_bh_3m', 'dynamic_bh_1m', 'static_bh_6m', 'static_bh_3m', 'static_bh_1m', 'ratio_1y_3m', 'ratio_3m_1y', 'turnaround', 'momentum_volatility_hybrid']:
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

    # Rebalance portfolio
    rebalance_portfolio(
        client,
        target_tickers,
        current_positions,
        INVESTMENT_PER_STOCK
    )

    # Summary
    print("\n" + "=" * 80)
    print(" LIVE TRADING COMPLETE")
    print("=" * 80)
    print(f" Portfolio should now hold: {target_tickers}")
    print(f" Check Alpaca dashboard: https://app.alpaca.markets/{'paper' if USE_PAPER_TRADING else 'live'}/dashboard")
    print("=" * 80)

    # 7. Rebalance portfolio
    rebalance_portfolio(
        client,
        target_tickers,
        current_positions,
        INVESTMENT_PER_STOCK
    )

    # 8. Summary
    print("\n" + "=" * 80)
    print(" LIVE TRADING COMPLETE")
    print("=" * 80)
    print(f" Portfolio should now hold: {target_tickers}")
    print(f" Check Alpaca dashboard: https://app.alpaca.markets/{'paper' if USE_PAPER_TRADING else 'live'}/dashboard")
    print("=" * 80)




