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


def load_strategy_selections_from_json(strategy_name: str) -> Optional[List[str]]:
    """
    Load strategy selections from the JSON file saved by backtesting.
    This is the preferred method - live trading should use backtest results, not recalculate.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'momentum_volatility_hybrid_6m')
    
    Returns:
        List of ticker symbols, or None if file not found or strategy not in file
    """
    import json
    from pathlib import Path
    
    selections_file = Path('logs/strategy_selections.json')
    
    if not selections_file.exists():
        print(f"   [WARN] Strategy selections file not found: {selections_file}")
        print(f"   [INFO] Run backtesting first to generate strategy selections")
        return None
    
    try:
        with open(selections_file, 'r') as f:
            data = json.load(f)
        
        # Check file age
        from datetime import datetime, timedelta
        timestamp = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        
        if age_hours > 24:
            print(f"   [WARN] Strategy selections are {age_hours:.1f} hours old")
            print(f"   [INFO] Consider re-running backtesting for fresh selections")
        
        # Get strategy tickers
        strategies = data.get('strategies', {})
        if strategy_name in strategies:
            tickers = strategies[strategy_name].get('tickers', [])
            print(f"   [INFO] Loaded {len(tickers)} tickers for {strategy_name} from JSON")
            print(f"   [INFO] Backtest end date: {data.get('backtest_end_date', 'unknown')}")
            return tickers
        else:
            print(f"   [WARN] Strategy '{strategy_name}' not found in selections file")
            print(f"   [INFO] Available strategies: {list(strategies.keys())}")
            return None
            
    except Exception as e:
        print(f"   [FAIL] Error loading strategy selections: {e}")
        return None


def get_all_strategy_selections() -> Optional[Dict]:
    """Load all strategy selections from JSON file."""
    import json
    from pathlib import Path
    
    selections_file = Path('logs/strategy_selections.json')
    
    if not selections_file.exists():
        return None
    
    try:
        with open(selections_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _prepare_ticker_data_grouped(all_tickers: List[str], all_tickers_data: pd.DataFrame, strategy_name: str = "Strategy") -> dict:
    """
    Prepare ticker data grouped by ticker with date as index.
    This is a shared helper for all strategies to ensure consistent data format.
    """
    ticker_data_grouped = {}
    if all_tickers_data is None:
        return ticker_data_grouped
    
    print(f"   [DEBUG] {strategy_name}: Data shape: {all_tickers_data.shape}")
    print(f"   [DEBUG] {strategy_name}: Columns: {list(all_tickers_data.columns[:5])}")
    
    # Debug: Check unique tickers in data
    if 'ticker' in all_tickers_data.columns:
        unique_tickers = all_tickers_data['ticker'].unique()
        print(f"   [DEBUG] {strategy_name}: Unique tickers in data: {len(unique_tickers)}")
        # Check if requested tickers are in data
        matching = [t for t in all_tickers[:5] if t in unique_tickers]
        print(f"   [DEBUG] {strategy_name}: First 5 requested tickers in data: {matching}")
    
    # Helper function to process ticker data
    def process_ticker_data(ticker, group):
        """Process a single ticker's data group."""
        group_copy = group.copy()
        if 'date' in group_copy.columns:
            group_copy['date'] = pd.to_datetime(group_copy['date'])
            group_copy = group_copy.set_index('date')
            # Remove ticker column since it's now the key
            group_copy = group_copy.drop('ticker', axis=1)
            return group_copy
        else:
            print(f"   [WARN] No 'date' column found for ticker {ticker}")
            return None
    
    # Data is in long format - check for 'ticker' column
    if 'ticker' in all_tickers_data.columns:
        print(f"   [DEBUG] {strategy_name}: Using long format (ticker in columns)")
        
        # Group by ticker
        for ticker, group in all_tickers_data.groupby('ticker'):
            processed_data = process_ticker_data(ticker, group)
            if processed_data is not None:
                ticker_data_grouped[ticker] = processed_data
                
    elif 'ticker' in all_tickers_data.index.names:
        # Long format: ticker in index
        print(f"   [DEBUG] {strategy_name}: Using long format (ticker in index)")
        for ticker, group in all_tickers_data.groupby('ticker'):
            processed_data = process_ticker_data(ticker, group)
            if processed_data is not None:
                ticker_data_grouped[ticker] = processed_data
                
    else:
        # Assume ticker columns (fallback - should not happen with new data format)
        print(f"   [WARN] {strategy_name}: Using ticker columns format (unexpected)")
        for ticker in all_tickers:
            if ticker in all_tickers_data.columns:
                ticker_data_grouped[ticker] = all_tickers_data[[ticker]].copy()
                ticker_data_grouped[ticker].columns = ['Close']
    
    print(f"   [DEBUG] {strategy_name}: Prepared {len(ticker_data_grouped)} ticker data groups")
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
    investment_per_stock: float,
    ticker_data_grouped: Dict[str, pd.DataFrame] = None
) -> None:
    """
    Rebalance portfolio to hold only the target tickers.

    Args:
        client: Alpaca trading client
        target_tickers: List of tickers to hold
        current_positions: Dict of current positions (ticker -> quantity)
        investment_per_stock: Amount to invest per stock
        ticker_data_grouped: Dict mapping ticker -> DataFrame with date index and OHLCV columns (same format as backtesting)
    """
    print(f"\n Rebalancing portfolio to hold: {target_tickers}")
    
    if not target_tickers:
        print(f"   [WARN] NO TARGET TICKERS SELECTED!")
        print(f"   [INFO] Possible reasons:")
        print(f"      - No stocks met the Risk-Adjusted Momentum criteria (min score: 30.0)")
        print(f"      - Insufficient data for momentum calculation")
        print(f"      - Data format issue")
        print(f"   [INFO]  Skipping rebalancing - portfolio unchanged")
        return
    
    # Get account information
    try:
        account = client.get_account()
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value)
        print(f"   [INFO] Account Info:")
        print(f"      Cash: ${cash:,.2f}")
        print(f"      Buying Power: ${buying_power:,.2f}")
        print(f"      Portfolio Value: ${portfolio_value:,.2f}")
        print(f"      Investment per stock: ${investment_per_stock:,.2f}")
        print(f"      Total needed for {len(target_tickers)} stocks: ${investment_per_stock * len(target_tickers):,.2f}")
        
        if buying_power < investment_per_stock * len(target_tickers):
            print(f"   [WARN]  INSUFFICIENT FUNDS: Need ${investment_per_stock * len(target_tickers):,.2f}, only have ${buying_power:,.2f}")
            print(f"   [INFO] Consider reducing INVESTMENT_PER_STOCK")
    except Exception as e:
        print(f"   [WARN] Could not get account info: {e}")

    # Calculate target quantities
    target_positions = {}
    for ticker in target_tickers:
        # Calculate quantity based on investment amount and current price
        try:
            # Get latest close price from ticker_data_grouped (same as backtesting)
            price = None
            if ticker_data_grouped is not None and ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    valid_prices = ticker_data['Close'].dropna()
                    if len(valid_prices) > 0:
                        price = float(valid_prices.iloc[-1])
            
            if price is None or price <= 0:
                print(f"   [WARN] {ticker}: No valid price found in downloaded data, skipping")
                continue
            
            qty = int(investment_per_stock / price)
            if qty <= 0:
                print(f"   [WARN] {ticker}: Price ${price:.2f} too high for ${investment_per_stock:,.2f} allocation, skipping")
                continue
            target_positions[ticker] = qty
            print(f"   [INFO] {ticker}: ${investment_per_stock:,.2f} → {qty} shares @ ${price:.2f}")
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
        print(f"   [INFO] Found {len(orders)} open orders")
        for order in orders:
            if order.symbol not in orders_by_ticker:
                orders_by_ticker[order.symbol] = []
            orders_by_ticker[order.symbol].append(order)
        if orders_by_ticker:
            print(f"   [INFO] Active orders for {len(orders_by_ticker)} tickers")
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
                print(f" [WARN] Could not check position for {ticker}: {e}")

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


def get_strategy_tickers(strategy: str, all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Get the tickers to hold based on the selected strategy.
    
    PREFERRED: Load from JSON file saved by backtesting (no recalculation needed).
    FALLBACK: Recalculate using shared strategy functions (slower, may differ from backtest).
    
    Args:
        strategy: Strategy name
        all_tickers: List of ticker symbols
        ticker_data_grouped: Dict mapping ticker -> DataFrame with date index and OHLCV columns (same format as backtesting)
    """
    print(f"   [DEBUG] Strategy passed = '{strategy}'")
    
    # FIRST: Try to load from JSON file (preferred - uses backtest results)
    json_tickers = load_strategy_selections_from_json(strategy)
    if json_tickers:
        print(f"   [INFO] Using backtest selections from JSON file")
        return json_tickers
    
    print(f"   [WARN] JSON not available, falling back to recalculation...")

    if strategy == 'ai_individual' or strategy == 'ai_strategy':
        # AI Strategy: REMOVED - fallback to momentum-based selection
        print(f"   [WARN] AI Strategy removed. Using momentum_volatility_hybrid_6m instead.")
        return get_momentum_volatility_hybrid_tickers(all_tickers, '6m', ticker_data_grouped)

    elif strategy == 'multitask':
        # Multi-Task Learning Strategy: REMOVED - fallback to momentum-based selection
        print(f"   [WARN] Multi-Task Learning removed. Using momentum_volatility_hybrid_6m instead.")
        return get_momentum_volatility_hybrid_tickers(all_tickers, '6m', ticker_data_grouped)

    elif strategy.startswith('dynamic_bh_'):
        # Dynamic BH Strategy: Select based on performance period
        period = strategy.replace('dynamic_bh_', '')  # '1y', '6m', '3m', or '1m'
        return get_dynamic_bh_tickers(all_tickers, period, ticker_data_grouped)

    elif strategy == 'risk_adj_mom':
        # Risk-Adjusted Momentum Strategy
        return get_risk_adj_mom_tickers(all_tickers, ticker_data_grouped)

    elif strategy == '3m_1y_ratio':
        # 3M/1Y Ratio Strategy
        return get_3m_1y_ratio_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'mean_reversion':
        # Mean Reversion Strategy
        return get_mean_reversion_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'volatility_adj_mom':
        # Volatility-Adjusted Momentum Strategy
        return get_volatility_adj_mom_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'quality_momentum':
        # Quality + Momentum Strategy
        return get_quality_momentum_tickers(all_tickers, ticker_data_grouped)

    elif strategy.startswith('static_bh_'):
        # Static BH Strategy: Select based on performance period but hold static
        period = strategy.replace('static_bh_', '')  # '1y', '6m', '3m', or '1m'
        return get_static_bh_tickers(all_tickers, period, ticker_data_grouped)
    
    elif strategy == 'turnaround':
        # Turnaround Strategy: Low 3Y but high 1Y performance
        return get_turnaround_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'ratio_3m_1y':
        # 3M/1Y Ratio Strategy: Strong 3M momentum vs 1Y performance
        return get_3m_1y_ratio_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'ratio_1y_3m':
        # 1Y/3M Ratio Strategy: Strong 1Y performance but weak 3M (buy on dip)
        return get_ratio_1y_3m_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_volatility_hybrid':
        # Momentum-Volatility Hybrid Strategy: Combines strong momentum with controlled volatility
        return get_momentum_volatility_hybrid_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_volatility_hybrid_6m':
        # Momentum-Volatility Hybrid 6M Strategy: 6-month lookback variant
        return get_momentum_volatility_hybrid_6m_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_volatility_hybrid_1y':
        # Momentum-Volatility Hybrid 1Y Strategy: 1-year lookback variant
        return get_momentum_volatility_hybrid_1y_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_volatility_hybrid_1y3m':
        # Momentum-Volatility Hybrid 1Y/3M Strategy: Strong 1Y, weak 3M (buy on dip)
        return get_momentum_volatility_hybrid_1y3m_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'adaptive_ensemble':
        # Adaptive Ensemble Strategy: Meta-ensemble combining multiple strategies
        return get_adaptive_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'volatility_ensemble':
        # Volatility-Adjusted Ensemble Strategy: Risk-managed position sizing
        return get_volatility_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'enhanced_volatility':
        # Enhanced Volatility Trader: ATR-based stops + take profits
        return get_enhanced_volatility_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'ai_volatility_ensemble':
        # AI-Enhanced Volatility Ensemble Strategy: AI-optimized weights and volatility caps
        return get_ai_volatility_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'correlation_ensemble':
        # Correlation-Filtered Ensemble Strategy: Diversification-focused
        return get_correlation_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'dynamic_pool':
        # Dynamic Strategy Pool Strategy: Rotates strategies based on performance
        return get_dynamic_pool_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'sentiment_ensemble':
        # Mom-Vol Hybrid 6M + Sentiment Strategy: Incorporates news sentiment
        return get_sentiment_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_breakout':
        # Momentum Breakout Strategy: 52-week high breakouts with volume
        return get_momentum_breakout_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'factor_rotation':
        # Factor Rotation Strategy: Rotates between Value/Growth/Momentum/Quality
        return get_factor_rotation_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'pairs_trading':
        # Pairs Trading Strategy: Statistical arbitrage on correlated pairs
        return get_pairs_trading_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'earnings_momentum':
        # Earnings Momentum (PEAD) Strategy: Post-earnings drift capture
        return get_earnings_momentum_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'insider_trading':
        # Insider Trading Signal Strategy: Follow insider buying patterns
        return get_insider_trading_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'options_sentiment':
        # Options-Based Sentiment Strategy: Put/call ratios and unusual activity
        return get_options_sentiment_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'ml_ensemble':
        # ML Ensemble Strategy: Weighted voting from multiple ML models
        return get_ml_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'price_acceleration':
        # Price Acceleration Strategy: Physics-based velocity and acceleration
        return get_price_acceleration_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'voting_ensemble':
        # Voting Ensemble Strategy: Consensus picks from multiple strategies
        return get_voting_ensemble_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'dual_momentum':
        # Dual Momentum Strategy: Antonacci style absolute + relative momentum
        return get_dual_momentum_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'momentum_ai_hybrid':
        # Momentum AI Hybrid Strategy: Combines momentum with AI predictions
        return get_momentum_ai_hybrid_tickers(all_tickers, ticker_data_grouped)
    
    elif strategy == 'elite_hybrid':
        # Elite Hybrid Strategy: Advanced multi-factor ensemble
        return get_elite_hybrid_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'ai_elite':
        # AI Elite Strategy: ML-powered scoring of momentum + dip opportunities
        return get_ai_elite_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'risk_adj_mom_6m':
        # Risk-Adj Mom 6M Strategy: 6-month risk-adjusted momentum
        return get_risk_adj_mom_6m_tickers(all_tickers, ticker_data_grouped)

    elif strategy == 'risk_adj_mom_3m':
        # Risk-Adj Mom 3M Strategy: 3-month risk-adjusted momentum
        return get_risk_adj_mom_3m_tickers(all_tickers, ticker_data_grouped)

    else:
        print(f" Unknown strategy: {strategy}, using dynamic_bh_3m")
        return get_dynamic_bh_tickers(all_tickers, '3m', ticker_data_grouped)


def get_ai_strategy_tickers(all_tickers: List[str]) -> List[str]:
    """AI Strategy: REMOVED - returns fallback to top tickers."""
    print(f"   [WARN] AI Strategy has been removed. Returning top {TOP_N_STOCKS} tickers.")
    return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_3m_1y_ratio_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """3M/1Y Ratio Strategy: Select stocks with strong 3M momentum vs 1Y performance."""
    from shared_strategies import select_3m_1y_ratio_stocks
    
    print(f"   [DEBUG] 3M/1Y Ratio: Processing {len(all_tickers)} tickers")
    
    current_date = datetime.now(timezone.utc)
    return select_3m_1y_ratio_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_risk_adj_mom_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Risk-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_risk_adj_mom_stocks
    
    print(f"   [DEBUG] Risk-Adj Mom: Processing {len(all_tickers)} tickers")
    print(f"   [DEBUG] Risk-Adj Mom: Data available: {ticker_data_grouped is not None}")
    
    current_date = datetime.now(timezone.utc)
    selected = select_risk_adj_mom_stocks(all_tickers, ticker_data_grouped, 
                                          current_date=current_date, 
                                          top_n=PORTFOLIO_SIZE)
    
    if selected:
        print(f"   [PASS] Risk-Adj Mom: Selected {len(selected)} stocks: {selected}")
    else:
        from config import RISK_ADJ_MOM_MIN_SCORE
        print(f"   [WARN]  Risk-Adj Mom: No stocks selected!")
        print(f"      - Check if data has 'Close' column")
        print(f"      - Check if stocks meet min score threshold ({RISK_ADJ_MOM_MIN_SCORE})")
    
    return selected


def get_mean_reversion_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Mean Reversion Strategy: Use shared strategy logic."""
    from shared_strategies import select_mean_reversion_stocks
    
    print(f"   [DEBUG] Mean Reversion: Processing {len(all_tickers)} tickers")
    
    current_date = datetime.now(timezone.utc)
    return select_mean_reversion_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_volatility_adj_mom_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Volatility-Adjusted Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_volatility_adj_mom_stocks
    
    print(f"   [DEBUG] Vol-Adj Mom: Processing {len(all_tickers)} tickers")
    
    current_date = datetime.now(timezone.utc)
    return select_volatility_adj_mom_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_dynamic_bh_tickers(all_tickers: List[str], period: str, ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Dynamic Buy & Hold Strategy: Use shared strategy logic."""
    from shared_strategies import select_top_performers
    
    print(f"   [DEBUG] Dynamic BH ({period}): Processing {len(all_tickers)} tickers")
    print(f"   [DEBUG] Dynamic BH ({period}): Data available: {ticker_data_grouped is not None}")
    
    current_date = datetime.now(timezone.utc)
    if period == '1y':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=365, top_n=PORTFOLIO_SIZE, apply_performance_filter=True, 
                                    filter_label="Dynamic BH 1Y")
    elif period == '6m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=180, top_n=PORTFOLIO_SIZE, apply_performance_filter=True,
                                    filter_label="Dynamic BH 6M")
    elif period == '3m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=90, top_n=PORTFOLIO_SIZE, apply_performance_filter=True,
                                    filter_label="Dynamic BH 3M")
    elif period == '1m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=30, top_n=PORTFOLIO_SIZE, apply_performance_filter=True,
                                    filter_label="Dynamic BH 1M")
    else:
        print(f"   [WARN] Unknown period: {period}, defaulting to 1y")
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=365, top_n=PORTFOLIO_SIZE, apply_performance_filter=True,
                                    filter_label="Dynamic BH 1Y")


def get_static_bh_tickers(all_tickers: List[str], period: str, ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Static Buy & Hold Strategy: Select top performers based on period and hold them."""
    from shared_strategies import select_top_performers
    
    print(f"   [DEBUG] Static BH ({period}): Processing {len(all_tickers)} tickers")
    print(f"   [DEBUG] Static BH ({period}): Data available: {ticker_data_grouped is not None}")
    
    current_date = datetime.now(timezone.utc)
    if period == '1y':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=365, top_n=PORTFOLIO_SIZE)
    elif period == '6m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=180, top_n=PORTFOLIO_SIZE)
    elif period == '3m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=90, top_n=PORTFOLIO_SIZE)
    elif period == '1m':
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=30, top_n=PORTFOLIO_SIZE)
    else:
        print(f"   [WARN] Unknown period: {period}, defaulting to 1y")
        return select_top_performers(all_tickers, ticker_data_grouped, current_date=current_date, 
                                    lookback_days=365, top_n=PORTFOLIO_SIZE)


def get_quality_momentum_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Quality + Momentum Strategy: Use shared strategy logic."""
    from shared_strategies import select_quality_momentum_stocks
    
    print(f"   [DEBUG] Quality+Mom: Processing {len(all_tickers)} tickers")
    current_date = datetime.now(timezone.utc)
    return select_quality_momentum_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_multitask_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Multi-Task Learning Strategy: DISABLED - fallback to momentum."""
    print(f"   [WARN] Multi-Task strategy disabled, falling back to momentum-based selection")
    try:
        return get_dynamic_bh_1y_tickers(all_tickers, ticker_data_grouped)
    except Exception as e:
        print(f"   [FAIL] Multi-Task failed: {e}. Falling back to top {TOP_N_STOCKS} tickers")
        return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_turnaround_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Turnaround Strategy: Select stocks with low 3Y but high 1Y performance."""
    from shared_strategies import select_turnaround_stocks
    
    print(f"   [DEBUG] Turnaround: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Turnaround")
    
    current_date = datetime.now(timezone.utc)
    return select_turnaround_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)




def get_ratio_1y_3m_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """1Y/3M Ratio Strategy: Select stocks with strong 1Y performance but weak 3M (buy on dip)."""
    from shared_strategies import select_1y_3m_ratio_stocks
    
    print(f"   [DEBUG] 1Y/3M Ratio: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "1Y/3M Ratio")
    
    current_date = datetime.now(timezone.utc)
    return select_1y_3m_ratio_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_volatility_hybrid_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Hybrid Momentum-Volatility Strategy: Combines strong momentum with controlled volatility."""
    from shared_strategies import select_momentum_volatility_hybrid_stocks
    
    print(f"   [DEBUG] Momentum-Volatility Hybrid: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Momentum-Volatility Hybrid")
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_volatility_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_volatility_hybrid_6m_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Hybrid Momentum-Volatility Strategy (6M variant): Combines strong 6-month momentum with controlled volatility."""
    from shared_strategies import select_momentum_volatility_hybrid_6m_stocks
    
    print(f"   [DEBUG] Mom-Vol Hybrid 6M: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Mom-Vol Hybrid 6M")
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_volatility_hybrid_6m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_volatility_hybrid_1y_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Hybrid Momentum-Volatility Strategy (1Y variant): Combines strong 1-year momentum with controlled volatility."""
    from shared_strategies import select_momentum_volatility_hybrid_1y_stocks
    
    print(f"   [DEBUG] Mom-Vol Hybrid 1Y: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Mom-Vol Hybrid 1Y")
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_volatility_hybrid_1y_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_momentum_volatility_hybrid_1y3m_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Hybrid Momentum-Volatility Strategy (1Y/3M variant): Strong 1Y performance, weak 3M (buy on dip)."""
    from shared_strategies import select_momentum_volatility_hybrid_1y3m_stocks
    
    print(f"   [DEBUG] Mom-Vol Hybrid 1Y/3M: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Mom-Vol Hybrid 1Y/3M")
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_volatility_hybrid_1y3m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_adaptive_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Adaptive Ensemble Strategy: Meta-ensemble combining multiple strategies dynamically."""
    from adaptive_ensemble import select_adaptive_ensemble_stocks
    
    print(f"   [DEBUG] Adaptive Ensemble: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Adaptive Ensemble")
    
    current_date = datetime.now(timezone.utc)
    return select_adaptive_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_volatility_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Volatility-Adjusted Ensemble Strategy: Risk-managed position sizing."""
    from volatility_ensemble import select_volatility_ensemble_stocks
    
    print(f"   [DEBUG] Volatility Ensemble: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Volatility Ensemble")
    
    current_date = datetime.now(timezone.utc)
    return select_volatility_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_ai_volatility_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """AI-Enhanced Volatility Ensemble Strategy: DISABLED - fallback to regular volatility ensemble."""
    print(f"   [WARN] AI Volatility Ensemble disabled, falling back to regular Volatility Ensemble")
    try:
        return get_volatility_ensemble_tickers(all_tickers, ticker_data_grouped)
    except Exception as e:
        print(f"   [FAIL] AI Volatility Ensemble failed: {e}. Falling back to top {TOP_N_STOCKS} tickers")
        return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_correlation_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Correlation-Filtered Ensemble Strategy: Diversification-focused."""
    from correlation_ensemble import select_correlation_ensemble_stocks
    
    print(f"   [DEBUG] Correlation Ensemble: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Correlation Ensemble")
    
    current_date = datetime.now(timezone.utc)
    return select_correlation_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_dynamic_pool_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Dynamic Strategy Pool Strategy: Rotates strategies based on performance."""
    from dynamic_pool import select_dynamic_pool_stocks
    
    print(f"   [DEBUG] Dynamic Pool: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Dynamic Pool")
    
    current_date = datetime.now(timezone.utc)
    return select_dynamic_pool_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_sentiment_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Mom-Vol Hybrid 6M + Sentiment Strategy: Incorporates news sentiment."""
    from sentiment_ensemble import select_sentiment_ensemble_stocks
    
    print(f"   [DEBUG] Mom-Vol 6M Sentiment: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Mom-Vol 6M Sentiment")
    
    current_date = datetime.now(timezone.utc)
    return select_sentiment_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def _get_latest_data_date(ticker_data_grouped: Dict, all_tickers: List[str]) -> datetime:
    """Get the latest available data date from ticker data."""
    latest_dates = [ticker_data_grouped[t].index.max() 
                   for t in all_tickers if t in ticker_data_grouped and len(ticker_data_grouped[t]) > 0]
    if latest_dates:
        latest_date = max(latest_dates)
        # Convert to pandas Timestamp first, then ensure timezone-aware
        latest_ts = pd.Timestamp(latest_date)
        if latest_ts.tz is None:
            latest_ts = latest_ts.tz_localize('UTC')
        return latest_ts
    else:
        return datetime.now(timezone.utc)


def get_momentum_breakout_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Momentum Breakout Strategy: 52-week high breakouts with volume confirmation."""
    from momentum_breakout import select_momentum_breakout_stocks
    
    print(f"   [DEBUG] Momentum Breakout: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Momentum Breakout")
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_breakout_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_enhanced_volatility_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Enhanced Volatility Trader: Combines volatility_ensemble + static_bh_3m with ATR stops."""
    from enhanced_volatility_trader import select_enhanced_volatility_stocks
    
    print(f"   Enhanced Volatility Trader: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Enhanced Volatility Trader")
    
    current_date = datetime.now(timezone.utc)
    return select_enhanced_volatility_stocks(all_tickers, ticker_data_grouped, 
                                           current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_factor_rotation_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Factor Rotation Strategy: Rotates between Value/Growth/Momentum/Quality based on regime."""
    from factor_rotation import select_factor_rotation_stocks
    
    print(f"   [DEBUG] Factor Rotation: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Factor Rotation")
    
    current_date = datetime.now(timezone.utc)
    return select_factor_rotation_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_pairs_trading_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Pairs Trading Strategy: Statistical arbitrage on correlated pairs."""
    from pairs_trading import select_pairs_trading_stocks
    
    print(f"   [DEBUG] Pairs Trading: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Pairs Trading")
    
    current_date = datetime.now(timezone.utc)
    return select_pairs_trading_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_earnings_momentum_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Earnings Momentum (PEAD) Strategy: Post-earnings announcement drift."""
    from earnings_momentum import select_earnings_momentum_stocks
    
    print(f"   [DEBUG] Earnings Momentum: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Earnings Momentum")
    
    current_date = datetime.now(timezone.utc)
    return select_earnings_momentum_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_insider_trading_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Insider Trading Signal Strategy: Follow insider buying patterns."""
    from insider_trading import select_insider_trading_stocks
    
    print(f"   [DEBUG] Insider Trading: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Insider Trading")
    
    current_date = datetime.now(timezone.utc)
    return select_insider_trading_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_options_sentiment_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Options-Based Sentiment Strategy: Put/call ratios and unusual activity."""
    from options_sentiment import select_options_sentiment_stocks
    
    print(f"   [DEBUG] Options Sentiment: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Options Sentiment")
    
    current_date = datetime.now(timezone.utc)
    return select_options_sentiment_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_ml_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """ML Ensemble Strategy: DISABLED - fallback to momentum."""
    print(f"   [WARN] ML Ensemble disabled, falling back to momentum-based selection")
    try:
        return get_dynamic_bh_1y_tickers(all_tickers, ticker_data_grouped)
    except Exception as e:
        print(f"   [FAIL] ML Ensemble failed: {e}. Falling back to top {TOP_N_STOCKS} tickers")
        return all_tickers[:TOP_N_STOCKS] if len(all_tickers) >= TOP_N_STOCKS else all_tickers


def get_price_acceleration_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Price Acceleration Strategy: Physics-based velocity (price change) and acceleration (velocity change)."""
    from shared_strategies import select_price_acceleration_stocks
    
    print(f"   🚀 Price Acceleration: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Price Acceleration")
    
    current_date = datetime.now(timezone.utc)
    return select_price_acceleration_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_voting_ensemble_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Voting Ensemble Strategy: Consensus picks from multiple strategies."""
    from shared_strategies import select_voting_ensemble_stocks
    
    print(f"   🗳️  Voting Ensemble: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Voting Ensemble")
    
    current_date = datetime.now(timezone.utc)
    return select_voting_ensemble_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_dual_momentum_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Dual Momentum Strategy: Antonacci style absolute + relative momentum."""
    from new_strategies import select_dual_momentum_stocks
    
    print(f"   [INFO] Dual Momentum: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Dual Momentum")
    
    current_date = datetime.now(timezone.utc)
    selected, is_risk_on = select_dual_momentum_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)
    
    # If risk-off, return empty list (strategy will go to cash)
    if not is_risk_on:
        print(f"   [WARN] Dual Momentum: RISK-OFF mode - holding cash")
        return []
    
    return selected


def get_momentum_ai_hybrid_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Momentum AI Hybrid Strategy: Calls shared function (same as backtesting)."""
    from shared_strategies import select_momentum_ai_hybrid_stocks
    
    current_date = datetime.now(timezone.utc)
    return select_momentum_ai_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_elite_hybrid_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Elite Hybrid Strategy: Advanced multi-factor ensemble."""
    from elite_hybrid_strategy import select_elite_hybrid_stocks
    
    print(f"   [INFO] Elite Hybrid: Processing {len(all_tickers)} tickers")
    # ticker_data_grouped already prepared in main.py "Elite Hybrid")
    
    current_date = datetime.now(timezone.utc)
    return select_elite_hybrid_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def run_live_trading_with_filtered_tickers(filtered_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None):
    """Live trading function that receives pre-filtered tickers and grouped data from main.py.
    
    Args:
        filtered_tickers: List of pre-filtered ticker symbols
        ticker_data_grouped: Dict mapping ticker -> DataFrame with date index and OHLCV columns (same format as backtesting)
    """
    # Use the filtered tickers instead of fetching all tickers
    valid_tickers = filtered_tickers
    
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
        'ai_strategy': 'AI Strategy (Individual Models)',
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
        'momentum_volatility_hybrid_6m': 'Mom-Vol Hybrid 6M (6-Month Controlled Momentum)',
        'momentum_volatility_hybrid_1y': 'Mom-Vol Hybrid 1Y (1-Year Controlled Momentum)',
        'momentum_volatility_hybrid_1y3m': 'Mom-Vol Hybrid 1Y/3M (Strong 1Y, Weak 3M)',
        'adaptive_ensemble': 'Adaptive Ensemble (Meta-Strategy)',
        'volatility_ensemble': 'Volatility Ensemble (Risk-Managed)',
        'ai_volatility_ensemble': 'AI Volatility Ensemble (AI-Enhanced)',
        'correlation_ensemble': 'Correlation Ensemble (Diversified)',
        'ai_elite': 'AI Elite (ML-Powered Momentum + Dip Scoring)',
        'dynamic_pool': 'Dynamic Pool (Adaptive)',
        'sentiment_ensemble': 'Mom-Vol 6M Sentiment (News-Enhanced)',
        'momentum_breakout': 'Momentum Breakout (52-Week High)',
        'factor_rotation': 'Factor Rotation (Value/Growth/Mom/Quality)',
        'pairs_trading': 'Pairs Trading (Statistical Arbitrage)',
        'earnings_momentum': 'Earnings Momentum (PEAD)',
        'insider_trading': 'Insider Trading Signal',
        'options_sentiment': 'Options Sentiment (Put/Call)',
        'ml_ensemble': 'ML Ensemble (Multi-Model Voting)',
        'price_acceleration': 'Price Acceleration (Physics-Based Momentum)',
        'voting_ensemble': 'Voting Ensemble (Consensus from Multiple Strategies)',
        'dual_momentum': 'Dual Momentum (Absolute + Relative)',
        'risk_adj_mom_6m': 'Risk-Adj Mom 6M (6-Month Risk-Adjusted Momentum)',
        'risk_adj_mom_3m': 'Risk-Adj Mom 3M (3-Month Risk-Adjusted Momentum)'
    }

    strategy_name = strategy_names.get(LIVE_TRADING_STRATEGY, LIVE_TRADING_STRATEGY)
    print(f" Strategy: {strategy_name}  Hold top {TOP_N_STOCKS}")
    print(f" Investment per stock: ${INVESTMENT_PER_STOCK:,.2f}")

    mode = " DRY-RUN" if not LIVE_TRADING_ENABLED else (" PAPER" if USE_PAPER_TRADING else "  LIVE")
    print(f" Mode: {mode}")
    print("=" * 80)

    # No longer require Alpaca connection - just show recommended trades from backtest
    current_positions = {}  # Empty - we don't track positions anymore

    # Get target tickers based on strategy
    print(f"\n [DEBUG] Running {LIVE_TRADING_STRATEGY} strategy...")
    print(f"    Available tickers: {len(valid_tickers)}")
    
    # Pass ticker_data_grouped if available for strategies that need it
    ticker_data_grouped_for_strategy = ticker_data_grouped if LIVE_TRADING_STRATEGY in ['risk_adj_mom', 'risk_adj_mom_6m', 'risk_adj_mom_3m', 'dynamic_bh_1y', 'dynamic_bh_6m', 'dynamic_bh_3m', 'dynamic_bh_1m', 'static_bh_6m', 'static_bh_3m', 'static_bh_1m', 'ratio_1y_3m', 'ratio_3m_1y', 'turnaround', 'momentum_volatility_hybrid', 'momentum_volatility_hybrid_6m', 'momentum_volatility_hybrid_1y', 'momentum_volatility_hybrid_1y3m', 'price_acceleration', 'voting_ensemble', 'ai_elite', 'elite_hybrid', 'elite_risk'] else None
    if LIVE_TRADING_STRATEGY in ['risk_adj_mom', 'risk_adj_mom_6m', 'risk_adj_mom_3m', 'dynamic_bh_1y', 'dynamic_bh_6m', 'dynamic_bh_3m', 'dynamic_bh_1m', 'static_bh_6m', 'static_bh_3m', 'static_bh_1m', 'ratio_1y_3m', 'ratio_3m_1y', 'turnaround', 'momentum_volatility_hybrid', 'momentum_volatility_hybrid_6m', 'momentum_volatility_hybrid_1y', 'momentum_volatility_hybrid_1y3m', 'price_acceleration', 'voting_ensemble', 'ai_elite', 'elite_hybrid', 'elite_risk']:
        print(f"    Data available: {ticker_data_grouped_for_strategy is not None}")
    
    target_tickers = get_strategy_tickers(LIVE_TRADING_STRATEGY, valid_tickers, ticker_data_grouped_for_strategy)
    
    print(f"\n[DEBUG] SELECTED STOCKS FOR TRADING:")
    print(f"   Strategy: {LIVE_TRADING_STRATEGY}")
    print(f"   Number of stocks: {len(target_tickers)}")
    print(f"   Stocks to buy: {target_tickers}")
    
    # Show recommended portfolio
    print(f"\n" + "=" * 80)
    print(f" RECOMMENDED PORTFOLIO ({LIVE_TRADING_STRATEGY})")
    print("=" * 80)
    print(f" Stocks to hold: {target_tickers}")
    print(f" Number of positions: {len(target_tickers)}")
    print(f" Investment per stock: ${INVESTMENT_PER_STOCK:,.2f}")
    print(f" Total investment: ${INVESTMENT_PER_STOCK * len(target_tickers):,.2f}")
    print("=" * 80)
    
    # Show all available strategies from JSON if available
    all_selections = get_all_strategy_selections()
    if all_selections:
        print(f"\n📊 ALL STRATEGY SELECTIONS (from backtest {all_selections.get('backtest_end_date', 'unknown')}):")
        print("-" * 80)
        for strat_name, strat_data in all_selections.get('strategies', {}).items():
            tickers = strat_data.get('tickers', [])
            perf = all_selections.get('performance', {}).get(strat_name, {})
            ret_pct = perf.get('return_pct', 0)
            print(f"   {strat_name:<30} {len(tickers):>2} stocks  {ret_pct:>+7.1f}%  {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
        print("-" * 80)
    
    print("\n✅ LIVE TRADING RECOMMENDATIONS COMPLETE")
    print("   Use these selections to manually execute trades on your broker")


def get_ai_elite_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """AI Elite Strategy: Calls shared function (same as backtesting)."""
    from shared_strategies import select_ai_elite_with_training
    
    current_date = _get_latest_data_date(ticker_data_grouped, all_tickers)
    selected, _ = select_ai_elite_with_training(
        all_tickers=all_tickers,
        ticker_data_grouped=ticker_data_grouped,
        current_date=current_date,
        top_n=PORTFOLIO_SIZE
    )
    return selected


def get_risk_adj_mom_6m_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Risk-Adj Mom 6M Strategy: 6-month risk-adjusted momentum (return/vol^0.5)."""
    from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks
    
    print(f"   📊 Risk-Adj Mom 6M: Processing {len(all_tickers)} tickers")
    
    current_date = datetime.now(timezone.utc)
    return select_risk_adj_mom_6m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)


def get_risk_adj_mom_3m_tickers(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame] = None) -> List[str]:
    """Risk-Adj Mom 3M Strategy: 3-month risk-adjusted momentum (return/vol^0.5)."""
    from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks
    
    print(f"   📊 Risk-Adj Mom 3M: Processing {len(all_tickers)} tickers")
    
    current_date = datetime.now(timezone.utc)
    return select_risk_adj_mom_3m_stocks(all_tickers, ticker_data_grouped, current_date=current_date, top_n=PORTFOLIO_SIZE)




