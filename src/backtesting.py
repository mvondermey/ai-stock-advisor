"""
backtesting.py
Final version – includes 1D sequential optimisation, compatible with main.py and accepts extra kwargs (e.g., top_tickers).
"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
from backtesting_env import RuleTradingEnv
from ml_models import initialize_ml_libraries, train_and_evaluate_models
from data_utils import load_prices, fetch_training_data, _ensure_dir, _calculate_technical_indicators
import logging
from pathlib import Path
from config import (
    GRU_TARGET_PERCENTAGE_OPTIONS, GRU_CLASS_HORIZON_OPTIONS,
    TRANSACTION_COST, SEED, INVESTMENT_PER_STOCK, PORTFOLIO_SIZE,
    BACKTEST_DAYS,
    N_TOP_TICKERS, USE_PERFORMANCE_BENCHMARK, PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK,
    DATA_CACHE_DIR, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY,
    FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, NUM_PROCESSES, SEQUENCE_LENGTH,
    RETRAIN_FREQUENCY_DAYS, PREDICTION_LOOKBACK_DAYS, PREDICTION_TIMEOUT,
    ENABLE_RISK_ADJ_MOM, ENABLE_MEAN_REVERSION, ENABLE_QUALITY_MOM, ENABLE_MOMENTUM_AI_HYBRID,
    ENABLE_VOLATILITY_ADJ_MOM, VOLATILITY_ADJ_MOM_LOOKBACK, VOLATILITY_ADJ_MOM_VOL_WINDOW, VOLATILITY_ADJ_MOM_MIN_SCORE,
    ENABLE_STATIC_BH_6M, ENABLE_STATIC_BH, ENABLE_DYNAMIC_BH_1Y, ENABLE_DYNAMIC_BH_6M, ENABLE_DYNAMIC_BH_3M, ENABLE_DYNAMIC_BH_1M,
    ENABLE_DYNAMIC_BH_1Y_VOL_FILTER, ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP, ENABLE_SECTOR_ROTATION,
    MIN_DATA_DAYS_1Y, MIN_DATA_DAYS_6M, MIN_DATA_DAYS_3M, MIN_DATA_DAYS_1M, MIN_DATA_DAYS_GENERAL,
    ENABLE_MULTITASK_LEARNING, ENABLE_3M_1Y_RATIO, ENABLE_MOMENTUM_VOLATILITY_HYBRID, ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M, ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y, ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M, ENABLE_ADAPTIVE_STRATEGY, ENABLE_TURNAROUND, ENABLE_PRICE_ACCELERATION,
    ENABLE_VOLATILITY_ENSEMBLE, ENABLE_ENHANCED_VOLATILITY, ENABLE_CORRELATION_ENSEMBLE, ENABLE_DYNAMIC_POOL, ENABLE_RISK_ADJ_MOM_SENTIMENT, ENABLE_AI_VOLATILITY_ENSEMBLE,
    ENABLE_PARALLEL_STRATEGIES, ENABLE_MULTI_TIMEFRAME_ENSEMBLE,
    CALENDAR_DAYS_PER_YEAR,
    ENABLE_MOMENTUM_ACCELERATION, ENABLE_CONCENTRATED_3M, ENABLE_DUAL_MOMENTUM, ENABLE_TREND_FOLLOWING_ATR,
    ENABLE_ELITE_HYBRID, ENABLE_ELITE_RISK, ENABLE_RISK_ADJ_MOM_6M, ENABLE_RISK_ADJ_MOM_3M, ENABLE_RISK_ADJ_MOM_3M_MONTHLY, ENABLE_AI_ELITE,
    CONCENTRATED_3M_REBALANCE_DAYS,
    AI_ELITE_RETRAIN_DAYS, AI_ELITE_TRAINING_LOOKBACK, AI_ELITE_FORWARD_DAYS, AI_ELITE_INTRADAY_LOOKBACK
)
import signal
from contextlib import contextmanager

class PredictionTimeoutError(Exception):
    """Raised when a prediction takes too long."""
    pass

@contextmanager
def prediction_timeout(seconds: int, ticker: str):
    """Context manager for prediction timeout using SIGALRM (Unix only)."""
    def timeout_handler(signum, frame):
        raise PredictionTimeoutError(f"Prediction for {ticker} timed out after {seconds}s")
    
    if seconds is None or seconds <= 0:
        yield
        return
    
    # Only use signal-based timeout on Unix systems
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # SIGALRM not available (Windows), just run without timeout
        yield
from config import (
    ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, PERIOD_HORIZONS,
    PYTORCH_AVAILABLE, CUDA_AVAILABLE, USE_LSTM, USE_GRU, # Moved from ml_models
    ENABLE_STATIC_BH, ENABLE_DYNAMIC_BH_1Y, ENABLE_DYNAMIC_BH_6M, ENABLE_DYNAMIC_BH_3M, ENABLE_DYNAMIC_BH_1M, ENABLE_RISK_ADJ_MOM, ENABLE_MEAN_REVERSION, ENABLE_QUALITY_MOM,
    ENABLE_MOMENTUM_AI_HYBRID, PORTFOLIO_SIZE, MOMENTUM_AI_HYBRID_BUY_THRESHOLD,
    MOMENTUM_AI_HYBRID_SELL_THRESHOLD, MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK,
    MOMENTUM_AI_HYBRID_STOP_LOSS, MOMENTUM_AI_HYBRID_TRAILING_STOP,
    STATIC_BH_1Y_REBALANCE_DAYS, STATIC_BH_6M_REBALANCE_DAYS, STATIC_BH_3M_REBALANCE_DAYS, STATIC_BH_1M_REBALANCE_DAYS,
    ENABLE_DYNAMIC_BH_1Y_VOL_FILTER, DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY,
    ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP, DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT,
    ENABLE_SECTOR_ROTATION, AI_REBALANCE_FREQUENCY_DAYS, ENABLE_PROFIT_GUARD, ENABLE_STOP_LOSS, STOP_LOSS_PCT, STRATEGY_STOP_LOSS_PCT, PORTFOLIO_BUFFER_SIZE,
    ENABLE_MULTITASK_LEARNING, ENABLE_3M_1Y_RATIO, ENABLE_MOMENTUM_VOLATILITY_HYBRID, ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M, ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y, ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M, ENABLE_ADAPTIVE_STRATEGY,
    ENABLE_VOLATILITY_ENSEMBLE, ENABLE_ENHANCED_VOLATILITY, ENABLE_CORRELATION_ENSEMBLE, ENABLE_DYNAMIC_POOL, ENABLE_RISK_ADJ_MOM_SENTIMENT,
    ENABLE_TURNAROUND, ENABLE_VOTING_ENSEMBLE,
    ENABLE_STATIC_BH_1Y_MONTHLY, ENABLE_STATIC_BH_6M_MONTHLY, ENABLE_STATIC_BH_3M_MONTHLY,
    RISK_ADJ_MOM_ENABLE_MOMENTUM_CONFIRMATION, RISK_ADJ_MOM_CONFIRM_SHORT, RISK_ADJ_MOM_CONFIRM_MEDIUM, RISK_ADJ_MOM_CONFIRM_LONG, RISK_ADJ_MOM_MIN_CONFIRMATIONS,
    RISK_ADJ_MOM_ENABLE_VOLUME_CONFIRMATION, RISK_ADJ_MOM_VOLUME_WINDOW, RISK_ADJ_MOM_VOLUME_MULTIPLIER,
    OPTIMIZE_REBALANCE_HORIZON,
    TRAINING_NUM_PROCESSES,
)
from scipy.stats import uniform, beta

# Global transaction cost tracking variables (initialized in main function)
ai_transaction_costs = 0
static_bh_transaction_costs = 0
static_bh_3m_transaction_costs = 0
static_bh_6m_transaction_costs = 0
static_bh_1m_transaction_costs = 0
dynamic_bh_transaction_costs = 0
dynamic_bh_1y_transaction_costs = 0
dynamic_bh_6m_transaction_costs = 0
dynamic_bh_3m_transaction_costs = 0
dynamic_bh_1m_transaction_costs = 0
risk_adj_mom_transaction_costs = 0
mean_reversion_transaction_costs = 0
quality_momentum_transaction_costs = 0
momentum_ai_hybrid_transaction_costs = 0
volatility_adj_mom_transaction_costs = 0
sector_rotation_transaction_costs = 0
multitask_transaction_costs = 0
ratio_3m_1y_transaction_costs = 0
ratio_1y_3m_transaction_costs = 0
turnaround_transaction_costs = 0
adaptive_strategy_transaction_costs = 0
adaptive_strategy_portfolio_value = 0
volatility_ensemble_transaction_costs = 0
enhanced_volatility_transaction_costs = 0
correlation_ensemble_transaction_costs = 0
dynamic_pool_transaction_costs = 0
sentiment_ensemble_transaction_costs = 0
voting_ensemble_transaction_costs = 0
momentum_volatility_hybrid_transaction_costs = 0
momentum_volatility_hybrid_6m_transaction_costs = 0
price_acceleration_transaction_costs = 0
dynamic_bh_1y_vol_filter_transaction_costs = 0
dynamic_bh_1y_trailing_stop_transaction_costs = 0
adaptive_ensemble_transaction_costs = 0
ai_volatility_ensemble_transaction_costs = 0
multi_tf_ensemble_transaction_costs = 0


def _collect_data_worker(args):
    """Top-level worker for ProcessPoolExecutor - collects training data for one ticker."""
    ticker, ticker_data, train_start, train_end, forward_days, market_returns = args
    from ai_elite_strategy_per_ticker import collect_ticker_training_data
    samples = collect_ticker_training_data(
        ticker=ticker, ticker_data=ticker_data,
        train_start_date=train_start, train_end_date=train_end,
        forward_days=forward_days, market_returns=market_returns
    )
    return ticker, samples


def _fine_tune_worker(args):
    """Top-level worker for ProcessPoolExecutor - fine-tunes base model for one ticker."""
    ticker, ticker_samples, base_model, save_path = args
    from ai_elite_strategy_per_ticker import fine_tune_per_ticker
    model = fine_tune_per_ticker(
        ticker=ticker, ticker_samples=ticker_samples,
        base_model=base_model, save_path=save_path
    )
    return ticker, model


def _last_valid_close_up_to(ticker_df: pd.DataFrame, current_date: datetime) -> Optional[float]:
    try:
        s = ticker_df.loc[:current_date, "Close"].dropna()
        if len(s) == 0:
            return None
        v = float(s.iloc[-1])
        return None if pd.isna(v) else v
    except Exception:
        return None


def _return_over_lookback(
    ticker_df: pd.DataFrame,
    current_date: datetime,
    lookback_days: int
) -> Optional[float]:
    """
    Return over lookback as a fraction (e.g. 0.10 = +10%).
    Uses first valid close at/after lookback start, and last valid close up to current_date.
    """
    try:
        start_date = current_date - timedelta(days=int(lookback_days))
        window = ticker_df.loc[start_date:current_date, "Close"].dropna()
        if len(window) < 2:
            return None
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        if start_price <= 0 or pd.isna(start_price) or pd.isna(end_price):
            return None
        return (end_price / start_price) - 1.0
    except Exception:
        return None


def _mark_to_market_value(
    positions: Dict[str, Dict],
    cash: float,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime
) -> float:
    total = float(cash or 0.0)
    for t, pos in (positions or {}).items():
        try:
            if t not in ticker_data_grouped:
                continue
            px = _last_valid_close_up_to(ticker_data_grouped[t], current_date)
            if px is None:
                continue
            shares = float(pos.get("shares", 0.0) or 0.0)
            total += shares * px
        except Exception:
            continue
    return float(total)


def _should_rebalance_by_profit_since_last_rebalance(
    current_stocks: List[str],
    new_stocks: List[str],
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    positions: Dict[str, Dict],
    cash: float,
    transaction_cost: float,
    last_rebalance_value: float
) -> Tuple[bool, str]:
    """
    Transaction cost guard: Only rebalance if the portfolio has grown enough
    since the last rebalance to cover the transaction costs.
    
    Logic:
    1. Save portfolio value when last rebalanced (last_rebalance_value)
    2. Calculate current portfolio value (mark-to-market)
    3. Calculate transaction costs for the stocks being changed
    4. Rebalance if: (current_value - transaction_costs) > last_rebalance_value
    
    This ensures we only trade if we are "up enough" since the last rebalance
    to pay for the next rebalance.
    """
    if not new_stocks:
        return False, "no new selection"
    if not current_stocks:
        # For initial allocation, always allow rebalancing regardless of transaction costs
        return True, "initial allocation"

    current_set = set(current_stocks)
    new_set = set(new_stocks)
    
    stocks_to_sell = current_set - new_set  # Stocks to exit
    stocks_to_buy = new_set - current_set   # Stocks to enter
    
    if len(stocks_to_sell) == 0 and len(stocks_to_buy) == 0:
        return False, "no change"

    # Calculate current portfolio value
    portfolio_value_now = _mark_to_market_value(positions, cash, ticker_data_grouped, current_date)
    
    # Calculate actual transaction costs for the specific trades
    total_sell_value = 0.0
    for ticker in stocks_to_sell:
        if ticker in positions:
            pos = positions[ticker]
            shares = pos.get('shares', 0)
            if ticker in ticker_data_grouped:
                price_data = ticker_data_grouped[ticker].loc[:current_date]
                if not price_data.empty:
                    current_price = price_data['Close'].dropna().iloc[-1]
                    total_sell_value += shares * current_price
    
    total_buy_value = 0.0
    # Estimate buy value based on equal allocation of freed capital
    if stocks_to_buy:
        # Capital available = cash + sell proceeds (after sell costs)
        sell_proceeds_after_cost = total_sell_value * (1 - transaction_cost)
        available_capital = cash + sell_proceeds_after_cost
        # Each new stock gets equal share
        per_stock_allocation = available_capital / len(new_stocks) if new_stocks else 0
        total_buy_value = per_stock_allocation * len(stocks_to_buy)
    
    # Transaction costs: sell cost + buy cost
    sell_cost = total_sell_value * transaction_cost
    buy_cost = total_buy_value * transaction_cost
    total_transaction_cost = sell_cost + buy_cost
    
    # Check if portfolio value after costs exceeds last rebalance value
    value_after_costs = portfolio_value_now - total_transaction_cost
    delta_vs_last = value_after_costs - float(last_rebalance_value or 0.0)
    
    if delta_vs_last > 0:
        return True, (
            f"value_after_cost>last (now ${portfolio_value_now:,.0f} - cost ${total_transaction_cost:,.0f} = "
            f"${value_after_costs:,.0f} > last ${last_rebalance_value:,.0f}; Δ=${delta_vs_last:,.0f})"
        )
    return False, (
        f"value_after_cost<=last (now ${portfolio_value_now:,.0f} - cost ${total_transaction_cost:,.0f} = "
        f"${value_after_costs:,.0f} <= last ${last_rebalance_value:,.0f}; Δ=${delta_vs_last:,.0f})"
    )


def _smart_rebalance_portfolio(
    strategy_name: str,
    current_stocks: List[str],
    new_stocks: List[str],
    positions: Dict[str, Dict],
    cash: float,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime,
    transaction_cost: float,
    portfolio_size: int = 10,
    force_rebalance: bool = False,
    buffer_size: int = None,
    strategy_stop_loss: float = None
) -> Tuple[Dict[str, Dict], float, List[str], float]:
    """
    Universal smart rebalancing function for all strategies.
    
    Implements selective rebalancing with individual stock profit guards.
    
    Args:
        strategy_name: Name of the strategy for logging
        current_stocks: Current portfolio positions
        new_stocks: New target positions
        positions: Current position details
        cash: Available cash
        ticker_data_grouped: Price data
        current_date: Current date
        transaction_cost: Transaction cost rate
        portfolio_size: Target portfolio size
        force_rebalance: Force rebalance regardless of profit guards
        buffer_size: Buffer size for portfolio rebalancing
        strategy_stop_loss: Strategy-specific stop loss percentage (optional)
    
    Returns:
        Tuple of (updated_positions, updated_cash, final_stocks, total_transaction_costs)
    """
    if not new_stocks:
        return positions, cash, current_stocks, 0.0
    
    # Use buffer size if provided, otherwise use config value
    effective_buffer_size = buffer_size if buffer_size is not None else PORTFOLIO_BUFFER_SIZE
    
    # Convert to sets for comparison
    current_positions_set = set(current_stocks)
    new_positions_set = set(new_stocks)
    
    # Create buffer set (top X stocks where X = buffer_size)
    # If buffer_size > portfolio_size, we keep more stocks than we target
    if len(new_stocks) >= effective_buffer_size:
        buffer_set = set(new_stocks[:effective_buffer_size])
    else:
        buffer_set = new_positions_set  # All new stocks are in buffer
    
    # Classify positions with buffer logic
    positions_to_sell = current_positions_set - buffer_set  # Sell if not in top buffer_size
    positions_to_buy = new_positions_set - current_positions_set   # Buy new top stocks
    positions_to_keep = current_positions_set & buffer_set   # Keep if in buffer
    
    print(f"   📊 {strategy_name} Rebalance summary (buffer={effective_buffer_size}): {len(positions_to_keep)} keep, {len(positions_to_sell)} sell, {len(positions_to_buy)} buy")
    
    total_transaction_costs = 0.0
    updated_positions = positions.copy()
    updated_cash = cash
    final_stocks = list(positions_to_keep)  # Start with positions we're keeping
    kept_unprofitable_positions = []  # Track positions kept due to profit guard
    
    # Sell positions that are no longer in target list
    for ticker in positions_to_sell:
        if ticker in ticker_data_grouped and ticker in updated_positions:
            price_data = ticker_data_grouped[ticker].loc[:current_date]
            if not price_data.empty:
                current_price = price_data['Close'].dropna().iloc[-1]
                shares = updated_positions[ticker]['shares']
                entry_price = updated_positions[ticker]['entry_price']
                
                # Calculate gain/loss
                gross_sale = shares * current_price
                gross_cost = shares * entry_price
                gain_loss = gross_sale - gross_cost
                sell_cost = gross_sale * transaction_cost
                net_gain = gain_loss - sell_cost
                loss_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                
                # Decision logic based on ENABLE_PROFIT_GUARD flag
                if ENABLE_PROFIT_GUARD:
                    # NEW behavior: Check stop loss first, then profit guard
                    should_sell = False
                    sell_reason = ""
                    
                    # Get strategy-specific stop loss, but only if global stop loss is enabled
                    if ENABLE_STOP_LOSS and strategy_name in STRATEGY_STOP_LOSS_PCT:
                        strategy_stop_loss = STRATEGY_STOP_LOSS_PCT[strategy_name]
                    elif ENABLE_STOP_LOSS:
                        strategy_stop_loss = STOP_LOSS_PCT  # Use global stop loss
                    else:
                        strategy_stop_loss = 0.0
                    
                    if strategy_stop_loss > 0 and loss_pct <= -strategy_stop_loss:
                        # Stop loss triggered - always sell
                        should_sell = True
                        sell_reason = f"STOP LOSS {loss_pct:.1%} (threshold: {strategy_stop_loss:.1%})"
                    elif net_gain > 0 or force_rebalance:
                        # Profitable to sell (net gain after costs) OR forced rebalance
                        should_sell = True
                        sell_reason = f"Profitable: ${net_gain:,.0f} ({loss_pct:+.1%})"
                    else:
                        # Unprofitable to sell - keep position to avoid locking in loss + transaction cost
                        should_sell = False
                        sell_reason = f"Unprofitable: would lose ${-net_gain:,.0f} ({loss_pct:+.1%})"
                else:
                    # OLD behavior: Always sell stocks not in buy list (no profit guard, no stop loss check)
                    should_sell = True
                    sell_reason = f"Not in buy list: {loss_pct:+.1%}"
                
                if should_sell:
                    if "STOP LOSS" in sell_reason:
                        print(f"   🛑 {strategy_name} {ticker}: {sell_reason}, selling")
                    else:
                        print(f"   💰 {strategy_name} Selling {ticker}: {sell_reason}")
                    total_transaction_costs += sell_cost
                    updated_cash += gross_sale - sell_cost
                    del updated_positions[ticker]
                else:
                    print(f"   🚫 {strategy_name} Holding {ticker}: {sell_reason}")
                    # Keep this position in final list
                    final_stocks.append(ticker)
                    kept_unprofitable_positions.append(ticker)
    
    # Calculate capital per stock AFTER sells (so freed-up cash is included)
    if positions_to_buy:
        # Get current portfolio value (cash after sells + value of kept positions)
        kept_positions_value = sum(
            updated_positions[ticker]['shares'] * ticker_data_grouped[ticker].loc[:current_date]['Close'].dropna().iloc[-1]
            for ticker in positions_to_keep
            if ticker in ticker_data_grouped and ticker in updated_positions
        )
        total_portfolio_value = updated_cash + kept_positions_value
        capital_per_stock = total_portfolio_value / len(new_stocks) if new_stocks else 0
    else:
        capital_per_stock = 0
    
    # Buy new positions - but skip some if we kept unprofitable positions to maintain target portfolio size
    num_positions_to_skip = len(kept_unprofitable_positions)
    positions_to_buy_list = list(positions_to_buy)
    
    if num_positions_to_skip > 0:
        # Skip the last N positions from the buy list (lowest priority)
        positions_to_actually_buy = positions_to_buy_list[:-num_positions_to_skip] if num_positions_to_skip < len(positions_to_buy_list) else []
        if positions_to_actually_buy:
            print(f"   ⚠️ {strategy_name} Kept {num_positions_to_skip} unprofitable position(s), skipping {num_positions_to_skip} new buy(s) to maintain portfolio size")
    else:
        positions_to_actually_buy = positions_to_buy_list
    
    for ticker in positions_to_actually_buy:
        if ticker in ticker_data_grouped:
            price_data = ticker_data_grouped[ticker].loc[:current_date]
            if not price_data.empty:
                current_price = price_data['Close'].dropna().iloc[-1]
                if current_price > 0:
                    shares = int(capital_per_stock / (current_price * (1 + transaction_cost)))
                    buy_value = shares * current_price
                    buy_cost = buy_value * transaction_cost
                    total_cost = buy_value + buy_cost
                    
                    # Check if we have enough cash
                    if total_cost <= updated_cash and shares > 0:
                        print(f"   🛒 {strategy_name} Buying {ticker}: {shares} shares @ ${current_price:.2f}")
                        total_transaction_costs += buy_cost
                        updated_cash -= total_cost
                        updated_positions[ticker] = {'shares': shares, 'entry_price': current_price}
                        final_stocks.append(ticker)
                    else:
                        print(f"   ❌ {strategy_name} Insufficient cash for {ticker}: need ${total_cost:,.0f}, have ${updated_cash:,.0f}")
    
    # Log skipped positions
    if num_positions_to_skip > 0 and num_positions_to_skip < len(positions_to_buy_list):
        skipped = positions_to_buy_list[-num_positions_to_skip:]
        print(f"   ⏭️ {strategy_name} Skipped buying {len(skipped)} position(s) to maintain size: {', '.join(skipped)}")
    
    return updated_positions, updated_cash, final_stocks, total_transaction_costs


from data_validation import validate_prediction_data, validate_features_after_engineering, InsufficientDataError
import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path("logs").mkdir(exist_ok=True)


def _prepare_model_for_multiprocessing(model):
    """Prepare PyTorch models for multiprocessing by converting to numpy arrays.
    
    Returns a dict with model info that can be pickled (numpy arrays only),
    and the model will be reconstructed on GPU in the worker process.
    """
    if model is None:
        return None
    if PYTORCH_AVAILABLE:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor
            if isinstance(model, (LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor)):
                # Force model to CPU first to ensure all tensors are on CPU
                model_cpu = model.cpu()
                
                # Extract state dict and convert tensors to numpy arrays for safe pickling
                state_dict = model_cpu.state_dict()
                numpy_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Convert to numpy array (detach to break computation graph, cpu to ensure no CUDA context)
                        numpy_state_dict[key] = value.detach().cpu().numpy().copy()
                    else:
                        numpy_state_dict[key] = value
                
                # Extract architecture info from model and state dict
                model_type_name = type(model_cpu).__name__
                
                # Handle TCNRegressor differently from RNN models
                if model_type_name == 'TCNRegressor':
                    # TCN uses Conv1d layers, get input_size from first conv layer
                    input_size = None
                    num_filters = 32
                    kernel_size = 3
                    num_levels = 2
                    dropout = 0.1
                    
                    # Get input_size from first conv layer weight
                    for key in numpy_state_dict.keys():
                        if 'net.0.weight' in key:  # First Conv1d layer
                            input_size = numpy_state_dict[key].shape[1]  # in_channels
                            num_filters = numpy_state_dict[key].shape[0]  # out_channels
                            break
                    
                    if input_size is None:
                        # Fallback: try to get from model's first conv layer
                        if hasattr(model_cpu, 'net') and len(model_cpu.net) > 0:
                            first_conv = model_cpu.net[0]
                            if hasattr(first_conv, 'in_channels'):
                                input_size = first_conv.in_channels
                    
                    if input_size is None:
                        sys.stderr.write("  ⚠️ ERROR in _prepare_model_for_multiprocessing: missing TCN input_size\n")
                        sys.stderr.flush()
                        return None
                    
                    model_info = {
                        'type': model_type_name,
                        'state_dict': numpy_state_dict,
                        'input_size': input_size,
                        'num_filters': num_filters,
                        'kernel_size': kernel_size,
                        'num_levels': num_levels,
                        'dropout': dropout,
                    }
                    sys.stderr.write(f"  [PREP] {model_info['type']}: in={input_size}, filters={num_filters}, levels={num_levels}\n")
                    sys.stderr.flush()
                    return model_info
                
                # For RNN models (GRU, LSTM)
                hidden_size = model_cpu.hidden_size if hasattr(model_cpu, 'hidden_size') else None
                num_layers = model_cpu.num_layers if hasattr(model_cpu, 'num_layers') else None
                
                # Infer input_size and output_size from state dict
                input_size = None
                output_size = None
                dropout = 0.0
                
                for key in numpy_state_dict.keys():
                    if 'weight_ih_l0' in key:  # First layer input weights
                        # Shape is [hidden_size*3 (GRU) or hidden_size*4 (LSTM), input_size]
                        input_size = numpy_state_dict[key].shape[1]
                    elif 'fc.weight' in key:
                        # For simple fc layer: 'fc.weight'
                        output_size = numpy_state_dict[key].shape[0]
                    elif 'fc.2.weight' in key:
                        # For nn.Sequential fc layer (GRURegressor): 'fc.2.weight' is the final layer
                        output_size = numpy_state_dict[key].shape[0]

                # Fallback: try to get output_size from model attributes or final Linear in fc
                if output_size is None and hasattr(model_cpu, 'output_size'):
                    output_size = model_cpu.output_size
                if output_size is None and hasattr(model_cpu, 'fc'):
                    try:
                        import torch.nn as nn
                        for module in reversed(list(model_cpu.fc.modules())):
                            if isinstance(module, nn.Linear):
                                output_size = module.out_features
                                break
                    except Exception:
                        pass
                
                # Get dropout from model's RNN layer
                if hasattr(model_cpu, 'gru'):
                    dropout = model_cpu.gru.dropout if hasattr(model_cpu.gru, 'dropout') else 0.0
                elif hasattr(model_cpu, 'lstm'):
                    dropout = model_cpu.lstm.dropout if hasattr(model_cpu.lstm, 'dropout') else 0.0

                # Defensive guard: ensure we have all dims
                if any(v is None for v in [input_size, hidden_size, num_layers, output_size]):
                    sys.stderr.write("  ⚠️ ERROR in _prepare_model_for_multiprocessing: missing model dimensions\n")
                    sys.stderr.write(f"     input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, output_size={output_size}\n")
                    sys.stderr.flush()
                    return None
                
                # Clear CUDA cache to ensure all references are gone
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                model_info = {
                    'type': type(model_cpu).__name__,
                    'state_dict': numpy_state_dict,  # Now contains numpy arrays, not tensors
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'output_size': output_size,
                    'dropout': dropout,
                }
                # DEBUG: Log model architecture for verification
                sys.stderr.write(f"  [PREP] {model_info['type']}: in={input_size}, hid={hidden_size}, layers={num_layers}, out={output_size}, dropout={dropout}\n")
                sys.stderr.flush()
                return model_info
            else:
                # For non-PyTorch models (LightGBM, XGBoost, etc.), return as-is
                return model
        except (ImportError, AttributeError, Exception) as e:
            sys.stderr.write(f"  ⚠️ ERROR in _prepare_model_for_multiprocessing: {e}\n")
            sys.stderr.write(f"     Model type: {type(model).__name__ if model else 'None'}\n")
            sys.stderr.flush()
            return None
    return model  # For non-PyTorch models, return as-is


def _reconstruct_model_from_info(model_info, device='cpu'):
    """Reconstruct a PyTorch model from model_info (with numpy arrays) on the specified device."""
    if model_info is None:
        return None
    if isinstance(model_info, dict) and 'type' in model_info:
        try:
            import torch
            import numpy as np
            from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, LSTMRegressor, TCNRegressor

            # Resolve device
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            torch_device = torch.device(device)

            model_type = model_info['type']
            numpy_state_dict = model_info['state_dict']

            # Convert numpy arrays back to tensors on CPU first
            state_dict = {}
            for key, value in numpy_state_dict.items():
                if isinstance(value, np.ndarray):
                    # Convert to numpy array (detach to break computation graph, cpu to ensure no CUDA context)
                    state_dict[key] = torch.from_numpy(value)  # keep on CPU for load_state_dict
                else:
                    state_dict[key] = value

            # Reconstruct model based on type (on CPU)
            if model_type == 'TCNRegressor':
                model = TCNRegressor(
                    model_info['input_size'],
                    num_filters=model_info.get('num_filters', 32),
                    kernel_size=model_info.get('kernel_size', 3),
                    num_levels=model_info.get('num_levels', 2),
                    dropout=model_info.get('dropout', 0.1)
                )
            elif model_type == 'GRUClassifier':
                model = GRUClassifier(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            elif model_type == 'GRURegressor':
                model = GRURegressor(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            elif model_type == 'LSTMClassifier':
                model = LSTMClassifier(
                    model_info['input_size'],
                    model_info['hidden_size'],
                    model_info['num_layers'],
                    model_info['output_size'],
                    model_info.get('dropout', 0.0)
                )
            else:
                return None

            # Load state dict on CPU, then move to target device
            model.load_state_dict(state_dict)
            model = model.to(torch_device)
            model.eval()
            return model
        except (ImportError, AttributeError, Exception) as e:
            import sys
            sys.stderr.write(f"  ⚠️ ERROR in _reconstruct_model_from_info: {e}\n")
            sys.stderr.write(f"     Model type: {model_info.get('type', 'unknown')}\n")
            sys.stderr.flush()
            return None
    return model_info  # For non-PyTorch models, return as-is


# -----------------------------------------------------------------------------
# Single-ticker optimisation worker
# -----------------------------------------------------------------------------
def optimize_single_ticker_worker(params):
    (
        ticker, train_data, capital_per_stock, class_horizon,
        force_thresholds_optimization, force_percentage_optimization,
        use_alpha_threshold_buy, use_alpha_threshold_sell,
        alpha_config, current_min_proba_buy, current_min_proba_sell,
        initial_class_horizon,
        target_percentage_options, class_horizon_options,
        seed, feature_set, model_buy, model_sell, scaler
    ) = params

    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Starting optimization...\n")

    df_backtest_opt = train_data.copy()
    if df_backtest_opt.empty:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: No backtest data. Skipping optimization.\n")
        return {
            'ticker': ticker,
            'min_proba_buy': current_min_proba_buy,
            'min_proba_sell': current_min_proba_sell,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no data)"
        }

    # Reconstruct PyTorch models on GPU if they were passed as model_info dicts
    if PYTORCH_AVAILABLE:
        import torch
        from config import FORCE_CPU
        device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)
    
    if model_buy is None or model_sell is None or scaler is None:
        sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Models or scaler not provided. Skipping optimization.\n")
        return {
            'ticker': ticker,
            'min_proba_buy': current_min_proba_buy,
            'min_proba_sell': current_min_proba_sell,
            'class_horizon': initial_class_horizon,
            'best_revenue': capital_per_stock,
            'optimization_status': "Failed (no models)"
        }

    best_alpha = -np.inf
    best_revenue = -np.inf
    best_min_proba_buy = current_min_proba_buy
    best_min_proba_sell = current_min_proba_sell
    best_class_horizon = initial_class_horizon

    # Store all tested combinations for backtesting
    tested_combinations = []  # List of dicts with params and revenue

    # ITERATIVE HILL-CLIMBING OPTIMIZATION
    # Start from current saved thresholds, try one step up/down, move if better
    # Probability threshold optimization removed - using simplified trading logic
    buy_options = [0.0]  # Single disabled threshold
    sell_options = [1.0]  # Single disabled threshold
    
    # Find closest indices to current thresholds
    current_buy_idx = min(range(len(buy_options)), key=lambda i: abs(buy_options[i] - current_min_proba_buy))
    current_sell_idx = min(range(len(sell_options)), key=lambda i: abs(sell_options[i] - current_min_proba_sell))
    
    p_buy = buy_options[current_buy_idx]
    p_sell = sell_options[current_sell_idx]
    
    print(f"  🔍 Iterative optimization for {ticker} starting from Buy={p_buy:.2f}, Sell={p_sell:.2f}...")
    
    # Helper function to test a single combination
    def _test_combination(p_buy, p_sell):
        env = RuleTradingEnv(
            df=df_backtest_opt.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model=model_buy,  # Use single model
            scaler=scaler,
            y_scaler=y_scaler,
            use_gate=False,  # Simplified buy-and-hold logic
            feature_set=feature_set
        )
        final_val, _, _, _, _, _ = env.run()
        revenue = final_val
        
        # Calculate buy & hold for this combination
        start_price_bh = float(df_backtest_opt["Close"].iloc[0])
        end_price_bh = float(df_backtest_opt["Close"].iloc[-1])
        shares_bh = int(capital_per_stock / start_price_bh) if start_price_bh > 0 else 0
        cash_bh = capital_per_stock - shares_bh * start_price_bh
        buy_hold_final_val = cash_bh + shares_bh * end_price_bh
        buy_hold_revenue = buy_hold_final_val - capital_per_stock
        
        # Calculate alpha from portfolio history vs buy & hold
        portfolio_history = env.portfolio_history
        if len(portfolio_history) > 1 and len(df_backtest_opt) > 1:
            # Calculate daily returns for strategy (portfolio value changes)
            strategy_values = pd.Series(portfolio_history)
            strategy_returns = strategy_values.pct_change(fill_method=None).dropna()
            
            # Calculate daily returns for buy & hold (price changes)
            close_prices = pd.to_numeric(df_backtest_opt["Close"], errors='coerce').dropna()
            bh_returns = close_prices.pct_change(fill_method=None).dropna()
            
            # Align returns - portfolio_history has same length as df_backtest_opt rows
            # Both should start from index 1 (after pct_change().dropna())
            min_len = min(len(strategy_returns), len(bh_returns))
            if min_len > 1:  # Need at least 2 data points for regression
                strategy_returns_aligned = strategy_returns.iloc[:min_len].values
                bh_returns_aligned = bh_returns.iloc[:min_len].values
                
                # Calculate alpha using OLS regression: strategy_ret = alpha + beta * bh_ret
                # Alpha is the intercept, annualized
                try:
                    X = np.column_stack([np.ones_like(bh_returns_aligned), bh_returns_aligned])
                    beta_coeffs = np.linalg.lstsq(X, strategy_returns_aligned, rcond=None)[0]
                    alpha_per_day = float(beta_coeffs[0])
                    alpha_annualized = alpha_per_day * CALENDAR_DAYS_PER_YEAR  # Annualize using calendar days per year
                except Exception as e:
                    alpha_annualized = 0.0
            else:
                alpha_annualized = 0.0
        
        # Calculate percentages
        revenue_pct = ((revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
        bh_revenue_pct = (buy_hold_revenue / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
        diff = revenue - buy_hold_final_val
        diff_pct = revenue_pct - bh_revenue_pct
        
        # Print immediately after each combination is tested
        # Use sys.stdout.write with flush to ensure ticker-specific output doesn't get mixed
        sys.stdout.write(f"  [{ticker}] Buy={p_buy:.2f}, Sell={p_sell:.2f} → "
              f"AI: ${revenue:,.2f} ({revenue_pct:+.2f}%), B&H: ${buy_hold_final_val:,.2f} ({bh_revenue_pct:+.2f}%), "
              f"Alpha: {alpha_annualized:.4f}\n")
        sys.stdout.flush()
        
        return revenue, alpha_annualized, buy_hold_final_val, buy_hold_revenue
    
    # Test initial position
    revenue_current, alpha_current, bh_val_current, bh_rev_current = _test_combination(p_buy, p_sell)
    best_revenue = revenue_current
    best_alpha = alpha_current
    best_min_proba_buy = p_buy
    best_min_proba_sell = p_sell
    
    tested_combinations.append({
        'min_proba_buy': p_buy,
        'min_proba_sell': p_sell,
        'class_horizon': initial_class_horizon,
        'revenue': revenue_current,
        'buy_hold_revenue': bh_rev_current,
        'buy_hold_final_val': bh_val_current,
        'alpha_annualized': alpha_current,
        'model_buy': model_buy,
        'model_sell': model_sell,
        'scaler': scaler
    })
    
    # Iterative hill-climbing: Test one step in each direction until no improvement
    max_iterations = 20  # Safety limit
    iteration = 0
    improvement_found = True
    
    while improvement_found and iteration < max_iterations:
        improvement_found = False
        iteration += 1
        
        # Get current indices
        current_buy_idx = buy_options.index(best_min_proba_buy)
        current_sell_idx = sell_options.index(best_min_proba_sell)
        
        # Test 4 neighbors: buy-1, buy+1, sell-1, sell+1
        neighbors = []
        
        if current_buy_idx > 0:
            neighbors.append((buy_options[current_buy_idx - 1], best_min_proba_sell, 'buy_down'))
        if current_buy_idx < len(buy_options) - 1:
            neighbors.append((buy_options[current_buy_idx + 1], best_min_proba_sell, 'buy_up'))
        if current_sell_idx > 0:
            neighbors.append((best_min_proba_buy, sell_options[current_sell_idx - 1], 'sell_down'))
        if current_sell_idx < len(sell_options) - 1:
            neighbors.append((best_min_proba_buy, sell_options[current_sell_idx + 1], 'sell_up'))
        
        # Test each neighbor
        for test_buy, test_sell, direction in neighbors:
            revenue, alpha, bh_val, bh_rev = _test_combination(test_buy, test_sell)
            
            tested_combinations.append({
                'min_proba_buy': test_buy,
                'min_proba_sell': test_sell,
                'class_horizon': initial_class_horizon,
                'revenue': revenue,
                'buy_hold_revenue': bh_rev,
                'buy_hold_final_val': bh_val,
                'alpha_annualized': alpha,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler
            })
            
            # If this neighbor is better, move there
            if alpha > best_alpha:
                best_alpha = alpha
                best_revenue = revenue
                best_min_proba_buy = test_buy
                best_min_proba_sell = test_sell
                improvement_found = True
                sys.stdout.write(f"  [{ticker}] ✨ Improvement found ({direction}): Buy={test_buy:.2f}, Sell={test_sell:.2f}, Alpha={alpha:.4f}\n")
                sys.stdout.flush()
                break  # Move to this position and start again
    
    if iteration > 1:
        sys.stdout.write(f"  [{ticker}] 🎯 Converged after {iteration} iterations with {len(tested_combinations)} tests\n")
        sys.stdout.flush()
    
    best_class_horizon = initial_class_horizon
    optimization_status = "Optimized" if iteration > 1 else "No Change"
    if not np.isclose(best_min_proba_buy, current_min_proba_buy) or \
       not np.isclose(best_min_proba_sell, current_min_proba_sell) or \
       not np.isclose(best_class_horizon, initial_class_horizon):
        optimization_status = "Optimized"

    # Calculate buy & hold for the best combination
    start_price_bh_best = float(df_backtest_opt["Close"].iloc[0])
    end_price_bh_best = float(df_backtest_opt["Close"].iloc[-1])
    shares_bh_best = int(capital_per_stock / start_price_bh_best) if start_price_bh_best > 0 else 0
    cash_bh_best = capital_per_stock - shares_bh_best * start_price_bh_best
    buy_hold_final_val_best = cash_bh_best + shares_bh_best * end_price_bh_best
    buy_hold_revenue_best = buy_hold_final_val_best - capital_per_stock
    revenue_pct_best = ((best_revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
    bh_revenue_pct_best = (buy_hold_revenue_best / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
    diff_best = best_revenue - buy_hold_final_val_best
    diff_pct_best = revenue_pct_best - bh_revenue_pct_best

    # Recalculate alpha for the best combination
    best_alpha_final = 0.0
    if best_revenue > -np.inf:
        # Find the best combination in tested_combinations
        for combo in tested_combinations:
            if (np.isclose(combo['min_proba_buy'], best_min_proba_buy) and
                np.isclose(combo['min_proba_sell'], best_min_proba_sell) and
                np.isclose(combo['class_horizon'], best_class_horizon)):
                best_alpha_final = combo.get('alpha_annualized', 0.0)
                break
    
    # Validation: Check if selected parameters beat B&H in revenue
    # If not, find the best combination that beats B&H (by revenue)
    revenue_beats_bh = best_revenue > buy_hold_final_val_best
    if not revenue_beats_bh and tested_combinations:
        # Find combinations that beat B&H
        combinations_beating_bh = [c for c in tested_combinations if c['revenue'] > c['buy_hold_final_val']]
        
        if combinations_beating_bh:
            # Among those that beat B&H, find the one with highest alpha
            best_beating_bh = max(combinations_beating_bh, key=lambda x: x.get('alpha_annualized', -np.inf))
            
            # If the best alpha combination doesn't beat B&H, but we found one that does, use it
            if best_beating_bh.get('alpha_annualized', -np.inf) > -np.inf:
                print(f"  ⚠️ [{ticker}] WARNING: Best alpha combination (Alpha={best_alpha_final:.4f}) does NOT beat B&H in revenue.")
                print(f"     Selecting alternative: Alpha={best_beating_bh.get('alpha_annualized', 0.0):.4f} that beats B&H")
                best_alpha_final = best_beating_bh.get('alpha_annualized', 0.0)
                best_revenue = best_beating_bh['revenue']
                best_min_proba_buy = best_beating_bh['min_proba_buy']
                best_min_proba_sell = best_beating_bh['min_proba_sell']
                best_class_horizon = best_beating_bh['class_horizon']
                buy_hold_final_val_best = best_beating_bh['buy_hold_final_val']
                buy_hold_revenue_best = best_beating_bh['buy_hold_revenue']
                revenue_pct_best = ((best_revenue - capital_per_stock) / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                bh_revenue_pct_best = (buy_hold_revenue_best / capital_per_stock * 100) if capital_per_stock > 0 else 0.0
                diff_best = best_revenue - buy_hold_final_val_best
                diff_pct_best = revenue_pct_best - bh_revenue_pct_best
                revenue_beats_bh = True
        else:
            # No combination beats B&H - warn but keep the best alpha
            print(f"  ⚠️ [{ticker}] WARNING: No tested combination beats Buy & Hold in revenue!")
            print(f"     Best alpha combination selected, but revenue is ${best_revenue:,.2f} vs B&H ${buy_hold_final_val_best:,.2f}")
    
    # Final validation: Ensure selected values beat B&H
    # Recalculate to be absolutely sure
    final_revenue_beats_bh = best_revenue > buy_hold_final_val_best
    
    # Print summary of selected values
    revenue_status = "✅ Beats B&H" if final_revenue_beats_bh else "❌ Below B&H"
    print(f"\n  ✅ [{ticker}] Optimization complete - Selected values (optimized for highest alpha):")
    print(f"     Horizon={best_class_horizon}, Buy={best_min_proba_buy:.2f}, Sell={best_min_proba_sell:.2f}")
    print(f"     Best Alpha (annualized): {best_alpha_final:.4f}")
    print(f"     Best AI Revenue: ${best_revenue:,.2f} ({revenue_pct_best:+.2f}%) {revenue_status}")
    print(f"     Buy & Hold Revenue: ${buy_hold_final_val_best:,.2f} ({bh_revenue_pct_best:+.2f}%)")
    print(f"     Difference: ${diff_best:,.2f} ({diff_pct_best:+.2f}%)")
    
    # Add explicit check result
    if not final_revenue_beats_bh:
        print(f"     ⚠️  WARNING: Selected parameters do NOT beat Buy & Hold in revenue!")
        print(f"        AI Strategy: ${best_revenue:,.2f} vs Buy & Hold: ${buy_hold_final_val_best:,.2f}")
        print(f"        Shortfall: ${buy_hold_final_val_best - best_revenue:,.2f}")
    else:
        print(f"     ✅ SUCCESS: Selected parameters beat Buy & Hold by ${diff_best:,.2f} ({diff_pct_best:+.2f}%)")
    
    print(f"     Status: {optimization_status}\n")

    sys.stderr.write(f"  [DEBUG] {current_process().name} - {ticker}: Optimization complete. Best Alpha={best_alpha_final:.4f}, Best Revenue=${best_revenue:,.2f}, Beats B&H={final_revenue_beats_bh}, Status: {optimization_status}\n")
    return {
        'ticker': ticker,
        'min_proba_buy': best_min_proba_buy,
        'min_proba_sell': best_min_proba_sell,
        'class_horizon': best_class_horizon,
        'best_revenue': best_revenue,
        'buy_hold_revenue': buy_hold_final_val_best,
        'revenue_beats_bh': final_revenue_beats_bh,  # Add explicit flag
        'optimization_status': optimization_status,
        'tested_combinations': tested_combinations  # Return all tested combinations
    }

def optimize_thresholds_for_portfolio_parallel(optimization_params, num_processes=None):
    num_processes = num_processes or max(1, cpu_count() - 5)
    print(f"\nOptimizing thresholds for {len(optimization_params)} tickers using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(optimize_single_ticker_worker, optimization_params),
            total=len(optimization_params),
            desc="Optimizing Thresholds"
        ))

    optimized_params = {}
    all_tested_combinations = {}  # Store all tested combinations per ticker
    
    for res in results:
        if res and res.get('ticker'):
            optimized_params[res['ticker']] = {
                k: res[k] for k in ['min_proba_buy', 'min_proba_sell', 'class_horizon', 'optimization_status', 'revenue_beats_bh']
            }
            if 'tested_combinations' in res and res['tested_combinations']:
                all_tested_combinations[res['ticker']] = res['tested_combinations']
            beats_bh_status = "✅ Beats B&H" if res.get('revenue_beats_bh', False) else "❌ Below B&H"
            print(f"Optimized {res['ticker']}: Buy>{res['min_proba_buy']:.2f}, Sell>{res['min_proba_sell']:.2f}, "
                  f"Horizon={res['class_horizon']}d → {res['optimization_status']} | {beats_bh_status}")

    return optimized_params, all_tested_combinations


# -----------------------------------------------------------------------------
# Backtest worker function
# -----------------------------------------------------------------------------
def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, y_scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy, horizon_days = params
    
    # Initial log to confirm the worker has started for a ticker
    with open("logs/worker_debug.log", "a") as f:
        f.write(f"Worker started for ticker: {ticker}\n")

    # Reconstruct PyTorch models from prepared dict format
    if PYTORCH_AVAILABLE:
        import torch
        from config import FORCE_CPU
        device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
        model_buy = _reconstruct_model_from_info(model_buy, device)
        model_sell = _reconstruct_model_from_info(model_sell, device)

    if df_backtest.empty:
        print(f"  ⚠️ Skipping backtest for {ticker}: DataFrame is empty.")
        return None
        
    # DEBUG: Log simplified approach
    import sys
    sys.stderr.write(f"\n[DEBUG {ticker}] Simplified backtest: Buy at start, hold until end\n")
    sys.stderr.flush()
    
    try:
        env = RuleTradingEnv(
            df=df_backtest.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model=model_buy,  # Use single model
            scaler=scaler,
            y_scaler=y_scaler,  # ✅ Pass y_scaler
            use_gate=False,  # Simplified buy-and-hold logic
            feature_set=feature_set,
            horizon_days=horizon_days
        )
        final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation = env.run()

        # Calculate individual Buy & Hold for the same period
        start_price_bh = float(df_backtest["Close"].iloc[0])
        end_price_bh = float(df_backtest["Close"].iloc[-1])
        individual_bh_return = ((end_price_bh - start_price_bh) / start_price_bh) * 100 if start_price_bh > 0 else 0.0
        
        # Analyze performance for this ticker
        perf_data = analyze_performance(trade_log, env.portfolio_history, df_backtest["Close"].tolist(), ticker)

        # Calculate Buy & Hold history for this ticker
        bh_history_for_ticker = []
        if not df_backtest.empty:
            start_price = float(df_backtest["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            for price_day in df_backtest["Close"].tolist():
                bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
        else:
            bh_history_for_ticker.append(capital_per_stock)

        # Print prediction summary and store stats for final summary
        if hasattr(env, 'all_predictions_buy') and len(env.all_predictions_buy) > 0:
            import numpy as np
            preds_buy = np.array(env.all_predictions_buy)
            print(f"\n📊 [{ticker}] BUY Prediction Summary:")
            print(f"   Min: {np.min(preds_buy)*100:.4f}%, Max: {np.max(preds_buy)*100:.4f}%, Mean: {np.mean(preds_buy)*100:.4f}%")
            print(f"   Positive predictions: {np.sum(preds_buy > 0)} out of {len(preds_buy)} days ({np.sum(preds_buy > 0)/len(preds_buy)*100:.1f}%)")
            print(f"   Threshold: {min_proba_buy*100:.2f}%")
            pred_min_pct = float(np.min(preds_buy) * 100)
            pred_max_pct = float(np.max(preds_buy) * 100)
            pred_mean_pct = float(np.mean(preds_buy) * 100)
        else:
            pred_min_pct = pred_max_pct = pred_mean_pct = None

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob,
            'shares_before_liquidation': shares_before_liquidation,
            'pred_min_pct': pred_min_pct,
            'pred_max_pct': pred_max_pct,
            'pred_mean_pct': pred_mean_pct,
            'buy_hold_history': bh_history_for_ticker
        }
    finally:
        # This block will execute whether an exception occurred or not.
        with open("logs/worker_debug.log", "a") as f:
            final_val_to_log = 'Error' if 'final_val' not in locals() else final_val
            f.write(f"Worker finished for ticker: {ticker}. Final Value: {final_val_to_log}\n")


def analyze_performance(
    trade_log: List[tuple],
    strategy_history: List[float],
    buy_hold_history: List[float],
    ticker: str
) -> Dict[str, float]:
    """Analyzes trades and calculates key performance metrics."""
    # --- Trade Analysis ---
    buys = [t for t in trade_log if t[1] == "BUY"]
    sells = [t for t in trade_log if t[1] == "SELL"]
    profits = []
    n = min(len(buys), len(sells))
    for i in range(n):
        pb, sb = float(buys[i][2]), float(sells[i][2])
        qb, qs = float(buys[i][3]), float(sells[i][3])
        qty = min(qb, qs)
        fee_b = float(buys[i][6]) if len(buys[i]) > 6 else 0.0
        fee_s = float(sells[i][6]) if len(sells[i]) > 6 else 0.0
        profits.append((sb - pb) * qty - (fee_b + fee_s))

    total_pnl = float(sum(profits))
    win_rate = (sum(1 for p in profits if p > 0) / len(profits)) if profits else 0.0
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  - Trades: {n}, Win Rate: {win_rate:.2%}")
    print(f"  - Total PnL: ${total_pnl:,.2f}")

    # --- Performance Metrics ---
    strat_returns = pd.Series(strategy_history).pct_change(fill_method=None).dropna()
    bh_returns = pd.Series(buy_hold_history).pct_change(fill_method=None).dropna()

    # Sharpe Ratio (annualized, using calendar days per year)
    sharpe_strat = (strat_returns.mean() / strat_returns.std()) * np.sqrt(CALENDAR_DAYS_PER_YEAR) if strat_returns.std() > 0 else 0
    sharpe_bh = (bh_returns.mean() / bh_returns.std()) * np.sqrt(CALENDAR_DAYS_PER_YEAR) if bh_returns.std() > 0 else 0

    # Max Drawdown
    strat_series = pd.Series(strategy_history)
    strat_cummax = strat_series.cummax()
    strat_drawdown = ((strat_series - strat_cummax) / strat_cummax).min()

    bh_series = pd.Series(buy_hold_history)
    bh_cummax = bh_series.cummax()
    bh_drawdown = ((bh_series - bh_cummax) / bh_cummax).min()

    print(f"\n📈 {ticker} Performance Metrics:")
    print(f"  | Metric         | Strategy      | Buy & Hold    |")
    print(f"  |----------------|---------------|---------------|")
    print(f"  | Sharpe Ratio   | {sharpe_strat:13.2f} | {sharpe_bh:13.2f} |")
    print(f"  | Max Drawdown   | {strat_drawdown:12.2%} | {bh_drawdown:12.2%} |")

    return {
        "trades": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "sharpe_ratio": sharpe_strat, "max_drawdown": strat_drawdown
    }


# -----------------------------------------------------------------------------
# Portfolio-level backtesting
# -----------------------------------------------------------------------------

def _run_portfolio_backtest_walk_forward(
    all_tickers_data: pd.DataFrame,
    backtest_start_date: datetime,
    backtest_end_date: datetime,
    initial_top_tickers: List[str],
    initial_models: Dict,  # Single regression models
    initial_scalers: Dict,
    initial_y_scalers: Dict,
    capital_per_stock: float,
    period_name: str,
    top_performers_data: List[Tuple],
    horizon_days: int = 20,
    enable_ai_strategy: bool = False  # AI Strategy removed - always disabled
) -> Tuple[float, List[float], List[str], List[Dict], Dict[str, List[float]], float, float, List[float], float, List[float], float, List[float]]:
    """
    Walk-forward backtest: Daily selection from top 40 stocks with 10-day retraining.

    Your desired approach (NOW IMPLEMENTED):
    - Initial selection: Top 40 stocks by momentum (N_TOP_TICKERS = 40)
    - Model retraining: Every 10 days for all 40 stocks
    - Daily selection: Use current models to pick best 3 from 40 stocks EVERY DAY
    - Portfolio: Rebalance only when selection changes (cost-effective)
    """

    if enable_ai_strategy:
        print(f"🔄 Walk-forward backtest for {period_name} (AI Strategy)")
        print(f"   📊 Universe: Top {len(initial_top_tickers)} stocks by momentum")
        print(f"   🧠 Model retraining: Every {RETRAIN_FREQUENCY_DAYS} days for all {len(initial_top_tickers)} stocks")
        print(f"   🎯 Daily selection: Pick best 3 from {len(initial_top_tickers)} stocks EVERY DAY using current models")
        print(f"   💰 Rebalance only when portfolio changes (transaction costs minimized)")
    else:
        print(f"🔄 Comparison strategies backtest for {period_name} (AI Strategy disabled)")
        print(f"   📊 Running comparison strategies only (BH_3m, Dynamic BH, etc.)")
        print(f"   ⚠️  AI Strategy disabled (using saved models only)")
    
    
    # Initialize
    current_models = initial_models.copy()  # Single regression models
    current_scalers = initial_scalers.copy()
    current_y_scalers = initial_y_scalers.copy()

    # Debug: Check initial models/scalers
    print(f"   🔍 Initial models: {len(current_models)} tickers, sample: {list(current_models.keys())[:3]}")
    print(f"   🔍 Initial scalers: {len(current_scalers)} tickers, sample: {list(current_scalers.keys())[:3]}")
    print(f"   🔍 Initial y_scalers: {len(current_y_scalers)} tickers, sample: {list(current_y_scalers.keys())[:3]}")

    # Check if any models are None
    none_models = [t for t, m in current_models.items() if m is None]
    if none_models:
        print(f"   ⚠️ Warning: {len(none_models)} models are None: {none_models[:5]}...")

    none_scalers = [t for t, s in current_scalers.items() if s is None]
    if none_scalers:
        print(f"   ⚠️ Warning: {len(none_scalers)} scalers are None: {none_scalers[:5]}...")

    # Track current portfolio (starts empty)
    current_portfolio_stocks = []
    total_portfolio_value = 0.0  # Start with no capital invested
    portfolio_values_history = [total_portfolio_value]

    # Track actual positions for proper portfolio management
    positions = {}  # ticker -> {'shares': float, 'avg_price': float, 'value': float}
    cash_balance = 0.0  # Available cash

    # Calculate initial capital that should be allocated (PORTFOLIO_SIZE stocks * capital_per_stock)
    from config import PORTFOLIO_SIZE
    initial_capital_needed = PORTFOLIO_SIZE * capital_per_stock
    cash_balance = initial_capital_needed  # Start with cash available for initial purchases
    
    # Track last rebalance value for transaction cost guard (same as other strategies)
    ai_strategy_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # Track STATIC BH 1Y PORTFOLIO (with optional periodic rebalancing)
    static_bh_1y_portfolio_value = initial_capital_needed
    static_bh_1y_portfolio_history = [static_bh_1y_portfolio_value]
    static_bh_1y_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_1y_cash = initial_capital_needed
    current_static_bh_1y_stocks = []
    static_bh_1y_days_since_rebalance = 0
    static_bh_1y_initialized = False

    # Track STATIC BH 3M PORTFOLIO (with optional periodic rebalancing)
    static_bh_3m_portfolio_value = initial_capital_needed
    static_bh_3m_portfolio_history = [static_bh_3m_portfolio_value]
    static_bh_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_3m_cash = initial_capital_needed
    current_static_bh_3m_stocks = []

    # Track STATIC BH 6M PORTFOLIO (with optional periodic rebalancing)
    static_bh_6m_portfolio_value = initial_capital_needed
    static_bh_6m_portfolio_history = [static_bh_6m_portfolio_value]
    static_bh_6m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_6m_cash = initial_capital_needed
    current_static_bh_6m_stocks = []
    static_bh_6m_days_since_rebalance = 0
    static_bh_6m_initialized = False
    static_bh_3m_days_since_rebalance = 0
    static_bh_3m_initialized = False

    # Track STATIC BH 1M PORTFOLIO (with optional periodic rebalancing)
    static_bh_1m_portfolio_value = initial_capital_needed
    static_bh_1m_portfolio_history = [static_bh_1m_portfolio_value]
    static_bh_1m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    static_bh_1m_cash = initial_capital_needed
    current_static_bh_1m_stocks = []
    static_bh_1m_days_since_rebalance = 0
    static_bh_1m_initialized = False

    # Track STATIC BH MONTHLY REBALANCE variants (rebalance on first trading day of each month)
    # Static BH 1Y Monthly
    static_bh_1y_monthly_portfolio_value = initial_capital_needed
    static_bh_1y_monthly_portfolio_history = [static_bh_1y_monthly_portfolio_value]
    static_bh_1y_monthly_positions = {}
    static_bh_1y_monthly_cash = initial_capital_needed
    current_static_bh_1y_monthly_stocks = []
    static_bh_1y_monthly_initialized = False
    static_bh_1y_monthly_transaction_costs = 0.0
    static_bh_1y_monthly_last_month = None  # Track last rebalanced month

    # Static BH 6M Monthly
    static_bh_6m_monthly_portfolio_value = initial_capital_needed
    static_bh_6m_monthly_portfolio_history = [static_bh_6m_monthly_portfolio_value]
    static_bh_6m_monthly_positions = {}
    static_bh_6m_monthly_cash = initial_capital_needed
    current_static_bh_6m_monthly_stocks = []
    static_bh_6m_monthly_initialized = False
    static_bh_6m_monthly_transaction_costs = 0.0
    static_bh_6m_monthly_last_month = None

    # Static BH 3M Monthly
    static_bh_3m_monthly_portfolio_value = initial_capital_needed
    static_bh_3m_monthly_portfolio_history = [static_bh_3m_monthly_portfolio_value]
    static_bh_3m_monthly_positions = {}
    static_bh_3m_monthly_cash = initial_capital_needed
    current_static_bh_3m_monthly_stocks = []
    static_bh_3m_monthly_initialized = False
    static_bh_3m_monthly_transaction_costs = 0.0
    static_bh_3m_monthly_last_month = None

    # Track DYNAMIC BH PORTFOLIO (rebalances to top N performers periodically)
    dynamic_bh_portfolio_value = initial_capital_needed
    dynamic_bh_portfolio_history = [dynamic_bh_portfolio_value]
    dynamic_bh_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_stocks = []  # Current top N stocks held by dynamic BH
    dynamic_bh_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track DYNAMIC BH 1Y + VOLATILITY FILTER PORTFOLIO (same as Dynamic BH 1Y but with volatility filter)
    dynamic_bh_1y_vol_filter_portfolio_value = initial_capital_needed
    dynamic_bh_1y_vol_filter_portfolio_history = [dynamic_bh_1y_vol_filter_portfolio_value]
    dynamic_bh_1y_vol_filter_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_1y_vol_filter_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1y_vol_filter_stocks = []  # Current top N stocks held by dynamic BH 1Y vol filter
    dynamic_bh_1y_vol_filter_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track DYNAMIC BH 1Y + TRAILING STOP PORTFOLIO (same as Dynamic BH 1Y but with 20% trailing stop)
    dynamic_bh_1y_trailing_stop_portfolio_value = initial_capital_needed
    dynamic_bh_1y_trailing_stop_portfolio_history = [dynamic_bh_1y_trailing_stop_portfolio_value]
    dynamic_bh_1y_trailing_stop_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'peak_price': float}
    dynamic_bh_1y_trailing_stop_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1y_trailing_stop_stocks = []  # Current top stocks held by dynamic BH 1Y trailing stop
    dynamic_bh_1y_trailing_stop_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance

    # Track SECTOR ROTATION PORTFOLIO
    sector_rotation_portfolio_value = initial_capital_needed
    sector_rotation_portfolio_history = [sector_rotation_portfolio_value]
    sector_rotation_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    sector_rotation_cash = initial_capital_needed  # Start with same capital as AI
    current_sector_rotation_etfs = []  # Current sector ETFs held
    sector_rotation_last_rebalance_value = initial_capital_needed  # Threshold value recorded at last rebalance
    sector_rotation_last_rebalance_date = backtest_start_date  # Initialize to backtest start date

    # Track DYNAMIC BH 3-MONTH PORTFOLIO (rebalances to top N based on 3-month performance)
    dynamic_bh_3m_portfolio_value = initial_capital_needed
    dynamic_bh_3m_portfolio_history = [dynamic_bh_3m_portfolio_value]
    dynamic_bh_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_3m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_3m_stocks = []  # Current top N stocks held by 3-month dynamic BH
    dynamic_bh_3m_last_rebalance_value = initial_capital_needed

    # Track DYNAMIC BH 6-MONTH PORTFOLIO (rebalances to top N based on 6-month performance)
    dynamic_bh_6m_portfolio_value = initial_capital_needed
    dynamic_bh_6m_portfolio_history = [dynamic_bh_6m_portfolio_value]
    dynamic_bh_6m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_6m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_6m_stocks = []  # Current top N stocks held by 6-month dynamic BH
    dynamic_bh_6m_last_rebalance_value = initial_capital_needed

    # Track DYNAMIC BH 1-MONTH PORTFOLIO (rebalances to top N based on 1-month performance)
    dynamic_bh_1m_portfolio_value = initial_capital_needed
    dynamic_bh_1m_portfolio_history = [dynamic_bh_1m_portfolio_value]
    dynamic_bh_1m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_bh_1m_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_bh_1m_stocks = []  # Current top N stocks held by 1-month dynamic BH
    dynamic_bh_1m_last_rebalance_value = initial_capital_needed

    # RISK-ADJUSTED MOMENTUM: Initialize portfolio tracking
    risk_adj_mom_portfolio_value = initial_capital_needed
    risk_adj_mom_portfolio_history = [risk_adj_mom_portfolio_value]
    risk_adj_mom_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    risk_adj_mom_cash = initial_capital_needed  # Start with same capital as AI
    current_risk_adj_mom_stocks = []  # Current top N stocks held by risk-adjusted momentum
    risk_adj_mom_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MULTI-TASK LEARNING: Initialize portfolio tracking
    multitask_portfolio_value = initial_capital_needed
    multitask_portfolio_history = [multitask_portfolio_value]
    multitask_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    multitask_cash = initial_capital_needed  # Start with same capital as AI
    current_multitask_stocks = []  # Current top stocks held by multi-task learning
    multitask_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # 3M/1Y RATIO: Initialize portfolio tracking
    ratio_3m_1y_portfolio_value = initial_capital_needed
    ratio_3m_1y_portfolio_history = [ratio_3m_1y_portfolio_value]
    ratio_3m_1y_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ratio_3m_1y_cash = initial_capital_needed  # Start with same capital as AI
    current_ratio_3m_1y_stocks = []  # Current top stocks held by 3M/1Y ratio strategy
    ratio_3m_1y_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # 1Y/3M RATIO: Initialize portfolio tracking
    ratio_1y_3m_portfolio_value = initial_capital_needed
    ratio_1y_3m_portfolio_history = [ratio_1y_3m_portfolio_value]
    ratio_1y_3m_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ratio_1y_3m_cash = initial_capital_needed  # Start with same capital as AI
    current_ratio_1y_3m_stocks = []  # Current top stocks held by 1Y/3M ratio strategy
    ratio_1y_3m_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MOMENTUM-VOLATILITY HYBRID: Initialize portfolio tracking
    momentum_volatility_hybrid_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_portfolio_history = [momentum_volatility_hybrid_portfolio_value]
    momentum_volatility_hybrid_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    momentum_volatility_hybrid_cash = initial_capital_needed  # Start with same capital as AI
    current_momentum_volatility_hybrid_stocks = []  # Current top stocks held by momentum-volatility hybrid strategy
    momentum_volatility_hybrid_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MOMENTUM-VOLATILITY HYBRID 6M: Initialize portfolio tracking
    momentum_volatility_hybrid_6m_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_6m_portfolio_history = [momentum_volatility_hybrid_6m_portfolio_value]
    momentum_volatility_hybrid_6m_positions = {}
    momentum_volatility_hybrid_6m_cash = initial_capital_needed
    current_momentum_volatility_hybrid_6m_stocks = []
    momentum_volatility_hybrid_6m_last_rebalance_value = initial_capital_needed

    # MOMENTUM-VOLATILITY HYBRID 1Y: Initialize portfolio tracking
    momentum_volatility_hybrid_1y_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_1y_portfolio_history = [momentum_volatility_hybrid_1y_portfolio_value]
    momentum_volatility_hybrid_1y_positions = {}
    momentum_volatility_hybrid_1y_cash = initial_capital_needed
    current_momentum_volatility_hybrid_1y_stocks = []
    momentum_volatility_hybrid_1y_last_rebalance_value = initial_capital_needed

    # MOMENTUM-VOLATILITY HYBRID 1Y/3M: Initialize portfolio tracking
    momentum_volatility_hybrid_1y3m_portfolio_value = initial_capital_needed
    momentum_volatility_hybrid_1y3m_portfolio_history = [momentum_volatility_hybrid_1y3m_portfolio_value]
    momentum_volatility_hybrid_1y3m_positions = {}
    momentum_volatility_hybrid_1y3m_cash = initial_capital_needed
    current_momentum_volatility_hybrid_1y3m_stocks = []
    momentum_volatility_hybrid_1y3m_last_rebalance_value = initial_capital_needed

    # PRICE ACCELERATION: Initialize portfolio tracking
    price_acceleration_portfolio_value = initial_capital_needed
    price_acceleration_portfolio_history = [price_acceleration_portfolio_value]
    price_acceleration_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    price_acceleration_cash = initial_capital_needed  # Start with same capital as AI
    current_price_acceleration_stocks = []  # Current top stocks held by price acceleration strategy
    price_acceleration_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # TURNAROUND: Initialize portfolio tracking
    turnaround_portfolio_value = initial_capital_needed
    turnaround_portfolio_history = [turnaround_portfolio_value]
    turnaround_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    turnaround_cash = initial_capital_needed  # Start with same capital as AI
    current_turnaround_stocks = []  # Current top stocks held by turnaround strategy
    turnaround_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # ADAPTIVE ENSEMBLE: Initialize portfolio tracking
    adaptive_ensemble_portfolio_value = initial_capital_needed
    adaptive_ensemble_portfolio_history = [adaptive_ensemble_portfolio_value]
    adaptive_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    adaptive_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_adaptive_ensemble_stocks = []  # Current top stocks held by adaptive ensemble
    adaptive_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # VOLATILITY ENSEMBLE: Initialize portfolio tracking
    volatility_ensemble_portfolio_value = initial_capital_needed
    volatility_ensemble_portfolio_history = [volatility_ensemble_portfolio_value]
    volatility_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    volatility_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_volatility_ensemble_stocks = []  # Current top stocks held by volatility ensemble
    volatility_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # ENHANCED VOLATILITY TRADER: Initialize portfolio tracking
    enhanced_volatility_portfolio_value = initial_capital_needed  # Start with full capital
    enhanced_volatility_portfolio_history = [enhanced_volatility_portfolio_value]
    enhanced_volatility_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'stop_loss': float, 'take_profit': float}
    enhanced_volatility_cash = initial_capital_needed  # Start with same capital as AI
    current_enhanced_volatility_stocks = []  # Current top stocks held by enhanced volatility trader
    enhanced_volatility_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # AI VOLATILITY ENSEMBLE: Initialize portfolio tracking
    ai_volatility_ensemble_portfolio_value = initial_capital_needed
    ai_volatility_ensemble_portfolio_history = [ai_volatility_ensemble_portfolio_value]
    ai_volatility_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    ai_volatility_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_ai_volatility_ensemble_stocks = []  # Current stocks held by AI volatility ensemble
    ai_volatility_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MULTI-TIMEFRAME ENSEMBLE: Initialize portfolio tracking
    multi_tf_ensemble_portfolio_value = initial_capital_needed
    multi_tf_ensemble_portfolio_history = [multi_tf_ensemble_portfolio_value]
    multi_tf_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    multi_tf_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_multi_tf_ensemble_stocks = []  # Current multi-timeframe ensemble stocks held
    multi_tf_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # CORRELATION ENSEMBLE: Initialize portfolio tracking
    correlation_ensemble_portfolio_value = initial_capital_needed
    correlation_ensemble_portfolio_history = [correlation_ensemble_portfolio_value]
    correlation_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    correlation_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_correlation_ensemble_stocks = []  # Current top stocks held by correlation ensemble
    correlation_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # DYNAMIC POOL: Initialize portfolio tracking
    dynamic_pool_portfolio_value = initial_capital_needed
    dynamic_pool_portfolio_history = [dynamic_pool_portfolio_value]
    dynamic_pool_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    dynamic_pool_cash = initial_capital_needed  # Start with same capital as AI
    current_dynamic_pool_stocks = []  # Current top stocks held by dynamic pool
    dynamic_pool_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # SENTIMENT ENSEMBLE: Initialize portfolio tracking
    sentiment_ensemble_portfolio_value = initial_capital_needed
    sentiment_ensemble_portfolio_history = [sentiment_ensemble_portfolio_value]
    sentiment_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    sentiment_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_sentiment_ensemble_stocks = []  # Current top stocks held by sentiment ensemble
    sentiment_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # RISK-ADJ MOMENTUM SENTIMENT: Initialize portfolio tracking
    risk_adj_mom_sentiment_portfolio_value = initial_capital_needed
    risk_adj_mom_sentiment_portfolio_history = [risk_adj_mom_sentiment_portfolio_value]
    risk_adj_mom_sentiment_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    risk_adj_mom_sentiment_cash = initial_capital_needed  # Start with same capital as AI
    current_risk_adj_mom_sentiment_stocks = []  # Current top stocks held by risk adj mom sentiment
    risk_adj_mom_sentiment_last_rebalance_value = initial_capital_needed  # Transaction cost guard
    risk_adj_mom_sentiment_transaction_costs = 0.0

    # VOTING ENSEMBLE: Initialize portfolio tracking
    voting_ensemble_portfolio_value = initial_capital_needed
    voting_ensemble_portfolio_history = [voting_ensemble_portfolio_value]
    voting_ensemble_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    voting_ensemble_cash = initial_capital_needed  # Start with same capital as AI
    current_voting_ensemble_stocks = []  # Current top stocks held by voting ensemble
    voting_ensemble_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # TOP 5 CONSISTENCY TRACKER: Count how many days each strategy is in top 5
    top5_consistency_counts = {}  # strategy_name -> count of days in top 5

    # AI CLASSIFICATION: Removed - stub variables for compatibility
    ai_classification_portfolio_value = 0
    ai_classification_portfolio_history = []
    ai_classification_positions = {}
    ai_classification_cash = 0
    ai_classification_transaction_costs = 0

    # MOMENTUM ACCELERATION: Initialize portfolio tracking
    mom_accel_portfolio_value = initial_capital_needed
    mom_accel_portfolio_history = [mom_accel_portfolio_value]
    mom_accel_positions = {}
    mom_accel_cash = initial_capital_needed
    current_mom_accel_stocks = []
    mom_accel_last_rebalance_value = initial_capital_needed

    # CONCENTRATED 3M: Initialize portfolio tracking
    concentrated_3m_portfolio_value = initial_capital_needed
    concentrated_3m_portfolio_history = [concentrated_3m_portfolio_value]
    concentrated_3m_positions = {}
    concentrated_3m_cash = initial_capital_needed
    current_concentrated_3m_stocks = []
    concentrated_3m_last_rebalance_value = initial_capital_needed
    concentrated_3m_days_since_rebalance = 0

    # DUAL MOMENTUM: Initialize portfolio tracking
    dual_mom_portfolio_value = initial_capital_needed
    dual_mom_portfolio_history = [dual_mom_portfolio_value]
    dual_mom_positions = {}
    dual_mom_cash = initial_capital_needed
    current_dual_mom_stocks = []
    dual_mom_is_risk_on = True

    # TREND FOLLOWING ATR: Initialize portfolio tracking
    trend_atr_portfolio_value = initial_capital_needed
    trend_atr_portfolio_history = [trend_atr_portfolio_value]
    trend_atr_positions = {}
    trend_atr_cash = initial_capital_needed
    current_trend_atr_stocks = []

    # ELITE HYBRID: Initialize portfolio tracking
    elite_hybrid_portfolio_value = initial_capital_needed
    elite_hybrid_portfolio_history = [elite_hybrid_portfolio_value]
    elite_hybrid_positions = {}
    elite_hybrid_cash = initial_capital_needed
    current_elite_hybrid_stocks = []
    elite_hybrid_last_rebalance_value = initial_capital_needed

    # ELITE RISK: Initialize portfolio tracking
    elite_risk_portfolio_value = initial_capital_needed
    elite_risk_portfolio_history = [elite_risk_portfolio_value]
    elite_risk_positions = {}
    elite_risk_cash = initial_capital_needed
    current_elite_risk_stocks = []
    elite_risk_last_rebalance_value = initial_capital_needed

    # RISK-ADJ MOM 6M: Initialize portfolio tracking
    risk_adj_mom_6m_portfolio_value = initial_capital_needed
    risk_adj_mom_6m_portfolio_history = [risk_adj_mom_6m_portfolio_value]
    risk_adj_mom_6m_positions = {}
    risk_adj_mom_6m_cash = initial_capital_needed
    current_risk_adj_mom_6m_stocks = []

    # RISK-ADJ MOM 3M: Initialize portfolio tracking
    risk_adj_mom_3m_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_portfolio_history = [risk_adj_mom_3m_portfolio_value]
    risk_adj_mom_3m_positions = {}
    risk_adj_mom_3m_cash = initial_capital_needed
    current_risk_adj_mom_3m_stocks = []

    # RISK-ADJ MOM 3M MONTHLY: Initialize portfolio tracking (rebalance start of month only)
    risk_adj_mom_3m_monthly_portfolio_value = initial_capital_needed
    risk_adj_mom_3m_monthly_portfolio_history = [risk_adj_mom_3m_monthly_portfolio_value]
    risk_adj_mom_3m_monthly_positions = {}
    risk_adj_mom_3m_monthly_cash = initial_capital_needed
    current_risk_adj_mom_3m_monthly_stocks = []
    risk_adj_mom_3m_monthly_initialized = False
    risk_adj_mom_3m_monthly_last_month = None

    # AI ELITE: Initialize portfolio tracking
    ai_elite_portfolio_value = initial_capital_needed
    ai_elite_portfolio_history = [ai_elite_portfolio_value]
    ai_elite_positions = {}
    ai_elite_cash = initial_capital_needed
    current_ai_elite_stocks = []
    ai_elite_last_rebalance_value = initial_capital_needed
    ai_elite_models = {}  # Per-ticker models: ticker -> model
    ai_elite_last_train_days = {}  # Per-ticker training tracking: ticker -> last_train_day

    # MEAN REVERSION: Initialize portfolio tracking
    mean_reversion_portfolio_value = initial_capital_needed
    mean_reversion_portfolio_history = [mean_reversion_portfolio_value]
    mean_reversion_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    mean_reversion_cash = initial_capital_needed  # Start with same capital as AI
    current_mean_reversion_stocks = []  # Current bottom N stocks held by mean reversion
    mean_reversion_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # QUALITY + MOMENTUM: Initialize portfolio tracking
    quality_momentum_portfolio_value = initial_capital_needed
    quality_momentum_portfolio_history = [quality_momentum_portfolio_value]
    quality_momentum_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    quality_momentum_cash = initial_capital_needed  # Start with same capital as AI
    current_quality_momentum_stocks = []  # Current top N stocks held by quality + momentum
    quality_momentum_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # MOMENTUM + AI HYBRID: Initialize portfolio tracking
    momentum_ai_hybrid_portfolio_value = initial_capital_needed
    momentum_ai_hybrid_portfolio_history = [momentum_ai_hybrid_portfolio_value]
    momentum_ai_hybrid_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float, 'entry_date': str, 'peak_price': float}
    momentum_ai_hybrid_cash = initial_capital_needed  # Start with same capital as AI
    current_momentum_ai_hybrid_stocks = []  # Current stocks held by momentum + AI hybrid
    last_momentum_ai_hybrid_rebalance_day = 0  # Track days since last rebalance

    # VOLATILITY-ADJUSTED MOMENTUM: Initialize portfolio tracking
    volatility_adj_mom_portfolio_value = initial_capital_needed
    volatility_adj_mom_portfolio_history = [volatility_adj_mom_portfolio_value]
    volatility_adj_mom_positions = {}  # ticker -> {'shares': float, 'entry_price': float, 'value': float}
    volatility_adj_mom_cash = initial_capital_needed  # Start with same capital as AI
    current_volatility_adj_mom_stocks = []  # Current top N stocks held by volatility-adjusted momentum
    volatility_adj_mom_last_rebalance_value = initial_capital_needed  # Transaction cost guard

    # Reset global transaction cost tracking variables for this backtest
    global ai_transaction_costs, static_bh_transaction_costs, static_bh_3m_transaction_costs, static_bh_6m_transaction_costs, static_bh_1m_transaction_costs, dynamic_bh_1y_transaction_costs, dynamic_bh_transaction_costs
    global dynamic_bh_3m_transaction_costs, dynamic_bh_6m_transaction_costs, dynamic_bh_1m_transaction_costs, risk_adj_mom_transaction_costs, mean_reversion_transaction_costs, quality_momentum_transaction_costs, momentum_ai_hybrid_transaction_costs, volatility_adj_mom_transaction_costs, dynamic_bh_1y_vol_filter_transaction_costs, dynamic_bh_1y_trailing_stop_transaction_costs, multitask_transaction_costs, ratio_3m_1y_transaction_costs, ratio_1y_3m_transaction_costs, turnaround_transaction_costs, adaptive_ensemble_transaction_costs, volatility_ensemble_transaction_costs, correlation_ensemble_transaction_costs, dynamic_pool_transaction_costs, sentiment_ensemble_transaction_costs
    global sector_rotation_transaction_costs, momentum_volatility_hybrid_transaction_costs, momentum_volatility_hybrid_6m_transaction_costs, momentum_volatility_hybrid_1y_transaction_costs, momentum_volatility_hybrid_1y3m_transaction_costs, enhanced_volatility_transaction_costs, ai_volatility_ensemble_transaction_costs, multi_tf_ensemble_transaction_costs
    ai_transaction_costs = 0.0
    static_bh_transaction_costs = 0.0  # Static BH has no transaction costs (buy once, hold)
    static_bh_3m_transaction_costs = 0.0
    static_bh_6m_transaction_costs = 0.0
    static_bh_1m_transaction_costs = 0.0
    dynamic_bh_1y_transaction_costs = 0.0
    dynamic_bh_transaction_costs = 0.0
    dynamic_bh_6m_transaction_costs = 0.0
    dynamic_bh_3m_transaction_costs = 0.0
    dynamic_bh_1m_transaction_costs = 0.0
    risk_adj_mom_transaction_costs = 0.0
    mean_reversion_transaction_costs = 0.0
    quality_momentum_transaction_costs = 0.0
    momentum_ai_hybrid_transaction_costs = 0.0
    volatility_adj_mom_transaction_costs = 0.0
    dynamic_bh_1y_vol_filter_transaction_costs = 0.0
    dynamic_bh_1y_trailing_stop_transaction_costs = 0.0
    multitask_transaction_costs = 0.0
    ratio_3m_1y_transaction_costs = 0.0
    ratio_1y_3m_transaction_costs = 0.0
    momentum_volatility_hybrid_transaction_costs = 0.0
    momentum_volatility_hybrid_6m_transaction_costs = 0.0
    momentum_volatility_hybrid_1y_transaction_costs = 0.0
    momentum_volatility_hybrid_1y3m_transaction_costs = 0.0
    turnaround_transaction_costs = 0.0
    sector_rotation_transaction_costs = 0.0
    adaptive_ensemble_transaction_costs = 0.0
    volatility_ensemble_transaction_costs = 0.0
    enhanced_volatility_transaction_costs = 0.0
    ai_volatility_ensemble_transaction_costs = 0.0
    multi_tf_ensemble_transaction_costs = 0.0
    correlation_ensemble_transaction_costs = 0.0
    dynamic_pool_transaction_costs = 0.0
    sentiment_ensemble_transaction_costs = 0.0
    voting_ensemble_transaction_costs = 0.0
    mom_accel_transaction_costs = 0.0
    concentrated_3m_transaction_costs = 0.0
    dual_mom_transaction_costs = 0.0
    trend_atr_transaction_costs = 0.0
    elite_hybrid_transaction_costs = 0.0
    elite_risk_transaction_costs = 0.0
    risk_adj_mom_6m_transaction_costs = 0.0
    risk_adj_mom_3m_transaction_costs = 0.0
    risk_adj_mom_3m_monthly_transaction_costs = 0.0
    ai_elite_transaction_costs = 0.0

    # Transaction cost tracking for strategies that don't initialize it elsewhere
    ratio_3m_1y_transaction_costs = 0.0
    ratio_1y_3m_transaction_costs = 0.0
    momentum_volatility_hybrid_transaction_costs = 0.0
    momentum_volatility_hybrid_6m_transaction_costs = 0.0
    momentum_volatility_hybrid_1y_transaction_costs = 0.0
    momentum_volatility_hybrid_1y3m_transaction_costs = 0.0
    price_acceleration_transaction_costs = 0.0
    turnaround_transaction_costs = 0.0
    ai_classification_transaction_costs = 0.0

    all_processed_tickers = []
    all_performance_metrics = []
    all_buy_hold_histories = {}
    
    # NEW: Track per-stock contributions
    # ✅ NEW: Track per-stock contributions
    stock_performance_tracking = {}  # ticker -> {'days_held': int, 'contribution': float, 'max_shares': float, 'entry_value': float, 'exit_value': float}

    # Get all trading days in the backtest period
    date_range = pd.date_range(start=backtest_start_date, end=backtest_end_date, freq='D')
    business_days = [d for d in date_range if d.weekday() < 5]  # Filter to weekdays

    print(f"   📅 Total trading days to process: {len(business_days)}")
    
    # ✅ OPTIMIZATION: Pre-group data by ticker ONCE (instead of filtering 5644 times per day!)
    print(f"   🔧 Pre-grouping data by ticker for fast lookups...", flush=True)
    ticker_data_grouped = {}
    grouped = all_tickers_data.groupby('ticker')
    available_tickers_in_data = set(all_tickers_data['ticker'].unique())
    missing_tickers = []
    
    for ticker in initial_top_tickers:
        try:
            ticker_df = grouped.get_group(ticker).copy()
            if 'date' in ticker_df.columns:
                ticker_df = ticker_df.set_index('date')
            ticker_df = ticker_df.drop('ticker', axis=1, errors='ignore')
            ticker_data_grouped[ticker] = ticker_df
        except KeyError:
            missing_tickers.append(ticker)
    print(f"   ✅ Pre-grouped {len(ticker_data_grouped)} tickers", flush=True)
    if missing_tickers:
        print(f"   ⚠️ {len(missing_tickers)} tickers NOT found in data: {missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}")
        print(f"   🔍 DEBUG: initial_top_tickers sample: {initial_top_tickers[:5]}")
        print(f"   🔍 DEBUG: available_tickers_in_data sample: {list(available_tickers_in_data)[:5]}")

    day_count = 0
    retrain_count = 0
    rebalance_count = 0

    # Meta-Strategy removed - no longer used
    meta_allocator = None

    # ✅ NEW: Track consecutive failures for fail-fast
    consecutive_no_predictions = 0
    consecutive_training_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # Abort if 5 days in a row fail
    
    # ✅ NEW: Track daily predictions vs actuals
    daily_prediction_log = []

    for current_date in business_days:
        day_count += 1

                    
        # STATIC BH PORTFOLIOS: Initialize on day 1 and optional periodic rebalancing
        # Static BH 1Y, 3M, and 1M are always initialized, then rebalance every N days if configured
        if ENABLE_STATIC_BH:
            # Increment days since last rebalance
            static_bh_1y_days_since_rebalance += 1
            static_bh_6m_days_since_rebalance += 1
            static_bh_3m_days_since_rebalance += 1
            static_bh_1m_days_since_rebalance += 1
            
            # STATIC BH 1Y: Initialize on day 1, then rebalance every N days if configured
            # Always initialize on first day, then only rebalance if REBALANCE_DAYS > 0
            should_init_or_rebalance_1y = (
                (not static_bh_1y_initialized) or  # Always initialize on first day
                (STATIC_BH_1Y_REBALANCE_DAYS > 0 and static_bh_1y_days_since_rebalance >= STATIC_BH_1Y_REBALANCE_DAYS)
            )
            
            if should_init_or_rebalance_1y:
                # Calculate top PORTFOLIO_SIZE by 1-year performance (always parallel)
                from parallel_backtest import calculate_parallel_performance
                perf_1y_list = calculate_parallel_performance(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date,
                    period_days=365
                )
                
                if perf_1y_list:
                    perf_1y_list.sort(key=lambda x: x[1], reverse=True)
                    new_static_bh_1y_stocks = [t for t, _ in perf_1y_list[:PORTFOLIO_SIZE]]
                    
                    if new_static_bh_1y_stocks != current_static_bh_1y_stocks:
                        if not static_bh_1y_initialized:
                            print(f"   🎯 Static BH 1Y: Initializing with top {len(new_static_bh_1y_stocks)} by 1Y performance: {new_static_bh_1y_stocks}")
                        else:
                            print(f"   🔄 Static BH 1Y: Smart rebalancing to top {len(new_static_bh_1y_stocks)} by 1Y performance: {new_static_bh_1y_stocks}")
                        
                        # Use universal smart rebalancing function
                        static_bh_1y_positions, static_bh_1y_cash, current_static_bh_1y_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Static BH 1Y",
                            current_stocks=current_static_bh_1y_stocks,
                            new_stocks=new_static_bh_1y_stocks,
                            positions=static_bh_1y_positions,
                            cash=static_bh_1y_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not static_bh_1y_initialized  # Force initial allocation
                        )
                        static_bh_transaction_costs += rebalance_costs
                        
                        static_bh_1y_initialized = True
                        static_bh_1y_days_since_rebalance = 0
            
            # STATIC BH 3M: Initialize on day 1, then rebalance every N days if configured
            should_init_or_rebalance_3m = (
                (not static_bh_3m_initialized) or  # Always initialize on first day
                (STATIC_BH_3M_REBALANCE_DAYS > 0 and static_bh_3m_days_since_rebalance >= STATIC_BH_3M_REBALANCE_DAYS)
            )
            
            if should_init_or_rebalance_3m:
                # Calculate top PORTFOLIO_SIZE by 3-month performance
                perf_3m_list = []
                for ticker in initial_top_tickers:
                    if ticker not in ticker_data_grouped:
                        continue
                    ticker_data = ticker_data_grouped[ticker]
                    perf_start = current_date - timedelta(days=90)
                    perf_data = ticker_data.loc[perf_start:current_date]
                    if len(perf_data) >= 30:
                        valid_close = perf_data['Close'].dropna()
                        if len(valid_close) >= 2:
                            start_p = valid_close.iloc[0]
                            end_p = valid_close.iloc[-1]
                            if start_p > 0:
                                perf_pct = ((end_p - start_p) / start_p) * 100
                                perf_3m_list.append((ticker, perf_pct))
                
                if perf_3m_list:
                    perf_3m_list.sort(key=lambda x: x[1], reverse=True)
                    new_static_bh_3m_stocks = [t for t, _ in perf_3m_list[:PORTFOLIO_SIZE]]
                    
                    if new_static_bh_3m_stocks:
                        if not static_bh_3m_initialized:
                            print(f"   🎯 Static BH 3M: Initializing with top {len(new_static_bh_3m_stocks)} by 3M performance: {new_static_bh_3m_stocks}")
                        else:
                            print(f"   🔄 Static BH 3M: Smart rebalancing to top {len(new_static_bh_3m_stocks)} by 3M performance: {new_static_bh_3m_stocks}")
                        
                        # Use universal smart rebalancing function
                        static_bh_3m_positions, static_bh_3m_cash, current_static_bh_3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Static BH 3M",
                            current_stocks=current_static_bh_3m_stocks,
                            new_stocks=new_static_bh_3m_stocks,
                            positions=static_bh_3m_positions,
                            cash=static_bh_3m_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not static_bh_3m_initialized  # Force initial allocation
                        )
                        static_bh_transaction_costs += rebalance_costs
                        
                        static_bh_3m_initialized = True
                        static_bh_3m_days_since_rebalance = 0
            
            # STATIC BH 6M: Initialize on day 1, then rebalance every N days if configured
        if ENABLE_STATIC_BH_6M:
            should_init_or_rebalance_6m = (
                (not static_bh_6m_initialized) or  # Always initialize on first day
                (STATIC_BH_6M_REBALANCE_DAYS > 0 and static_bh_6m_days_since_rebalance >= STATIC_BH_6M_REBALANCE_DAYS)
            )
            
            if should_init_or_rebalance_6m:
                # Calculate top PORTFOLIO_SIZE by 6-month performance
                perf_6m_list = []
                for ticker in initial_top_tickers:
                    if ticker not in ticker_data_grouped:
                        continue
                    ticker_data = ticker_data_grouped[ticker]
                    
                    # Calculate 6-month performance (126 trading days ~ 6 months)
                    perf_6m = _return_over_lookback(ticker_data, current_date, 126)
                    if perf_6m is not None:
                        perf_6m_list.append((ticker, perf_6m))
                
                # Sort by 6-month performance (best first) and take top PORTFOLIO_SIZE
                perf_6m_list.sort(key=lambda x: x[1], reverse=True)
                new_static_bh_6m_stocks = [ticker for ticker, _ in perf_6m_list[:PORTFOLIO_SIZE]]
                
                if new_static_bh_6m_stocks:
                    if not static_bh_6m_initialized:
                        print(f"   🎯 Static BH 6M: Initializing with top {len(new_static_bh_6m_stocks)} by 6M performance: {new_static_bh_6m_stocks}")
                    else:
                        print(f"   🔄 Static BH 6M: Smart rebalancing to top {len(new_static_bh_6m_stocks)} by 6M performance: {new_static_bh_6m_stocks}")
                    
                    # Use universal smart rebalancing function
                    static_bh_6m_positions, static_bh_6m_cash, current_static_bh_6m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Static BH 6M",
                        current_stocks=current_static_bh_6m_stocks,
                        new_stocks=new_static_bh_6m_stocks,
                        positions=static_bh_6m_positions,
                        cash=static_bh_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not static_bh_6m_initialized  # Force initial allocation
                    )
                    static_bh_transaction_costs += rebalance_costs
                    
                    static_bh_6m_initialized = True
                    static_bh_6m_days_since_rebalance = 0
            
            # STATIC BH 1M: Initialize on day 1, then rebalance every N days if configured
            should_init_or_rebalance_1m = (
                (not static_bh_1m_initialized) or  # Always initialize on first day
                (STATIC_BH_1M_REBALANCE_DAYS > 0 and static_bh_1m_days_since_rebalance >= STATIC_BH_1M_REBALANCE_DAYS)
            )
            
            if should_init_or_rebalance_1m:
                # Calculate top PORTFOLIO_SIZE by 1-month performance
                perf_1m_list = []
                for ticker in initial_top_tickers:
                    if ticker not in ticker_data_grouped:
                        continue
                    ticker_data = ticker_data_grouped[ticker]
                    
                    # Calculate 1-month performance (21 trading days ~ 1 month)
                    perf_start_date = current_date - timedelta(days=21)
                    perf_data = ticker_data.loc[perf_start_date:current_date]
                    
                    if len(perf_data) >= MIN_DATA_DAYS_1M:  # Need minimum data for 1-month
                        valid_close = perf_data['Close'].dropna()
                        if len(valid_close) >= 2:
                            start_p = valid_close.iloc[0]
                            end_p = valid_close.iloc[-1]
                            if start_p > 0:
                                perf_pct = ((end_p - start_p) / start_p) * 100
                                # Only include stocks with positive momentum
                                if perf_pct > 0:
                                    perf_1m_list.append((ticker, perf_pct))
                
                if perf_1m_list:
                    perf_1m_list.sort(key=lambda x: x[1], reverse=True)
                    new_static_bh_1m_stocks = [t for t, _ in perf_1m_list[:PORTFOLIO_SIZE]]
                    
                    if new_static_bh_1m_stocks:
                        if not static_bh_1m_initialized:
                            print(f"   🎯 Static BH 1M: Initializing with top {len(new_static_bh_1m_stocks)} by 1M performance: {new_static_bh_1m_stocks}")
                        else:
                            print(f"   🔄 Static BH 1M: Smart rebalancing to top {len(new_static_bh_1m_stocks)} by 1M performance: {new_static_bh_1m_stocks}")
                        
                        # Use universal smart rebalancing function
                        static_bh_1m_positions, static_bh_1m_cash, current_static_bh_1m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Static BH 1M",
                            current_stocks=current_static_bh_1m_stocks,
                            new_stocks=new_static_bh_1m_stocks,
                            positions=static_bh_1m_positions,
                            cash=static_bh_1m_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not static_bh_1m_initialized  # Force initial allocation
                        )
                        static_bh_1m_transaction_costs += rebalance_costs
                        
                        static_bh_1m_initialized = True
                        static_bh_1m_days_since_rebalance = 0

        # STATIC BH MONTHLY REBALANCE VARIANTS: Rebalance on first trading day of each month
        is_first_trading_day_of_month = (
            (day_count == 1) or  # First day of backtest
            (current_date.month != (business_days[business_days.index(current_date) - 1].month if business_days.index(current_date) > 0 else current_date.month))
        )

        # Static BH 1Y Monthly
        if ENABLE_STATIC_BH_1Y_MONTHLY:
            should_rebalance_1y_monthly = (not static_bh_1y_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_1y_monthly:
                from parallel_backtest import calculate_parallel_performance
                perf_1y_m_list = calculate_parallel_performance(initial_top_tickers, ticker_data_grouped, current_date, 365)
                if perf_1y_m_list:
                    perf_1y_m_list.sort(key=lambda x: x[1], reverse=True)
                    new_stocks = [t for t, _ in perf_1y_m_list[:PORTFOLIO_SIZE]]
                    if new_stocks:
                        if not static_bh_1y_monthly_initialized:
                            print(f"   🎯 Static BH 1Y Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Static BH 1Y Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        static_bh_1y_monthly_positions, static_bh_1y_monthly_cash, current_static_bh_1y_monthly_stocks, rc = _smart_rebalance_portfolio(
                            strategy_name="Static BH 1Y Mth", current_stocks=current_static_bh_1y_monthly_stocks,
                            new_stocks=new_stocks, positions=static_bh_1y_monthly_positions, cash=static_bh_1y_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_1y_monthly_initialized)
                        static_bh_1y_monthly_transaction_costs += rc
                        static_bh_1y_monthly_initialized = True
                        static_bh_1y_monthly_last_month = current_date.month

        # Static BH 6M Monthly
        if ENABLE_STATIC_BH_6M_MONTHLY:
            should_rebalance_6m_monthly = (not static_bh_6m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_6m_monthly:
                from parallel_backtest import calculate_parallel_performance
                perf_6m_m_list = calculate_parallel_performance(initial_top_tickers, ticker_data_grouped, current_date, 180)
                if perf_6m_m_list:
                    perf_6m_m_list.sort(key=lambda x: x[1], reverse=True)
                    new_stocks = [t for t, _ in perf_6m_m_list[:PORTFOLIO_SIZE]]
                    if new_stocks:
                        if not static_bh_6m_monthly_initialized:
                            print(f"   🎯 Static BH 6M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Static BH 6M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        static_bh_6m_monthly_positions, static_bh_6m_monthly_cash, current_static_bh_6m_monthly_stocks, rc = _smart_rebalance_portfolio(
                            strategy_name="Static BH 6M Mth", current_stocks=current_static_bh_6m_monthly_stocks,
                            new_stocks=new_stocks, positions=static_bh_6m_monthly_positions, cash=static_bh_6m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_6m_monthly_initialized)
                        static_bh_6m_monthly_transaction_costs += rc
                        static_bh_6m_monthly_initialized = True
                        static_bh_6m_monthly_last_month = current_date.month

        # Static BH 3M Monthly
        if ENABLE_STATIC_BH_3M_MONTHLY:
            should_rebalance_3m_monthly = (not static_bh_3m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_3m_monthly:
                from parallel_backtest import calculate_parallel_performance
                perf_3m_m_list = calculate_parallel_performance(initial_top_tickers, ticker_data_grouped, current_date, 90)
                if perf_3m_m_list:
                    perf_3m_m_list.sort(key=lambda x: x[1], reverse=True)
                    new_stocks = [t for t, _ in perf_3m_m_list[:PORTFOLIO_SIZE]]
                    if new_stocks:
                        if not static_bh_3m_monthly_initialized:
                            print(f"   🎯 Static BH 3M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Static BH 3M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        static_bh_3m_monthly_positions, static_bh_3m_monthly_cash, current_static_bh_3m_monthly_stocks, rc = _smart_rebalance_portfolio(
                            strategy_name="Static BH 3M Mth", current_stocks=current_static_bh_3m_monthly_stocks,
                            new_stocks=new_stocks, positions=static_bh_3m_monthly_positions, cash=static_bh_3m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped, current_date=current_date, transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE, force_rebalance=not static_bh_3m_monthly_initialized)
                        static_bh_3m_monthly_transaction_costs += rc
                        static_bh_3m_monthly_initialized = True
                        static_bh_3m_monthly_last_month = current_date.month

        # DYNAMIC BH 1Y PORTFOLIO: Rebalance to current top N performers DAILY
        if ENABLE_DYNAMIC_BH_1Y:
            print(f"\n🔄 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): Dynamic BH 1Y Rebalancing...")

            try:
                # Apply performance filters if enabled
                from performance_filters import filter_tickers_by_performance
                filtered_tickers_1y = filter_tickers_by_performance(
                    initial_top_tickers, ticker_data_grouped, current_date, "Dynamic BH 1Y"
                )
                
                # Calculate current top N performers based on 1-year performance
                current_top_performers = []

                # Calculate performances (always parallel)
                from parallel_backtest import calculate_parallel_performance
                current_top_performers = calculate_parallel_performance(
                    filtered_tickers_1y,
                    ticker_data_grouped,
                    current_date,
                    period_days=365
                )

                # Sort by performance and get top N
                if current_top_performers and ENABLE_DYNAMIC_BH_1Y:
                    current_top_performers.sort(key=lambda x: x[1], reverse=True)
                    new_dynamic_bh_stocks = [ticker for ticker, perf in current_top_performers[:PORTFOLIO_SIZE]]

                    print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-year): {', '.join(new_dynamic_bh_stocks)}")

                    # Use universal smart rebalancing function
                    dynamic_bh_positions, dynamic_bh_cash, current_dynamic_bh_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Dynamic BH 1Y",
                        current_stocks=current_dynamic_bh_stocks,
                        new_stocks=new_dynamic_bh_stocks,
                        positions=dynamic_bh_positions,
                        cash=dynamic_bh_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_bh_stocks  # Force initial allocation
                    )
                    dynamic_bh_transaction_costs += rebalance_costs
                    dynamic_bh_last_rebalance_value = _mark_to_market_value(
                        dynamic_bh_positions, dynamic_bh_cash, ticker_data_grouped, current_date
                    )

            except Exception as e:
                print(f"   ⚠️ Dynamic BH 1Y error: {e}")

        # DYNAMIC BH 6M PORTFOLIO: Rebalance to current top N performers DAILY
        if ENABLE_DYNAMIC_BH_6M:
            print(f"\n🔄 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): Dynamic BH 6M Rebalancing...")

            try:
                # Apply performance filters if enabled
                from performance_filters import filter_tickers_by_performance
                filtered_tickers_6m = filter_tickers_by_performance(
                    initial_top_tickers, ticker_data_grouped, current_date, "Dynamic BH 6M"
                )
                
                # Calculate current top N performers based on 6-month performance (always parallel)
                from parallel_backtest import calculate_parallel_performance
                current_top_performers_6m = calculate_parallel_performance(
                    filtered_tickers_6m, ticker_data_grouped, current_date, 180)

                # Sort by performance and get top N
                if current_top_performers_6m:
                    current_top_performers_6m.sort(key=lambda x: x[1], reverse=True)
                    new_dynamic_bh_6m_stocks = [ticker for ticker, perf in current_top_performers_6m[:PORTFOLIO_SIZE]]

                    print(f"   🏆 Top {PORTFOLIO_SIZE} performers (6-month): {', '.join(new_dynamic_bh_6m_stocks)}")

                    # Use universal smart rebalancing function
                    dynamic_bh_6m_positions, dynamic_bh_6m_cash, current_dynamic_bh_6m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Dynamic BH 6M",
                        current_stocks=current_dynamic_bh_6m_stocks,
                        new_stocks=new_dynamic_bh_6m_stocks,
                        positions=dynamic_bh_6m_positions,
                        cash=dynamic_bh_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_bh_6m_stocks  # Force initial allocation
                    )
                    dynamic_bh_6m_transaction_costs += rebalance_costs
                    dynamic_bh_6m_last_rebalance_value = _mark_to_market_value(
                        dynamic_bh_6m_positions, dynamic_bh_6m_cash, ticker_data_grouped, current_date
                    )

            except Exception as e:
                print(f"   ⚠️ Dynamic BH 6M error: {e}")

        # DYNAMIC BH 3M PORTFOLIO: Rebalance to current top N performers DAILY
        if ENABLE_DYNAMIC_BH_3M:
            # Apply performance filters if enabled
            from performance_filters import filter_tickers_by_performance
            filtered_tickers_3m = filter_tickers_by_performance(
                initial_top_tickers, ticker_data_grouped, current_date, "Dynamic BH 3M"
            )
            
            print(f"   🔍 Dynamic BH 3M: Processing {len(filtered_tickers_3m)} tickers (filtered from {len(initial_top_tickers)}) for 3-month performance...")
            from parallel_backtest import calculate_parallel_performance
            current_top_performers_3m = calculate_parallel_performance(
                filtered_tickers_3m, ticker_data_grouped, current_date, 90)

            # Sort by performance and get top N
            if current_top_performers_3m:
                current_top_performers_3m.sort(key=lambda x: x[1], reverse=True)
                new_dynamic_bh_3m_stocks = [ticker for ticker, perf in current_top_performers_3m[:PORTFOLIO_SIZE]]
                print(f"   🏆 Top {PORTFOLIO_SIZE} performers (3-month): {', '.join(new_dynamic_bh_3m_stocks)}")
                
                # Use universal smart rebalancing function
                dynamic_bh_3m_positions, dynamic_bh_3m_cash, current_dynamic_bh_3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 3M",
                    current_stocks=current_dynamic_bh_3m_stocks,
                    new_stocks=new_dynamic_bh_3m_stocks,
                    positions=dynamic_bh_3m_positions,
                    cash=dynamic_bh_3m_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_3m_stocks  # Force initial allocation
                )
                dynamic_bh_3m_transaction_costs += rebalance_costs
                dynamic_bh_3m_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_3m_positions, dynamic_bh_3m_cash, ticker_data_grouped, current_date
                )

        # DYNAMIC BH 1M PORTFOLIO: Rebalance to current top N performers DAILY
        if ENABLE_DYNAMIC_BH_1M:
            # Apply performance filters if enabled
            from performance_filters import filter_tickers_by_performance
            filtered_tickers_1m = filter_tickers_by_performance(
                initial_top_tickers, ticker_data_grouped, current_date, "Dynamic BH 1M"
            )
            
            print(f"   🔍 Dynamic BH 1M: Processing {len(filtered_tickers_1m)} tickers (filtered from {len(initial_top_tickers)}) for 1-month performance...")
            from parallel_backtest import calculate_parallel_performance
            current_top_performers_1m = calculate_parallel_performance(
                filtered_tickers_1m, ticker_data_grouped, current_date, 21)

            # Sort by performance and get top N
            if current_top_performers_1m:
                current_top_performers_1m.sort(key=lambda x: x[1], reverse=True)
                new_dynamic_bh_1m_stocks = [ticker for ticker, perf in current_top_performers_1m[:PORTFOLIO_SIZE]]
                print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-month): {', '.join(new_dynamic_bh_1m_stocks)}")
                
                # Use universal smart rebalancing function
                dynamic_bh_1m_positions, dynamic_bh_1m_cash, current_dynamic_bh_1m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1M",
                    current_stocks=current_dynamic_bh_1m_stocks,
                    new_stocks=new_dynamic_bh_1m_stocks,
                    positions=dynamic_bh_1m_positions,
                    cash=dynamic_bh_1m_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1m_stocks  # Force initial allocation
                )
                dynamic_bh_1m_transaction_costs += rebalance_costs
                dynamic_bh_1m_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1m_positions, dynamic_bh_1m_cash, ticker_data_grouped, current_date
                )

        # DYNAMIC BH 1Y + VOLATILITY FILTER: Same as Dynamic BH 1Y but with volatility filter
        if ENABLE_DYNAMIC_BH_1Y_VOL_FILTER:
            current_top_performers_vol_filter = []
            stocks_evaluated = 0
            stocks_passed_vol_filter = 0
            
            # Calculate performances (always parallel)
            from parallel_backtest import calculate_parallel_performance
            base_performances = calculate_parallel_performance(
                initial_top_tickers,
                ticker_data_grouped,
                current_date,
                period_days=365
            )
            
            # Apply volatility filter
            for ticker, perf_pct in base_performances:
                try:
                    ticker_data = ticker_data_grouped[ticker]
                    perf_start_date = current_date - timedelta(days=MIN_DATA_DAYS_1Y)
                    perf_data = ticker_data.loc[perf_start_date:current_date]
                    
                    if len(perf_data) >= 50:
                        valid_close = perf_data['Close'].dropna()
                        if len(valid_close) >= 2:
                            # Calculate annualized volatility
                            daily_returns = valid_close.pct_change(fill_method=None).dropna()
                            if len(daily_returns) > 10:
                                # Annualize volatility: std_dev * sqrt(252) * 100%
                                annualized_volatility = daily_returns.std() * (252 ** 0.5) * 100
                                
                                # Apply volatility filter
                                if annualized_volatility <= DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY:
                                    current_top_performers_vol_filter.append((ticker, perf_pct, annualized_volatility))
                                    stocks_passed_vol_filter += 1
                except Exception:
                    continue

            # Sort by performance and get top N (from filtered list)
            if current_top_performers_vol_filter:
                current_top_performers_vol_filter.sort(key=lambda x: x[1], reverse=True)
                new_dynamic_bh_1y_vol_filter_stocks = [ticker for ticker, perf, vol in current_top_performers_vol_filter[:PORTFOLIO_SIZE]]
                
                # Show volatility info
                vol_info = [(ticker, f"{perf:.1f}%", f"{vol:.1f}%") for ticker, perf, vol in current_top_performers_vol_filter[:PORTFOLIO_SIZE]]
                print(f"   🎯 Top {PORTFOLIO_SIZE} performers (1Y + Vol Filter): {', '.join([f'{t}({p}/{v})' for t, p, v in vol_info])}")
                print(f"   🔍 Filter stats: {stocks_passed_vol_filter}/{stocks_evaluated} stocks passed volatility filter")

                # Use universal smart rebalancing function
                dynamic_bh_1y_vol_filter_positions, dynamic_bh_1y_vol_filter_cash, current_dynamic_bh_1y_vol_filter_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1Y+Vol",
                    current_stocks=current_dynamic_bh_1y_vol_filter_stocks,
                    new_stocks=new_dynamic_bh_1y_vol_filter_stocks,
                    positions=dynamic_bh_1y_vol_filter_positions,
                    cash=dynamic_bh_1y_vol_filter_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1y_vol_filter_stocks  # Force initial allocation
                )
                dynamic_bh_1y_vol_filter_transaction_costs += rebalance_costs
                dynamic_bh_1y_vol_filter_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1y_vol_filter_positions, dynamic_bh_1y_vol_filter_cash, ticker_data_grouped, current_date
                )
            else:
                print(f"   ⚠️ No stocks passed volatility filter (max {DYNAMIC_BH_1Y_VOL_FILTER_MAX_VOLATILITY:.1f}% annualized)")
                print(f"   🔍 Filter stats: {stocks_passed_vol_filter}/{stocks_evaluated} stocks passed volatility filter")
                # Debug: show what volatilities we saw
                if current_top_performers_vol_filter:
                    vol_debug = [(ticker, f"{vol:.1f}%") for ticker, perf, vol in current_top_performers_vol_filter[:5]]
                    print(f"   🔍 Sample volatilities: {', '.join([f'{t}({v})' for t, v in vol_debug])}")
                else:
                    print(f"   🔍 No stocks had valid volatility calculations")

        # DYNAMIC BH 1Y + TRAILING STOP: Same as Dynamic BH 1Y but with 20% trailing stop
        if ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP:
            # First, check trailing stops on existing positions
            positions_to_sell = []
            for ticker in list(current_dynamic_bh_1y_trailing_stop_stocks):
                if ticker in dynamic_bh_1y_trailing_stop_positions:
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is None or ticker_data.empty:
                            continue
                        ticker_data = ticker_data.loc[ticker_data.index == current_date]
                        if len(ticker_data) > 0:
                            current_price = ticker_data['Close'].iloc[0]
                            position = dynamic_bh_1y_trailing_stop_positions[ticker]
                            peak_price = position.get('peak_price', position['entry_price'])
                            
                            # Update peak price if current price is higher
                            if current_price > peak_price:
                                dynamic_bh_1y_trailing_stop_positions[ticker]['peak_price'] = current_price
                                peak_price = current_price
                            
                            # Check if trailing stop triggered (20% drop from peak)
                            stop_price = peak_price * (1 - DYNAMIC_BH_1Y_TRAILING_STOP_PERCENT / 100)
                            if current_price <= stop_price:
                                positions_to_sell.append(ticker)
                                print(f"   🛑 Trailing stop triggered for {ticker}: ${current_price:.2f} <= ${stop_price:.2f} (peak: ${peak_price:.2f})")
                    except Exception as e:
                        continue
            
            # Sell positions that hit trailing stop
            for ticker in positions_to_sell:
                if ticker in dynamic_bh_1y_trailing_stop_positions:
                    position = dynamic_bh_1y_trailing_stop_positions[ticker]
                    shares = position['shares']
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is None or ticker_data.empty:
                            continue
                        ticker_data = ticker_data.loc[ticker_data.index == current_date]
                        if len(ticker_data) > 0:
                            current_price = ticker_data['Close'].iloc[0]
                            sell_value = shares * current_price
                            sell_cost = sell_value * TRANSACTION_COST
                            dynamic_bh_1y_trailing_stop_transaction_costs += sell_cost
                            dynamic_bh_1y_trailing_stop_cash += (sell_value - sell_cost)
                            del dynamic_bh_1y_trailing_stop_positions[ticker]
                            current_dynamic_bh_1y_trailing_stop_stocks.remove(ticker)
                    except Exception as e:
                        continue
            
            # Regular rebalancing (same as Dynamic BH 1Y) - always parallel
            from parallel_backtest import calculate_parallel_performance
            current_top_performers_ts = calculate_parallel_performance(
                initial_top_tickers, ticker_data_grouped, current_date, period_days=365
            )

            if current_top_performers_ts:
                current_top_performers_ts.sort(key=lambda x: x[1], reverse=True)
                new_dynamic_bh_1y_trailing_stop_stocks = [ticker for ticker, perf in current_top_performers_ts[:PORTFOLIO_SIZE]]
                print(f"   🏆 Top {PORTFOLIO_SIZE} performers (1-year + Trailing Stop): {', '.join(new_dynamic_bh_1y_trailing_stop_stocks)}")

                # Use universal smart rebalancing function
                dynamic_bh_1y_trailing_stop_positions, dynamic_bh_1y_trailing_stop_cash, current_dynamic_bh_1y_trailing_stop_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="Dynamic BH 1Y+TS",
                    current_stocks=current_dynamic_bh_1y_trailing_stop_stocks,
                    new_stocks=new_dynamic_bh_1y_trailing_stop_stocks,
                    positions=dynamic_bh_1y_trailing_stop_positions,
                    cash=dynamic_bh_1y_trailing_stop_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_dynamic_bh_1y_trailing_stop_stocks  # Force initial allocation
                )
                dynamic_bh_1y_trailing_stop_transaction_costs += rebalance_costs
                dynamic_bh_1y_trailing_stop_last_rebalance_value = _mark_to_market_value(
                    dynamic_bh_1y_trailing_stop_positions, dynamic_bh_1y_trailing_stop_cash, ticker_data_grouped, current_date
                )

        # SECTOR ROTATION: Rebalance to top performing sector ETFs
        if ENABLE_SECTOR_ROTATION:
            # Check if it's time to rebalance (or if this is initial allocation)
            days_since_rebalance = (current_date - sector_rotation_last_rebalance_date).days if 'sector_rotation_last_rebalance_date' in locals() else AI_REBALANCE_FREQUENCY_DAYS
            is_initial_allocation = not current_sector_rotation_etfs  # Force day 1 investment
            
            if is_initial_allocation or days_since_rebalance >= AI_REBALANCE_FREQUENCY_DAYS:
                print(f"   🏢 Sector Rotation rebalancing (every {AI_REBALANCE_FREQUENCY_DAYS} days)...")
                
                # Select top sector ETFs based on momentum
                from shared_strategies import select_sector_rotation_etfs
                new_sector_rotation_etfs = select_sector_rotation_etfs(
                    initial_top_tickers, ticker_data_grouped, current_date, PORTFOLIO_SIZE
                )
                
                if new_sector_rotation_etfs:
                    # Use universal smart rebalancing function
                    sector_rotation_positions, sector_rotation_cash, current_sector_rotation_etfs, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Sector Rotation",
                        current_stocks=current_sector_rotation_etfs,
                        new_stocks=new_sector_rotation_etfs,
                        positions=sector_rotation_positions,
                        cash=sector_rotation_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_sector_rotation_etfs  # Force initial allocation
                    )
                    sector_rotation_transaction_costs += rebalance_costs
                    sector_rotation_last_rebalance_date = current_date
                    sector_rotation_last_rebalance_value = _mark_to_market_value(
                        sector_rotation_positions, sector_rotation_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No sector ETFs selected for rotation")
            else:
                print(f"   ⏭️ Skip Sector Rotation rebalance: {days_since_rebalance} days since last (rebalance every {AI_REBALANCE_FREQUENCY_DAYS} days)")

        # RISK-ADJUSTED MOMENTUM: Rebalance to current top N using shared strategy
        if ENABLE_RISK_ADJ_MOM:
            from shared_strategies import select_risk_adj_mom_stocks
            
            # Use shared strategy for consistent selection
            new_risk_adj_mom_stocks = select_risk_adj_mom_stocks(
                initial_top_tickers, 
                ticker_data_grouped, 
                current_date,
                top_n=PORTFOLIO_SIZE
            )

            if new_risk_adj_mom_stocks:
                # Use universal smart rebalancing function
                risk_adj_mom_positions, risk_adj_mom_cash, current_risk_adj_mom_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="Risk-Adj Mom",
                    current_stocks=current_risk_adj_mom_stocks,
                    new_stocks=new_risk_adj_mom_stocks,
                    positions=risk_adj_mom_positions,
                    cash=risk_adj_mom_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_risk_adj_mom_stocks  # Force initial allocation
                )
                risk_adj_mom_transaction_costs += rebalance_costs
                risk_adj_mom_last_rebalance_value = _mark_to_market_value(
                    risk_adj_mom_positions, risk_adj_mom_cash, ticker_data_grouped, current_date
                )

        # === MULTI-TASK LEARNING STRATEGY ===
        # Runs independently of Dynamic BH performance data
        if ENABLE_MULTITASK_LEARNING:
            from shared_strategies import select_multitask_learning_stocks
            
            # Calculate training end date (day before current date to avoid lookahead bias)
            multitask_train_end = current_date - timedelta(days=1)
            
            try:
                # Use multi-task learning strategy
                new_multitask_stocks = select_multitask_learning_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped, 
                    current_date,
                    multitask_train_end,    # Required for multi-task training
                    top_n=PORTFOLIO_SIZE
                )

                if new_multitask_stocks:
                    # Use universal smart rebalancing function
                    multitask_positions, multitask_cash, current_multitask_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Multi-Task Learning",
                        current_stocks=current_multitask_stocks,
                        new_stocks=new_multitask_stocks,
                        positions=multitask_positions,
                        cash=multitask_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_multitask_stocks  # Force initial allocation
                    )
                    multitask_transaction_costs += rebalance_costs
                    multitask_last_rebalance_value = _mark_to_market_value(
                        multitask_positions, multitask_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ⚠️ Multi-Task Learning: No stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Multi-Task Learning strategy error: {e}")

        # 3M/1Y RATIO: Rebalance to highest 3M/1Y ratio stocks DAILY
        if ENABLE_3M_1Y_RATIO:
            try:
                from shared_strategies import select_3m_1y_ratio_stocks
                
                print(f"   📊 3M/1Y Ratio Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Use shared strategy for consistent selection
                new_ratio_3m_1y_stocks = select_3m_1y_ratio_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped, 
                    current_date,  # Add the current date parameter!
                    top_n=PORTFOLIO_SIZE  # Use configured portfolio size (default 10)
                )
                
                if new_ratio_3m_1y_stocks:
                    # Use universal smart rebalancing function
                    ratio_3m_1y_positions, ratio_3m_1y_cash, current_ratio_3m_1y_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="3M/1Y Ratio",
                        current_stocks=current_ratio_3m_1y_stocks,
                        new_stocks=new_ratio_3m_1y_stocks,
                        positions=ratio_3m_1y_positions,
                        cash=ratio_3m_1y_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_ratio_3m_1y_stocks  # Force initial allocation
                    )
                    ratio_3m_1y_transaction_costs += rebalance_costs
                    ratio_3m_1y_last_rebalance_value = _mark_to_market_value(
                        ratio_3m_1y_positions, ratio_3m_1y_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No 3M/1Y Ratio stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ 3M/1Y Ratio strategy error: {e}")

        # 1Y/3M RATIO: Rebalance to highest 1Y/3M ratio stocks DAILY
        try:
            from shared_strategies import select_1y_3m_ratio_stocks
            
            print(f"   📊 1Y/3M Ratio Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
            
            # Use shared strategy for consistent selection
            new_ratio_1y_3m_stocks = select_1y_3m_ratio_stocks(
                initial_top_tickers, 
                ticker_data_grouped, 
                current_date,  # Add the current date parameter!
                top_n=PORTFOLIO_SIZE  # Use configured portfolio size (default 10)
            )
            
            if new_ratio_1y_3m_stocks:
                # Use universal smart rebalancing function
                ratio_1y_3m_positions, ratio_1y_3m_cash, current_ratio_1y_3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                    strategy_name="1Y/3M Ratio",
                    current_stocks=current_ratio_1y_3m_stocks,
                    new_stocks=new_ratio_1y_3m_stocks,
                    positions=ratio_1y_3m_positions,
                    cash=ratio_1y_3m_cash,
                    ticker_data_grouped=ticker_data_grouped,
                    current_date=current_date,
                    transaction_cost=TRANSACTION_COST,
                    portfolio_size=PORTFOLIO_SIZE,
                    force_rebalance=not current_ratio_1y_3m_stocks  # Force initial allocation
                )
                ratio_1y_3m_transaction_costs += rebalance_costs
                ratio_1y_3m_last_rebalance_value = _mark_to_market_value(
                    ratio_1y_3m_positions, ratio_1y_3m_cash, ticker_data_grouped, current_date
                )
            else:
                print(f"   ❌ No 1Y/3M Ratio stocks selected")
                
        except Exception as e:
            print(f"   ⚠️ 1Y/3M Ratio strategy error: {e}")

        # TURNAROUND: Rebalance to best turnaround stocks DAILY
        if ENABLE_TURNAROUND:
            try:
                from shared_strategies import select_turnaround_stocks
                
                print(f"   📊 Turnaround Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Use shared strategy for consistent selection
                new_turnaround_stocks = select_turnaround_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped, 
                    current_date,  # Add the current date parameter!
                    top_n=PORTFOLIO_SIZE  # Use configured portfolio size (default 10)
                )
                
                if new_turnaround_stocks:
                    # Use universal smart rebalancing function
                    turnaround_positions, turnaround_cash, current_turnaround_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Turnaround",
                        current_stocks=current_turnaround_stocks,
                        new_stocks=new_turnaround_stocks,
                        positions=turnaround_positions,
                        cash=turnaround_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=min(len(new_turnaround_stocks), 15),  # Allow up to 15 positions
                        force_rebalance=not current_turnaround_stocks  # Force initial allocation
                    )
                    turnaround_transaction_costs += rebalance_costs
                    turnaround_last_rebalance_value = _mark_to_market_value(
                        turnaround_positions, turnaround_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Turnaround stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Turnaround strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID: Rebalance using hybrid strategy DAILY
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_stocks
                
                print(f"   🎯 Momentum-Volatility Hybrid Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Use momentum-volatility hybrid for stock selection
                new_momentum_volatility_hybrid_stocks = select_momentum_volatility_hybrid_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE  # Use configured portfolio size (default 10)
                )
                
                if new_momentum_volatility_hybrid_stocks:
                    # Use universal smart rebalancing function
                    momentum_volatility_hybrid_positions, momentum_volatility_hybrid_cash, current_momentum_volatility_hybrid_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Momentum-Vol Hybrid",
                        current_stocks=current_momentum_volatility_hybrid_stocks,
                        new_stocks=new_momentum_volatility_hybrid_stocks,
                        positions=momentum_volatility_hybrid_positions,
                        cash=momentum_volatility_hybrid_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_stocks  # Force initial allocation
                    )
                    momentum_volatility_hybrid_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_positions, momentum_volatility_hybrid_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Momentum-Volatility Hybrid stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Momentum-Volatility Hybrid strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 6M: Rebalance using hybrid strategy (6M lookback) DAILY
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_6m_stocks
                
                print(f"   🎯 Mom-Vol Hybrid 6M Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_momentum_volatility_hybrid_6m_stocks = select_momentum_volatility_hybrid_6m_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_momentum_volatility_hybrid_6m_stocks:
                    momentum_volatility_hybrid_6m_positions, momentum_volatility_hybrid_6m_cash, current_momentum_volatility_hybrid_6m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 6M",
                        current_stocks=current_momentum_volatility_hybrid_6m_stocks,
                        new_stocks=new_momentum_volatility_hybrid_6m_stocks,
                        positions=momentum_volatility_hybrid_6m_positions,
                        cash=momentum_volatility_hybrid_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_6m_stocks
                    )
                    momentum_volatility_hybrid_6m_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_6m_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_6m_positions, momentum_volatility_hybrid_6m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 6M stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 6M strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 1Y: Rebalance using hybrid strategy (1Y lookback) DAILY
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_1y_stocks
                
                print(f"   🎯 Mom-Vol Hybrid 1Y Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_momentum_volatility_hybrid_1y_stocks = select_momentum_volatility_hybrid_1y_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_momentum_volatility_hybrid_1y_stocks:
                    momentum_volatility_hybrid_1y_positions, momentum_volatility_hybrid_1y_cash, current_momentum_volatility_hybrid_1y_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 1Y",
                        current_stocks=current_momentum_volatility_hybrid_1y_stocks,
                        new_stocks=new_momentum_volatility_hybrid_1y_stocks,
                        positions=momentum_volatility_hybrid_1y_positions,
                        cash=momentum_volatility_hybrid_1y_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_1y_stocks
                    )
                    momentum_volatility_hybrid_1y_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_1y_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_1y_positions, momentum_volatility_hybrid_1y_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 1Y stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 1Y strategy error: {e}")

        # MOMENTUM-VOLATILITY HYBRID 1Y/3M: Rebalance using hybrid strategy (1Y/3M ratio) DAILY
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M:
            try:
                from shared_strategies import select_momentum_volatility_hybrid_1y3m_stocks
                
                print(f"   🎯 Mom-Vol Hybrid 1Y/3M Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_momentum_volatility_hybrid_1y3m_stocks = select_momentum_volatility_hybrid_1y3m_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_momentum_volatility_hybrid_1y3m_stocks:
                    momentum_volatility_hybrid_1y3m_positions, momentum_volatility_hybrid_1y3m_cash, current_momentum_volatility_hybrid_1y3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Mom-Vol Hybrid 1Y/3M",
                        current_stocks=current_momentum_volatility_hybrid_1y3m_stocks,
                        new_stocks=new_momentum_volatility_hybrid_1y3m_stocks,
                        positions=momentum_volatility_hybrid_1y3m_positions,
                        cash=momentum_volatility_hybrid_1y3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_momentum_volatility_hybrid_1y3m_stocks
                    )
                    momentum_volatility_hybrid_1y3m_transaction_costs += rebalance_costs
                    momentum_volatility_hybrid_1y3m_last_rebalance_value = _mark_to_market_value(
                        momentum_volatility_hybrid_1y3m_positions, momentum_volatility_hybrid_1y3m_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Mom-Vol Hybrid 1Y/3M stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Mom-Vol Hybrid 1Y/3M strategy error: {e}")

        # PRICE ACCELERATION: Rebalance using velocity/acceleration strategy DAILY
        if ENABLE_PRICE_ACCELERATION:
            try:
                from shared_strategies import select_price_acceleration_stocks
                
                print(f"   🚀 Price Acceleration Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Use price acceleration for stock selection
                new_price_acceleration_stocks = select_price_acceleration_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE  # Use configured portfolio size (default 10)
                )
                
                if new_price_acceleration_stocks:
                    # Use universal smart rebalancing function
                    price_acceleration_positions, price_acceleration_cash, current_price_acceleration_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Price Acceleration",
                        current_stocks=current_price_acceleration_stocks,
                        new_stocks=new_price_acceleration_stocks,
                        positions=price_acceleration_positions,
                        cash=price_acceleration_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_price_acceleration_stocks  # Force initial allocation
                    )
                    price_acceleration_transaction_costs += rebalance_costs
                    price_acceleration_last_rebalance_value = _mark_to_market_value(
                        price_acceleration_positions, price_acceleration_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Price Acceleration stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Price Acceleration strategy error: {e}")

        # ADAPTIVE ENSEMBLE: Rebalance using meta-ensemble strategy DAILY
        if ENABLE_ADAPTIVE_STRATEGY:
            try:
                from adaptive_ensemble import select_adaptive_ensemble_stocks, reset_ensemble_state
                
                print(f"   📊 Adaptive Ensemble Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Use adaptive ensemble for stock selection
                new_adaptive_ensemble_stocks = select_adaptive_ensemble_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_adaptive_ensemble_stocks:
                    # Use universal smart rebalancing function
                    adaptive_ensemble_positions, adaptive_ensemble_cash, current_adaptive_ensemble_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Adaptive Ensemble",
                        current_stocks=current_adaptive_ensemble_stocks,
                        new_stocks=new_adaptive_ensemble_stocks,
                        positions=adaptive_ensemble_positions,
                        cash=adaptive_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_adaptive_ensemble_stocks  # Force initial allocation on day 1
                    )
                    adaptive_ensemble_transaction_costs += rebalance_costs
                    adaptive_ensemble_last_rebalance_value = _mark_to_market_value(
                        adaptive_ensemble_positions, adaptive_ensemble_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Adaptive Ensemble stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Adaptive Ensemble strategy error: {e}")
                import traceback
                traceback.print_exc()

        # VOLATILITY ENSEMBLE: Rebalance using volatility-adjusted strategy DAILY
        if ENABLE_VOLATILITY_ENSEMBLE:
            try:
                from volatility_ensemble import select_volatility_ensemble_stocks, reset_vol_ensemble_state
                
                print(f"   📊 Volatility Ensemble Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_volatility_ensemble_stocks = select_volatility_ensemble_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                                        top_n=PORTFOLIO_SIZE
                )
                
                if new_volatility_ensemble_stocks:
                    # Use universal smart rebalancing function
                    volatility_ensemble_positions, volatility_ensemble_cash, current_volatility_ensemble_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Volatility Ensemble",
                        current_stocks=current_volatility_ensemble_stocks,
                        new_stocks=new_volatility_ensemble_stocks,
                        positions=volatility_ensemble_positions,
                        cash=volatility_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_volatility_ensemble_stocks  # Force initial allocation on day 1
                    )
                    volatility_ensemble_transaction_costs += rebalance_costs
                    volatility_ensemble_last_rebalance_value = _mark_to_market_value(
                        volatility_ensemble_positions, volatility_ensemble_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Volatility Ensemble stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Volatility Ensemble strategy error: {e}")

        # CORRELATION ENSEMBLE: Select stocks with low correlation for diversification
        if ENABLE_CORRELATION_ENSEMBLE:
            try:
                print(f"   📊 Correlation Ensemble: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                # Calculate correlation-filtered stocks (select stocks with low correlation to each other)
                correlation_scores = []
                
                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]
                        
                        # Get 60-day returns for correlation calculation
                        lookback_start = current_date - timedelta(days=60)
                        data_slice = ticker_data.loc[lookback_start:current_date]
                        
                        if len(data_slice) >= 30:
                            returns = data_slice['Close'].pct_change().dropna()
                            if len(returns) >= 20:
                                # Calculate momentum (positive returns preferred)
                                momentum = (data_slice['Close'].iloc[-1] / data_slice['Close'].iloc[0]) - 1
                                # Calculate volatility
                                volatility = returns.std() * np.sqrt(252)
                                # Score: momentum adjusted by volatility
                                # Only include stocks with positive momentum
                                if volatility > 0 and momentum > 0:
                                    score = momentum / volatility
                                    correlation_scores.append((ticker, score, returns))
                    except Exception:
                        continue
                
                if correlation_scores:
                    # Sort by score and select top candidates
                    correlation_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select stocks with low correlation to each other
                    selected_stocks = []
                    for ticker, score, returns in correlation_scores:
                        if len(selected_stocks) >= PORTFOLIO_SIZE:
                            break
                        
                        # Check correlation with already selected stocks
                        is_correlated = False
                        for sel_ticker, _, sel_returns in [s for s in correlation_scores if s[0] in selected_stocks]:
                            try:
                                corr = returns.corr(sel_returns)
                                if abs(corr) > 0.7:  # High correlation threshold
                                    is_correlated = True
                                    break
                            except Exception:
                                pass
                        
                        if not is_correlated:
                            selected_stocks.append(ticker)
                    
                    new_correlation_ensemble_stocks = selected_stocks
                    
                    if new_correlation_ensemble_stocks:
                        print(f"   🎯 Correlation Ensemble: Selected {len(new_correlation_ensemble_stocks)} low-correlation stocks")
                        
                        # Use universal smart rebalancing function
                        correlation_ensemble_positions, correlation_ensemble_cash, current_correlation_ensemble_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Correlation Ensemble",
                            current_stocks=current_correlation_ensemble_stocks,
                            new_stocks=new_correlation_ensemble_stocks,
                            positions=correlation_ensemble_positions,
                            cash=correlation_ensemble_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_correlation_ensemble_stocks  # Force initial allocation on day 1
                        )
                        correlation_ensemble_transaction_costs += rebalance_costs
                        correlation_ensemble_last_rebalance_value = _mark_to_market_value(
                            correlation_ensemble_positions, correlation_ensemble_cash, ticker_data_grouped, current_date
                        )
                    else:
                        print(f"   ❌ No low-correlation stocks found")
                else:
                    print(f"   ❌ No correlation data available")
                    
            except Exception as e:
                print(f"   ⚠️ Correlation Ensemble strategy error: {e}")

        # DYNAMIC POOL: Rebalance using dynamic strategy pool DAILY
        if ENABLE_DYNAMIC_POOL:
            try:
                from dynamic_pool import select_dynamic_pool_stocks
                
                print(f"   📊 Dynamic Pool Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_dynamic_pool_stocks = select_dynamic_pool_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                                        top_n=PORTFOLIO_SIZE
                )
                
                if new_dynamic_pool_stocks:
                    # Use universal smart rebalancing function
                    dynamic_pool_positions, dynamic_pool_cash, current_dynamic_pool_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Dynamic Pool",
                        current_stocks=current_dynamic_pool_stocks,
                        new_stocks=new_dynamic_pool_stocks,
                        positions=dynamic_pool_positions,
                        cash=dynamic_pool_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dynamic_pool_stocks  # Force initial allocation on day 1
                    )
                    dynamic_pool_transaction_costs += rebalance_costs
                    dynamic_pool_last_rebalance_value = _mark_to_market_value(
                        dynamic_pool_positions, dynamic_pool_cash, ticker_data_grouped, current_date
                    )
                else:
                    print(f"   ❌ No Dynamic Pool stocks selected")
                    
            except Exception as e:
                print(f"   ⚠️ Dynamic Pool strategy error: {e}")

        # RISK-ADJ MOMENTUM SENTIMENT: Rebalance using risk-adjusted momentum + sentiment strategy DAILY
        if ENABLE_RISK_ADJ_MOM_SENTIMENT:
            try:
                from risk_adj_mom_sentiment import select_risk_adj_mom_sentiment_stocks
                
                print(f"   📊 Risk-Adj Mom Sentiment Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_risk_adj_mom_sentiment_stocks = select_risk_adj_mom_sentiment_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped, 
                    current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_risk_adj_mom_sentiment_stocks:
                    # Use universal smart rebalancing function
                    risk_adj_mom_sentiment_positions, risk_adj_mom_sentiment_cash, current_risk_adj_mom_sentiment_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom Sentiment",
                        current_stocks=current_risk_adj_mom_sentiment_stocks,
                        new_stocks=new_risk_adj_mom_sentiment_stocks,
                        positions=risk_adj_mom_sentiment_positions,
                        cash=risk_adj_mom_sentiment_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        buffer_size=PORTFOLIO_BUFFER_SIZE,
                        strategy_stop_loss=STRATEGY_STOP_LOSS_PCT.get('Risk-Adj Mom Sentiment', STOP_LOSS_PCT)
                    )
                    
                    risk_adj_mom_sentiment_transaction_costs += rebalance_costs
                    risk_adj_mom_sentiment_last_rebalance_value = risk_adj_mom_sentiment_portfolio_value
                
            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom Sentiment strategy error: {e}")

        # VOTING ENSEMBLE: Rebalance using consensus voting strategy DAILY
        if ENABLE_VOTING_ENSEMBLE:
            try:
                from shared_strategies import select_voting_ensemble_stocks
                
                print(f"   🗳️  Voting Ensemble Strategy: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")
                
                new_voting_ensemble_stocks = select_voting_ensemble_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_voting_ensemble_stocks:
                    # Use universal smart rebalancing function
                    voting_ensemble_positions, voting_ensemble_cash, current_voting_ensemble_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Voting Ensemble",
                        current_stocks=current_voting_ensemble_stocks,
                        new_stocks=new_voting_ensemble_stocks,
                        positions=voting_ensemble_positions,
                        cash=voting_ensemble_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_voting_ensemble_stocks  # Force initial allocation
                    )
                    voting_ensemble_transaction_costs += rebalance_costs
                    voting_ensemble_last_rebalance_value = voting_ensemble_portfolio_value
                
            except Exception as e:
                print(f"   ⚠️ Voting Ensemble strategy error: {e}")

        # === MEAN REVERSION, QUALITY+MOM, VOL-ADJ MOM STRATEGIES ===
        # These strategies run independently of Dynamic BH performance data
        
        # MEAN REVERSION: Rebalance to bottom N performers DAILY
        if ENABLE_MEAN_REVERSION:
            try:
                # Calculate current bottom N performers based on recent short-term performance
                # Mean reversion: buy stocks that have declined recently (expecting bounce back)
                current_bottom_performers = []
                
                print(f"   🔍 Mean Reversion: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # ✅ OPTIMIZED: Use pre-grouped data
                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]
                        
                        # Use 1-month performance for mean reversion (opposite of momentum)
                        # ✅ FIX: Use explicit date range like other working strategies
                        perf_start_date_mr = current_date - timedelta(days=30)
                        data_slice = ticker_data.loc[perf_start_date_mr:current_date]
                        if len(data_slice) >= 10:  # Relaxed: at least 10 days of data
                            recent_data = data_slice.tail(21) if len(data_slice) >= 21 else data_slice
                            if len(recent_data) >= 2:
                                start_price = recent_data['Close'].iloc[0]
                                end_price = recent_data['Close'].iloc[-1]
                                if start_price > 0:
                                    monthly_return = ((end_price - start_price) / start_price) * 100
                                    current_bottom_performers.append((ticker, monthly_return))

                    except Exception as e:
                        continue

                print(f"   📊 Mean Reversion: Found {len(current_bottom_performers)} tickers with valid data")

                if current_bottom_performers:
                    current_bottom_performers.sort(key=lambda x: x[1])  # Sort by return (ascending = worst performers)
                    # FIXED: Select stocks with moderate losses (not worst performers) for mean reversion
                    # Filter for stocks with -20% to -5% returns (oversold but not crashing)
                    moderate_losers = [t for t, ret in current_bottom_performers if -20 <= ret <= -5]
                    new_mean_reversion_stocks = [ticker for ticker in moderate_losers[:PORTFOLIO_SIZE]]
                    print(f"   🎯 Mean Reversion: Selected {new_mean_reversion_stocks} (from {len(moderate_losers)} moderate losers)")

                    if new_mean_reversion_stocks:
                        # Use universal smart rebalancing function
                        mean_reversion_positions, mean_reversion_cash, current_mean_reversion_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Mean Reversion",
                            current_stocks=current_mean_reversion_stocks,
                            new_stocks=new_mean_reversion_stocks,
                            positions=mean_reversion_positions,
                            cash=mean_reversion_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_mean_reversion_stocks  # Force initial allocation
                        )
                        mean_reversion_transaction_costs += rebalance_costs
                        mean_reversion_last_rebalance_value = _mark_to_market_value(
                            mean_reversion_positions, mean_reversion_cash, ticker_data_grouped, current_date
                        )
                else:
                    print(f"   ⚠️ Mean Reversion: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Mean reversion selection failed: {e}")

        # QUALITY + MOMENTUM: Rebalance to top performers by combined quality+momentum score DAILY
        if ENABLE_QUALITY_MOM:
            try:
                # Calculate combined quality + momentum scores
                quality_momentum_scores = []
                
                print(f"   🔍 Quality+Mom: Analyzing {len(initial_top_tickers)} tickers on {current_date.strftime('%Y-%m-%d')}")

                # ✅ OPTIMIZED: Use pre-grouped data
                for ticker in initial_top_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_data = ticker_data_grouped[ticker]
                        
                        # Use 3-month period for both quality and momentum assessment
                        # ✅ FIX: Use explicit date range like other working strategies
                        perf_start_date_qm = current_date - timedelta(days=90)
                        data_slice = ticker_data.loc[perf_start_date_qm:current_date]
                        if len(data_slice) >= 30:  # Relaxed: at least 30 days of data
                            recent_data = data_slice.tail(63) if len(data_slice) >= 63 else data_slice

                            if len(recent_data) >= 10:
                                # MOMENTUM SCORE: 3-month return
                                start_price = recent_data['Close'].iloc[0]
                                end_price = recent_data['Close'].iloc[-1]
                                momentum_score = ((end_price - start_price) / start_price) * 100 if start_price > 0 else -100

                                # QUALITY SCORE: Consistency (low volatility) + trend strength
                                returns = recent_data['Close'].pct_change(fill_method=None).dropna()
                                if len(returns) > 5:
                                    # Volatility (lower = higher quality)
                                    volatility = returns.std() * np.sqrt(252)  # Annualized
                                    quality_volatility = max(0, 50 - volatility * 100)  # Higher score for lower volatility

                                    # Trend consistency (higher = higher quality)
                                    trend_strength = abs(momentum_score) * (1 - volatility)  # Strong trend with low volatility

                                    # Combined score: 70% momentum, 30% quality
                                    combined_score = (momentum_score * 0.7) + (quality_volatility * 0.3)

                                    quality_momentum_scores.append((ticker, combined_score, momentum_score, quality_volatility))

                    except Exception as e:
                        continue

                print(f"   📊 Quality+Mom: Found {len(quality_momentum_scores)} tickers with valid data")

                if quality_momentum_scores:
                    # Sort by combined score (descending)
                    quality_momentum_scores.sort(key=lambda x: x[1], reverse=True)
                    new_quality_momentum_stocks = [ticker for ticker, score, mom, qual in quality_momentum_scores[:PORTFOLIO_SIZE]]
                    print(f"   🎯 Quality+Mom: Selected {new_quality_momentum_stocks}")

                    if new_quality_momentum_stocks:
                        # Use universal smart rebalancing function
                        quality_momentum_positions, quality_momentum_cash, current_quality_momentum_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Quality+Mom",
                            current_stocks=current_quality_momentum_stocks,
                            new_stocks=new_quality_momentum_stocks,
                            positions=quality_momentum_positions,
                            cash=quality_momentum_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_quality_momentum_stocks  # Force initial allocation
                        )
                        quality_momentum_transaction_costs += rebalance_costs
                        quality_momentum_last_rebalance_value = _mark_to_market_value(
                            quality_momentum_positions, quality_momentum_cash, ticker_data_grouped, current_date
                        )
                else:
                    print(f"   ⚠️ Quality+Mom: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Quality + momentum selection failed: {e}")

        # VOLATILITY-ADJUSTED MOMENTUM: Rebalance to top performers by volatility-adjusted momentum DAILY
        if ENABLE_VOLATILITY_ADJ_MOM:
            try:
                # Apply performance filters if enabled
                from performance_filters import filter_tickers_by_performance
                available_tickers = initial_top_tickers if initial_top_tickers else []
                filtered_tickers = filter_tickers_by_performance(
                    available_tickers, ticker_data_grouped, current_date, "Vol-Adj Mom"
                )
                
                # Calculate volatility-adjusted momentum scores
                volatility_adj_mom_scores = []
                
                print(f"   🔍 Vol-Adj Mom: Analyzing {len(filtered_tickers)} tickers (filtered from {len(available_tickers)}) on {current_date.strftime('%Y-%m-%d')}")
                
                for ticker in filtered_tickers:
                    try:
                        if ticker not in ticker_data_grouped:
                            continue
                        ticker_history = ticker_data_grouped[ticker].reset_index()
                        # ✅ FIX: Use explicit date range like other working strategies
                        lookback_start = current_date - timedelta(days=VOLATILITY_ADJ_MOM_LOOKBACK + VOLATILITY_ADJ_MOM_VOL_WINDOW + 30)
                        ticker_history = ticker_history[
                            (ticker_history['date'] >= lookback_start) & 
                            (ticker_history['date'] <= current_date)
                        ]
                        
                        # Relaxed requirement: need at least 60 days instead of full lookback
                        min_required = min(60, VOLATILITY_ADJ_MOM_LOOKBACK + VOLATILITY_ADJ_MOM_VOL_WINDOW)
                        if len(ticker_history) >= min_required:
                            # Calculate volatility-adjusted momentum score
                            vol_adj_score = calculate_volatility_adjusted_momentum(
                                ticker_history, 
                                min(VOLATILITY_ADJ_MOM_LOOKBACK, len(ticker_history) - VOLATILITY_ADJ_MOM_VOL_WINDOW), 
                                VOLATILITY_ADJ_MOM_VOL_WINDOW
                            )
                            
                            # Relaxed threshold: accept any positive score
                            if vol_adj_score > 0:
                                volatility_adj_mom_scores.append((ticker, vol_adj_score))
                    
                    except Exception:
                        continue
                
                print(f"   📊 Vol-Adj Mom: Found {len(volatility_adj_mom_scores)} tickers with valid data")
                
                if volatility_adj_mom_scores:
                    # Sort by volatility-adjusted score and get top N
                    volatility_adj_mom_scores.sort(key=lambda x: x[1], reverse=True)
                    new_volatility_adj_mom_stocks = [ticker for ticker, score in volatility_adj_mom_scores[:PORTFOLIO_SIZE]]
                    print(f"   🎯 Vol-Adj Mom: Selected {new_volatility_adj_mom_stocks}")
                    
                    if new_volatility_adj_mom_stocks:
                        # Use universal smart rebalancing function
                        volatility_adj_mom_positions, volatility_adj_mom_cash, current_volatility_adj_mom_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Vol-Adj Mom",
                            current_stocks=current_volatility_adj_mom_stocks,
                            new_stocks=new_volatility_adj_mom_stocks,
                            positions=volatility_adj_mom_positions,
                            cash=volatility_adj_mom_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not current_volatility_adj_mom_stocks  # Force initial allocation
                        )
                        volatility_adj_mom_transaction_costs += rebalance_costs
                        volatility_adj_mom_last_rebalance_value = _mark_to_market_value(
                            volatility_adj_mom_positions, volatility_adj_mom_cash, ticker_data_grouped, current_date
                        )
                else:
                    print(f"   ⚠️ Vol-Adj Mom: No valid tickers found on {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"   ⚠️ Volatility-adjusted momentum selection failed: {e}")

        # === ENHANCED VOLATILITY TRADER STRATEGY ===
        if ENABLE_ENHANCED_VOLATILITY:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if 'last_enhanced_volatility_rebalance_day' not in locals():
                    last_enhanced_volatility_rebalance_day = 0
                
                if last_enhanced_volatility_rebalance_day == 0 or (day_count - last_enhanced_volatility_rebalance_day) >= 7:  # Weekly rebalance
                    print(f"\n🔄 Enhanced Volatility Trader: Evaluating portfolio (Day {day_count})...")
                    
                    # Calculate volatility-adjusted momentum scores
                    enhanced_vol_scores = []
                    available_tickers = initial_top_tickers if initial_top_tickers else []
                    
                    for ticker in available_tickers:
                        try:
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_data = ticker_data_grouped[ticker]
                            
                            # Get data for last 30 days
                            end_date = current_date
                            start_date = current_date - timedelta(days=30)
                            data_slice = ticker_data.loc[start_date:end_date]
                            
                            if len(data_slice) >= 20:
                                # Calculate returns
                                returns = data_slice['Close'].pct_change().dropna()
                                
                                if len(returns) > 0:
                                    # Calculate volatility (annualized)
                                    volatility = returns.std() * np.sqrt(252)
                                    
                                    # Calculate momentum (20-day return)
                                    momentum = (data_slice['Close'].iloc[-1] / data_slice['Close'].iloc[0]) - 1
                                    
                                    # Calculate ATR for stop loss
                                    high_low = data_slice['High'] - data_slice['Low']
                                    high_close = np.abs(data_slice['High'] - data_slice['Close'].shift())
                                    low_close = np.abs(data_slice['Low'] - data_slice['Close'].shift())
                                    atr = np.maximum(high_low, np.maximum(high_close, low_close)).mean()
                                    
                                    # Enhanced score: momentum adjusted by volatility (lower volatility = higher score)
                                    if volatility > 0 and momentum > 0:  # Only positive momentum
                                        enhanced_score = momentum / volatility
                                        enhanced_vol_scores.append((ticker, enhanced_score, volatility, atr))
                        
                        except Exception as e:
                            continue
                    
                    if enhanced_vol_scores:
                        # Sort by enhanced score (descending)
                        enhanced_vol_scores.sort(key=lambda x: x[1], reverse=True)
                        top_enhanced_vol_stocks = [(t, s, v, a) for t, s, v, a in enhanced_vol_scores[:PORTFOLIO_SIZE]]
                        
                        print(f"   📈 Top {PORTFOLIO_SIZE} Enhanced Volatility stocks: {[(t, f'{s*100:.2f}', f'{v*100:.1f}%') for t, s, v, a in top_enhanced_vol_stocks]}")
                        
                        # Rebalance portfolio
                        total_value = enhanced_volatility_cash + sum(pos.get('value', 0) for pos in enhanced_volatility_positions.values())
                        capital_per_stock = total_value / PORTFOLIO_SIZE if PORTFOLIO_SIZE > 0 else 0
                        
                        # Sell existing positions
                        for ticker in list(enhanced_volatility_positions.keys()):
                            if ticker not in [t for t, _, _, _ in top_enhanced_vol_stocks]:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        shares = enhanced_volatility_positions[ticker]['shares']
                                        gross_sale = shares * current_price
                                        sell_cost = gross_sale * TRANSACTION_COST
                                        enhanced_volatility_cash += gross_sale - sell_cost
                                        del enhanced_volatility_positions[ticker]
                                except Exception:
                                    continue
                        
                        # Buy new positions with ATR-based stops
                        for ticker, score, volatility, atr in top_enhanced_vol_stocks:
                            if ticker not in enhanced_volatility_positions and capital_per_stock > 0:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                                            if shares > 0:
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                enhanced_volatility_cash -= (buy_value + buy_cost)
                                                
                                                # Set ATR-based stop loss (2x ATR) and take profit (3x ATR)
                                                stop_loss = current_price - (2 * atr)
                                                take_profit = current_price + (3 * atr)
                                                
                                                enhanced_volatility_positions[ticker] = {
                                                    'shares': shares,
                                                    'entry_price': current_price,
                                                    'value': buy_value,
                                                    'stop_loss': stop_loss,
                                                    'take_profit': take_profit,
                                                    'atr': atr
                                                }
                                except Exception:
                                    continue
                        
                        last_enhanced_volatility_rebalance_day = day_count
                        print(f"   ✅ Enhanced Volatility: Rebalanced to {len(enhanced_volatility_positions)} positions")
                
                # Check stop losses and take profits daily
                positions_to_close = []
                for ticker, pos in enhanced_volatility_positions.items():
                    try:
                        ticker_data = ticker_data_grouped[ticker]
                        price_data = ticker_data.loc[:current_date]
                        if not price_data.empty:
                            current_price = price_data['Close'].dropna().iloc[-1]
                            
                            # Check stop loss
                            if current_price <= pos['stop_loss']:
                                positions_to_close.append((ticker, current_price, 'Stop Loss'))
                            # Check take profit
                            elif current_price >= pos['take_profit']:
                                positions_to_close.append((ticker, current_price, 'Take Profit'))
                    except Exception:
                        continue
                
                # Close positions that hit stop loss or take profit
                for ticker, price, reason in positions_to_close:
                    try:
                        shares = enhanced_volatility_positions[ticker]['shares']
                        gross_sale = shares * price
                        sell_cost = gross_sale * TRANSACTION_COST
                        enhanced_volatility_cash += gross_sale - sell_cost
                        del enhanced_volatility_positions[ticker]
                        print(f"   🛑 Enhanced Volatility: Sold {ticker} @ ${price:.2f} ({reason})")
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"   ⚠️ Enhanced Volatility Trader error: {e}")

        # === AI VOLATILITY ENSEMBLE STRATEGY ===
        if ENABLE_AI_VOLATILITY_ENSEMBLE:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if 'last_ai_vol_ensemble_rebalance_day' not in locals():
                    last_ai_vol_ensemble_rebalance_day = 0
                
                if last_ai_vol_ensemble_rebalance_day == 0 or (day_count - last_ai_vol_ensemble_rebalance_day) >= 7:  # Weekly rebalance
                    print(f"\n🔄 AI Volatility Ensemble: Evaluating portfolio (Day {day_count})...")
                    
                    # Calculate AI-enhanced volatility scores
                    ai_vol_scores = []
                    available_tickers = initial_top_tickers if initial_top_tickers else []
                    
                    for ticker in available_tickers:
                        try:
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_data = ticker_data_grouped[ticker]
                            
                            # Get data for last 60 days
                            end_date = current_date
                            start_date = current_date - timedelta(days=60)
                            data_slice = ticker_data.loc[start_date:end_date]
                            
                            if len(data_slice) >= 30:
                                # Calculate multiple volatility metrics
                                returns = data_slice['Close'].pct_change().dropna()
                                
                                if len(returns) > 5:
                                    # 1. Realized volatility (20-day)
                                    real_vol = returns.tail(20).std() * np.sqrt(252)
                                    
                                    # 2. Volatility trend (is volatility increasing or decreasing?)
                                    vol_short = returns.tail(10).std() * np.sqrt(252)
                                    vol_long = returns.head(20).std() * np.sqrt(252)
                                    vol_trend = (vol_short - vol_long) / vol_long if vol_long > 0 else 0
                                    
                                    # 3. Price momentum (30-day)
                                    price_momentum = (data_slice['Close'].iloc[-1] / data_slice['Close'].iloc[-30]) - 1
                                    
                                    # 4. Volume confirmation
                                    volume_ratio = data_slice['Volume'].tail(10).mean() / data_slice['Volume'].head(30).mean()
                                    
                                    # AI-enhanced score: combine multiple factors
                                    # Prefer: moderate volatility, decreasing vol trend, positive momentum, high volume
                                    if real_vol > 0:
                                        # Normalize factors
                                        vol_score = 1 / (1 + real_vol)  # Lower volatility = higher score
                                        trend_score = 1 - max(0, vol_trend)  # Decreasing volatility = higher score
                                        momentum_score = max(0, price_momentum)  # Positive momentum = higher score
                                        volume_score = min(2, volume_ratio)  # Higher volume = higher score (capped at 2x)
                                        
                                        # Weighted combination (AI-optimized weights)
                                        ai_score = (0.3 * vol_score + 
                                                   0.25 * trend_score + 
                                                   0.25 * momentum_score + 
                                                   0.2 * volume_score)
                                        
                                        # Only include stocks with positive momentum
                                        if price_momentum > 0:
                                            ai_vol_scores.append((ticker, ai_score, real_vol, price_momentum, vol_trend))
                        
                        except Exception:
                            continue
                    
                    if ai_vol_scores:
                        # Sort by AI score (descending)
                        ai_vol_scores.sort(key=lambda x: x[1], reverse=True)
                        top_ai_vol_stocks = [(t, s, v, m, tr) for t, s, v, m, tr in ai_vol_scores[:PORTFOLIO_SIZE]]
                        
                        print(f"   🤖 Top {PORTFOLIO_SIZE} AI Volatility stocks: {[(t, f'{s*100:.1f}', f'{v*100:.1f}%', f'{m*100:.1f}%') for t, s, v, m, tr in top_ai_vol_stocks]}")
                        
                        # Rebalance portfolio with volatility caps
                        total_value = ai_volatility_ensemble_cash + sum(pos.get('value', 0) for pos in ai_volatility_ensemble_positions.values())
                        capital_per_stock = total_value / PORTFOLIO_SIZE if PORTFOLIO_SIZE > 0 else 0
                        
                        # Apply volatility cap: max 15% allocation per position
                        max_position_value = total_value * 0.15
                        
                        # Sell existing positions
                        for ticker in list(ai_volatility_ensemble_positions.keys()):
                            if ticker not in [t for t, _, _, _, _ in top_ai_vol_stocks]:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        shares = ai_volatility_ensemble_positions[ticker]['shares']
                                        gross_sale = shares * current_price
                                        sell_cost = gross_sale * TRANSACTION_COST
                                        ai_volatility_ensemble_cash += gross_sale - sell_cost
                                        del ai_volatility_ensemble_positions[ticker]
                                except Exception:
                                    continue
                        
                        # Buy new positions with volatility caps
                        for ticker, ai_score, vol, momentum, vol_trend in top_ai_vol_stocks:
                            if ticker not in ai_volatility_ensemble_positions and capital_per_stock > 0:
                                try:
                                    ticker_data = ticker_data_grouped[ticker]
                                    price_data = ticker_data.loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            # Apply volatility cap
                                            position_value = min(capital_per_stock, max_position_value)
                                            shares = int(position_value / (current_price * (1 + TRANSACTION_COST)))
                                            if shares > 0:
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                ai_volatility_ensemble_cash -= (buy_value + buy_cost)
                                                
                                                ai_volatility_ensemble_positions[ticker] = {
                                                    'shares': shares,
                                                    'entry_price': current_price,
                                                    'value': buy_value,
                                                    'ai_score': ai_score,
                                                    'volatility': vol,
                                                    'momentum': momentum
                                                }
                                except Exception:
                                    continue
                        
                        last_ai_vol_ensemble_rebalance_day = day_count
                        print(f"   ✅ AI Volatility Ensemble: Rebalanced to {len(ai_volatility_ensemble_positions)} positions")
                        
            except Exception as e:
                print(f"   ⚠️ AI Volatility Ensemble error: {e}")

        # === MOMENTUM + AI HYBRID STRATEGY ===
        if ENABLE_MOMENTUM_AI_HYBRID:
            try:
                # Check if it's time to rebalance (weekly) or initial allocation
                if last_momentum_ai_hybrid_rebalance_day == 0 or (day_count - last_momentum_ai_hybrid_rebalance_day) >= AI_REBALANCE_FREQUENCY_DAYS:
                    print(f"\n🔄 Momentum+AI Hybrid: Evaluating portfolio (Day {day_count})...")
                    
                    # Calculate momentum for all available tickers
                    momentum_scores = []
                    # Use initial_top_tickers (the tickers we trained models for)
                    available_tickers = initial_top_tickers if initial_top_tickers else []
                    # ✅ OPTIMIZED: Use pre-grouped data
                    for ticker in available_tickers:
                        try:
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_history = ticker_data_grouped[ticker].reset_index()
                            ticker_history = ticker_history[ticker_history['date'] <= current_date].tail(MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK + 10)
                            
                            if len(ticker_history) >= MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK:
                                lookback_data = ticker_history.tail(MOMENTUM_AI_HYBRID_MOMENTUM_LOOKBACK)
                                start_price = lookback_data.iloc[0]['Close']
                                end_price = lookback_data.iloc[-1]['Close']
                                
                                if start_price > 0:
                                    momentum_return = (end_price - start_price) / start_price
                                    momentum_scores.append((ticker, momentum_return))
                        
                        except Exception:
                            continue
                    
                    if momentum_scores:
                        # Sort by momentum (descending)
                        momentum_scores.sort(key=lambda x: x[1], reverse=True)
                        top_momentum_stocks = [ticker for ticker, score in momentum_scores[:PORTFOLIO_SIZE]]
                        
                        print(f"   📈 Top {PORTFOLIO_SIZE} momentum stocks: {[(t, f'{s*100:.1f}%') for t, s in momentum_scores[:PORTFOLIO_SIZE]]}")
                        
                        # Rebalance using AI signals and universal smart rebalancing
                        momentum_ai_hybrid_positions, momentum_ai_hybrid_cash, current_momentum_ai_hybrid_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Momentum+AI Hybrid",
                            current_stocks=current_momentum_ai_hybrid_stocks,
                            new_stocks=top_momentum_stocks,
                            positions=momentum_ai_hybrid_positions,
                            cash=momentum_ai_hybrid_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=last_momentum_ai_hybrid_rebalance_day == 0  # Force initial allocation
                        )
                        momentum_ai_hybrid_transaction_costs += rebalance_costs
                        
                        last_momentum_ai_hybrid_rebalance_day = day_count
            
            except Exception as e:
                print(f"   ⚠️ Momentum+AI hybrid failed: {e}")
                import traceback
                traceback.print_exc()

        # === NEW ADVANCED STRATEGIES ===
        
        # MOMENTUM ACCELERATION STRATEGY
        if ENABLE_MOMENTUM_ACCELERATION:
            try:
                from new_strategies import select_momentum_acceleration_stocks
                
                print(f"   📈 Momentum Acceleration: Analyzing {len(initial_top_tickers)} tickers...")
                
                new_mom_accel_stocks = select_momentum_acceleration_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE
                )
                
                if new_mom_accel_stocks:
                    total_value = mom_accel_cash + sum(pos.get('value', 0) for pos in mom_accel_positions.values())
                    capital_per_stock = total_value / len(new_mom_accel_stocks)
                    
                    # Sell positions not in new selection
                    for ticker in list(mom_accel_positions.keys()):
                        if ticker not in new_mom_accel_stocks:
                            if ticker in ticker_data_grouped:
                                price_data = ticker_data_grouped[ticker].loc[:current_date]
                                if not price_data.empty:
                                    current_price = price_data['Close'].dropna().iloc[-1]
                                    shares = mom_accel_positions[ticker]['shares']
                                    gross_sale = shares * current_price
                                    sell_cost = gross_sale * TRANSACTION_COST
                                    mom_accel_transaction_costs += sell_cost
                                    mom_accel_cash += gross_sale - sell_cost
                            del mom_accel_positions[ticker]
                    
                    # Buy new positions
                    for ticker in new_mom_accel_stocks:
                        if ticker not in mom_accel_positions:
                            if ticker in ticker_data_grouped:
                                price_data = ticker_data_grouped[ticker].loc[:current_date]
                                if not price_data.empty:
                                    current_price = price_data['Close'].dropna().iloc[-1]
                                    if current_price > 0:
                                        shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                                        if shares > 0 and mom_accel_cash >= shares * current_price * (1 + TRANSACTION_COST):
                                            buy_value = shares * current_price
                                            buy_cost = buy_value * TRANSACTION_COST
                                            mom_accel_transaction_costs += buy_cost
                                            mom_accel_cash -= (buy_value + buy_cost)
                                            mom_accel_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}
                    
                    current_mom_accel_stocks = new_mom_accel_stocks
                    
            except Exception as e:
                print(f"   ⚠️ Momentum Acceleration error: {e}")

        # CONCENTRATED 3M STRATEGY
        if ENABLE_CONCENTRATED_3M:
            try:
                concentrated_3m_days_since_rebalance += 1
                
                # Only rebalance monthly
                if concentrated_3m_days_since_rebalance >= CONCENTRATED_3M_REBALANCE_DAYS or not current_concentrated_3m_stocks:
                    from new_strategies import select_concentrated_3m_stocks
                    
                    print(f"   🎯 Concentrated 3M: Analyzing {len(initial_top_tickers)} tickers...")
                    
                    new_concentrated_3m_stocks = select_concentrated_3m_stocks(
                        initial_top_tickers, 
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=3
                    )
                    
                    if new_concentrated_3m_stocks:
                        # Use universal smart rebalancing function
                        concentrated_3m_positions, concentrated_3m_cash, current_concentrated_3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="Concentrated 3M",
                            current_stocks=current_concentrated_3m_stocks,
                            new_stocks=new_concentrated_3m_stocks,
                            positions=concentrated_3m_positions,
                            cash=concentrated_3m_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=3,
                            force_rebalance=not current_concentrated_3m_stocks  # Force initial allocation
                        )
                        concentrated_3m_transaction_costs += rebalance_costs
                        concentrated_3m_days_since_rebalance = 0
                        
            except Exception as e:
                print(f"   ⚠️ Concentrated 3M error: {e}")

        # DUAL MOMENTUM STRATEGY
        if ENABLE_DUAL_MOMENTUM:
            try:
                from new_strategies import select_dual_momentum_stocks
                
                print(f"   📊 Dual Momentum: Analyzing {len(initial_top_tickers)} tickers...")
                
                new_dual_mom_stocks, is_risk_on = select_dual_momentum_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE
                )
                
                # If risk-off, sell all positions
                if not is_risk_on:
                    for ticker in list(dual_mom_positions.keys()):
                        if ticker in ticker_data_grouped:
                            price_data = ticker_data_grouped[ticker].loc[:current_date]
                            if not price_data.empty:
                                current_price = price_data['Close'].dropna().iloc[-1]
                                shares = dual_mom_positions[ticker]['shares']
                                gross_sale = shares * current_price
                                sell_cost = gross_sale * TRANSACTION_COST
                                dual_mom_transaction_costs += sell_cost
                                dual_mom_cash += gross_sale - sell_cost
                        del dual_mom_positions[ticker]
                    current_dual_mom_stocks = []
                    dual_mom_is_risk_on = False
                elif new_dual_mom_stocks:
                    dual_mom_is_risk_on = True
                    # Use universal smart rebalancing function
                    dual_mom_positions, dual_mom_cash, current_dual_mom_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Dual Momentum",
                        current_stocks=current_dual_mom_stocks,
                        new_stocks=new_dual_mom_stocks,
                        positions=dual_mom_positions,
                        cash=dual_mom_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_dual_mom_stocks  # Force initial allocation
                    )
                    dual_mom_transaction_costs += rebalance_costs
                    
            except Exception as e:
                print(f"   ⚠️ Dual Momentum error: {e}")

        # TREND FOLLOWING ATR STRATEGY
        if ENABLE_TREND_FOLLOWING_ATR:
            try:
                from new_strategies import select_trend_following_atr_stocks, reset_trend_atr_state
                
                # Reset on first day
                if day_count == 1:
                    reset_trend_atr_state()
                
                print(f"   📈 Trend Following ATR: Analyzing {len(initial_top_tickers)} tickers...")
                
                stocks_to_buy, stocks_to_sell = select_trend_following_atr_stocks(
                    initial_top_tickers, ticker_data_grouped, current_date, top_n=PORTFOLIO_SIZE
                )
                
                # Process sells first
                for ticker in stocks_to_sell:
                    if ticker in trend_atr_positions:
                        if ticker in ticker_data_grouped:
                            price_data = ticker_data_grouped[ticker].loc[:current_date]
                            if not price_data.empty:
                                current_price = price_data['Close'].dropna().iloc[-1]
                                shares = trend_atr_positions[ticker]['shares']
                                gross_sale = shares * current_price
                                sell_cost = gross_sale * TRANSACTION_COST
                                trend_atr_transaction_costs += sell_cost
                                trend_atr_cash += gross_sale - sell_cost
                        del trend_atr_positions[ticker]
                
                # Process buys
                if stocks_to_buy:
                    total_value = trend_atr_cash + sum(pos.get('value', 0) for pos in trend_atr_positions.values())
                    available_slots = PORTFOLIO_SIZE - len(trend_atr_positions)
                    if available_slots > 0:
                        capital_per_stock = total_value / PORTFOLIO_SIZE
                        
                        for ticker in stocks_to_buy[:available_slots]:
                            if ticker not in trend_atr_positions:
                                if ticker in ticker_data_grouped:
                                    price_data = ticker_data_grouped[ticker].loc[:current_date]
                                    if not price_data.empty:
                                        current_price = price_data['Close'].dropna().iloc[-1]
                                        if current_price > 0:
                                            shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                                            if shares > 0 and trend_atr_cash >= shares * current_price * (1 + TRANSACTION_COST):
                                                buy_value = shares * current_price
                                                buy_cost = buy_value * TRANSACTION_COST
                                                trend_atr_transaction_costs += buy_cost
                                                trend_atr_cash -= (buy_value + buy_cost)
                                                trend_atr_positions[ticker] = {'shares': shares, 'entry_price': current_price, 'value': buy_value}
                
                current_trend_atr_stocks = list(trend_atr_positions.keys())
                    
            except Exception as e:
                print(f"   ⚠️ Trend Following ATR error: {e}")

        # ELITE HYBRID STRATEGY (Mom-Vol 6M + 1Y/3M Ratio)
        if ENABLE_ELITE_HYBRID:
            try:
                from elite_hybrid_strategy import select_elite_hybrid_stocks
                
                print(f"   🏆 Elite Hybrid: Analyzing {len(initial_top_tickers)} tickers...")
                
                new_elite_hybrid_stocks = select_elite_hybrid_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )
                
                if new_elite_hybrid_stocks:
                    elite_hybrid_positions, elite_hybrid_cash, current_elite_hybrid_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Elite Hybrid",
                        current_stocks=current_elite_hybrid_stocks,
                        new_stocks=new_elite_hybrid_stocks,
                        positions=elite_hybrid_positions,
                        cash=elite_hybrid_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_elite_hybrid_stocks
                    )
                    elite_hybrid_transaction_costs += rebalance_costs
                    elite_hybrid_last_rebalance_value = elite_hybrid_portfolio_value
                    
            except Exception as e:
                print(f"   ⚠️ Elite Hybrid error: {e}")

        # ELITE RISK STRATEGY (Risk-Adj Mom base + Elite Hybrid dip/vol bonuses)
        if ENABLE_ELITE_RISK:
            try:
                from elite_risk_strategy import select_elite_risk_stocks

                new_elite_risk_stocks = select_elite_risk_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )

                if new_elite_risk_stocks:
                    elite_risk_positions, elite_risk_cash, current_elite_risk_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Elite Risk",
                        current_stocks=current_elite_risk_stocks,
                        new_stocks=new_elite_risk_stocks,
                        positions=elite_risk_positions,
                        cash=elite_risk_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_elite_risk_stocks
                    )
                    elite_risk_transaction_costs += rebalance_costs
                    elite_risk_last_rebalance_value = elite_risk_portfolio_value

            except Exception as e:
                print(f"   ⚠️ Elite Risk error: {e}")

        # RISK-ADJ MOM 6M STRATEGY
        if ENABLE_RISK_ADJ_MOM_6M:
            try:
                from risk_adj_mom_6m_strategy import select_risk_adj_mom_6m_stocks

                new_risk_adj_mom_6m_stocks = select_risk_adj_mom_6m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )

                if new_risk_adj_mom_6m_stocks:
                    risk_adj_mom_6m_positions, risk_adj_mom_6m_cash, current_risk_adj_mom_6m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 6M",
                        current_stocks=current_risk_adj_mom_6m_stocks,
                        new_stocks=new_risk_adj_mom_6m_stocks,
                        positions=risk_adj_mom_6m_positions,
                        cash=risk_adj_mom_6m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_6m_stocks
                    )
                    risk_adj_mom_6m_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 6M error: {e}")

        # RISK-ADJ MOM 3M STRATEGY
        if ENABLE_RISK_ADJ_MOM_3M:
            try:
                from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks

                new_risk_adj_mom_3m_stocks = select_risk_adj_mom_3m_stocks(
                    initial_top_tickers,
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE
                )

                if new_risk_adj_mom_3m_stocks:
                    risk_adj_mom_3m_positions, risk_adj_mom_3m_cash, current_risk_adj_mom_3m_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="Risk-Adj Mom 3M",
                        current_stocks=current_risk_adj_mom_3m_stocks,
                        new_stocks=new_risk_adj_mom_3m_stocks,
                        positions=risk_adj_mom_3m_positions,
                        cash=risk_adj_mom_3m_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_risk_adj_mom_3m_stocks
                    )
                    risk_adj_mom_3m_transaction_costs += rebalance_costs

            except Exception as e:
                print(f"   ⚠️ Risk-Adj Mom 3M error: {e}")

        # RISK-ADJ MOM 3M MONTHLY STRATEGY (same scoring, rebalance start of month only)
        if ENABLE_RISK_ADJ_MOM_3M_MONTHLY:
            should_rebalance_3m_mom_monthly = (not risk_adj_mom_3m_monthly_initialized) or is_first_trading_day_of_month
            if should_rebalance_3m_mom_monthly:
                try:
                    from risk_adj_mom_3m_strategy import select_risk_adj_mom_3m_stocks

                    new_stocks = select_risk_adj_mom_3m_stocks(
                        initial_top_tickers,
                        ticker_data_grouped,
                        current_date=current_date,
                        top_n=PORTFOLIO_SIZE
                    )

                    if new_stocks:
                        if not risk_adj_mom_3m_monthly_initialized:
                            print(f"   🎯 Risk-Adj Mom 3M Monthly: Initializing with {new_stocks}")
                        else:
                            print(f"   🔄 Risk-Adj Mom 3M Monthly: Start-of-month rebalance ({current_date.strftime('%b %Y')})")
                        risk_adj_mom_3m_monthly_positions, risk_adj_mom_3m_monthly_cash, current_risk_adj_mom_3m_monthly_stocks, rc = _smart_rebalance_portfolio(
                            strategy_name="RiskAdj 3M Mth",
                            current_stocks=current_risk_adj_mom_3m_monthly_stocks,
                            new_stocks=new_stocks,
                            positions=risk_adj_mom_3m_monthly_positions,
                            cash=risk_adj_mom_3m_monthly_cash,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not risk_adj_mom_3m_monthly_initialized)
                        risk_adj_mom_3m_monthly_transaction_costs += rc
                        risk_adj_mom_3m_monthly_initialized = True
                        risk_adj_mom_3m_monthly_last_month = current_date.month

                except Exception as e:
                    print(f"   ⚠️ Risk-Adj Mom 3M Monthly error: {e}")

        # AI ELITE STRATEGY (ML-powered scoring) - PER-TICKER MODELS
        if ENABLE_AI_ELITE:
            try:
                from ai_elite_strategy import select_ai_elite_stocks
                from ai_elite_strategy_per_ticker import train_shared_base_model
                import os
                import pickle
                
                models_dir = "logs/models"
                os.makedirs(models_dir, exist_ok=True)
                
                # Try loading model from disk if not in memory
                base_model_path = os.path.join(models_dir, "_shared_base_ai_elite.joblib")
                if ai_elite_models.get('_shared_base') is None and os.path.exists(base_model_path):
                    try:
                        with open(base_model_path, 'rb') as f:
                            loaded_model = pickle.load(f)
                        ai_elite_models['_shared_base'] = loaded_model
                        for ticker in initial_top_tickers:
                            ai_elite_models[ticker] = loaded_model
                            ai_elite_last_train_days[ticker] = 0  # Mark as loaded from disk (day 0)
                        print(f"   ✅ AI Elite: Loaded model from disk for {len(initial_top_tickers)} tickers")
                    except Exception as e:
                        print(f"   ⚠️ AI Elite: Failed to load model from disk: {e}")
                
                # Check if we should train (day 1 ALWAYS trains, otherwise check retrain interval)
                should_train_ai_elite = False
                if day_count == 1:
                    should_train_ai_elite = True  # Always train on day 1
                    print(f"   📊 AI Elite: Day 1 - forced training (warm-start from {'disk model' if ai_elite_models.get('_shared_base') is not None else 'scratch'})")
                elif ai_elite_models.get('_shared_base') is None:
                    should_train_ai_elite = True  # No model exists (in memory or disk), need to train
                else:
                    # Check retraining based on AI_ELITE_RETRAIN_DAYS
                    last_train_day = max(ai_elite_last_train_days.values()) if ai_elite_last_train_days else 0
                    days_since_train = day_count - last_train_day
                    if days_since_train >= AI_ELITE_RETRAIN_DAYS:
                        should_train_ai_elite = True
                        print(f"   📊 AI Elite: Retraining triggered (day {day_count}, last train day {last_train_day}, interval {AI_ELITE_RETRAIN_DAYS})")
                
                if should_train_ai_elite:
                    # HYBRID: shared base model + per-ticker fine-tuning
                    train_end = current_date
                    train_start = train_end - timedelta(days=AI_ELITE_TRAINING_LOOKBACK)
                    
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    n_workers = min(TRAINING_NUM_PROCESSES, len(initial_top_tickers))
                    
                    # Pre-compute market returns for all sample dates (needed for excess return labels)
                    from ai_elite_strategy import _calculate_market_return
                    from datetime import timezone as tz_utc
                    market_returns = {}
                    sample_date_iter = train_start
                    while sample_date_iter <= train_end:
                        mr = _calculate_market_return(ticker_data_grouped, sample_date_iter, AI_ELITE_FORWARD_DAYS)
                        # Store with UTC key to match collect_ticker_training_data's UTC dates
                        utc_key = sample_date_iter.replace(tzinfo=tz_utc.utc) if sample_date_iter.tzinfo is None else sample_date_iter
                        market_returns[utc_key] = mr if mr is not None else 0.0
                        sample_date_iter += timedelta(days=2)
                    
                    # Step 1: Collect training data from ALL tickers in parallel
                    print(f"   📊 AI Elite: Collecting data from {len(initial_top_tickers)} tickers ({n_workers} processes, {AI_ELITE_TRAINING_LOOKBACK}d lookback)...")
                    collect_args = [
                        (t, ticker_data_grouped.get(t), train_start, train_end, AI_ELITE_FORWARD_DAYS, market_returns)
                        for t in initial_top_tickers
                    ]
                    
                    all_training_data = []
                    ticker_samples_map = {}  # ticker -> its samples
                    
                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        futures = {executor.submit(_collect_data_worker, a): a[0] for a in collect_args}
                        for future in as_completed(futures):
                            try:
                                ticker, samples = future.result()
                                if samples:
                                    all_training_data.extend(samples)
                                    ticker_samples_map[ticker] = samples
                            except Exception as e:
                                print(f"   ⚠️ AI Elite: Data collection failed for {futures[future]}: {e}")
                    
                    print(f"   📊 AI Elite: Collected {len(all_training_data)} samples from {len(ticker_samples_map)} tickers")
                    
                    # Step 2: Train shared base model on ALL data (single process, large dataset)
                    base_model_path = os.path.join(models_dir, "_shared_base_ai_elite.joblib")
                    existing_base = ai_elite_models.get('_shared_base')
                    base_model, base_kappa = train_shared_base_model(
                        all_training_data, save_path=base_model_path,
                        existing_model=existing_base
                    )
                    
                    if base_model:
                        ai_elite_models['_shared_base'] = base_model
                        
                        # Use shared base model for ALL tickers (skip per-ticker fine-tuning to avoid overfitting)
                        print(f"   ✅ AI Elite: Using shared base model for all tickers (R² {base_kappa:.3f})")
                        for ticker in initial_top_tickers:
                            ai_elite_models[ticker] = base_model
                            ai_elite_last_train_days[ticker] = day_count
                        trained = len(initial_top_tickers)
                    else:
                        print(f"   ⚠️ AI Elite: Training failed, no model produced")
                else:
                    # Using existing model (loaded from disk or trained earlier), no retraining needed
                    last_train_day = max(ai_elite_last_train_days.values()) if ai_elite_last_train_days else 0
                    if last_train_day == 0:
                        print(f"   📊 AI Elite: Using model loaded from disk (fresh load)")
                    else:
                        days_ago = day_count - last_train_day
                        print(f"   📊 AI Elite: Using existing model (trained {days_ago} days ago, day {last_train_day})")
                
                # Select stocks using ML model (always runs - with fresh or existing model)
                print(f"   🤖 AI Elite: Analyzing {len(initial_top_tickers)} tickers...")
                
                new_ai_elite_stocks = select_ai_elite_stocks(
                    initial_top_tickers, 
                    ticker_data_grouped,
                    current_date=current_date,
                    top_n=PORTFOLIO_SIZE,
                    per_ticker_models=ai_elite_models
                )
                
                if new_ai_elite_stocks:
                    ai_elite_positions, ai_elite_cash, current_ai_elite_stocks, rebalance_costs = _smart_rebalance_portfolio(
                        strategy_name="AI Elite",
                        current_stocks=current_ai_elite_stocks,
                        new_stocks=new_ai_elite_stocks,
                        positions=ai_elite_positions,
                        cash=ai_elite_cash,
                        ticker_data_grouped=ticker_data_grouped,
                        current_date=current_date,
                        transaction_cost=TRANSACTION_COST,
                        portfolio_size=PORTFOLIO_SIZE,
                        force_rebalance=not current_ai_elite_stocks
                    )
                    ai_elite_transaction_costs += rebalance_costs
                    ai_elite_last_rebalance_value = ai_elite_portfolio_value
                    
            except Exception as e:
                print(f"   ⚠️ AI Elite error: {e}")

        # Daily stock selection: Use current models to pick best 3 from 40 stocks
        try:
            predictions = []
            selected_stocks = []
            day_predictions = {'date': current_date, 'day': day_count, 'predictions': []}

            # Skip AI predictions if disabled - just keep empty portfolio
            if not enable_ai_strategy:
                print(f"   🤖 AI predictions disabled (ENABLE_AI_STRATEGY={enable_ai_strategy})")
                valid_predictions = 0
                # Explicitly reset consecutive counter when AI is disabled
                consecutive_no_predictions = 0
            else:
                # Get predictions for all 40 stocks using current models
                valid_predictions = 0
                # ✅ OPTIMIZED: Use pre-grouped data
                debug_count = 0
                for ticker in initial_top_tickers:
                    is_debug = debug_count < 3  # Debug first 3 tickers
                    debug_count += 1
                    
                    if is_debug:
                        print(f"   🔍 Checking {ticker}: in models={ticker in current_models}, model not None={current_models.get(ticker) is not None if ticker in current_models else False}")
                    if ticker in current_models and current_models[ticker] is not None:
                        try:
                            # Get data up to previous day for prediction (avoid look-ahead bias)
                            if ticker not in ticker_data_grouped:
                                continue
                            ticker_data = ticker_data_grouped[ticker]
                            
                            prediction_date = current_date - timedelta(days=1)
                            
                            # ✅ FIX: Ensure prediction_date timezone matches the DataFrame index
                            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                                if prediction_date.tzinfo is None:
                                    prediction_date = prediction_date.replace(tzinfo=ticker_data.index.tz)
                                else:
                                    prediction_date = prediction_date.astimezone(ticker_data.index.tz)
                            elif prediction_date.tzinfo is not None:
                                # Index is naive, make prediction_date naive too
                                prediction_date = prediction_date.replace(tzinfo=None)
                            
                            data_slice = ticker_data.loc[:prediction_date]
                            
                            # DEBUG: Show data slice info for first 3 tickers only
                            if is_debug:
                                print(f"   🔍 {ticker}: data_slice length={len(data_slice)}, required={PREDICTION_LOOKBACK_DAYS}, prediction_date={prediction_date.date()}")

                            if len(data_slice) >= PREDICTION_LOOKBACK_DAYS:  # Need minimum lookback days for features
                                print(f"   📊 {ticker}: Calling prediction with {len(data_slice.tail(PREDICTION_LOOKBACK_DAYS))} rows, model={type(current_models[ticker]).__name__ if current_models[ticker] else None}, scaler={type(current_scalers.get(ticker)).__name__ if current_scalers.get(ticker) else None}")
                                pred = _quick_predict_return(
                                    ticker, data_slice.tail(PREDICTION_LOOKBACK_DAYS),  # Use last N days for features
                                    current_models[ticker],  # Single model
                                    current_scalers.get(ticker),
                                    current_y_scalers.get(ticker),
                                    horizon_days
                                )
                                print(f"   📊 {ticker}: Prediction result = {pred}")
                                # ✅ FIX 4: Only add valid predictions
                                if pred != -np.inf:
                                    predictions.append((ticker, pred))
                                    valid_predictions += 1
                            else:
                                # Debug: Show actual data slice length for first few tickers
                                if ticker in ['WDC', 'STX', 'WBD'] and day_count <= 3:
                                    print(f"   ⚠️ {ticker}: Only {len(data_slice)} rows available, need >={PREDICTION_LOOKBACK_DAYS} for feature engineering")
                                    print(f"      📅 Data slice range: {data_slice.index.min()} to {data_slice.index.max()}")
                                    print(f"      📅 Current date: {current_date}, Prediction date: {prediction_date}")
                                # ✅ FIX 4: Don't reference undefined 'pred' variable

                        except Exception:
                            continue

            # Debug: Show prediction summary
            if day_count == 1 or day_count % 10 == 0:
                print(f"   🔮 Day {day_count}: {valid_predictions} valid predictions from {len(initial_top_tickers)} tickers")

            # ✅ FIX: Check if no predictions are being made (only when AI is enabled)
            # DEBUG: Print current state
            if day_count <= 3 or day_count % 10 == 0:  # Debug first few days and every 10th day
                print(f"   🔍 DEBUG: enable_ai_strategy={enable_ai_strategy}, valid_predictions={valid_predictions}, consecutive_no_predictions={consecutive_no_predictions}")

            # Explicit check: only count as failure if AI is enabled AND no predictions
            should_count_as_failure = enable_ai_strategy and (valid_predictions == 0)

            if should_count_as_failure:
                consecutive_no_predictions += 1
                if consecutive_no_predictions >= MAX_CONSECUTIVE_FAILURES:
                    # ✅ FIX: Just log error, don't abort - let other strategies continue
                    print(f"\n⚠️ AI Strategy: No valid predictions for {consecutive_no_predictions} consecutive days")
                    print(f"   💡 AI Strategy will remain in cash, other strategies continue normally")
                    # Reset counter to avoid spamming the same message
                    consecutive_no_predictions = 0
            elif enable_ai_strategy:
                consecutive_no_predictions = 0  # Reset on success (only when AI is enabled)
            
            # ✅ NEW: Store predictions with metadata
            day_predictions = {
                'date': current_date,
                'day': day_count,
                'predictions': [(t, p) for t, p in predictions]  # Store all predictions made
            }

            # Initialize selected_stocks variable
            selected_stocks = []

            # Select top N by predicted return
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Apply momentum filter: only consider stocks with positive 3M or 6M momentum
                filtered_predictions = []
                for ticker, pred in predictions:
                    try:
                        if ticker in ticker_data_grouped:
                            ticker_data = ticker_data_grouped[ticker]
                            # Get momentum up to prediction date
                            prediction_date = current_date - timedelta(days=1)
                            if hasattr(ticker_data.index, 'tz') and ticker_data.index.tz is not None:
                                if prediction_date.tzinfo is None:
                                    prediction_date = prediction_date.replace(tzinfo=ticker_data.index.tz)
                            data_slice = ticker_data.loc[:prediction_date]
                            if len(data_slice) >= 63:  # Need at least 63 days for 3M momentum
                                close_3m_ago = data_slice['Close'].iloc[-63]
                                close_now = data_slice['Close'].iloc[-1]
                                momentum_3m = (close_now / close_3m_ago - 1) * 100
                                # Only include if 3M momentum is positive
                                if momentum_3m > 0:
                                    filtered_predictions.append((ticker, pred, momentum_3m))
                    except Exception:
                        # If momentum calculation fails, include the stock anyway
                        filtered_predictions.append((ticker, pred, 0))
                
                # Use filtered predictions if any remain, otherwise use original
                if filtered_predictions:
                    # Sort by prediction (already sorted) and select top N
                    predictions = [(t, p) for t, p, _ in filtered_predictions]
                    if day_count <= 3 or day_count % 10 == 0:
                        print(f"   📊 AI Strategy: Filtered from {len(predictions)} to {len(filtered_predictions)} stocks with positive 3M momentum")
                
                # Select top N stocks (or all if fewer available)
                num_to_select = min(PORTFOLIO_SIZE, len(predictions))
                selected_stocks = [ticker for ticker, _ in predictions[:num_to_select]]

                # Check if portfolio changed (only then incur transaction costs)
                skip_rebalance = False  # Initialize before try-except
                if set(selected_stocks) != set(current_portfolio_stocks):
                    # ✅ Transaction Cost Guard: Same logic as other strategies (Dynamic BH, Risk-Adj Mom, etc.)
                    # Only rebalance if portfolio has grown enough since last rebalance to cover transaction costs
                    if current_portfolio_stocks:
                        try:
                            # Create ticker data group for transaction cost guard function
                            # ✅ FIX: Use different variable name to avoid overwriting main ticker_data_grouped
                            rebal_ticker_data = {}
                            for ticker in set(current_portfolio_stocks) | set(selected_stocks):
                                ticker_data = ticker_data_grouped.get(ticker)
                                if ticker_data is not None and not ticker_data.empty:
                                    rebal_ticker_data[ticker] = ticker_data
                            
                            # Apply transaction cost guard (same as Dynamic BH and other strategies)
                            should_rebal, reason = _should_rebalance_by_profit_since_last_rebalance(
                                current_portfolio_stocks,
                                selected_stocks,
                                positions,
                                cash_balance,
                                TRANSACTION_COST,
                                ai_strategy_last_rebalance_value,
                                rebal_ticker_data,
                                current_date
                            )
                            
                            if not should_rebal:
                                if day_count % 5 == 0 or day_count <= 3:
                                    print(f"   💤 AI Strategy: Skipping rebalance ({reason})")
                                # Do not rebalance; keep current portfolio
                                selected_stocks = current_portfolio_stocks
                                skip_rebalance = True
                            else:
                                skip_rebalance = False
                                # Update last rebalance value after successful rebalance
                                current_portfolio_value = float(cash_balance)
                                for t, pos in positions.items():
                                    try:
                                        if pos and pos.get('shares', 0) > 0:
                                            td = ticker_data_grouped.get(t)
                                            if td and not td.empty:
                                                current_price_data = td.loc[:current_date]
                                                if not current_price_data.empty:
                                                    current_price = current_price_data['Close'].dropna().iloc[-1]
                                                    position_value = positions[ticker]['shares'] * current_price
                                                    current_portfolio_value += position_value

                                        # Update stored position value
                                        positions[ticker]['value'] = position_value
                                    except Exception:
                                        pass
                                ai_strategy_last_rebalance_value = current_portfolio_value
                            
                        except Exception as e:
                            print(f"   ⚠️ AI Strategy transaction cost guard failed: {e}")
                            # Fall back to allowing rebalance
                            skip_rebalance = False

                    if not skip_rebalance:
                        rebalance_count += 1
                    print(f"📊 Day {day_count} ({current_date.strftime('%Y-%m-%d')}): New portfolio: {selected_stocks}")
                    old_portfolio = current_portfolio_stocks.copy()

                    # Use universal smart rebalancing function
                    try:
                        positions, cash_balance, current_portfolio_stocks, rebalance_costs = _smart_rebalance_portfolio(
                            strategy_name="AI Strategy",
                            current_stocks=old_portfolio,
                            new_stocks=selected_stocks,
                            positions=positions,
                            cash=cash_balance,
                            ticker_data_grouped=ticker_data_grouped,
                            current_date=current_date,
                            transaction_cost=TRANSACTION_COST,
                            portfolio_size=PORTFOLIO_SIZE,
                            force_rebalance=not old_portfolio  # Force initial allocation
                        )
                        ai_transaction_costs += rebalance_costs

                        if old_portfolio:
                            print(f"   🔄 Rebalanced from {old_portfolio} to {selected_stocks}")
                        else:
                            print(f"   🆕 Initial portfolio: {selected_stocks}")

                        # Debug: Check if positions were updated
                        total_shares = sum(p.get('shares', 0) for p in positions.values())
                        total_value = sum(p.get('value', 0) for p in positions.values())
                        print(f"      📊 After rebalance: {len(positions)} positions, {total_shares:.0f} total shares, ${total_value:,.0f} total value, ${cash_balance:,.0f} cash")

                    except Exception as e:
                        print(f"   ⚠️ Rebalancing failed: {e}. Keeping current portfolio.")
                        import traceback
                        traceback.print_exc()
                        current_portfolio_stocks = old_portfolio
            else:
                # No stocks selected - this might happen on early days
                if day_count == 1:
                    print(f"   ⚠️ Day {day_count}: No valid predictions - portfolio remains unallocated")
                elif day_count % 10 == 0:
                    print(f"   📊 Day {day_count}: No portfolio changes needed")

                    # Calculate actual portfolio value based on current positions
                    if selected_stocks:
                        # Calculate current portfolio value from positions
                        invested_value = 0.0
                        individual_returns = []

                        for ticker in selected_stocks:
                            if ticker in positions and positions[ticker]['shares'] > 0:
                                # Get current price
                                try:
                                    ticker_data = ticker_data_grouped.get(ticker)
                                    if ticker_data is not None and not ticker_data.empty:
                                        # ticker_data already has date as index
                                        current_price_data = ticker_data.loc[:current_date]
                                        if not current_price_data.empty:
                                            current_price = current_price_data['Close'].dropna().iloc[-1]
                                            position_value = positions[ticker]['shares'] * current_price
                                            invested_value += position_value

                                            # Calculate individual stock performance
                                            entry_price = positions[ticker]['avg_price']
                                            if entry_price > 0:
                                                stock_return_pct = (current_price / entry_price - 1) * 100
                                                individual_returns.append({
                                                    'ticker': ticker,
                                                    'total_return_pct': stock_return_pct,
                                                    'current_price': current_price,
                                                    'entry_price': entry_price,
                                                    'shares': positions[ticker]['shares'],
                                                    'value': position_value
                                                })
                                except Exception as e:
                                    print(f"   ⚠️ Could not calculate value for {ticker}: {e}")

                        # Portfolio value = invested value + cash balance
                        total_portfolio_value = invested_value + cash_balance

                        # Print portfolio status
                        print(f"   💼 Portfolio Status: ${total_portfolio_value:,.0f} total (${invested_value:,.0f} invested + ${cash_balance:,.0f} cash)")
                        print(f"   📊 Positions: {len([p for p in positions.values() if p['shares'] > 0])} stocks held")

                        # Print individual stock performance
                        if individual_returns:
                            print(f"   📋 Individual Stock Performance:")
                            for stock in individual_returns:
                                print(f"      • {stock['ticker']}: {stock['total_return_pct']:+.1f}% (${stock['current_price']:.2f} vs ${stock['entry_price']:.2f})")

                    # In real implementation: execute trades here
                    # For simulation: allocate capital to new stocks

                # Even if no rebalancing, portfolio value is updated at the end of the day loop

        except Exception as e:
            print(f"   ⚠️ Day {day_count}: Stock selection failed: {e}")
            # Keep existing portfolio if selection fails

        # Update DYNAMIC BH 1Y portfolio value daily (skip if disabled)
        dynamic_bh_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_1Y:
          # Iterate over actual positions, not just current_dynamic_bh_stocks list
          print(f"   🔧 DEBUG: Dyn BH 1Y - iterating over {len(dynamic_bh_positions)} positions")
          for ticker in list(dynamic_bh_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_positions[ticker]['shares'] * current_price
                                    dynamic_bh_positions[ticker]['value'] = position_value
                                    dynamic_bh_invested_value += position_value
                                else:
                                    dynamic_bh_invested_value += dynamic_bh_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_invested_value += dynamic_bh_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH position for {ticker}: {e}")

        dynamic_bh_portfolio_value = dynamic_bh_invested_value + dynamic_bh_cash
        dynamic_bh_portfolio_history.append(dynamic_bh_portfolio_value)

        # Update DYNAMIC BH 1Y + VOLATILITY FILTER portfolio value daily (skip if disabled)
        dynamic_bh_1y_vol_filter_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_1Y_VOL_FILTER:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1y_vol_filter_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1y_vol_filter_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1y_vol_filter_positions[ticker]['value'] = position_value
                                    dynamic_bh_1y_vol_filter_invested_value += position_value
                                else:
                                    dynamic_bh_1y_vol_filter_invested_value += dynamic_bh_1y_vol_filter_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1y_vol_filter_invested_value += dynamic_bh_1y_vol_filter_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1Y+Vol position for {ticker}: {e}")

        dynamic_bh_1y_vol_filter_portfolio_value = dynamic_bh_1y_vol_filter_invested_value + dynamic_bh_1y_vol_filter_cash
        dynamic_bh_1y_vol_filter_portfolio_history.append(dynamic_bh_1y_vol_filter_portfolio_value)

        # Update DYNAMIC BH 1Y + TRAILING STOP portfolio value daily (skip if disabled)
        dynamic_bh_1y_trailing_stop_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1y_trailing_stop_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1y_trailing_stop_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1y_trailing_stop_positions[ticker]['value'] = position_value
                                    dynamic_bh_1y_trailing_stop_invested_value += position_value
                                else:
                                    dynamic_bh_1y_trailing_stop_invested_value += dynamic_bh_1y_trailing_stop_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1y_trailing_stop_invested_value += dynamic_bh_1y_trailing_stop_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1Y+TS position for {ticker}: {e}")

        dynamic_bh_1y_trailing_stop_portfolio_value = dynamic_bh_1y_trailing_stop_invested_value + dynamic_bh_1y_trailing_stop_cash
        dynamic_bh_1y_trailing_stop_portfolio_history.append(dynamic_bh_1y_trailing_stop_portfolio_value)

        # Update SECTOR ROTATION portfolio value daily (skip if disabled)
        if ENABLE_SECTOR_ROTATION:
            sector_rotation_invested_value = 0.0
            for etf in current_sector_rotation_etfs:
                if etf in sector_rotation_positions:
                    try:
                        etf_data = ticker_data_grouped.get(etf)
                        if etf_data and not etf_data.empty:
                            current_price_data = etf_data.loc[etf_data.index == current_date]
                            if len(current_price_data) > 0:
                                valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = sector_rotation_positions[etf]['shares'] * current_price
                                    sector_rotation_positions[etf]['value'] = position_value
                                    sector_rotation_invested_value += position_value
                                else:
                                    sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                            else:
                                sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                        else:
                            sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)
                    except Exception as e:
                        sector_rotation_invested_value += sector_rotation_positions[etf].get('value', 0.0)

            sector_rotation_portfolio_value = sector_rotation_invested_value + sector_rotation_cash
            sector_rotation_portfolio_history.append(sector_rotation_portfolio_value)

        # Update DYNAMIC BH 6-MONTH portfolio value daily (skip if disabled)
        dynamic_bh_6m_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_6M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_6m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_6m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_6m_positions[ticker]['value'] = position_value
                                    dynamic_bh_6m_invested_value += position_value
                                else:
                                    dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)
                except Exception:
                    dynamic_bh_6m_invested_value += dynamic_bh_6m_positions[ticker].get('value', 0.0)

        # Update DYNAMIC BH 3-MONTH portfolio value daily (skip if disabled)
        dynamic_bh_3m_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_3M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_3m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_3m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_3m_positions[ticker]['value'] = position_value
                                    dynamic_bh_3m_invested_value += position_value
                                else:
                                    dynamic_bh_3m_invested_value += dynamic_bh_3m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_3m_invested_value += dynamic_bh_3m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 3M position for {ticker}: {e}")

        dynamic_bh_6m_portfolio_value = dynamic_bh_6m_invested_value + dynamic_bh_6m_cash
        dynamic_bh_6m_portfolio_history.append(dynamic_bh_6m_portfolio_value)

        dynamic_bh_3m_portfolio_value = dynamic_bh_3m_invested_value + dynamic_bh_3m_cash
        dynamic_bh_3m_portfolio_history.append(dynamic_bh_3m_portfolio_value)

        # Update dynamic BH 1-month portfolio value (skip if disabled)
        dynamic_bh_1m_invested_value = 0.0
        if ENABLE_DYNAMIC_BH_1M:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(dynamic_bh_1m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = dynamic_bh_1m_positions[ticker]['shares'] * current_price
                                    dynamic_bh_1m_positions[ticker]['value'] = position_value
                                    dynamic_bh_1m_invested_value += position_value
                                else:
                                    dynamic_bh_1m_invested_value += dynamic_bh_1m_positions[ticker].get('value', 0.0)
                            else:
                                dynamic_bh_1m_invested_value += dynamic_bh_1m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic BH 1M position for {ticker}: {e}")

        dynamic_bh_1m_portfolio_value = dynamic_bh_1m_invested_value + dynamic_bh_1m_cash
        dynamic_bh_1m_portfolio_history.append(dynamic_bh_1m_portfolio_value)

        # Update RISK-ADJUSTED MOMENTUM portfolio value daily (skip if disabled)
        risk_adj_mom_invested_value = 0.0
        if ENABLE_RISK_ADJ_MOM:
          # Iterate over actual positions, not just current stocks list
          for ticker in list(risk_adj_mom_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = risk_adj_mom_positions[ticker]['shares'] * current_price
                                    risk_adj_mom_positions[ticker]['value'] = position_value
                                    risk_adj_mom_invested_value += position_value
                                else:
                                    risk_adj_mom_invested_value += risk_adj_mom_positions[ticker].get('value', 0.0)
                            else:
                                risk_adj_mom_invested_value += risk_adj_mom_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating risk-adjusted momentum position for {ticker}: {e}")

        risk_adj_mom_portfolio_value = risk_adj_mom_invested_value + risk_adj_mom_cash
        risk_adj_mom_portfolio_history.append(risk_adj_mom_portfolio_value)

        # Update RISK-ADJ MOMENTUM SENTIMENT portfolio value daily (skip if disabled)
        if ENABLE_RISK_ADJ_MOM_SENTIMENT:
            risk_adj_mom_sentiment_invested_value = 0.0
            for ticker in list(risk_adj_mom_sentiment_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = risk_adj_mom_sentiment_positions[ticker]['shares'] * current_price
                                    risk_adj_mom_sentiment_positions[ticker]['value'] = position_value
                                    risk_adj_mom_sentiment_invested_value += position_value
                                else:
                                    risk_adj_mom_sentiment_invested_value += risk_adj_mom_sentiment_positions[ticker].get('value', 0.0)
                            else:
                                risk_adj_mom_sentiment_invested_value += risk_adj_mom_sentiment_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating risk adj mom sentiment position for {ticker}: {e}")

            risk_adj_mom_sentiment_portfolio_value = risk_adj_mom_sentiment_invested_value + risk_adj_mom_sentiment_cash
            risk_adj_mom_sentiment_portfolio_history.append(risk_adj_mom_sentiment_portfolio_value)

        # Update MULTI-TASK LEARNING portfolio value daily (skip if disabled)
        multitask_invested_value = 0.0
        if ENABLE_MULTITASK_LEARNING:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(multitask_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            # ticker_data already has date as index
                            current_price_data = ticker_data.loc[:current_date]
                            if not current_price_data.empty:
                                # Drop NaN values to avoid NaN propagation
                                valid_prices = current_price_data['Close'].dropna()
                                if len(valid_prices) > 0:
                                    current_price = valid_prices.iloc[-1]
                                    if not pd.isna(current_price) and current_price > 0:
                                        position_value = multitask_positions[ticker]['shares'] * current_price
                                        multitask_positions[ticker]['value'] = position_value
                                        multitask_invested_value += position_value
                                else:
                                    multitask_invested_value += multitask_positions[ticker].get('value', 0.0)
                            else:
                                multitask_invested_value += multitask_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        print(f"   ⚠️ Error updating multi-task learning position for {ticker}: {e}")
                        multitask_invested_value += multitask_positions[ticker].get('value', 0.0)

            multitask_portfolio_value = multitask_invested_value + multitask_cash
            multitask_portfolio_history.append(multitask_portfolio_value)

        # Update 3M/1Y RATIO portfolio value daily (skip if disabled)
        if ENABLE_3M_1Y_RATIO:
            ratio_3m_1y_invested_value = 0.0
            for ticker in list(ratio_3m_1y_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ratio_3m_1y_positions[ticker]['shares']
                            position_value = shares * current_price
                            ratio_3m_1y_positions[ticker]['value'] = position_value
                            ratio_3m_1y_invested_value += position_value
                        else:
                            ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)
                    else:
                        ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating 3M/1Y ratio position for {ticker}: {e}")
                    ratio_3m_1y_invested_value += ratio_3m_1y_positions[ticker].get('value', 0.0)

            ratio_3m_1y_portfolio_value = ratio_3m_1y_invested_value + ratio_3m_1y_cash
            ratio_3m_1y_portfolio_history.append(ratio_3m_1y_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID portfolio value daily
        momentum_volatility_hybrid_invested_value = 0.0
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(momentum_volatility_hybrid_positions.keys()):
                    try:
                        # Try to get current price from grouped data first
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        print(f"   ⚠️ Error updating Momentum-Volatility Hybrid position for {ticker}: {e}")
                        # Use previous value as fallback
                        momentum_volatility_hybrid_invested_value += momentum_volatility_hybrid_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_portfolio_value = momentum_volatility_hybrid_invested_value + momentum_volatility_hybrid_cash
        momentum_volatility_hybrid_portfolio_history.append(momentum_volatility_hybrid_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 6M portfolio value daily
        momentum_volatility_hybrid_6m_invested_value = 0.0
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M:
            for ticker in list(momentum_volatility_hybrid_6m_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_6m_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_6m_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_6m_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_6m_invested_value += momentum_volatility_hybrid_6m_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_6m_portfolio_value = momentum_volatility_hybrid_6m_invested_value + momentum_volatility_hybrid_6m_cash
        momentum_volatility_hybrid_6m_portfolio_history.append(momentum_volatility_hybrid_6m_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 1Y portfolio value daily
        momentum_volatility_hybrid_1y_invested_value = 0.0
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y:
            for ticker in list(momentum_volatility_hybrid_1y_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_1y_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_1y_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_1y_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_1y_invested_value += momentum_volatility_hybrid_1y_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_1y_portfolio_value = momentum_volatility_hybrid_1y_invested_value + momentum_volatility_hybrid_1y_cash
        momentum_volatility_hybrid_1y_portfolio_history.append(momentum_volatility_hybrid_1y_portfolio_value)

        # Update MOMENTUM-VOLATILITY HYBRID 1Y/3M portfolio value daily
        momentum_volatility_hybrid_1y3m_invested_value = 0.0
        if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M:
            for ticker in list(momentum_volatility_hybrid_1y3m_positions.keys()):
                    try:
                        ticker_df = ticker_data_grouped.get(ticker)
                        if ticker_df is not None:
                            current_price = _last_valid_close_up_to(ticker_df, current_date)
                            if current_price is not None:
                                shares = momentum_volatility_hybrid_1y3m_positions[ticker]['shares']
                                position_value = shares * current_price
                                momentum_volatility_hybrid_1y3m_positions[ticker]['value'] = position_value
                                momentum_volatility_hybrid_1y3m_invested_value += position_value
                            else:
                                momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)
                        else:
                            momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        momentum_volatility_hybrid_1y3m_invested_value += momentum_volatility_hybrid_1y3m_positions[ticker].get('value', 0.0)

        momentum_volatility_hybrid_1y3m_portfolio_value = momentum_volatility_hybrid_1y3m_invested_value + momentum_volatility_hybrid_1y3m_cash
        momentum_volatility_hybrid_1y3m_portfolio_history.append(momentum_volatility_hybrid_1y3m_portfolio_value)

        # Update PRICE ACCELERATION portfolio value daily
        price_acceleration_invested_value = 0.0
        if ENABLE_PRICE_ACCELERATION:
            for ticker in list(price_acceleration_positions.keys()):
                try:
                    # Get ticker data using the same method as other strategies
                    ticker_data = ticker_data_grouped.get(ticker) if isinstance(ticker_data_grouped, dict) else None
                    if ticker_data is None and hasattr(ticker_data_grouped, 'get_group'):
                        try:
                            ticker_data = ticker_data_grouped.get_group(ticker)
                        except KeyError:
                            ticker_data = None
                    
                    if ticker_data is not None and not ticker_data.empty:
                        current_price = _last_valid_close_up_to(ticker_data, current_date)
                        if current_price is not None:
                            shares = price_acceleration_positions[ticker]['shares']
                            position_value = shares * current_price
                            price_acceleration_positions[ticker]['value'] = position_value
                            price_acceleration_invested_value += position_value
                        else:
                            price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)
                    else:
                        price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Price Acceleration position for {ticker}: {e}")
                    price_acceleration_invested_value += price_acceleration_positions[ticker].get('value', 0.0)

        price_acceleration_portfolio_value = price_acceleration_invested_value + price_acceleration_cash
        price_acceleration_portfolio_history.append(price_acceleration_portfolio_value)

        # Update 1Y/3M RATIO portfolio value daily
        ratio_1y_3m_invested_value = 0.0
        for ticker in list(ratio_1y_3m_positions.keys()):
            try:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    current_price = _last_valid_close_up_to(ticker_df, current_date)
                    if current_price is not None:
                        shares = ratio_1y_3m_positions[ticker]['shares']
                        position_value = shares * current_price
                        ratio_1y_3m_positions[ticker]['value'] = position_value
                        ratio_1y_3m_invested_value += position_value
                    else:
                        ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)
                else:
                    ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)
            except Exception as e:
                print(f"   ⚠️ Error updating 1Y/3M ratio position for {ticker}: {e}")
                ratio_1y_3m_invested_value += ratio_1y_3m_positions[ticker].get('value', 0.0)

        ratio_1y_3m_portfolio_value = ratio_1y_3m_invested_value + ratio_1y_3m_cash
        ratio_1y_3m_portfolio_history.append(ratio_1y_3m_portfolio_value)

        # Update TURNAROUND portfolio value daily
        turnaround_invested_value = 0.0
        for ticker in list(turnaround_positions.keys()):
            try:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is not None:
                    current_price = _last_valid_close_up_to(ticker_df, current_date)
                    if current_price is not None:
                        shares = turnaround_positions[ticker]['shares']
                        position_value = shares * current_price
                        turnaround_positions[ticker]['value'] = position_value
                        turnaround_invested_value += position_value
                    else:
                        turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)
                else:
                    turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)
            except Exception as e:
                print(f"   ⚠️ Error updating turnaround position for {ticker}: {e}")
                turnaround_invested_value += turnaround_positions[ticker].get('value', 0.0)

        turnaround_portfolio_value = turnaround_invested_value + turnaround_cash
        turnaround_portfolio_history.append(turnaround_portfolio_value)

        # Update ADAPTIVE ENSEMBLE portfolio value daily (skip if disabled)
        if ENABLE_ADAPTIVE_STRATEGY:
            adaptive_ensemble_invested_value = 0.0
            for ticker in list(adaptive_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = adaptive_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            adaptive_ensemble_positions[ticker]['value'] = position_value
                            adaptive_ensemble_invested_value += position_value
                        else:
                            adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating adaptive ensemble position for {ticker}: {e}")
                    adaptive_ensemble_invested_value += adaptive_ensemble_positions[ticker].get('value', 0.0)

            adaptive_ensemble_portfolio_value = adaptive_ensemble_invested_value + adaptive_ensemble_cash
            adaptive_ensemble_portfolio_history.append(adaptive_ensemble_portfolio_value)

        # Update VOLATILITY ENSEMBLE portfolio value daily (skip if disabled)
        volatility_ensemble_invested_value = 0.0
        if ENABLE_VOLATILITY_ENSEMBLE:
            for ticker in list(volatility_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = volatility_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            volatility_ensemble_positions[ticker]['value'] = position_value
                            volatility_ensemble_invested_value += position_value
                        else:
                            volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating volatility ensemble position for {ticker}: {e}")
                    volatility_ensemble_invested_value += volatility_ensemble_positions[ticker].get('value', 0.0)

        volatility_ensemble_portfolio_value = volatility_ensemble_invested_value + volatility_ensemble_cash
        volatility_ensemble_portfolio_history.append(volatility_ensemble_portfolio_value)

        # Update ENHANCED VOLATILITY TRADER portfolio value daily (skip if disabled)
        enhanced_volatility_invested_value = 0.0
        if ENABLE_ENHANCED_VOLATILITY:
            for ticker in list(enhanced_volatility_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = enhanced_volatility_positions[ticker]['shares']
                            position_value = shares * current_price
                            enhanced_volatility_positions[ticker]['value'] = position_value
                            enhanced_volatility_invested_value += position_value
                        else:
                            enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)
                    else:
                        enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating enhanced volatility position for {ticker}: {e}")
                    enhanced_volatility_invested_value += enhanced_volatility_positions[ticker].get('value', 0.0)

        enhanced_volatility_portfolio_value = enhanced_volatility_invested_value + enhanced_volatility_cash
        enhanced_volatility_portfolio_history.append(enhanced_volatility_portfolio_value)
        
        # DEBUG: Log Enhanced Volatility state
        if day_count <= 5:
            print(f"   🔧 DEBUG: Enhanced Volatility Day {day_count} - Cash: ${enhanced_volatility_cash:.2f}, Invested: ${enhanced_volatility_invested_value:.2f}, Total: ${enhanced_volatility_portfolio_value:.2f}, Positions: {len(enhanced_volatility_positions)}")

        # Update AI VOLATILITY ENSEMBLE portfolio value daily (skip if disabled)
        ai_volatility_ensemble_invested_value = 0.0
        if ENABLE_AI_VOLATILITY_ENSEMBLE:
            for ticker in list(ai_volatility_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_volatility_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_volatility_ensemble_positions[ticker]['value'] = position_value
                            ai_volatility_ensemble_invested_value += position_value
                        else:
                            ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating AI volatility ensemble position for {ticker}: {e}")
                    ai_volatility_ensemble_invested_value += ai_volatility_ensemble_positions[ticker].get('value', 0.0)

        ai_volatility_ensemble_portfolio_value = ai_volatility_ensemble_invested_value + ai_volatility_ensemble_cash
        ai_volatility_ensemble_portfolio_history.append(ai_volatility_ensemble_portfolio_value)

        # Update MULTI-TIMEFRAME ENSEMBLE portfolio value daily (skip if disabled)
        multi_tf_ensemble_invested_value = 0.0
        if ENABLE_MULTI_TIMEFRAME_ENSEMBLE:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(multi_tf_ensemble_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = multi_tf_ensemble_positions[ticker]['shares'] * current_price
                                    multi_tf_ensemble_positions[ticker]['value'] = position_value
                                    multi_tf_ensemble_invested_value += position_value
                                else:
                                    multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)
                            else:
                                multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating multi-timeframe ensemble position for {ticker}: {e}")
                    multi_tf_ensemble_invested_value += multi_tf_ensemble_positions[ticker].get('value', 0.0)

        multi_tf_ensemble_portfolio_value = multi_tf_ensemble_invested_value + multi_tf_ensemble_cash
        multi_tf_ensemble_portfolio_history.append(multi_tf_ensemble_portfolio_value)

        # Update CORRELATION ENSEMBLE portfolio value daily (skip if disabled)
        correlation_ensemble_invested_value = 0.0
        if ENABLE_CORRELATION_ENSEMBLE:
            for ticker in list(correlation_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = correlation_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            correlation_ensemble_positions[ticker]['value'] = position_value
                            correlation_ensemble_invested_value += position_value
                        else:
                            correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating correlation ensemble position for {ticker}: {e}")
                    correlation_ensemble_invested_value += correlation_ensemble_positions[ticker].get('value', 0.0)

        correlation_ensemble_portfolio_value = correlation_ensemble_invested_value + correlation_ensemble_cash
        correlation_ensemble_portfolio_history.append(correlation_ensemble_portfolio_value)

        # Update DYNAMIC POOL portfolio value daily (skip if disabled)
        if ENABLE_DYNAMIC_POOL:
            dynamic_pool_invested_value = 0.0
            for ticker in list(dynamic_pool_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = dynamic_pool_positions[ticker]['shares']
                            position_value = shares * current_price
                            dynamic_pool_positions[ticker]['value'] = position_value
                            dynamic_pool_invested_value += position_value
                        else:
                            dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)
                    else:
                        dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating dynamic pool position for {ticker}: {e}")
                    dynamic_pool_invested_value += dynamic_pool_positions[ticker].get('value', 0.0)

            dynamic_pool_portfolio_value = dynamic_pool_invested_value + dynamic_pool_cash
            dynamic_pool_portfolio_history.append(dynamic_pool_portfolio_value)

        # Update VOTING ENSEMBLE portfolio value daily (skip if disabled)
        if ENABLE_VOTING_ENSEMBLE:
            voting_ensemble_invested_value = 0.0
            for ticker in list(voting_ensemble_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = voting_ensemble_positions[ticker]['shares']
                            position_value = shares * current_price
                            voting_ensemble_positions[ticker]['value'] = position_value
                            voting_ensemble_invested_value += position_value
                        else:
                            voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)
                    else:
                        voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating voting ensemble position for {ticker}: {e}")
                    voting_ensemble_invested_value += voting_ensemble_positions[ticker].get('value', 0.0)

            voting_ensemble_portfolio_value = voting_ensemble_invested_value + voting_ensemble_cash
            voting_ensemble_portfolio_history.append(voting_ensemble_portfolio_value)

        # Update MOMENTUM ACCELERATION portfolio value daily
        mom_accel_invested_value = 0.0
        if ENABLE_MOMENTUM_ACCELERATION:
            for ticker in list(mom_accel_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = mom_accel_positions[ticker]['shares']
                            position_value = shares * current_price
                            mom_accel_positions[ticker]['value'] = position_value
                            mom_accel_invested_value += position_value
                        else:
                            mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
                    else:
                        mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
                except Exception:
                    mom_accel_invested_value += mom_accel_positions[ticker].get('value', 0.0)
        mom_accel_portfolio_value = mom_accel_invested_value + mom_accel_cash
        mom_accel_portfolio_history.append(mom_accel_portfolio_value)

        # Update CONCENTRATED 3M portfolio value daily
        concentrated_3m_invested_value = 0.0
        if ENABLE_CONCENTRATED_3M:
            for ticker in list(concentrated_3m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = concentrated_3m_positions[ticker]['shares']
                            position_value = shares * current_price
                            concentrated_3m_positions[ticker]['value'] = position_value
                            concentrated_3m_invested_value += position_value
                        else:
                            concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
                    else:
                        concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
                except Exception:
                    concentrated_3m_invested_value += concentrated_3m_positions[ticker].get('value', 0.0)
        concentrated_3m_portfolio_value = concentrated_3m_invested_value + concentrated_3m_cash
        concentrated_3m_portfolio_history.append(concentrated_3m_portfolio_value)

        # Update DUAL MOMENTUM portfolio value daily
        dual_mom_invested_value = 0.0
        if ENABLE_DUAL_MOMENTUM:
            for ticker in list(dual_mom_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = dual_mom_positions[ticker]['shares']
                            position_value = shares * current_price
                            dual_mom_positions[ticker]['value'] = position_value
                            dual_mom_invested_value += position_value
                        else:
                            dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
                    else:
                        dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
                except Exception:
                    dual_mom_invested_value += dual_mom_positions[ticker].get('value', 0.0)
        dual_mom_portfolio_value = dual_mom_invested_value + dual_mom_cash
        dual_mom_portfolio_history.append(dual_mom_portfolio_value)

        # Update TREND FOLLOWING ATR portfolio value daily
        trend_atr_invested_value = 0.0
        if ENABLE_TREND_FOLLOWING_ATR:
            for ticker in list(trend_atr_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = trend_atr_positions[ticker]['shares']
                            position_value = shares * current_price
                            trend_atr_positions[ticker]['value'] = position_value
                            trend_atr_invested_value += position_value
                        else:
                            trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
                    else:
                        trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
                except Exception:
                    trend_atr_invested_value += trend_atr_positions[ticker].get('value', 0.0)
        trend_atr_portfolio_value = trend_atr_invested_value + trend_atr_cash
        trend_atr_portfolio_history.append(trend_atr_portfolio_value)

        # Update ELITE HYBRID portfolio value daily
        elite_hybrid_invested_value = 0.0
        if ENABLE_ELITE_HYBRID:
            for ticker in list(elite_hybrid_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = elite_hybrid_positions[ticker]['shares']
                            position_value = shares * current_price
                            elite_hybrid_positions[ticker]['value'] = position_value
                            elite_hybrid_invested_value += position_value
                        else:
                            elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
                    else:
                        elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
                except Exception:
                    elite_hybrid_invested_value += elite_hybrid_positions[ticker].get('value', 0.0)
        elite_hybrid_portfolio_value = elite_hybrid_invested_value + elite_hybrid_cash
        elite_hybrid_portfolio_history.append(elite_hybrid_portfolio_value)

        # Update ELITE RISK portfolio value daily
        elite_risk_invested_value = 0.0
        if ENABLE_ELITE_RISK:
            for ticker in list(elite_risk_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = elite_risk_positions[ticker]['shares']
                            position_value = shares * current_price
                            elite_risk_positions[ticker]['value'] = position_value
                            elite_risk_invested_value += position_value
                        else:
                            elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
                    else:
                        elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
                except Exception:
                    elite_risk_invested_value += elite_risk_positions[ticker].get('value', 0.0)
        elite_risk_portfolio_value = elite_risk_invested_value + elite_risk_cash
        elite_risk_portfolio_history.append(elite_risk_portfolio_value)

        # Update RISK-ADJ MOM 6M portfolio value daily
        risk_adj_mom_6m_invested_value = 0.0
        if ENABLE_RISK_ADJ_MOM_6M:
            for ticker in list(risk_adj_mom_6m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_6m_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_6m_positions[ticker]['value'] = position_value
                            risk_adj_mom_6m_invested_value += position_value
                        else:
                            risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_6m_invested_value += risk_adj_mom_6m_positions[ticker].get('value', 0.0)
        risk_adj_mom_6m_portfolio_value = risk_adj_mom_6m_invested_value + risk_adj_mom_6m_cash
        risk_adj_mom_6m_portfolio_history.append(risk_adj_mom_6m_portfolio_value)

        # Update RISK-ADJ MOM 3M portfolio value daily
        risk_adj_mom_3m_invested_value = 0.0
        if ENABLE_RISK_ADJ_MOM_3M:
            for ticker in list(risk_adj_mom_3m_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_invested_value += position_value
                        else:
                            risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_invested_value += risk_adj_mom_3m_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_portfolio_value = risk_adj_mom_3m_invested_value + risk_adj_mom_3m_cash
        risk_adj_mom_3m_portfolio_history.append(risk_adj_mom_3m_portfolio_value)

        # Update RISK-ADJ MOM 3M MONTHLY portfolio value daily
        risk_adj_mom_3m_monthly_invested_value = 0.0
        if ENABLE_RISK_ADJ_MOM_3M_MONTHLY:
            for ticker in list(risk_adj_mom_3m_monthly_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = risk_adj_mom_3m_monthly_positions[ticker]['shares']
                            position_value = shares * current_price
                            risk_adj_mom_3m_monthly_positions[ticker]['value'] = position_value
                            risk_adj_mom_3m_monthly_invested_value += position_value
                        else:
                            risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
                    else:
                        risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
                except Exception:
                    risk_adj_mom_3m_monthly_invested_value += risk_adj_mom_3m_monthly_positions[ticker].get('value', 0.0)
        risk_adj_mom_3m_monthly_portfolio_value = risk_adj_mom_3m_monthly_invested_value + risk_adj_mom_3m_monthly_cash
        risk_adj_mom_3m_monthly_portfolio_history.append(risk_adj_mom_3m_monthly_portfolio_value)

        # Update AI ELITE portfolio value daily
        ai_elite_invested_value = 0.0
        if ENABLE_AI_ELITE:
            for ticker in list(ai_elite_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = ai_elite_positions[ticker]['shares']
                            position_value = shares * current_price
                            ai_elite_positions[ticker]['value'] = position_value
                            ai_elite_invested_value += position_value
                        else:
                            ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
                    else:
                        ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
                except Exception:
                    ai_elite_invested_value += ai_elite_positions[ticker].get('value', 0.0)
        ai_elite_portfolio_value = ai_elite_invested_value + ai_elite_cash
        ai_elite_portfolio_history.append(ai_elite_portfolio_value)

        # Update MEAN REVERSION portfolio value daily (skip if disabled)
        mean_reversion_invested_value = 0.0
        if ENABLE_MEAN_REVERSION:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(mean_reversion_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            current_price_data = ticker_data.loc[ticker_data.index == current_date]
                            if not current_price_data.empty:
                                current_price = current_price_data['Close'].iloc[0]
                            else:
                                current_price = None
                        else:
                            current_price = None

                        if current_price is not None and current_price > 0:
                            shares = mean_reversion_positions[ticker]['shares']
                            value = shares * current_price
                            mean_reversion_positions[ticker]['value'] = value
                            mean_reversion_invested_value += value
                    except Exception as e:
                        print(f"   ⚠️ Error updating mean reversion position for {ticker}: {e}")

        mean_reversion_portfolio_value = mean_reversion_invested_value + mean_reversion_cash
        mean_reversion_portfolio_history.append(mean_reversion_portfolio_value)

        # Update QUALITY + MOMENTUM portfolio value daily (skip if disabled)
        quality_momentum_invested_value = 0.0
        if ENABLE_QUALITY_MOM:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(quality_momentum_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            current_price_data = ticker_data.loc[ticker_data.index == current_date]
                            if not current_price_data.empty:
                                current_price = current_price_data['Close'].iloc[0]
                            else:
                                current_price = None
                        else:
                            current_price = None

                        if current_price is not None and current_price > 0:
                            shares = quality_momentum_positions[ticker]['shares']
                            value = shares * current_price
                            quality_momentum_positions[ticker]['value'] = value
                            quality_momentum_invested_value += value
                    except Exception as e:
                        print(f"   ⚠️ Error updating quality + momentum position for {ticker}: {e}")

        quality_momentum_portfolio_value = quality_momentum_invested_value + quality_momentum_cash
        quality_momentum_portfolio_history.append(quality_momentum_portfolio_value)

        # Update VOLATILITY-ADJUSTED MOMENTUM portfolio value daily (skip if disabled)
        if ENABLE_VOLATILITY_ADJ_MOM:
            volatility_adj_mom_invested_value = 0.0
            for ticker, pos in volatility_adj_mom_positions.items():
                try:
                    # Get current price
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        current_price_data = ticker_data.loc[ticker_data.index == current_date]
                        if not current_price_data.empty:
                            price_data = current_price_data['Close']
                        else:
                            price_data = pd.Series()
                    else:
                        price_data = pd.Series()
                    
                    if not price_data.empty:
                        current_price = price_data.iloc[0]
                        position_value = pos['shares'] * current_price
                        volatility_adj_mom_invested_value += position_value
                        
                        # Update stored position value
                        pos['value'] = position_value
                    else:
                        # Use previous value if current price is invalid
                        volatility_adj_mom_invested_value += pos.get('value', 0.0)
                except Exception:
                    # Keep previous value if price lookup fails
                    volatility_adj_mom_invested_value += pos.get('value', 0.0)

        volatility_adj_mom_portfolio_value = volatility_adj_mom_invested_value + volatility_adj_mom_cash
        volatility_adj_mom_portfolio_history.append(volatility_adj_mom_portfolio_value)

        # === MOMENTUM + AI HYBRID: Update portfolio value ===
        if ENABLE_MOMENTUM_AI_HYBRID:
            momentum_ai_hybrid_invested_value = 0.0
            for ticker in list(momentum_ai_hybrid_positions.keys()):
                try:
                    ticker_df = ticker_data_grouped.get(ticker)
                    if ticker_df is not None:
                        current_price = _last_valid_close_up_to(ticker_df, current_date)
                        if current_price is not None:
                            shares = momentum_ai_hybrid_positions[ticker]['shares']
                            position_value = shares * current_price
                            momentum_ai_hybrid_positions[ticker]['value'] = position_value
                            momentum_ai_hybrid_invested_value += position_value
                        else:
                            momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)
                    else:
                        momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)
                except Exception:
                    momentum_ai_hybrid_invested_value += momentum_ai_hybrid_positions[ticker].get('value', 0.0)
            
            momentum_ai_hybrid_portfolio_value = momentum_ai_hybrid_invested_value + momentum_ai_hybrid_cash
            momentum_ai_hybrid_portfolio_history.append(momentum_ai_hybrid_portfolio_value)

        # Update STATIC BH 1Y portfolio value daily (skip if disabled)
        static_bh_1y_invested_value = 0.0
        if ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_1y_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_1y_positions[ticker]['shares'] * current_price
                                    static_bh_1y_positions[ticker]['value'] = position_value
                                    static_bh_1y_invested_value += position_value
                                else:
                                    static_bh_1y_invested_value += static_bh_1y_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_1y_invested_value += static_bh_1y_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 1Y position for {ticker}: {e}")

        static_bh_1y_portfolio_value = static_bh_1y_invested_value + static_bh_1y_cash
        static_bh_1y_portfolio_history.append(static_bh_1y_portfolio_value)

        # Update STATIC BH 6M portfolio value daily (skip if disabled)
        static_bh_6m_invested_value = 0.0
        if ENABLE_STATIC_BH_6M:
            for ticker in list(static_bh_6m_positions.keys()):
                try:
                    if ticker in ticker_data_grouped:
                        ticker_data = ticker_data_grouped[ticker]
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_6m_positions[ticker]['shares'] * current_price
                                    static_bh_6m_positions[ticker]['value'] = position_value
                                    static_bh_6m_invested_value += position_value
                                else:
                                    static_bh_6m_invested_value += static_bh_6m_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_6m_invested_value += static_bh_6m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 6M position for {ticker}: {e}")

        static_bh_6m_portfolio_value = static_bh_6m_invested_value + static_bh_6m_cash
        static_bh_6m_portfolio_history.append(static_bh_6m_portfolio_value)

        # Update STATIC BH 3M portfolio value daily (skip if disabled)
        static_bh_3m_invested_value = 0.0
        if ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_3m_positions.keys()):
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = static_bh_3m_positions[ticker]['shares'] * current_price
                                    static_bh_3m_positions[ticker]['value'] = position_value
                                    static_bh_3m_invested_value += position_value
                                else:
                                    static_bh_3m_invested_value += static_bh_3m_positions[ticker].get('value', 0.0)
                            else:
                                static_bh_3m_invested_value += static_bh_3m_positions[ticker].get('value', 0.0)
                except Exception as e:
                    print(f"   ⚠️ Error updating Static BH 3M position for {ticker}: {e}")

        static_bh_3m_portfolio_value = static_bh_3m_invested_value + static_bh_3m_cash
        static_bh_3m_portfolio_history.append(static_bh_3m_portfolio_value)

        # Update STATIC BH 1M portfolio value daily (skip if disabled)
        static_bh_1m_invested_value = 0.0
        if ENABLE_STATIC_BH:
            # Iterate over actual positions, not just current stocks list
            for ticker in list(static_bh_1m_positions.keys()):
                    try:
                        ticker_data = ticker_data_grouped.get(ticker)
                        if ticker_data is not None and not ticker_data.empty:
                            # ticker_data already has date as index
                            current_price_data = ticker_data.loc[:current_date]
                            if not current_price_data.empty:
                                # Drop NaN values to avoid NaN propagation
                                valid_prices = current_price_data['Close'].dropna()
                                if len(valid_prices) > 0:
                                    current_price = valid_prices.iloc[-1]
                                    if not pd.isna(current_price) and current_price > 0:
                                        position_value = static_bh_1m_positions[ticker]['shares'] * current_price
                                        static_bh_1m_positions[ticker]['value'] = position_value
                                        static_bh_1m_invested_value += position_value
                                    else:
                                        static_bh_1m_invested_value += static_bh_1m_positions[ticker].get('value', 0.0)
                                else:
                                    static_bh_1m_invested_value += static_bh_1m_positions[ticker].get('value', 0.0)
                    except Exception as e:
                        print(f"   ⚠️ Error updating Static BH 1M position for {ticker}: {e}")

        static_bh_1m_portfolio_value = static_bh_1m_invested_value + static_bh_1m_cash
        static_bh_1m_portfolio_history.append(static_bh_1m_portfolio_value)

        # Update STATIC BH MONTHLY variants portfolio values daily
        for monthly_var in [
            ('1Y', ENABLE_STATIC_BH_1Y_MONTHLY, static_bh_1y_monthly_positions, 'static_bh_1y_monthly'),
            ('6M', ENABLE_STATIC_BH_6M_MONTHLY, static_bh_6m_monthly_positions, 'static_bh_6m_monthly'),
            ('3M', ENABLE_STATIC_BH_3M_MONTHLY, static_bh_3m_monthly_positions, 'static_bh_3m_monthly'),
        ]:
            label, enabled, positions_dict, var_prefix = monthly_var
            if enabled:
                invested = 0.0
                for ticker in list(positions_dict.keys()):
                    try:
                        if ticker in ticker_data_grouped:
                            td = ticker_data_grouped[ticker]
                            price_data = td.loc[:current_date]
                            if not price_data.empty:
                                valid_close = price_data['Close'].dropna()
                                if len(valid_close) > 0:
                                    cp = valid_close.iloc[-1]
                                    positions_dict[ticker]['value'] = positions_dict[ticker]['shares'] * cp
                                    invested += positions_dict[ticker]['value']
                                else:
                                    invested += positions_dict[ticker].get('value', 0.0)
                            else:
                                invested += positions_dict[ticker].get('value', 0.0)
                    except Exception:
                        invested += positions_dict[ticker].get('value', 0.0)
                if var_prefix == 'static_bh_1y_monthly':
                    static_bh_1y_monthly_portfolio_value = invested + static_bh_1y_monthly_cash
                    static_bh_1y_monthly_portfolio_history.append(static_bh_1y_monthly_portfolio_value)
                elif var_prefix == 'static_bh_6m_monthly':
                    static_bh_6m_monthly_portfolio_value = invested + static_bh_6m_monthly_cash
                    static_bh_6m_monthly_portfolio_history.append(static_bh_6m_monthly_portfolio_value)
                elif var_prefix == 'static_bh_3m_monthly':
                    static_bh_3m_monthly_portfolio_value = invested + static_bh_3m_monthly_cash
                    static_bh_3m_monthly_portfolio_history.append(static_bh_3m_monthly_portfolio_value)

        # Update portfolio value (invested + cash) at end of each day
        invested_value = 0.0
        for ticker in current_portfolio_stocks:
            if ticker in positions and positions[ticker]['shares'] > 0:
                # Get current price
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # ticker_data already has date as index
                        current_price_data = ticker_data.loc[:current_date]
                        if not current_price_data.empty:
                            # Drop NaN values to avoid NaN propagation
                            valid_prices = current_price_data['Close'].dropna()
                            if len(valid_prices) > 0:
                                current_price = valid_prices.iloc[-1]
                                if not pd.isna(current_price) and current_price > 0:
                                    position_value = positions[ticker]['shares'] * current_price
                                    invested_value += position_value
                                    
                                    # ✅ NEW: Track daily contribution for this stock
                                    old_value = positions[ticker].get('value', position_value)
                                    daily_change = position_value - old_value
                                    
                                    if ticker not in stock_performance_tracking:
                                        stock_performance_tracking[ticker] = {
                                            'days_held': 0,
                                            'contribution': 0.0,
                                            'max_shares': 0.0,
                                            'entry_value': position_value,
                                            'total_invested': 0.0
                                        }
                                    
                                    stock_performance_tracking[ticker]['days_held'] += 1
                                    stock_performance_tracking[ticker]['contribution'] += daily_change
                                    stock_performance_tracking[ticker]['max_shares'] = max(
                                        stock_performance_tracking[ticker]['max_shares'],
                                        positions[ticker]['shares']
                                    )
                                    
                                    # Update stored position value
                                    positions[ticker]['value'] = position_value
                                else:
                                    # Use previous value if current price is invalid
                                    invested_value += positions[ticker].get('value', 0.0)
                            else:
                                invested_value += positions[ticker].get('value', 0.0)
                except Exception:
                    # Keep previous value if price lookup fails
                    invested_value += positions[ticker].get('value', 0.0)

        # Validate values before calculation
        if pd.isna(invested_value):
            invested_value = 0.0
        if pd.isna(cash_balance):
            cash_balance = 0.0
            
        total_portfolio_value = invested_value + cash_balance
        
        # Debug: Log if portfolio value is 0 (might indicate an issue)
        if day_count == 1 and total_portfolio_value == 0:
            print(f"   ⚠️ DEBUG: Day 1 portfolio value is 0 (invested: {invested_value}, cash: {cash_balance})")

        # Update portfolio value history
        portfolio_values_history.append(total_portfolio_value)

        # ✅ NEW: Calculate actual returns vs predictions at end of each day
        if day_predictions['predictions'] and day_count > 1:
            # Calculate actual returns for the next prediction horizon (e.g., 20 days)
            future_date = current_date + timedelta(days=horizon_days)
            
            prediction_results = []
            # ✅ OPTIMIZED: Use pre-grouped data
            for ticker, predicted_return in day_predictions['predictions']:
                try:
                    # Get current and future price
                    if ticker not in ticker_data_grouped:
                        continue
                    ticker_data = ticker_data_grouped[ticker]
                    
                    # Current price
                    current_data = ticker_data.loc[:current_date]
                    if not current_data.empty:
                        current_price = current_data['Close'].iloc[-1]
                        
                        # Validate current price
                        if pd.isna(current_price) or current_price <= 0:
                            # Skip if current price is invalid
                            continue
                        
                        # Future price (if available in our data)
                        future_data = ticker_data.loc[:future_date]
                        if not future_data.empty and len(future_data) > len(current_data):
                            future_price = future_data['Close'].iloc[-1]
                            
                            # Check if future price is valid
                            if pd.isna(future_price) or future_price <= 0:
                                # Future price exists but is invalid - mark as NaN
                                actual_return = np.nan
                                bh_return = np.nan
                                future_price = np.nan
                            else:
                                actual_return = (future_price / current_price - 1)
                                bh_return = actual_return  # Same as actual
                            
                            prediction_results.append({
                                'ticker': ticker,
                                'predicted_return': predicted_return,
                                'actual_return': actual_return,
                                'bh_return': bh_return,
                                'prediction_error': abs(predicted_return - actual_return) if not pd.isna(actual_return) else np.nan,
                                'current_price': current_price,
                                'future_price': future_price,
                                'model_status': '✅' if ticker in current_models and current_models[ticker] is not None else '❌'
                            })
                        else:
                            # Future data not available yet - still show the prediction but mark actual as NaN
                            prediction_results.append({
                                'ticker': ticker,
                                'predicted_return': predicted_return,
                                'actual_return': np.nan,
                                'bh_return': np.nan,
                                'prediction_error': np.nan,
                                'current_price': current_price,
                                'future_price': np.nan,
                                'model_status': '✅' if ticker in current_models and current_models[ticker] is not None else '❌'
                            })
                except Exception:
                    continue
            
            # Store results
            if prediction_results:
                day_predictions['results'] = prediction_results
                daily_prediction_log.append(day_predictions)
                
                # Print daily comparison (every day)
                if True:  # Show every day for better visibility
                    print(f"\n   📊 Day {day_count} - AI Predictions vs Buy & Hold (Next {horizon_days} days):")
                    print(f"   {'Ticker':<8} {'AI Pred':<10} {'Buy & Hold':<12} {'Error':<10} {'Dir':<5} {'Models':<8} {'Port':<6}")
                    print(f"   {'-'*75}")
                    
                    # Sort all predictions by AI performance (highest first)
                    sorted_predictions = sorted(prediction_results, key=lambda x: x['predicted_return'], reverse=True)
                    
                    for i, res in enumerate(sorted_predictions):
                        # Handle NaN values gracefully in display
                        if pd.isna(res['bh_return']):
                            bh_str = "N/A".rjust(10)
                            error_str = "N/A".rjust(8)
                            direction = "?"
                        else:
                            # Check if direction is correct
                            pred_up = res['predicted_return'] > 0
                            actual_up = res['bh_return'] > 0
                            direction = "✓" if pred_up == actual_up else "✗"
                            error_str = f"{abs(res['prediction_error']*100):.1f}%".rjust(8)
                            bh_str = f"{res['bh_return']*100:+.2f}%".rjust(10)
                        
                        # Show if stock is in top 10 (selected for portfolio)
                        portfolio_mark = "🟢" if i < PORTFOLIO_SIZE else "  "
                        
                        pred_str = f"{res['predicted_return']*100:+.1f}%".rjust(8)
                        print(f"   {res['ticker']:<8} {pred_str} {bh_str} {error_str} {direction:<5} {res['model_status']:<8} {portfolio_mark}")
        
        # Periodic progress update
        if day_count % 50 == 0:
            print(f"   📈 Processed {day_count}/{len(business_days)} days, portfolio: {current_portfolio_stocks}")
        
        # === DAILY SUMMARY ===
        if True:  # Print daily summary every day
            print(f"\n📊 DAILY SUMMARY - Day {day_count} ({current_date.strftime('%Y-%m-%d')})")
            print("=" * 80)
            
            # Get current portfolio values for all strategies
            strategy_values = [
                ("Static BH 1Y", static_bh_1y_portfolio_value if ENABLE_STATIC_BH else None),
                ("Static BH 6M", static_bh_6m_portfolio_value if ENABLE_STATIC_BH_6M else None),
                ("Static BH 3M", static_bh_3m_portfolio_value if ENABLE_STATIC_BH else None),
                ("Static BH 1M", static_bh_1m_portfolio_value if ENABLE_STATIC_BH else None),
                ("Dynamic BH 1Y", dynamic_bh_portfolio_value if ENABLE_DYNAMIC_BH_1Y else None),
                ("Dynamic BH 6M", dynamic_bh_6m_portfolio_value if ENABLE_DYNAMIC_BH_6M else None),
                ("Dynamic BH 3M", dynamic_bh_3m_portfolio_value if ENABLE_DYNAMIC_BH_3M else None),
                ("Dynamic BH 1M", dynamic_bh_1m_portfolio_value if ENABLE_DYNAMIC_BH_1M else None),
                ("Risk-Adj Mom", risk_adj_mom_portfolio_value if ENABLE_RISK_ADJ_MOM else None),
                ("Mean Reversion", mean_reversion_portfolio_value if ENABLE_MEAN_REVERSION else None),
                ("Quality+Mom", quality_momentum_portfolio_value if ENABLE_QUALITY_MOM else None),
                ("Momentum+AI", momentum_ai_hybrid_portfolio_value if ENABLE_MOMENTUM_AI_HYBRID else None),
                ("Vol-Adj Mom", volatility_adj_mom_portfolio_value if ENABLE_VOLATILITY_ADJ_MOM else None),
                ("Dynamic BH 1Y+Vol", dynamic_bh_1y_vol_filter_portfolio_value if ENABLE_DYNAMIC_BH_1Y_VOL_FILTER else None),
                ("Dynamic BH 1Y+TS", dynamic_bh_1y_trailing_stop_portfolio_value if ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP else None),
                ("Sector Rotation", sector_rotation_portfolio_value if ENABLE_SECTOR_ROTATION else None),
                ("Multi-Task", multitask_portfolio_value if ENABLE_MULTITASK_LEARNING else None),
                ("3M/1Y Ratio", ratio_3m_1y_portfolio_value if ENABLE_3M_1Y_RATIO else None),
                ("1Y/3M Ratio", ratio_1y_3m_portfolio_value),
                ("Mom-Vol Hybrid", momentum_volatility_hybrid_portfolio_value if ENABLE_MOMENTUM_VOLATILITY_HYBRID else None),
                ("Mom-Vol Hybrid 6M", momentum_volatility_hybrid_6m_portfolio_value if ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M else None),
                ("Mom-Vol Hybrid 1Y", momentum_volatility_hybrid_1y_portfolio_value if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y else None),
                ("Mom-Vol Hybrid 1Y/3M", momentum_volatility_hybrid_1y3m_portfolio_value if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M else None),
                ("Price Acceleration", price_acceleration_portfolio_value if ENABLE_PRICE_ACCELERATION else None),
                ("Turnaround", turnaround_portfolio_value if ENABLE_TURNAROUND else None),
                ("Adaptive Ensemble", adaptive_ensemble_portfolio_value if ENABLE_ADAPTIVE_STRATEGY else None),
                ("Volatility Ensemble", volatility_ensemble_portfolio_value if ENABLE_VOLATILITY_ENSEMBLE else None),
                ("Enhanced Volatility", enhanced_volatility_portfolio_value if ENABLE_ENHANCED_VOLATILITY else None),
                ("AI Volatility Ensemble", ai_volatility_ensemble_portfolio_value if ENABLE_AI_VOLATILITY_ENSEMBLE else None),
                ("Correlation Ensemble", correlation_ensemble_portfolio_value if ENABLE_CORRELATION_ENSEMBLE else None),
                ("Dynamic Pool", dynamic_pool_portfolio_value if ENABLE_DYNAMIC_POOL else None),
                ("Risk-Adj Mom Sentiment", risk_adj_mom_sentiment_portfolio_value if ENABLE_RISK_ADJ_MOM_SENTIMENT else None),
                ("Voting Ensemble", voting_ensemble_portfolio_value if ENABLE_VOTING_ENSEMBLE else None),
                ("Mom Acceleration", mom_accel_portfolio_value if ENABLE_MOMENTUM_ACCELERATION else None),
                ("Concentrated 3M", concentrated_3m_portfolio_value if ENABLE_CONCENTRATED_3M else None),
                ("Dual Momentum", dual_mom_portfolio_value if ENABLE_DUAL_MOMENTUM else None),
                ("Trend ATR", trend_atr_portfolio_value if ENABLE_TREND_FOLLOWING_ATR else None),
                ("Elite Hybrid", elite_hybrid_portfolio_value if ENABLE_ELITE_HYBRID else None),
                ("Elite Risk", elite_risk_portfolio_value if ENABLE_ELITE_RISK else None),
                ("Risk-Adj Mom 6M", risk_adj_mom_6m_portfolio_value if ENABLE_RISK_ADJ_MOM_6M else None),
                ("Risk-Adj Mom 3M", risk_adj_mom_3m_portfolio_value if ENABLE_RISK_ADJ_MOM_3M else None),
                ("RiskAdj 3M Mth", risk_adj_mom_3m_monthly_portfolio_value if ENABLE_RISK_ADJ_MOM_3M_MONTHLY else None),
                ("AI Elite", ai_elite_portfolio_value if ENABLE_AI_ELITE else None),
                ("BH 1Y Monthly", static_bh_1y_monthly_portfolio_value if ENABLE_STATIC_BH_1Y_MONTHLY else None),
                ("BH 6M Monthly", static_bh_6m_monthly_portfolio_value if ENABLE_STATIC_BH_6M_MONTHLY else None),
                ("BH 3M Monthly", static_bh_3m_monthly_portfolio_value if ENABLE_STATIC_BH_3M_MONTHLY else None),
            ]
            
            # Filter out None values and sort by performance
            active_strategies = [(name, value) for name, value in strategy_values if value is not None]
            active_strategies.sort(key=lambda x: x[1], reverse=True)
            
            # Track Top 5 consistency
            for rank, (name, value) in enumerate(active_strategies[:5], 1):
                if name not in top5_consistency_counts:
                    top5_consistency_counts[name] = 0
                top5_consistency_counts[name] += 1
            
            # Show ALL strategies (not just top 10) with cash and allocation info
            print(f"{'Rank':<5} {'Strategy':<20} {'Value':<12} {'Return':<10} {'Ann. Ret':<10} {'Cash':<12} {'Positions':<10}")
            print("-" * 85)
            
            # Prepare strategy data with cash and position info
            strategy_details = []
            for name, value in active_strategies:
                # Get cash and positions for each strategy
                strat_cash = 0
                num_positions = 0
                invested = 0
                
                if name == "AI Strategy" and enable_ai_strategy:
                    strat_cash = cash_balance
                    num_positions = len(current_portfolio_stocks)
                    invested = value - strat_cash
                elif name == "Static BH 1Y" and ENABLE_STATIC_BH:
                    strat_cash = static_bh_1y_cash
                    num_positions = len(current_static_bh_1y_stocks)
                    invested = value - strat_cash
                elif name == "Static BH 6M" and ENABLE_STATIC_BH_6M:
                    strat_cash = static_bh_6m_cash
                    num_positions = len(current_static_bh_6m_stocks)
                    invested = value - strat_cash
                elif name == "Static BH 3M" and ENABLE_STATIC_BH:
                    strat_cash = static_bh_3m_cash
                    num_positions = len(current_static_bh_3m_stocks)
                    invested = value - strat_cash
                elif name == "Static BH 1M" and ENABLE_STATIC_BH:
                    strat_cash = static_bh_1m_cash
                    num_positions = len(current_static_bh_1m_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 1Y" and ENABLE_DYNAMIC_BH_1Y:
                    strat_cash = dynamic_bh_cash
                    num_positions = len(current_dynamic_bh_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 6M" and ENABLE_DYNAMIC_BH_6M:
                    strat_cash = dynamic_bh_6m_cash
                    num_positions = len(current_dynamic_bh_6m_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 3M" and ENABLE_DYNAMIC_BH_3M:
                    strat_cash = dynamic_bh_3m_cash
                    num_positions = len(current_dynamic_bh_3m_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 1M" and ENABLE_DYNAMIC_BH_1M:
                    strat_cash = dynamic_bh_1m_cash
                    num_positions = len(current_dynamic_bh_1m_stocks)
                    invested = value - strat_cash
                elif name == "Risk-Adj Mom" and ENABLE_RISK_ADJ_MOM:
                    strat_cash = risk_adj_mom_cash
                    num_positions = len(current_risk_adj_mom_stocks)
                    invested = value - strat_cash
                elif name == "Mean Reversion" and ENABLE_MEAN_REVERSION:
                    strat_cash = mean_reversion_cash
                    num_positions = len(current_mean_reversion_stocks)
                    invested = value - strat_cash
                elif name == "Quality+Mom" and ENABLE_QUALITY_MOM:
                    strat_cash = quality_momentum_cash
                    num_positions = len(current_quality_momentum_stocks)
                    invested = value - strat_cash
                elif name == "Momentum+AI" and ENABLE_MOMENTUM_AI_HYBRID:
                    strat_cash = momentum_ai_hybrid_cash
                    num_positions = len(momentum_ai_hybrid_positions)
                    invested = value - strat_cash
                elif name == "Vol-Adj Mom" and ENABLE_VOLATILITY_ADJ_MOM:
                    strat_cash = volatility_adj_mom_cash
                    num_positions = len(current_volatility_adj_mom_stocks)
                    invested = value - strat_cash
                elif name == "Enhanced Volatility" and ENABLE_ENHANCED_VOLATILITY:
                    strat_cash = enhanced_volatility_cash
                    num_positions = len(enhanced_volatility_positions)
                    invested = value - strat_cash
                elif name == "AI Volatility Ensemble" and ENABLE_AI_VOLATILITY_ENSEMBLE:
                    strat_cash = ai_volatility_ensemble_cash
                    num_positions = len(ai_volatility_ensemble_positions)
                    invested = value - strat_cash
                elif name == "Turnaround" and ENABLE_TURNAROUND:
                    strat_cash = turnaround_cash
                    num_positions = len(current_turnaround_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 1Y+Vol" and ENABLE_DYNAMIC_BH_1Y_VOL_FILTER:
                    strat_cash = dynamic_bh_1y_vol_filter_cash
                    num_positions = len(current_dynamic_bh_1y_vol_filter_stocks)
                    invested = value - strat_cash
                elif name == "Dynamic BH 1Y+TS" and ENABLE_DYNAMIC_BH_1Y_TRAILING_STOP:
                    strat_cash = dynamic_bh_1y_trailing_stop_cash
                    num_positions = len(current_dynamic_bh_1y_trailing_stop_stocks)
                    invested = value - strat_cash
                elif name == "Volatility Ensemble" and ENABLE_VOLATILITY_ENSEMBLE:
                    strat_cash = volatility_ensemble_cash
                    num_positions = len(volatility_ensemble_positions)
                    invested = value - strat_cash
                elif name == "Correlation Ensemble" and ENABLE_CORRELATION_ENSEMBLE:
                    strat_cash = correlation_ensemble_cash
                    num_positions = len(correlation_ensemble_positions)
                    invested = value - strat_cash
                elif name == "1Y/3M Ratio":
                    strat_cash = ratio_1y_3m_cash
                    num_positions = len(current_ratio_1y_3m_stocks) if 'current_ratio_1y_3m_stocks' in dir() else 0
                    invested = value - strat_cash
                elif name == "Mom Acceleration" and ENABLE_MOMENTUM_ACCELERATION:
                    strat_cash = mom_accel_cash
                    num_positions = len(mom_accel_positions)
                    invested = value - strat_cash
                elif name == "Concentrated 3M" and ENABLE_CONCENTRATED_3M:
                    strat_cash = concentrated_3m_cash
                    num_positions = len(concentrated_3m_positions)
                    invested = value - strat_cash
                elif name == "Dual Momentum" and ENABLE_DUAL_MOMENTUM:
                    strat_cash = dual_mom_cash
                    num_positions = len(dual_mom_positions)
                    invested = value - strat_cash
                elif name == "Trend ATR" and ENABLE_TREND_FOLLOWING_ATR:
                    strat_cash = trend_atr_cash
                    num_positions = len(trend_atr_positions)
                    invested = value - strat_cash
                elif name == "Elite Hybrid" and ENABLE_ELITE_HYBRID:
                    strat_cash = elite_hybrid_cash
                    num_positions = len(elite_hybrid_positions)
                    invested = value - strat_cash
                elif name == "Elite Risk" and ENABLE_ELITE_RISK:
                    strat_cash = elite_risk_cash
                    num_positions = len(elite_risk_positions)
                    invested = value - strat_cash
                elif name == "Risk-Adj Mom 6M" and ENABLE_RISK_ADJ_MOM_6M:
                    strat_cash = risk_adj_mom_6m_cash
                    num_positions = len(risk_adj_mom_6m_positions)
                    invested = value - strat_cash
                elif name == "Risk-Adj Mom 3M" and ENABLE_RISK_ADJ_MOM_3M:
                    strat_cash = risk_adj_mom_3m_cash
                    num_positions = len(risk_adj_mom_3m_positions)
                    invested = value - strat_cash
                elif name == "RiskAdj 3M Mth" and ENABLE_RISK_ADJ_MOM_3M_MONTHLY:
                    strat_cash = risk_adj_mom_3m_monthly_cash
                    num_positions = len(risk_adj_mom_3m_monthly_positions)
                    invested = value - strat_cash
                elif name == "AI Elite" and ENABLE_AI_ELITE:
                    strat_cash = ai_elite_cash
                    num_positions = len(ai_elite_positions)
                    invested = value - strat_cash
                elif name == "Risk-Adj Mom Sentiment" and ENABLE_RISK_ADJ_MOM_SENTIMENT:
                    strat_cash = risk_adj_mom_sentiment_cash
                    num_positions = len(risk_adj_mom_sentiment_positions)
                    invested = value - strat_cash
                elif name == "Voting Ensemble" and ENABLE_VOTING_ENSEMBLE:
                    strat_cash = voting_ensemble_cash
                    num_positions = len(voting_ensemble_positions)
                    invested = value - strat_cash
                elif name == "Adaptive Ensemble" and ENABLE_ADAPTIVE_STRATEGY:
                    strat_cash = adaptive_ensemble_cash
                    num_positions = len(adaptive_ensemble_positions)
                    invested = value - strat_cash
                elif name == "Dynamic Pool" and ENABLE_DYNAMIC_POOL:
                    strat_cash = dynamic_pool_cash
                    num_positions = len(dynamic_pool_positions)
                    invested = value - strat_cash
                elif name == "Multi-Task" and ENABLE_MULTITASK_LEARNING:
                    strat_cash = multitask_cash
                    num_positions = len(multitask_positions)
                    invested = value - strat_cash
                elif name == "Sector Rotation" and ENABLE_SECTOR_ROTATION:
                    strat_cash = sector_rotation_cash
                    num_positions = len(sector_rotation_positions)
                    invested = value - strat_cash
                elif name == "Mom-Vol Hybrid" and ENABLE_MOMENTUM_VOLATILITY_HYBRID:
                    strat_cash = momentum_volatility_hybrid_cash
                    num_positions = len(momentum_volatility_hybrid_positions)
                    invested = value - strat_cash
                elif name == "Mom-Vol Hybrid 6M" and ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M:
                    strat_cash = momentum_volatility_hybrid_6m_cash
                    num_positions = len(momentum_volatility_hybrid_6m_positions)
                    invested = value - strat_cash
                elif name == "Mom-Vol Hybrid 1Y" and ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y:
                    strat_cash = momentum_volatility_hybrid_1y_cash
                    num_positions = len(momentum_volatility_hybrid_1y_positions)
                    invested = value - strat_cash
                elif name == "Mom-Vol Hybrid 1Y/3M" and ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M:
                    strat_cash = momentum_volatility_hybrid_1y3m_cash
                    num_positions = len(momentum_volatility_hybrid_1y3m_positions)
                    invested = value - strat_cash
                elif name == "3M/1Y Ratio" and ENABLE_3M_1Y_RATIO:
                    strat_cash = ratio_3m_1y_cash
                    num_positions = len(ratio_3m_1y_positions)
                    invested = value - strat_cash
                elif name == "Multi-TF Ensemble" and ENABLE_MULTI_TIMEFRAME_ENSEMBLE:
                    strat_cash = multi_tf_ensemble_cash
                    num_positions = len(multi_tf_ensemble_positions)
                    invested = value - strat_cash
                elif name == "Price Acceleration" and ENABLE_PRICE_ACCELERATION:
                    strat_cash = price_acceleration_cash
                    num_positions = len(price_acceleration_positions)
                    invested = value - strat_cash
                elif name == "BH 1Y Monthly" and ENABLE_STATIC_BH_1Y_MONTHLY:
                    strat_cash = static_bh_1y_monthly_cash
                    num_positions = len(static_bh_1y_monthly_positions)
                    invested = value - strat_cash
                elif name == "BH 6M Monthly" and ENABLE_STATIC_BH_6M_MONTHLY:
                    strat_cash = static_bh_6m_monthly_cash
                    num_positions = len(static_bh_6m_monthly_positions)
                    invested = value - strat_cash
                elif name == "BH 3M Monthly" and ENABLE_STATIC_BH_3M_MONTHLY:
                    strat_cash = static_bh_3m_monthly_cash
                    num_positions = len(static_bh_3m_monthly_positions)
                    invested = value - strat_cash
                elif name in ("Meta-Strategy ML", "Meta-Strategy Mom") and meta_allocator is not None:
                    strat_cash = 0
                    num_positions = meta_allocator.top_k
                    invested = value
                
                strategy_details.append((name, value, strat_cash, num_positions, invested))
            
            # Sort by value and display
            strategy_details.sort(key=lambda x: x[1], reverse=True)
            for i, (name, value, strat_cash, num_pos, invested) in enumerate(strategy_details, 1):
                # Use TOTAL_CAPITAL as the starting capital for return calculation
                # (all strategies start with the same initial investment)
                from config import TOTAL_CAPITAL
                starting_capital = TOTAL_CAPITAL
                return_pct = ((value - starting_capital) / starting_capital) * 100
                # Calculate annualized return: (1 + total_return)^(252/days) - 1
                if day_count > 0:
                    total_return_multiplier = value / starting_capital
                    annualized_return = (total_return_multiplier ** (252.0 / day_count) - 1) * 100
                else:
                    annualized_return = 0.0
                allocation_pct = (invested / value * 100) if value > 0 and invested > 0 else 0
                print(f"{i:<5} {name:<20} ${value:<11,.0f} {return_pct:+.1f}% {annualized_return:+.1f}% ${strat_cash:<11,.0f} {num_pos:<10} ({allocation_pct:.0f}%)")
            
            # Show current AI strategy if enabled
            if enable_ai_strategy and current_portfolio_stocks:
                print(f"\n🎯 AI Strategy (Day {day_count}): {', '.join(current_portfolio_stocks[:5])}")
                if len(current_portfolio_stocks) > 5:
                    print(f"   ... and {len(current_portfolio_stocks) - 5} more")
            
            # Show Top 5 Consistency Score
            if day_count >= 3 and top5_consistency_counts:
                print(f"\n🏆 TOP 5 CONSISTENCY (Days in Top 5 / {day_count} total days):")
                sorted_consistency = sorted(top5_consistency_counts.items(), key=lambda x: x[1], reverse=True)
                for rank, (strat_name, count) in enumerate(sorted_consistency[:10], 1):
                    pct = (count / day_count) * 100
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    print(f"   {rank:<3} {strat_name:<20} {count:>3}/{day_count} days ({pct:>5.1f}%) {bar}")
            
            print("=" * 80)
        else:
            print(f"\n📊 DAILY SUMMARY - Day {day_count} ({current_date.strftime('%Y-%m-%d')})")
            print("=" * 80)
            print(f"   📊 No strategies enabled for daily summary.")
            print("=" * 80)
        
        # === DAILY ALL STRATEGIES SUMMARY TABLE ===
        print("\n" + "="*120)
        print("                     📊 DAILY ALL STRATEGIES PERFORMANCE SUMMARY 📊")
        print("="*120)
        
        # Collect all strategy data for each day
        strategy_data = []

        # Build (name, history) list for the summary table
        strategy_histories = [
            ("Static BH 1Y",        static_bh_1y_portfolio_history        if ENABLE_STATIC_BH else None),
            ("Static BH 6M",        static_bh_6m_portfolio_history        if ENABLE_STATIC_BH_6M else None),
            ("Static BH 3M",        static_bh_3m_portfolio_history        if ENABLE_STATIC_BH else None),
            ("Dynamic BH 1Y",       dynamic_bh_portfolio_history          if ENABLE_DYNAMIC_BH_1Y else None),
            ("Dynamic BH 6M",       dynamic_bh_6m_portfolio_history       if ENABLE_DYNAMIC_BH_6M else None),
            ("Dynamic BH 3M",       dynamic_bh_3m_portfolio_history       if ENABLE_DYNAMIC_BH_3M else None),
            ("Dynamic BH 1M",       dynamic_bh_1m_portfolio_history       if ENABLE_DYNAMIC_BH_1M else None),
            ("Risk-Adj Mom",        risk_adj_mom_portfolio_history        if ENABLE_RISK_ADJ_MOM else None),
            ("Mean Reversion",      mean_reversion_portfolio_history      if ENABLE_MEAN_REVERSION else None),
            ("Quality+Mom",         quality_momentum_portfolio_history    if ENABLE_QUALITY_MOM else None),
            ("Mom-AI Hybrid",       momentum_ai_hybrid_portfolio_history  if ENABLE_MOMENTUM_AI_HYBRID else None),
            ("Vol-Adj Mom",         volatility_adj_mom_portfolio_history  if ENABLE_VOLATILITY_ADJ_MOM else None),
            ("1Y/3M Ratio",         ratio_1y_3m_portfolio_history         if ENABLE_3M_1Y_RATIO else None),
            ("3M/1Y Ratio",         ratio_3m_1y_portfolio_history         if ENABLE_3M_1Y_RATIO else None),
            ("Mom-Vol Hybrid",      momentum_volatility_hybrid_portfolio_history    if ENABLE_MOMENTUM_VOLATILITY_HYBRID else None),
            ("Mom-Vol Hybrid 6M",   momentum_volatility_hybrid_6m_portfolio_history if ENABLE_MOMENTUM_VOLATILITY_HYBRID_6M else None),
            ("Mom-Vol Hybrid 1Y",   momentum_volatility_hybrid_1y_portfolio_history if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y else None),
            ("Mom-Vol Hybrid 1Y/3M",momentum_volatility_hybrid_1y3m_portfolio_history if ENABLE_MOMENTUM_VOLATILITY_HYBRID_1Y3M else None),
            ("Enhanced Volatility", enhanced_volatility_portfolio_history if ENABLE_ENHANCED_VOLATILITY else None),
            ("Trend ATR",           trend_atr_portfolio_history           if ENABLE_TREND_FOLLOWING_ATR else None),
            ("Dual Momentum",       dual_mom_portfolio_history            if ENABLE_DUAL_MOMENTUM else None),
            ("Mom Acceleration",    mom_accel_portfolio_history           if ENABLE_MOMENTUM_ACCELERATION else None),
            ("Concentrated 3M",     concentrated_3m_portfolio_history     if ENABLE_CONCENTRATED_3M else None),
            ("Price Acceleration",  price_acceleration_portfolio_history  if ENABLE_PRICE_ACCELERATION else None),
            ("Elite Hybrid",        elite_hybrid_portfolio_history        if ENABLE_ELITE_HYBRID else None),
            ("Elite Risk",          elite_risk_portfolio_history          if ENABLE_ELITE_RISK else None),
            ("Risk-Adj Mom 6M",     risk_adj_mom_6m_portfolio_history     if ENABLE_RISK_ADJ_MOM_6M else None),
            ("Risk-Adj Mom 3M",     risk_adj_mom_3m_portfolio_history     if ENABLE_RISK_ADJ_MOM_3M else None),
            ("RiskAdj 3M Mth",     risk_adj_mom_3m_monthly_portfolio_history if ENABLE_RISK_ADJ_MOM_3M_MONTHLY else None),
            ("AI Elite",            ai_elite_portfolio_history            if ENABLE_AI_ELITE else None),
            ("BH 1Y Monthly",       static_bh_1y_monthly_portfolio_history if ENABLE_STATIC_BH_1Y_MONTHLY else None),
            ("BH 6M Monthly",       static_bh_6m_monthly_portfolio_history if ENABLE_STATIC_BH_6M_MONTHLY else None),
            ("BH 3M Monthly",       static_bh_3m_monthly_portfolio_history if ENABLE_STATIC_BH_3M_MONTHLY else None),
        ]

        # Filter out disabled strategies
        active_strategies = [(name, history) for name, history in strategy_histories if history is not None and len(history) > 0]

        if len(active_strategies) == 0:
            print(f"⚠️ No strategies enabled for daily summary.")
        else:
            # Create dynamic header based on number of strategies
            num_strategies = len(active_strategies)
            header_parts = [f"{'Date':<12}"]

            # Add column for each strategy
            for i, (name, _) in enumerate(active_strategies):
                short_name = name[:15] if len(name) > 15 else name
                header_parts.append(f"{short_name:<15}")

            header_parts.append(f"{'Best':<10} {'Worst':<10}")
            header_line = " ".join(header_parts)
            print(header_line)
            print("-" * len(header_line))

            # Process each day
            max_days = max(len(history) for _, history in active_strategies)
            for day_idx, current_date in enumerate(business_days[:max_days]):
                daily_values = []

                for strategy_name, history in active_strategies:
                    if day_idx < len(history):
                        value = history[day_idx]
                        if not pd.isna(value) and value > 0:
                            if day_idx == 0:
                                daily_return = 0.0
                            else:
                                prev_value = history[day_idx - 1]
                                if prev_value > 0 and not pd.isna(prev_value):
                                    daily_return = ((value - prev_value) / prev_value) * 100
                                else:
                                    daily_return = 0.0
                            daily_values.append((strategy_name, value, daily_return))

                if len(daily_values) > 0:
                    daily_values.sort(key=lambda x: x[1], reverse=True)
                    best_strategy = daily_values[0][0]
                    worst_strategy = daily_values[-1][0]

                    date_str = current_date.strftime('%Y-%m-%d')
                    line_parts = [f"{date_str:<12}"]

                    for strategy_name, history in active_strategies:
                        if day_idx < len(history):
                            value = history[day_idx]
                            if not pd.isna(value) and value > 0:
                                display_value = f"${value:>8,.0f}"
                                if strategy_name == best_strategy:
                                    display_value = "🥇" + display_value[1:]
                                elif strategy_name == worst_strategy:
                                    display_value = "🔻" + display_value[1:]
                                line_parts.append(f"{display_value:<15}")
                            else:
                                line_parts.append(f"{'N/A':<15}")
                        else:
                            line_parts.append(f"{'--':<15}")

                    best_short = best_strategy[:8] if len(best_strategy) > 8 else best_strategy
                    worst_short = worst_strategy[:8] if len(worst_strategy) > 8 else worst_strategy
                    line_parts.append(f"{best_short:<10} {worst_short:<10}")
                    print(" ".join(line_parts))

            print("-" * 120)
            print(f"\n📈 Summary: Showing all strategies with best/worst identification for each trading day")
            print(f"📊 Total strategies tracked: {len(active_strategies)}")
            print(f"📅 Trading days analyzed: {len(business_days)}")
            print(f"🥇 Best strategy each day marked with 🥇 symbol")
            print(f"🔻 Worst strategy each day marked with 🔻 symbol")

        print("="*120)

    # Build results dictionary - replaces the fragile 149-value return tuple
    # Each strategy has: value, history, costs, cash_deployed
    # Adding a new strategy = just add one dict entry here, no main.py changes needed
    def _strat(value, history, costs, cash):
        return {'value': value, 'history': history, 'costs': costs, 'cash_deployed': cash}

    results = {
        'general': {
            'final_strategy_value': total_portfolio_value,
            'portfolio_values_history': portfolio_values_history,
            'processed_tickers': initial_top_tickers,
            'performance_metrics': [],
            'buy_hold_histories': {},
            'day_count': day_count,
        },
        'strategies': {
            'ai_strategy':              _strat(total_portfolio_value, portfolio_values_history, ai_transaction_costs, initial_capital_needed),
            'static_bh_1y':             _strat(static_bh_1y_portfolio_value, static_bh_1y_portfolio_history, static_bh_transaction_costs, static_bh_1y_cash),
            'static_bh_6m':             _strat(static_bh_6m_portfolio_value, static_bh_6m_portfolio_history, static_bh_6m_transaction_costs, static_bh_6m_cash),
            'static_bh_3m':             _strat(static_bh_3m_portfolio_value, static_bh_3m_portfolio_history, static_bh_3m_transaction_costs, static_bh_3m_cash),
            'static_bh_1m':             _strat(static_bh_1m_portfolio_value, static_bh_1m_portfolio_history, static_bh_1m_transaction_costs, static_bh_1m_cash),
            'dynamic_bh_1y':            _strat(dynamic_bh_portfolio_value, dynamic_bh_portfolio_history, dynamic_bh_transaction_costs, dynamic_bh_cash),
            'dynamic_bh_6m':            _strat(dynamic_bh_6m_portfolio_value, dynamic_bh_6m_portfolio_history, dynamic_bh_6m_transaction_costs, dynamic_bh_6m_cash),
            'dynamic_bh_3m':            _strat(dynamic_bh_3m_portfolio_value, dynamic_bh_3m_portfolio_history, dynamic_bh_3m_transaction_costs, dynamic_bh_3m_cash),
            'dynamic_bh_1m':            _strat(dynamic_bh_1m_portfolio_value, dynamic_bh_1m_portfolio_history, dynamic_bh_1m_transaction_costs, dynamic_bh_1m_cash),
            'dynamic_bh_1y_vol_filter': _strat(dynamic_bh_1y_vol_filter_portfolio_value, dynamic_bh_1y_vol_filter_portfolio_history, dynamic_bh_1y_vol_filter_transaction_costs, dynamic_bh_1y_vol_filter_cash),
            'dynamic_bh_1y_trailing_stop': _strat(dynamic_bh_1y_trailing_stop_portfolio_value, dynamic_bh_1y_trailing_stop_portfolio_history, dynamic_bh_1y_trailing_stop_transaction_costs, dynamic_bh_1y_trailing_stop_cash),
            'risk_adj_mom':             _strat(risk_adj_mom_portfolio_value, risk_adj_mom_portfolio_history, risk_adj_mom_transaction_costs, risk_adj_mom_cash),
            'multitask':                _strat(multitask_portfolio_value, multitask_portfolio_history, multitask_transaction_costs, multitask_cash),
            'mean_reversion':           _strat(mean_reversion_portfolio_value, mean_reversion_portfolio_history, mean_reversion_transaction_costs, mean_reversion_cash),
            'quality_momentum':         _strat(quality_momentum_portfolio_value, quality_momentum_portfolio_history, quality_momentum_transaction_costs, quality_momentum_cash),
            'momentum_ai_hybrid':       _strat(momentum_ai_hybrid_portfolio_value, momentum_ai_hybrid_portfolio_history, momentum_ai_hybrid_transaction_costs, momentum_ai_hybrid_cash),
            'volatility_adj_mom':       _strat(volatility_adj_mom_portfolio_value, volatility_adj_mom_portfolio_history, volatility_adj_mom_transaction_costs, volatility_adj_mom_cash),
            'sector_rotation':          _strat(sector_rotation_portfolio_value, sector_rotation_portfolio_history, sector_rotation_transaction_costs, sector_rotation_cash),
            'ratio_3m_1y':              _strat(ratio_3m_1y_portfolio_value, ratio_3m_1y_portfolio_history, ratio_3m_1y_transaction_costs, ratio_3m_1y_cash),
            'ratio_1y_3m':              _strat(ratio_1y_3m_portfolio_value, ratio_1y_3m_portfolio_history, ratio_1y_3m_transaction_costs, ratio_1y_3m_cash),
            'momentum_volatility_hybrid': _strat(momentum_volatility_hybrid_portfolio_value, momentum_volatility_hybrid_portfolio_history, momentum_volatility_hybrid_transaction_costs, momentum_volatility_hybrid_cash),
            'momentum_volatility_hybrid_6m': _strat(momentum_volatility_hybrid_6m_portfolio_value, momentum_volatility_hybrid_6m_portfolio_history, momentum_volatility_hybrid_6m_transaction_costs, momentum_volatility_hybrid_6m_cash),
            'momentum_volatility_hybrid_1y': _strat(momentum_volatility_hybrid_1y_portfolio_value, momentum_volatility_hybrid_1y_portfolio_history, momentum_volatility_hybrid_1y_transaction_costs, momentum_volatility_hybrid_1y_cash),
            'momentum_volatility_hybrid_1y3m': _strat(momentum_volatility_hybrid_1y3m_portfolio_value, momentum_volatility_hybrid_1y3m_portfolio_history, momentum_volatility_hybrid_1y3m_transaction_costs, momentum_volatility_hybrid_1y3m_cash),
            'price_acceleration':       _strat(price_acceleration_portfolio_value, price_acceleration_portfolio_history, price_acceleration_transaction_costs, price_acceleration_cash),
            'turnaround':               _strat(turnaround_portfolio_value, turnaround_portfolio_history, turnaround_transaction_costs, turnaround_cash),
            'adaptive_ensemble':        _strat(adaptive_ensemble_portfolio_value, adaptive_ensemble_portfolio_history, adaptive_ensemble_transaction_costs, adaptive_ensemble_cash),
            'volatility_ensemble':      _strat(volatility_ensemble_portfolio_value, volatility_ensemble_portfolio_history, volatility_ensemble_transaction_costs, volatility_ensemble_cash),
            'enhanced_volatility':      _strat(enhanced_volatility_portfolio_value, enhanced_volatility_portfolio_history, enhanced_volatility_transaction_costs, enhanced_volatility_cash),
            'ai_volatility_ensemble':   _strat(ai_volatility_ensemble_portfolio_value, ai_volatility_ensemble_portfolio_history, ai_volatility_ensemble_transaction_costs, ai_volatility_ensemble_cash),
            'multi_tf_ensemble':        _strat(multi_tf_ensemble_portfolio_value, multi_tf_ensemble_portfolio_history, multi_tf_ensemble_transaction_costs, multi_tf_ensemble_cash),
            'correlation_ensemble':     _strat(correlation_ensemble_portfolio_value, correlation_ensemble_portfolio_history, correlation_ensemble_transaction_costs, correlation_ensemble_cash),
            'dynamic_pool':             _strat(dynamic_pool_portfolio_value, dynamic_pool_portfolio_history, dynamic_pool_transaction_costs, dynamic_pool_cash),
            'sentiment_ensemble':       _strat(sentiment_ensemble_portfolio_value, sentiment_ensemble_portfolio_history, sentiment_ensemble_transaction_costs, sentiment_ensemble_cash),
            'voting_ensemble':          _strat(voting_ensemble_portfolio_value, voting_ensemble_portfolio_history, voting_ensemble_transaction_costs, voting_ensemble_cash),
            'mom_accel':                _strat(mom_accel_portfolio_value, mom_accel_portfolio_history, mom_accel_transaction_costs, mom_accel_cash),
            'concentrated_3m':          _strat(concentrated_3m_portfolio_value, concentrated_3m_portfolio_history, concentrated_3m_transaction_costs, concentrated_3m_cash),
            'dual_momentum':            _strat(dual_mom_portfolio_value, dual_mom_portfolio_history, dual_mom_transaction_costs, dual_mom_cash),
            'trend_atr':                _strat(trend_atr_portfolio_value, trend_atr_portfolio_history, trend_atr_transaction_costs, trend_atr_cash),
            'elite_hybrid':             _strat(elite_hybrid_portfolio_value, elite_hybrid_portfolio_history, elite_hybrid_transaction_costs, elite_hybrid_cash),
            'elite_risk':               _strat(elite_risk_portfolio_value, elite_risk_portfolio_history, elite_risk_transaction_costs, elite_risk_cash),
            'risk_adj_mom_6m':          _strat(risk_adj_mom_6m_portfolio_value, risk_adj_mom_6m_portfolio_history, risk_adj_mom_6m_transaction_costs, risk_adj_mom_6m_cash),
            'risk_adj_mom_3m':          _strat(risk_adj_mom_3m_portfolio_value, risk_adj_mom_3m_portfolio_history, risk_adj_mom_3m_transaction_costs, risk_adj_mom_3m_cash),
            'risk_adj_mom_3m_monthly':  _strat(risk_adj_mom_3m_monthly_portfolio_value, risk_adj_mom_3m_monthly_portfolio_history, risk_adj_mom_3m_monthly_transaction_costs, risk_adj_mom_3m_monthly_cash),
            'ai_elite':                 _strat(ai_elite_portfolio_value, ai_elite_portfolio_history, ai_elite_transaction_costs, ai_elite_cash),
            'risk_adj_mom_sentiment':   _strat(risk_adj_mom_sentiment_portfolio_value, risk_adj_mom_sentiment_portfolio_history, risk_adj_mom_sentiment_transaction_costs, risk_adj_mom_sentiment_cash),
            'bh_1y_monthly':            _strat(static_bh_1y_monthly_portfolio_value, static_bh_1y_monthly_portfolio_history, static_bh_1y_monthly_transaction_costs, static_bh_1y_monthly_cash),
            'bh_6m_monthly':            _strat(static_bh_6m_monthly_portfolio_value, static_bh_6m_monthly_portfolio_history, static_bh_6m_monthly_transaction_costs, static_bh_6m_monthly_cash),
            'bh_3m_monthly':            _strat(static_bh_3m_monthly_portfolio_value, static_bh_3m_monthly_portfolio_history, static_bh_3m_monthly_transaction_costs, static_bh_3m_monthly_cash),
        }
    }

    return results




def _quick_predict_return(ticker: str, df_recent: pd.DataFrame, model, scaler, y_scaler, horizon_days: int) -> float:
    """Quick prediction of return for stock reselection during walk-forward backtest."""
    # Import PyTorch models if available
    if PYTORCH_AVAILABLE:
        from ml_models import TCNRegressor, GRURegressor, LSTMRegressor, LSTMClassifier, GRUClassifier
    
    try:
        # Wrap entire prediction in timeout
        with prediction_timeout(PREDICTION_TIMEOUT, ticker):
            if model is None:
                print(f"   ⚠️ {ticker}: model is None")
                return -np.inf
            if scaler is None:
                print(f"   ⚠️ {ticker}: scaler is None")
                return -np.inf
            if df_recent.empty:
                print(f"   ⚠️ {ticker}: df_recent is empty")
                return -np.inf

            # ✅ VALIDATION: Check if we have enough data for prediction
            try:
                validate_prediction_data(df_recent, ticker)
            except InsufficientDataError as e:
                print(f"   {str(e)}")
                return -np.inf

            print(f"   🔍 {ticker}: Starting prediction with {len(df_recent)} rows, model type: {type(model).__name__}")

            # Engineer features - same as training
            df_with_features = df_recent.copy()

            print(f"   🔧 {ticker}: Initial features: {list(df_with_features.columns)}")

            # Add financial features that might be in the data (fill with 0 if missing)
            financial_features = [col for col in df_with_features.columns if col.startswith('Fin_')]
            for col in financial_features:
                df_with_features[col] = pd.to_numeric(df_with_features[col], errors='coerce').fillna(0)

            df_with_features = _calculate_technical_indicators(df_with_features)
            print(f"   🔧 {ticker}: After technical indicators: {len(df_with_features)} rows, {len(df_with_features.columns)} features")

            # Add annualized BH return feature (same as in training)
            if len(df_with_features) > 1:
                start_price = df_with_features["Close"].iloc[0]
                end_price = df_with_features["Close"].iloc[-1]
                
                # Fix: Handle date calculation properly when index is named 'date'
                try:
                    total_days = (df_with_features.index[-1] - df_with_features.index[0]).days
                except Exception:
                    # Fallback: convert to datetime if needed
                    total_days = (pd.to_datetime(df_with_features.index[-1]) - pd.to_datetime(df_with_features.index[0])).days

                if total_days > 0 and start_price > 0:
                    total_return = (end_price / start_price) - 1.0
                    annualized_return = (1 + total_return) ** (365.0 / total_days) - 1
                    df_with_features["Annualized_BH_Return"] = annualized_return
                else:
                    df_with_features["Annualized_BH_Return"] = 0.0
            else:
                df_with_features["Annualized_BH_Return"] = 0.0

            # Only drop rows with NaN if we have enough rows to spare
            rows_before_dropna = len(df_with_features)
            df_with_features = df_with_features.dropna()
            rows_after_dropna = len(df_with_features)
            print(f"   🔧 {ticker}: After dropna: {rows_after_dropna} rows (dropped {rows_before_dropna - rows_after_dropna})")

            # ✅ VALIDATION: Check if enough rows remain after feature engineering
            if rows_after_dropna == 0:
                print(f"   ❌ {ticker}: All rows dropped during prediction feature engineering!")
                print(f"   💡 This usually means:")
                print(f"      - Not enough historical data for technical indicators")
                print(f"      - Too many NaN/missing values in source data")
                print(f"      - Feature calculation window is too large for available data")
                return -np.inf
            
            try:
                validate_features_after_engineering(df_with_features, ticker, min_rows=1, context="prediction")
            except InsufficientDataError as e:
                print(f"   {str(e)}")
                return -np.inf

            # Get latest data point
            latest_data = df_with_features.iloc[-1:]
            print(f"   📊 {ticker}: Latest data shape: {latest_data.shape}, features: {list(latest_data.columns)}")

            # Align features to match scaler's expectations
            scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else []
            print(f"   🔧 {ticker}: Scaler expects {len(scaler_features)} features: {scaler_features[:5]}...")
            if scaler_features:
                # Ensure we have all expected features, fill missing ones with 0
                for feature in scaler_features:
                    if feature not in latest_data.columns:
                        latest_data[feature] = 0.0
                # Reorder columns to match scaler expectations
                latest_data = latest_data[scaler_features]
                print(f"   🔧 {ticker}: After alignment: {latest_data.shape}")

            # Scale features
            try:
                # Pass DataFrame directly to preserve feature names and avoid sklearn warning
                features_scaled = scaler.transform(latest_data)
                print(f"   🔧 {ticker}: Features scaled successfully, shape: {features_scaled.shape}")
            except Exception as e:
                print(f"   ❌ {ticker}: Scaling failed: {e}")
                return -np.inf

            # Predict return
            try:
                # Check if model is a PyTorch sequence model (TCN, GRU, LSTM)
                if PYTORCH_AVAILABLE and isinstance(model, (TCNRegressor, GRURegressor, LSTMRegressor, LSTMClassifier, GRUClassifier)):
                    import torch
                    
                    # For sequence models, we need a sequence of data, not just the latest point
                    # Use the last SEQUENCE_LENGTH rows from df_with_features
                    sequence_length = SEQUENCE_LENGTH  # Default is 60
                    
                    if len(df_with_features) < sequence_length:
                        # If not enough data, pad with zeros
                        sequence_data = df_with_features[scaler_features].copy()
                        padding_needed = sequence_length - len(sequence_data)
                        padding_df = pd.DataFrame(
                            np.zeros((padding_needed, len(scaler_features))),
                            columns=scaler_features
                        )
                        sequence_data = pd.concat([padding_df, sequence_data], ignore_index=True)
                    else:
                        # Get the last sequence_length rows
                        sequence_data = df_with_features[scaler_features].tail(sequence_length)
                    
                    # Scale the entire sequence (pass DataFrame to preserve feature names)
                    sequence_scaled = scaler.transform(sequence_data)
                    print(f"   🔧 {ticker}: Sequence scaled, shape: {sequence_scaled.shape}")
                    
                    # Convert to PyTorch tensor with shape (batch_size=1, sequence_length, num_features)
                    X_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)
                    
                    # Move to appropriate device
                    from config import FORCE_CPU
                    device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
                    X_tensor = X_tensor.to(device)
                    model.to(device)
                    
                    # Make prediction
                    model.eval()
                    with torch.no_grad():
                        output_tensor = model(X_tensor)
                        # Handle different output shapes - check tensor dim before converting to numpy
                        if output_tensor.dim() > 1:
                            prediction = float(output_tensor.cpu().numpy()[0][0])
                        else:
                            prediction = float(output_tensor.cpu().numpy()[0])
                        print(f"   🤖 {ticker}: PyTorch model prediction successful: {prediction:.4f}")
                
                elif hasattr(model, 'predict'):
                    # Scikit-learn style models
                    prediction = model.predict(features_scaled)[0]
                    print(f"   🤖 {ticker}: Model.predict() successful: {float(prediction):.4f}")
                else:
                    # Fallback for other model types
                    prediction = model(features_scaled)[0]
                    print(f"   🤖 {ticker}: Model call successful: {float(prediction):.4f}")

                # ✅ FIX: Clip prediction BEFORE inverse transform to prevent extrapolation
                # For models outputting scaled values, clip to [-1, 1] range
                if y_scaler and hasattr(y_scaler, 'inverse_transform'):
                    # Clip to valid scaled range before inverse transform
                    prediction_clipped = np.clip(float(prediction), -1.0, 1.0)
                    if abs(prediction_clipped - float(prediction)) > 0.01:
                        print(f"   ⚠️ {ticker}: Clipped prediction from {float(prediction):.4f} to {prediction_clipped:.4f}")
                    prediction_pct = y_scaler.inverse_transform(np.array([[prediction_clipped]]))[0][0]
                    # ✅ Convert from percentage to decimal (y_scaler returns percentage like 50.0 for 50%)
                    prediction = prediction_pct / 100.0
                    print(f"   🔄 {ticker}: Y-scaler applied: {prediction_pct:.4f}% → {float(prediction):.4f} decimal")

                # ✅ FIX: Final validation - clip to reasonable return range (-100% to +200%)
                # No stock can lose more than 100%, and >200% returns are rare outliers
                final_prediction = np.clip(float(prediction), -1.0, 2.0)
                if abs(final_prediction - float(prediction)) > 0.01:
                    print(f"   ⚠️ {ticker}: Clipped final prediction from {float(prediction)*100:.2f}% to {final_prediction*100:.2f}%")

                print(f"   ✅ {ticker}: Final prediction = {final_prediction*100:.2f}%")
                return float(final_prediction)

            except Exception as e:
                print(f"   ❌ {ticker}: Prediction failed: {e}")
                return -np.inf

    except PredictionTimeoutError as e:
        print(f"   ⏰ {ticker}: {e}")
        return -np.inf
    except Exception as e:
        print(f"   ⚠️ Prediction failed for {ticker}: {type(e).__name__}: {str(e)[:100]}")
        return -np.inf


# -----------------------------------------------------------------------------
# Final Summary Printing
# -----------------------------------------------------------------------------
def print_final_summary(
    sorted_final_results: List[Dict],
    models: Dict,  # Changed from models_buy and models_sell
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
    top_performers_data: List[Tuple],
    final_dynamic_bh_value_1y: float = None,
    dynamic_bh_1y_return: float = None,
    final_dynamic_bh_3m_value_1y: float = None,
    dynamic_bh_3m_1y_return: float = None,
    **kwargs
) -> None:
    """Prints the final summary of the backtest results."""
    print("\n" + "="*80)
    print("                     🚀 AI-POWERED STOCK ADVISOR FINAL SUMMARY 🚀")
    print("="*80)

    print("\n📊 Overall Portfolio Performance:")
    print(f"  Initial Capital: ${initial_balance_used:,.2f}")
    print(f"  Number of Tickers Analyzed: {num_tickers_analyzed}")
    print("-" * 40)
    print(f"  1-Year AI Strategy Value: ${final_strategy_value_1y:,.2f} ({ai_1y_return:+.2f}%)")
    print(f"  1-Year Simple Rule Value: ${final_simple_rule_value_1y:,.2f} ({simple_rule_1y_return:+.2f}%)")
    print(f"  1-Year Static BH Value: ${final_buy_hold_value_1y:,.2f} ({((final_buy_hold_value_1y - initial_balance_used) / abs(initial_balance_used)) * 100 if initial_balance_used != 0 else 0.0:+.2f}%)")
    if final_dynamic_bh_value_1y is not None:
        print(f"  1-Year Dynamic BH Value: ${final_dynamic_bh_value_1y:,.2f} ({dynamic_bh_1y_return:+.2f}%)")
    if final_dynamic_bh_3m_value_1y is not None:
        print(f"  1-Year Dynamic BH 3M Value: ${final_dynamic_bh_3m_value_1y:,.2f} ({dynamic_bh_3m_1y_return:+.2f}%)")
    print("="*80)

    print("\n📈 Individual Ticker Performance (AI Strategy - Sorted by 1-Year Performance):")
    print("-" * 280)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10} | {'Class Horiz':>13} | {'Opt. Status':<25} | {'Shares Before Liquidation':>25}")
    print("-" * 280)
    for res in sorted_final_results:
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        # Probability thresholds removed
        buy_thresh = 0.0  # Disabled
        sell_thresh = 1.0  # Disabled
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)
        class_horiz = optimized_params.get('class_horizon', PERIOD_HORIZONS.get("1-Year", 10))  # 10 calendar days
        opt_status = optimized_params.get('optimization_status', 'N/A')

        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('performance', 0.0) - allocated_capital

        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"
        
        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%} | {class_horiz:>12} | {opt_status:<25} | {shares_before_liquidation_str}")
    print("-" * 290)

    print("\n📈 Individual Ticker Performance (Simple Rule Strategy - Sorted by 1-Year Performance):")
    print("-" * 126)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'Sharpe':>12} | {'Last Action':<16} | {'Shares Before Liquidation':>25}")
    print("-" * 126)
    
    sorted_simple_rule_results = sorted(performance_metrics_simple_rule_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_simple_rule_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = res.get('final_val', 0.0) - allocated_capital
        
        one_year_perf_benchmark = np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        last_action_str = str(res.get('last_ai_action', 'HOLD'))
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {sharpe_str} | {last_action_str:<16} | {shares_before_liquidation_str}")
    print("-" * 126)

    print("\n📈 Individual Ticker Performance (Buy & Hold Strategy - Sorted by 1-Year Performance):")
    print("-" * 126)
    print(f"{'Ticker':<10} | {'Allocated Capital':>18} | {'Strategy Gain':>15} | {'1Y Perf':>10} | {'Sharpe':>12} | {'Shares Before Liquidation':>25}")
    print("-" * 126)
    
    sorted_buy_hold_results = sorted(performance_metrics_buy_hold_1y, key=lambda x: x.get('individual_bh_return', -np.inf) if pd.notna(x.get('individual_bh_return')) else -np.inf, reverse=True)

    for res in sorted_buy_hold_results:
        ticker = str(res.get('ticker', 'N/A'))
        allocated_capital = INVESTMENT_PER_STOCK
        strategy_gain = (res.get('final_val', 0.0) - allocated_capital) if res.get('final_val') is not None else 0.0
        
        one_year_perf_benchmark = np.nan
        for t, p1y, pytd in top_performers_data:
            if t == ticker:
                one_year_perf_benchmark = p1y if pd.notna(p1y) else np.nan
                break

        one_year_perf_str = f"{one_year_perf_benchmark:>9.2f}%" if pd.notna(one_year_perf_benchmark) else "N/A".rjust(10)
        sharpe_str = f"{res['perf_data']['sharpe_ratio']:>11.2f}" if pd.notna(res['perf_data']['sharpe_ratio']) else "N/A".rjust(12)
        shares_before_liquidation_str = f"{res.get('shares_before_liquidation', 0.0):>24.2f}"

        print(f"{ticker:<10} | ${allocated_capital:>16,.2f} | ${strategy_gain:>13,.2f} | {one_year_perf_str} | {sharpe_str} | {shares_before_liquidation_str}")
    print("-" * 126)

    print("\n🤖 ML Model Status:")
    for ticker in sorted_final_results:
        t = ticker['ticker']
        model_status = "✅ Trained" if models.get(t) else "❌ Not Trained"
        print(f"  - {t}: TargetReturn Model: {model_status}")
    print("="*80)

    print("\n💡 Next Steps:")
    print("  - Review individual ticker performance and trade logs for deeper insights.")
    print("  - Experiment with different `MARKET_SELECTION` options and `N_TOP_TICKERS`.")
    print("  - Adjust `TARGET_PERCENTAGE` and `RISK_PER_TRADE` for different risk appetites.")
    print("  - Consider enabling `USE_MARKET_FILTER` and `USE_PERFORMANCE_BENCHMARK` for additional filtering.")
    print("  - Explore advanced ML models or feature engineering for further improvements.")
    print("="*80)


# Import shared function
from shared_strategies import calculate_volatility_adjusted_momentum


def _rebalance_volatility_adj_mom_portfolio(new_stocks, current_date, ticker_data_grouped,
                                           volatility_adj_mom_positions, volatility_adj_mom_cash, capital_per_stock):
    """
    Rebalance volatility-adjusted momentum portfolio to hold the new top N stocks.
    
    Args:
        new_stocks: List of tickers to hold
        current_date: Current date for rebalancing
        ticker_data_grouped: Dict of ticker DataFrames indexed by date
        volatility_adj_mom_positions: Current positions dictionary
        volatility_adj_mom_cash: Available cash
        capital_per_stock: Target investment per stock
    
    Returns:
        Updated cash balance after rebalancing
    """
    global volatility_adj_mom_transaction_costs
    
    # Sell stocks not in new selection
    stocks_to_sell = [ticker for ticker in volatility_adj_mom_positions if ticker not in new_stocks]
    for ticker in stocks_to_sell:
        try:
            # Get current price
            ticker_df = ticker_data_grouped.get(ticker)
            if ticker_df is None or ticker_df.empty:
                print(f"   ⚠️ No price data for {ticker} up to {current_date.date()}, skipping sell")
                continue

            price_data = ticker_df.loc[:current_date]
            if price_data.empty:
                print(f"   ⚠️ No price data for {ticker} up to {current_date.date()}, skipping sell")
                continue

            valid_prices = price_data['Close'].dropna()
            if len(valid_prices) == 0:
                print(f"   ⚠️ No valid price data for {ticker} up to {current_date.date()}, skipping sell")
                continue

            current_price = valid_prices.iloc[-1]
            shares = volatility_adj_mom_positions[ticker]['shares']
            proceeds = shares * current_price
            fee = proceeds * TRANSACTION_COST
            volatility_adj_mom_cash += proceeds - fee
            volatility_adj_mom_transaction_costs += fee
            
            del volatility_adj_mom_positions[ticker]
            
        except Exception as e:
            print(f"   ⚠️ Error selling {ticker}: {e}")
    
    # Buy new stocks
    stocks_to_buy = [t for t in new_stocks if t not in volatility_adj_mom_positions]
    if stocks_to_buy:
        # Split available cash across remaining buys, accounting for transaction costs
        target_value_per_stock = volatility_adj_mom_cash / (len(stocks_to_buy) * (1 + TRANSACTION_COST))
        
        for ticker in stocks_to_buy:
            try:
                ticker_df = ticker_data_grouped.get(ticker)
                if ticker_df is None or ticker_df.empty:
                    print(f"   ⚠️ No price data for {ticker} up to {current_date.date()}, skipping buy")
                    continue

                price_data = ticker_df.loc[:current_date]
                if price_data.empty:
                    print(f"   ⚠️ No price data for {ticker} up to {current_date.date()}, skipping buy")
                    continue

                valid_prices = price_data['Close'].dropna()
                if len(valid_prices) == 0:
                    print(f"   ⚠️ No valid price data for {ticker} up to {current_date.date()}, skipping buy")
                    continue

                current_price = valid_prices.iloc[-1]
                
                if current_price <= 0:
                    print(f"   ⚠️ Invalid price for {ticker}: ${current_price}, skipping buy")
                    continue
                
                # Calculate shares to buy (accounting for transaction cost)
                max_affordable_shares = int(target_value_per_stock / (current_price * (1 + TRANSACTION_COST)))
                
                if max_affordable_shares > 0:
                    cost = max_affordable_shares * current_price
                    fee = cost * TRANSACTION_COST
                    total_cost = cost + fee
                    
                    if total_cost <= volatility_adj_mom_cash:
                        volatility_adj_mom_cash -= total_cost
                        volatility_adj_mom_transaction_costs += fee
                        volatility_adj_mom_positions[ticker] = {
                            'shares': max_affordable_shares,
                            'entry_price': current_price,
                            'value': cost
                        }
                        
            except Exception as e:
                print(f"   ⚠️ Error buying {ticker}: {e}")
    
    return volatility_adj_mom_cash, volatility_adj_mom_positions


def _rebalance_ratio_3m_1y_portfolio(new_stocks, current_date, ticker_data_grouped,
                                     ratio_3m_1y_positions, ratio_3m_1y_cash, capital_per_stock):
    """
    Rebalance 3M/1Y ratio portfolio to hold the new top N stocks.
    Uses the same logic as other rebalance functions.
    """
    global ratio_3m_1y_transaction_costs
    
    # Sell all current positions that are not in the new stock list
    stocks_to_sell = [ticker for ticker in ratio_3m_1y_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = ratio_3m_1y_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            ratio_3m_1y_transaction_costs += sell_cost
                            ratio_3m_1y_cash += gross_sale - sell_cost
                            del ratio_3m_1y_positions[ticker]
                        else:
                            print(f"   ⚠️ 3M/1Y Ratio: Invalid price {current_price} for {ticker}, skipping sell")
                    else:
                        print(f"   ⚠️ 3M/1Y Ratio: No valid price data for {ticker}, skipping sell")
                else:
                    print(f"   ⚠️ 3M/1Y Ratio: No price data available for {ticker}, skipping sell")
            else:
                print(f"   ⚠️ 3M/1Y Ratio: {ticker} not found in ticker_data_grouped, skipping sell")
        except Exception as e:
            print(f"   ⚠️ 3M/1Y Ratio: Error selling {ticker}: {e}")
            continue
    
    # Buy new positions
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            # Calculate affordable shares
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= ratio_3m_1y_cash:
                                    ratio_3m_1y_cash -= total_cost
                                    ratio_3m_1y_transaction_costs += fee
                                    ratio_3m_1y_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
                                else:
                                    print(f"   ⚠️ 3M/1Y Ratio: Insufficient cash for {ticker}, need ${total_cost:.2f}, have ${ratio_3m_1y_cash:.2f}")
                            else:
                                print(f"   ⚠️ 3M/1Y Ratio: Cannot afford even 1 share of {ticker} at ${current_price:.2f}")
                        else:
                            print(f"   ⚠️ 3M/1Y Ratio: Invalid price {current_price} for {ticker}, skipping buy")
                    else:
                        print(f"   ⚠️ 3M/1Y Ratio: No valid price data for {ticker}, skipping buy")
                else:
                    print(f"   ⚠️ 3M/1Y Ratio: No price data available for {ticker}, skipping buy")
            except Exception as e:
                print(f"   ⚠️ 3M/1Y Ratio: Error buying {ticker}: {e}")
                continue
        else:
            print(f"   ⚠️ 3M/1Y Ratio: {ticker} not found in ticker_data_grouped, skipping buy")
    
    return ratio_3m_1y_cash, ratio_3m_1y_positions


def _rebalance_momentum_volatility_hybrid_portfolio(new_stocks, current_date, ticker_data_grouped,
                                                   momentum_volatility_hybrid_positions, momentum_volatility_hybrid_cash, capital_per_stock):
    """
    Rebalance momentum-volatility hybrid portfolio to hold the new top N stocks.
    Uses the same logic as other rebalance functions.
    """
    global momentum_volatility_hybrid_transaction_costs
    
    # Sell all current positions that are not in the new stock list
    stocks_to_sell = [ticker for ticker in momentum_volatility_hybrid_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = momentum_volatility_hybrid_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            momentum_volatility_hybrid_transaction_costs += sell_cost
                            momentum_volatility_hybrid_cash += gross_sale - sell_cost
                            del momentum_volatility_hybrid_positions[ticker]
                        else:
                            print(f"   ⚠️ Momentum-Volatility Hybrid: Invalid price {current_price} for {ticker}, skipping sell")
                    else:
                        print(f"   ⚠️ Momentum-Volatility Hybrid: No valid price data for {ticker}, skipping sell")
                else:
                    print(f"   ⚠️ Momentum-Volatility Hybrid: No price data available for {ticker}, skipping sell")
            else:
                print(f"   ⚠️ Momentum-Volatility Hybrid: {ticker} not found in ticker_data_grouped, skipping sell")
        except Exception as e:
            print(f"   ⚠️ Momentum-Volatility Hybrid: Error selling {ticker}: {e}")
            continue
    
    # Buy new positions
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            # Calculate affordable shares
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= momentum_volatility_hybrid_cash:
                                    momentum_volatility_hybrid_cash -= total_cost
                                    momentum_volatility_hybrid_transaction_costs += fee
                                    momentum_volatility_hybrid_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
                                else:
                                    print(f"   ⚠️ Momentum-Volatility Hybrid: Insufficient cash for {ticker}, need ${total_cost:.2f}, have ${momentum_volatility_hybrid_cash:.2f}")
                            else:
                                print(f"   ⚠️ Momentum-Volatility Hybrid: Cannot afford even 1 share of {ticker} at ${current_price:.2f}")
                        else:
                            print(f"   ⚠️ Momentum-Volatility Hybrid: Invalid price {current_price} for {ticker}, skipping buy")
                    else:
                        print(f"   ⚠️ Momentum-Volatility Hybrid: No valid price data for {ticker}, skipping buy")
                else:
                    print(f"   ⚠️ Momentum-Volatility Hybrid: No price data available for {ticker}, skipping buy")
            except Exception as e:
                print(f"   ⚠️ Momentum-Volatility Hybrid: Error buying {ticker}: {e}")
                continue
        else:
            print(f"   ⚠️ Momentum-Volatility Hybrid: {ticker} not found in ticker_data_grouped, skipping buy")
    
    return momentum_volatility_hybrid_cash, momentum_volatility_hybrid_positions


def _rebalance_ratio_1y_3m_portfolio(new_stocks, current_date, ticker_data_grouped,
                                     ratio_1y_3m_positions, ratio_1y_3m_cash, capital_per_stock):
    """
    Rebalance 1Y/3M ratio portfolio to hold the new top N stocks.
    Uses the same logic as other rebalance functions.
    """
    global ratio_1y_3m_transaction_costs
    
    # Sell all current positions that are not in the new stock list
    stocks_to_sell = [ticker for ticker in ratio_1y_3m_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = ratio_1y_3m_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            ratio_1y_3m_transaction_costs += sell_cost
                            ratio_1y_3m_cash += gross_sale - sell_cost
                            del ratio_1y_3m_positions[ticker]
                        else:
                            print(f"   ⚠️ 1Y/3M Ratio: Invalid price {current_price} for {ticker}, skipping sell")
                    else:
                        print(f"   ⚠️ 1Y/3M Ratio: No valid price data for {ticker}, skipping sell")
                else:
                    print(f"   ⚠️ 1Y/3M Ratio: No price data available for {ticker}, skipping sell")
            else:
                print(f"   ⚠️ 1Y/3M Ratio: {ticker} not found in ticker_data_grouped, skipping sell")
        except Exception as e:
            print(f"   ⚠️ 1Y/3M Ratio: Error selling {ticker}: {e}")
            continue
    
    # Buy new positions
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            # Calculate affordable shares
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= ratio_1y_3m_cash:
                                    ratio_1y_3m_cash -= total_cost
                                    ratio_1y_3m_transaction_costs += fee
                                    ratio_1y_3m_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
                                else:
                                    print(f"   ⚠️ 1Y/3M Ratio: Insufficient cash for {ticker}, need ${total_cost:.2f}, have ${ratio_1y_3m_cash:.2f}")
                            else:
                                print(f"   ⚠️ 1Y/3M Ratio: Cannot afford even 1 share of {ticker} at ${current_price:.2f}")
                        else:
                            print(f"   ⚠️ 1Y/3M Ratio: Invalid price {current_price} for {ticker}, skipping buy")
                    else:
                        print(f"   ⚠️ 1Y/3M Ratio: No valid price data for {ticker}, skipping buy")
                else:
                    print(f"   ⚠️ 1Y/3M Ratio: No price data available for {ticker}, skipping buy")
            except Exception as e:
                print(f"   ⚠️ 1Y/3M Ratio rebalancing failed: {e}")
        
    return ratio_1y_3m_cash, ratio_1y_3m_positions


def _rebalance_turnaround_portfolio(new_stocks, current_date, ticker_data_grouped,
                                  turnaround_positions, turnaround_cash, capital_per_stock):
    """
    Rebalance turnaround portfolio to hold the new top N stocks.
    Uses the same logic as other rebalance functions.
    """
    global turnaround_transaction_costs
    
    # Sell all current positions that are not in the new stock list
    stocks_to_sell = [ticker for ticker in turnaround_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = turnaround_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            turnaround_transaction_costs += sell_cost
                            turnaround_cash += gross_sale - sell_cost
                            del turnaround_positions[ticker]
                        else:
                            print(f"   ⚠️ Turnaround: Invalid price {current_price} for {ticker}, skipping sell")
                    else:
                        print(f"   ⚠️ Turnaround: No valid price data for {ticker}, skipping sell")
                else:
                    print(f"   ⚠️ Turnaround: No price data available for {ticker}, skipping sell")
            else:
                print(f"   ⚠️ Turnaround: {ticker} not found in ticker_data_grouped, skipping sell")
        except Exception as e:
            print(f"   ⚠️ Turnaround: Error selling {ticker}: {e}")
            continue
    
    # Buy new positions - 🔧 FIX: Only buy what we can afford
    affordable_stocks = []
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            # Calculate affordable shares
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= turnaround_cash:
                                    affordable_stocks.append((ticker, current_price, max_affordable_shares, cost, fee, total_cost))
                                else:
                                    print(f"   ⚠️ Turnaround: Insufficient cash for {ticker}, need ${total_cost:.2f}, have ${turnaround_cash:.2f}")
                            else:
                                print(f"   ⚠️ Turnaround: Cannot afford even 1 share of {ticker} at ${current_price:.2f}")
                        else:
                            print(f"   ⚠️ Turnaround: Invalid price {current_price} for {ticker}, skipping buy")
                    else:
                        print(f"   ⚠️ Turnaround: No valid price data for {ticker}, skipping buy")
                else:
                    print(f"   ⚠️ Turnaround: No price data available for {ticker}, skipping buy")
            except Exception as e:
                print(f"   ⚠️ Turnaround: Error buying {ticker}: {e}")
                continue
        else:
            print(f"   ⚠️ Turnaround: {ticker} not found in ticker_data_grouped, skipping buy")
    
    # 🔧 FIX: Sort by affordability and buy only what we can
    affordable_stocks.sort(key=lambda x: x[5])  # Sort by total_cost (lowest first)
    
    bought_count = 0
    for ticker, current_price, max_affordable_shares, cost, fee, total_cost in affordable_stocks:
        if total_cost <= turnaround_cash:
            turnaround_cash -= total_cost
            turnaround_transaction_costs += fee
            turnaround_positions[ticker] = {
                'shares': max_affordable_shares,
                'entry_price': current_price,
                'value': cost
            }
            bought_count += 1
            print(f"   🛒 Turnaround bought {ticker}: {max_affordable_shares} shares @ ${current_price:.2f} = ${cost:.0f} (+${fee:.0f} cost)")
        else:
            break  # Can't afford more
    
    # 🔧 FALLBACK: If couldn't buy any stocks, try buying 1 share of the cheapest stock
    if bought_count == 0 and affordable_stocks:
        cheapest_stock = affordable_stocks[0]  # Already sorted by cost
        ticker, current_price, _, _, fee, total_cost = cheapest_stock
        
        # Try to buy just 1 share if we can afford it
        if total_cost <= turnaround_cash:
            cost = current_price  # Just 1 share
            fee = cost * TRANSACTION_COST
            total_cost = cost + fee
            
            if total_cost <= turnaround_cash:
                turnaround_cash -= total_cost
                turnaround_transaction_costs += fee
                turnaround_positions[ticker] = {
                    'shares': 1,
                    'entry_price': current_price,
                    'value': cost
                }
                bought_count += 1
                print(f"   🛒 Turnaround fallback: bought 1 share of {ticker} @ ${current_price:.2f} = ${cost:.0f} (+${fee:.0f} cost)")
    
    if bought_count > 0:
        print(f"   📊 Turnaround bought {bought_count} of {len(new_stocks)} selected stocks")
    else:
        print(f"   ❌ Turnaround: Could not afford any of the {len(new_stocks)} selected stocks")
    
    return turnaround_cash, turnaround_positions


def _rebalance_adaptive_ensemble_portfolio(new_stocks, current_date, ticker_data_grouped,
                                          adaptive_ensemble_positions, adaptive_ensemble_cash, capital_per_stock):
    """
    Rebalance adaptive ensemble portfolio to hold the new top N stocks.
    Uses the same logic as other rebalance functions.
    """
    global adaptive_ensemble_transaction_costs
    
    # Sell all current positions that are not in the new stock list
    stocks_to_sell = [ticker for ticker in adaptive_ensemble_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = adaptive_ensemble_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            adaptive_ensemble_transaction_costs += sell_cost
                            adaptive_ensemble_cash += gross_sale - sell_cost
                            del adaptive_ensemble_positions[ticker]
                        else:
                            print(f"   ⚠️ Adaptive Ensemble: Invalid price {current_price} for {ticker}, skipping sell")
                    else:
                        print(f"   ⚠️ Adaptive Ensemble: No valid price data for {ticker}, skipping sell")
                else:
                    print(f"   ⚠️ Adaptive Ensemble: No price data available for {ticker}, skipping sell")
            else:
                print(f"   ⚠️ Adaptive Ensemble: {ticker} not found in ticker_data_grouped, skipping sell")
        except Exception as e:
            print(f"   ⚠️ Adaptive Ensemble: Error selling {ticker}: {e}")
            continue
    
    # Buy new positions
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            # Calculate affordable shares
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= adaptive_ensemble_cash:
                                    adaptive_ensemble_cash -= total_cost
                                    adaptive_ensemble_transaction_costs += fee
                                    adaptive_ensemble_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
                                else:
                                    print(f"   ⚠️ Adaptive Ensemble: Insufficient cash for {ticker}, need ${total_cost:.2f}, have ${adaptive_ensemble_cash:.2f}")
                            else:
                                print(f"   ⚠️ Adaptive Ensemble: Cannot afford even 1 share of {ticker} at ${current_price:.2f}")
                        else:
                            print(f"   ⚠️ Adaptive Ensemble: Invalid price {current_price} for {ticker}, skipping buy")
                    else:
                        print(f"   ⚠️ Adaptive Ensemble: No valid price data for {ticker}, skipping buy")
                else:
                    print(f"   ⚠️ Adaptive Ensemble: No price data available for {ticker}, skipping buy")
            except Exception as e:
                print(f"   ⚠️ Adaptive Ensemble: Error buying {ticker}: {e}")
                continue
        else:
            print(f"   ⚠️ Adaptive Ensemble: {ticker} not found in ticker_data_grouped, skipping buy")
    
    return adaptive_ensemble_cash, adaptive_ensemble_positions


def _rebalance_volatility_ensemble_portfolio(new_stocks, current_date, ticker_data_grouped,
                                             volatility_ensemble_positions, volatility_ensemble_cash, capital_per_stock):
    """Rebalance volatility ensemble portfolio."""
    global volatility_ensemble_transaction_costs
    
    stocks_to_sell = [ticker for ticker in volatility_ensemble_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = volatility_ensemble_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            volatility_ensemble_transaction_costs += sell_cost
                            volatility_ensemble_cash += gross_sale - sell_cost
                            del volatility_ensemble_positions[ticker]
        except Exception:
            continue
    
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= volatility_ensemble_cash:
                                    volatility_ensemble_cash -= total_cost
                                    volatility_ensemble_transaction_costs += fee
                                    volatility_ensemble_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
            except Exception:
                continue
    
    return volatility_ensemble_cash, volatility_ensemble_positions


def _rebalance_enhanced_volatility_portfolio(new_stocks, current_date, ticker_data_grouped,
                                           enhanced_volatility_positions, enhanced_volatility_cash, capital_per_stock):
    """Rebalance enhanced volatility trader portfolio with ATR-based stops."""
    global enhanced_volatility_transaction_costs
    
    stocks_to_sell = [ticker for ticker in enhanced_volatility_positions if ticker not in new_stocks]
    
    # Sell positions not in new selection
    for ticker in stocks_to_sell:
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is not None:
                current_price = _last_valid_close_up_to(ticker_data, current_date)
                if current_price is not None and current_price > 0 and ticker in enhanced_volatility_positions:
                    shares = enhanced_volatility_positions[ticker]['shares']
                    gross_sale = shares * current_price
                    sell_cost = gross_sale * TRANSACTION_COST
                    enhanced_volatility_transaction_costs += sell_cost
                    enhanced_volatility_cash += gross_sale - sell_cost
                    del enhanced_volatility_positions[ticker]
        except Exception:
            continue
    
    # Buy new positions with ATR-based stops
    for ticker in new_stocks:
        if ticker not in enhanced_volatility_positions:
            try:
                ticker_data = ticker_data_grouped.get(ticker)
                if ticker_data is not None:
                    current_price = _last_valid_close_up_to(ticker_data, current_date)
                    if current_price is None:
                        continue
                    
                    try:
                        current_price = float(current_price)
                    except Exception:
                        continue

                    if current_price > 0:
                        max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                        if max_affordable_shares > 0:
                            cost = max_affordable_shares * current_price
                            fee = cost * TRANSACTION_COST
                            total_cost = cost + fee

                            if total_cost <= enhanced_volatility_cash:
                                # Calculate ATR/stops BEFORE committing cash.
                                from enhanced_volatility_trader import EnhancedVolatilityTrader
                                trader = EnhancedVolatilityTrader()
                                atr = trader.calculate_atr(ticker, ticker_data_grouped, current_date)
                                try:
                                    atr = float(atr)
                                except Exception:
                                    atr = 0.0

                                # Safeguard: ensure ATR is reasonable (at least 1% of price)
                                min_atr = current_price * 0.01  # 1% minimum ATR
                                atr = max(atr, min_atr)

                                stop_loss = current_price - (atr * 2.0)  # 2x ATR stop loss
                                take_profit = current_price + min(atr * 3.0, current_price * 0.15)  # 3x ATR or 15% cap

                                # Commit trade
                                enhanced_volatility_cash -= total_cost
                                enhanced_volatility_transaction_costs += fee

                                enhanced_volatility_positions[ticker] = {
                                    'shares': max_affordable_shares,
                                    'entry_price': current_price,
                                    'value': cost,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit
                                }
            except Exception:
                continue
    
    return enhanced_volatility_cash, enhanced_volatility_positions


def _check_enhanced_volatility_stops(current_date, ticker_data_grouped, enhanced_volatility_positions, 
                                    enhanced_volatility_cash):
    """Check and execute stop-loss and take-profit orders for enhanced volatility positions."""
    global enhanced_volatility_transaction_costs
    
    positions_to_close = []
    
    for ticker, position in list(enhanced_volatility_positions.items()):
        try:
            ticker_data = ticker_data_grouped.get(ticker)
            if ticker_data is not None:
                current_price = _last_valid_close_up_to(ticker_data, current_date)
                if current_price is not None:
                    stop_loss = position['stop_loss']
                    take_profit = position['take_profit']
                    
                    # Check stop loss
                    if current_price <= stop_loss:
                        positions_to_close.append((ticker, current_price, 'STOP_LOSS'))
                    # Check take profit
                    elif current_price >= take_profit:
                        positions_to_close.append((ticker, current_price, 'TAKE_PROFIT'))
        except Exception:
            continue
    
    # Execute stops and take profits
    for ticker, exit_price, reason in positions_to_close:
        if ticker in enhanced_volatility_positions:
            try:
                shares = enhanced_volatility_positions[ticker]['shares']
                gross_sale = shares * exit_price
                sell_cost = gross_sale * TRANSACTION_COST
                enhanced_volatility_transaction_costs += sell_cost
                enhanced_volatility_cash += gross_sale - sell_cost
                del enhanced_volatility_positions[ticker]
                
                # Optional: Print stop/take notifications (commented out for cleaner output)
                # print(f"   {reason}: {ticker} at ${exit_price:.2f}")
                
            except Exception:
                continue
    
    return enhanced_volatility_cash, enhanced_volatility_positions


def _rebalance_multi_tf_ensemble_portfolio(
    new_stocks: List[str],
    current_stocks: List[str],
    positions: Dict,
    capital_per_stock: float,
    transaction_costs: float,
    cash: float,
    ticker_data_grouped: Dict[str, pd.DataFrame],
    current_date: datetime
) -> Tuple[float, float]:
    """
    Rebalance multi-timeframe ensemble portfolio
    
    Returns:
        Tuple of (updated_transaction_costs, updated_cash)
    """
    # Sell stocks not in new selection
    for ticker in current_stocks:
        if ticker not in new_stocks and ticker in positions:
            ticker_data = ticker_data_grouped[ticker]
            current_price = ticker_data.loc[current_date, 'Close'] if current_date in ticker_data.index else None
            
            if current_price and current_price > 0:
                shares = positions[ticker]['shares']
                sale_value = shares * current_price
                
                # Apply transaction cost
                sell_cost = sale_value * TRANSACTION_COST
                net_sale_value = sale_value - sell_cost
                transaction_costs += sell_cost
                
                # Add to cash
                cash += net_sale_value
                
                # Remove position
                del positions[ticker]
    
    # Buy new stocks
    for ticker in new_stocks:
        if ticker not in positions:
            ticker_data = ticker_data_grouped[ticker]
            current_price = ticker_data.loc[current_date, 'Close'] if current_date in ticker_data.index else None
            
            if current_price and current_price > 0 and cash >= capital_per_stock:
                shares_to_buy = capital_per_stock / current_price
                buy_value = shares_to_buy * current_price
                
                # Apply transaction cost
                buy_cost = buy_value * TRANSACTION_COST
                total_buy_cost = buy_value + buy_cost
                
                if cash >= total_buy_cost:
                    transaction_costs += buy_cost
                    cash -= total_buy_cost
                    
                    # Update position
                    positions[ticker] = {
                        'shares': shares_to_buy,
                        'entry_price': current_price,
                        'value': buy_value
                    }
    
    return transaction_costs, cash


def _rebalance_ai_volatility_ensemble_portfolio(new_stocks, current_date, ticker_data_grouped,
                                                ai_volatility_ensemble_positions, ai_volatility_ensemble_cash, capital_per_stock):
    """Rebalance AI volatility ensemble portfolio."""
    global ai_volatility_ensemble_transaction_costs
    
    stocks_to_sell = [ticker for ticker in ai_volatility_ensemble_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = ai_volatility_ensemble_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            ai_volatility_ensemble_transaction_costs += sell_cost
                            ai_volatility_ensemble_cash += gross_sale - sell_cost
                            del ai_volatility_ensemble_positions[ticker]
        except Exception:
            continue
    
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= ai_volatility_ensemble_cash:
                                    ai_volatility_ensemble_cash -= total_cost
                                    ai_volatility_ensemble_transaction_costs += fee
                                    ai_volatility_ensemble_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
            except Exception:
                continue
    
    return ai_volatility_ensemble_cash, ai_volatility_ensemble_positions


def _rebalance_correlation_ensemble_portfolio(new_stocks, current_date, ticker_data_grouped,
                                             correlation_ensemble_positions, correlation_ensemble_cash, capital_per_stock):
    """Rebalance correlation ensemble portfolio."""
    global correlation_ensemble_transaction_costs
    
    stocks_to_sell = [ticker for ticker in correlation_ensemble_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = correlation_ensemble_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            correlation_ensemble_transaction_costs += sell_cost
                            correlation_ensemble_cash += gross_sale - sell_cost
                            del correlation_ensemble_positions[ticker]
        except Exception:
            continue
    
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= correlation_ensemble_cash:
                                    correlation_ensemble_cash -= total_cost
                                    correlation_ensemble_transaction_costs += fee
                                    correlation_ensemble_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
            except Exception:
                continue
    
    return correlation_ensemble_cash, correlation_ensemble_positions


def _rebalance_dynamic_pool_portfolio(new_stocks, current_date, ticker_data_grouped,
                                     dynamic_pool_positions, dynamic_pool_cash, capital_per_stock):
    """Rebalance dynamic pool portfolio."""
    global dynamic_pool_transaction_costs
    
    stocks_to_sell = [ticker for ticker in dynamic_pool_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = dynamic_pool_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            dynamic_pool_transaction_costs += sell_cost
                            dynamic_pool_cash += gross_sale - sell_cost
                            del dynamic_pool_positions[ticker]
        except Exception:
            continue
    
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= dynamic_pool_cash:
                                    dynamic_pool_cash -= total_cost
                                    dynamic_pool_transaction_costs += fee
                                    dynamic_pool_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
            except Exception:
                continue
    
    return dynamic_pool_cash, dynamic_pool_positions


def _rebalance_sentiment_ensemble_portfolio(new_stocks, current_date, ticker_data_grouped,
                                            sentiment_ensemble_positions, sentiment_ensemble_cash, capital_per_stock):
    """Rebalance sentiment ensemble portfolio."""
    global sentiment_ensemble_transaction_costs
    
    stocks_to_sell = [ticker for ticker in sentiment_ensemble_positions if ticker not in new_stocks]
    
    for ticker in stocks_to_sell:
        try:
            if ticker in ticker_data_grouped:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            shares = sentiment_ensemble_positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            sentiment_ensemble_transaction_costs += sell_cost
                            sentiment_ensemble_cash += gross_sale - sell_cost
                            del sentiment_ensemble_positions[ticker]
        except Exception:
            continue
    
    for ticker in new_stocks:
        if ticker in ticker_data_grouped:
            try:
                ticker_data = ticker_data_grouped[ticker]
                price_data = ticker_data.loc[:current_date]
                if not price_data.empty:
                    valid_prices = price_data['Close'].dropna()
                    if not valid_prices.empty:
                        current_price = valid_prices.iloc[-1]
                        if current_price > 0:
                            max_affordable_shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                            if max_affordable_shares > 0:
                                cost = max_affordable_shares * current_price
                                fee = cost * TRANSACTION_COST
                                total_cost = cost + fee
                                
                                if total_cost <= sentiment_ensemble_cash:
                                    sentiment_ensemble_cash -= total_cost
                                    sentiment_ensemble_transaction_costs += fee
                                    sentiment_ensemble_positions[ticker] = {
                                        'shares': max_affordable_shares,
                                        'entry_price': current_price,
                                        'value': cost
                                    }
            except Exception:
                continue
    
    return sentiment_ensemble_cash, sentiment_ensemble_positions


# Missing rebalancing functions - simple implementations
def _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock, strategy_name=None):
    """Generic rebalancing function for strategies without specific implementations."""
    try:
        # Sell existing positions not in new_stocks
        for ticker in list(positions.keys()):
            if ticker not in new_stocks:
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # 🔧 FIX: Get price up to current_date, not last price in dataset
                        # ticker_data already has date as index, use loc for filtering
                        ticker_data = ticker_data.loc[ticker_data.index <= current_date]
                        if ticker_data is not None and not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            shares = positions[ticker]['shares']
                            gross_sale = shares * current_price
                            sell_cost = gross_sale * TRANSACTION_COST
                            cash += gross_sale - sell_cost
                            del positions[ticker]
                except Exception:
                    continue
        
        # Buy new positions
        for ticker in new_stocks:
            if ticker not in positions and capital_per_stock > 0:
                try:
                    ticker_data = ticker_data_grouped.get(ticker)
                    if ticker_data is not None and not ticker_data.empty:
                        # 🔧 FIX: Get price up to current_date, not last price in dataset
                        # ticker_data already has date as index, use loc for filtering
                        ticker_data = ticker_data.loc[ticker_data.index <= current_date]
                        if ticker_data is not None and not ticker_data.empty:
                            current_price = ticker_data['Close'].iloc[-1]
                            if current_price > 0:
                                shares = int(capital_per_stock / (current_price * (1 + TRANSACTION_COST)))
                                if shares > 0:
                                    buy_value = shares * current_price
                                    buy_cost = buy_value * TRANSACTION_COST
                                    total_cost = buy_value + buy_cost
                                    # Check if we have enough cash before buying
                                    if total_cost <= cash:
                                        cash -= total_cost
                                        positions[ticker] = {
                                            'shares': shares,
                                            'entry_price': current_price,
                                            'value': buy_value
                                        }
                                    else:
                                        print(f"   ⚠️ Insufficient cash for {ticker}: need ${total_cost:.2f}, have ${cash:.2f}")
                except Exception:
                    continue
    except Exception as e:
        print(f"   ⚠️ Generic rebalancing error: {e}")
    
    return cash, positions


def _rebalance_sector_rotation_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock):
    """Sector rotation rebalancing function."""
    return _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock)


def _rebalance_multitask_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock):
    """Multi-task learning rebalancing function."""
    return _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock)


def _rebalance_mean_reversion_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock):
    """Mean reversion rebalancing function."""
    return _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock)


def _rebalance_quality_momentum_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock):
    """Quality+Momentum rebalancing function."""
    return _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock)




def _rebalance_risk_adj_mom_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock):
    """Risk-Adjusted Momentum rebalancing function."""
    return _rebalance_generic_portfolio(new_stocks, current_date, ticker_data_grouped, positions, cash, capital_per_stock)


# Module for backtesting functions - not meant to be run directly
