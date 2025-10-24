# -*- coding: utf-8 -*-
"""
Live Trading component for the AI Stock Advisor
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from multiprocessing import Pool, cpu_count
from main import train_worker, TARGET_PERCENTAGE, TRAIN_LOOKBACK_DAYS, get_all_tickers, GRUClassifier, LSTMClassifier, SEQUENCE_LENGTH, PYTORCH_AVAILABLE, CUDA_AVAILABLE # Import GRUClassifier, LSTMClassifier, SEQUENCE_LENGTH, PYTORCH_AVAILABLE, CUDA_AVAILABLE
import joblib
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import time # Import the time module
import yfinance as yf # Import yfinance
import torch # Import torch for tensor operations

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")

# --- Configuration for Model Retraining ---
MODEL_RETRAIN_DAYS = 90 # Retrain models if they are older than this many days

# --- Live Trading Configuration ---
LIVE_TRADING_ENABLED = True # Set to True to enable actual trade execution on Alpaca
INVESTMENT_PER_STOCK_LIVE = 10000.0 # Fixed amount to invest per stock in live trading

def _get_alpaca_buying_power(trading_client: TradingClient) -> Optional[float]:
    """Fetches the current Reg T buying power from the Alpaca trading account."""
    try:
        account = trading_client.get_account()
        if account and account.regt_buying_power:
            buying_power = float(account.regt_buying_power)
            print(f"✅ Fetched Alpaca Reg T Buying Power: ${buying_power:,.2f}")
            return buying_power
        else:
            print("⚠️ Could not retrieve Alpaca Reg T buying power.")
            return None
    except Exception as e:
        print(f"  ⚠️ Error fetching Alpaca account balance: {e}")
        return None

def _place_alpaca_order(trading_client: TradingClient, ticker: str, qty: int, side: OrderSide) -> bool:
    """
    Places a market order on Alpaca.
    Returns True if the order was successfully placed, False otherwise.
    """
    if not LIVE_TRADING_ENABLED:
        print(f"  ℹ️ Live trading is disabled. Skipping order placement for {side.value} {qty} shares of {ticker}.")
        return False

    if qty <= 0:
        print(f"  ⚠️ Cannot place order for {side.value} {qty} shares of {ticker}. Quantity must be positive.")
        return False

    try:
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(market_order_data)
        print(f"  ✅ Successfully placed {side.value} order for {qty} shares of {ticker}. Order ID: {order.id}")
        return True
    except APIError as e:
        print(f"  ❌ Alpaca API Error placing {side.value} order for {ticker}: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error placing {side.value} order for {ticker}: {e}")
        return False

def verify_train_and_recommend_worker(args: Tuple) -> Dict:
    """
    Worker function that handles the entire pipeline for a single ticker and
    always returns a status dictionary.
    """
    ticker, models_dir, optimized_params = args
    
    # 1. Check model and train if needed (this will also serve as data verification)
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    current_time = datetime.now(timezone.utc)
    retrain_threshold = timedelta(days=MODEL_RETRAIN_DAYS)
    
    needs_training = False
    if not model_buy_path.exists():
        needs_training = True
    else:
        model_mod_time = datetime.fromtimestamp(model_buy_path.stat().st_mtime, timezone.utc)
        if (current_time - model_mod_time) > retrain_threshold:
            needs_training = True

    if needs_training:
        print(f"DEBUG: {ticker}: Model is missing or outdated. Skipping ticker.")
        return {'status': 'failure', 'ticker': ticker, 'reason': 'Model missing or outdated'}
    else:
        print(f"DEBUG: {ticker}: Model already exists and is up-to-date.")

    # 2. Generate recommendation
    try:
        print(f"DEBUG: {ticker}: Generating recommendation.")
        recommendation = process_ticker(ticker, models_dir, optimized_params)
        if recommendation:
            print(f"DEBUG: {ticker}: Recommendation generated: {recommendation['action']}.")
            return {'status': 'success', 'data': recommendation}
        else:
            # The process_ticker function now handles missing real-time price internally
            # and returns None if it fails to get a price.
            print(f"DEBUG: {ticker}: Recommendation generation failed.")
            return {'status': 'failure', 'ticker': ticker, 'reason': 'Recommendation generation failed'}
    except Exception as e:
        print(f"DEBUG: {ticker}: Exception during recommendation generation: {e}")
        return {'status': 'failure', 'ticker': ticker, 'reason': f'Exception in recommendation: {e}'}

def _get_alpaca_portfolio_positions(trading_client: TradingClient) -> Tuple[List[Dict], float]:
    """
    Fetches current portfolio positions from Alpaca.
    Returns a tuple containing a list of position dictionaries and the total market value.
    """
    if not trading_client:
        return [], 0.0
    
    total_market_value = 0.0
    formatted_positions = []
    
    try:
        positions = trading_client.get_all_positions()
        if positions:
            for pos in positions:
                market_value = float(pos.market_value)
                formatted_positions.append({
                    'ticker': pos.symbol,
                    'qty': float(pos.qty),
                    'current_price': float(pos.current_price),
                    'market_value': market_value
                })
                total_market_value += market_value
            print(f"✅ Fetched {len(positions)} Alpaca portfolio positions. Total market value: ${total_market_value:,.2f}")
        else:
            print("ℹ️ No open positions found in Alpaca portfolio.")
            
        return formatted_positions, total_market_value
    except Exception as e:
        print(f"  ⚠️ Error fetching Alpaca portfolio positions: {e}")
        return [], 0.0

def _get_latest_price_from_yahoo(ticker: str) -> Optional[float]:
    """
    Fetches the latest real-time price for a ticker from Yahoo Finance.
    This function is called by parallel workers for recommendations, so no internal delay/retry is needed here.
    """
    try:
        stock = yf.Ticker(ticker)
        latest_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        
        if latest_price is not None:
            print(f"✅ Fetched latest price for {ticker} from Yahoo Finance: ${latest_price:.2f}")
            return float(latest_price)
        else:
            print(f"⚠️ Could not retrieve latest price for {ticker} from Yahoo Finance.")
            return None
    except Exception as e:
        print(f"  ⚠️ Error fetching latest price for {ticker} from Yahoo Finance: {e}")
        return None

def get_recommendation(ticker: str, model_buy, model_sell, scaler, data: pd.DataFrame, buy_thresh: float, sell_thresh: float) -> Tuple[str, float, float]:
    """Generates a buy, sell, or hold recommendation and the buy/sell probabilities."""
    if data.empty:
        return "HOLD", 0.0, 0.0

    latest_data = data.iloc[-1:].copy()
    required_features = scaler.feature_names_in_
    for feature in required_features:
        if feature not in latest_data.columns:
            latest_data[feature] = 0

    latest_data = latest_data[required_features]
    scaled_data_np = scaler.transform(latest_data)
    
    buy_prob = 0.0
    sell_prob = 0.0

    # Handle PyTorch models (LSTM/GRU) separately
    if PYTORCH_AVAILABLE and isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
        # For PyTorch models, we need to create a sequence from the scaled data
        # The `scaled_data_np` is a 2D array (1, num_features) representing the latest day.
        # We need to create a sequence of `SEQUENCE_LENGTH` days.
        # For live trading, we'll assume `data` contains enough historical context.
        
        # Get the last SEQUENCE_LENGTH rows from the original `data` (before `latest_data` slicing)
        # and then scale them.
        if len(data) < SEQUENCE_LENGTH:
            # Not enough historical data to form a sequence, return neutral probabilities
            return "HOLD", 0.5, 0.5

        # Get the last SEQUENCE_LENGTH rows for sequencing
        historical_data_for_seq = data.iloc[-SEQUENCE_LENGTH:][required_features].copy()
        
        # Ensure all columns are numeric and fill any NaNs
        for col in historical_data_for_seq.columns:
            historical_data_for_seq[col] = pd.to_numeric(historical_data_for_seq[col], errors='coerce').fillna(0.0)

        # Scale the sequence data using the same scaler that was used for training
        X_scaled_seq = scaler.transform(historical_data_for_seq)
        
        # Convert to tensor and add batch dimension (batch_size=1)
        X_tensor = torch.tensor(X_scaled_seq, dtype=torch.float32).unsqueeze(0)
        
        # Move to appropriate device
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        X_tensor = X_tensor.to(device)

        model_buy.eval() # Set model to evaluation mode
        model_sell.eval() # Set model to evaluation mode
        with torch.no_grad():
            buy_output = model_buy(X_tensor)
            sell_output = model_sell(X_tensor)
            buy_prob = float(buy_output.cpu().numpy()[0][0])
            sell_prob = float(sell_output.cpu().numpy()[0][0])
    else:
        # For traditional ML models, use the existing scaling and prediction logic
        scaled_data = pd.DataFrame(scaled_data_np, columns=required_features)
        buy_prob = model_buy.predict_proba(scaled_data)[0][1]
        sell_prob = model_sell.predict_proba(scaled_data)[0][1]

    if buy_prob > buy_thresh:
        return "BUY", buy_prob, sell_prob
    elif sell_prob > sell_thresh:
        return "SELL", buy_prob, sell_prob
    else:
        return "HOLD", buy_prob, sell_prob

def process_ticker(ticker: str, models_dir: Path, optimized_params: Dict) -> Optional[Dict]:
    """Worker function to process a single ticker and generate a recommendation."""
    try:
        model_buy = joblib.load(models_dir / f"{ticker}_model_buy.joblib")
        model_sell = joblib.load(models_dir / f"{ticker}_model_sell.joblib")
        scaler = joblib.load(models_dir / f"{ticker}_scaler.joblib")

        ticker_params = optimized_params.get(ticker, {})
        buy_thresh = ticker_params.get('min_proba_buy', 0.5)
        sell_thresh = ticker_params.get('min_proba_sell', 0.5)

        end_date = datetime.now(timezone.utc)
        # Fetch historical data for feature calculation (e.g., SMAs, Volatility)
        # We need data up to the day before 'today' for consistent feature calculation
        hist_end_date = end_date - timedelta(days=1)
        hist_start_date = hist_end_date - timedelta(days=365) # Use a lookback period for features
        from main import load_prices_robust
        historical_data = load_prices_robust(ticker, hist_start_date, hist_end_date) # Fetch full lookback period

        if historical_data.empty:
            print(f"  ⚠️ No sufficient historical data for {ticker}. Skipping recommendation.")
            return None

        # Get the latest real-time price from Yahoo Finance
        latest_realtime_price = _get_latest_price_from_yahoo(ticker)
        if latest_realtime_price is None:
            print(f"  ⚠️ Could not get latest real-time price for {ticker}. Skipping recommendation.")
            return None

        # Create a "live" data point for today using the latest real-time price
        # This simulates a new bar for today with the real-time close price
        live_bar_data = {
            'Open': historical_data['Close'].iloc[-1], # Use previous day's close as today's open
            'High': max(historical_data['Close'].iloc[-1], latest_realtime_price), # Placeholder for high
            'Low': min(historical_data['Close'].iloc[-1], latest_realtime_price),  # Placeholder for low
            'Close': latest_realtime_price,
            'Volume': 0 # Live volume is not easily available from this WS feed
        }
        # Add financial features from the last historical day, as they don't change intraday
        for col in historical_data.columns:
            if col.startswith('Fin_'):
                live_bar_data[col] = historical_data[col].iloc[-1]

        live_bar_df = pd.DataFrame([live_bar_data], index=[end_date], columns=historical_data.columns)
        live_bar_df.index.name = "Date"
        
        # Append the live bar to historical data for feature calculation
        # Ensure column consistency before concat
        missing_cols_hist = set(live_bar_df.columns) - set(historical_data.columns)
        for col in missing_cols_hist:
            historical_data[col] = 0 # Add missing columns to historical_data with default 0
        
        missing_cols_live = set(historical_data.columns) - set(live_bar_df.columns)
        for col in missing_cols_live:
            live_bar_df[col] = 0 # Add missing columns to live_bar_df with default 0

        # Reorder columns to match
        live_bar_df = live_bar_df[historical_data.columns]

        combined_data = pd.concat([historical_data, live_bar_df])
        
        # Re-calculate features on the combined data to include the latest price
        from main import fetch_training_data
        # fetch_training_data returns (ready_df, actual_feature_set)
        # We only need the ready_df for the latest row to get features for prediction
        processed_combined_data, _ = fetch_training_data(ticker, combined_data, TARGET_PERCENTAGE)

        if processed_combined_data.empty:
            print(f"  ⚠️ Failed to process combined data for {ticker}. Skipping recommendation.")
            return None

        # Get recommendation using the newly processed data (which includes the live price)
        recommendation, buy_prob, sell_prob = get_recommendation(ticker, model_buy, model_sell, scaler, processed_combined_data, buy_thresh, sell_thresh)
        
        return {
            'ticker': ticker,
            'action': recommendation,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'buy_thresh': buy_thresh,
            'sell_thresh': sell_thresh,
            'latest_price': latest_realtime_price # Include the latest price
        }
    except Exception as e:
        print(f"  ⚠️ Error processing ticker {ticker}: {e}")
        return None

def run_live_trading():
    """Main function to run the live trading bot with a streaming, end-to-end process for each ticker."""
    import time # Explicitly import time here to ensure it's bound within this function's scope
    print("🚀 Starting Live Trading Bot...")

    # --- Display API Key Status ---
    print("\n--- API Key Status ---")
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        print("✅ Alpaca API Keys: Set")
    else:
        print("❌ Alpaca API Keys: NOT Set")
    
    print("----------------------")
    
    alpaca_trading_client = None
    tradable_tickers: List[str] = [] # Initialize tradable_tickers here

    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            alpaca_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            print("✅ Alpaca Paper Trading Client initialized.")

            # --- Print Initial Alpaca Portfolio ---
            positions_list, total_market_value = _get_alpaca_portfolio_positions(alpaca_trading_client)
            print("\n--- Initial Alpaca Portfolio ---")
            if positions_list:
                for pos in positions_list:
                    print(f"  - {pos['ticker']}: {pos['qty']} shares @ ${pos['current_price']:.2f} (Value: ${pos['market_value']:,.2f})")
                print(f"  Total Portfolio Market Value: ${total_market_value:,.2f}")
            else:
                print("  No open positions.")
            print("------------------------------")

        except Exception as e:
            print(f"⚠️ Error initializing Alpaca Trading Client: {e}. Live trading actions will be disabled.")
            return
    else:
        print("⚠️ Alpaca API keys not set. Live trading is disabled.")
        return

    models_dir = Path("logs/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Optimized Thresholds ---
    optimized_params = {}
    optimized_params_path = Path("logs/optimized_per_ticker_params.json")
    if optimized_params_path.exists():
        with open(optimized_params_path, 'r') as f:
            optimized_params = json.load(f)
    else:
        print(f"⚠️ Optimized parameters file not found at {optimized_params_path}. Using default thresholds.")

    # --- Fetch all potential tradable tickers based on MARKET_SELECTION in main.py ---
    try:
        all_potential_tickers = get_all_tickers()
        if not all_potential_tickers:
            print("❌ No potential tradable tickers found based on MARKET_SELECTION. Aborting live trading.")
            return
        print(f"✅ Fetched {len(all_potential_tickers)} potential tradable tickers based on MARKET_SELECTION.")
    except Exception as e:
        print(f"⚠️ Could not fetch potential tradable tickers ({e}). Aborting live trading.")
        return

    # Filter tickers to only include those with existing and up-to-date models
    models_dir = Path("logs/models")
    tradable_tickers = []
    current_time = datetime.now(timezone.utc)
    retrain_threshold = timedelta(days=MODEL_RETRAIN_DAYS)

    print("\n--- Filtering tickers based on available and up-to-date models ---")
    for ticker in all_potential_tickers:
        model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
        if model_buy_path.exists():
            model_mod_time = datetime.fromtimestamp(model_buy_path.stat().st_mtime, timezone.utc)
            if (current_time - model_mod_time) <= retrain_threshold:
                tradable_tickers.append(ticker)
            else:
                print(f"  ℹ️ Skipping {ticker}: Model is outdated (older than {MODEL_RETRAIN_DAYS} days).")
        else:
            print(f"  ℹ️ Skipping {ticker}: No trained model found.")

    if not tradable_tickers:
        print("\n--- No tickers with existing and up-to-date models found after filtering. Aborting live trading. ---")
        return
    print(f"✅ Identified {len(tradable_tickers)} tickers with existing and up-to-date models for live trading.")

    # The script will now proceed with the filtered list of all available tradable tickers.
    print(f"DEBUG: Subscribing to all {len(tradable_tickers)} available tickers.")


    # --- Initial Pass: Verify/Train Models and Filter Tickers ---
    print("\n--- Initial Pass: Verifying/Training Models and Filtering Tickers ---")
    print("DEBUG: Starting initial pass for model verification/training.")
    num_processes = max(1, cpu_count() - 1)
    
    initial_worker_args = [(ticker, models_dir, optimized_params) for ticker in tradable_tickers]
    
    model_ready_tickers_initial = []
    skipped_tickers_info = []

    print(f"DEBUG: Initializing Pool with {num_processes} processes for model verification/training.")
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tradable_tickers), desc="Verifying/Training Models") as pbar:
            for result in pool.imap_unordered(verify_train_and_recommend_worker, initial_worker_args):
                if result['status'] == 'success':
                    model_ready_tickers_initial.append(result['data']['ticker'])
                else:
                    skipped_tickers_info.append({'ticker': result['ticker'], 'reason': result['reason']})
                    pbar.write(f"  ⚠️ Skipping {result['ticker']} for live trading: {result['reason']}")
                pbar.update()
    print("DEBUG: Initial pass for model verification/training completed.")

    if not model_ready_tickers_initial:
        print("\n--- No tickers with successfully trained models and prices found. Aborting live trading. ---")
        return

    print(f"\n✅ Identified {len(model_ready_tickers_initial)} tickers with trained models and prices for live trading.")
    if skipped_tickers_info:
        print(f"ℹ️ {len(skipped_tickers_info)} tickers were skipped due to various issues.")

    # --- Main Pass: Generate Recommendations for Model-Ready Tickers ---
    print("\n--- Generating Recommendations for Model-Ready Tickers ---")
    print("DEBUG: Starting main pass for recommendation generation.")

    main_worker_args = [(ticker, models_dir, optimized_params) for ticker in model_ready_tickers_initial]
    
    all_generated_recommendations = []
    print(f"DEBUG: Initializing Pool with {num_processes} processes for recommendation generation.")
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(model_ready_tickers_initial), desc="Generating Recommendations") as pbar:
            for result in pool.imap_unordered(verify_train_and_recommend_worker, main_worker_args):
                if result['status'] == 'success':
                    rec = result['data']
                    all_generated_recommendations.append(rec)
                    buy_prob_str = f"{rec['buy_prob'] * 100:.2f}%"
                    sell_prob_str = f"{rec['sell_prob'] * 100:.2f}%"
                    pbar.write(f"  - Rec for {rec['ticker']}: {rec['action']} (Buy: {buy_prob_str}, Sell: {sell_prob_str})")
                else:
                    pbar.write(f"  ❌ Failed to generate recommendation for {result['ticker']}: {result['reason']}")
                pbar.update()
    print("DEBUG: Main pass for recommendation generation completed.")

    if not all_generated_recommendations:
        print("\n--- No valid recommendations were generated for model-ready tickers. ---")
        return
        
    print("\n--- Full Recommendation Summary ---")
    print(f"{'Ticker':<10} | {'Recommendation':<15} | {'Buy Prob %':>12} | {'Sell Prob %':>12} | {'Buy Thresh':>12} | {'Sell Thresh':>12}")
    print("-" * 80)
    sorted_generated_recs = sorted(all_generated_recommendations, key=lambda x: x['buy_prob'], reverse=True)
    for rec in sorted_generated_recs:
        buy_prob_str = f"{rec['buy_prob'] * 100:.2f}%"
        sell_prob_str = f"{rec['sell_prob'] * 100:.2f}%"
        buy_thresh_str = f"{rec['buy_thresh']:.2f}"
        sell_thresh_str = f"{rec['sell_thresh']:.2f}"
        print(f"{rec['ticker']:<10} | {rec['action']:<15} | {buy_prob_str:>12} | {sell_prob_str:>12} | {buy_thresh_str:>12} | {sell_thresh_str:>12}")
    print("-" * 120)

    # --- Execute Trades based on Recommendations ---
    if LIVE_TRADING_ENABLED and alpaca_trading_client:
        print("\n--- Executing Live Trades on Alpaca ---")
        
        available_buying_power = _get_alpaca_buying_power(alpaca_trading_client)
        if available_buying_power is None or available_buying_power <= 0:
            print("  ⚠️ No available buying power on Alpaca. Skipping all BUY orders.")
            # Still proceed with SELL orders if any
        
        current_positions_raw = alpaca_trading_client.get_all_positions()
        current_positions_map = {}
        for pos in current_positions_raw:
            current_positions_map[pos.symbol] = {
                'qty': float(pos.qty),
                'qty_available': float(pos.qty_available) # Get available quantity
            }
        
        for rec in sorted_generated_recs:
            ticker = rec['ticker']
            action = rec['action']
            latest_price = rec.get('latest_price') # Get the latest price from the recommendation result
            
            if latest_price is None:
                print(f"  ⚠️ Recommendation for {ticker} did not include latest price. Skipping trade execution.")
                continue

            if action == "BUY":
                if ticker not in current_positions_map:
                    if available_buying_power is not None and available_buying_power > 0:
                        # Calculate quantity based on fixed investment amount, but limited by buying power
                        max_qty_by_investment = int(INVESTMENT_PER_STOCK_LIVE / latest_price)
                        max_qty_by_buying_power = int(available_buying_power / latest_price)
                        
                        qty_to_buy = min(max_qty_by_investment, max_qty_by_buying_power)
                        
                        if qty_to_buy > 0:
                            _place_alpaca_order(alpaca_trading_client, ticker, qty_to_buy, OrderSide.BUY)
                            # Deduct the cost from available buying power for subsequent trades
                            available_buying_power -= (qty_to_buy * latest_price) # Approximate deduction
                        else:
                            print(f"  ⚠️ Insufficient buying power for {ticker} (needed ${INVESTMENT_PER_STOCK_LIVE:.2f}, available ${available_buying_power:.2f}). Skipping BUY order.")
                    else:
                        print(f"  ⚠️ No buying power available for {ticker}. Skipping BUY order.")
                else:
                    print(f"  ℹ️ Already holding {ticker}. Skipping BUY order.")
            elif action == "SELL":
                if ticker in current_positions_map:
                    qty_to_sell = int(current_positions_map[ticker]['qty_available']) # Use qty_available for selling
                    if qty_to_sell > 0:
                        _place_alpaca_order(alpaca_trading_client, ticker, qty_to_sell, OrderSide.SELL)
                    else:
                        print(f"  ℹ️ No available quantity to sell for {ticker}. Skipping SELL order.")
                else:
                    print(f"  ℹ️ Not holding {ticker}. Skipping SELL order.")
            elif action == "HOLD":
                print(f"  ℹ️ Recommendation for {ticker} is HOLD. No action taken.")
    else:
        print("\n--- Live Trading Execution is DISABLED ---")
        print("  Set `LIVE_TRADING_ENABLED = True` in `src/live_trading.py` to enable actual trade execution.")
        print("  Ensure Alpaca API keys are set as environment variables.")

    print("\n💡 Next Steps:")
    print("  - Review the recommendations and trade executions.")
    print("  - Adjust `MARKET_SELECTION` in `main.py` or `TARGET_PERCENTAGE` for different strategies.")
    print("="*80)

if __name__ == "__main__":
    run_live_trading()
