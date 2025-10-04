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
from main import train_worker, TARGET_PERCENTAGE, TRAIN_LOOKBACK_DAYS, TWELVEDATA_API_KEY, get_all_tickers # Import get_all_tickers
import joblib
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import time # Import the time module
from twelvedata_ws import TwelveDataWebSocketClient
from twelvedata import TDClient # Import the TwelveData REST API client

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")

# --- Configuration for Model Retraining ---
MODEL_RETRAIN_DAYS = 90 # Retrain models if they are older than this many days

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

def verify_train_and_recommend_worker(args: Tuple) -> Dict:
    """
    Worker function that handles the entire pipeline for a single ticker and
    always returns a status dictionary.
    """
    ticker, models_dir, optimized_params, latest_prices_dict = args
    
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
        train_end = datetime.now(timezone.utc) - timedelta(days=1)
        train_start = train_end - timedelta(days=TRAIN_LOOKBACK_DAYS)
        from main import load_prices_robust, train_worker
        
        # This call now serves as our data verification. If it fails, the ticker is invalid.
        df_train_period = load_prices_robust(ticker, train_start, train_end)
        if df_train_period.empty:
            return {'status': 'failure', 'ticker': ticker, 'reason': 'Failed to load historical training data'}
            
        training_result = train_worker((ticker, df_train_period, TARGET_PERCENTAGE, None))
        if not training_result or not training_result.get('model_buy'):
            return {'status': 'failure', 'ticker': ticker, 'reason': 'Model training failed'}

    # 2. Generate recommendation
    try:
        recommendation = process_ticker(ticker, models_dir, optimized_params, latest_prices_dict)
        if recommendation:
            return {'status': 'success', 'data': recommendation}
        else:
            return {'status': 'failure', 'ticker': ticker, 'reason': 'Recommendation generation failed'}
    except Exception as e:
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

def _get_latest_price_from_twelvedata(ticker: str, latest_prices_dict: Dict[str, float]) -> Optional[float]:
    """
    Fetches the latest price for a ticker from TwelveData, preferring the passed dictionary.
    If not in dictionary, it falls back to REST API.
    """
    price_from_dict = latest_prices_dict.get(ticker)
    if price_from_dict is not None:
        return price_from_dict
    
    # If price not in dict, fall back to REST API

    if not TWELVEDATA_API_KEY:
        print(f"  ⚠️ TwelveData API key not set. Cannot fetch latest price for {ticker}.")
        return None

    try:
        tdc = TDClient(apikey=TWELVEDATA_API_KEY)
        
        # Fetch latest price using the TwelveData client
        # The `TDClient.price` method is suitable for getting the latest price
        price_data = tdc.price(symbol=ticker).as_json()

        if not price_data or 'price' not in price_data:
            print(f"  ℹ️ No latest price data found for {ticker} from TwelveData.")
            return None

        return float(price_data['price'])
            
    except Exception as e:
        print(f"  ❌ TwelveData API Error for {ticker}: {e}. Could not fetch latest price.")
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
    scaled_data = pd.DataFrame(scaled_data_np, columns=required_features)
    
    buy_prob = model_buy.predict_proba(scaled_data)[0][1]
    sell_prob = model_sell.predict_proba(scaled_data)[0][1]

    if buy_prob > buy_thresh:
        return "BUY", buy_prob, sell_prob
    elif sell_prob > sell_thresh:
        return "SELL", buy_prob, sell_prob
    else:
        return "HOLD", buy_prob, sell_prob

def process_ticker(ticker: str, models_dir: Path, optimized_params: Dict, latest_prices_dict: Dict[str, float]) -> Optional[Dict]:
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

        # Get the latest real-time price from the passed dictionary or fallback to REST API
        latest_realtime_price = _get_latest_price_from_twelvedata(ticker, latest_prices_dict)
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
            'sell_thresh': sell_thresh
        }
    except Exception as e:
        print(f"  ⚠️ Error processing ticker {ticker}: {e}")
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
            'sell_thresh': sell_thresh
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
    
    if TWELVEDATA_API_KEY:
        print("✅ TwelveData API Key: Set")
    else:
        print("❌ TwelveData API Key: NOT Set")
    print("----------------------")
    
    alpaca_trading_client = None
    twelvedata_ws_client: Optional[TwelveDataWebSocketClient] = None # Declare WebSocket client
    tradable_tickers: List[str] = [] # Initialize tradable_tickers here

    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            alpaca_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            print("✅ Alpaca Paper Trading Client initialized.")
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

    # --- Fetch all tradable tickers based on MARKET_SELECTION in main.py ---
    try:
        # Use get_all_tickers from main.py to respect MARKET_SELECTION
        tradable_tickers = get_all_tickers()
        if not tradable_tickers:
            print("❌ No tradable tickers found based on MARKET_SELECTION. Aborting live trading.")
            return
        print(f"✅ Fetched {len(tradable_tickers)} tradable tickers based on MARKET_SELECTION.")
    except Exception as e:
        print(f"⚠️ Could not fetch tradable tickers ({e}). Aborting live trading.")
        return

    if TWELVEDATA_API_KEY:
        twelvedata_ws_client = TwelveDataWebSocketClient(TWELVEDATA_API_KEY, symbols_to_subscribe=tradable_tickers)
        twelvedata_ws_client.connect()
    else:
        print("⚠️ TwelveData API key not set. Real-time price fetching via TwelveData WebSocket will be unavailable.")

    # Subscriptions are now handled by the WebSocket client's _on_open method
    if twelvedata_ws_client:
        print("ℹ️ TwelveData WebSocket client initialized. Subscriptions will be handled on connection.")
        time.sleep(15) # Give more time for initial price data to stream in and subscriptions to process

        successful_count = twelvedata_ws_client.get_successful_subscriptions_count()
        print(f"✅ Successfully subscribed to {successful_count} out of {len(tradable_tickers)} NASDAQ 100 tickers via TwelveData WebSocket.")
        if successful_count < len(tradable_tickers):
            print(f"ℹ️ Some subscriptions failed. The client will periodically retry these failed subscriptions.")
        print("\n--- Note on Multiple WebSocket Connections ---")
        print("The current implementation uses a single WebSocket connection for simplicity and to manage API limits.")
        print("Implementing 8 concurrent WebSocket connections would require a significant architectural change")
        print("to manage multiple client instances, their threads, and data aggregation. This would be a separate, more complex task.")

    # --- Process all tickers in parallel from verification to recommendation ---
    print("\n--- Verifying, Training, and Generating Recommendations in Parallel ---")
    num_processes = max(1, cpu_count() - 1)
    # Collect latest prices from the WebSocket client before passing to workers
    latest_prices_from_ws = {}
    if twelvedata_ws_client:
        # Give some time for initial price data to stream in
        print("ℹ️ Waiting for initial WebSocket price data...")
        time.sleep(10) # Increased sleep to allow more data to come in
        latest_prices_from_ws = twelvedata_ws_client.latest_prices.copy()
        print(f"✅ Collected {len(latest_prices_from_ws)} latest prices from WebSocket.")
    
    print("\n--- Process all tickers in parallel from verification to recommendation ---")
    num_processes = max(1, cpu_count() - 1)
    print(f"ℹ️ Using {num_processes} parallel processes for ticker processing.")
    worker_args = [(ticker, models_dir, optimized_params, latest_prices_from_ws) for ticker in tradable_tickers]
    
    all_generated_recommendations = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tradable_tickers), desc="Processing Tickers") as pbar:
            for result in pool.imap_unordered(verify_train_and_recommend_worker, worker_args):
                if result['status'] == 'success':
                    rec = result['data']
                    all_generated_recommendations.append(rec)
                    buy_prob_str = f"{rec['buy_prob'] * 100:.2f}%"
                    sell_prob_str = f"{rec['sell_prob'] * 100:.2f}%"
                    pbar.write(f"  - Rec for {rec['ticker']}: {rec['action']} (Buy: {buy_prob_str}, Sell: {sell_prob_str})")
                else:
                    # Print a message for skipped tickers to provide feedback
                    pbar.write(f"  ⚠️ Skipping {result['ticker']}: {result['reason']}")
                pbar.update()

    # --- Print Full Recommendation Summary ---
    if not all_generated_recommendations:
        print("\n--- No valid tickers or recommendations were generated. ---")
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
    print("-" * 80)

    all_recommendations = [rec for rec in all_generated_recommendations if rec['action'] != "HOLD"]
    if not all_recommendations:
        print("\n--- No new BUY/SELL recommendations generated. No trades to execute. ---")
        return

    # --- Execution Step ---
    print("\n--- Executing Trades Based on Recommendations ---")
    positions_list, _ = _get_alpaca_portfolio_positions(alpaca_trading_client)
    portfolio = {pos['ticker']: pos['qty'] for pos in positions_list}

    buy_recs = sorted([r for r in all_recommendations if r['action'] == 'BUY'], key=lambda x: x['buy_prob'], reverse=True)
    sell_recs = [r for r in all_recommendations if r['action'] == 'SELL']

    # --- 1. Execute all SELL orders first ---
    print("\n--- Phase 1: Executing SELL Recommendations ---")
    print(f"{'Ticker':<10} | {'Recommendation':<15} | {'Action / Status':<60}")
    print("-" * 100)
    for rec in sell_recs:
        ticker = rec['ticker']
        status = ""
        if ticker in portfolio:
            qty_to_sell = portfolio[ticker]
            try:
                market_order_data = MarketOrderRequest(symbol=ticker, qty=qty_to_sell, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                alpaca_trading_client.submit_order(order_data=market_order_data)
                status = f"✅ Submitted SELL order for {qty_to_sell} shares."
            except Exception as e:
                status = f"❌ Error submitting SELL order: {e}"
        else:
            status = "ℹ️ No position to sell (not in portfolio)."
        print(f"{ticker:<10} | {'SELL':<15} | {status:<60}")
    print("-" * 100)

    # --- 2. Re-evaluate buying power and execute BUY orders ---
    print("\n--- Phase 2: Executing BUY Recommendations ---")
    # Re-fetch buying power after sells have been submitted
    print("...waiting a moment for sell orders to potentially fill and update buying power...")
    import time
    time.sleep(5) # Wait 5 seconds
    buying_power = _get_alpaca_buying_power(alpaca_trading_client)
    
    CAPITAL_PER_TRADE = 1000.0
    
    print(f"\n{'Ticker':<10} | {'Recommendation':<15} | {'Buy Prob %':>12} | {'Action / Status':<60}")
    print("-" * 100)

    if not buy_recs:
        print("No BUY recommendations to process.")
    elif buying_power is None or buying_power < CAPITAL_PER_TRADE:
        print(f"Insufficient buying power (${buying_power:,.2f}) to execute any trades with a ${CAPITAL_PER_TRADE:,.2f} minimum. Skipping all BUY trades.")
    else:
        print(f"Starting BUY execution with available buying power: ${buying_power:,.2f}")
        for rec in buy_recs:
            ticker = rec['ticker']
            buy_prob = rec['buy_prob'] * 100
            buy_prob_str = f"{buy_prob:.2f}%"
            status = ""

            if buying_power < CAPITAL_PER_TRADE:
                status = f"ℹ️ Insufficient buying power left (${buying_power:,.2f}) to place trade. Halting further BUYs."
                print(f"{ticker:<10} | {'BUY':<15} | {buy_prob_str:>12} | {status:<60}")
                break # Stop processing more buy recommendations

            if ticker in portfolio:
                status = f"ℹ️ Skipping BUY because {ticker} is already in the portfolio."
            else:
                price = _get_latest_price_from_twelvedata(ticker, twelvedata_ws_client) # Use TwelveData for latest price
                if price and price > 0:
                    qty_to_buy = int(CAPITAL_PER_TRADE / price)
                    if qty_to_buy > 0:
                        try:
                            market_order_data = MarketOrderRequest(symbol=ticker, qty=qty_to_buy, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                            alpaca_trading_client.submit_order(order_data=market_order_data)
                            status = f"✅ Submitted BUY order for {qty_to_buy} shares of {ticker} (~${CAPITAL_PER_TRADE:,.2f})."
                            buying_power -= (qty_to_buy * price) # Decrement buying power
                        except Exception as e:
                            status = f"❌ Error submitting BUY order: {e}"
                    else:
                        status = f"ℹ️ Not enough capital (${CAPITAL_PER_TRADE:,.2f}) to buy 1 share at ${price:,.2f}."
                else:
                    status = f"⚠️ Could not fetch real-time price. Skipping trade."
            
            print(f"{ticker:<10} | {'BUY':<15} | {buy_prob_str:>12} | {status:<60}")

    print("-" * 100)

    # --- Print Final Live Trading Summary ---
    print("\n" + "="*80)
    print("                     🚀 LIVE TRADING SESSION SUMMARY 🚀")
    print("="*80)

    if not all_generated_recommendations:
        print("\n--- No valid tickers or recommendations were generated during this session. ---")
    else:
        print("\n📈 Individual Ticker Recommendations (Sorted by Buy Probability):")
        print("-" * 120)
        print(f"{'Ticker':<10} | {'Recommendation':<15} | {'Buy Prob %':>12} | {'Sell Prob %':>12} | {'Buy Thresh':>12} | {'Sell Thresh':>12}")
        print("-" * 120)
        sorted_final_recs = sorted(all_generated_recommendations, key=lambda x: x['buy_prob'], reverse=True)
        for rec in sorted_final_recs:
            buy_prob_str = f"{rec['buy_prob'] * 100:.2f}%"
            sell_prob_str = f"{rec['sell_prob'] * 100:.2f}%"
            buy_thresh_str = f"{rec['buy_thresh']:.2f}"
            sell_thresh_str = f"{rec['sell_thresh']:.2f}"
            print(f"{rec['ticker']:<10} | {rec['action']:<15} | {buy_prob_str:>12} | {sell_prob_str:>12} | {buy_thresh_str:>12} | {sell_thresh_str:>12}")
        print("-" * 120)

    print("\n💡 Next Steps:")
    print("  - Review the recommendations and trade executions.")
    print("  - Adjust `MARKET_SELECTION` in `main.py` or `TARGET_PERCENTAGE` for different strategies.")
    print("  - Monitor TwelveData WebSocket logs for persistent subscription failures.")
    print("="*80)

    # Disconnect WebSocket client
    if twelvedata_ws_client:
        twelvedata_ws_client.disconnect()

if __name__ == "__main__":
    run_live_trading()
