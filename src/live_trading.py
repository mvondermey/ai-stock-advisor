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
from main import train_worker, TARGET_PERCENTAGE, TRAIN_LOOKBACK_DAYS
import joblib
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

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

def verify_ticker_data(ticker: str) -> Optional[str]:
    """Worker function to verify data availability for a single ticker."""
    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = data_client.get_stock_bars(request_params)
        if not bars.df.empty and 'close' in bars.df.columns:
            return ticker
    except APIError:
        pass
    except Exception as e:
        print(f"  ⚠️ Error verifying ticker {ticker}: {e}")
    return None


def verify_train_and_recommend_worker(args: Tuple) -> Optional[Dict]:
    """
    Worker function that handles the entire pipeline for a single ticker:
    1. Verifies data availability.
    2. Trains/retrains the model if necessary.
    3. Generates a recommendation.
    """
    ticker, models_dir, optimized_params = args
    
    # 1. Verify data
    if not verify_ticker_data(ticker):
        return None # Ticker is not valid

    # 2. Check model and train if needed
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    current_time = datetime.now(timezone.utc)
    retrain_threshold = timedelta(days=MODEL_RETRAIN_DAYS)
    
    needs_training = False
    if not model_buy_path.exists():
        needs_training = True
        print(f"  ℹ️ No models found for {ticker}. Queuing for training.")
    else:
        model_mod_time = datetime.fromtimestamp(model_buy_path.stat().st_mtime, timezone.utc)
        if (current_time - model_mod_time) > retrain_threshold:
            needs_training = True
            print(f"  ℹ️ Models for {ticker} are stale. Queuing for retraining.")

    if needs_training:
        train_end = datetime.now(timezone.utc) - timedelta(days=1)
        train_start = train_end - timedelta(days=TRAIN_LOOKBACK_DAYS)
        from main import load_prices_robust, train_worker
        
        df_train_period = load_prices_robust(ticker, train_start, train_end)
        
        training_result = train_worker((ticker, df_train_period, TARGET_PERCENTAGE, None))
        if not training_result or not training_result.get('model_buy'):
            print(f"  ❌ Failed to train models for {ticker}. Skipping recommendation.")
            return None

    # 3. Generate recommendation
    try:
        recommendation = process_ticker(ticker, models_dir, optimized_params)
        if recommendation and recommendation['action'] != "HOLD":
             # Print recommendation as soon as it's generated
            buy_prob_str = f"{recommendation['buy_prob'] * 100:.2f}%"
            sell_prob_str = f"{recommendation['sell_prob'] * 100:.2f}%"
            print(f"  ✅ Recommendation for {ticker}: {recommendation['action']} (Buy Prob: {buy_prob_str}, Sell Prob: {sell_prob_str})")
        return recommendation
    except Exception as e:
        print(f"  ⚠️ Error generating recommendation for {ticker}: {e}")
        return None

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

def _get_latest_price_from_alpaca(ticker: str) -> Optional[float]:
    """
    Fetches the latest price for a ticker from Alpaca.
    It first tries to get the latest daily bar and uses the closing price.
    If that fails, it falls back to the latest quote, then the latest trade.
    """
    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Primary method: Fetch the latest daily bar
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=5) # Look back 5 days for the latest bar
        
        bars_request = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        latest_bars = data_client.get_stock_bars(bars_request)
        
        if ticker in latest_bars.df.index.get_level_values('symbol'):
            ticker_bars = latest_bars.df.loc[ticker]
            if not ticker_bars.empty:
                return float(ticker_bars.iloc[-1]['close'])

        # Fallback 1: Try to get the latest quote
        print(f"  ℹ️ Could not get latest bar for {ticker}, falling back to latest quote.")
        quote_request_params = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
        latest_quote = data_client.get_stock_latest_quote(quote_request_params)
        
        if ticker in latest_quote and latest_quote[ticker] and latest_quote[ticker].ask_price is not None:
            return float(latest_quote[ticker].ask_price)
            
        # Fallback 2: Try the latest trade
        print(f"  ℹ️ Could not get latest quote for {ticker}, falling back to latest trade.")
        trade_request_params = StockLatestTradeRequest(symbol_or_symbols=[ticker])
        latest_trade = data_client.get_stock_latest_trade(trade_request_params)
        
        if ticker in latest_trade and latest_trade[ticker] and latest_trade[ticker].price is not None:
            return float(latest_trade[ticker].price)
            
        print(f"  ⚠️ Could not retrieve any latest price for {ticker} from Alpaca.")
        return None
        
    except APIError as e:
        print(f"  ❌ Alpaca API Error for {ticker}: {e}. Could not fetch latest price.")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error fetching latest price for {ticker}: {e}")
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
        start_date = end_date - timedelta(days=365)
        from main import load_prices_robust
        live_data = load_prices_robust(ticker, start_date, end_date)

        recommendation, buy_prob, sell_prob = get_recommendation(ticker, model_buy, model_sell, scaler, live_data, buy_thresh, sell_thresh)
        
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
    print("🚀 Starting Live Trading Bot...")
    
    alpaca_trading_client = None
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

    # --- Fetch all tradable tickers ---
    try:
        search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = alpaca_trading_client.get_all_assets(search_params)
        tradable_tickers = [a.symbol for a in assets if a.tradable]
        print(f"✅ Fetched {len(tradable_tickers)} tradable US equity tickers from Alpaca.")
    except Exception as e:
        print(f"⚠️ Could not fetch asset list from Alpaca ({e}). Aborting.")
        return

    # --- Process all tickers in parallel from verification to recommendation ---
    print("\n--- Verifying, Training, and Generating Recommendations in Parallel ---")
    num_processes = max(1, cpu_count() - 1)
    worker_args = [(ticker, models_dir, optimized_params) for ticker in tradable_tickers]
    
    all_generated_recommendations = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tradable_tickers), desc="Processing Tickers") as pbar:
            for result in pool.imap_unordered(verify_train_and_recommend_worker, worker_args):
                if result:
                    all_generated_recommendations.append(result)
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
                price = _get_latest_price_from_alpaca(ticker)
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

if __name__ == "__main__":
    run_live_trading()
