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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone

# Alpaca API credentials (set as environment variables for security)
ALPACA_API_KEY          = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY       = os.environ.get("ALPACA_SECRET_KEY")

def _get_alpaca_account_balance(trading_client: TradingClient) -> Optional[float]:
    """Fetches the current equity from the Alpaca trading account."""
    try:
        account = trading_client.get_account()
        if account and account.equity:
            print(f"✅ Fetched Alpaca account equity: ${float(account.equity):,.2f}")
            return float(account.equity)
        else:
            print("⚠️ Could not retrieve Alpaca account equity.")
            return None
    except Exception as e:
        print(f"  ⚠️ Error fetching Alpaca account balance: {e}")
        return None

def _get_alpaca_portfolio_positions(trading_client: TradingClient) -> Tuple[List[Dict], float]:
    """
    Fetches current portfolio positions from Alpaca and returns a list of dictionaries
    with ticker, quantity, current price, market value, and the total market value of positions.
    """
    if not trading_client:
        return [], 0.0
    
    portfolio_positions = []
    total_positions_market_value = 0.0
    
    try:
        positions = trading_client.get_all_positions()
        if positions:
            print(f"✅ Fetched Alpaca portfolio positions:")
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                current_price = float(pos.current_price)
                market_value = float(pos.market_value)
                
                portfolio_positions.append({
                    'ticker': symbol,
                    'qty': qty,
                    'current_price': current_price,
                    'market_value': market_value
                })
                total_positions_market_value += market_value
            return portfolio_positions, total_positions_market_value
        else:
            print("ℹ️ No open positions found in Alpaca portfolio.")
            return [], 0.0
    except Exception as e:
        print(f"  ⚠️ Error fetching Alpaca portfolio positions: {e}")
        return [], 0.0

def _get_latest_price_from_alpaca(ticker: str) -> Optional[float]:
    """Fetches the latest trade price for a ticker from Alpaca."""
    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        request_params = StockLatestTradeRequest(symbol_or_symbols=[ticker])
        latest_trade = data_client.get_stock_latest_trade(request_params)
        if ticker in latest_trade and latest_trade[ticker]:
            price = float(latest_trade[ticker].price)
            print(f"  ℹ️ Fetched real-time price for {ticker}: ${price:.2f}")
            return price
        print(f"  ⚠️ No latest trade data found for {ticker}.")
        return None
    except Exception as e:
        print(f"  ⚠️ Error fetching latest price for {ticker}: {e}")
        return None

def execute_final_trades(
    alpaca_trading_client: TradingClient,
    final_recommendations: List[Dict],
    capital_per_stock: float
):
    """Executes trades on Alpaca based on the final recommendations."""
    if not alpaca_trading_client:
        print("⚠️ Alpaca trading client not available. Skipping live trades.")
        return

    print("\n" + "="*80)
    print("                     📈 EXECUTING FINAL TRADES ON ALPACA 📈")
    print("="*80)

    # Get current portfolio positions first
    portfolio_positions, _ = _get_alpaca_portfolio_positions(alpaca_trading_client)
    held_tickers = {pos['ticker']: pos['qty'] for pos in portfolio_positions}

    for rec in final_recommendations:
        ticker = rec['ticker']
        action = rec['last_ai_action']
        
        if action == "BUY":
            price = rec.get('price')
            if price and price > 0:
                qty_to_buy = int(capital_per_stock / price)
                if qty_to_buy > 0:
                    try:
                        market_order_data = MarketOrderRequest(
                            symbol=ticker,
                            qty=qty_to_buy,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        alpaca_trading_client.submit_order(order_data=market_order_data)
                        print(f"  ✅ Submitted BUY order for {qty_to_buy} shares of {ticker}.")
                    except Exception as e:
                        print(f"  ⚠️ Error submitting BUY order for {ticker}: {e}")
                else:
                    print(f"  ℹ️ Not enough capital (${capital_per_stock:.2f}) to buy 1 share of {ticker} at ${price:.2f}.")
            else:
                print(f"  ⚠️ Could not determine price for {ticker}. Skipping BUY order.")
        
        elif action == "SELL":
            if ticker in held_tickers:
                qty_to_sell = held_tickers[ticker]
                try:
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=qty_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    alpaca_trading_client.submit_order(order_data=market_order_data)
                    print(f"  ✅ Submitted SELL order for {qty_to_sell} shares of {ticker}.")
                except Exception as e:
                    print(f"  ⚠️ Unexpected error submitting SELL order for {ticker}: {e}")
            else:
                print(f"  ℹ️ No position to sell for {ticker} (not in portfolio).")

    print("="*80)

def get_recommendation(ticker: str, model_buy, model_sell, scaler, data: pd.DataFrame) -> str:
    """Generates a buy, sell, or hold recommendation for a single ticker."""
    if data.empty:
        return "HOLD"

    # Prepare the latest data point for prediction
    latest_data = data.iloc[-1:].copy()
    
    # Ensure all required features are present
    required_features = scaler.feature_names_in_
    for feature in required_features:
        if feature not in latest_data.columns:
            latest_data[feature] = 0 # or some other default value

    latest_data = latest_data[required_features]
    
    # Scale the features
    scaled_data_np = scaler.transform(latest_data)
    scaled_data = pd.DataFrame(scaled_data_np, columns=required_features)
    
    # Get buy and sell probabilities
    buy_prob = model_buy.predict_proba(scaled_data)[0][1]
    sell_prob = model_sell.predict_proba(scaled_data)[0][1]

    if buy_prob > 0.5:
        return "BUY"
    elif sell_prob > 0.5:
        return "SELL"
    else:
        return "HOLD"

def print_recommendation_summary(final_recommendations: List[Dict]):
    """Prints a summary table of the final recommendations from the last backtest."""
    
    optimized_params_per_ticker = {}
    optimized_params_file = Path("logs/optimized_per_ticker_params.json")
    if optimized_params_file.exists():
        try:
            with open(optimized_params_file, 'r') as f:
                optimized_params_per_ticker = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Could not decode JSON from {optimized_params_file}. Thresholds will use defaults.")
        except Exception as e:
            print(f"⚠️ Error reading {optimized_params_file}: {e}. Thresholds will use defaults.")

    print("\n" + "="*160)
    print(" " * 60 + "🚀 FINAL RECOMMENDATION SUMMARY 🚀")
    print("="*160)
    print("\n📈 Individual Ticker Performance (from last backtest):")
    print("-" * 160)
    print(f"{'Ticker':<10} | {'1Y Perf':>10} | {'YTD Perf':>10} | {'AI Sharpe':>12} | {'Last AI Action':<16} | {'Buy Prob':>10} | {'Sell Prob':>10} | {'Buy Thresh':>12} | {'Sell Thresh':>12} | {'Target %':>10}")
    print("-" * 160)

    # Defaults from main.py for fallback
    MIN_PROBA_BUY = 0.4
    MIN_PROBA_SELL = 0.4
    TARGET_PERCENTAGE = 0.01

    for res in final_recommendations:
        ticker = str(res.get('ticker', 'N/A'))
        optimized_params = optimized_params_per_ticker.get(ticker, {})
        
        buy_thresh = optimized_params.get('min_proba_buy', MIN_PROBA_BUY)
        sell_thresh = optimized_params.get('min_proba_sell', MIN_PROBA_SELL)
        target_perc = optimized_params.get('target_percentage', TARGET_PERCENTAGE)

        one_year_perf_str = f"{res.get('one_year_perf', 0.0):>9.2f}%" if pd.notna(res.get('one_year_perf')) else "N/A".rjust(10)
        ytd_perf_str = f"{res.get('ytd_perf', 0.0):>9.2f}%" if pd.notna(res.get('ytd_perf')) else "N/A".rjust(10)
        sharpe_str = f"{res.get('sharpe', 0.0):>11.2f}" if pd.notna(res.get('sharpe')) else "N/A".rjust(12)
        buy_prob_str = f"{res.get('buy_prob', 0.0):>9.2f}" if pd.notna(res.get('buy_prob')) else "N/A".rjust(10)
        sell_prob_str = f"{res.get('sell_prob', 0.0):>9.2f}" if pd.notna(res.get('sell_prob')) else "N/A".rjust(10)
        last_ai_action_str = str(res.get('last_ai_action', 'HOLD'))
        
        print(f"{ticker:<10} | {one_year_perf_str} | {ytd_perf_str} | {sharpe_str} | {last_ai_action_str:<16} | {buy_prob_str} | {sell_prob_str} | {buy_thresh:>11.2f} | {sell_thresh:>11.2f} | {target_perc:>9.2%}")
    
    print("-" * 160)


def run_live_trading():
    """Main function to run the live trading bot."""
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
    if not models_dir.exists():
        print(f"❌ Models directory not found at {models_dir}. Please run the backtest and training first.")
        return

    # Get a list of all tickers with trained models
    tickers = sorted(list(set([f.name.split('_')[0] for f in models_dir.glob('*_model_buy.joblib')])))
    
    if not tickers:
        print("ℹ️ No trained models found.")
        return

    recommendations = []
    for ticker in tickers:
        try:
            model_buy = joblib.load(models_dir / f"{ticker}_model_buy.joblib")
            model_sell = joblib.load(models_dir / f"{ticker}_model_sell.joblib")
            scaler = joblib.load(models_dir / f"{ticker}_scaler.joblib")

            # Fetch the latest data for the ticker
            # You might need to adjust the start date based on your feature engineering requirements
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=365) 
            
            # This is a placeholder for a function that fetches live data
            # In a real scenario, you would replace this with a call to your data provider
            from main import load_prices_robust
            live_data = load_prices_robust(ticker, start_date, end_date)

            recommendation = get_recommendation(ticker, model_buy, model_sell, scaler, live_data)
            
            if recommendation != "HOLD":
                latest_price = _get_latest_price_from_alpaca(ticker)
                if latest_price is None:
                    # Fallback to last close if real-time price fails
                    latest_price = live_data['Close'].iloc[-1] if not live_data.empty else 0
                    print(f"  ⚠️ Could not fetch real-time price for {ticker}. Falling back to last close price: ${latest_price:.2f}")
                
                recommendations.append({
                    'ticker': ticker,
                    'last_ai_action': recommendation,
                    'price': latest_price
                })
        except Exception as e:
            print(f"  ⚠️ Error processing ticker {ticker}: {e}")

    if not recommendations:
        print("ℹ️ No new trading recommendations generated.")
    else:
        # --- Dynamically allocate capital based on buying power ---
        buying_power = _get_alpaca_account_balance(alpaca_trading_client)
        capital_per_stock = 0
        
        if buying_power is not None and buying_power > 0:
            buy_recommendations = [rec for rec in recommendations if rec['last_ai_action'] == 'BUY']
            num_buys = len(buy_recommendations)

            if num_buys > 0:
                capital_per_stock = buying_power / num_buys
                print(f"ℹ️ Available buying power: ${buying_power:,.2f}. Allocating ${capital_per_stock:,.2f} per BUY trade for {num_buys} tickers.")
        else:
            print("⚠️ No buying power available. Skipping BUY trades.")
            
        execute_final_trades(alpaca_trading_client, recommendations, capital_per_stock)

    # --- Display the summary from the last backtest ---
    recommendations_path = Path("logs/recommendations.json")
    if recommendations_path.exists():
        try:
            with open(recommendations_path, 'r') as f:
                backtest_recommendations = json.load(f)
            print_recommendation_summary(backtest_recommendations)
        except json.JSONDecodeError:
            print(f"\n⚠️ Could not decode JSON from {recommendations_path}. Cannot display summary.")
        except Exception as e:
            print(f"\n⚠️ Error reading {recommendations_path}: {e}. Cannot display summary.")
    else:
        print("\nℹ️ No backtest recommendation file found at 'logs/recommendations.json'. Run main.py to generate one.")


if __name__ == "__main__":
    run_live_trading()
