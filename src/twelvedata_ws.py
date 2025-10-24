import websocket
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, List # Added List
import os # Added os for API key

from src.twelvedata_api import TwelveDataAPI # Import the REST API client

class TwelveDataWebSocketClient:
    def __init__(self, api_key: str, symbols_to_subscribe: Optional[List[str]] = None, on_message_callback: Optional[Callable[[Dict], None]] = None):
        self.api_key = api_key
        self.twelvedata_api = TwelveDataAPI(api_key) # Initialize the REST API client
        # The base URL without the API key, as it will be sent in headers
        self.ws_url = "wss://ws.twelvedata.com/v1/quotes/price" 
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.latest_prices: Dict[str, float] = {} # Stores latest price for each subscribed symbol
        self.subscribed_symbols: set = set(symbols_to_subscribe) if symbols_to_subscribe else set() # Initialize with symbols
        self.successful_subscriptions_count: int = 0 # New: Counter for successful subscriptions
        # Stores symbols that failed to subscribe and their last failure time
        self.failed_subscriptions: Dict[str, datetime] = {} 
        self.api_fallback_symbols: set = set() # New: Stores symbols currently using API fallback
        self.on_message_callback = on_message_callback
        self._lock = threading.Lock()
        self._connect_thread: Optional[threading.Thread] = None
        self._retry_thread: Optional[threading.Thread] = None # New: Thread for retrying failed subscriptions
        self.retry_interval_seconds = 300 # Retry failed subscriptions every 5 minutes
        self._rate_limit_pause_until: Optional[datetime] = None # New: Global pause for rate limits
        # Removed max_ws_retries as API fallback will now happen after first retry interval

    def _on_open(self, ws):
        self.is_connected = True
        print(f"✅ TwelveData WebSocket connection opened.")
        with self._lock:
            # Attempt to subscribe to initial symbols and any previously failed ones
            # Reset successful count on new connection
            self.successful_subscriptions_count = 0 
            all_symbols_to_attempt = list(self.subscribed_symbols) + list(self.failed_subscriptions.keys())
            if all_symbols_to_attempt:
                self._send_subscribe_message(all_symbols_to_attempt)
        self._start_retry_thread() # Start the retry thread when connected

    def _on_message(self, ws, message):
        data = json.loads(message)
        if data.get('event') == 'price':
            symbol = data.get('symbol')
            price = data.get('price')
            if symbol and price is not None:
                with self._lock:
                    self.latest_prices[symbol] = float(price)
                    # If a price is received for a symbol that was previously failed, remove it from failed_subscriptions
                    if symbol in self.failed_subscriptions:
                        del self.failed_subscriptions[symbol]
                        self.successful_subscriptions_count += 1 # Count as successful if it starts streaming after a retry
                    # If a price is received for a symbol that was in API fallback, remove it from API fallback
                    if symbol in self.api_fallback_symbols:
                        self.api_fallback_symbols.remove(symbol)
                        print(f"  [WS] {symbol} is now streaming via WebSocket, removed from API fallback.")
                print(f"  [WS] Received price data for {symbol}: {data}") # Added print for received price data
                if self.on_message_callback:
                    self.on_message_callback(data)
        elif data.get('event') == 'subscribe-status':
            if data.get('status') == 'ok':
                for s in data.get('success', []):
                    symbol = s.get('symbol')
                    print(f"  [WS] Successfully subscribed to {symbol}")
                    with self._lock:
                        if symbol in self.failed_subscriptions:
                            del self.failed_subscriptions[symbol] # Remove from failed list on success
                        # Increment successful count for any successful subscription
                        self.successful_subscriptions_count += 1 
                        self.subscribed_symbols.add(symbol) # Ensure it's in subscribed list
                        if symbol in self.api_fallback_symbols:
                            self.api_fallback_symbols.remove(symbol) # If WS subscription is now ok, remove from API fallback
                            print(f"  [WS] {symbol} successfully subscribed via WebSocket, removed from API fallback.")
            if data.get('fails'):
                for f in data.get('fails', []):
                    symbol = f.get('symbol')
                    reason = f.get('reason', 'Unknown reason')
                    print(f"  [WS] Failed to subscribe to {symbol}: {reason}")
                    with self._lock:
                        now = datetime.now(timezone.utc) # Define now here
                        # Attempt immediate API fallback
                        print(f"  [API] Attempting immediate API fallback for {symbol} after WS failure.")
                        price, error_type = self.twelvedata_api.get_price(symbol)
                        if price is not None:
                            self.latest_prices[symbol] = price
                            self.api_fallback_symbols.add(symbol)
                            self.successful_subscriptions_count += 1
                            print(f"  [API] Successfully fetched price for {symbol} via immediate API fallback: {price}")
                            if self.on_message_callback:
                                self.on_message_callback({'event': 'price', 'symbol': symbol, 'price': price, 'source': 'API_IMMEDIATE_FALLBACK'})
                        else:
                            print(f"  [API] Immediate API fallback failed for {symbol}. Error type: {error_type}. Will retry periodically.")
                            # Store last_failure_time and error_type for periodic API retries
                            self.failed_subscriptions[symbol] = {'last_failure_time': now, 'error_type': error_type}
                        # Do NOT remove from self.subscribed_symbols here, as per user request to retry
        elif data.get('event') == 'heartbeat':
            pass # Ignore heartbeat messages
        elif data.get('event') == 'liveness':
            pass # Ignore liveness messages
        else:
            print(f"  [WS] Received: {data}")

    def _on_error(self, ws, error):
        print(f"❌ TwelveData WebSocket error: {error}")
        self.is_connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        print(f"⚠️ TwelveData WebSocket connection closed: {close_status_code} - {close_msg}")

    def _run_forever(self):
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            header={"X-TD-APIKEY": self.api_key} # Pass API key in header
        )
        # run_forever will automatically reconnect on disconnects
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def connect(self):
        if self._connect_thread is None or not self._connect_thread.is_alive():
            print("Attempting to connect to TwelveData WebSocket...")
            self._connect_thread = threading.Thread(target=self._run_forever)
            self._connect_thread.daemon = True
            self._connect_thread.start()
            # Give it a moment to connect
            # time.sleep(2) 

    def _retry_failed_subscriptions_loop(self):
        while self.is_connected:
            time.sleep(1) # Check more frequently to respect global pause
            with self._lock:
                now = datetime.now(timezone.utc)

                # --- Global Rate Limit Check ---
                if self._rate_limit_pause_until and now < self._rate_limit_pause_until:
                    remaining_pause = (self._rate_limit_pause_until - now).total_seconds()
                    # print(f"  [API] Global rate limit active. Pausing API fallbacks for {remaining_pause:.1f}s.")
                    continue # Skip this iteration, wait for pause to expire
                else:
                    self._rate_limit_pause_until = None # Reset if pause expired

                symbols_to_api_fallback_attempt = []
                
                for symbol, failure_info in list(self.failed_subscriptions.items()):
                    last_failure_time = failure_info['last_failure_time']
                    error_type = failure_info['error_type']
                    
                    current_retry_interval = 60 if error_type == 'rate_limit' else self.retry_interval_seconds
                    
                    if (now - last_failure_time).total_seconds() >= current_retry_interval:
                        symbols_to_api_fallback_attempt.append(symbol)
                
                if symbols_to_api_fallback_attempt:
                    print(f"  [API] Attempting API fallback for: {', '.join(symbols_to_api_fallback_attempt)}")
                    for symbol in symbols_to_api_fallback_attempt:
                        if symbol not in self.api_fallback_symbols: # Only fetch if not already in API fallback
                            price, error_type = self.twelvedata_api.get_price(symbol)
                            if price is not None:
                                self.latest_prices[symbol] = price
                                self.api_fallback_symbols.add(symbol)
                                del self.failed_subscriptions[symbol] # Remove from failed list if API fallback is successful
                                self.successful_subscriptions_count += 1
                                print(f"  [API] Successfully fetched price for {symbol} via API: {price}")
                                if self.on_message_callback:
                                    # Notify callback that data came from API
                                    self.on_message_callback({'event': 'price', 'symbol': symbol, 'price': price, 'source': 'API'})
                            else:
                                print(f"  [API] Failed to get price for {symbol} via API. Error type: {error_type}. Will retry.")
                                self.failed_subscriptions[symbol] = {'last_failure_time': now, 'error_type': error_type} # Update last failure time and error type
                                if error_type == 'rate_limit':
                                    self._rate_limit_pause_until = now + timedelta(minutes=1) # Set global pause
                                    print(f"  [API] Global rate limit triggered. Pausing all API fallbacks until {self._rate_limit_pause_until.strftime('%H:%M:%S UTC')}.")
                                # Keep it in failed_subscriptions to be picked up by API retry again
                        else:
                            # If already in API fallback, just ensure its price is up-to-date
                            price, error_type = self.twelvedata_api.get_price(symbol)
                            if price is not None:
                                self.latest_prices[symbol] = price
                                print(f"  [API] Refreshed price for {symbol} via API: {price}")
                                if self.on_message_callback:
                                    self.on_message_callback({'event': 'price', 'symbol': symbol, 'price': price, 'source': 'API_REFRESH'})
                            else:
                                print(f"  [API] Failed to refresh price for {symbol} via API. Error type: {error_type}. Will retry.")
                                self.failed_subscriptions[symbol] = {'last_failure_time': now, 'error_type': error_type} # Update last failure time and error type
                                if error_type == 'rate_limit':
                                    self._rate_limit_pause_until = now + timedelta(minutes=1) # Set global pause
                                    print(f"  [API] Global rate limit triggered. Pausing all API fallbacks until {self._rate_limit_pause_until.strftime('%H:%M:%S UTC')}.")
                                # Keep it in failed_subscriptions to be picked up by API retry again
                
                # Also, for any symbols that are still in subscribed_symbols but not in latest_prices
                # and not in self.failed_subscriptions, attempt a WS subscribe again.
                # This handles cases where WS might have dropped a symbol without explicitly failing.
                symbols_to_re_ws_subscribe = []
                for symbol in self.subscribed_symbols:
                    if symbol not in self.latest_prices and symbol not in self.failed_subscriptions and symbol not in self.api_fallback_symbols:
                        symbols_to_re_ws_subscribe.append(symbol)
                
                if symbols_to_re_ws_subscribe:
                    print(f"  [WS] Re-attempting WebSocket subscription for potentially dropped symbols: {', '.join(symbols_to_re_ws_subscribe)}")
                    self._send_subscribe_message(symbols_to_re_ws_subscribe)
                    for symbol in symbols_to_re_ws_subscribe:
                        self.failed_subscriptions[symbol] = {'last_failure_time': now, 'error_type': 'ws_re_subscribe'} # Mark as failed for now, will be removed on success

    def _start_retry_thread(self):
        if self._retry_thread is None or not self._retry_thread.is_alive():
            self._retry_thread = threading.Thread(target=self._retry_failed_subscriptions_loop)
            self._retry_thread.daemon = True
            self._retry_thread.start()

    def _stop_retry_thread(self):
        if self._retry_thread and self._retry_thread.is_alive():
            # No direct way to stop a daemon thread, rely on main thread exit or is_connected flag
            pass # The loop condition `while self.is_connected` will handle stopping

    def disconnect(self):
        if self.ws:
            print("Disconnecting from TwelveData WebSocket...")
            self._stop_retry_thread() # Ensure retry thread is stopped before closing WS
            self.ws.close()
            if self._connect_thread and self._connect_thread.is_alive():
                self._connect_thread.join(timeout=5) # Wait for thread to finish
            print("TwelveData WebSocket disconnected.")

    def _send_subscribe_message(self, symbols: List[str]):
        if not symbols:
            return

        if self.ws and self.is_connected:
            symbols_str = ",".join(symbols)
            subscribe_message = {
                "action": "subscribe",
                "params": {
                    "symbols": symbols_str
                }
            }
            print(f"  [WS] Sending subscribe message for {symbols_str}: {json.dumps(subscribe_message)}")
            self.ws.send(json.dumps(subscribe_message))
            print(f"  [WS] Attempting to subscribe to {symbols_str}...")
        else:
            print(f"  [WS] Not connected to WebSocket, cannot subscribe to {','.join(symbols)}. Will attempt to subscribe on connect.")

    def subscribe(self, symbols: List[str]):
        with self._lock:
            new_symbols_to_subscribe = []
            for symbol in symbols:
                if symbol not in self.subscribed_symbols and symbol not in self.failed_subscriptions:
                    self.subscribed_symbols.add(symbol)
                    new_symbols_to_subscribe.append(symbol)
                elif symbol in self.failed_subscriptions:
                    print(f"  [WS] {symbol} is in failed subscriptions. Will be retried periodically.")
                else:
                    print(f"  [WS] Already subscribed to {symbol}.")
            
            if new_symbols_to_subscribe and self.is_connected:
                self._send_subscribe_message(new_symbols_to_subscribe)
            elif new_symbols_to_subscribe:
                print(f"  [WS] Added {','.join(new_symbols_to_subscribe)} to pending subscriptions. Will subscribe on connect.")

    def unsubscribe(self, symbols: List[str]):
        with self._lock:
            symbols_to_unsubscribe = []
            for symbol in symbols:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.remove(symbol)
                    self.successful_subscriptions_count -= 1 # Decrement count on unsubscribe
                    symbols_to_unsubscribe.append(symbol)
                elif symbol in self.failed_subscriptions:
                    del self.failed_subscriptions[symbol] # Also remove from failed list if explicitly unsubscribed
                    print(f"  [WS] Removed {symbol} from failed subscriptions list.")
                elif symbol in self.api_fallback_symbols:
                    self.api_fallback_symbols.remove(symbol) # Remove from API fallback list if explicitly unsubscribed
                    print(f"  [WS] Removed {symbol} from API fallback list.")
                else:
                    print(f"  [WS] Not subscribed to {symbol}.")
            
            if symbols_to_unsubscribe and self.ws and self.is_connected:
                unsubscribe_message = {
                    "action": "unsubscribe",
                    "params": {
                        "symbols": ",".join(symbols_to_unsubscribe)
                    }
                }
                self.ws.send(json.dumps(unsubscribe_message))
                print(f"  [WS] Unsubscribed from {','.join(symbols_to_unsubscribe)}.")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self.latest_prices.get(symbol)

    def get_successful_subscriptions_count(self) -> int:
        with self._lock:
            return self.successful_subscriptions_count

if __name__ == "__main__":
    # Example Usage:
    # Set your TwelveData API key as an environment variable or replace directly
    api_key = os.environ.get("TWELVEDATA_API_KEY")

    if not api_key:
        print("Please set the TWELVEDATA_API_KEY environment variable.")
    else:
        def my_custom_callback(data):
            print(f"Custom callback received: {data}")

        # Subscribe to some symbols
        symbols_to_subscribe = ["AAPL", "MSFT", "GOOGL"]
        ws_client = TwelveDataWebSocketClient(api_key, symbols_to_subscribe=symbols_to_subscribe, on_message_callback=my_custom_callback)
        ws_client.connect()

        print("\nFetching latest prices for a few seconds...")
        for _ in range(10):
            for symbol in symbols_to_subscribe:
                price = ws_client.get_latest_price(symbol)
                if price:
                    print(f"  Latest price for {symbol}: {price}")
                else:
                    print(f"  No price yet for {symbol}")
            time.sleep(1)

        # Unsubscribe from one symbol
        ws_client.unsubscribe(["MSFT"])
        print("\nUnsubscribed from MSFT. Still fetching prices for AAPL and GOOGL...")
        for _ in range(5):
            for symbol in symbols_to_subscribe:
                price = ws_client.get_latest_price(symbol)
                if price:
                    print(f"  Latest price for {symbol}: {price}")
                else:
                    print(f"  No price yet for {symbol}")
            time.sleep(1)

        ws_client.disconnect()
