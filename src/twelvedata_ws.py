import websocket
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, List # Added List

class TwelveDataWebSocketClient:
    def __init__(self, api_key: str, symbols_to_subscribe: Optional[List[str]] = None, on_message_callback: Optional[Callable[[Dict], None]] = None):
        self.api_key = api_key
        self.ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.latest_prices: Dict[str, float] = {} # Stores latest price for each subscribed symbol
        self.subscribed_symbols: set = set(symbols_to_subscribe) if symbols_to_subscribe else set() # Initialize with symbols
        self.successful_subscriptions_count: int = 0 # New: Counter for successful subscriptions
        self.failed_subscriptions: Dict[str, datetime] = {} # Stores symbols that failed to subscribe and their last failure time
        self.on_message_callback = on_message_callback
        self._lock = threading.Lock()
        self._connect_thread: Optional[threading.Thread] = None
        self._retry_thread: Optional[threading.Thread] = None # New: Thread for retrying failed subscriptions
        self.retry_interval_seconds = 300 # Retry failed subscriptions every 5 minutes

    def _on_open(self, ws):
        self.is_connected = True
        print(f"✅ TwelveData WebSocket connection opened.")
        with self._lock:
            # Attempt to subscribe to initial symbols and any previously failed ones
            # Reset successful count on new connection
            self.successful_subscriptions_count = 0 
            all_symbols_to_attempt = list(self.subscribed_symbols) + list(self.failed_subscriptions.keys())
            for symbol in all_symbols_to_attempt:
                self._send_subscribe_message(symbol)
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
                # print(f"  [WS] Received price for {symbol}: {price}")
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
                        self.subscribed_symbols.add(symbol) # Ensure it's in subscribed list
                        self.successful_subscriptions_count += 1 # Increment successful count
            if data.get('fails'):
                for f in data.get('fails', []):
                    symbol = f.get('symbol')
                    reason = f.get('reason', 'Unknown reason')
                    print(f"  [WS] Failed to subscribe to {symbol}: {reason}")
                    with self._lock:
                        self.failed_subscriptions[symbol] = datetime.now(timezone.utc) # Add to failed_subscriptions for retry
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
            on_close=self._on_close
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
            time.sleep(2) 

    def _retry_failed_subscriptions_loop(self):
        while self.is_connected:
            time.sleep(self.retry_interval_seconds)
            with self._lock:
                now = datetime.now(timezone.utc)
                symbols_to_retry = []
                for symbol, last_failure_time in list(self.failed_subscriptions.items()):
                    # Only retry if retry_interval_seconds has passed since last failure
                    if (now - last_failure_time).total_seconds() >= self.retry_interval_seconds:
                        symbols_to_retry.append(symbol)
                
                if symbols_to_retry:
                    print(f"  [WS] Retrying failed subscriptions for: {', '.join(symbols_to_retry)}")
                    for symbol in symbols_to_retry:
                        self._send_subscribe_message(symbol)
                        # Update last failure time to prevent immediate re-retry if it fails again
                        self.failed_subscriptions[symbol] = now 

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

    def _send_subscribe_message(self, symbol: str):
        if self.ws and self.is_connected:
            subscribe_message = {
                "action": "subscribe",
                "params": {
                    "symbols": symbol
                }
            }
            self.ws.send(json.dumps(subscribe_message))
            time.sleep(0.6) # Increased delay to respect 100 events per minute limit (1 event per 0.6 seconds)
        else:
            print(f"  [WS] Not connected to WebSocket, cannot subscribe to {symbol}. Will attempt to subscribe on connect.")

    def subscribe(self, symbols: List[str]):
        with self._lock:
            for symbol in symbols:
                # Add to subscribed_symbols if not already there, and not currently in failed_subscriptions
                if symbol not in self.subscribed_symbols and symbol not in self.failed_subscriptions:
                    self.subscribed_symbols.add(symbol)
                    if self.is_connected: # Only send subscribe message if already connected
                        self._send_subscribe_message(symbol)
                    else:
                        print(f"  [WS] Added {symbol} to pending subscriptions. Will subscribe on connect.")
                elif symbol in self.failed_subscriptions:
                    # If it's in failed_subscriptions, it means we've tried before.
                    # We don't re-add it to subscribed_symbols here, as the retry thread will handle it.
                    print(f"  [WS] {symbol} is in failed subscriptions. Will be retried periodically.")
                else:
                    print(f"  [WS] Already subscribed to {symbol}.")

    def unsubscribe(self, symbols: List[str]):
        with self._lock:
            for symbol in symbols:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.remove(symbol)
                    self.successful_subscriptions_count -= 1 # Decrement count on unsubscribe
                    if self.ws and self.is_connected:
                        unsubscribe_message = {
                            "action": "unsubscribe",
                            "params": {
                                "symbols": symbol
                            }
                        }
                        self.ws.send(json.dumps(unsubscribe_message))
                        print(f"  [WS] Unsubscribed from {symbol}.")
                elif symbol in self.failed_subscriptions:
                    del self.failed_subscriptions[symbol] # Also remove from failed list if explicitly unsubscribed
                    print(f"  [WS] Removed {symbol} from failed subscriptions list.")
                else:
                    print(f"  [WS] Not subscribed to {symbol}.")

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
