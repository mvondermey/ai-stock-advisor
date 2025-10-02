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
        self.on_message_callback = on_message_callback
        self._lock = threading.Lock()
        self._connect_thread: Optional[threading.Thread] = None

    def _on_open(self, ws):
        self.is_connected = True
        print(f"✅ TwelveData WebSocket connection opened.")
        with self._lock:
            for symbol in list(self.subscribed_symbols): # Subscribe/Resubscribe on connect/reconnect
                self._send_subscribe_message(symbol)

    def _on_message(self, ws, message):
        data = json.loads(message)
        if data.get('event') == 'price':
            symbol = data.get('symbol')
            price = data.get('price')
            if symbol and price is not None:
                with self._lock:
                    self.latest_prices[symbol] = float(price)
                # print(f"  [WS] Received price for {symbol}: {price}")
                if self.on_message_callback:
                    self.on_message_callback(data)
        elif data.get('event') == 'subscribe-status' and data.get('status') == 'ok':
            print(f"  [WS] Successfully subscribed to {data.get('symbol')}")
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

    def disconnect(self):
        if self.ws:
            print("Disconnecting from TwelveData WebSocket...")
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
            time.sleep(0.1) # Add a small delay to respect rate limits
        else:
            print(f"  [WS] Not connected to WebSocket, cannot subscribe to {symbol}. Will attempt to subscribe on connect.")

    def subscribe(self, symbols: List[str]):
        with self._lock:
            for symbol in symbols:
                if symbol not in self.subscribed_symbols:
                    self.subscribed_symbols.add(symbol)
                    if self.is_connected: # Only send subscribe message if already connected
                        self._send_subscribe_message(symbol)
                    else:
                        print(f"  [WS] Added {symbol} to pending subscriptions. Will subscribe on connect.")
                else:
                    print(f"  [WS] Already subscribed to {symbol}.")

    def unsubscribe(self, symbols: List[str]):
        with self._lock:
            for symbol in symbols:
                if symbol in self.subscribed_symbols:
                    self.subscribed_symbols.remove(symbol)
                    if self.ws and self.is_connected:
                        unsubscribe_message = {
                            "action": "unsubscribe",
                            "params": {
                                "symbols": symbol
                            }
                        }
                        self.ws.send(json.dumps(unsubscribe_message))
                        print(f"  [WS] Unsubscribed from {symbol}.")
                else:
                    print(f"  [WS] Not subscribed to {symbol}.")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self.latest_prices.get(symbol)

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
