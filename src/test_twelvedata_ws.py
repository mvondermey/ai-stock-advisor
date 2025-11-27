import os
import time
from typing import Dict
from src.twelvedata_ws import TwelveDataWebSocketClient # Corrected import
import time
import os

# Global variable to track if API fallback was observed
api_fallback_observed = False

def on_message_callback(data: Dict):
    """Custom callback function to process received WebSocket messages."""
    global api_fallback_observed
    print(f"Custom Callback: Received data: {data}")
    if data.get('source') == 'API':
        api_fallback_observed = True
        print(f"!!! API Fallback observed for {data.get('symbol')} !!!")

def run_websocket_test():
    """
    Runs a standalone test for the TwelveData WebSocket client,
    including testing the API fallback mechanism.
    """
    global api_fallback_observed
    api_fallback_observed = False # Reset for each test run

    api_key = os.environ.get("TWELVEDATA_API_KEY")

    if not api_key:
        print("Please set the TWELVEDATA_API_KEY environment variable.")
        return

    # Use a mix of real symbols and a fictitious one to force API fallback
    symbols_to_test = ["AAPL", "MSFT", "GOOGL", "FAILSYM"] 
    print(f"Attempting to subscribe to: {', '.join(symbols_to_test)}")

    ws_client = TwelveDataWebSocketClient(
        api_key=api_key,
        symbols_to_subscribe=symbols_to_test,
        on_message_callback=on_message_callback
    )

    print("Connecting to TwelveData WebSocket...")
    ws_client.connect()

    # Wait for connection and initial subscriptions
    # Give enough time for API fallback to kick in (after one retry interval)
    test_duration = ws_client.retry_interval_seconds + 30 # 5 minutes + 30 seconds buffer
    print(f"\nMonitoring WebSocket and API fallback for {test_duration} seconds...")
    
    start_time = time.time()
    while time.time() - start_time < test_duration:
        time.sleep(5) # Check every 5 seconds
        successful_count = ws_client.get_successful_subscriptions_count()
        print(f"Current successful subscriptions: {successful_count} / {len(symbols_to_test)}")
        print(f"Latest prices: {ws_client.latest_prices}")
        print(f"Failed subscriptions: {ws_client.failed_subscriptions}")
        print(f"API fallback symbols: {ws_client.api_fallback_symbols}")

        if "FAILSYM" in ws_client.api_fallback_symbols and api_fallback_observed:
            print("\nSUCCESS: 'FAILSYM' detected in API fallback and callback received API data!")
            break
    else:
        print("\nWARNING: 'FAILSYM' was not detected in API fallback or API data was not received via callback within the test duration.")

    print("\nDisconnecting from TwelveData WebSocket.")
    ws_client.disconnect()
    print("Test finished.")

if __name__ == "__main__":
    run_websocket_test()
