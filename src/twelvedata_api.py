import requests
import json
from typing import Optional, Dict, Tuple, Union

class TwelveDataAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"

    def get_quote(self, symbol: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Fetches real-time quote data for a given symbol from the TwelveData REST API.
        Returns (quote_data_dict, None) on success, or (None, error_type_string) on failure.
        error_type_string can be 'rate_limit', 'http_error', 'invalid_data', etc.
        """
        endpoint = f"{self.base_url}/quote"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            # Check if essential data fields are present
            if data and data.get('symbol') == symbol and 'close' in data:
                return data, None # Success
            else:
                # Print the full response data for better debugging
                error_message = data.get('message', 'Expected data fields missing or invalid.')
                print(f"  [API] Failed to get quote for {symbol}. Expected data not found or invalid. Full response: {json.dumps(data, indent=2)}")
                return None, 'invalid_data'
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                message = http_err.response.json().get('message', 'Rate limit exceeded.')
                print(f"  [API] HTTP 429 Rate Limit occurred for {symbol}: {message}")
                return None, 'rate_limit'
            else:
                print(f"  [API] HTTP error occurred for {symbol}: {http_err}")
                return None, 'http_error'
        except requests.exceptions.ConnectionError as conn_err:
            print(f"  [API] Connection error occurred for {symbol}: {conn_err}")
            return None, 'connection_error'
        except requests.exceptions.Timeout as timeout_err:
            print(f"  [API] Timeout error occurred for {symbol}: {timeout_err}")
            return None, 'timeout_error'
        except requests.exceptions.RequestException as req_err:
            print(f"  [API] An unexpected error occurred for {symbol}: {req_err}")
            return None, 'request_error'

    def get_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Fetches the latest price for a given symbol from the TwelveData REST API.
        Returns (price_float, None) on success, or (None, error_type_string) on failure.
        error_type_string can be 'rate_limit', 'invalid_data', 'http_error', etc.
        """
        quote_data, error_type = self.get_quote(symbol)
        
        if error_type:
            return None, error_type # Propagate error type
        
        if quote_data and 'close' in quote_data:
            try:
                return float(quote_data['close']), None # Success
            except ValueError:
                print(f"  [API] Could not convert price to float for {symbol}: {quote_data['close']}")
                return None, 'value_error'
        return None, 'unknown_error' # Should not be reached if get_quote is robust

if __name__ == "__main__":
    import os
    # Example Usage:
    api_key = os.environ.get("TWELVEDATA_API_KEY")

    if not api_key:
        print("Please set the TWELVEDATA_API_KEY environment variable.")
    else:
        td_api = TwelveDataAPI(api_key)

        symbol = "AAPL"
        print(f"Fetching quote for {symbol} via REST API...")
        quote = td_api.get_quote(symbol)
        if quote:
            print(f"  Quote for {symbol}: {json.dumps(quote, indent=2)}")
            price = td_api.get_price(symbol)
            if price:
                print(f"  Latest price for {symbol}: {price}")
        else:
            print(f"Failed to retrieve quote for {symbol}.")

        symbol = "MSFT"
        print(f"\nFetching price for {symbol} via REST API...")
        price = td_api.get_price(symbol)
        if price:
            print(f"  Latest price for {symbol}: {price}")
        else:
            print(f"Failed to retrieve price for {symbol}.")
