import requests
import json

class SwissquoteAPI:
    def __init__(self, base_url, access_token):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def _make_request(self, method, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=data)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, data=json.dumps(data))
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status() # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response content: {response.text}")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            return None
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected error occurred: {req_err}")
            return None

    def get_open_orders(self):
        """
        Returns all open orders to which the user of the API has access to.
        """
        return self._make_request("GET", "/orders")

    def get_order_by_id(self, client_order_id):
        """
        Returns a single security order by its clientOrderId.
        """
        return self._make_request("GET", f"/orders/{client_order_id}")

    def place_new_order(self, order_details):
        """
        Posts a new order.
        order_details should be a dictionary conforming to the API's schema for a new order.
        """
        return self._make_request("POST", "/orders", data=order_details)

    def cancel_order(self, client_order_id):
        """
        Cancellation of a specific order.
        """
        return self._make_request("DELETE", f"/orders/{client_order_id}")

# Example Usage (for testing purposes, replace with actual token and order details)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual base URL and access token
    # The sandbox URL is: https://bankingapi.simulator.swissquote.ch/ow-trading/api/v1
    # The production URL is: https://bankingapi.swissquote.ch/ow-trading/api/v1
    BASE_URL = "https://bankingapi.simulator.swissquote.ch/ow-trading/api/v1"
    ACCESS_TOKEN = "YOUR_SWISSQUOTE_OAUTH2_TOKEN" # This needs to be obtained by the user

    if ACCESS_TOKEN == "YOUR_SWISSQUOTE_OAUTH2_TOKEN":
        print("Please replace 'YOUR_SWISSQUOTE_OAUTH2_TOKEN' with your actual Swissquote OAuth2 token.")
    else:
        api = SwissquoteAPI(BASE_URL, ACCESS_TOKEN)

        # Example: Get open orders
        print("Fetching open orders...")
        open_orders = api.get_open_orders()
        if open_orders:
            print("Open Orders:", json.dumps(open_orders, indent=2))
        else:
            print("Failed to retrieve open orders.")

        # Example: Place a new order (this is a placeholder, actual details depend on API schema)
        # You would need to construct a valid order_details dictionary based on the API documentation
        # For example:
        # order_details = {
        #     "bulkOrderDetails": {
        #         "clientOrderId": "unique-order-id-123",
        #         "executionType": "market",
        #         "timeInForce": "goodTillDate",
        #         "goodTillDate": "2025-12-31",
        #         "currency": "USD",
        #         "financialInstrumentDetails": {
        #             "financialInstrumentIdentification": {
        #                 "identification": "US0378331005", # Apple ISIN
        #                 "type": "ISIN"
        #             },
        #             "placeOfTrade": {
        #                 "mic": "XNAS" # NASDAQ
        #             }
        #         },
        #         "orderQuantity": {
        #             "amount": 10,
        #             "unit": "SHARE"
        #         },
        #         "side": "BUY"
        #     },
        #     "requestedAllocationList": [
        #         {
        #             "accountId": "YOUR_ACCOUNT_ID",
        #             "allocationQuantity": {
        #                 "amount": 10,
        #                 "unit": "SHARE"
        #             }
        #         }
        #     ]
        # }
        # print("\nPlacing a new order (example)...")
        # new_order_response = api.place_new_order(order_details)
        # if new_order_response:
        #     print("New Order Response:", json.dumps(new_order_response, indent=2))
        # else:
        #     print("Failed to place new order.")

        # Example: Cancel an order (replace with a real clientOrderId)
        # print("\nCancelling an order (example)...")
        # cancel_response = api.cancel_order("some-client-order-id")
        # if cancel_response:
        #     print("Cancel Order Response:", json.dumps(cancel_response, indent=2))
        # else:
        #     print("Failed to cancel order.")
