
Alpha thresholds (BUY & SELL) integration

1) Place these files under src/:
   - alpha_training.py
   - alpha_integration.py

2) In src/main.py (near imports, after class RuleTradingEnv definition or at the top if you prefer),
   add:
       from alpha_integration import enable_alpha_thresholds, USE_ALPHA_THRESHOLD_BUY, USE_ALPHA_THRESHOLD_SELL
       USE_ALPHA_THRESHOLD_BUY = True
       USE_ALPHA_THRESHOLD_SELL = True
       enable_alpha_thresholds()

   That's it. Continue to build RuleTradingEnv(...) exactly as before. When models/scaler are present,
   per_ticker_min_proba_buy / per_ticker_min_proba_sell will be auto-chosen to maximize alpha vs buy-and-hold.

3) Run as before:
       python src/main.py

4) Toggle flags to disable any side:
       USE_ALPHA_THRESHOLD_BUY = False
       USE_ALPHA_THRESHOLD_SELL = False

Notes:
- Horizon days defaults to 5; adjust ALPHA_HORIZON_DAYS in alpha_integration.py if needed.
- Costs/slippage configurable there as well.
