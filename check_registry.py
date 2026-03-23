import sys
sys.path.insert(0, 'src')
import json
import inspect

from shared_strategies import _get_strategy_registry

registry = _get_strategy_registry()

# Load live_trading_selections.json
with open('logs/live_trading_selections.json', 'r') as f:
    live_data = json.load(f)

live_strategies = live_data['strategies']

# Check what functions the working vs non-working strategies call
working = []
not_working = []

for s, tickers in live_strategies.items():
    if s in registry:
        count = len(tickers) if isinstance(tickers, list) else 0
        if count > 0:
            working.append(s)
        else:
            not_working.append(s)

# Compare the lambda bytecode to see what functions they call
print("Checking lambda closures:")
print()

# Get the closure variables for a few strategies
test_pairs = [
    ('static_bh_1y', 'dynamic_bh_1y_trailing_stop'),  # Both should call select_top_performers
    ('risk_adj_mom', 'risk_adj_mom_6m_monthly'),  # Both should call select_risk_adj_mom_stocks
]

for working_s, not_working_s in test_pairs:
    w_func = registry.get(working_s)
    nw_func = registry.get(not_working_s)
    print(f"{working_s} (works): {w_func}")
    print(f"{not_working_s} (empty): {nw_func}")
    print()
