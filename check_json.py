import json

with open('logs/live_trading_selections.json', 'r') as f:
    data = json.load(f)

strategies = data['strategies']
with_selections = sum(1 for v in strategies.values() if isinstance(v, list) and len(v) > 0)
empty_lists = sum(1 for v in strategies.values() if isinstance(v, list) and len(v) == 0)
total = len(strategies)

print(f"Strategies with selections: {with_selections}")
print(f"Strategies with empty lists: {empty_lists}")
print(f"Total strategies: {total}")

# Check specific strategies
print("\nSpecific strategies:")
print(f"vol_sweet_mom: {strategies.get('vol_sweet_mom', 'NOT FOUND')}")
print(f"bh_1y_vol_adj_rebal: {strategies.get('bh_1y_vol_adj_rebal', 'NOT FOUND')}")
print(f"bh_1y_volsweet_accel: {strategies.get('bh_1y_volsweet_accel', 'NOT FOUND')}")
