#!/usr/bin/env python3
"""Extract and format strategy performance from the backtest results."""

# Data from the backtest results
strategies = [
    ("AI Strategy", 0.0, 0),
    ("Static BH 1Y", 3819, 294),
    ("Static BH 3M", 8603, 295),
    ("Static BH 1M", -88.1, 299),
    ("Dynamic BH 1Y", 907.6, 1811),
    ("Dynamic BH 3M", 1383, 1974),
    ("AI Portfolio", 26.4, 893),
    ("Dynamic BH 1M", 203.9, 2132),
    ("Risk-Adj Mom", 852.2, 1866),
    ("Mean Reversion", -16.2, 1469),
    ("Quality+Mom", 1150, 2348),
    ("Vol-Adj Mom", 872.6, 1642),
    ("Mom+AI Hybrid", 64.7, 261),
    ("Dynamic BH 1Y+Vol", 2718, 770),
    ("Dynamic BH 1Y+TS", 2718, 770),
    ("Multi-Task", 0.0, 0),
    ("Sector Rotation", 0.0, 0),
    ("3M/1Y Ratio", -7.7, 287),
]

# Sort by annualized return (descending)
strategies.sort(key=lambda x: x[1], reverse=True)

# Print table
print("=" * 80)
print("STRATEGY PERFORMANCE SUMMARY (Annualized Returns & Transaction Costs)")
print("=" * 80)
print(f"{'Strategy':<20} {'Annualized Return':<18} {'Transaction Cost':<15} {'Return/Cost Ratio':<18}")
print("-" * 80)

for strategy, annual_ret, txn_cost in strategies:
    if txn_cost > 0:
        ratio = annual_ret / txn_cost
        ratio_str = f"{ratio:.1f}x"
    else:
        ratio_str = "N/A"
    
    ret_str = f"+{annual_ret:.1f}%" if annual_ret >= 0 else f"{annual_ret:.1f}%"
    cost_str = f"${txn_cost:,}"
    
    print(f"{strategy:<20} {ret_str:<18} {cost_str:<15} {ratio_str:<18}")

print("-" * 80)

# Summary stats
positive_returns = [s for s in strategies if s[1] > 0]
negative_returns = [s for s in strategies if s[1] < 0]

print(f"\nSUMMARY:")
print(f"Total strategies: {len(strategies)}")
print(f"Positive returns: {len(positive_returns)}")
print(f"Negative returns: {len(negative_returns)}")
print(f"\nBest performer: {strategies[0][0]} (+{strategies[0][1]:.1f}%)")
print(f"Worst performer: {strategies[-1][0]} ({strategies[-1][1]:.1f}%)")
print(f"Average positive return: +{sum(s[1] for s in positive_returns)/len(positive_returns):.1f}%")
print(f"Average transaction cost: ${sum(s[2] for s in strategies if s[2] > 0)/len([s for s in strategies if s[2] > 0]):,.0f}")
