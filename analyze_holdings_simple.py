#!/usr/bin/env python3
"""Simple analysis of holdings based on previous strategy runs."""

# Your holdings (converted from ISINs)
your_holdings = ['AFRM', 'DHI', 'DASH', 'GE', 'HLT', 'NKE', 'CHTR', 'NVDA', 'PANW', 'PEP', 
                 'RIO', 'RCL', 'AVGO', 'SEM', 'SPY', 'STX', 'WFC', 'ENB', 'DSCSY', 'FTNT']

# Strategy picks from your last run
risk_adj_mom_picks = ['STX', 'MU', 'NDX1.DE', 'HOT.DE', 'RHM.DE', 'NEM', 'TKA.DE', 'PLTR', 'SLV', 'GBF.DE']
dynamic_bh_1y_picks = ['STX', 'MU', 'HOT.DE', 'NDX1.DE', 'NEM', 'SLV', 'RHM.DE', 'GBF.DE', 'TKA.DE', 'PLTR']

# Combined picks (union of both strategies)
all_strategy_picks = list(set(risk_adj_mom_picks + dynamic_bh_1y_picks))

print('=' * 80)
print('📊 HOLDINGS ANALYSIS')
print('=' * 80)

# Categorize holdings
in_strategies = [t for t in your_holdings if t in all_strategy_picks]
not_in_strategies = [t for t in your_holdings if t not in all_strategy_picks]

# Check which strategy each is in
in_risk_adj = [t for t in your_holdings if t in risk_adj_mom_picks]
in_dynamic_bh = [t for t in your_holdings if t in dynamic_bh_1y_picks]

print(f"\n📈 Your Holdings: {len(your_holdings)} total")
print(f"   In at least one strategy: {len(in_strategies)}")
print(f"   Not in any strategy: {len(not_in_strategies)}")

print(f"\n✅ KEEP - In strategy selections:")
print(f"   Risk-Adj Mom ({len(in_risk_adj)}): {in_risk_adj}")
print(f"   Dynamic BH 1Y ({len(in_dynamic_bh)}): {in_dynamic_bh}")

print(f"\n❌ CONSIDER SELLING - Not in any strategy ({len(not_in_strategies)}):")
for i, ticker in enumerate(not_in_strategies, 1):
    print(f"   {i:2d}. {ticker}")

print(f"\n💡 STRATEGY WANTS - Not in your portfolio:")
missing = [t for t in all_strategy_picks if t not in your_holdings]
for i, ticker in enumerate(missing, 1):
    print(f"   {i:2d}. {ticker}")

print("\n" + "=" * 80)
print("📋 ACTION PLAN")
print("=" * 80)

# Create action plan
actions = []
for ticker in your_holdings:
    if ticker in all_strategy_picks:
        actions.append(('KEEP', ticker, 'In strategy selection'))
    else:
        actions.append(('SELL', ticker, 'Not in any strategy'))

# Add missing stocks to buy
for ticker in missing:
    actions.append(('BUY', ticker, 'Strategy wants this stock'))

# Sort by action
keep_actions = [a for a in actions if a[0] == 'KEEP']
sell_actions = [a for a in actions if a[0] == 'SELL']
buy_actions = [a for a in actions if a[0] == 'BUY']

print(f"\n🟢 KEEP ({len(keep_actions)}):")
for action, ticker, reason in keep_actions:
    print(f"   {ticker} - {reason}")

print(f"\n🔴 SELL ({len(sell_actions)}):")
for action, ticker, reason in sell_actions:
    print(f"   {ticker} - {reason}")

print(f"\n🟡 BUY ({len(buy_actions)}):")
for action, ticker, reason in buy_actions:
    print(f"   {ticker} - {reason}")

print(f"\n📊 SUMMARY:")
print(f"   Portfolio size: {len(your_holdings)} → {len(keep_actions) + len(buy_actions)}")
print(f"   Trades needed: {len(sell_actions)} sells + {len(buy_actions)} buys = {len(sell_actions) + len(buy_actions)} total")
