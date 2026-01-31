#!/usr/bin/env python3
"""Analyze user holdings against strategy picks."""

# Your holdings (converted from ISINs)
your_holdings = ['AFRM', 'DHI', 'DASH', 'GE', 'HLT', 'NKE', 'CHTR', 'NVDA', 'PANW', 'PEP', 
                 'RIO', 'RCL', 'AVGO', 'SEM', 'SPY', 'STX', 'WFC', 'ENB', 'DSCSY', 'FTNT']

# Current strategy selections (from your last run)
risk_adj_mom_picks = ['STX', 'MU', 'NDX1.DE', 'HOT.DE', 'RHM.DE', 'NEM', 'TKA.DE', 'PLTR', 'SLV', 'GBF.DE']
dynamic_bh_1y_picks = ['STX', 'MU', 'HOT.DE', 'NDX1.DE', 'NEM', 'SLV', 'RHM.DE', 'GBF.DE', 'TKA.DE', 'PLTR']

print('=' * 70)
print('PORTFOLIO ANALYSIS - YOUR HOLDINGS vs STRATEGY PICKS')
print('=' * 70)

# Check overlap
in_risk_adj = [t for t in your_holdings if t in risk_adj_mom_picks]
in_dynamic_bh = [t for t in your_holdings if t in dynamic_bh_1y_picks]
not_in_any = [t for t in your_holdings if t not in risk_adj_mom_picks and t not in dynamic_bh_1y_picks]

print(f'\nIN risk_adj_mom picks ({len(in_risk_adj)}): {in_risk_adj}')
print(f'IN dynamic_bh_1y picks ({len(in_dynamic_bh)}): {in_dynamic_bh}')
print(f'\nNOT in any strategy ({len(not_in_any)}): {not_in_any}')

print('\n' + '=' * 70)
print('RECOMMENDATION SUMMARY')
print('=' * 70)
keep_list = list(set(in_risk_adj + in_dynamic_bh))
print(f'\n[KEEP] In strategy ({len(keep_list)}): {keep_list}')
print(f'[SELL] Not in strategy ({len(not_in_any)}): {not_in_any}')
print(f'\n[BUY] Missing from your portfolio (strategy wants):')
missing = [t for t in risk_adj_mom_picks if t not in your_holdings]
print(f'   {missing}')
