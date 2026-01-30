#!/usr/bin/env python3
"""
Check variable names
"""

def check_var():
    expected = 'ratio_3m_1y_transaction_costs_1y'
    found_in_return = 'ratio_3m_1y_transaction_costs' in open('/home/mvondermey/ai-stock-advisor/src/backtesting.py').read()
    
    print(f'Expected: {expected}')
    print(f'Found in return: ratio_3m_1y_transaction_costs')
    print(f'Match: {expected == "ratio_3m_1y_transaction_costs"}')
    print(f'Need to add _1y suffix: {"_1y" not in "ratio_3m_1y_transaction_costs"}')

if __name__ == "__main__":
    check_var()
