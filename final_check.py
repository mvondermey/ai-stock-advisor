#!/usr/bin/env python3
"""
Final check of the unpacking error fix
"""

def final_check():
    with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'r') as f:
        lines = f.readlines()
    
    # Find the return statement
    for i, line in enumerate(lines):
        if 'return total_portfolio_value' in line and i > 5390:
            # Count variables in this line
            var_count = line.count(',') + 1
            print(f'✅ Return line {i+1}: {var_count} variables')
            print(f'✅ Expected: 113')
            print(f'✅ Status: {"FIXED" if var_count == 113 else "NEEDS FIX"}')
            
            # Check for the missing variable
            if 'ratio_3m_1y_transaction_costs_1y' in line:
                print('✅ ratio_3m_1y_transaction_costs_1y is present')
            else:
                print('❌ ratio_3m_1y_transaction_costs_1y is missing')
            return var_count == 113
    return False

if __name__ == "__main__":
    final_check()
