#!/usr/bin/env python3
"""
Verify the unpacking error fix
"""

def verify_fix():
    with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'r') as f:
        content = f.read()
    
    # Count the return values
    import re
    return_match = re.search(r'return total_portfolio_value.*sentiment_ensemble_cash_deployed', content, re.DOTALL)
    if return_match:
        return_statement = return_match.group(0)
        var_count = return_statement.count(',') + 1
        print(f'✅ Return values count: {var_count}')
        print(f'✅ Expected: 113')
        print(f'✅ Status: {"FIXED" if var_count == 113 else "NEEDS FIX"}')
        
        # Check for the specific variable
        if 'ratio_3m_1y_transaction_costs_1y' in return_statement:
            print('✅ ratio_3m_1y_transaction_costs_1y is present')
        else:
            print('❌ ratio_3m_1y_transaction_costs_1y is missing')
            
        # Check if variable is defined
        if 'ratio_3m_1y_transaction_costs_1y = ratio_3m_1y_transaction_costs' in content:
            print('✅ Variable is defined before return')
        else:
            print('❌ Variable definition not found')
            
        return var_count == 113
    else:
        print('❌ Return statement not found')
        return False

if __name__ == "__main__":
    verify_fix()
