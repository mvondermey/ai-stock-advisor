#!/usr/bin/env python3
"""Simplified Strategy Test"""
from datetime import datetime, timedelta
# Mock data for AAPL - prices increasing steadily
aapl_data = {
    'Close': [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    'Volume': [1000000, 1200000, 1100000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
}
def risk_adj_mom_score(data):
    """Simplified risk-adjusted momentum calculation"""
    # 1-year return
    start_price = data['Close'][0]
    end_price = data['Close'][-1]
    returns = (end_price - start_price) / start_price * 100
    
    # Volatility (average daily return percentage)
    daily_returns = []
    for i in range(1, len(data['Close'])):
        daily_return = (data['Close'][i] - data['Close'][i-1]) / data['Close'][i-1]
        daily_returns.append(daily_return)
    
    avg_daily_return = sum(daily_returns) / len(daily_returns)
    volatility = avg_daily_return * 100  # Convert to percentage
    
    # Risk-adjusted score (return divided by square root of volatility)
    score = returns / (abs(volatility)**0.5 + 0.001)
    return score, returns, volatility
# Calculate and display results
score, return_pct, volatility = risk_adj_mom_score(aapl_data)
print(f"AAPL Risk-Adjusted Score: {score:.2f}")
print(f"Return: {return_pct:.1f}%")
print(f"Volatility: {volatility:.1f}%")