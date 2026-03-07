"""
Elite Hybrid Strategy Improvements
Based on Risk-Adj Mom's superior performance
"""

def improved_elite_hybrid_scoring(momentum_6m, volatility, perf_1y, perf_3m, volume):
    """
    Improved Elite Hybrid scoring inspired by Risk-Adj Mom success
    
    Key improvements:
    1. Use sqrt(volatility) like Risk-Adj Mom
    2. Add volume confirmation
    3. Better dip ratio handling
    4. Risk-adjusted momentum focus
    """
    
    # 1. Risk-adjusted momentum (Risk-Adj Mom style)
    annualized_6m = momentum_6m * 2  # 6M -> 1Y
    risk_adj_momentum = annualized_6m / ((volatility**0.5) + 0.001)
    
    # 2. Volume confirmation (missing in original)
    volume_score = min(volume / 1000000, 2.0)  # Cap at 2x bonus
    
    # 3. Improved dip ratio (more stable)
    if perf_3m > 0 and perf_1y > 0:
        dip_ratio = (perf_1y - perf_3m) / perf_3m
        dip_ratio = max(min(dip_ratio, 3.0), 0.1)  # Tighter range
    else:
        dip_ratio = 0.1
    
    # 4. Combined score (Risk-Adj Mom inspired)
    base_score = risk_adj_momentum
    dip_bonus = 1 + dip_ratio * 0.2  # Smaller bonus
    volume_bonus = 1 + (volume_score - 1) * 0.1  # Volume confirmation
    
    improved_score = base_score * dip_bonus * volume_bonus
    
    return improved_score

# Key improvements needed:
IMPROVEMENTS = {
    "1. Volatility Penalty": "Use sqrt(volatility) instead of full volatility",
    "2. Volume Confirmation": "Add volume filter like Risk-Adj Mom", 
    "3. Dip Ratio": "Tighter range (0.1-3.0) vs (0.5-5.0)",
    "4. Risk Focus": "Prioritize risk-adjusted returns over raw momentum",
    "5. Simpler Logic": "Less complex additive scoring"
}

print("Elite Hybrid Improvements Needed:")
for key, value in IMPROVEMENTS.items():
    print(f"  {key}: {value}")
