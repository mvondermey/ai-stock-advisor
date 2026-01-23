#!/usr/bin/env python3

# Test to show how cash utilization will appear in the final summary
def test_cash_utilization_display():
    print("Testing Cash Utilization Display:")
    print("=" * 60)
    
    # Simulate different portfolio scenarios
    initial_capital = 30000.0
    
    scenarios = [
        # (strategy_name, final_value, description)
        ("Static BH", 42355, "Full investment - all capital deployed"),
        ("Dyn BH 3M", 74483, "Full investment - all capital deployed"),
        ("Risk-Adj Mom", 84615, "Full investment - all capital deployed"),
        ("Quality+Mom", 30000, "No stocks selected - 100% cash"),
        ("Mean Reversion", 35, "Almost no investment - 99.9% cash"),
        ("AI Portfolio", 0, "Complete failure - 100% cash"),
        ("3M/1Y Ratio", 17485, "Partial investment - some cash uninvested"),
        ("Sector Rotation", 12000, "Partial investment - significant cash"),
    ]
    
    def calculate_cash_utilization(final_value, initial_capital):
        """Calculate cash utilization as percentage of capital invested."""
        if final_value is None or initial_capital == 0:
            return "N/A"
        utilization = (final_value / initial_capital) * 100
        return f"{utilization:.1f}%"
    
    print(f"📊 Initial Capital: ${initial_capital:,.2f}")
    print(f"📊 Portfolio Size Target: 10 stocks")
    print()
    
    # Show the cash utilization row format
    print("🎋 CASH UTILIZATION ROW EXAMPLE:")
    print("-" * 40)
    
    cash_util_row = "Cash Util    | "
    for strategy_name, final_value, description in scenarios:
        cash_util = calculate_cash_utilization(final_value, initial_capital)
        cash_util_row += f"{cash_util:<17} | "
    
    # Trim the last " | "
    cash_util_row = cash_util_row[:-3]
    print(cash_util_row)
    
    print()
    print("📈 DETAILED ANALYSIS:")
    print("-" * 40)
    
    for strategy_name, final_value, description in scenarios:
        cash_util = calculate_cash_utilization(final_value, initial_capital)
        cash_uninvested = initial_capital - final_value
        cash_uninvested_pct = ((initial_capital - final_value) / initial_capital) * 100
        
        print(f"\n{strategy_name}:")
        print(f"   Final Value: ${final_value:,.0f}")
        print(f"   Cash Utilization: {cash_util}")
        print(f"   Uninvested Cash: ${cash_uninvested:,.0f} ({cash_uninvested_pct:.1f}%)")
        print(f"   Scenario: {description}")
        
        # Interpretation
        if cash_util == "100.0%":
            print(f"   ✅ Fully invested - optimal capital deployment")
        elif cash_util == "0.0%":
            print(f"   ❌ Complete failure - no stocks selected")
        elif float(cash_util.rstrip('%')) >= 80.0:
            print(f"   ⚠️  Good utilization - minimal cash drag")
        elif float(cash_util.rstrip('%')) >= 50.0:
            print(f"   ⚠️  Moderate utilization - some cash drag")
        else:
            print(f"   ❌ Poor utilization - significant cash drag")

if __name__ == "__main__":
    test_cash_utilization_display()
