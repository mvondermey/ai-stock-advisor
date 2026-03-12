import re

with open('src/backtesting.py', 'r') as f:
    lines = f.readlines()

# Find all if new_*_stocks: patterns and check next few lines for print
strategies_with = []
strategies_missing = []

for i, line in enumerate(lines):
    match = re.match(r'\s+if (new_\w+_stocks):', line)
    if match:
        var_name = match.group(1)
        # Check next 5 lines for print with emoji
        found_print = False
        for j in range(i+1, min(i+6, len(lines))):
            if 'print(f"' in lines[j] and ('📊' in lines[j] or '🎯' in lines[j] or '🏆' in lines[j] or '🔄' in lines[j] or '📈' in lines[j]):
                found_print = True
                break
        if found_print:
            strategies_with.append(var_name)
        else:
            strategies_missing.append(var_name)

print(f"Strategies WITH summary ({len(strategies_with)}):")
for s in sorted(strategies_with):
    print(f"  ✓ {s}")

print(f"\nStrategies MISSING summary ({len(strategies_missing)}):")
for s in sorted(strategies_missing):
    print(f"  ✗ {s}")

print(f"\nTotal: {len(strategies_with) + len(strategies_missing)}")
