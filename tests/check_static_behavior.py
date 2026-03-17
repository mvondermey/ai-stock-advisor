#!/usr/bin/env python3
"""Detect potential static behavior patterns in strategy code."""
import sys
import os
import re

# Look for patterns that indicate static behavior
static_patterns = [
    r'max\(latest_dates\)',  # Using max date instead of current_date
    r'current_date is None',  # Not handling current_date properly
    r'timedelta\(days=365\)',  # Hardcoded periods (should use constants)
    r'timedelta\(days=90\)',
    r'timedelta\(days=21\)',
]

issues_found = []

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in static_patterns:
                        if re.search(pattern, line):
                            issues_found.append(f'{filepath}:{i}: {line.strip()}')

if issues_found:
    print('⚠️  Potential static behavior patterns found:')
    for issue in issues_found[:10]:  # Limit output
        print(f'   {issue}')
    if len(issues_found) > 10:
        print(f'   ... and {len(issues_found) - 10} more')
    print('🔍 Review these patterns to ensure rolling windows work correctly')
else:
    print('✅ No obvious static behavior patterns detected')
