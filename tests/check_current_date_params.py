#!/usr/bin/env python3
"""Check for strategy function calls missing current_date parameter."""
import sys
import os
import re

sys.path.append('src')

# Check for strategy function calls without current_date
pattern = r'select_\w+_stocks\([^,)]+(?!,\s*current_date)'
files_to_check = []

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
                if re.search(pattern, content):
                    files_to_check.append(filepath)

if files_to_check:
    print('❌ Missing current_date parameter detected in:')
    for f in files_to_check:
        print(f'   {f}')
    sys.exit(1)
else:
    print('✅ All strategy calls include current_date parameter')
