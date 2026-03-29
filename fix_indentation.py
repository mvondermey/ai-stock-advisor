#!/usr/bin/env python3
"""
Fix indentation issues in backtesting.py where except blocks are at wrong indentation level.
The problem is that except blocks are nested inside if blocks but should be at the try block level.
"""

# Read the file
with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all try blocks and their corresponding except blocks
# The issue is that except blocks have 4 extra spaces of indentation

fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this is an except line that needs to be dedented
    if line.strip().startswith('except Exception as e:'):
        # Get the current indentation
        current_indent = len(line) - len(line.lstrip())
        
        # Check if the next line is a print statement
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip().startswith('print(f"   '):
                # This is likely a misindented except block
                # Dedent both the except and print by 4 spaces
                if current_indent >= 4:
                    # Dedent the except line
                    fixed_except = ' ' * (current_indent - 4) + line.lstrip()
                    fixed_lines.append(fixed_except)
                    
                    # The print line should be at current_indent (same as original except)
                    fixed_print = ' ' * current_indent + next_line.lstrip()
                    fixed_lines.append(fixed_print)
                    i += 2
                    continue
    
    fixed_lines.append(line)
    i += 1

# Write the file back
with open('/home/mvondermey/ai-stock-advisor/src/backtesting.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print('Fixed indentation issues')
