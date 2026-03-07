#!/usr/bin/env python3
import re

with open("output.log", "r") as f:
    content = f.read()

# Find all daily summary sections with AI Elite
# Pattern: Day X Portfolio Summary followed by table data
sections = re.findall(r'Day (\d+) Portfolio Summary.*?Best.*?Worst.*?\n', content, re.DOTALL)

# Look for AI Elite line pattern in each day
for day_num in range(1, 100):
    # Find Day X section
    pattern = rf'Day {day_num} Portfolio Summary.*?\n.*?AI Elite.*?(\$[0-9,]+).*?([+-]?[0-9.]+)%'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        value = match.group(1)
        ret = match.group(2)
        print(f"Day {day_num}: AI Elite Value={value}, Return={ret}%")
    else:
        # Try alternative pattern
        day_section = re.search(rf'Day {day_num}.*?Portfolio Summary.*?(?=Day {day_num+1}|Daily Stock Selection|$)', content, re.DOTALL)
        if day_section:
            section = day_section.group(0)
            ai_match = re.search(r'AI Elite.*?\$([0-9,]+).*?([+-]?[0-9.]+)%', section)
            if ai_match:
                print(f"Day {day_num}: AI Elite Value=${ai_match.group(1)}, Return={ai_match.group(2)}%")
