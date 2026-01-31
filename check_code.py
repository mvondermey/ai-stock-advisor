#!/usr/bin/env python3
"""
Quick check of the current multitask_strategy.py file
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_current_code():
    """Check if the fix is properly applied."""
    
    print("🔍 Checking Current Code State")
    print("=" * 40)
    
    try:
        with open('src/multitask_strategy.py', 'r') as f:
            lines = f.readlines()
        
        # Find the prepare_data method
        for i, line in enumerate(lines):
            if 'def prepare_data' in line:
                print(f"Found prepare_data at line {i+1}")
                # Show next 30 lines
                for j in range(min(30, len(lines) - i)):
                    print(f"{i+j+1:3d}: {lines[i+j].rstrip()}")
                break
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    check_current_code()
