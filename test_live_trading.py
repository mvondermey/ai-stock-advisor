#!/usr/bin/env python3
"""
Test script for live trading setup
Tests the Risk-Adjusted Momentum selection logic
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.live_trading import run_live_trading

if __name__ == "__main__":
    print("🧪 Testing Live Trading Setup")
    print("=" * 50)
    
    # Test the live trading function (will run in DRY-RUN mode by default)
    run_live_trading()
