"""
Backtesting Phase Module
Handles portfolio backtesting for all periods (1-Year, YTD, 3-Month, 1-Month).
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd

# Import the backtesting function from backtesting.py
from backtesting import _run_portfolio_backtest

# Re-export for convenience
__all__ = ['_run_portfolio_backtest']

