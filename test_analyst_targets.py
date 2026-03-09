#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
from analyst_recommendation_strategy import fetch_analyst_data, calculate_analyst_score
from datetime import datetime

data = fetch_analyst_data("AAPL")
print("Checking price target usage...")
score, actions = calculate_analyst_score(data, datetime(2026, 1, 15), 200.0, 60)
print("AAPL at $200: score=%.1f, actions=%d" % (score, actions))
score2, actions2 = calculate_analyst_score(data, datetime(2026, 1, 15), 350.0, 60)
print("AAPL at $350: score=%.1f, actions=%d" % (score2, actions2))
