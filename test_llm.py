#!/usr/bin/env python3
"""Test LLM response parsing."""
import requests
import json
import re

test_summary = """Ticker: AAPL
Current Price: $185.50

RETURNS:
- 5-day: +1.2%
- 20-day (1 month): +3.2%
- 60-day (3 months): +8.5%
- 1-year: +25.3%

TECHNICAL INDICATORS:
- RSI (14): 62
- Trend: strong uptrend (SMA10 > SMA20 > SMA50)
- Price vs SMA20: above
- Price vs SMA50: above
- Volatility (annualized): 22.5%

PRICE LEVELS:
- 52-week high: $199.62 (-7.1% from current)
- 52-week low: $142.00 (+30.6% from current)
- Position in 20d range: 75% (0=low, 100=high)

VOLUME:
- Average daily volume: 52.3M
- Recent volume trend: stable"""

prompt = f"""You are a quantitative stock analyst. Analyze this stock and provide a prediction score.

{test_summary}

Based on the data above, rate this stock's expected performance over the next 10 trading days.
Respond with ONLY a JSON object in this exact format:
{{"score": <number between -1 and 1>, "reasoning": "<brief 1-sentence explanation>"}}

Respond with ONLY the JSON, no other text."""

print("Sending request to Ollama...")
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300}
    },
    timeout=120
)

if response.status_code == 200:
    text = response.json().get("response", "")
    print("RAW RESPONSE:")
    print("-" * 50)
    print(text[:1500])
    print("-" * 50)
    
    # Try parsing
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    print("\nCLEANED (no think tags):")
    print(text_clean[:500])
    
    if '{' in text_clean and '}' in text_clean:
        json_start = text_clean.index('{')
        json_end = text_clean.rindex('}') + 1
        json_str = text_clean[json_start:json_end]
        print("\nEXTRACTED JSON:")
        print(json_str)
        try:
            parsed = json.loads(json_str)
            print("\nPARSED:", parsed)
        except json.JSONDecodeError as e:
            print(f"\nJSON PARSE ERROR: {e}")
else:
    print(f"Error: {response.status_code}")
