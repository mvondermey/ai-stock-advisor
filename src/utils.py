import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

def _ensure_dir(p: Path) -> None:
    """Ensures that a directory exists."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _normalize_symbol(symbol: str, provider: str) -> str:
    """Normalizes a ticker symbol for the given data provider."""
    s_ticker = str(symbol).strip()
    if '$' in s_ticker:
        return "" # Or handle as an invalid ticker
    
    # For now, the main normalization is for Yahoo/Stooq US tickers
    if provider.lower() in ['yahoo', 'stooq']:
        if s_ticker.endswith(('.DE', '.MI', '.SW', '.PA', '.AS', '.HE', '.LS', '.BR', '.MC')):
            return s_ticker
        else:
            return s_ticker.replace('.', '-')
    # Alpaca expects symbols without suffixes like '.US'
    if provider.lower() == 'alpaca':
        return s_ticker.replace('.', '-').split('.US')[0]
        
    return s_ticker

def _to_utc(ts):
    """Return a pandas UTC-aware Timestamp for any datetime-like input."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')
