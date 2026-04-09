from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd


CACHE_SCHEMA_VERSION = 1


def get_cache_root() -> Path:
    return Path("logs/cache/strategy")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_cache_hash(namespace: str, key_parts: Dict[str, Any]) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "key_parts": _normalize_value(key_parts),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_dir_paths(namespace: str, key_parts: Dict[str, Any]) -> tuple[Path, Path]:
    cache_hash = build_cache_hash(namespace, key_parts)
    cache_dir = get_cache_root() / namespace / cache_hash
    metadata_path = cache_dir / "meta.json"
    return cache_dir, metadata_path


def get_cache_dir(namespace: str, key_parts: Dict[str, Any], create: bool = True) -> Path:
    cache_dir, metadata_path = _cache_dir_paths(namespace, key_parts)
    if not create:
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        metadata = {
            "schema": CACHE_SCHEMA_VERSION,
            "namespace": namespace,
            "cache_hash": build_cache_hash(namespace, key_parts),
            "key_parts": _normalize_value(key_parts),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return cache_dir


def save_joblib_cache(namespace: str, key_parts: Dict[str, Any], value: Any, filename: str = "artifact.joblib") -> Path:
    cache_dir = get_cache_dir(namespace, key_parts, create=True)
    artifact_path = cache_dir / filename
    joblib.dump(value, artifact_path)
    return artifact_path


def load_joblib_cache(namespace: str, key_parts: Dict[str, Any], filename: str = "artifact.joblib") -> Optional[Any]:
    cache_dir = get_cache_dir(namespace, key_parts, create=False)
    artifact_path = cache_dir / filename
    if not artifact_path.exists():
        return None
    try:
        return joblib.load(artifact_path)
    except Exception:
        return None


def save_json_cache(namespace: str, key_parts: Dict[str, Any], value: Any, filename: str = "artifact.json") -> Path:
    cache_dir = get_cache_dir(namespace, key_parts, create=True)
    artifact_path = cache_dir / filename
    artifact_path.write_text(json.dumps(_normalize_value(value), indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def load_json_cache(namespace: str, key_parts: Dict[str, Any], filename: str = "artifact.json") -> Optional[Any]:
    cache_dir = get_cache_dir(namespace, key_parts, create=False)
    artifact_path = cache_dir / filename
    if not artifact_path.exists():
        return None
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def dataframe_signature(data: Optional[pd.DataFrame]) -> str:
    if data is None or data.empty:
        return "empty"

    normalized = data.sort_index()
    columns = [str(column) for column in normalized.columns]

    index_values = normalized.index
    first_index = str(index_values[0]) if len(index_values) else None
    last_index = str(index_values[-1]) if len(index_values) else None

    signature_payload: Dict[str, Any] = {
        "rows": int(len(normalized)),
        "columns": columns,
        "first_index": first_index,
        "last_index": last_index,
    }

    for column_name in ("Close", "Volume", "High", "Low"):
        if column_name not in normalized.columns:
            continue
        series = pd.to_numeric(normalized[column_name], errors="coerce").dropna()
        signature_payload[f"{column_name.lower()}_count"] = int(len(series))
        if not series.empty:
            signature_payload[f"{column_name.lower()}_first"] = float(series.iloc[0])
            signature_payload[f"{column_name.lower()}_last"] = float(series.iloc[-1])

    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def universe_signature_from_frames(
    ticker_data_grouped: Dict[str, pd.DataFrame],
    tickers: Optional[list[str]] = None,
) -> str:
    selected_tickers = list(tickers) if tickers is not None else sorted(ticker_data_grouped.keys())
    payload = []
    for ticker in sorted(selected_tickers):
        frame = ticker_data_grouped.get(ticker)
        payload.append(
            {
                "ticker": ticker,
                "signature": dataframe_signature(frame),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def universe_signature_from_price_cache(
    date_ns_by_ticker: Dict[str, np.ndarray],
    close_by_ticker: Dict[str, np.ndarray],
) -> str:
    payload = []
    for ticker in sorted(close_by_ticker.keys()):
        date_ns = date_ns_by_ticker.get(ticker)
        close_values = close_by_ticker.get(ticker)
        if date_ns is None or close_values is None:
            continue
        payload.append(
            {
                "ticker": ticker,
                "date_count": int(len(date_ns)),
                "close_count": int(len(close_values)),
                "first_date": int(date_ns[0]) if len(date_ns) else None,
                "last_date": int(date_ns[-1]) if len(date_ns) else None,
                "first_close": float(close_values[0]) if len(close_values) else None,
                "last_close": float(close_values[-1]) if len(close_values) else None,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
