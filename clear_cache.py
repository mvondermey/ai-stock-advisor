#!/usr/bin/env python3
"""
Clear corrupted data cache files
"""
import os
import glob
from pathlib import Path

cache_dir = Path("data_cache")

if cache_dir.exists():
    csv_files = list(cache_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} cache files to delete...")
    
    for csv_file in csv_files:
        try:
            csv_file.unlink()
            print(f"  ✅ Deleted {csv_file.name}")
        except Exception as e:
            print(f"  ❌ Failed to delete {csv_file.name}: {e}")
    
    print(f"\n✅ Cache cleared! Deleted {len(csv_files)} files.")
    print("💡 The system will automatically re-download fresh data on next run.")
else:
    print("❌ data_cache directory not found!")

