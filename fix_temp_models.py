#!/usr/bin/env python3
"""
Fix temporary model naming issue.
Renames *_temp_*.joblib files to final names.
"""

import os
import glob
from pathlib import Path

def fix_temp_models():
    """Fix temporary model files by renaming them to final names."""
    
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print("❌ models directory not found")
        return
    
    print("🔧 Fixing temporary model names...")
    
    # Find all temp model files
    temp_files = list(models_dir.glob("*_temp_*.joblib"))
    
    if not temp_files:
        print("✅ No temporary model files found")
        return
    
    print(f"📁 Found {len(temp_files)} temporary model files")
    
    renamed_count = 0
    for temp_file in temp_files:
        # Create final filename by removing "_temp"
        final_name = temp_file.name.replace("_temp_", "_")
        final_file = models_dir / final_name
        
        # Skip if final file already exists
        if final_file.exists():
            print(f"⚠️  Skipping {temp_file.name} - final file already exists")
            continue
        
        # Rename temp to final
        try:
            temp_file.rename(final_file)
            print(f"✅ Renamed: {temp_file.name} → {final_name}")
            renamed_count += 1
        except Exception as e:
            print(f"❌ Failed to rename {temp_file.name}: {e}")
    
    print(f"\n🎉 Successfully renamed {renamed_count} model files")
    
    # Show summary
    final_files = list(models_dir.glob("*_model.joblib"))
    print(f"📊 Total final model files: {len(final_files)}")

if __name__ == "__main__":
    fix_temp_models()
