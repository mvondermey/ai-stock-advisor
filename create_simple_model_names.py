#!/usr/bin/env python3
"""
Create simple model names for better compatibility.
Creates {ticker}_model.joblib copies of the TargetReturn models.
"""

import os
import shutil
from pathlib import Path
import glob

def create_simple_model_names():
    """Create simple model names for better compatibility."""
    
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print("ERROR: Models directory does not exist")
        return
    
    print("Creating simple model names...")
    
    # Get all TargetReturn models
    target_models = glob.glob(str(models_dir / "*_TargetReturn_model.joblib"))
    
    if not target_models:
        print("ERROR: No TargetReturn models found")
        return
    
    print(f"Found {len(target_models)} TargetReturn models")
    
    success_count = 0
    
    for target_model in target_models:
        # Extract ticker from filename
        ticker = Path(target_model).stem.replace('_TargetReturn_model', '')
        
        # Create simple model name
        simple_model_path = models_dir / f"{ticker}_model.joblib"
        simple_scaler_path = models_dir / f"{ticker}_scaler.joblib"
        
        # Source paths
        target_scaler_path = target_model.replace('_TargetReturn_model.joblib', '_TargetReturn_scaler.joblib')
        
        try:
            print(f"Creating simple names for {ticker}...")
            
            # Copy model
            shutil.copy2(target_model, simple_model_path)
            print(f"  Model: {simple_model_path.name}")
            
            # Copy scaler
            if os.path.exists(target_scaler_path):
                shutil.copy2(target_scaler_path, simple_scaler_path)
                print(f"  Scaler: {simple_scaler_path.name}")
            
            success_count += 1
            print(f"  SUCCESS")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("=" * 50)
    print(f"Successfully created simple names for {success_count}/{len(target_models)} models")
    
    # Test the naming patterns
    test_ticker = target_models[0] if target_models else None
    if test_ticker:
        ticker = Path(test_ticker).stem.replace('_TargetReturn_model', '')
        
        naming_patterns = [
            f"{ticker}_model.joblib",
            f"{ticker}_scaler.joblib",
            f"{ticker}_TargetReturn_model.joblib",
            f"{ticker}_TargetReturn_scaler.joblib"
        ]
        
        print(f"Testing naming patterns for {ticker}:")
        for pattern in naming_patterns:
            test_path = models_dir / pattern
            exists = test_path.exists()
            print(f"  {pattern}: {'EXISTS' if exists else 'MISSING'}")

if __name__ == "__main__":
    create_simple_model_names()
