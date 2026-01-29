#!/usr/bin/env python3
"""
Test that models can be loaded properly by the prediction system.
"""

import joblib
from pathlib import Path
import glob

def test_model_loading():
    """Test loading a few models to verify they work."""
    
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print("ERROR: Models directory does not exist")
        return
    
    print("Testing model loading...")
    
    # Get a few final models to test
    final_models = glob.glob(str(models_dir / "*_TargetReturn_model.joblib"))
    
    if not final_models:
        print("ERROR: No final models found")
        return
    
    print(f"Found {len(final_models)} final models")
    
    # Test first 5 models
    success_count = 0
    for model_file in final_models[:5]:
        ticker = Path(model_file).stem.replace('_TargetReturn_model', '')
        scaler_file = model_file.replace('_TargetReturn_model.joblib', '_TargetReturn_scaler.joblib')
        
        try:
            print(f"Testing {ticker}...")
            
            # Load model
            model = joblib.load(model_file)
            print(f"  Model loaded: {type(model)}")
            
            # Load scaler
            scaler = joblib.load(scaler_file)
            print(f"  Scaler loaded: {type(scaler)}")
            
            success_count += 1
            print(f"  SUCCESS: {ticker}")
            
        except Exception as e:
            print(f"  ERROR loading {ticker}: {e}")
    
    print("=" * 50)
    print(f"Successfully tested {success_count}/{min(5, len(final_models))} models")
    
    # Check if models are accessible to prediction system
    print("Testing prediction system access...")
    
    # Test the naming patterns that prediction.py expects
    test_ticker = final_models[0] if final_models else None
    if test_ticker:
        ticker = Path(test_ticker).stem.replace('_TargetReturn_model', '')
        
        # Test the naming patterns from prediction.py
        naming_patterns = [
            f"{ticker}_model.joblib",
            f"{ticker}_LSTM_model.joblib", 
            f"{ticker}_TargetReturn_model.joblib",
            f"{ticker}_TCN_model.joblib"
        ]
        
        print(f"Testing naming patterns for {ticker}:")
        for pattern in naming_patterns:
            test_path = models_dir / pattern
            exists = test_path.exists()
            print(f"  {pattern}: {'EXISTS' if exists else 'MISSING'}")

if __name__ == "__main__":
    test_model_loading()
