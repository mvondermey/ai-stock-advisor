#!/usr/bin/env python3
"""
Fix model consolidation by manually converting temp models to final models.
"""

import os
import joblib
from pathlib import Path
import glob

def fix_model_consolidation():
    models_dir = Path("logs/models")
    if not models_dir.exists():
        print("ERROR: Models directory does not exist")
        return
    
    print("Fixing model consolidation...")
    
    # Get all temp model files
    temp_model_files = glob.glob(str(models_dir / "*_temp_model.joblib"))
    
    if not temp_model_files:
        print("ERROR: No temp model files found")
        return
    
    print(f"Found {len(temp_model_files)} temp model files")
    
    # Group temp models by ticker
    ticker_models = {}
    
    for temp_file in temp_model_files:
        filename = Path(temp_file).stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            ticker = '_'.join(parts[:-2])
            model_type = parts[-2]
            
            if ticker not in ticker_models:
                ticker_models[ticker] = []
            
            ticker_models[ticker].append({
                'ticker': ticker,
                'model_type': model_type,
                'temp_model_path': temp_file,
                'temp_scaler_path': temp_file.replace('_temp_model.joblib', '_temp_scaler.joblib'),
                'temp_y_scaler_path': temp_file.replace('_temp_model.joblib', '_temp_y_scaler.joblib'),
            })
    
    print(f"Grouped into {len(ticker_models)} tickers")
    
    # Model priority (LightGBM is best)
    model_priority = {
        'LightGBM': 1,
        'XGBoost': 2, 
        'RandomForest': 3,
        'TCN': 4,
        'LSTM': 5,
        'GRU': 6
    }
    
    success_count = 0
    
    for ticker, models in ticker_models.items():
        if not models:
            continue
        
        # Sort by model priority
        models.sort(key=lambda x: model_priority.get(x['model_type'], 999))
        best_model = models[0]
        
        try:
            print(f"Processing {ticker} - Best: {best_model['model_type']}")
            
            # Check if temp files exist
            if not os.path.exists(best_model['temp_model_path']):
                print(f"  WARNING: Temp model file missing")
                continue
                
            if not os.path.exists(best_model['temp_scaler_path']):
                print(f"  WARNING: Temp scaler file missing")
                continue
            
            # Load model and scaler
            model = joblib.load(best_model['temp_model_path'])
            scaler = joblib.load(best_model['temp_scaler_path'])
            
            # Load y_scaler if it exists
            y_scaler = None
            if os.path.exists(best_model['temp_y_scaler_path']):
                y_scaler = joblib.load(best_model['temp_y_scaler_path'])
            
            # Save with final naming
            final_model_path = models_dir / f"{ticker}_TargetReturn_model.joblib"
            final_scaler_path = models_dir / f"{ticker}_TargetReturn_scaler.joblib"
            final_y_scaler_path = models_dir / f"{ticker}_TargetReturn_y_scaler.joblib"
            
            print(f"  Saving to {final_model_path.name}")
            joblib.dump(model, final_model_path)
            joblib.dump(scaler, final_scaler_path)
            
            if y_scaler is not None:
                joblib.dump(y_scaler, final_y_scaler_path)
            
            success_count += 1
            
            # Clean up temp files for this ticker
            for model_info in models:
                try:
                    os.remove(model_info['temp_model_path'])
                    os.remove(model_info['temp_scaler_path'])
                    if os.path.exists(model_info['temp_y_scaler_path']):
                        os.remove(model_info['temp_y_scaler_path'])
                except:
                    pass
            
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            continue
    
    print("=" * 50)
    print(f"Successfully consolidated {success_count}/{len(ticker_models)} models")
    
    # List final models
    final_models = glob.glob(str(models_dir / "*_TargetReturn_model.joblib"))
    print(f"Final models available: {len(final_models)}")
    
    for model_file in sorted(final_models)[:5]:
        ticker = Path(model_file).stem.replace('_TargetReturn_model', '')
        print(f"  {ticker}")
    
    if len(final_models) > 5:
        print(f"  ... and {len(final_models) - 5} more")

if __name__ == "__main__":
    fix_model_consolidation()
