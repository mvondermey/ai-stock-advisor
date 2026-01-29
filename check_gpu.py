#!/usr/bin/env python3
"""Check GPU availability for AI Stock Advisor."""
import sys
sys.path.insert(0, 'src')

print("=" * 60)
print("GPU STATUS CHECK")
print("=" * 60)

# Check PyTorch CUDA
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ PyTorch cannot see GPU - check CUDA installation")
except ImportError:
    print("PyTorch: NOT INSTALLED")
    cuda_available = False

# Check XGBoost GPU
try:
    import xgboost as xgb
    print(f"\nXGBoost version: {xgb.__version__}")
    # XGBoost GPU test
    try:
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
        model.fit(X, y)
        print(f"  XGBoost GPU: ✅ WORKING")
    except Exception as e:
        print(f"  XGBoost GPU: ❌ NOT WORKING - {e}")
except ImportError:
    print("XGBoost: NOT INSTALLED")

# Check LightGBM GPU
try:
    import lightgbm as lgb
    print(f"\nLightGBM version: {lgb.__version__}")
    try:
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        model = lgb.LGBMClassifier(device='gpu', n_estimators=2, verbose=-1)
        model.fit(X, y)
        print(f"  LightGBM GPU: ✅ WORKING")
    except Exception as e:
        print(f"  LightGBM GPU: ❌ NOT WORKING - {str(e)[:100]}")
except ImportError:
    print("LightGBM: NOT INSTALLED")

# Check config settings
print("\n" + "=" * 60)
print("CONFIG SETTINGS")
print("=" * 60)
try:
    from config import CUDA_AVAILABLE, XGBOOST_USE_GPU, PYTORCH_USE_GPU
    print(f"CUDA_AVAILABLE: {CUDA_AVAILABLE}")
    print(f"XGBOOST_USE_GPU: {XGBOOST_USE_GPU}")
    print(f"PYTORCH_USE_GPU: {PYTORCH_USE_GPU}")
    
    if XGBOOST_USE_GPU and CUDA_AVAILABLE:
        print("\n✅ AI Portfolio training SHOULD use GPU (XGBoost/LightGBM)")
    else:
        print("\n⚠️ AI Portfolio training will use CPU")
        if not CUDA_AVAILABLE:
            print("   Reason: CUDA not available")
        if not XGBOOST_USE_GPU:
            print("   Reason: XGBOOST_USE_GPU is False")
except Exception as e:
    print(f"Could not load config: {e}")

print("\n" + "=" * 60)
