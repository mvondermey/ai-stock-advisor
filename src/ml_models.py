import numpy as np
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
import json
from pathlib import Path
from alpha_training import alpha_sample_weights
import joblib # For model saving/loading
import sys # For current_process in train_worker
import matplotlib.pyplot as plt # For SHAP plots


def _safe_to_cpu(model_or_tensor):
    """Safely move a PyTorch model/tensor to CPU, handling CUDA errors gracefully.
    
    Returns:
        The model/tensor on CPU, or None if failed.
    """
    import torch
    if model_or_tensor is None:
        return None
    try:
        return model_or_tensor.cpu()
    except RuntimeError as e:
        if "CUDA" in str(e):
            try:
                # Try to clear CUDA cache first
                torch.cuda.empty_cache()
                return model_or_tensor.cpu()
            except RuntimeError:
                # CUDA context is corrupted, can't recover this model
                print(f"⚠️ CUDA error moving model to CPU, model lost: {e}")
                return None
        raise

def _alpha_metrics_report(label: str, proba, df_like: pd.DataFrame, freq: str = "D") -> None:
    """Print Alpha (annualized) and Active-IR vs buy-and-hold for a probability vector aligned to df_like."""
    try:
        import numpy as np, pandas as pd, math

        def _infer_ppy(f: str) -> int:
            return {"D": 252, "W": 52, "M": 12}.get(f.upper(), 252)

        def _ols_alpha(r_s: pd.Series, r_b: pd.Series, f: str) -> float:
            s = pd.Series(r_s).dropna()
            b = pd.Series(r_b).reindex_like(s).fillna(0.0)
            if s.empty:
                return 0.0
            x = b.values
            y = s.values
            X = np.column_stack([np.ones_like(x), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[0]) * _infer_ppy(f)

        def _active_ir(active: pd.Series, f: str) -> float:
            ppy = _infer_ppy(f)
            mu = active.mean() * ppy
            vol = active.std(ddof=0) * math.sqrt(ppy)
            return float(mu / vol) if vol > 0 else 0.0

        idx = getattr(proba, "index", None)
        s_proba = pd.Series(proba, index=idx) if idx is not None else pd.Series(proba, index=df_like.index[:len(proba)])

        fut = None
        for c in ("future_ret", "Future_Returns", "future_return"):
            if c in df_like.columns:
                fut = pd.Series(df_like[c].values, index=df_like.index)
                break
        if fut is None and "Close" in df_like.columns:
            close = pd.to_numeric(df_like["Close"], errors="coerce")
            fut = (close.shift(-5) / close - 1.0)
        if fut is None:
            print(f"    - {label} alpha metrics: (skipped: cannot derive future returns)")
            return

        fut = fut.reindex(s_proba.index).fillna(0.0)
        bench = fut.copy()
        entries = (s_proba >= 0.5).astype(int)  # quick diagnostic threshold
        strat_ret = fut * entries
        active = strat_ret - bench
        alpha = _ols_alpha(strat_ret, bench, freq)
        ir = _active_ir(active, freq)
        print(f"    - {label} Alpha (annualized): {alpha:.4f} | Active-IR: {ir:.3f}")
    except Exception as _e:
        print(f"    - {label} alpha metrics: (error: {_e})")


def _alpha_fit_params(d: pd.DataFrame, X_index) -> dict | None:
    """Build sklearn fit_params with alpha-aware sample weights.

    Looks for future returns columns in `d`: 'future_ret', 'Future_Returns', or 'future_return'.
    If none are present, derive a 5-day forward return from 'Close'.
    If 'bench_future_ret' is present, use it as the benchmark; otherwise, use the asset itself.
    Returns {'sample_weight': np.ndarray} or None if not possible.
    """
    try:
        fut_col = next((c for c in ("future_ret", "Future_Returns", "future_return") if c in d.columns), None)
        if fut_col is None:
            if "Close" not in d.columns:
                return None
            close = pd.to_numeric(d["Close"], errors="coerce")
            fut = (close.shift(-5) / close - 1.0)
        else:
            fut = pd.Series(d[fut_col].values, index=d.index)

        if "bench_future_ret" in d.columns:
            bench = pd.Series(d["bench_future_ret"].values, index=d.index)
        else:
            bench = fut.copy()

        w = alpha_sample_weights(fut, bench)
        if X_index is not None:
            w = w.reindex(X_index)
        w = w.fillna(w.min())
        return {"sample_weight": w.values}
    except Exception:
        return None


def _alpha_fit_params(d: pd.DataFrame, X_index) -> dict | None:
    """Build sklearn fit_params with alpha-aware sample weights.

    Looks for future returns columns in `d`: 'future_ret', 'Future_Returns', or 'future_return'.
    If none are present, derive a 5-day forward return from 'Close'.
    If 'bench_future_ret' is present, use it as the benchmark; otherwise, use the asset itself.
    """
    try:
        fut_col = next((c for c in ("future_ret", "Future_Returns", "future_return") if c in d.columns), None)
        if fut_col is None:
            if "Close" not in d.columns:
                return None
            close = pd.to_numeric(d["Close"], errors="coerce")
            fut = (close.shift(-5) / close - 1.0)
        else:
            fut = pd.Series(d[fut_col].values, index=d.index)

        if "bench_future_ret" in d.columns:
            bench = pd.Series(d["bench_future_ret"].values, index=d.index)
        else:
            bench = fut.copy()

        w = alpha_sample_weights(fut, bench)
        if X_index is not None:
            w = w.reindex(X_index)
        w = w.fillna(w.min())
        return {"sample_weight": w.values}
    except Exception:
        return None


def _infer_ppy(freq: str = "D") -> int:
    return {"D": 252, "W": 52, "M": 12}.get(freq.upper(), 252)

def _active_ir(active_ret, freq: str = "D") -> float:
    import numpy as np, pandas as pd, math
    s = pd.Series(active_ret).dropna()
    if s.empty:
        return 0.0
    ppy = _infer_ppy(freq)
    mu = s.mean() * ppy
    vol = s.std(ddof=0) * math.sqrt(ppy)
    return float(mu / vol) if vol and vol > 0 else 0.0

def _ols_alpha(r_s, r_b, freq: str = "D") -> float:
    import numpy as np, pandas as pd
    s = pd.Series(r_s).dropna()
    b = pd.Series(r_b).reindex_like(s).fillna(0.0)
    if s.empty:
        return 0.0
    x = b.values
    y = s.values
    X = np.column_stack([np.ones_like(x), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_per_period = float(beta[0])
        return alpha_per_period * _infer_ppy(freq)
    except Exception:
        return 0.0

def _derive_future_returns(df, horizon: int = 5):
    import pandas as pd
    if "future_ret" in df.columns:
        return pd.Series(df["future_ret"].values, index=df.index)
    if "Future_Returns" in df.columns:
        return pd.Series(df["Future_Returns"].values, index=df.index)
    if "future_return" in df.columns:
        return pd.Series(df["future_return"].values, index=df.index)
    if "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")
        return (close.shift(-horizon) / close - 1.0)
    return None

def _alpha_metrics_report(label: str, proba, df_like: pd.DataFrame, freq: str = "D") -> None:
    """Print Alpha (annualized) and Active-IR vs buy-and-hold for a probability vector aligned to df_like."""
    try:
        import numpy as np, pandas as pd, math

        def _infer_ppy(f: str) -> int:
            return {"D": 252, "W": 52, "M": 12}.get(f.upper(), 252)

        def _ols_alpha(r_s: pd.Series, r_b: pd.Series, f: str) -> float:
            s = pd.Series(r_s).dropna()
            b = pd.Series(r_b).reindex_like(s).fillna(0.0)
            if s.empty:
                return 0.0
            x = b.values
            y = s.values
            X = np.column_stack([np.ones_like(x), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[0]) * _infer_ppy(f)

        def _active_ir(active: pd.Series, f: str) -> float:
            ppy = _infer_ppy(f)
            mu = active.mean() * ppy
            vol = active.std(ddof=0) * math.sqrt(ppy)
            return float(mu / vol) if vol > 0 else 0.0

        idx = getattr(proba, "index", None)
        s_proba = pd.Series(proba, index=idx) if idx is not None else pd.Series(proba, index=df_like.index[:len(proba)])

        fut = None
        for c in ("future_ret", "Future_Returns", "future_return"):
            if c in df_like.columns:
                fut = pd.Series(df_like[c].values, index=df_like.index)
                break
        if fut is None and "Close" in df_like.columns:
            close = pd.to_numeric(df_like["Close"], errors="coerce")
            fut = (close.shift(-5) / close - 1.0)
        if fut is None:
            print(f"    - {label} alpha metrics: (skipped: cannot derive future returns)")
            return

        fut = fut.reindex(s_proba.index).fillna(0.0)
        bench = fut.copy()
        entries = (s_proba >= 0.5).astype(int)  # quick diagnostic threshold
        strat_ret = fut * entries
        active = strat_ret - bench
        alpha = _ols_alpha(strat_ret, bench, freq)
        ir = _active_ir(active, freq)
        print(f"    - {label} Alpha (annualized): {alpha:.4f} | Active-IR: {ir:.3f}")
    except Exception as _e:
        print(f"    - {label} alpha metrics: (error: {_e})")


    try:
        import pandas as pd, numpy as np
        idx = getattr(proba, "index", None)
        s_proba = pd.Series(proba, index=idx) if idx is not None else pd.Series(proba, index=df_like.index[:len(proba)])
        fut = _derive_future_returns(df_like)
        if fut is None:
            print(f"    - {label} alpha metrics: (skipped: cannot derive future returns)")
            return
        fut = pd.Series(fut).reindex(s_proba.index).fillna(0.0)
        bench = fut.copy()
        entries = (s_proba >= 0.5).astype(int)  # neutral threshold; final run uses alpha-optimized
        strat_ret = fut * entries
        active = strat_ret - bench
        alpha = _ols_alpha(strat_ret, bench, freq=freq)
        ir = _active_ir(active, freq=freq)
        print(f"    - {label} Alpha (annualized): {alpha:.4f} | Active-IR: {ir:.3f}")
    except Exception as _e:
        print(f"    - {label} alpha metrics: (error: {_e})")


def _alpha_fit_params(df: pd.DataFrame, X_index) -> dict | None:
    
    try:
        cand = ["future_ret", "Future_Returns", "future_return"]
        col = next((c for c in cand if c in df.columns), None)
        if col is None:
            return None
        fut = pd.Series(df[col].values, index=df.index)
        if "bench_future_ret" in df.columns:
            bench = pd.Series(df["bench_future_ret"].values, index=df.index)
        else:
            bench = fut.copy()
        w = alpha_sample_weights(fut, bench)
        if X_index is not None:
            w = w.reindex(X_index)
        w = w.fillna(w.min()).values
        return {"sample_weight": w}
    except Exception:
        return None


# Alpha-aware training
USE_ALPHA_WEIGHTS: bool = True

# Scikit-learn imports (fallback for CPU or if cuML not available)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint

# Added for XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Added for PyTorch and LSTM/GRU models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Added for SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import configuration from config.py
from config import (
    SEED, FEAT_SMA_LONG, SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    GRU_HIDDEN_SIZE_OPTIONS, GRU_NUM_LAYERS_OPTIONS, GRU_DROPOUT_OPTIONS,
    GRU_LEARNING_RATE_OPTIONS, GRU_BATCH_SIZE_OPTIONS, GRU_EPOCHS_OPTIONS,
    ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION, SAVE_PLOTS,
    FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING,
    USE_LOGISTIC_REGRESSION, USE_RANDOM_FOREST, USE_SVM, USE_MLP_CLASSIFIER,
    USE_LSTM, USE_GRU, USE_LIGHTGBM, USE_XGBOOST,
    TRY_LSTM_INSTEAD_OF_GRU,
    USE_TCN, USE_ELASTIC_NET, USE_RIDGE
)
# PYTORCH_AVAILABLE is set locally below based on actual import, not from config

# Import data fetching for training data
from data_utils import fetch_training_data

# --- Globals for ML library status ---
_ml_libraries_initialized = False
CUDA_AVAILABLE = False
CUML_AVAILABLE = False
LGBMClassifier = None
XGBClassifier = None
XGBRegressor = None
# RandomForestRegressor imported from sklearn, don't set to None
# Cache GPU availability to avoid retesting in worker processes
_lgbm_gpu_available = None
_xgb_gpu_available = None
cuMLRandomForestClassifier = None
cuMLLogisticRegression = None
cuMLStandardScaler = None
models_and_params: Dict = {}

# --- Global tracking for model selection statistics ---
_model_selection_stats = {}

# Helper function (copied from main.py)
# _ensure_dir moved to utils.py to avoid duplication
from utils import _ensure_dir

# Helper function to safely move model to device
def safe_to_device(model, device):
    """Safely move model to device with fallback to CPU on CUDA errors."""
    try:
        return model.to(device)
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print(f"⚠️ CUDA error when moving model to device: {e}. Falling back to CPU.")
            return model.cpu()
        else:
            raise  # Re-raise non-CUDA errors

# Define LSTM/GRU model architecture
if PYTORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
            super(LSTMClassifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out

    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
            super(GRUClassifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            if x.dim() == 4:
                x = x.squeeze(0)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out
    
    class GRURegressor(nn.Module):
        """GRU for regression - predicts continuous values (e.g., return percentages)"""
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
            super(GRURegressor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
            # ✅ Add Batch Normalization for stable training
            self.bn = nn.BatchNorm1d(hidden_size)
            # ✅ Multi-layer output with Tanh to bound predictions to [-1, 1]
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_size),
                nn.Tanh()  # ✅ Bound output to [-1, 1] range
            )

        def forward(self, x):
            if x.dim() == 4:
                x = x.squeeze(0)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = out[:, -1, :]
            # ✅ Apply batch normalization
            out = self.bn(out)
            # ✅ Apply FC layers with Tanh activation
            out = self.fc(out)
            return out

    class LSTMRegressor(nn.Module):
        """LSTM for regression - alternative to GRU"""
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
            super(LSTMRegressor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0.0)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_size),
                nn.Tanh()
            )

        def forward(self, x):
            if x.dim() == 4:
                x = x.squeeze(0)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.bn(out)
            out = self.fc(out)
            return out

    class TCNRegressor(nn.Module):
        """Temporal Convolutional Network for regression"""
        def __init__(self, input_size, num_filters=32, kernel_size=3, num_levels=2, dropout=0.1):
            super().__init__()
            layers = []
            in_ch = input_size
            for _ in range(num_levels):
                layers.append(nn.Conv1d(in_ch, num_filters, kernel_size, padding="same"))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_ch = num_filters
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(num_filters, 1)
        
        def forward(self, x):
            if x.dim() == 4:
                x = x.squeeze(0)
            x = x.transpose(1, 2)  # (batch, seq, feat) -> (batch, feat, seq)
            x = self.net(x)
            x = x.mean(dim=2)      # global average pool over time
            return self.head(x).squeeze(-1)

            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.bn(out)
            out = self.fc(out)
            return out

def initialize_ml_libraries():
    """Initializes ML libraries and prints their status only once."""
    global _ml_libraries_initialized, CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier, XGBClassifier, XGBRegressor, models_and_params, \
           cuMLRandomForestClassifier, cuMLLogisticRegression, cuMLStandardScaler, _lgbm_gpu_available, _xgb_gpu_available
    
    if _ml_libraries_initialized:
        return models_and_params

    # CUML_AVAILABLE is initially False and set to True if imports succeed.
    # The explicit disablement and fallback message are removed to allow cuML to be enabled if available.

    try:
        if PYTORCH_AVAILABLE and (USE_LSTM or USE_GRU):
            torch.manual_seed(SEED)
            # Check CUDA availability once and configure PyTorch accordingly
            if torch.cuda.is_available():
                try:
                    # Test CUDA by creating a small tensor and moving it to GPU
                    test_tensor = torch.randn(1).cuda()
                    test_tensor.cpu()  # Move back to free memory
                    CUDA_AVAILABLE = True
                    torch.cuda.manual_seed_all(SEED)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    print("✅ CUDA is available and working. GPU acceleration enabled with deterministic algorithms.")
                except RuntimeError as e:
                    CUDA_AVAILABLE = False
                    print(f"⚠️ CUDA device available but not working: {e}. Falling back to CPU.")
            else:
                CUDA_AVAILABLE = False
                if USE_LSTM or USE_GRU:
                    print("⚠️ CUDA is not available. GPU acceleration will not be used.")
        elif PYTORCH_AVAILABLE:
            # PyTorch is available but LSTM/GRU are not enabled, still check CUDA for other models
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.randn(1).cuda()
                    test_tensor.cpu()
                    CUDA_AVAILABLE = True
                except RuntimeError:
                    CUDA_AVAILABLE = False
            else:
                CUDA_AVAILABLE = False
        else:
            CUDA_AVAILABLE = False
            if USE_LSTM or USE_GRU:
                print("⚠️ PyTorch not installed. Run: pip install torch. CUDA availability check skipped.")
    except NameError:
        CUDA_AVAILABLE = False
        if USE_LSTM or USE_GRU:
            print("⚠️ PyTorch not installed. Run: pip install torch. CUDA availability check skipped.")

    try:
        from cuml.ensemble import RandomForestClassifier as cuMLRandomForestClassifier_
        from cuml.linear_model import LogisticRegression as cuMLLogisticRegression_
        from cuml.preprocessing import StandardScaler as cuMLStandardScaler_
        cuMLRandomForestClassifier = cuMLRandomForestClassifier_
        cuMLLogisticRegression = cuMLLogisticRegression_
        cuMLStandardScaler = cuMLStandardScaler_
    except ImportError:
        pass
    except Exception as e:
        pass

    if USE_LIGHTGBM:
        try:
            from lightgbm import LGBMClassifier as lgbm
            LGBMClassifier = lgbm
            
            # Force CPU for LightGBM due to OpenCL/PoCL compatibility issues with NVIDIA in WSL2
            # OpenCL requires a different runtime than CUDA and often fails with PoCL
            _lgbm_gpu_available = False
            
            lgbm_model_params = {
                "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='cpu'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
            }
            models_and_params["LightGBM (CPU)"] = lgbm_model_params
            print("ℹ️ LightGBM: using CPU (OpenCL not compatible with NVIDIA/PoCL in WSL2).")
        except ImportError:
            print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")

    if USE_XGBOOST and XGBOOST_AVAILABLE:
        XGBClassifier = xgb.XGBClassifier
        XGBRegressor = xgb.XGBRegressor
        xgb_model_params = {
            "model": XGBRegressor(random_state=SEED),
            "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        }
        if CUDA_AVAILABLE:
            xgb_model_params["model"].set_params(tree_method='gpu_hist')
            models_and_params["XGBoost (GPU)"] = xgb_model_params
            print(f"✅ XGBoostRegressor found. Configured for GPU (gpu_hist tree_method).")
        else:
            xgb_model_params["model"].set_params(tree_method='hist')
            models_and_params["XGBoost (CPU)"] = xgb_model_params
            print(f"ℹ️ XGBoostRegressor found. Will use CPU (CUDA not available).")

    _ml_libraries_initialized = True
    return models_and_params

def analyze_shap_for_gru(model, scaler: MinMaxScaler, X_df: pd.DataFrame, feature_names: List[str], ticker: str, target_col: str):
    """
    Calculates and visualizes SHAP values for a GRU model.
    """
    if not SHAP_AVAILABLE:
        print(f"  [{ticker}] SHAP is not available. Skipping SHAP analysis.")
        return

    # Check if model is a GRU model (GRUClassifier might not be defined if PyTorch is not available)
    if PYTORCH_AVAILABLE and hasattr(model, '__class__') and 'GRU' in model.__class__.__name__:
        print(f"  [{ticker}] SHAP KernelExplainer is not directly compatible with GRU models due to sequential input. Skipping SHAP analysis for GRU.")
        return

    print(f"  [{ticker}] Calculating SHAP values for GRU model ({target_col})...")
    
    try:
        def gru_predict_proba_wrapper_for_kernel(X_unsequenced_np):
            X_scaled_np = scaler.transform(X_unsequenced_np)
            
            X_sequences_for_pred = []
            if len(X_scaled_np) < SEQUENCE_LENGTH:
                return np.full(len(X_unsequenced_np), 0.5)

            for i in range(len(X_scaled_np) - SEQUENCE_LENGTH + 1):
                X_sequences_for_pred.append(X_scaled_np[i:i + SEQUENCE_LENGTH])
            
            if not X_sequences_for_pred:
                return np.full(len(X_unsequenced_np), 0.5)

            X_sequences_tensor = torch.tensor(np.array(X_sequences_for_pred), dtype=torch.float32)
            
            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            model.to(device)
            X_sequences_tensor = X_sequences_tensor.to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(X_sequences_tensor)
                return outputs.cpu().numpy().flatten()

        num_background_samples = min(50, len(X_df))
        background_data_for_kernel = X_df.sample(num_background_samples, random_state=SEED).values
        num_explain_samples = min(20, len(X_df))
        explain_data_for_kernel = X_df.sample(num_explain_samples, random_state=SEED).values

        if explain_data_for_kernel.shape[0] == 0:
            print(f"  [{ticker}] Explain data for KernelExplainer is empty. Skipping SHAP calculation.")
            return

        explainer = shap.KernelExplainer(
            gru_predict_proba_wrapper_for_kernel,
            background_data_for_kernel
        )
        
        shap_values = explainer.shap_values(explain_data_for_kernel)
        
        shap_values_for_plot = None
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values_for_plot = shap_values[1]
            elif len(shap_values) == 1:
                shap_values_for_plot = shap_values[0]
            else:
                print(f"  [{ticker}] SHAP values list has unexpected length ({len(shap_values)}). Skipping plot.")
                return
        elif isinstance(shap_values, np.ndarray):
            shap_values_for_plot = shap_values
        else:
            print(f"  [{ticker}] SHAP values are not a list or numpy array. Type: {type(shap_values)}. Skipping plot.")
            return

        if shap_values_for_plot is None or shap_values_for_plot.size == 0:
            print(f"  [{ticker}] SHAP values for plotting are empty or None. Skipping plot.")
            return

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, explain_data_for_kernel, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for {ticker} GRU ({target_col})")
        plt.tight_layout()
        
        shap_plot_path = Path(f"logs/shap_plots/{ticker}_GRU_SHAP_{target_col}.png")
        _ensure_dir(shap_plot_path.parent)
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"  [{ticker}] SHAP summary plot saved to {shap_plot_path}")

    except Exception as e:
        print(f"  [{ticker}] Error during SHAP analysis for GRU model ({target_col}): {e}")
        import traceback
        traceback.print_exc()

def analyze_shap_for_tree_model(model, X_df: pd.DataFrame, feature_names: List[str], ticker: str, target_col: str):
    """
    Calculates and visualizes SHAP values for tree-based models (XGBoost, RandomForest).
    """
    if not SHAP_AVAILABLE:
        print(f"  [{ticker}] SHAP is not available. Skipping SHAP analysis.")
        return

    print(f"  [{ticker}] Calculating SHAP values for tree model ({target_col})...")
    
    try:
        explainer = shap.TreeExplainer(model)
        
        num_explain_samples = min(100, len(X_df))
        explain_data = X_df.sample(num_explain_samples, random_state=SEED)

        shap_values = explainer.shap_values(explain_data)
        
        shap_values_for_plot = None
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values_for_plot = shap_values[1]
            elif len(shap_values) == 1:
                shap_values_for_plot = shap_values[0]
            else:
                print(f"  [{ticker}] SHAP values list has unexpected length ({len(shap_values)}). Skipping plot.")
                return
        elif isinstance(shap_values, np.ndarray):
            shap_values_for_plot = shap_values
        else:
            print(f"  [{ticker}] SHAP values are not a list or numpy array. Type: {type(shap_values)}. Skipping plot.")
            return

        if shap_values_for_plot is None or shap_values_for_plot.size == 0:
            print(f"  [{ticker}] SHAP values for plotting are empty or None. Skipping plot.")
            return
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, explain_data, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance for {ticker} Tree Model ({target_col})")
        plt.tight_layout()
        
        shap_plot_path = Path(f"logs/shap_plots/{ticker}_TREE_SHAP_{target_col}.png")
        _ensure_dir(shap_plot_path.parent)
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"  [{ticker}] SHAP summary plot saved to {shap_plot_path}")

    except Exception as e:
        print(f"  [{ticker}] Error during SHAP analysis for tree model ({target_col}): {e}")
        import traceback
        traceback.print_exc()

def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str = "TargetReturnBuy",
    feature_set: Optional[List[str]] = None,
    ticker: str = "UNKNOWN",
    initial_model=None,
    loaded_gru_hyperparams: Optional[Dict] = None,
    models_and_params_global: Optional[Dict] = None,
    perform_gru_hp_optimization: bool = True,
    default_target_percentage: float = None,  # Will be set from config if None
    default_class_horizon: int = None  # Will be set from config if None
):
    # --- Alpha-aware weights (optional) ---
    _fit_params = None
    if 'd' in locals() and 'X_df' in locals():
        if USE_ALPHA_WEIGHTS:
            try:
                _fit_params = _alpha_fit_params(d, X_df.index)
            except Exception:
                _fit_params = None
    
    """Train and compare multiple classifiers for a given target, returning the best one."""
    models_and_params = models_and_params_global if models_and_params_global is not None else initialize_ml_libraries()
    
    # Use default values from config if not provided
    if default_target_percentage is None:
        from config import TARGET_PERCENTAGE
        default_target_percentage = TARGET_PERCENTAGE
    if default_class_horizon is None:
        from config import CLASS_HORIZON
        default_class_horizon = CLASS_HORIZON
    d = df.copy()
    
    if target_col not in d.columns:
        print(f"  [DIAGNOSTIC] {ticker}: Target column '{target_col}' not found. Skipping.")
        return None, None, None

    if feature_set is None:
        print("⚠️ feature_set was None in train_and_evaluate_models. Inferring features from DataFrame columns.")
        final_feature_names = [col for col in d.columns if col not in ["Target", "TargetReturn"]]
        if not final_feature_names:
            print("⚠️ No features found in DataFrame after excluding target columns. Skipping model training.")
            return None, None, None
    else:
        final_feature_names = [f for f in feature_set if f in d.columns]
        if len(final_feature_names) != len(feature_set):
            missing_features = set(feature_set) - set(final_feature_names)
            print(f"⚠️ Missing features in DataFrame 'd' that were expected in feature_set: {missing_features}. Proceeding with available features.")
        if not final_feature_names:
            print("⚠️ No valid features to train with after filtering. Skipping model training.")
            return None, None, None

    required_cols_for_training = final_feature_names + [target_col]
    if not all(col in d.columns for col in required_cols_for_training):
        missing = [col for col in required_cols_for_training if col not in d.columns]
        print(f"⚠️ Missing critical columns for model comparison (target: {target_col}, missing: {missing}). Skipping.")
        return None, None, None

    d = d[required_cols_for_training].dropna()
    print(f"  [DIAGNOSTIC] {ticker}: train_and_evaluate_models - Rows after dropping NaNs: {len(d)}")
    
    if len(d) < 50:
        print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after feature prep ({len(d)} rows, need >= 50). Skipping.")
        return None, None, None
    
    X_df = d[final_feature_names]
    y = d[target_col].values

    # Handle infinity and extremely large values before scaling
    inf_mask = np.isinf(X_df.values).any(axis=1)
    large_val_threshold = np.finfo(np.float64).max / 10  # Conservative threshold
    large_mask = (np.abs(X_df.values) > large_val_threshold).any(axis=1)
    invalid_mask = inf_mask | large_mask
    
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"  [WARNING] {ticker}: Removing {n_invalid} rows with inf/extremely large values before scaling.")
        X_df = X_df[~invalid_mask]
        y = y[~invalid_mask]
        
        if len(y) < 50:
            print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after removing inf values ({len(y)} rows, need >= 50). Skipping.")
            return None, None, None

    # Also replace any remaining inf in individual columns (shouldn't happen but safety net)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    # Drop any rows that now have NaN from the replacement
    valid_rows = ~X_df.isna().any(axis=1)
    if not valid_rows.all():
        X_df = X_df[valid_rows]
        y = y[valid_rows]
        
        if len(y) < 50:
            print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after final inf cleanup ({len(y)} rows, need >= 50). Skipping.")
            return None, None, None

    # Regression: check we have enough samples
    if len(y) < 10:
        print(f"  [DIAGNOSTIC] {ticker}: Not enough samples for '{target_col}' (only {len(y)} samples, need >= 10). Skipping.")
        return None, None, None
    n_splits = min(3, len(y) // 5)  # Use 3-fold or less if dataset is tiny

    if CUML_AVAILABLE and cuMLStandardScaler:
        try:
            import cuml
            scaler = cuMLStandardScaler()
            X_gpu_np = X_df.values
            X_scaled = scaler.fit_transform(X_gpu_np)
            X = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_df.index)
        except Exception as e:
            print(f"⚠️ Error using cuML StandardScaler: {e}. Falling back to sklearn.StandardScaler.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df)
            X = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_df.index)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_df.index)
    
    scaler.feature_names_in_ = list(final_feature_names) 

    models_and_params_local = {} 

    if CUML_AVAILABLE and USE_LOGISTIC_REGRESSION:
        models_and_params_local["cuML Logistic Regression"] = {
            "model": cuMLLogisticRegression(class_weight="balanced", solver='qn'),
            "params": {'C': [0.1, 1.0, 10.0]}
        }
    if CUML_AVAILABLE and USE_RANDOM_FOREST:
        models_and_params_local["cuML Random Forest"] = {
            "model": cuMLRandomForestClassifier(random_state=SEED),
            "params": {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, None]}
        }
    
    if USE_LOGISTIC_REGRESSION:
        models_and_params_local["Logistic Regression"] = {
            "model": LogisticRegression(random_state=SEED, class_weight="balanced", solver='liblinear'),
            "params": {'C': [0.1, 1.0, 10.0, 100.0]}
        }
    if USE_ELASTIC_NET:
        models_and_params_local["ElasticNet"] = {
            "model": ElasticNet(random_state=SEED, max_iter=2000),
            "params": {'alpha': [0.0005, 0.001, 0.005, 0.01], 'l1_ratio': [0.1, 0.3, 0.5, 0.7]}
        }
    if USE_RIDGE:
        models_and_params_local["Ridge"] = {
            "model": Ridge(random_state=SEED, max_iter=2000, solver="lsqr"),
            "params": {'alpha': [0.1, 1.0, 5.0, 10.0]}
        }
    if USE_RANDOM_FOREST:
        models_and_params_local["Random Forest"] = {
            "model": RandomForestRegressor(random_state=SEED),
            "params": {'n_estimators': [100, 200], 'max_depth': [10, 15]}  # Reduced grid
        }
    if USE_SVM:
        from sklearn.svm import SVR
        models_and_params_local["SVR"] = {
            "model": SVR(),
            "params": {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['rbf', 'linear'], 'epsilon': [0.01, 0.1, 1.0]}
        }
    if USE_MLP_CLASSIFIER:
        from sklearn.neural_network import MLPRegressor
        models_and_params_local["MLPRegressor"] = {
            "model": MLPRegressor(random_state=SEED, max_iter=500, early_stopping=True),
            "params": {'hidden_layer_sizes': [(100,), (100, 50), (50, 25)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]}
        }

    if USE_LIGHTGBM:
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier as LGBMClf
        except Exception as e:
            print(f"⚠️ LightGBM not available: {e}")
            LGBMRegressor = None
            LGBMClf = None
        # Always use regression (default behavior)
        if LGBMRegressor:
            lgbm_model_params = {
                "model": LGBMRegressor(random_state=SEED, verbosity=-1, device='cpu'),
                "params": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [-1, 5, 7]}
            }
            models_and_params_local["LightGBM Regressor (CPU)"] = lgbm_model_params

    if USE_XGBOOST and XGBOOST_AVAILABLE and XGBClassifier:
        # Prefer GPU if the build supports CUDA; use new device API (XGBoost>=2.0)
        use_cuda_in_xgb = False
        if hasattr(xgb, "build_info"):
            try:
                use_cuda_in_xgb = bool(xgb.build_info().get("USE_CUDA", False)) and CUDA_AVAILABLE
            except Exception:
                use_cuda_in_xgb = False

        device_param = "cuda" if use_cuda_in_xgb else None  # None => CPU
        common_kwargs = {
            "random_state": SEED,
            "tree_method": "hist",  # recommended with device param
        }
        if device_param:
            common_kwargs["device"] = device_param

        xgb_model_params = {
            "model": XGBRegressor(**common_kwargs),
            "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 7]}  # Reduced grid
        }
        models_and_params_local["XGBoost"] = xgb_model_params

    if PYTORCH_AVAILABLE:
        # Scale features to [0, 1]
        dl_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled_dl = dl_scaler.fit_transform(X_df)
        
        # ✅ CRITICAL FIX: Scale targets (y) to [-1, 1] to match GRU Tanh output
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_sequences = []
        y_sequences = []
        for i in range(len(X_scaled_dl) - SEQUENCE_LENGTH):
            X_sequences.append(X_scaled_dl[i:i + SEQUENCE_LENGTH])
            y_sequences.append(y_scaled[i + SEQUENCE_LENGTH])  # Use scaled y
        
        if not X_sequences:
            print(f"  [DIAGNOSTIC] {ticker}: Not enough data to create sequences for DL models (need > {SEQUENCE_LENGTH} rows). Skipping DL models.")
        else:
            X_sequences = torch.tensor(np.array(X_sequences), dtype=torch.float32)
            y_sequences = torch.tensor(np.array(y_sequences), dtype=torch.float32).unsqueeze(1)

            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            
            dataset = TensorDataset(X_sequences, y_sequences)
            dataloader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

            input_size = X_sequences.shape[2]
            
            # Always use MSE loss for regression (predicting returns)
            criterion = nn.MSELoss()
            print(f"    - Using MSE loss for regression (predicting returns)")

            if USE_LSTM:
                from ml_models import LSTMRegressor
                lstm_model = safe_to_device(LSTMRegressor(input_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 1, LSTM_DROPOUT), device)
                if initial_model and isinstance(initial_model, LSTMClassifier):
                    try:
                        lstm_model.load_state_dict(initial_model.state_dict())
                        print(f"    - Loaded existing LSTM model state for {ticker} to continue training.")
                    except Exception as e:
                        print(f"    - Error loading LSTM model state for {ticker}: {e}. Training from scratch.")
                
                optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LSTM_LEARNING_RATE)

                for epoch in range(LSTM_EPOCHS):
                    try:
                        for batch_X, batch_y in dataloader:
                            try:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    device = torch.device("cpu")
                                    lstm_model = _safe_to_cpu(lstm_model)
                                    if lstm_model is None:
                                        break
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                else:
                                    raise
                            if lstm_model is None:
                                break
                            optimizer_lstm.zero_grad()
                            outputs = lstm_model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer_lstm.step()
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"⚠️ CUDA error during LSTM training epoch {epoch}, recreating model on CPU...")
                            try:
                                # Clear CUDA cache and recreate model on CPU
                                torch.cuda.empty_cache()
                                device = torch.device("cpu")
                                # Recreate model on CPU instead of moving corrupted model
                                lstm_model = type(lstm_model)(
                                    input_size=lstm_model.lstm.input_size,
                                    hidden_size=lstm_model.lstm.hidden_size,
                                    num_layers=lstm_model.lstm.num_layers,
                                    output_size=lstm_model.fc.out_features
                                ).to(device)
                                optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                            except Exception:
                                # If recreation fails, skip LSTM for this ticker
                                print(f"⚠️ Could not recover LSTM, skipping...")
                                lstm_model = None
                                break
                            
                            if lstm_model is not None:
                                # Restart this epoch on CPU
                                for batch_X, batch_y in dataloader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer_lstm.zero_grad()
                                    outputs = lstm_model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    optimizer_lstm.step()
                        else:
                            raise
                
                lstm_model.eval()
                with torch.no_grad():
                    all_outputs = []
                    for batch_X, _ in dataloader:
                        try:
                            batch_X = batch_X.to(device)
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                device = torch.device("cpu")
                                lstm_model = _safe_to_cpu(lstm_model)
                                if lstm_model is None:
                                    break
                                batch_X = batch_X.to(device)
                            else:
                                raise
                        if lstm_model is None:
                            break
                        outputs = lstm_model(batch_X)
                        all_outputs.append(outputs.cpu().numpy())
                    y_pred_proba_lstm = np.concatenate(all_outputs).flatten() if all_outputs else np.array([])
                
                try:
                    # Always use regression (default behavior)
                    from sklearn.metrics import mean_squared_error, r2_score
                    y_true = y_sequences.cpu().numpy()
                    y_pred = y_pred_proba_lstm
                    mse_lstm = mean_squared_error(y_true, y_pred)
                    r2_lstm = r2_score(y_true, y_pred)
                    rmse_lstm = mse_lstm ** 0.5
                    models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler, "auc": -mse_lstm}  # Negative MSE (higher is better)
                    print(f"      📊 LSTM Regression Metrics:")
                    print(f"         MSE: {mse_lstm:.6f}")
                    print(f"         RMSE: {rmse_lstm:.6f}")
                    print(f"         R² Score: {r2_lstm:.4f} ({'Good' if r2_lstm > 0.5 else 'Poor'} - {abs(r2_lstm)*100:.1f}% variance explained)")
                except ValueError:
                    models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler, "auc": 0.0}

            # --- TCN Regressor (lightweight) ---
            if USE_TCN:
                tcn_model = safe_to_device(TCNRegressor(input_size, num_filters=32, kernel_size=3, num_levels=2, dropout=0.1), device)
                optimizer_tcn = optim.Adam(tcn_model.parameters(), lr=LSTM_LEARNING_RATE)

                for epoch in range(LSTM_EPOCHS):
                    try:
                        for batch_X, batch_y in dataloader:
                            try:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    device = torch.device("cpu")
                                    tcn_model = _safe_to_cpu(tcn_model)
                                    if tcn_model is None:
                                        break
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                else:
                                    raise
                            if tcn_model is None:
                                break
                            optimizer_tcn.zero_grad()
                            outputs = tcn_model(batch_X)
                            loss = criterion(outputs, batch_y.squeeze())
                            loss.backward()
                            optimizer_tcn.step()
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"⚠️ CUDA error during TCN training epoch {epoch}, recreating model on CPU...")
                            try:
                                torch.cuda.empty_cache()
                                device = torch.device("cpu")
                                tcn_model = TCNRegressor(input_size, num_filters=32, kernel_size=3, num_levels=2, dropout=0.1).to(device)
                                optimizer_tcn = optim.Adam(tcn_model.parameters(), lr=LSTM_LEARNING_RATE)
                            except Exception:
                                print(f"⚠️ Could not recover TCN, skipping...")
                                tcn_model = None
                                break
                            
                            if tcn_model is not None:
                                for batch_X, batch_y in dataloader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer_tcn.zero_grad()
                                    outputs = tcn_model(batch_X)
                                    loss = criterion(outputs, batch_y.squeeze())
                                    loss.backward()
                                    optimizer_tcn.step()
                        else:
                            raise
                    if tcn_model is None:
                        break

                if tcn_model is not None:
                    tcn_model.eval()
                    with torch.no_grad():
                        all_outputs = []
                        for batch_X, _ in dataloader:
                            try:
                                batch_X = batch_X.to(device)
                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    device = torch.device("cpu")
                                    tcn_model = _safe_to_cpu(tcn_model)
                                    if tcn_model is None:
                                        break
                                    batch_X = batch_X.to(device)
                                else:
                                    raise
                            if tcn_model is None:
                                break
                            outputs = tcn_model(batch_X)
                            all_outputs.append(outputs.cpu().numpy())
                        y_pred_tcn = np.concatenate(all_outputs).flatten() if all_outputs else np.array([])

                if tcn_model is not None and len(y_pred_tcn) > 0:
                    try:
                        from sklearn.metrics import mean_squared_error, r2_score
                        y_true = y_sequences.cpu().numpy()
                        mse_tcn = mean_squared_error(y_true, y_pred_tcn)
                        r2_tcn = r2_score(y_true, y_pred_tcn)
                        rmse_tcn = mse_tcn ** 0.5
                        models_and_params_local["TCN"] = {"model": tcn_model, "scaler": dl_scaler, "auc": -mse_tcn, "params": None}
                        print(f"      📊 TCN Regression Metrics:")
                        print(f"         MSE: {mse_tcn:.6f}")
                        print(f"         RMSE: {rmse_tcn:.6f}")
                        print(f"         R² Score: {r2_tcn:.4f}")
                    except ValueError:
                        models_and_params_local["TCN"] = {"model": tcn_model, "scaler": dl_scaler, "auc": 0.0, "params": None}

            if USE_GRU:
                if perform_gru_hp_optimization and ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION:
                    print(f"    - Starting GRU hyperparameter optimization for {ticker} ({target_col}) (HP_OPT={perform_gru_hp_optimization}, ENABLE_HP_OPT={ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION}, Horizon={default_class_horizon})...")
                    # For regression we minimize MSE; we still store negative MSE in "auc" field for compatibility
                    best_gru_mse = np.inf
                    best_gru_auc = -np.inf  # kept for compatibility with downstream structure
                    best_gru_model = None
                    best_gru_scaler = None
                    best_gru_hyperparams = {}

                    def create_focused_range(base_val, step, min_val=None, max_val=None, is_float=False, options_list=None):
                        if options_list:
                            return sorted(list(set([x for x in options_list if (min_val is None or x >= min_val) and (max_val is None or x <= max_val)])))
                        
                        if is_float:
                            options = [base_val - step, base_val, base_val + step]
                            options = [round(x, 4) for x in options]
                        else:
                            options = [base_val - step, base_val, base_val + step]
                        
                        if min_val is not None:
                            options = [max(min_val, x) for x in options]
                        if max_val is not None:
                            options = [min(max_val, x) for x in options]
                        return sorted(list(set(options)))

                    base_hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE) if loaded_gru_hyperparams else LSTM_HIDDEN_SIZE
                    base_num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS) if loaded_gru_hyperparams else LSTM_NUM_LAYERS
                    base_dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT) if loaded_gru_hyperparams else LSTM_DROPOUT
                    base_learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE) if loaded_gru_hyperparams else LSTM_LEARNING_RATE
                    base_batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE) if loaded_gru_hyperparams else LSTM_BATCH_SIZE
                    base_epochs = loaded_gru_hyperparams.get("epochs", LSTM_EPOCHS) if loaded_gru_hyperparams else LSTM_EPOCHS

                    hidden_size_options = create_focused_range(base_hidden_size, 32, min_val=32, options_list=GRU_HIDDEN_SIZE_OPTIONS)
                    num_layers_options = create_focused_range(base_num_layers, 1, min_val=1, options_list=GRU_NUM_LAYERS_OPTIONS)
                    dropout_rate_options = create_focused_range(base_dropout_rate, 0.1, min_val=0.0, max_val=0.5, is_float=True, options_list=GRU_DROPOUT_OPTIONS)
                    learning_rate_options = create_focused_range(base_learning_rate, base_learning_rate * 0.5, min_val=0.0001, is_float=True, options_list=GRU_LEARNING_RATE_OPTIONS)
                    batch_size_options = create_focused_range(base_batch_size, base_batch_size // 2, min_val=16, options_list=GRU_BATCH_SIZE_OPTIONS)
                    epochs_options = create_focused_range(base_epochs, 20, min_val=10, options_list=GRU_EPOCHS_OPTIONS)

                    best_gru_hyperparams = {
                        "hidden_size": base_hidden_size, "num_layers": base_num_layers, "dropout_rate": base_dropout_rate,
                        "learning_rate": base_learning_rate, "batch_size": base_batch_size, "epochs": base_epochs
                    }
                    best_gru_mse = np.inf
                    best_gru_auc = -np.inf  # legacy name; stores -MSE for regression
                    best_gru_model = None
                    best_gru_scaler = dl_scaler

                    hyperparameter_dimensions = [
                        ("hidden_size", GRU_HIDDEN_SIZE_OPTIONS, 32, None, False, 32),
                        ("num_layers", GRU_NUM_LAYERS_OPTIONS, 1, None, False, 1),
                        ("dropout_rate", GRU_DROPOUT_OPTIONS, 0.0, 0.5, True, 0.1),
                        ("learning_rate", GRU_LEARNING_RATE_OPTIONS, 0.0001, None, True, None),
                        ("batch_size", GRU_BATCH_SIZE_OPTIONS, 16, None, False, None),
                        ("epochs", GRU_EPOCHS_OPTIONS, 10, None, False, 20)
                    ]

                    total_combinations = 0
                    for param_name, options_list, min_val, max_val, is_float, step_size_for_range in hyperparameter_dimensions:
                        current_best_val = best_gru_hyperparams[param_name]
                        if param_name == "learning_rate":
                            current_options = create_focused_range(current_best_val, current_best_val * 0.5, min_val=min_val, is_float=True)
                        elif param_name == "batch_size":
                            current_options = create_focused_range(current_best_val, current_best_val // 2, min_val=min_val, is_float=False)
                        elif param_name == "epochs":
                            current_options = create_focused_range(current_best_val, step_size_for_range, min_val=min_val, is_float=False)
                        else:
                            current_options = create_focused_range(current_best_val, step_size_for_range, min_val=min_val, max_val=max_val, is_float=is_float, options_list=options_list)
                        total_combinations += len(current_options)

                    current_iteration = 0
                    for param_name, options_list, min_val, max_val, is_float, step_size_for_range in hyperparameter_dimensions:
                        current_best_val = best_gru_hyperparams[param_name]
                        
                        if param_name == "learning_rate":
                            current_options = create_focused_range(current_best_val, current_best_val * 0.5, min_val=min_val, is_float=True)
                        elif param_name == "batch_size":
                            current_options = create_focused_range(current_best_val, current_best_val // 2, min_val=min_val, is_float=False)
                        elif param_name == "epochs":
                            current_options = create_focused_range(current_best_val, step_size_for_range, min_val=min_val, is_float=False)
                        else:
                            current_options = create_focused_range(current_best_val, step_size_for_range, min_val=min_val, max_val=max_val, is_float=is_float, options_list=options_list)

                        for value in current_options:
                            current_iteration += 1
                            temp_hyperparams = best_gru_hyperparams.copy()
                            temp_hyperparams[param_name] = value
                            
                            current_dropout_rate = temp_hyperparams["dropout_rate"] if temp_hyperparams["num_layers"] > 1 else 0.0
                            temp_hyperparams["dropout_rate"] = current_dropout_rate

                            # Choose model based on architecture preference and task type
                            if TRY_LSTM_INSTEAD_OF_GRU:
                                gru_model = safe_to_device(LSTMRegressor(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]), device)
                            else:
                                gru_model = safe_to_device(GRURegressor(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]), device)
                            optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
                            
                            current_dataloader = DataLoader(dataset, batch_size=temp_hyperparams["batch_size"], shuffle=True)

                            for epoch in range(temp_hyperparams["epochs"]):
                                try:
                                    for batch_X, batch_y in current_dataloader:
                                        try:
                                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                        except RuntimeError as e:
                                            if "CUDA" in str(e):
                                                # Fallback to CPU if CUDA fails during training
                                                device = torch.device("cpu")
                                                gru_model = _safe_to_cpu(gru_model)
                                                if gru_model is None:
                                                    break
                                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                            else:
                                                raise
                                        if gru_model is None:
                                            break
                                        optimizer_gru.zero_grad()
                                        outputs = gru_model(batch_X)
                                        loss = criterion(outputs, batch_y)
                                        loss.backward()
                                        optimizer_gru.step()
                                except RuntimeError as e:
                                    if "CUDA" in str(e):
                                        print(f"⚠️ CUDA error during training epoch {epoch}, recreating GRU on CPU...")
                                        try:
                                            torch.cuda.empty_cache()
                                            device = torch.device("cpu")
                                            if TRY_LSTM_INSTEAD_OF_GRU:
                                                gru_model = LSTMRegressor(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                                            else:
                                                gru_model = GRURegressor(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                                            optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
                                        except Exception:
                                            print(f"⚠️ Could not recover GRU, skipping...")
                                            gru_model = None
                                            break
                                        
                                        if gru_model is not None:
                                            for batch_X, batch_y in current_dataloader:
                                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                                optimizer_gru.zero_grad()
                                                outputs = gru_model(batch_X)
                                                loss = criterion(outputs, batch_y)
                                                loss.backward()
                                                optimizer_gru.step()
                                    else:
                                        raise
                                if gru_model is None:
                                    break
                            
                            if gru_model is not None:
                                gru_model.eval()
                                with torch.no_grad():
                                    all_outputs = []
                                    for batch_X, _ in current_dataloader:
                                        try:
                                            batch_X = batch_X.to(device)
                                        except RuntimeError:
                                            device = torch.device("cpu")
                                            gru_model = _safe_to_cpu(gru_model)
                                            if gru_model is None:
                                                break
                                            batch_X = batch_X.to(device)
                                        if gru_model is None:
                                            break
                                        outputs = gru_model(batch_X)
                                        all_outputs.append(outputs.cpu().numpy())
                                    y_pred_proba_gru = np.concatenate(all_outputs).flatten() if all_outputs else np.array([])

                            if gru_model is not None and len(y_pred_proba_gru) > 0:
                                try:
                                    # Always use regression (default behavior)
                                    from sklearn.metrics import mean_squared_error, r2_score
                                    y_true = y_sequences.cpu().numpy()
                                    y_pred = y_pred_proba_gru
                                    mse_gru = mean_squared_error(y_true, y_pred)
                                    r2_gru = r2_score(y_true, y_pred)
                                    auc_gru = -mse_gru  # keep legacy key; negative MSE for compatibility
                                    print(f"            GRU MSE: {mse_gru:.6f}, R²: {r2_gru:.4f} | {param_name}={value} (HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']})")
                                    better = mse_gru < best_gru_mse

                                    if better:
                                        best_gru_auc = auc_gru
                                        best_gru_mse = mse_gru
                                        best_gru_model = gru_model
                                        best_gru_scaler = dl_scaler # dl_scaler is already fitted
                                        best_gru_hyperparams = temp_hyperparams.copy() # Update best_gru_hyperparams
                                except ValueError:
                                    print(f"            GRU AUC: Not enough samples with positive class for AUC calculation. | {param_name}={value} (HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']})")
                                
                    if best_gru_model:
                        models_and_params_local["GRU"] = {"model": best_gru_model, "scaler": best_gru_scaler, "y_scaler": y_scaler, "auc": best_gru_auc, "hyperparams": best_gru_hyperparams}
                        model_name = "LSTM" if TRY_LSTM_INSTEAD_OF_GRU else "GRU"
                        # Always use regression (default behavior)
                        print(f"      Best {model_name} found for {ticker} ({target_col}) with MSE: {best_gru_mse:.6f}, Hyperparams: {best_gru_hyperparams}")
                        print(f"DEBUG: SAVE_PLOTS={SAVE_PLOTS}, SHAP_AVAILABLE={SHAP_AVAILABLE}")
                        if SAVE_PLOTS and SHAP_AVAILABLE:
                            analyze_shap_for_gru(best_gru_model, best_gru_scaler, X_df, final_feature_names, ticker, target_col)
                    else:
                        models_and_params_local["GRU"] = {"model": None, "scaler": None, "y_scaler": None, "auc": 0.0}
                else: # ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION is False, use fixed or loaded hyperparameters
                    if loaded_gru_hyperparams:
                        # Use loaded hyperparameters
                        model_name = "LSTM" if TRY_LSTM_INSTEAD_OF_GRU else "GRU"
                        print(f"    - Training {model_name} for {ticker} ({target_col}) with loaded hyperparameters...")
                        hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE)
                        num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS)
                        dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT)
                        learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE)
                        batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE)
                        epochs = LSTM_EPOCHS
                        print(f"      Loaded {model_name} Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}")
                    else:
                        model_name = "LSTM" if TRY_LSTM_INSTEAD_OF_GRU else "GRU"
                        print(f"    - Training {model_name} for {ticker} ({target_col}) with default fixed hyperparameters (HP_OPT={perform_gru_hp_optimization}, ENABLE_HP_OPT={ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION})...")
                        hidden_size = LSTM_HIDDEN_SIZE
                        num_layers = LSTM_NUM_LAYERS
                        dropout_rate = LSTM_DROPOUT
                        learning_rate = LSTM_LEARNING_RATE
                        batch_size = LSTM_BATCH_SIZE
                        epochs = LSTM_EPOCHS
                        print(f"      Default {model_name} Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, Horizon={default_class_horizon}")

                    # Choose model based on architecture preference
                    if TRY_LSTM_INSTEAD_OF_GRU:
                        gru_model = safe_to_device(LSTMRegressor(input_size, hidden_size, num_layers, 1, dropout_rate), device)
                        model_type = "LSTM"
                        if initial_model and isinstance(initial_model, LSTMRegressor):
                            try:
                                gru_model.load_state_dict(initial_model.state_dict())
                                print(f"    - Loaded existing LSTM regressor state for {ticker} to continue training.")
                            except Exception as e:
                                print(f"    - Error loading LSTM regressor state for {ticker}: {e}. Training from scratch.")
                    else:
                        model_type = "GRU"
                        gru_model = safe_to_device(GRURegressor(input_size, hidden_size, num_layers, 1, dropout_rate), device)
                        if initial_model and isinstance(initial_model, GRURegressor):
                            try:
                                gru_model.load_state_dict(initial_model.state_dict())
                                print(f"    - Loaded existing GRU regressor state for {ticker} to continue training.")
                            except Exception as e:
                                print(f"    - Error loading GRU regressor state for {ticker}: {e}. Training from scratch.")
                    
                    optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)
                    
                    # Create DataLoader for current batch_size
                    current_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                    for epoch in range(epochs):
                        try:
                            for batch_X, batch_y in current_dataloader:
                                try:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                except RuntimeError as e:
                                    if "CUDA" in str(e):
                                        device = torch.device("cpu")
                                        gru_model = _safe_to_cpu(gru_model)
                                        if gru_model is None:
                                            break
                                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    else:
                                        raise
                                if gru_model is None:
                                    break
                                optimizer_gru.zero_grad()
                                outputs = gru_model(batch_X)
                                loss = criterion(outputs, batch_y)
                                loss.backward()
                                optimizer_gru.step()
                        except RuntimeError as e:
                            if "CUDA" in str(e):
                                print(f"⚠️ CUDA error during training epoch {epoch}, recreating GRU on CPU...")
                                try:
                                    torch.cuda.empty_cache()
                                    device = torch.device("cpu")
                                    if TRY_LSTM_INSTEAD_OF_GRU:
                                        gru_model = LSTMRegressor(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                                    else:
                                        gru_model = GRURegressor(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                                    optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)
                                except Exception:
                                    print(f"⚠️ Could not recover GRU, skipping...")
                                    gru_model = None
                                    break
                                
                                if gru_model is not None:
                                    for batch_X, batch_y in current_dataloader:
                                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                        optimizer_gru.zero_grad()
                                        outputs = gru_model(batch_X)
                                        loss = criterion(outputs, batch_y)
                                        loss.backward()
                                        optimizer_gru.step()
                            else:
                                raise
                        if gru_model is None:
                            break
                    
                    # Evaluate GRU
                    if gru_model is not None:
                        gru_model.eval()
                        with torch.no_grad():
                            all_outputs = []
                            for batch_X, _ in current_dataloader:
                                try:
                                    batch_X = batch_X.to(device)
                                except RuntimeError:
                                    device = torch.device("cpu")
                                    gru_model = _safe_to_cpu(gru_model)
                                    if gru_model is None:
                                        break
                                    batch_X = batch_X.to(device)
                                if gru_model is None:
                                    break
                                outputs = gru_model(batch_X)
                                all_outputs.append(outputs.cpu().numpy())
                            y_pred_proba_gru = np.concatenate(all_outputs).flatten() if all_outputs else np.array([])

                    if gru_model is not None and len(y_pred_proba_gru) > 0:
                        try:
                            # Always use regression (default behavior)
                            from sklearn.metrics import mean_squared_error, r2_score
                            y_true = y_sequences.cpu().numpy()
                            y_pred = y_pred_proba_gru
                            mse_gru = mean_squared_error(y_true, y_pred)
                            r2_gru = r2_score(y_true, y_pred)
                            rmse_gru = mse_gru ** 0.5  # Root Mean Squared Error
                            auc_gru = -mse_gru  # Negative MSE (higher is better for comparison)
                            current_gru_hyperparams = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout_rate": dropout_rate, "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs}
                            models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler, "y_scaler": y_scaler, "auc": auc_gru, "hyperparams": current_gru_hyperparams}
                            print(f"      📊 GRU Regression Metrics:")
                            print(f"         MSE: {mse_gru:.6f}")
                            print(f"         RMSE: {rmse_gru:.6f}")
                            print(f"         R² Score: {r2_gru:.4f} ({'Good' if r2_gru > 0.5 else 'Poor'} - {abs(r2_gru)*100:.1f}% variance explained)")
                            print(f"DEBUG: SAVE_PLOTS={SAVE_PLOTS}, SHAP_AVAILABLE={SHAP_AVAILABLE}")
                            if SAVE_PLOTS and SHAP_AVAILABLE:
                                analyze_shap_for_gru(gru_model, dl_scaler, X_df, final_feature_names, ticker, target_col)
                        except ValueError:
                            print(f"      GRU AUC (fixed/loaded params): Not enough samples with positive class for AUC calculation.")
                            models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler, "y_scaler": y_scaler, "auc": 0.0}
                    else:
                        models_and_params_local["GRU"] = {"model": None, "scaler": None, "y_scaler": None, "auc": 0.0}

    best_model_overall = None
    best_hyperparams_overall: Optional[Dict] = None  # New: To store GRU hyperparams if GRU is best
    # Always use regression (default behavior)
    best_mse_overall = np.inf  # lower is better
    best_auc_overall = None    # unused in regression mode
    
    # Always use KFold for regression (default behavior)
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    results = {} # Initialize the results dictionary here

    # Always use regression (default behavior)
    print("  🔬 Comparing regressor performance (MSE via cross-validation with GridSearchCV):")
    for name, mp in models_and_params_local.items():  # Iterate over local models_and_params
        if name in ["LSTM", "GRU", "TCN"]:
            # For DL models, we stored negative MSE in "auc" for compatibility.
            # Always use regression (default behavior)
            current_mse = -mp["auc"]  # convert back to positive MSE
            results[name] = current_mse
            print(f"    - {name}: MSE={current_mse:.4f}")
            if current_mse < best_mse_overall:
                best_mse_overall = current_mse
                best_model_overall = mp["model"]
                scaler = mp["scaler"]  # Use the DL scaler for DL models
                if name == "GRU":  # If GRU is the best, store its hyperparams
                    best_hyperparams_overall = mp.get("hyperparams")
        else:
            model = mp["model"]
            params = mp["params"]
            
            def _run_grid(estimator):
                gs = GridSearchCV(estimator, params, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
                gs.fit(X, y)
                return gs
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
                    
                    grid_search = _run_grid(model)
                    best_score = -grid_search.best_score_
                    results[name] = best_score
                    print(f"    - {name}: MSE={best_score:.4f} (Best Params: {grid_search.best_params_})")
                    if best_score < best_mse_overall:
                        best_mse_overall = best_score
                        best_model_overall = grid_search.best_estimator_
                        best_hyperparams_overall = None

            except Exception as e:
                # LightGBM GPU fallback to CPU if build/device fails at fit time
                if "LightGBM" in name and hasattr(model, "get_params"):
                    try:
                        print(f"    - {name}: GPU fit failed ({e}), retrying on CPU.")
                        model_cpu = model.__class__(**{**model.get_params(), "device": "cpu"})
                        grid_search = _run_grid(model_cpu)
                        best_score = -grid_search.best_score_
                        results[name] = best_score
                        print(f"    - {name} (CPU fallback): MSE={best_score:.4f} (Best Params: {grid_search.best_params_})")
                        if best_score < best_mse_overall:
                            best_mse_overall = best_score
                            best_model_overall = grid_search.best_estimator_
                            best_hyperparams_overall = None
                        continue
                    except Exception as e2:
                        print(f"    - {name}: CPU fallback also failed ({e2}).")
                print(f"    - {name}: Failed evaluation. Error: {e}")
                results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None, None, None

    # Select best model based on lowest MSE (regression) or highest AUC (classification)
    best_model_name = min(results, key=results.get)  # Lowest MSE wins
    best_score = results[best_model_name]
    print(f"  🏆 WINNER for {ticker} ({target_col}): {best_model_name} with MSE={best_score:.4f}")
    
    # Track model selection for statistics (store in a way that can be aggregated later)
    global _model_selection_stats
    key = f"{ticker}_{target_col}"
    _model_selection_stats[key] = best_model_name
    
    # Alpha metrics (validation-like quick check)
    try:
        if best_model_overall is not None and 'X_df' in locals() and hasattr(best_model_overall, 'predict_proba'):
            _proba = best_model_overall.predict_proba(X_df)[:, 1]
            _alpha_metrics_report("Best model (quick)", _proba, d if 'd' in locals() else X_df)
    except Exception as _e:
        print(f"    - Alpha metrics (quick) skipped: {_e}")

    # If the best model is a DL model, ensure its specific scaler is returned
    if best_model_name in ["LSTM", "GRU"]:
        y_scaler = models_and_params_local[best_model_name].get("y_scaler", None)
        return models_and_params_local[best_model_name]["model"], models_and_params_local[best_model_name]["scaler"], y_scaler, best_hyperparams_overall, best_model_name
    else:
        # Otherwise, return the best traditional ML model and the StandardScaler
        from sklearn.ensemble import RandomForestRegressor as RFRegressor
        try:
            from xgboost import XGBRegressor as XGBReg
        except ImportError:
            XGBReg = type(None)
        
        if SAVE_PLOTS and SHAP_AVAILABLE and isinstance(best_model_overall, (RandomForestClassifier, XGBClassifier, RFRegressor, XGBReg)):
            analyze_shap_for_tree_model(best_model_overall, X_df, final_feature_names, ticker, target_col)
        return best_model_overall, scaler, None, best_hyperparams_overall, best_model_name  # None for y_scaler (not used in traditional ML)

def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_path = models_dir / f"{ticker}_model.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_path = models_dir / f"{ticker}_TargetReturn_gru_optimized_params.json"

    model, scaler = None, None
    
    # Flag to indicate if we successfully loaded a model to continue training
    loaded_for_retraining = False

    # Attempt to load model and GRU hyperparams if CONTINUE_TRAINING_FROM_EXISTING is True
    if CONTINUE_TRAINING_FROM_EXISTING and model_path.exists() and scaler_path.exists():
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if gru_hyperparams_path.exists():
                with open(gru_hyperparams_path, 'r') as f:
                    loaded_gru_hyperparams = json.load(f)

            print(f"  ✅ Loaded existing model and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

    # If FORCE_TRAINING is False and we didn't load for retraining, then we just load and skip training
    if not FORCE_TRAINING and not loaded_for_retraining and model_path.exists() and scaler_path.exists():
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            if gru_hyperparams_path.exists():
                with open(gru_hyperparams_path, 'r') as f:
                    loaded_gru_hyperparams = json.load(f)

            print(f"  ✅ Loaded existing model and GRU hyperparams for {ticker} (FORCE_TRAINING is False).")
            return {
                'ticker': ticker,
                'model': model,
                'scaler': scaler,
                'gru_hyperparams': loaded_gru_hyperparams,
                'status': 'loaded',
                'reason': None
            }
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker}: {e}. Training from scratch.")
            # Fall through to training from scratch if loading fails

    print(f"  ⚙️ Training models for {ticker} (FORCE_TRAINING is {FORCE_TRAINING}, CONTINUE_TRAINING_FROM_EXISTING is {CONTINUE_TRAINING_FROM_EXISTING})...")
    print(f"  [DEBUG] {current_process().name} - {ticker}: Initiating feature extraction for training.")
    
    df_train, actual_feature_set = fetch_training_data(ticker, df_train_period, target_percentage)

    if df_train.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data.")
        return {'ticker': ticker, 'model': None, 'scaler': None}

    # Train single regression model for both buy and sell decisions
    model, scaler, gru_hyperparams = train_and_evaluate_models(df_train, "TargetReturn", actual_feature_set, ticker=ticker, initial_model=model_buy if loaded_for_retraining else None, loaded_gru_hyperparams=loaded_gru_hyperparams_buy)

    # Use same model for both buy and sell decisions (single model approach)
    model_buy = model_sell = model
    scaler_buy = scaler_sell = scaler
    gru_hyperparams_buy = gru_hyperparams_sell = gru_hyperparams

    final_scaler = scaler

    if model and final_scaler:
        try:
            joblib.dump(model, model_path)
            joblib.dump(final_scaler, scaler_path)

            if gru_hyperparams:
                with open(gru_hyperparams_path, 'w') as f:
                    json.dump(gru_hyperparams, f, indent=4)

            print(f"  ✅ Model, scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving model or GRU hyperparams for {ticker}: {e}")
            
        return {
            'ticker': ticker,
            'model': model,
            'scaler': final_scaler,
            'gru_hyperparams': gru_hyperparams,
            'status': 'trained',
            'reason': None
        }
    else:
        reason = "Insufficient training data" # Default reason
        if df_train.empty:
            reason = f"Insufficient training data (initial rows: {len(df_train_period)})"
        elif len(df_train) < 50:
            reason = f"Not enough rows after feature prep ({len(df_train)} rows, need >= 50)"
        
        print(f"  ❌ Failed to train models for {ticker}. Reason: {reason}")
        return {'ticker': ticker, 'model': None, 'scaler': None, 'status': 'failed', 'reason': reason}
