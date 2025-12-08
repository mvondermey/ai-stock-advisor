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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
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
    ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION, SAVE_PLOTS, MIN_PROBA_BUY, MIN_PROBA_SELL,
    FORCE_TRAINING, CONTINUE_TRAINING_FROM_EXISTING, FORCE_PERCENTAGE_OPTIMIZATION,
    USE_LOGISTIC_REGRESSION, USE_RANDOM_FOREST, USE_SVM, USE_MLP_CLASSIFIER,
    USE_LSTM, USE_GRU, USE_LIGHTGBM, USE_XGBOOST,
    USE_REGRESSION_MODEL
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
# Cache GPU availability to avoid retesting in worker processes
_lgbm_gpu_available = None
_xgb_gpu_available = None
cuMLRandomForestClassifier = None
cuMLLogisticRegression = None
cuMLStandardScaler = None
models_and_params: Dict = {}

# Helper function (copied from main.py)
# _ensure_dir moved to utils.py to avoid duplication
from utils import _ensure_dir

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

def initialize_ml_libraries():
    """Initializes ML libraries and prints their status only once."""
    global _ml_libraries_initialized, CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier, XGBClassifier, models_and_params, \
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
                CUDA_AVAILABLE = True
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print("✅ CUDA is available. GPU acceleration enabled with deterministic algorithms.")
            else:
                CUDA_AVAILABLE = False
                if USE_LSTM or USE_GRU:
                    print("⚠️ CUDA is not available. GPU acceleration will not be used.")
        elif PYTORCH_AVAILABLE:
            # PyTorch is available but LSTM/GRU are not enabled, still check CUDA for other models
            if torch.cuda.is_available():
                CUDA_AVAILABLE = True
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
            # LightGBM GPU requires OpenCL, not CUDA. If CUDA is available, try GPU (OpenCL might be available too).
            # If GPU fails during training, it will be caught by error handling.
            if CUDA_AVAILABLE:
                lgbm_model_params = {
                    "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='gpu'),
                    "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
                }
                models_and_params["LightGBM (GPU)"] = lgbm_model_params
                print("✅ LightGBM found. Configured for GPU (OpenCL).")
            else:
                lgbm_model_params = {
                    "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='cpu'),
                    "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
                }
                models_and_params["LightGBM (CPU)"] = lgbm_model_params
                print("ℹ️ LightGBM found. Will use CPU (CUDA not available).")
        except ImportError:
            print("⚠️ lightgbm not installed. Run: pip install lightgbm. It will be skipped.")

    if USE_XGBOOST and XGBOOST_AVAILABLE:
        XGBClassifier = xgb.XGBClassifier
        xgb_model_params = {
            "model": XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=1),
            "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        }
        if CUDA_AVAILABLE:
            xgb_model_params["model"].set_params(tree_method='gpu_hist')
            models_and_params["XGBoost (GPU)"] = xgb_model_params
            print("✅ XGBoost found. Configured for GPU (gpu_hist tree_method).")
        else:
            xgb_model_params["model"].set_params(tree_method='hist')
            models_and_params["XGBoost (CPU)"] = xgb_model_params
            print("ℹ️ XGBoost found. Will use CPU (CUDA not available).")

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
    target_col: str = "TargetClassBuy",
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
        final_feature_names = [col for col in d.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell", "TargetReturnBuy", "TargetReturnSell"]]
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

    # For classification, check class balance. For regression, skip this check.
    if USE_REGRESSION_MODEL:
        # Regression: no class balance needed, just check we have enough samples
        if len(y) < 10:
            print(f"  [DIAGNOSTIC] {ticker}: Not enough samples for '{target_col}' (only {len(y)} samples, need >= 10). Skipping.")
            return None, None, None
        n_splits = min(3, len(y) // 5)  # Use 3-fold or less if dataset is tiny
    else:
        # Classification: check class balance
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            print(f"  [DIAGNOSTIC] {ticker}: Not enough class diversity for '{target_col}' (only 1 class found: {unique_classes}). Skipping.")
            return None, None, None
        
        # Use 3-fold CV for small datasets to allow training with fewer examples
        n_splits = min(3, min(counts))  # Adaptive: use 3-fold CV, or less if needed
        min_samples_required = 2  # Minimum 2 samples per class
        if any(c < min_samples_required for c in counts):
            print(f"  [DIAGNOSTIC] {ticker}: Least populated class in '{target_col}' has {min(counts)} members (needs >= {min_samples_required}). Skipping.")
            return None, None, None

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
    if USE_RANDOM_FOREST:
        models_and_params_local["Random Forest"] = {
            "model": RandomForestClassifier(random_state=SEED, class_weight="balanced"),
            "params": {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, None]}
        }
    if USE_SVM:
        models_and_params_local["SVM"] = {
            "model": SVC(probability=True, random_state=SEED, class_weight="balanced"),
            "params": {'C': [0.1, 1.0, 10.0, 100.0], 'kernel': ['rbf', 'linear']}
        }
    if USE_MLP_CLASSIFIER:
        models_and_params_local["MLPClassifier"] = {
            "model": MLPClassifier(random_state=SEED, max_iter=500, early_stopping=True),
            "params": {'hidden_layer_sizes': [(100,), (100, 50), (50, 25)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate_init': [0.001, 0.01]}
        }

    if USE_LIGHTGBM and LGBMClassifier:
        # LightGBM GPU requires OpenCL, not CUDA. If CUDA is available, try GPU (OpenCL might be available too).
        # If GPU fails during training, it will be caught by error handling.
        if CUDA_AVAILABLE:
            lgbm_model_params = {
                "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='gpu'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
            }
            models_and_params_local["LightGBM (GPU)"] = lgbm_model_params
        else:
            lgbm_model_params = {
                "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='cpu'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
            }
            models_and_params_local["LightGBM (CPU)"] = lgbm_model_params

    if USE_XGBOOST and XGBOOST_AVAILABLE and XGBClassifier:
        if CUDA_AVAILABLE:
            xgb_model_params = {
                "model": XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=1, tree_method='gpu_hist'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            }
            models_and_params_local["XGBoost (GPU)"] = xgb_model_params
        else:
            xgb_model_params = {
                "model": XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=1, tree_method='hist'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            }
            models_and_params_local["XGBoost (CPU)"] = xgb_model_params

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
            
            # Choose loss function based on model type
            if USE_REGRESSION_MODEL:
                criterion = nn.MSELoss()  # Mean Squared Error for regression
                print(f"    - Using MSE loss for regression (predicting returns)")
            else:
                criterion = nn.BCELoss()  # Binary Cross Entropy for classification
                print(f"    - Using BCE loss for classification (predicting up/down)")

            if USE_LSTM:
                lstm_model = LSTMClassifier(input_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 1, LSTM_DROPOUT).to(device)
                if initial_model and isinstance(initial_model, LSTMClassifier):
                    try:
                        lstm_model.load_state_dict(initial_model.state_dict())
                        print(f"    - Loaded existing LSTM model state for {ticker} to continue training.")
                    except Exception as e:
                        print(f"    - Error loading LSTM model state for {ticker}: {e}. Training from scratch.")
                
                optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LSTM_LEARNING_RATE)

                for epoch in range(LSTM_EPOCHS):
                    for batch_X, batch_y in dataloader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer_lstm.zero_grad()
                        outputs = lstm_model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer_lstm.step()
                
                lstm_model.eval()
                with torch.no_grad():
                    all_outputs = []
                    for batch_X, _ in dataloader:
                        batch_X = batch_X.to(device)
                        outputs = lstm_model(batch_X)
                        all_outputs.append(outputs.cpu().numpy())
                    y_pred_proba_lstm = np.concatenate(all_outputs).flatten()
                
                try:
                    if USE_REGRESSION_MODEL:
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
                    else:
                        from sklearn.metrics import roc_auc_score
                        auc_lstm = roc_auc_score(y_sequences.cpu().numpy(), y_pred_proba_lstm)
                        models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler, "auc": auc_lstm}
                        print(f"      LSTM AUC (classification): {auc_lstm:.4f}")
                except ValueError:
                    models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler, "auc": 0.0}

            if USE_GRU:
                if perform_gru_hp_optimization and ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION:
                    print(f"    - Starting GRU hyperparameter optimization for {ticker} ({target_col}) (HP_OPT={perform_gru_hp_optimization}, ENABLE_HP_OPT={ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION}, Target={default_target_percentage:.4f}, Horizon={default_class_horizon})...")
                    best_gru_auc = -np.inf
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
                    best_gru_auc = -np.inf
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

                            # Choose model based on regression vs classification
                            if USE_REGRESSION_MODEL:
                                gru_model = GRURegressor(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                            else:
                                gru_model = GRUClassifier(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                            optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
                            
                            current_dataloader = DataLoader(dataset, batch_size=temp_hyperparams["batch_size"], shuffle=True)

                            for epoch in range(temp_hyperparams["epochs"]):
                                for batch_X, batch_y in current_dataloader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer_gru.zero_grad()
                                    outputs = gru_model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    optimizer_gru.step()
                            
                            gru_model.eval()
                            with torch.no_grad():
                                all_outputs = []
                                for batch_X, _ in current_dataloader:
                                    batch_X = batch_X.to(device)
                                    outputs = gru_model(batch_X)
                                    all_outputs.append(outputs.cpu().numpy())
                                y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                            try:
                                if USE_REGRESSION_MODEL:
                                    from sklearn.metrics import mean_squared_error, r2_score
                                    y_true = y_sequences.cpu().numpy()
                                    y_pred = y_pred_proba_gru
                                    mse_gru = mean_squared_error(y_true, y_pred)
                                    r2_gru = r2_score(y_true, y_pred)
                                    auc_gru = -mse_gru  # Negative MSE (higher is better for comparison)
                                    print(f"            GRU MSE: {mse_gru:.6f}, R²: {r2_gru:.4f} | {param_name}={value} (HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']})")
                                else:
                                    from sklearn.metrics import roc_auc_score
                                    auc_gru = roc_auc_score(y_sequences.cpu().numpy(), y_pred_proba_gru)
                                    print(f"            GRU AUC: {auc_gru:.4f} | {param_name}={value} (HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']})")

                                if auc_gru > best_gru_auc:
                                    best_gru_auc = auc_gru
                                    best_gru_model = gru_model
                                    best_gru_scaler = dl_scaler # dl_scaler is already fitted
                                    best_gru_hyperparams = temp_hyperparams.copy() # Update best_gru_hyperparams
                            except ValueError:
                                print(f"            GRU AUC: Not enough samples with positive class for AUC calculation. | {param_name}={value} (HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']})")
                                
                    if best_gru_model:
                        models_and_params_local["GRU"] = {"model": best_gru_model, "scaler": best_gru_scaler, "y_scaler": y_scaler, "auc": best_gru_auc, "hyperparams": best_gru_hyperparams}
                        print(f"      Best GRU found for {ticker} ({target_col}) with AUC: {best_gru_auc:.4f}, Hyperparams: {best_gru_hyperparams}")
                        print(f"DEBUG: SAVE_PLOTS={SAVE_PLOTS}, SHAP_AVAILABLE={SHAP_AVAILABLE}")
                        if SAVE_PLOTS and SHAP_AVAILABLE:
                            analyze_shap_for_gru(best_gru_model, best_gru_scaler, X_df, final_feature_names, ticker, target_col)
                    else:
                        models_and_params_local["GRU"] = {"model": None, "scaler": None, "y_scaler": None, "auc": 0.0}
                else: # ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION is False, use fixed or loaded hyperparameters
                    if loaded_gru_hyperparams and not FORCE_PERCENTAGE_OPTIMIZATION:
                        # Use loaded hyperparams only if FORCE_PERCENTAGE_OPTIMIZATION is False
                        print(f"    - Training GRU for {ticker} ({target_col}) with loaded hyperparameters (HP_OPT={perform_gru_hp_optimization}, ENABLE_HP_OPT={ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION})...")
                        hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE)
                        num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS)
                        dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT)
                        learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE)
                        batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE)
                        epochs = LSTM_EPOCHS
                        print(f"      Loaded GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, Target={default_target_percentage:.4f}, Horizon={default_class_horizon}")
                    elif loaded_gru_hyperparams and FORCE_PERCENTAGE_OPTIMIZATION:
                        # FORCE_PERCENTAGE_OPTIMIZATION is True: Use model architecture from loaded, but Target/Horizon from config
                        print(f"    - Training GRU for {ticker} ({target_col}) with loaded model architecture but FORCED config Target/Horizon (FORCE_PCT_OPT=True)...")
                        hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE)
                        num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS)
                        dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT)
                        learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE)
                        batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE)
                        epochs = LSTM_EPOCHS
                        # *** KEY FIX: Use config values for Target and Horizon, not loaded ones ***
                        print(f"      Forced Config: Target={default_target_percentage:.4f}, Horizon={default_class_horizon} (ignoring loaded Target/Horizon)")
                        print(f"      Loaded Model Arch: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}")
                    else:
                        print(f"    - Training GRU for {ticker} ({target_col}) with default fixed hyperparameters (HP_OPT={perform_gru_hp_optimization}, ENABLE_HP_OPT={ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION})...")
                        hidden_size = LSTM_HIDDEN_SIZE
                        num_layers = LSTM_NUM_LAYERS
                        dropout_rate = LSTM_DROPOUT
                        learning_rate = LSTM_LEARNING_RATE
                        batch_size = LSTM_BATCH_SIZE
                        epochs = LSTM_EPOCHS
                        print(f"      Default GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, Target={default_target_percentage:.4f}, Horizon={default_class_horizon}")

                    # Choose model based on regression vs classification
                    if USE_REGRESSION_MODEL:
                        gru_model = GRURegressor(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                        if initial_model and isinstance(initial_model, GRURegressor):
                            try:
                                gru_model.load_state_dict(initial_model.state_dict())
                                print(f"    - Loaded existing GRU regressor state for {ticker} to continue training.")
                            except Exception as e:
                                print(f"    - Error loading GRU regressor state for {ticker}: {e}. Training from scratch.")
                    else:
                        gru_model = GRUClassifier(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                        if initial_model and isinstance(initial_model, GRUClassifier):
                            try:
                                gru_model.load_state_dict(initial_model.state_dict())
                                print(f"    - Loaded existing GRU model state for {ticker} to continue training.")
                            except Exception as e:
                                print(f"    - Error loading GRU model state for {ticker}: {e}. Training from scratch.")
                    
                    optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)
                    
                    # Create DataLoader for current batch_size
                    current_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                    for epoch in range(epochs):
                        for batch_X, batch_y in current_dataloader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            optimizer_gru.zero_grad()
                            outputs = gru_model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer_gru.step()
                    
                    # Evaluate GRU
                    gru_model.eval()
                    with torch.no_grad():
                        all_outputs = []
                        for batch_X, _ in current_dataloader:
                            batch_X = batch_X.to(device)
                            outputs = gru_model(batch_X)
                            all_outputs.append(outputs.cpu().numpy())
                        y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                    try:
                        if USE_REGRESSION_MODEL:
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
                        else:
                            from sklearn.metrics import roc_auc_score
                            auc_gru = roc_auc_score(y_sequences.cpu().numpy(), y_pred_proba_gru)
                            current_gru_hyperparams = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout_rate": dropout_rate, "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs}
                            models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler, "y_scaler": y_scaler, "auc": auc_gru, "hyperparams": current_gru_hyperparams}
                            print(f"      GRU AUC (classification, fixed/loaded params): {auc_gru:.4f}")
                        print(f"DEBUG: SAVE_PLOTS={SAVE_PLOTS}, SHAP_AVAILABLE={SHAP_AVAILABLE}")
                        if SAVE_PLOTS and SHAP_AVAILABLE:
                            analyze_shap_for_gru(gru_model, dl_scaler, X_df, final_feature_names, ticker, target_col)
                    except ValueError:
                        print(f"      GRU AUC (fixed/loaded params): Not enough samples with positive class for AUC calculation.")
                        models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler, "y_scaler": y_scaler, "auc": 0.0}

    best_model_overall = None
    best_auc_overall = -np.inf
    best_hyperparams_overall: Optional[Dict] = None # New: To store GRU hyperparams if GRU is best
    
    # Use KFold for regression, StratifiedKFold for classification
    if USE_REGRESSION_MODEL:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    results = {} # Initialize the results dictionary here

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation with GridSearchCV):")
    for name, mp in models_and_params_local.items(): # Iterate over local models_and_params
        if name in ["LSTM", "GRU"]:
            # For DL models, we already have AUC from direct training
            current_auc = mp["auc"]
            results[name] = current_auc
            print(f"    - {name}: {current_auc:.4f}")
            if current_auc > best_auc_overall:
                best_auc_overall = current_auc
                best_model_overall = mp["model"]
                scaler = mp["scaler"] # Use the DL scaler for DL models
                if name == "GRU": # If GRU is the best, store its hyperparams
                    best_hyperparams_overall = mp.get("hyperparams")
        else:
            model = mp["model"]
            params = mp["params"]
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
                    
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
                    grid_search.fit(X, y)
                    
                    best_score = grid_search.best_score_
                    results[name] = best_score
                    print(f"    - {name}: {best_score:.4f} (Best Params: {grid_search.best_params_})")

                    if best_score > best_auc_overall:
                        best_auc_overall = best_score
                        best_model_overall = grid_search.best_estimator_ # Store the best estimator from GridSearchCV
                        best_hyperparams_overall = None # Reset if a non-GRU model is best

            except Exception as e:
                print(f"    - {name}: Failed evaluation. Error: {e}")
                results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None, None, None

    best_model_name = max(results, key=results.get)
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
        return models_and_params_local[best_model_name]["model"], models_and_params_local[best_model_name]["scaler"], y_scaler, best_hyperparams_overall
    else:
        # Otherwise, return the best traditional ML model and the StandardScaler
        if SAVE_PLOTS and SHAP_AVAILABLE and isinstance(best_model_overall, (RandomForestClassifier, XGBClassifier)):
            analyze_shap_for_tree_model(best_model_overall, X_df, final_feature_names, ticker, target_col)
        return best_model_overall, scaler, None, best_hyperparams_overall  # None for y_scaler (not used in traditional ML)

def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    model_sell_path = models_dir / f"{ticker}_model_sell.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_buy_path = models_dir / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
    gru_hyperparams_sell_path = models_dir / f"{ticker}_TargetClassSell_gru_optimized_params.json"

    model_buy, model_sell, scaler = None, None, None
    
    # Flag to indicate if we successfully loaded a model to continue training
    loaded_for_retraining = False

    # Attempt to load models and GRU hyperparams if CONTINUE_TRAINING_FROM_EXISTING is True
    if CONTINUE_TRAINING_FROM_EXISTING and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

    # If FORCE_TRAINING is False and we didn't load for retraining, then we just load and skip training
    if not FORCE_TRAINING and not loaded_for_retraining and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                with open(gru_hyperparams_sell_path, 'r') as f:
                    loaded_gru_hyperparams_sell = json.load(f)

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} (FORCE_TRAINING is False).")
            return {
                'ticker': ticker,
                'model_buy': model_buy,
                'model_sell': model_sell,
                'scaler': scaler,
                'gru_hyperparams_buy': loaded_gru_hyperparams_buy,
                'gru_hyperparams_sell': loaded_gru_hyperparams_sell,
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
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for BUY target.")
    # Train BUY model, passing the potentially loaded model and GRU hyperparams
    model_buy, scaler_buy, gru_hyperparams_buy = train_and_evaluate_models(df_train, "TargetClassBuy", actual_feature_set, ticker=ticker, initial_model=model_buy if loaded_for_retraining else None, loaded_gru_hyperparams=loaded_gru_hyperparams_buy)
    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for SELL target.")
    # Train SELL model, passing the potentially loaded model and GRU hyperparams
    model_sell, scaler_sell, gru_hyperparams_sell = train_and_evaluate_models(df_train, "TargetClassSell", actual_feature_set, ticker=ticker, initial_model=model_sell if loaded_for_retraining else None, loaded_gru_hyperparams=loaded_gru_hyperparams_sell)

    # For simplicity, we'll use the scaler from the buy model for both if they are different.
    # In a more complex scenario, you might want to ensure feature_set consistency or use separate scalers.
    final_scaler = scaler_buy if scaler_buy else scaler_sell

    if model_buy and model_sell and final_scaler:
        try:
            joblib.dump(model_buy, model_buy_path)
            joblib.dump(model_sell, model_sell_path)
            joblib.dump(final_scaler, scaler_path)
            
            if gru_hyperparams_buy:
                with open(gru_hyperparams_buy_path, 'w') as f:
                    json.dump(gru_hyperparams_buy, f, indent=4)
            if gru_hyperparams_sell:
                with open(gru_hyperparams_sell_path, 'w') as f:
                    json.dump(gru_hyperparams_sell, f, indent=4)

            print(f"  ✅ Models, scaler, and GRU hyperparams saved for {ticker}.")
        except Exception as e:
            print(f"  ⚠️ Error saving models or GRU hyperparams for {ticker}: {e}")
            
        return {
            'ticker': ticker,
            'model_buy': model_buy,
            'model_sell': model_sell,
            'scaler': final_scaler,
            'gru_hyperparams_buy': gru_hyperparams_buy,
            'gru_hyperparams_sell': gru_hyperparams_sell,
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
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'status': 'failed', 'reason': reason}
