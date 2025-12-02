import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import from ml_models and utils
from ml_models import LSTMClassifier, GRUClassifier, PYTORCH_AVAILABLE, CUDA_AVAILABLE, SHAP_AVAILABLE
from utils import _ensure_dir

# Scikit-learn imports for tree models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb # Assuming XGBClassifier is imported or defined in ml_models

def analyze_shap_for_gru(model: GRUClassifier, scaler: nn.Module, X_df: pd.DataFrame, feature_names: List[str], ticker: str, target_col: str, sequence_length: int):
    """
    Calculates and visualizes SHAP values for a GRU model.
    """
    if not SHAP_AVAILABLE:
        print(f"  [{ticker}] SHAP is not available. Skipping SHAP analysis.")
        return

    if isinstance(model, GRUClassifier):
        print(f"  [{ticker}] SHAP KernelExplainer is not directly compatible with GRU models due to sequential input. Skipping SHAP analysis for GRU.")
        return

    print(f"  [{ticker}] Calculating SHAP values for GRU model ({target_col})...")
    
    try:
        def gru_predict_proba_wrapper_for_kernel(X_unsequenced_np):
            X_scaled_np = scaler.transform(X_unsequenced_np)
            
            X_sequences_for_pred = []
            if len(X_scaled_np) < sequence_length:
                return np.full(len(X_unsequenced_np), 0.5)

            for i in range(len(X_scaled_np) - sequence_length + 1):
                X_sequences_for_pred.append(X_scaled_np[i:i + sequence_length])
            
            if not X_sequences_for_pred:
                return np.full(len(X_unsequenced_np), 0.5)

            X_sequences_tensor = torch.tensor(np.array(X_sequences_for_pred), dtype=torch.float32)
            
            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            model.to(device)
            X_sequences_tensor = X_sequences_tensor.to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(X_sequences_tensor)
                return torch.sigmoid(outputs).cpu().numpy().flatten()

        num_background_samples = min(50, len(X_df))
        background_data_for_kernel = X_df.sample(num_background_samples, random_state=42).values

        num_explain_samples = min(20, len(X_df))
        explain_data_for_kernel = X_df.sample(num_explain_samples, random_state=42).values

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
        explain_data = X_df.sample(num_explain_samples, random_state=42)

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
