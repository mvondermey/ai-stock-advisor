import json
import random
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neural_network import MLPClassifier

# Import from config, ml_models, feature_engineering, ml_analysis, and utils
from config import (
    SEED, SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    GRU_HIDDEN_SIZE_OPTIONS, GRU_NUM_LAYERS_OPTIONS, GRU_DROPOUT_OPTIONS,
    GRU_LEARNING_RATE_OPTIONS, GRU_BATCH_SIZE_OPTIONS, GRU_EPOCHS_OPTIONS,
    GRU_CLASS_HORIZON_OPTIONS, GRU_TARGET_PERCENTAGE_OPTIONS,
    ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION,
    USE_LOGISTIC_REGRESSION, USE_SVM, USE_MLP_CLASSIFIER, USE_LIGHTGBM,
    USE_XGBOOST, USE_LSTM, USE_GRU, USE_RANDOM_FOREST,
    TARGET_PERCENTAGE, CLASS_HORIZON, SAVE_PLOTS, FORCE_TRAINING,
    CONTINUE_TRAINING_FROM_EXISTING
)
from ml_models import (
    initialize_ml_libraries, LSTMClassifier, GRUClassifier, PYTORCH_AVAILABLE,
    CUDA_AVAILABLE, CUML_AVAILABLE, LGBMClassifier, XGBClassifier,
    cuMLRandomForestClassifier, cuMLLogisticRegression, cuMLStandardScaler, SHAP_AVAILABLE
)
from feature_engineering import fetch_training_data
from ml_analysis import analyze_shap_for_gru, analyze_shap_for_tree_model
from utils import _ensure_dir

def optimize_gru_hyperparameters_for_ticker(
    ticker: str,
    raw_df_dl: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    target_col: str,
    default_target_percentage: float,
    default_class_horizon: int,
    loaded_gru_hyperparams: Optional[Dict] = None
) -> Tuple[Optional[GRUClassifier], Optional[MinMaxScaler], Optional[Dict]]:
    """
    Performs GRU hyperparameter optimization for a single ticker and returns the best model, scaler, and hyperparams.
    """
    print(f"    - Starting GRU hyperparameter randomized search for {ticker} ({target_col})...")
    best_gru_auc = -np.inf
    best_gru_model = None
    best_gru_scaler = None
    best_gru_hyperparams = {}
    
    gru_param_distributions = {
        "hidden_size": GRU_HIDDEN_SIZE_OPTIONS,
        "num_layers": GRU_NUM_LAYERS_OPTIONS,
        "dropout_rate": GRU_DROPOUT_OPTIONS,
        "learning_rate": GRU_LEARNING_RATE_OPTIONS,
        "batch_size": GRU_BATCH_SIZE_OPTIONS,
        "epochs": GRU_EPOCHS_OPTIONS,
        "class_horizon": GRU_CLASS_HORIZON_OPTIONS,
        "target_percentage": GRU_TARGET_PERCENTAGE_OPTIONS
    }
    
    n_trials = 20

    print(f"      GRU Hyperparameter Randomized Search for {ticker} ({target_col}) with {n_trials} trials:")

    for trial in range(n_trials):
        temp_hyperparams = {
            "hidden_size": random.choice(gru_param_distributions["hidden_size"]),
            "num_layers": random.choice(gru_param_distributions["num_layers"]),
            "dropout_rate": random.choice(gru_param_distributions["dropout_rate"]),
            "learning_rate": random.choice(gru_param_distributions["learning_rate"]),
            "batch_size": random.choice(gru_param_distributions["batch_size"]),
            "epochs": random.choice(gru_param_distributions["epochs"]),
            "class_horizon": random.choice(gru_param_distributions["class_horizon"]),
            "target_percentage": random.choice(gru_param_distributions["target_percentage"])
        }
        
        current_dropout_rate = temp_hyperparams["dropout_rate"] if temp_hyperparams["num_layers"] > 1 else 0.0
        temp_hyperparams["dropout_rate"] = current_dropout_rate

        print(f"          Testing GRU (Trial {trial + 1}/{n_trials}) with: HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']}, CH={temp_hyperparams['class_horizon']}, TP={temp_hyperparams['target_percentage']:.4f}")

        df_train_trial, actual_feature_set_trial = fetch_training_data(ticker, raw_df_dl.loc[train_start:train_end].copy(), temp_hyperparams["target_percentage"], temp_hyperparams["class_horizon"])
        
        if df_train_trial.empty:
            print(f"          [DIAGNOSTIC] {ticker}: Insufficient training data for GRU trial. Skipping.")
            continue
        
        X_df_trial = df_train_trial[actual_feature_set_trial]
        y_trial = df_train_trial[target_col].values

        if len(X_df_trial) < SEQUENCE_LENGTH + 1:
            print(f"          [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in GRU trial (need > {SEQUENCE_LENGTH} rows). Skipping.")
            continue

        dl_scaler_trial = MinMaxScaler(feature_range=(0, 1))
        X_scaled_dl_trial = dl_scaler_trial.fit_transform(X_df_trial)
        
        X_sequences_trial = []
        y_sequences_trial = []
        for i in range(len(X_scaled_dl_trial) - SEQUENCE_LENGTH):
            X_sequences_trial.append(X_scaled_dl_trial[i:i + SEQUENCE_LENGTH])
            y_sequences_trial.append(y_trial[i + SEQUENCE_LENGTH])
        
        if not X_sequences_trial:
            print(f"          [DIAGNOSTIC] {ticker}: Not enough data to create sequences for GRU trial. Skipping.")
            continue
        
        X_sequences_trial = torch.tensor(np.array(X_sequences_trial), dtype=torch.float32)
        y_sequences_trial = torch.tensor(np.array(y_sequences_trial), dtype=torch.float32).unsqueeze(1)

        input_size = X_sequences_trial.shape[2]
        
        device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        neg_count_trial = (y_sequences_trial == 0).sum()
        pos_count_trial = (y_sequences_trial == 1).sum()
        if pos_count_trial > 0 and neg_count_trial > 0:
            pos_weight_trial = torch.tensor([neg_count_trial / pos_count_trial], device=device, dtype=torch.float32)
            criterion_trial = nn.BCEWithLogitsLoss(pos_weight=pos_weight_trial)
        else:
            criterion_trial = nn.BCEWithLogitsLoss()

        gru_model = GRUClassifier(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
        optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
        
        current_dataloader = DataLoader(TensorDataset(X_sequences_trial, y_sequences_trial), batch_size=temp_hyperparams["batch_size"], shuffle=True)

        for epoch in range(temp_hyperparams["epochs"]):
            for batch_X, batch_y in current_dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer_gru.zero_grad()
                outputs = gru_model(batch_X)
                loss = criterion_trial(outputs, batch_y)
                loss.backward()
                optimizer_gru.step()
        
        gru_model.eval()
        with torch.no_grad():
            all_outputs = []
            for batch_X, _ in current_dataloader:
                batch_X = batch_X.to(device)
                outputs = gru_model(batch_X)
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            y_pred_proba_gru = np.concatenate(all_outputs).flatten()

        try:
            from sklearn.metrics import roc_auc_score
            auc_gru = roc_auc_score(y_sequences_trial.cpu().numpy(), y_pred_proba_gru)
            print(f"            GRU AUC: {auc_gru:.4f}")

            if auc_gru > best_gru_auc:
                best_gru_auc = auc_gru
                best_gru_model = gru_model
                best_gru_scaler = dl_scaler_trial
                best_gru_hyperparams = temp_hyperparams.copy()
        except ValueError:
            print(f"            GRU AUC: Not enough samples with positive class for AUC calculation.")
                                
    if best_gru_model:
        print(f"      Best GRU found for {ticker} ({target_col}) with AUC: {best_gru_auc:.4f}, Hyperparams: {best_gru_hyperparams}")
        return best_gru_model, best_gru_scaler, best_gru_hyperparams
    else:
        print(f"      No valid GRU model found after hyperparameter search for {ticker} ({target_col}).")
        return None, None, None

def train_and_evaluate_models(
    df: pd.DataFrame,
    target_col: str = "TargetClassBuy",
    feature_set: Optional[List[str]] = None,
    ticker: str = "UNKNOWN",
    initial_model=None,
    loaded_gru_hyperparams: Optional[Dict] = None,
    models_and_params_global: Optional[Dict] = None,
    perform_gru_hp_optimization: bool = False,
    default_target_percentage: float = TARGET_PERCENTAGE,
    default_class_horizon: int = CLASS_HORIZON
):
    """Train and compare multiple classifiers for a given target, returning the best one."""
    models_and_params = models_and_params_global if models_and_params_global is not None else initialize_ml_libraries()
    
    X_non_dl = pd.DataFrame()
    y_non_dl = np.array([])
    scaler_non_dl = None
    final_feature_names_non_dl = []

    if not (USE_GRU and perform_gru_hp_optimization):
        d = df.copy()
        if d.empty:
            print(f"  [DIAGNOSTIC] {ticker}: Input DataFrame for non-DL models is empty. Skipping non-DL models.")
        else:
            if feature_set is None:
                print("⚠️ feature_set was None for non-DL models. Inferring features from DataFrame columns.")
                final_feature_names_non_dl = [col for col in d.columns if col not in ["Target", "TargetClassBuy", "TargetClassSell"]]
            else:
                final_feature_names_non_dl = [f for f in feature_set if f in d.columns]
            
            required_cols_for_training_non_dl = final_feature_names_non_dl + [target_col]
            if not all(col in d.columns for col in required_cols_for_training_non_dl):
                missing = [col for col in required_cols_for_training_non_dl if col not in d.columns]
                print(f"⚠️ Missing critical columns for non-DL model training (target: {target_col}, missing: {missing}). Skipping non-DL models.")
            else:
                d = d[required_cols_for_training_non_dl].dropna()
                if len(d) < 50:
                    print(f"  [DIAGNOSTIC] {ticker}: Not enough rows after feature prep for non-DL models ({len(d)} rows, need >= 50). Skipping non-DL models.")
                else:
                    X_df_non_dl = d[final_feature_names_non_dl]
                    y_non_dl = d[target_col].values

                    if CUML_AVAILABLE and cuMLStandardScaler:
                        try:
                            scaler_non_dl = cuMLStandardScaler()
                            X_gpu_np = X_df_non_dl.values
                            X_scaled_non_dl = scaler_non_dl.fit_transform(X_gpu_np)
                            X_non_dl = pd.DataFrame(X_scaled_non_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                        except Exception as e:
                            print(f"⚠️ Error using cuML StandardScaler for non-DL models: {e}. Falling back to sklearn.StandardScaler.")
                            scaler_non_dl = StandardScaler()
                            X_scaled_non_dl = scaler_non_dl.fit_transform(X_df_non_dl)
                            X_non_dl = pd.DataFrame(X_scaled_non_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                    else:
                        scaler_non_dl = StandardScaler()
                        X_scaled_non_dl = scaler_non_dl.fit_transform(X_df_non_dl)
                        X_non_dl = pd.DataFrame(X_scaled_non_dl, columns=final_feature_names_non_dl, index=X_df_non_dl.index)
                    scaler_non_dl.feature_names_in_ = list(final_feature_names_non_dl)
    
    models_and_params_local = {} 

    if not X_non_dl.empty and len(y_non_dl) > 0:
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

        if LGBMClassifier and USE_LIGHTGBM:
            lgbm_model_params = {
                "model": LGBMClassifier(random_state=SEED, class_weight="balanced", verbosity=-1, device='cpu'),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
            }
            models_and_params_local["LightGBM (CPU)"] = lgbm_model_params
            print("ℹ️ LightGBM found. Will use CPU.")

        if XGBClassifier and USE_XGBOOST:
            xgb_device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            xgb_tree_method = 'gpu_hist' if CUDA_AVAILABLE else 'hist'
            xgb_predictor = 'gpu_predictor' if CUDA_AVAILABLE else 'cpu_predictor'
            xgb_model_params = {
                "model": XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, scale_pos_weight=1, tree_method=xgb_tree_method, predictor=xgb_predictor),
                "params": {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7]}
            }
            models_and_params_local[f"XGBoost ({xgb_device.upper()})"] = xgb_model_params
            print(f"ℹ️ XGBoost found. Will use {xgb_device.upper()}.")

    if PYTORCH_AVAILABLE and (USE_LSTM or USE_GRU):
        raw_df_dl = df.copy()

        if raw_df_dl.empty:
            print(f"  [DIAGNOSTIC] {ticker}: Input DataFrame for DL models is empty. Skipping DL models.")
        else:
            device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
            
            if USE_GRU:
                if perform_gru_hp_optimization and ENABLE_GRU_HYPERPARAMETER_OPTIMIZATION:
                    print(f"    - Starting GRU hyperparameter randomized search for {ticker} ({target_col})...")
                    best_gru_auc = -np.inf
                    best_gru_model = None
                    best_gru_scaler = None
                    best_gru_hyperparams = {}
                    
                    gru_param_distributions = {
                        "hidden_size": GRU_HIDDEN_SIZE_OPTIONS,
                        "num_layers": GRU_NUM_LAYERS_OPTIONS,
                        "dropout_rate": GRU_DROPOUT_OPTIONS,
                        "learning_rate": GRU_LEARNING_RATE_OPTIONS,
                        "batch_size": GRU_BATCH_SIZE_OPTIONS,
                        "epochs": GRU_EPOCHS_OPTIONS,
                        "class_horizon": GRU_CLASS_HORIZON_OPTIONS,
                        "target_percentage": GRU_TARGET_PERCENTAGE_OPTIONS
                    }
                    
                    n_trials = 20

                    print(f"      GRU Hyperparameter Randomized Search for {ticker} ({target_col}) with {n_trials} trials:")

                    for trial in range(n_trials):
                        temp_hyperparams = {
                            "hidden_size": random.choice(gru_param_distributions["hidden_size"]),
                            "num_layers": random.choice(gru_param_distributions["num_layers"]),
                            "dropout_rate": random.choice(gru_param_distributions["dropout_rate"]),
                            "learning_rate": random.choice(gru_param_distributions["learning_rate"]),
                            "batch_size": random.choice(gru_param_distributions["batch_size"]),
                            "epochs": random.choice(gru_param_distributions["epochs"]),
                            "class_horizon": random.choice(gru_param_distributions["class_horizon"]),
                            "target_percentage": random.choice(gru_param_distributions["target_percentage"])
                        }
                        
                        current_dropout_rate = temp_hyperparams["dropout_rate"] if temp_hyperparams["num_layers"] > 1 else 0.0
                        temp_hyperparams["dropout_rate"] = current_dropout_rate

                        print(f"          Testing GRU (Trial {trial + 1}/{n_trials}) with: HS={temp_hyperparams['hidden_size']}, NL={temp_hyperparams['num_layers']}, DO={temp_hyperparams['dropout_rate']:.2f}, LR={temp_hyperparams['learning_rate']:.5f}, BS={temp_hyperparams['batch_size']}, E={temp_hyperparams['epochs']}, CH={temp_hyperparams['class_horizon']}, TP={temp_hyperparams['target_percentage']:.4f}")

                        df_train_trial, actual_feature_set_trial = fetch_training_data(ticker, raw_df_dl.copy(), temp_hyperparams["target_percentage"], temp_hyperparams["class_horizon"])
                        
                        if df_train_trial.empty:
                            print(f"          [DIAGNOSTIC] {ticker}: Insufficient training data for GRU trial. Skipping.")
                            continue
                        
                        X_df_trial = df_train_trial[actual_feature_set_trial]
                        y_trial = df_train_trial[target_col].values

                        if len(X_df_trial) < SEQUENCE_LENGTH + 1:
                            print(f"          [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in GRU trial (need > {SEQUENCE_LENGTH} rows). Skipping.")
                            continue

                        dl_scaler_trial = MinMaxScaler(feature_range=(0, 1))
                        X_scaled_dl_trial = dl_scaler_trial.fit_transform(X_df_trial)
                        
                        X_sequences_trial = []
                        y_sequences_trial = []
                        for i in range(len(X_scaled_dl_trial) - SEQUENCE_LENGTH):
                            X_sequences_trial.append(X_scaled_dl_trial[i:i + SEQUENCE_LENGTH])
                            y_sequences_trial.append(y_trial[i + SEQUENCE_LENGTH])
                        
                        if not X_sequences_trial:
                            print(f"          [DIAGNOSTIC] {ticker}: Not enough data to create sequences for GRU trial. Skipping.")
                            continue
                        
                        X_sequences_trial = torch.tensor(np.array(X_sequences_trial), dtype=torch.float32)
                        y_sequences_trial = torch.tensor(np.array(y_sequences_trial), dtype=torch.float32).unsqueeze(1)

                        input_size = X_sequences_trial.shape[2]
                        
                        neg_count_trial = (y_sequences_trial == 0).sum()
                        pos_count_trial = (y_sequences_trial == 1).sum()
                        if pos_count_trial > 0 and neg_count_trial > 0:
                            pos_weight_trial = torch.tensor([neg_count_trial / pos_count_trial], device=device, dtype=torch.float32)
                            criterion_trial = nn.BCEWithLogitsLoss(pos_weight=pos_weight_trial)
                        else:
                            criterion_trial = nn.BCEWithLogitsLoss()

                        gru_model = GRUClassifier(input_size, temp_hyperparams["hidden_size"], temp_hyperparams["num_layers"], 1, temp_hyperparams["dropout_rate"]).to(device)
                        optimizer_gru = optim.Adam(gru_model.parameters(), lr=temp_hyperparams["learning_rate"])
                        
                        current_dataloader = DataLoader(TensorDataset(X_sequences_trial, y_sequences_trial), batch_size=temp_hyperparams["batch_size"], shuffle=True)

                        for epoch in range(temp_hyperparams["epochs"]):
                            for batch_X, batch_y in current_dataloader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                optimizer_gru.zero_grad()
                                outputs = gru_model(batch_X)
                                loss = criterion_trial(outputs, batch_y)
                                loss.backward()
                                optimizer_gru.step()
                        
                        gru_model.eval()
                        with torch.no_grad():
                            all_outputs = []
                            for batch_X, _ in current_dataloader:
                                batch_X = batch_X.to(device)
                                outputs = gru_model(batch_X)
                                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                            y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                        try:
                            from sklearn.metrics import roc_auc_score
                            auc_gru = roc_auc_score(y_sequences_trial.cpu().numpy(), y_pred_proba_gru)
                            print(f"            GRU AUC: {auc_gru:.4f}")

                            if auc_gru > best_gru_auc:
                                best_gru_auc = auc_gru
                                best_gru_model = gru_model
                                best_gru_scaler = dl_scaler_trial
                                best_gru_hyperparams = temp_hyperparams.copy()
                        except ValueError:
                            print(f"            GRU AUC: Not enough samples with positive class for AUC calculation.")
                                
                    if best_gru_model:
                        models_and_params_local["GRU"] = {"model": best_gru_model, "scaler": best_gru_scaler, "auc": best_gru_auc, "hyperparams": best_gru_hyperparams}
                        print(f"      Best GRU found for {ticker} ({target_col}) with AUC: {best_gru_auc:.4f}, Hyperparams: {best_gru_hyperparams}")
                        if SAVE_PLOTS and SHAP_AVAILABLE:
                            analyze_shap_for_gru(best_gru_model, best_gru_scaler, X_df_trial, actual_feature_set_trial, ticker, target_col, SEQUENCE_LENGTH)
                    else:
                        print(f"      No valid GRU model found after hyperparameter search for {ticker} ({target_col}).")
                        models_and_params_local["GRU"] = {"model": None, "scaler": None, "auc": 0.0}

                else:
                    if loaded_gru_hyperparams:
                        print(f"    - Training GRU for {ticker} ({target_col}) with loaded hyperparameters...")
                        hidden_size = loaded_gru_hyperparams.get("hidden_size", LSTM_HIDDEN_SIZE)
                        num_layers = loaded_gru_hyperparams.get("num_layers", LSTM_NUM_LAYERS)
                        dropout_rate = loaded_gru_hyperparams.get("dropout_rate", LSTM_DROPOUT)
                        learning_rate = loaded_gru_hyperparams.get("learning_rate", LSTM_LEARNING_RATE)
                        batch_size = loaded_gru_hyperparams.get("batch_size", LSTM_BATCH_SIZE)
                        epochs = loaded_gru_hyperparams.get("epochs", LSTM_EPOCHS)
                        class_horizon_fixed = loaded_gru_hyperparams.get("class_horizon", default_class_horizon)
                        target_percentage_fixed = loaded_gru_hyperparams.get("target_percentage", default_target_percentage)
                        print(f"      Loaded GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, CH={class_horizon_fixed}, TP={target_percentage_fixed:.4f}")
                    else:
                        print(f"    - Training GRU for {ticker} ({target_col}) with default fixed hyperparameters...")
                        hidden_size = LSTM_HIDDEN_SIZE
                        num_layers = LSTM_NUM_LAYERS
                        dropout_rate = LSTM_DROPOUT
                        learning_rate = LSTM_LEARNING_RATE
                        batch_size = LSTM_BATCH_SIZE
                        epochs = LSTM_EPOCHS
                        class_horizon_fixed = default_class_horizon
                        target_percentage_fixed = default_target_percentage
                        print(f"      Default GRU Hyperparams: HS={hidden_size}, NL={num_layers}, DO={dropout_rate}, LR={learning_rate}, BS={batch_size}, E={epochs}, CH={class_horizon_fixed}, TP={target_percentage_fixed:.4f}")

                    df_train_fixed_gru, actual_feature_set_fixed_gru = fetch_training_data(ticker, raw_df_dl.copy(), target_percentage_fixed, class_horizon_fixed)
                    
                    if df_train_fixed_gru.empty:
                        print(f"    [DIAGNOSTIC] {ticker}: Insufficient training data for fixed GRU. Skipping.")
                    else:
                        X_df_fixed_gru = df_train_fixed_gru[actual_feature_set_fixed_gru]
                        y_fixed_gru = df_train_fixed_gru[target_col].values

                        if len(X_df_fixed_gru) < SEQUENCE_LENGTH + 1:
                            print(f"    [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in fixed GRU (need > {SEQUENCE_LENGTH} rows). Skipping.")
                        else:
                            dl_scaler_fixed_gru = MinMaxScaler(feature_range=(0, 1))
                            X_scaled_dl_fixed_gru = dl_scaler_fixed_gru.fit_transform(X_df_fixed_gru)
                            
                            X_sequences_fixed_gru = []
                            y_sequences_fixed_gru = []
                            for i in range(len(X_scaled_dl_fixed_gru) - SEQUENCE_LENGTH):
                                X_sequences_fixed_gru.append(X_scaled_dl_fixed_gru[i:i + SEQUENCE_LENGTH])
                                y_sequences_fixed_gru.append(y_fixed_gru[i + SEQUENCE_LENGTH])
                            
                            if not X_sequences_fixed_gru:
                                print(f"    [DIAGNOSTIC] {ticker}: Not enough data to create sequences for fixed GRU. Skipping.")
                            else:
                                X_sequences_fixed_gru = torch.tensor(np.array(X_sequences_fixed_gru), dtype=torch.float32)
                                y_sequences_fixed_gru = torch.tensor(np.array(y_sequences_fixed_gru), dtype=torch.float32).unsqueeze(1)

                                input_size = X_sequences_fixed_gru.shape[2]
                                
                                neg_count_fixed_gru = (y_sequences_fixed_gru == 0).sum()
                                pos_count_fixed_gru = (y_sequences_fixed_gru == 1).sum()
                                if pos_count_fixed_gru > 0 and neg_count_fixed_gru > 0:
                                    pos_weight_fixed_gru = torch.tensor([neg_count_fixed_gru / pos_count_fixed_gru], device=device, dtype=torch.float32)
                                    criterion_fixed_gru = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fixed_gru)
                                else:
                                    criterion_fixed_gru = nn.BCEWithLogitsLoss()

                                gru_model = GRUClassifier(input_size, hidden_size, num_layers, 1, dropout_rate).to(device)
                                if initial_model and isinstance(initial_model, GRUClassifier):
                                    try:
                                        gru_model.load_state_dict(initial_model.state_dict())
                                        print(f"    - Loaded existing GRU model state for {ticker} to continue training.")
                                    except Exception as e:
                                        print(f"    - Error loading GRU model state for {ticker}: {e}. Training from scratch.")
                                
                                optimizer_gru = optim.Adam(gru_model.parameters(), lr=learning_rate)
                                
                                current_dataloader = DataLoader(TensorDataset(X_sequences_fixed_gru, y_sequences_fixed_gru), batch_size=batch_size, shuffle=True)

                                for epoch in range(epochs):
                                    for batch_X, batch_y in current_dataloader:
                                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                        optimizer_gru.zero_grad()
                                        outputs = gru_model(batch_X)
                                        loss = criterion_fixed_gru(outputs, batch_y)
                                        loss.backward()
                                        optimizer_gru.step()
                            
                            gru_model.eval()
                            with torch.no_grad():
                                all_outputs = []
                                for batch_X, _ in current_dataloader:
                                    batch_X = batch_X.to(device)
                                    outputs = gru_model(batch_X)
                                    all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                                y_pred_proba_gru = np.concatenate(all_outputs).flatten()

                            try:
                                from sklearn.metrics import roc_auc_score
                                auc_gru = roc_auc_score(y_sequences_fixed_gru.cpu().numpy(), y_pred_proba_gru)
                                current_gru_hyperparams = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout_rate": dropout_rate, "learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs, "class_horizon": class_horizon_fixed, "target_percentage": target_percentage_fixed}
                                models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler_fixed_gru, "auc": auc_gru, "hyperparams": current_gru_hyperparams}
                                print(f"      GRU AUC (fixed/loaded params): {auc_gru:.4f}")
                                if SAVE_PLOTS and SHAP_AVAILABLE:
                                    analyze_shap_for_gru(gru_model, dl_scaler_fixed_gru, X_df_fixed_gru, actual_feature_set_fixed_gru, ticker, target_col, SEQUENCE_LENGTH)
                            except ValueError:
                                print(f"      GRU AUC (fixed/loaded params): Not enough samples with positive class for AUC calculation.")
                                models_and_params_local["GRU"] = {"model": gru_model, "scaler": dl_scaler_fixed_gru, "auc": 0.0}
            if USE_LSTM:
                print(f"    - Training LSTM for {ticker}...")
                
                df_train_fixed_lstm, actual_feature_set_fixed_lstm = fetch_training_data(ticker, raw_df_dl.copy(), default_target_percentage, default_class_horizon)
                
                if df_train_fixed_lstm.empty:
                    print(f"    [DIAGNOSTIC] {ticker}: Insufficient training data for LSTM. Skipping.")
                else:
                    X_df_fixed_lstm = df_train_fixed_lstm[actual_feature_set_fixed_lstm]
                    y_fixed_lstm = df_train_fixed_lstm[target_col].values

                    if len(X_df_fixed_lstm) < SEQUENCE_LENGTH + 1:
                        print(f"    [DIAGNOSTIC] {ticker}: Not enough rows for sequencing in LSTM (need > {SEQUENCE_LENGTH} rows). Skipping.")
                    else:
                        dl_scaler_fixed_lstm = MinMaxScaler(feature_range=(0, 1))
                        X_scaled_dl_fixed_lstm = dl_scaler_fixed_lstm.fit_transform(X_df_fixed_lstm)
                        
                        X_sequences_fixed_lstm = []
                        y_sequences_fixed_lstm = []
                        for i in range(len(X_scaled_dl_fixed_lstm) - SEQUENCE_LENGTH):
                            X_sequences_fixed_lstm.append(X_scaled_dl_fixed_lstm[i:i + SEQUENCE_LENGTH])
                            y_sequences_fixed_lstm.append(y_fixed_lstm[i + SEQUENCE_LENGTH])
                        
                        if not X_sequences_fixed_lstm:
                            print(f"    [DIAGNOSTIC] {ticker}: Not enough data to create sequences for LSTM. Skipping.")
                        else:
                            X_sequences_fixed_lstm = torch.tensor(np.array(X_sequences_fixed_lstm), dtype=torch.float32)
                            y_sequences_fixed_lstm = torch.tensor(np.array(y_sequences_fixed_lstm), dtype=torch.float32).unsqueeze(1)

                            input_size = X_sequences_fixed_lstm.shape[2]
                            
                            neg_count_fixed_lstm = (y_sequences_fixed_lstm == 0).sum()
                            pos_count_fixed_lstm = (y_sequences_fixed_lstm == 1).sum()
                            if pos_count_fixed_lstm > 0 and neg_count_fixed_lstm > 0:
                                pos_weight_fixed_lstm = torch.tensor([neg_count_fixed_lstm / pos_count_fixed_lstm], device=device, dtype=torch.float32)
                                criterion_fixed_lstm = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fixed_lstm)
                            else:
                                criterion_fixed_lstm = nn.BCEWithLogitsLoss()

                            lstm_model = LSTMClassifier(input_size, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 1, LSTM_DROPOUT).to(device)
                            if initial_model and isinstance(initial_model, LSTMClassifier):
                                try:
                                    lstm_model.load_state_dict(initial_model.state_dict())
                                    print(f"    - Loaded existing LSTM model state for {ticker} to continue training.")
                                except Exception as e:
                                    print(f"    - Error loading LSTM model state for {ticker}: {e}. Training from scratch.")
                            
                            optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LSTM_LEARNING_RATE)
                            
                            current_dataloader = DataLoader(TensorDataset(X_sequences_fixed_lstm, y_sequences_fixed_lstm), batch_size=LSTM_BATCH_SIZE, shuffle=True)

                            for epoch in range(LSTM_EPOCHS):
                                for batch_X, batch_y in current_dataloader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer_lstm.zero_grad()
                                    outputs = lstm_model(batch_X)
                                    loss = criterion_fixed_lstm(outputs, batch_y)
                                    loss.backward()
                                    optimizer_lstm.step()
                            
                                lstm_model.eval()
                                with torch.no_grad():
                                    all_outputs = []
                                    for batch_X, _ in current_dataloader:
                                        batch_X = batch_X.to(device)
                                        outputs = lstm_model(batch_X)
                                        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                                    y_pred_proba_lstm = np.concatenate(all_outputs).flatten()

                                try:
                                    from sklearn.metrics import roc_auc_score
                                    auc_lstm = roc_auc_score(y_sequences_fixed_lstm.cpu().numpy(), y_pred_proba_lstm)
                                    models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler_fixed_lstm, "auc": auc_lstm}
                                    print(f"      LSTM AUC: {auc_lstm:.4f}")
                                except ValueError:
                                    print(f"      LSTM AUC: Not enough samples with positive class for AUC calculation.")
                                    models_and_params_local["LSTM"] = {"model": lstm_model, "scaler": dl_scaler_fixed_lstm, "auc": 0.0}
    best_model_overall = None
    best_auc_overall = -np.inf
    best_hyperparams_overall: Optional[Dict] = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    print("  🔬 Comparing classifier performance (AUC score via 5-fold cross-validation with GridSearchCV):")
    for name, mp in models_and_params_local.items():
        if name in ["LSTM", "GRU"]:
            current_auc = mp["auc"]
            results[name] = current_auc
            print(f"    - {name}: {current_auc:.4f}")
            if current_auc > best_auc_overall:
                best_auc_overall = current_auc
                best_model_overall = mp["model"]
                scaler_non_dl = mp["scaler"]
                if name == "GRU":
                    best_hyperparams_overall = mp.get("hyperparams")
        else:
            if X_non_dl.empty or len(y_non_dl) == 0:
                print(f"    - {name}: Skipping evaluation due to empty non-DL training data.")
                results[name] = 0.0
                continue

            model = mp["model"]
            params = mp["params"]
            
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
                    
                    grid_search = GridSearchCV(model, params, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
                    grid_search.fit(X_non_dl, y_non_dl)
                    
                    best_score = grid_search.best_score_
                    results[name] = best_score
                    print(f"    - {name}: {best_score:.4f} (Best Params: {grid_search.best_params_})")

                    if best_score > best_auc_overall:
                        best_auc_overall = best_score
                        best_model_overall = grid_search.best_estimator_
                        best_hyperparams_overall = None

            except Exception as e:
                print(f"    - {name}: Failed evaluation. Error: {e}")
                results[name] = 0.0

    if not any(results.values()):
        print("  ⚠️ All models failed evaluation. No model will be used.")
        return None, None, None

    best_model_name = max(results, key=results.get)
    print(f"  🏆 Best model: {best_model_name} with AUC = {best_auc_overall:.4f}")

    if best_model_name in ["LSTM", "GRU"]:
        return models_and_params_local[best_model_name]["model"], models_and_params_local[best_model_name]["scaler"], best_hyperparams_overall
    else:
        if SAVE_PLOTS and SHAP_AVAILABLE and isinstance(best_model_overall, (RandomForestClassifier, XGBClassifier)):
            analyze_shap_for_tree_model(best_model_overall, X_df_non_dl, final_feature_names_non_dl, ticker, target_col)
        return best_model_overall, scaler_non_dl, best_hyperparams_overall

def train_worker(params: Tuple) -> Dict:
    """Worker function for parallel model training."""
    ticker, df_train_period, target_percentage, class_horizon, feature_set, loaded_gru_hyperparams_buy, loaded_gru_hyperparams_sell = params
    
    models_dir = Path("logs/models")
    _ensure_dir(models_dir)
    
    model_buy_path = models_dir / f"{ticker}_model_buy.joblib"
    model_sell_path = models_dir / f"{ticker}_model_sell.joblib"
    scaler_path = models_dir / f"{ticker}_scaler.joblib"
    gru_hyperparams_buy_path = models_dir / f"{ticker}_TargetClassBuy_gru_optimized_params.json"
    gru_hyperparams_sell_path = models_dir / f"{ticker}_TargetClassSell_gru_optimized_params.json"

    model_buy, model_sell, scaler = None, None, None
    
    loaded_for_retraining = False

    if CONTINUE_TRAINING_FROM_EXISTING and model_buy_path.exists() and model_sell_path.exists() and scaler_path.exists():
        try:
            model_buy = joblib.load(model_buy_path)
            model_sell = joblib.load(model_sell_path)
            scaler = joblib.load(scaler_path)
            
            if gru_hyperparams_buy_path.exists():
                with open(gru_hyperparams_buy_path, 'r') as f:
                    loaded_gru_hyperparams_buy = json.load(f)
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU buy hyperparams for {ticker}: {e}")
            if gru_hyperparams_sell_path.exists():
                try:
                    with open(gru_hyperparams_sell_path, 'r') as f:
                        loaded_gru_hyperparams_sell = json.load(f)
                except Exception as e:
                    print(f"  ⚠️ Error loading existing GRU sell hyperparams for {ticker}: {e}")

            print(f"  ✅ Loaded existing models and GRU hyperparams for {ticker} to continue training.")
            loaded_for_retraining = True
        except Exception as e:
            print(f"  ⚠️ Error loading models or GRU hyperparams for {ticker} for retraining: {e}. Training from scratch.")

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
            if PYTORCH_AVAILABLE:
                if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                    model_buy = model_buy.cpu()
                if isinstance(model_sell, (LSTMClassifier, GRUClassifier)):
                    model_sell = model_sell.cpu()
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

    print(f"  ⚙️ Training models for {ticker} (FORCE_TRAINING is {FORCE_TRAINING}, CONTINUE_TRAINING_FROM_EXISTING is {CONTINUE_TRAINING_FROM_EXISTING})...")
    print(f"  [DEBUG] {current_process().name} - {ticker}: Initiating feature extraction for training.")
    
    current_target_percentage_buy = loaded_gru_hyperparams_buy.get('target_percentage', target_percentage) if loaded_gru_hyperparams_buy else target_percentage
    current_class_horizon_buy = loaded_gru_hyperparams_buy.get('class_horizon', class_horizon) if loaded_gru_hyperparams_buy else class_horizon
    
    df_train_buy, actual_feature_set_buy = fetch_training_data(ticker, df_train_period.copy(), current_target_percentage_buy, current_class_horizon_buy)

    if df_train_buy.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data for BUY model.")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for BUY target.")
    global_models_and_params = initialize_ml_libraries()
    model_buy, scaler_buy, gru_hyperparams_buy = train_and_evaluate_models(
        df_train_buy, "TargetClassBuy", actual_feature_set_buy, ticker=ticker,
        initial_model=model_buy if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_buy,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=current_target_percentage_buy,
        default_class_horizon=current_class_horizon_buy
    )
    
    current_target_percentage_sell = loaded_gru_hyperparams_sell.get('target_percentage', target_percentage) if loaded_gru_hyperparams_sell else target_percentage
    current_class_horizon_sell = loaded_gru_hyperparams_sell.get('class_horizon', class_horizon) if loaded_gru_hyperparams_sell else class_horizon

    df_train_sell, actual_feature_set_sell = fetch_training_data(ticker, df_train_period.copy(), current_target_percentage_sell, current_class_horizon_sell)

    if df_train_sell.empty:
        print(f"  ❌ Skipping {ticker}: Insufficient training data for SELL model.")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None}

    print(f"  [DEBUG] {current_process().name} - {ticker}: Calling train_and_evaluate_models for SELL target.")
    model_sell, scaler_sell, gru_hyperparams_sell = train_and_evaluate_models(
        df_train_sell, "TargetClassSell", actual_feature_set_sell, ticker=ticker,
        initial_model=model_sell if loaded_for_retraining else None,
        loaded_gru_hyperparams=loaded_gru_hyperparams_sell,
        models_and_params_global=global_models_and_params,
        perform_gru_hp_optimization=False,
        default_target_percentage=current_target_percentage_sell,
        default_class_horizon=current_class_horizon_sell
    )

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
            
        if PYTORCH_AVAILABLE:
            if isinstance(model_buy, (LSTMClassifier, GRUClassifier)):
                model_buy = model_buy.cpu()
                model_sell = model_sell.cpu()

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
        reason = "Insufficient training data"
        if df_train_period.empty:
            reason = f"Insufficient training data (initial rows: {len(df_train_period)})"
        elif (df_train_buy.empty or df_train_sell.empty): # Check if either buy or sell training data is empty
            reason = "Insufficient training data for buy or sell model after feature prep"
        
        print(f"  ❌ Failed to train models for {ticker}. Reason: {reason}")
        return {'ticker': ticker, 'model_buy': None, 'model_sell': None, 'scaler': None, 'status': 'failed', 'reason': reason}
