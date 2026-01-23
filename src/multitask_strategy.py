"""
Multi-Task Learning Strategy
Implements unified model training for all tickers instead of separate models per ticker.

Benefits:
- Single model learns market-wide patterns
- Knowledge sharing between tickers
- Faster training and lower memory usage
- Better generalization

Architecture:
- Shared feature extractor (learns market patterns)
- Ticker embedding (ticker-specific context)
- Task-specific heads (LSTM, XGBoost, LightGBM, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import config for parallel training settings
try:
    from config import FORCE_CPU, CUDA_AVAILABLE, TRAINING_NUM_PROCESSES, XGBOOST_USE_GPU
except ImportError:
    FORCE_CPU = False
    CUDA_AVAILABLE = False
    TRAINING_NUM_PROCESSES = 1
    XGBOOST_USE_GPU = True

# ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
    LIGHTGBM_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False


class TickerDataset(Dataset):
    """Dataset for multi-task learning with ticker embeddings."""
    
    def __init__(self, features: np.ndarray, ticker_ids: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.ticker_ids = torch.LongTensor(ticker_ids)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.ticker_ids[idx], self.targets[idx]


class MultiTaskLSTM(nn.Module):
    """Multi-task LSTM with shared layers and ticker-specific heads."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_tickers: int, dropout: float = 0.2):
        super(MultiTaskLSTM, self).__init__()
        
        # Ticker embedding layer
        self.ticker_embedding = nn.Embedding(num_tickers, 16)
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size + 16,  # features + ticker embedding
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Shared dense layers
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output = nn.Linear(64, 1)
    
    def forward(self, features, ticker_ids):
        batch_size, seq_len, feature_dim = features.shape
        
        # Get ticker embeddings
        ticker_emb = self.ticker_embedding(ticker_ids)  # [batch_size, 16]
        ticker_emb = ticker_emb.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, 16]
        
        # Concatenate features with ticker embedding
        combined = torch.cat([features, ticker_emb], dim=-1)  # [batch_size, seq_len, feature_dim+16]
        
        # Shared LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Take last time step
        lstm_last = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Shared dense layers
        shared_features = self.shared_dense(lstm_last)
        
        # Output
        return self.output(shared_features).squeeze(-1)


class MultiTaskXGBoost:
    """Multi-task XGBoost with ticker encoding."""
    
    def __init__(self, num_tickers: int):
        self.num_tickers = num_tickers
        self.model = None
        self.scaler = StandardScaler()
        self.ticker_encoder = None
    
    def fit(self, features: np.ndarray, ticker_ids: np.ndarray, targets: np.ndarray):
        """Train XGBoost with ticker encoding as additional feature."""
        # One-hot encode ticker IDs
        ticker_encoded = np.zeros((len(ticker_ids), self.num_tickers))
        ticker_encoded[np.arange(len(ticker_ids)), ticker_ids] = 1
        
        # Combine features with ticker encoding
        combined_features = np.hstack([features, ticker_encoded])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Configure XGBoost with parallel training and GPU support (same as ml_models.py)
        use_gpu = XGBOOST_USE_GPU and CUDA_AVAILABLE
        common_kwargs = {
            "random_state": 42,
            "tree_method": "hist",  # XGBoost 2.0+ uses 'hist' for both CPU and GPU
            "nthread": TRAINING_NUM_PROCESSES,
        }
        if use_gpu:
            common_kwargs["device"] = "cuda"  # This enables GPU in XGBoost 2.0+
            print(f"   🚀 XGBoost: Using GPU acceleration")
        else:
            print(f"   💻 XGBoost: Using CPU (n_jobs={TRAINING_NUM_PROCESSES})")
        
        # Create model with training parameters
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            **common_kwargs
        )
        
        self.model.fit(scaled_features, targets)
    
    def predict(self, features: np.ndarray, ticker_ids: np.ndarray):
        """Predict with ticker encoding."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # One-hot encode ticker IDs
        ticker_encoded = np.zeros((len(ticker_ids), self.num_tickers))
        ticker_encoded[np.arange(len(ticker_ids)), ticker_ids] = 1
        
        # Combine features with ticker encoding
        combined_features = np.hstack([features, ticker_encoded])
        
        # Scale features
        scaled_features = self.scaler.transform(combined_features)
        
        return self.model.predict(scaled_features)


class MultiTaskStrategy:
    """Multi-task learning strategy for stock selection."""
    
    def __init__(self):
        self.models = {}
        self.ticker_to_id = {}
        self.id_to_ticker = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_data(self, all_tickers_data: pd.DataFrame, 
                    train_start_date: datetime, train_end_date: datetime,
                    sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for multi-task learning."""
        try:
            print(f"   🧠 Multi-Task: Preparing data from {len(all_tickers_data)} rows...")
            print(f"   📅 Training period: {train_start_date.date()} to {train_end_date.date()}")
            
            # Debug: Check data structure
            print(f"   🔍 Data columns: {list(all_tickers_data.columns)}")
            print(f"   🔍 Data index: {all_tickers_data.index.name}")
            print(f"   🔍 Data index type: {type(all_tickers_data.index)}")
            
            # Handle date column vs date index
            if 'date' in all_tickers_data.columns:
                print(f"   📅 Date range in data: {all_tickers_data['date'].min()} to {all_tickers_data['date'].max()}")
                print(f"   📅 Date type: {type(all_tickers_data['date'].iloc[0])}")
                date_col = 'date'
            elif (hasattr(all_tickers_data.index, 'to_series') and 
                  len(all_tickers_data.index) > 0 and
                  (isinstance(all_tickers_data.index, pd.DatetimeIndex) or
                   (hasattr(all_tickers_data.index[0], 'year') and 
                    hasattr(all_tickers_data.index[0], 'month') and 
                    hasattr(all_tickers_data.index[0], 'day')))):
                print(f"   📅 Date range in index: {all_tickers_data.index.min()} to {all_tickers_data.index.max()}")
                print(f"   📅 Date type: {type(all_tickers_data.index[0])}")
                print(f"   📅 Index name: {all_tickers_data.index.name}")
                # Reset index to make date a column
                all_tickers_data = all_tickers_data.reset_index()
                # Rename the index column to 'date' if it doesn't have a name
                if 'index' in all_tickers_data.columns:
                    all_tickers_data = all_tickers_data.rename(columns={'index': 'date'})
                date_col = 'date'
            else:
                print(f"   ❌ No date column or index found")
                print(f"   💡 Available columns: {list(all_tickers_data.columns)}")
                print(f"   💡 Index name: {all_tickers_data.index.name}")
                print(f"   💡 Index type: {type(all_tickers_data.index)}")
                if len(all_tickers_data.index) > 0:
                    print(f"   💡 Sample index value: {all_tickers_data.index[0]}")
                return None, None, None
            
            # Filter data to training period
            train_data = all_tickers_data[
                (all_tickers_data[date_col] >= train_start_date) & 
                (all_tickers_data[date_col] <= train_end_date)
            ].copy()
            
            if train_data.empty:
                print(f"   ⚠️ No data available in training period")
                print(f"   💡 Try expanding training period or checking data availability")
                return None, None, None
            
            # Get unique tickers and create mappings
            unique_tickers = sorted(train_data['ticker'].unique())
            self.ticker_to_id = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
            self.id_to_ticker = {idx: ticker for idx, ticker in enumerate(unique_tickers)}
            
            print(f"   📊 Found {len(unique_tickers)} unique tickers")
            
            # Prepare sequences and targets
            all_sequences = []
            all_ticker_ids = []
            all_targets = []
            
            for ticker in unique_tickers:
                ticker_data = train_data[train_data['ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values(date_col)
                
                if len(ticker_data) < sequence_length + 5:
                    continue
                
                # Extract features
                feature_cols = [col for col in ticker_data.columns if col not in [date_col, 'ticker']]
                features = ticker_data[feature_cols].values
                
                # Calculate target (5-day forward return)
                returns = ticker_data['Close'].pct_change().fillna(0)
                targets = returns.shift(-5).fillna(0)  # 5-day forward return
                
                # Create sequences
                for i in range(len(features) - sequence_length - 5):
                    seq_features = features[i:i+sequence_length]
                    target_return = targets.iloc[i+sequence_length]  # Use iloc for positional indexing
                    
                    all_sequences.append(seq_features)
                    all_ticker_ids.append(self.ticker_to_id[ticker])
                    all_targets.append(target_return)
            
            if not all_sequences:
                raise ValueError("No sequences created")
            
            X = np.array(all_sequences)
            ticker_ids = np.array(all_ticker_ids)
            y = np.array(all_targets)
            
            print(f"   ✅ Created {len(X)} training sequences")
            print(f"   📐 Shape: {X.shape}, Targets: {y.shape}")
            
            return X, ticker_ids, y
            
        except Exception as e:
            print(f"   ❌ Multi-Task data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train_models(self, X: np.ndarray, ticker_ids: np.ndarray, y: np.ndarray):
        """Train multi-task models."""
        
        # Check if data is valid
        if X is None or ticker_ids is None or y is None:
            print(f"   ⚠️ Multi-Task: Cannot train - invalid data")
            return
        
        print(f"   🚀 Multi-Task: Training unified models...")
        
        # Split data
        X_train, X_val, ticker_ids_train, ticker_ids_val, y_train, y_val = train_test_split(
            X, ticker_ids, y, test_size=0.2, random_state=42
        )
        
        num_tickers = len(self.ticker_to_id)
        sequence_length, num_features = X.shape[1], X.shape[2]
        
        # Train LSTM if PyTorch available
        if PYTORCH_AVAILABLE:
            print(f"   🧠 Training Multi-Task LSTM...")
            
            # Device selection
            device = torch.device("cpu" if FORCE_CPU else ("cuda" if CUDA_AVAILABLE else "cpu"))
            print(f"   🖥️ Using device: {device}")
            
            # Create model
            lstm_model = MultiTaskLSTM(
                input_size=num_features,
                hidden_size=64,
                num_layers=2,
                num_tickers=num_tickers,
                dropout=0.2
            )
            lstm_model.to(device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
            
            # Create datasets
            train_dataset = TickerDataset(X_train, ticker_ids_train, y_train)
            val_dataset = TickerDataset(X_val, ticker_ids_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(50):
                lstm_model.train()
                train_loss = 0
                
                for batch_features, batch_ticker_ids, batch_targets in train_loader:
                    # Move tensors to device
                    batch_features = batch_features.to(device)
                    batch_ticker_ids = batch_ticker_ids.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    optimizer.zero_grad()
                    predictions = lstm_model(batch_features, batch_ticker_ids)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                lstm_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_features, batch_ticker_ids, batch_targets in val_loader:
                        # Move tensors to device
                        batch_features = batch_features.to(device)
                        batch_ticker_ids = batch_ticker_ids.to(device)
                        batch_targets = batch_targets.to(device)
                        
                        predictions = lstm_model(batch_features, batch_ticker_ids)
                        val_loss += criterion(predictions, batch_targets).item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.models['lstm'] = lstm_model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
                
                if epoch % 10 == 0:
                    print(f"      Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            print(f"   ✅ LSTM trained (Best Val Loss: {best_val_loss:.6f})")
        
        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            print(f"   🌳 Training Multi-Task XGBoost...")
            
            # Flatten sequences for XGBoost
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            xgb_model = MultiTaskXGBoost(num_tickers)
            xgb_model.fit(X_train_flat, ticker_ids_train, y_train)
            
            # Validate
            val_predictions = xgb_model.predict(X_val_flat, ticker_ids_val)
            val_mse = np.mean((val_predictions - y_val) ** 2)
            
            self.models['xgboost'] = xgb_model
            print(f"   ✅ XGBoost trained (Val MSE: {val_mse:.6f})")
        
        # Train LightGBM if available
        if LIGHTGBM_AVAILABLE:
            print(f"   💡 Training Multi-Task LightGBM...")
            
            # Flatten sequences for LightGBM
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # One-hot encode ticker IDs
            ticker_encoded_train = np.zeros((len(ticker_ids_train), num_tickers))
            ticker_encoded_train[np.arange(len(ticker_ids_train)), ticker_ids_train] = 1
            X_train_combined = np.hstack([X_train_flat, ticker_encoded_train])
            
            ticker_encoded_val = np.zeros((len(ticker_ids_val), num_tickers))
            ticker_encoded_val[np.arange(len(ticker_ids_val)), ticker_ids_val] = 1
            X_val_combined = np.hstack([X_val_flat, ticker_encoded_val])
            
            # Train LightGBM with parallel training (same as ml_models.py)
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
                device='cpu',  # LightGBM always uses CPU in this setup
                n_jobs=TRAINING_NUM_PROCESSES  # Use parallel training
            )
            
            print(f"   💡 LightGBM: Using CPU (n_jobs={TRAINING_NUM_PROCESSES})")
            
            lgb_model.fit(X_train_combined, y_train)
            
            # Validate
            val_predictions = lgb_model.predict(X_val_combined)
            val_mse = np.mean((val_predictions - y_val) ** 2)
            
            self.models['lightgbm'] = lgb_model
            print(f"   ✅ LightGBM trained (Val MSE: {val_mse:.6f})")
        
        self.is_trained = True
        print(f"   🎉 Multi-Task training complete! Models: {list(self.models.keys())}")
    
    def predict_returns_grouped(self, ticker_data_grouped: Dict[str, pd.DataFrame], 
                                current_date: datetime, top_n: int = 3) -> List[str]:
        """Predict returns using ticker_data_grouped format (with date as index)."""
        
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        print(f"   🔮 Multi-Task: Predicting returns for {len(self.ticker_to_id)} tickers...")
        
        predictions = []
        sequence_length = 30
        
        for ticker, ticker_id in self.ticker_to_id.items():
            try:
                # Get ticker data from grouped format
                if ticker not in ticker_data_grouped:
                    continue
                
                ticker_data = ticker_data_grouped[ticker].copy()
                
                # Get data up to current_date (date is index)
                if hasattr(ticker_data.index, 'to_series'):
                    recent_data = ticker_data[ticker_data.index <= current_date].tail(sequence_length + 5)
                else:
                    continue
                
                if len(recent_data) < sequence_length:
                    continue
                
                # Extract features (exclude date index and ticker column if present)
                feature_cols = [col for col in recent_data.columns if col != 'ticker']
                features = recent_data[feature_cols].values[-sequence_length:]
                
                # Make ensemble prediction
                ensemble_pred = 0
                model_count = 0
                
                # LSTM prediction
                if 'lstm' in self.models and PYTORCH_AVAILABLE:
                    lstm_model = MultiTaskLSTM(
                        input_size=features.shape[1],
                        hidden_size=64,
                        num_layers=2,
                        num_tickers=len(self.ticker_to_id)
                    )
                    lstm_model.load_state_dict(self.models['lstm'])
                    lstm_model.eval()
                    
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    ticker_id_tensor = torch.LongTensor([ticker_id])
                    
                    with torch.no_grad():
                        pred = lstm_model(features_tensor, ticker_id_tensor).item()
                        ensemble_pred += pred
                        model_count += 1
                
                # XGBoost prediction
                if 'xgboost' in self.models and XGBOOST_AVAILABLE:
                    xgb_model = self.models['xgboost']
                    features_flat = features.flatten().reshape(1, -1)
                    pred = xgb_model.predict(features_flat, np.array([ticker_id]))[0]
                    ensemble_pred += pred
                    model_count += 1
                
                # LightGBM prediction
                if 'lightgbm' in self.models and LIGHTGBM_AVAILABLE:
                    lgb_model = self.models['lightgbm']
                    features_flat = features.flatten().reshape(1, -1)
                    
                    # One-hot encode ticker
                    ticker_encoded = np.zeros((1, len(self.ticker_to_id)))
                    ticker_encoded[0, ticker_id] = 1
                    
                    features_combined = np.hstack([features_flat, ticker_encoded])
                    pred = lgb_model.predict(features_combined)[0]
                    ensemble_pred += pred
                    model_count += 1
                
                if model_count > 0:
                    avg_prediction = ensemble_pred / model_count
                    predictions.append((ticker, avg_prediction))
                
            except Exception as e:
                continue
        
        # Sort by prediction and select top N
        if predictions:
            predictions.sort(key=lambda x: x[1], reverse=True)
            selected_tickers = [ticker for ticker, pred in predictions[:top_n]]
            
            print(f"   🎯 Multi-Task selected {len(selected_tickers)} stocks:")
            for ticker, pred in predictions[:top_n]:
                print(f"      {ticker}: {pred:+.3f}")
            
            return selected_tickers
        else:
            print(f"   ❌ No valid predictions made")
            return []
    
    def predict_returns(self, all_tickers_data: pd.DataFrame, 
                      current_date: datetime, top_n: int = 3) -> List[str]:
        """Predict returns for all tickers and select top N."""
        
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        print(f"   🔮 Multi-Task: Predicting returns for {len(self.ticker_to_id)} tickers...")
        
        predictions = []
        sequence_length = 30
        
        for ticker, ticker_id in self.ticker_to_id.items():
            try:
                # Get recent data for this ticker
                ticker_data = all_tickers_data[all_tickers_data['ticker'] == ticker].copy()
                
                # Handle date column vs index
                if 'date' in ticker_data.columns:
                    ticker_data = ticker_data.sort_values('date')
                    recent_data = ticker_data[ticker_data['date'] <= current_date].tail(sequence_length + 5)
                    feature_cols = [col for col in recent_data.columns if col not in ['date', 'ticker']]
                elif hasattr(ticker_data.index, 'to_series') and ticker_data.index.name == 'date':
                    ticker_data = ticker_data.sort_index()
                    recent_data = ticker_data[ticker_data.index <= current_date].tail(sequence_length + 5)
                    feature_cols = [col for col in recent_data.columns if col != 'ticker']
                else:
                    continue
                
                if len(recent_data) < sequence_length:
                    continue
                
                # Extract features
                features = recent_data[feature_cols].values[-sequence_length:]
                
                # Make ensemble prediction
                ensemble_pred = 0
                model_count = 0
                
                # LSTM prediction
                if 'lstm' in self.models and PYTORCH_AVAILABLE:
                    lstm_model = MultiTaskLSTM(
                        input_size=features.shape[1],
                        hidden_size=64,
                        num_layers=2,
                        num_tickers=len(self.ticker_to_id)
                    )
                    lstm_model.load_state_dict(self.models['lstm'])
                    lstm_model.eval()
                    
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        ticker_id_tensor = torch.LongTensor([ticker_id])
                        pred = lstm_model(features_tensor, ticker_id_tensor).item()
                        ensemble_pred += pred
                        model_count += 1
                
                # XGBoost prediction
                if 'xgboost' in self.models and XGBOOST_AVAILABLE:
                    xgb_model = self.models['xgboost']
                    features_flat = features.flatten().reshape(1, -1)
                    pred = xgb_model.predict(features_flat, np.array([ticker_id]))[0]
                    ensemble_pred += pred
                    model_count += 1
                
                # LightGBM prediction
                if 'lightgbm' in self.models and LIGHTGBM_AVAILABLE:
                    lgb_model = self.models['lightgbm']
                    features_flat = features.flatten().reshape(1, -1)
                    
                    # One-hot encode ticker
                    ticker_encoded = np.zeros((1, len(self.ticker_to_id)))
                    ticker_encoded[0, ticker_id] = 1
                    
                    features_combined = np.hstack([features_flat, ticker_encoded])
                    pred = lgb_model.predict(features_combined)[0]
                    ensemble_pred += pred
                    model_count += 1
                
                if model_count > 0:
                    avg_prediction = ensemble_pred / model_count
                    predictions.append((ticker, avg_prediction))
            
            except Exception as e:
                continue
        
        # Sort by prediction and select top N
        if predictions:
            predictions.sort(key=lambda x: x[1], reverse=True)
            selected_tickers = [ticker for ticker, _ in predictions[:top_n]]
            
            print(f"   🎯 Multi-Task selected {len(selected_tickers)} stocks:")
            for ticker, pred in predictions[:top_n]:
                print(f"      {ticker}: {pred:+.3f}")
            
            return selected_tickers
        else:
            print(f"   ❌ No valid predictions made")
            return []


def select_multitask_stocks(all_tickers: List[str], ticker_data_grouped: Dict[str, pd.DataFrame], 
                           current_date: datetime = None, train_start_date: datetime = None,
                           train_end_date: datetime = None, top_n: int = 3) -> List[str]:
    """
    Multi-Task Learning stock selection strategy.
    
    This strategy implements unified model training for all tickers,
    enabling knowledge sharing and better generalization.
    
    Args:
        all_tickers: List of ticker symbols to analyze
        ticker_data_grouped: Dict mapping ticker -> price data (with date as index)
        current_date: Current date for analysis
        train_start_date: Start date for training
        train_end_date: End date for training
        top_n: Number of stocks to select
        
    Returns:
        List[str]: Selected ticker symbols
    """
    
    if not PYTORCH_AVAILABLE and not XGBOOST_AVAILABLE:
        print("   ⚠️ Multi-Task: No ML libraries available, using fallback")
        return []
    
    print(f"   🧠 Multi-Task Learning Strategy")
    print(f"   📊 Analyzing {len(all_tickers)} tickers with unified models")
    
    try:
        # Initialize strategy
        strategy = MultiTaskStrategy()
        
        # Prepare data directly from ticker_data_grouped
        print(f"   🧠 Multi-Task: Preparing data from grouped format...")
        print(f"   📅 Training period: {train_start_date.date()} to {train_end_date.date()}")
        
        # Collect all data with proper date handling
        all_data_rows = []
        for ticker in all_tickers:
            if ticker not in ticker_data_grouped:
                continue
            
            ticker_data = ticker_data_grouped[ticker].copy()
            
            # Reset index to make date a column
            if hasattr(ticker_data.index, 'to_series'):
                ticker_data = ticker_data.reset_index()
                if 'index' in ticker_data.columns:
                    ticker_data = ticker_data.rename(columns={'index': 'date'})
            
            # Add ticker column
            ticker_data['ticker'] = ticker
            all_data_rows.append(ticker_data)
        
        if not all_data_rows:
            print("   ❌ No data available")
            return []
        
        all_tickers_data = pd.concat(all_data_rows, ignore_index=True)
        
        # Prepare and train models
        X, ticker_ids, y = strategy.prepare_data(
            all_tickers_data, train_start_date, train_end_date
        )
        
        # Check if data preparation was successful
        if X is None or ticker_ids is None or y is None:
            print("   ❌ Multi-Task: Data preparation failed")
            return []
        
        strategy.train_models(X, ticker_ids, y)
        
        # Make predictions using the original grouped data format
        selected_tickers = strategy.predict_returns_grouped(
            ticker_data_grouped, current_date, top_n
        )
        
        return selected_tickers
        
    except Exception as e:
        print(f"   ❌ Multi-Task strategy error: {e}")
        import traceback
        traceback.print_exc()
        return []
