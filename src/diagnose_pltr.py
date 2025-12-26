"""
Diagnostic script to understand why PLTR made 0 trades in 1-Year backtest.
Checks model predictions, probability distributions, and threshold compatibility.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle

print("=" * 80)
print("PLTR 1-Year Trading Diagnostic")
print("=" * 80)

# Load the trained models
models_dir = Path("logs/models")
buy_model_path = models_dir / "PLTR_TargetClassBuy_GRUClassifier_model.pkl"
sell_model_path = models_dir / "PLTR_TargetClassSell_GRUClassifier_model.pkl"
scaler_path = models_dir / "PLTR_scaler.pkl"

if not buy_model_path.exists():
    print(f"❌ Buy model not found at {buy_model_path}")
    exit(1)

print(f"\n📂 Loading models from {models_dir}...")
with open(buy_model_path, 'rb') as f:
    model_buy = pickle.load(f)
with open(sell_model_path, 'rb') as f:
    model_sell = pickle.load(f)
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"✅ Models loaded successfully")
print(f"   Buy Model: {type(model_buy).__name__}")
print(f"   Sell Model: {type(model_sell).__name__}")
print(f"   Scaler features: {len(scaler.feature_names_in_)}")

# Load PLTR backtest data
print(f"\n📊 Analyzing PLTR backtest data...")
import yfinance as yf
from datetime import datetime, timedelta

bt_end = datetime(2024, 12, 6)
bt_start = bt_end - timedelta(days=365)

print(f"   Backtest period: {bt_start.date()} to {bt_end.date()}")

ticker_data = yf.download("PLTR", start=bt_start, end=bt_end, progress=False)

if ticker_data.empty:
    print("❌ Could not download PLTR data")
    exit(1)

print(f"   Downloaded {len(ticker_data)} days of data")

# Calculate required features (simplified version)
from src.main import _calculate_technical_indicators

df = ticker_data.copy()
df = df.rename(columns={"Adj Close": "Close"})
df = _calculate_technical_indicators(df)

# Get feature set
feature_cols = scaler.feature_names_in_

print(f"\n🔍 Checking feature availability:")
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"   ⚠️  Missing features: {missing_features[:10]}")
    # Fill missing with 0
    for f in missing_features:
        df[f] = 0.0

# Select features and scale
X = df[feature_cols].fillna(0).values
X_scaled = scaler.transform(X)

# Make predictions with PyTorch model
model_buy.eval()
model_sell.eval()

with torch.no_grad():
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Reshape for GRU: (batch, seq_len, features) -> use seq_len=1
    X_tensor = X_tensor.unsqueeze(1)  # (N, 1, features)
    
    # Get predictions
    buy_probs = torch.softmax(model_buy(X_tensor), dim=1)[:, 1].numpy()
    sell_probs = torch.softmax(model_sell(X_tensor), dim=1)[:, 1].numpy()

print(f"\n📈 Probability Statistics:")
print(f"   Buy Probabilities:")
print(f"      Mean: {buy_probs.mean():.4f}")
print(f"      Median: {np.median(buy_probs):.4f}")
print(f"      Min: {buy_probs.min():.4f}, Max: {buy_probs.max():.4f}")
print(f"      Std: {buy_probs.std():.4f}")
print(f"\n   Sell Probabilities:")
print(f"      Mean: {sell_probs.mean():.4f}")
print(f"      Median: {np.median(sell_probs):.4f}")
print(f"      Min: {sell_probs.min():.4f}, Max: {sell_probs.max():.4f}")
print(f"      Std: {sell_probs.std():.4f}")

# Check against optimized thresholds
opt_buy_thresh = 0.27
opt_sell_thresh = 0.11

print(f"\n🎯 Threshold Analysis:")
print(f"   Optimized Buy Threshold: {opt_buy_thresh:.2f}")
print(f"   Optimized Sell Threshold: {opt_sell_thresh:.2f}")
print(f"\n   Days where Buy Prob > Threshold: {(buy_probs > opt_buy_thresh).sum()} / {len(buy_probs)} ({(buy_probs > opt_buy_thresh).sum() / len(buy_probs) * 100:.1f}%)")
print(f"   Days where Sell Prob > Threshold: {(sell_probs > opt_sell_thresh).sum()} / {len(sell_probs)} ({(sell_probs > opt_sell_thresh).sum() / len(sell_probs) * 100:.1f}%)")

# Show top buy signals
print(f"\n🔝 Top 10 Buy Signals:")
top_buy_idx = np.argsort(buy_probs)[-10:][::-1]
for idx in top_buy_idx:
    date = df.index[idx].strftime('%Y-%m-%d')
    prob = buy_probs[idx]
    price = df['Close'].iloc[idx]
    print(f"   {date}: Prob={prob:.4f}, Price=${price:.2f}")

print(f"\n🔝 Top 10 Sell Signals:")
top_sell_idx = np.argsort(sell_probs)[-10:][::-1]
for idx in top_sell_idx:
    date = df.index[idx].strftime('%Y-%m-%d')
    prob = sell_probs[idx]
    price = df['Close'].iloc[idx]
    print(f"   {date}: Prob={prob:.4f}, Price=${price:.2f}")

# Recommendations
print(f"\n💡 Recommendations:")
if buy_probs.max() < opt_buy_thresh:
    print(f"   ⚠️  Max buy probability ({buy_probs.max():.4f}) < threshold ({opt_buy_thresh:.2f})")
    print(f"       Recommend lowering buy threshold to ~{buy_probs.quantile(0.75):.2f} (75th percentile)")
if sell_probs.max() < opt_sell_thresh:
    print(f"   ⚠️  Max sell probability ({sell_probs.max():.4f}) < threshold ({opt_sell_thresh:.2f})")
    print(f"       Recommend lowering sell threshold to ~{sell_probs.quantile(0.75):.2f} (75th percentile)")

# Check model quality
print(f"\n🔬 Model Quality Check:")
print(f"   Buy model output range: [{buy_probs.min():.4f}, {buy_probs.max():.4f}]")
print(f"   Sell model output range: [{sell_probs.min():.4f}, {sell_probs.max():.4f}]")

if buy_probs.std() < 0.05:
    print(f"   ⚠️  Buy model has very low variance (std={buy_probs.std():.4f})")
    print(f"       This suggests the model is not well-calibrated or undertrained")
if sell_probs.std() < 0.05:
    print(f"   ⚠️  Sell model has very low variance (std={sell_probs.std():.4f})")
    print(f"       This suggests the model is not well-calibrated or undertrained")

print(f"\n" + "=" * 80)
print("Diagnosis Complete")
print("=" * 80)
















